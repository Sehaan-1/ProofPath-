"""Local ProofPath MCP server exposing allowlisted read-only Evidence & Opportunity tools.

Slice 1.7:
- Sub-part 1.7.1a: Typed `MCPServer("proofpath")` (`FastMCP` in MCP SDK v1, renamed
  to `MCPServer` in official `mcp>=2.0`) supporting `stdio` and Streamable HTTP.
- Sub-part 1.7.2a: Read-only tools `search_evidence`, `get_opportunity`,
  `get_application_status`, and `get_action_draft` backed directly by Slice 1.1
  repositories and Slice 1.3 hybrid search.
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Final, Literal, TypeAlias

from mcp.server.mcpserver import Context
from mcp.server.mcpserver import MCPServer as MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools.base import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.applications import Starlette

from proofpath.config import LogLevel, Settings, get_settings
from proofpath.db.repositories.draft import DraftRepo, draft_record
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.db.session import DatabaseSessionManager
from proofpath.policy.tool_allowlist import (
    READ_ONLY_TOOL_NAMES,
    ToolNotAllowedError,
    require_allowed,
)
from proofpath.retrieval.embeddings import EmbeddingClient, create_embedding_client
from proofpath.retrieval.hybrid_search import DEFAULT_TOP_K, HybridSearcher
from proofpath.schemas.actions import (
    ActionDraftRecord,
    ApplicationStatusResponse,
)
from proofpath.schemas.evidence import EvidenceSearchResponse
from proofpath.schemas.opportunity import OpportunityResponse, RequirementRecord
from proofpath.schemas.tools import (
    GetActionDraftInput,
    GetApplicationStatusInput,
    GetOpportunityInput,
    SearchEvidenceInput,
)

# In official MCP Python SDK v2 (`mcp>=2.0`), `FastMCP` was renamed to `MCPServer`.
# Export `FastMCP` as an explicit typed alias alongside `MCPServer`.
FastMCP = MCPServer

SERVER_NAME: Final[str] = "proofpath"
SERVER_TITLE: Final[str] = "ProofPath"
SERVER_VERSION: Final[str] = "0.1.0"
SERVER_INSTRUCTIONS: Final[str] = (
    "Read-only access to the personal ProofPath Evidence Ledger, target opportunities, "
    "application tracker stages, and pending/historical action drafts. Every supported "
    "claim must cite verified user-owned evidence chunks, and no mutating action is "
    "executed by these read-only tools."
)

MCPTransport: TypeAlias = Literal["stdio", "streamable-http"]
MCPLogLevel: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
SessionContextFactory: TypeAlias = Callable[[], AbstractAsyncContextManager[AsyncSession]]

_LOG_LEVEL_MAP: Final[dict[LogLevel, MCPLogLevel]] = {
    LogLevel.DEBUG: "DEBUG",
    LogLevel.INFO: "INFO",
    LogLevel.WARNING: "WARNING",
    LogLevel.ERROR: "ERROR",
    LogLevel.CRITICAL: "CRITICAL",
}

_READ_ONLY_ANNOTATIONS: Final[ToolAnnotations] = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


class _SearchEvidenceArgs(SearchEvidenceInput, ArgModelBase):
    """Strict MCP argument model (`extra='forbid'`) for `search_evidence`."""


class _GetOpportunityArgs(GetOpportunityInput, ArgModelBase):
    """Strict MCP argument model (`extra='forbid'`) for `get_opportunity`."""


class _GetApplicationStatusArgs(GetApplicationStatusInput, ArgModelBase):
    """Strict MCP argument model (`extra='forbid'`) for `get_application_status`."""


class _GetActionDraftArgs(GetActionDraftInput, ArgModelBase):
    """Strict MCP argument model (`extra='forbid'`) for `get_action_draft`."""


class MCPServerRuntime:
    """Lifespan-scoped database and embedding resources for the ProofPath MCP server."""

    def __init__(
        self,
        *,
        settings: Settings,
        open_session: SessionContextFactory,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings
        self.open_session = open_session
        self._embedder = embedder

    def get_embedder(self) -> EmbeddingClient:
        """Lazily initialize the configured `EmbeddingClient` on first search call."""
        if self._embedder is None:
            try:
                self._embedder = create_embedding_client(self.settings)
            except ValueError as exc:
                raise ToolError(
                    "Embedding provider is not configured. Set LLM_API_KEY or EMBEDDING_MODEL=hash."
                ) from exc
        return self._embedder


def _build_read_only_tool(
    fn: Callable[..., Any],
    *,
    arg_model: type[ArgModelBase],
) -> Tool:
    """Register an allowlisted read-only tool with its strict `extra='forbid'` schema."""
    spec = require_allowed(fn.__name__)
    if not spec.is_read_only:
        raise ToolNotAllowedError(spec.tool_name)
    if arg_model.model_config.get("extra") != "forbid" or set(arg_model.model_fields) != set(
        spec.input_schema.model_fields
    ):
        raise TypeError(
            f"Argument model for {spec.tool_name!r} must match strict {spec.input_schema.__name__}."
        )
    tool = Tool.from_function(
        fn,
        name=spec.tool_name,
        description=spec.description,
        annotations=_READ_ONLY_ANNOTATIONS,
        structured_output=True,
    )
    tool.fn_metadata.arg_model = arg_model
    tool.parameters = spec.input_schema.model_json_schema(by_alias=True)
    return tool


async def _execute_search_evidence(
    session: AsyncSession,
    embedder: EmbeddingClient,
    *,
    query: str,
    top_k: int,
) -> EvidenceSearchResponse:
    validated = SearchEvidenceInput(query=query, top_k=top_k)
    searcher = HybridSearcher(session, embedder)
    results = await searcher.search(validated.query, top_k=validated.top_k)
    return EvidenceSearchResponse(query=validated.query, results=results)


async def _execute_get_opportunity(
    session: AsyncSession,
    *,
    opportunity_id: uuid.UUID,
) -> OpportunityResponse:
    validated = GetOpportunityInput(opportunity_id=opportunity_id)
    opportunity = await OpportunityRepo(session).get_by_id(
        validated.opportunity_id,
        include_requirements=True,
    )
    if opportunity is None:
        raise ToolError(f"Opportunity {validated.opportunity_id} not found.")
    if any(not item.entity for item in opportunity.requirements):
        raise ToolError("Requirement entities must be recovered before reading this opportunity.")
    return OpportunityResponse(
        id=opportunity.id,
        company=opportunity.company,
        role_title=opportunity.role_title,
        source_url=opportunity.source_url,
        created_at=opportunity.created_at,
        requirements=[
            RequirementRecord(
                id=row.id,
                opportunity_id=row.opportunity_id,
                requirement_text=row.requirement_text,
                entity=row.entity,
                category=row.category,
                priority=row.priority,
            )
            for row in opportunity.requirements
            if row.entity
        ],
    )


async def _execute_get_application_status(
    session: AsyncSession,
    *,
    opportunity_id: uuid.UUID,
) -> ApplicationStatusResponse:
    validated = GetApplicationStatusInput(opportunity_id=opportunity_id)
    tracker = await TrackerRepo(session).get_by_opportunity_id(validated.opportunity_id)
    if tracker is not None:
        return ApplicationStatusResponse(
            opportunity_id=tracker.opportunity_id,
            tracked=True,
            tracker_id=tracker.id,
            stage=tracker.stage,
            notes=tracker.notes,
            updated_at=tracker.updated_at,
        )
    if await OpportunityRepo(session).get_by_id(validated.opportunity_id) is None:
        raise ToolError(f"Opportunity {validated.opportunity_id} not found.")
    return ApplicationStatusResponse(
        opportunity_id=validated.opportunity_id,
        tracked=False,
    )


async def _execute_get_action_draft(
    session: AsyncSession,
    *,
    draft_id: uuid.UUID,
) -> ActionDraftRecord:
    validated = GetActionDraftInput(draft_id=draft_id)
    draft = await DraftRepo(session).get_by_id(validated.draft_id)
    if draft is None:
        raise ToolError(f"Action draft {validated.draft_id} not found.")
    try:
        return draft_record(draft)
    except ValidationError as exc:
        raise ToolError(
            f"Stored payload for action draft {validated.draft_id} is invalid."
        ) from exc


def create_mcp_server(
    *,
    settings: Settings | None = None,
    session_factory: SessionContextFactory | None = None,
    embedder: EmbeddingClient | None = None,
) -> MCPServer[MCPServerRuntime]:
    """Create a typed `MCPServer("proofpath")` instance with Slice 1.7 read-only tools."""

    @asynccontextmanager
    async def _lifespan(_server: MCPServer[MCPServerRuntime]) -> AsyncIterator[MCPServerRuntime]:
        active_settings = settings if settings is not None else get_settings()
        if session_factory is not None:
            yield MCPServerRuntime(
                settings=active_settings,
                open_session=session_factory,
                embedder=embedder,
            )
            return
        db_manager = DatabaseSessionManager(active_settings.asyncpg_database_url)
        try:
            yield MCPServerRuntime(
                settings=active_settings,
                open_session=db_manager.session,
                embedder=embedder,
            )
        finally:
            await db_manager.close()

    @asynccontextmanager
    async def _active_runtime(
        ctx: Context[MCPServerRuntime, Any] | None,
    ) -> AsyncIterator[MCPServerRuntime]:
        if ctx is not None:
            try:
                yield ctx.request_context.lifespan_context
                return
            except ValueError:
                pass
        active_settings = settings if settings is not None else get_settings()
        if session_factory is not None:
            yield MCPServerRuntime(
                settings=active_settings,
                open_session=session_factory,
                embedder=embedder,
            )
            return
        db_manager = DatabaseSessionManager(active_settings.asyncpg_database_url)
        try:
            yield MCPServerRuntime(
                settings=active_settings,
                open_session=db_manager.session,
                embedder=embedder,
            )
        finally:
            await db_manager.close()

    async def search_evidence(
        query: str,
        top_k: int = DEFAULT_TOP_K,
        ctx: Context[MCPServerRuntime, Any] | None = None,
    ) -> EvidenceSearchResponse:
        """Hybrid pgvector + full-text RRF search over the user's Evidence Ledger."""
        async with _active_runtime(ctx) as runtime:
            active_embedder = runtime.get_embedder()
            async with runtime.open_session() as session:
                return await _execute_search_evidence(
                    session,
                    active_embedder,
                    query=query,
                    top_k=top_k,
                )

    async def get_opportunity(
        opportunity_id: uuid.UUID,
        ctx: Context[MCPServerRuntime, Any] | None = None,
    ) -> OpportunityResponse:
        """Fetch a persisted target opportunity and its structured requirements."""
        async with _active_runtime(ctx) as runtime, runtime.open_session() as session:
            return await _execute_get_opportunity(session, opportunity_id=opportunity_id)

    async def get_application_status(
        opportunity_id: uuid.UUID,
        ctx: Context[MCPServerRuntime, Any] | None = None,
    ) -> ApplicationStatusResponse:
        """Read the personal application-tracker stage for an opportunity."""
        async with _active_runtime(ctx) as runtime, runtime.open_session() as session:
            return await _execute_get_application_status(session, opportunity_id=opportunity_id)

    async def get_action_draft(
        draft_id: uuid.UUID,
        ctx: Context[MCPServerRuntime, Any] | None = None,
    ) -> ActionDraftRecord:
        """Read a pending or historical action draft without executing side effects."""
        async with _active_runtime(ctx) as runtime, runtime.open_session() as session:
            return await _execute_get_action_draft(session, draft_id=draft_id)

    tools = [
        _build_read_only_tool(search_evidence, arg_model=_SearchEvidenceArgs),
        _build_read_only_tool(get_opportunity, arg_model=_GetOpportunityArgs),
        _build_read_only_tool(get_application_status, arg_model=_GetApplicationStatusArgs),
        _build_read_only_tool(get_action_draft, arg_model=_GetActionDraftArgs),
    ]
    registered_names = frozenset(tool.name for tool in tools)
    if registered_names != READ_ONLY_TOOL_NAMES:
        raise RuntimeError(
            f"Expected read-only tool set {sorted(READ_ONLY_TOOL_NAMES)}, "
            f"got {sorted(registered_names)}."
        )

    active_log_level: MCPLogLevel = (
        _LOG_LEVEL_MAP[settings.log_level] if settings is not None else "INFO"
    )
    return MCPServer(
        SERVER_NAME,
        title=SERVER_TITLE,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        tools=tools,
        lifespan=_lifespan,
        log_level=active_log_level,
    )


def create_streamable_http_app(
    *,
    server: MCPServer[MCPServerRuntime] | None = None,
    settings: Settings | None = None,
    session_factory: SessionContextFactory | None = None,
    embedder: EmbeddingClient | None = None,
    host: str = "127.0.0.1",
    streamable_http_path: str = "/mcp",
    json_response: bool = False,
    stateless_http: bool = False,
    transport_security: TransportSecuritySettings | None = None,
) -> Starlette:
    """Create a Starlette ASGI app serving ProofPath over Streamable HTTP transport."""
    active_server = (
        server
        if server is not None
        else create_mcp_server(
            settings=settings,
            session_factory=session_factory,
            embedder=embedder,
        )
    )
    return active_server.streamable_http_app(
        host=host,
        streamable_http_path=streamable_http_path,
        json_response=json_response,
        stateless_http=stateless_http,
        transport_security=transport_security,
    )


def run_mcp_server(
    *,
    transport: MCPTransport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
    json_response: bool = False,
    stateless_http: bool = False,
    settings: Settings | None = None,
) -> None:
    """Start the ProofPath MCP server using `stdio` or `streamable-http` transport."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535.")
    if not streamable_http_path.startswith("/"):
        raise ValueError("streamable_http_path must start with '/'.")

    server = create_mcp_server(settings=settings)
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "streamable-http":
        server.run(
            transport="streamable-http",
            host=host,
            port=port,
            streamable_http_path=streamable_http_path,
            json_response=json_response,
            stateless_http=stateless_http,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported MCP transport: {transport!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for `python -m proofpath.mcp_server.server` (`make mcp-dev`)."""
    default_transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    default_host = os.getenv("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_port = os.getenv("MCP_PORT", "8000").strip()
    try:
        default_port = int(raw_port)
    except ValueError:
        default_port = 8000
    default_path = os.getenv("MCP_PATH", "/mcp").strip() or "/mcp"

    parser = argparse.ArgumentParser(
        prog="proofpath-mcp",
        description="Run the local ProofPath MCP server over stdio or Streamable HTTP.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=default_transport if default_transport in {"stdio", "streamable-http"} else "stdio",
        help="MCP transport protocol (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help="Bind host for streamable-http transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Bind port for streamable-http transport (default: 8000).",
    )
    parser.add_argument(
        "--path",
        default=default_path,
        help="Mount path for streamable-http transport (default: /mcp).",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        default=False,
        help="Return direct JSON responses instead of SSE streams over streamable-http.",
    )
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        default=False,
        help="Run streamable-http without persistent per-session server state.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m proofpath.mcp_server.server`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if not args.path.startswith("/"):
        parser.error("--path must start with '/'.")

    transport: MCPTransport = "streamable-http" if args.transport == "streamable-http" else "stdio"
    run_mcp_server(
        transport=transport,
        host=args.host,
        port=args.port,
        streamable_http_path=args.path,
        json_response=bool(args.json_response),
        stateless_http=bool(args.stateless_http),
    )
    return 0


mcp: MCPServer[MCPServerRuntime] = create_mcp_server()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
