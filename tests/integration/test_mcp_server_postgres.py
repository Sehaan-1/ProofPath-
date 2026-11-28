"""Automated MCP client integration tests against live PostgreSQL + pgvector (Slice 1.7.2b)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import jsonschema
import pytest
import uvicorn
from mcp import Client, StdioServerParameters
from pydantic import PostgresDsn
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.config import Settings
from proofpath.db.models import (
    EMBEDDING_DIM,
    ActionDraftStatus,
    ActionType,
    ApplicationStage,
    EvidenceSource,
    EvidenceSourceType,
    RequirementCategory,
    RequirementPriority,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.draft import DraftRepo
from proofpath.db.repositories.evidence import EvidenceChunkInput, EvidenceRepo
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.db.session import create_db_engine, create_session_factory
from proofpath.mcp_server import create_mcp_server, create_streamable_http_app
from proofpath.policy.tool_allowlist import FORBIDDEN_TOOL_NAMES, READ_ONLY_TOOL_NAMES
from proofpath.retrieval import EmbeddingClient, HashEmbeddingBackend
from proofpath.schemas import (
    TOOL_INPUT_REGISTRY,
    ActionDraftRecord,
    ApplicationStatusResponse,
    EvidenceSearchResponse,
    OpportunityResponse,
    TrackerUpsertPayload,
)

pytestmark = pytest.mark.integration


def _digest(label: str) -> str:
    return hashlib.sha256(f"mcp-int-{label}-{uuid.uuid4()}".encode()).hexdigest()


async def test_mcp_client_lists_tools_validates_schemas_and_searches_seeded_postgres(
    db_session: AsyncSession,
    integration_database_url: str,
) -> None:
    evidence_repo = EvidenceRepo(db_session)
    opp_repo = OpportunityRepo(db_session)
    analysis_repo = AnalysisRepo(db_session)
    tracker_repo = TrackerRepo(db_session)
    draft_repo = DraftRepo(db_session)

    embedder = EmbeddingClient(
        HashEmbeddingBackend(dimensions=EMBEDDING_DIM),
        model="hash",
        dimensions=EMBEDDING_DIM,
    )

    mcp_chunk_text = (
        "Shipped a typed ProofPath MCP server in Python using FastAPI, SQLAlchemy, "
        "PostgreSQL, and pgvector Reciprocal Rank Fusion."
    )
    sap_chunk_text = "Configured SAP S/4HANA finance workflows during an enterprise internship."
    notes_chunk_text = "Maintained CI pipelines with ruff, mypy, and pytest integration suites."
    mcp_vec, sap_vec, notes_vec = await embedder.embed(
        [mcp_chunk_text, sap_chunk_text, notes_chunk_text]
    )

    source = await evidence_repo.create_source(
        source_type=EvidenceSourceType.RESUME,
        title="ProofPath Engineer Resume",
        canonical_uri="file:///resumes/proofpath_engineer.md",
        content_sha256=_digest("resume"),
        metadata_json={
            "chunk_anchors": [
                {
                    "chunk_index": 0,
                    "canonical_uri": "file:///resumes/proofpath_engineer.md#Projects",
                },
                {
                    "chunk_index": 1,
                    "canonical_uri": "file:///resumes/proofpath_engineer.md#Experience",
                },
            ]
        },
    )
    inserted_chunks = await evidence_repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Projects",
                content=mcp_chunk_text,
                token_count=20,
                embedding=mcp_vec,
            ),
            EvidenceChunkInput(
                chunk_index=1,
                section_label="Experience",
                content=sap_chunk_text,
                token_count=12,
                embedding=sap_vec,
            ),
            EvidenceChunkInput(
                chunk_index=2,
                section_label="Skills",
                content=notes_chunk_text,
                token_count=12,
                embedding=notes_vec,
            ),
        ],
    )

    tracked_opp = await opp_repo.create_opportunity(
        company="Palo Alto Networks",
        role_title="AI Agent Platform Engineer",
        source_url="https://jobs.example.com/panw-mcp",
        raw_text="Build MCP servers and RAG pipelines with Python and pgvector.",
        sanitized_text="Build MCP servers and RAG pipelines with Python and pgvector.",
    )
    tracked_reqs = await opp_repo.add_requirements(
        tracked_opp.id,
        [
            RoleRequirementInput(
                requirement_text="Build MCP tool servers in Python",
                entity="MCP",
                category=RequirementCategory.AI_AGENT,
                priority=RequirementPriority.MUST_HAVE,
            ),
            RoleRequirementInput(
                requirement_text="PostgreSQL pgvector hybrid retrieval",
                entity="pgvector",
                category=RequirementCategory.FRAMEWORK,
                priority=RequirementPriority.MUST_HAVE,
            ),
        ],
    )
    tracker_entry = await tracker_repo.upsert(
        opportunity_id=tracked_opp.id,
        stage=ApplicationStage.PREPARING,
        notes="Validating Evidence Ledger citations",
    )

    untracked_opp = await opp_repo.create_opportunity(
        company="Example Cloud",
        role_title="Backend Engineer",
        raw_text="FastAPI and PostgreSQL required.",
        sanitized_text="FastAPI and PostgreSQL required.",
    )

    analysis_run = await analysis_repo.create_run(
        opportunity_id=tracked_opp.id,
        thread_id=f"mcp-thread-{uuid.uuid4()}",
    )
    draft_payload = TrackerUpsertPayload(
        action_type=ActionType.TRACKER_UPSERT,
        opportunity_id=tracked_opp.id,
        stage=ApplicationStage.READY_TO_APPLY,
        notes="Draft ready after policy check",
    )
    draft_row = await draft_repo.create_draft(
        analysis_run_id=analysis_run.id,
        action_type=ActionType.TRACKER_UPSERT,
        payload_json=draft_payload.model_dump(mode="json"),
        idempotency_key=f"proofpath:{analysis_run.id}:tracker_upsert",
    )
    await db_session.flush()

    @asynccontextmanager
    async def _tx_session_factory() -> AsyncIterator[AsyncSession]:
        yield db_session

    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        embedding_model="hash",
    )
    server = create_mcp_server(
        settings=settings,
        session_factory=_tx_session_factory,
        embedder=embedder,
    )

    async with Client(server) as client:
        # 1) List tools and validate JSON schemas.
        listed = await client.list_tools()
        tools_by_name = {tool.name: tool for tool in listed.tools}
        assert set(tools_by_name) == set(READ_ONLY_TOOL_NAMES)
        assert set(tools_by_name).isdisjoint(FORBIDDEN_TOOL_NAMES)

        for name, tool in tools_by_name.items():
            jsonschema.Draft202012Validator.check_schema(tool.input_schema)
            assert tool.input_schema == TOOL_INPUT_REGISTRY[name].model_json_schema(by_alias=True)
            assert tool.input_schema.get("additionalProperties") is False
            assert tool.output_schema is not None
            jsonschema.Draft202012Validator.check_schema(tool.output_schema)

        # 2) Invoke `search_evidence` against seeded PostgreSQL + pgvector data.
        search_call = await client.call_tool(
            "search_evidence",
            {"query": "MCP pgvector Reciprocal Rank Fusion", "top_k": 2},
        )
        assert search_call.is_error is False
        assert search_call.structured_content is not None
        jsonschema.validate(
            instance=search_call.structured_content,
            schema=tools_by_name["search_evidence"].output_schema or {},
        )
        search_response = EvidenceSearchResponse.model_validate(search_call.structured_content)
        assert search_response.query == "MCP pgvector Reciprocal Rank Fusion"
        assert 1 <= len(search_response.results) <= 2
        assert search_response.results[0].chunk_id == inserted_chunks[0].id
        assert search_response.results[0].source_id == source.id
        assert (
            search_response.results[0].canonical_uri
            == "file:///resumes/proofpath_engineer.md#Projects"
        )
        assert search_response.results[0].score is not None
        assert search_response.results[0].score > 0
        # Text content also deserializes cleanly for unstructured MCP clients.
        text_body = json.loads(search_call.content[0].text)  # type: ignore[union-attr]
        assert EvidenceSearchResponse.model_validate(text_body) == search_response

        sap_call = await client.call_tool(
            "search_evidence",
            {"query": "SAP S/4HANA", "top_k": 2},
        )
        assert sap_call.is_error is False
        sap_response = EvidenceSearchResponse.model_validate(sap_call.structured_content)
        assert any(hit.chunk_id == inserted_chunks[1].id for hit in sap_response.results)

        # 3) Verify `get_opportunity`, `get_application_status`, and `get_action_draft`.
        opp_call = await client.call_tool(
            "get_opportunity",
            {"opportunity_id": str(tracked_opp.id)},
        )
        assert opp_call.is_error is False
        jsonschema.validate(
            instance=opp_call.structured_content,
            schema=tools_by_name["get_opportunity"].output_schema or {},
        )
        opp_response = OpportunityResponse.model_validate(opp_call.structured_content)
        assert opp_response.id == tracked_opp.id
        assert {r.id for r in opp_response.requirements} == {r.id for r in tracked_reqs}

        tracked_status_call = await client.call_tool(
            "get_application_status",
            {"opportunity_id": str(tracked_opp.id)},
        )
        assert tracked_status_call.is_error is False
        jsonschema.validate(
            instance=tracked_status_call.structured_content,
            schema=tools_by_name["get_application_status"].output_schema or {},
        )
        tracked_status = ApplicationStatusResponse.model_validate(
            tracked_status_call.structured_content
        )
        assert tracked_status.tracked is True
        assert tracked_status.tracker_id == tracker_entry.id
        assert tracked_status.stage is ApplicationStage.PREPARING

        untracked_status_call = await client.call_tool(
            "get_application_status",
            {"opportunity_id": str(untracked_opp.id)},
        )
        assert untracked_status_call.is_error is False
        untracked_status = ApplicationStatusResponse.model_validate(
            untracked_status_call.structured_content
        )
        assert untracked_status.tracked is False
        assert untracked_status.stage is None

        draft_call = await client.call_tool(
            "get_action_draft",
            {"draft_id": str(draft_row.id)},
        )
        assert draft_call.is_error is False
        jsonschema.validate(
            instance=draft_call.structured_content,
            schema=tools_by_name["get_action_draft"].output_schema or {},
        )
        draft_record = ActionDraftRecord.model_validate(draft_call.structured_content)
        assert draft_record.id == draft_row.id
        assert draft_record.status is ActionDraftStatus.PENDING_APPROVAL

    # Read-only invariant: draft remains pending_approval and untracked opportunity stays untracked.
    refreshed_draft = await draft_repo.get_by_id(draft_row.id)
    assert refreshed_draft is not None
    assert refreshed_draft.status is ActionDraftStatus.PENDING_APPROVAL
    assert await tracker_repo.get_by_opportunity_id(untracked_opp.id) is None


async def test_mcp_stdio_and_streamable_http_transports_search_postgres(
    integration_database_url: str,
) -> None:
    unique_token = f"mcpwire{uuid.uuid4().hex}"
    canonical_uri = f"file:///proofpath-mcp-transport/{uuid.uuid4()}"
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        embedding_model="hash",
    )
    embedder = EmbeddingClient(
        HashEmbeddingBackend(dimensions=EMBEDDING_DIM),
        model="hash",
        dimensions=EMBEDDING_DIM,
    )
    chunk_text = f"Verified transport integration proof token {unique_token} with pgvector."
    [chunk_vec] = await embedder.embed([chunk_text])

    engine = create_db_engine(settings=settings)
    factory = create_session_factory(engine)
    source_id: uuid.UUID | None = None
    try:
        async with factory.begin() as session:
            repo = EvidenceRepo(session)
            source = await repo.create_source(
                source_type=EvidenceSourceType.EXPERIENCE_NOTE,
                title="Transport integration note",
                canonical_uri=canonical_uri,
                content_sha256=_digest("transport"),
            )
            source_id = source.id
            await repo.add_chunks(
                source.id,
                [
                    EvidenceChunkInput(
                        chunk_index=0,
                        section_label="Transport",
                        content=chunk_text,
                        token_count=10,
                        embedding=chunk_vec,
                    )
                ],
            )

        # 1) Live `stdio` subprocess transport (`python -m proofpath.mcp_server.server`).
        stdio_env = {
            **os.environ,
            "DATABASE_URL": settings.asyncpg_database_url,
            "EMBEDDING_MODEL": "hash",
            "LOG_LEVEL": "WARNING",
        }
        stdio_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "proofpath.mcp_server.server", "--transport", "stdio"],
            env=stdio_env,
        )
        async with Client(stdio_params) as stdio_client_conn:
            stdio_tools = await stdio_client_conn.list_tools()
            assert {t.name for t in stdio_tools.tools} == set(READ_ONLY_TOOL_NAMES)
            stdio_res = await stdio_client_conn.call_tool(
                "search_evidence", {"query": unique_token, "top_k": 3}
            )
            assert stdio_res.is_error is False
            stdio_parsed = EvidenceSearchResponse.model_validate(stdio_res.structured_content)
            assert any(item.source_id == source_id for item in stdio_parsed.results)

        # 2) Live Streamable HTTP transport (`uvicorn` over loopback TCP).
        http_app = create_streamable_http_app(settings=settings, host="127.0.0.1")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

        ready = asyncio.Event()

        class _ReadyServer(uvicorn.Server):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets=sockets)
                ready.set()

        uv_server = _ReadyServer(
            uvicorn.Config(http_app, host="127.0.0.1", port=port, log_level="warning")
        )
        serve_task = asyncio.create_task(uv_server.serve(sockets=[sock]))
        try:
            await asyncio.wait_for(ready.wait(), timeout=5.0)
            async with Client(f"http://127.0.0.1:{port}/mcp") as http_client_conn:
                http_tools = await http_client_conn.list_tools()
                assert {t.name for t in http_tools.tools} == set(READ_ONLY_TOOL_NAMES)
                http_res = await http_client_conn.call_tool(
                    "search_evidence", {"query": unique_token, "top_k": 3}
                )
                assert http_res.is_error is False
                http_parsed = EvidenceSearchResponse.model_validate(http_res.structured_content)
                assert any(item.source_id == source_id for item in http_parsed.results)
        finally:
            uv_server.should_exit = True
            await serve_task
            sock.close()
    finally:
        if source_id is not None:
            async with factory.begin() as session:
                await session.execute(delete(EvidenceSource).where(EvidenceSource.id == source_id))
        await engine.dispose()
