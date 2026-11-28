"""Bind graph nodes to services and a mandatory durable checkpointer.

Compile inside ``postgres_checkpointer``'s context, and keep that context open
while using the graph. Invoke with the same thread_config on resume and
``durability='sync'``. Each database node commits before returning; parsing and
approval-draft creation are replay-safe if a checkpoint write then fails.
``create_action_drafts`` is followed by ``approval_gate``, which pauses the run
with LangGraph's native ``interrupt()``; resume the *same* thread with
``BoundedWorkflow.aresume({"decision": ..., "draft_id": ...})``.
Node failures route to a terminal fallback with a sanitized error and best-effort
audit record. Checkpoint and process-level failures cannot be handled by a node.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import AuditActor
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.graph.approval import approval_gate_node, pending_approval_request
from proofpath.graph.checkpointer import thread_config
from proofpath.graph.nodes import (
    EvidenceComparator,
    SprintPlanner,
    analyse_role_node,
    compare_evidence_node,
    create_action_drafts_node,
    planner_route,
    planner_router,
    policy_route,
    policy_verifier_node,
    recommend_contribution_sprint_node,
    retrieve_evidence_node,
)
from proofpath.graph.state import ApprovalRequest, GraphError, ProofPathState
from proofpath.ingestion.analyse_role import RoleParserBackend, is_retryable_role_parse_error
from proofpath.policy.ledger_validator import LedgerValidator
from proofpath.retrieval.embeddings import EmbeddingClient
from proofpath.retrieval.hybrid_search import HybridSearcher

MAX_GRAPH_STEPS = 100


Node = Callable[[ProofPathState, RunnableConfig], Awaitable[dict[str, object]]]


class StepBudgetExceededError(RuntimeError):
    """Raised internally when a run has consumed its execution budget."""


class BoundedWorkflow:
    """Async graph facade that enforces a fixed thread, recursion limit and sync durability."""

    def __init__(
        self,
        graph: CompiledStateGraph[ProofPathState, None, ProofPathState, ProofPathState],
        thread_id: str,
        step_budget: int,
    ) -> None:
        self._graph = graph
        self._thread_id = thread_id
        self._step_budget = step_budget

    def _config(self, config: RunnableConfig | None) -> RunnableConfig:
        result: RunnableConfig = dict(config or {})  # type: ignore[assignment]
        configurable = dict(result.get("configurable", {}))
        if configurable.get("thread_id", self._thread_id) != self._thread_id:
            raise ValueError("Graph invocation thread_id does not match analysis run")
        configurable["thread_id"] = self._thread_id
        result["configurable"] = configurable
        result["recursion_limit"] = self._step_budget + 4
        return result

    async def ainvoke(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute with a fixed upper bound; caller overrides cannot relax it."""
        kwargs["durability"] = "sync"
        return await self._graph.ainvoke(input, self._config(config), **kwargs)

    async def aresume(
        self,
        value: object,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Resume a paused interrupt on this thread with synchronous durability.

        ``value`` is the interrupt's resume value; the approval gate expects
        ``{"decision": "approved" | "rejected", "draft_id": "<uuid>"}``.
        """
        return await self.ainvoke(Command(resume=value), config, **kwargs)

    async def apending_approval(self) -> ApprovalRequest | None:
        """Return the approval request when the thread is paused at the gate."""
        return pending_approval_request(await self.aget_state())

    async def astream(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream with the same bound and durability as invocation."""
        kwargs["durability"] = "sync"
        async for item in self._graph.astream(input, self._config(config), **kwargs):
            yield item

    async def aget_state(
        self, config: RunnableConfig | None = None, *, subgraphs: bool = False
    ) -> Any:
        """Read the checkpoint for the bound thread."""
        return await self._graph.aget_state(self._config(config), subgraphs=subgraphs)


def build_workflow(
    *,
    checkpointer: AsyncPostgresSaver,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    analysis_run_id: UUID,
    thread_id: str,
    parser: RoleParserBackend,
    embedder: EmbeddingClient,
    comparator: EvidenceComparator,
    sprint_planner: SprintPlanner,
    calendar_start: datetime | None = None,
    calendar_end: datetime | None = None,
    step_budget: int = 50,
    top_k: int = 8,
) -> BoundedWorkflow:
    """Build a graph tied to one persisted run and thread; no memory fallback.

    Calendar times are optional; if provided they must be timezone-aware.
    Invocation enforces a hard recursion limit.
    Reopen the saver before rebuilding
    a graph after process restart, with the same run, thread and dependencies.
    """
    if not thread_id.strip() or not 1 <= step_budget <= MAX_GRAPH_STEPS or not 1 <= top_k <= 50:
        raise ValueError("Invalid thread_id, step_budget or top_k")
    if not isinstance(checkpointer, AsyncPostgresSaver):
        raise TypeError("A live AsyncPostgresSaver is required")

    def guard(state: ProofPathState, config: RunnableConfig) -> None:
        if config.get("configurable", {}).get("thread_id") != thread_id:
            raise ValueError("Graph invocation thread_id does not match analysis run")
        if state.get("step_count", 0) >= min(state.get("step_budget", step_budget), step_budget):
            raise StepBudgetExceededError("Graph step budget exhausted")

    async def planner(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        async with session_factory.begin() as session:
            run = await AnalysisRepo(session).get_run_by_id(analysis_run_id)
            if (
                run is None
                or run.thread_id != thread_id
                or run.opportunity_id != state["opportunity_id"]
            ):
                raise ValueError("Analysis run, thread and opportunity do not match")
            return await planner_router(
                state, session=session, settings=settings, step_budget=step_budget
            )

    async def analyse(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        async with session_factory.begin() as session:
            return await analyse_role_node(state, session=session, parser=parser)

    async def retrieve(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        async with session_factory.begin() as session:
            return await retrieve_evidence_node(
                state, searcher=HybridSearcher(session, embedder), top_k=top_k
            )

    async def compare(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        return await compare_evidence_node(state, comparator=comparator)

    async def policy(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        async with session_factory.begin() as session:
            validator = LedgerValidator(EvidenceRepo(session), AuditRepo(session))
            return await policy_verifier_node(
                state, validator=validator, analysis_run_id=analysis_run_id
            )

    async def sprint(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        return await recommend_contribution_sprint_node(state, planner=sprint_planner)

    async def drafts(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        async with session_factory.begin() as session:
            return await create_action_drafts_node(
                state,
                session=session,
                analysis_run_id=analysis_run_id,
                calendar_start=calendar_start,
                calendar_end=calendar_end,
            )

    async def approval(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        return await approval_gate_node(state, analysis_run_id=analysis_run_id)

    def wrap_node(name: str, node: Node) -> Node:
        async def guarded(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
            # Thread mismatch is an authorization/configuration failure, not a graph error.
            if config.get("configurable", {}).get("thread_id") != thread_id:
                raise ValueError("Graph invocation thread_id does not match analysis run")
            try:
                return await node(state, config)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                exhausted = isinstance(exc, StepBudgetExceededError)
                error: GraphError = {
                    "node": name,
                    "code": "step_budget_exhausted" if exhausted else "node_failed",
                    "message": "Graph step budget exhausted."
                    if exhausted
                    else f"Node {name} failed ({type(exc).__name__}).",
                    "retryable": isinstance(exc, (TimeoutError, ConnectionError))
                    or is_retryable_role_parse_error(exc)
                    or (exc.__cause__ is not None and is_retryable_role_parse_error(exc.__cause__)),
                }
                return {"pending_error": error}

        return guarded

    async def fallback_error_node(state: ProofPathState) -> dict[str, object]:
        error = state.get("pending_error")
        if error is None:
            raise ValueError("Fallback requires a pending error")
        errors = [*state.get("errors", []), error]
        try:
            async with session_factory.begin() as session:
                await AuditRepo(session).record(
                    actor=AuditActor.AGENT,
                    event_type="graph.node_failed",
                    resource_id=analysis_run_id,
                    details_json={
                        **error,
                        "thread_id": thread_id,
                        "analysis_run_id": str(analysis_run_id),
                    },
                )
        except GraphBubbleUp:
            raise
        except Exception:
            errors.append(
                {
                    "node": "fallback_error",
                    "code": "audit_write_failed",
                    "message": "Failed to record graph failure in audit log.",
                    "retryable": False,
                }
            )
        updates: dict[str, object] = {
            "errors": errors,
            "pending_error": None,
            "policy_retry": False,
        }
        if not state.get("policy_verified", False):
            # Unverified model output must not be presented as evidence-backed.
            updates.update({"verdicts": [], "contribution_sprint": None})
        return updates

    def route(default: str) -> Callable[[ProofPathState], str]:
        def choose(state: ProofPathState) -> str:
            return "fallback_error" if state.get("pending_error") else default

        return choose

    def route_planner(state: ProofPathState) -> str:
        return "fallback_error" if state.get("pending_error") else planner_route(state)

    def route_approval(state: ProofPathState) -> str:
        return "fallback_error" if state.get("pending_error") else END

    def route_policy(state: ProofPathState) -> str:
        if state.get("pending_error"):
            return "fallback_error"
        return "compare_evidence" if policy_route(state) == "compare_evidence" else "sprint"

    builder = StateGraph(ProofPathState)
    for name, node in (
        ("planner", planner),
        ("analyse_role", analyse),
        ("retrieve_evidence", retrieve),
        ("compare_evidence", compare),
        ("policy", policy),
        ("sprint", sprint),
        ("drafts", drafts),
        ("approval_gate", approval),
    ):
        builder.add_node(name, wrap_node(name, node))  # type: ignore[call-overload]
    builder.add_node("fallback_error", fallback_error_node)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner", route_planner, ["analyse_role", "retrieve_evidence", "fallback_error"]
    )
    builder.add_conditional_edges(
        "analyse_role", route("retrieve_evidence"), ["retrieve_evidence", "fallback_error"]
    )
    builder.add_conditional_edges(
        "retrieve_evidence", route("compare_evidence"), ["compare_evidence", "fallback_error"]
    )
    builder.add_conditional_edges("compare_evidence", route("policy"), ["policy", "fallback_error"])
    builder.add_conditional_edges(
        "policy", route_policy, ["compare_evidence", "sprint", "fallback_error"]
    )
    builder.add_conditional_edges("sprint", route("drafts"), ["drafts", "fallback_error"])
    builder.add_conditional_edges(
        "drafts", route("approval_gate"), ["approval_gate", "fallback_error"]
    )
    builder.add_conditional_edges("approval_gate", route_approval, [END, "fallback_error"])
    builder.add_edge("fallback_error", END)
    return BoundedWorkflow(builder.compile(checkpointer=checkpointer), thread_id, step_budget)


def workflow_config(thread_id: str, *, step_budget: int = 50) -> RunnableConfig:
    """Create a thread configuration with a recursion limit matching the budget."""
    if not 1 <= step_budget <= MAX_GRAPH_STEPS:
        raise ValueError("step_budget is outside the allowed range")
    config = thread_config(thread_id)
    config["recursion_limit"] = step_budget + 4
    return config
