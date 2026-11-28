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

Every invocation (including approval resumes) is wrapped by the run's
observability tracer (Sub-slice 2.4.1): the configured backend's LangGraph
callback handler is attached to the invocation config and bound to one unified
trace ID derived from the analysis run, then flushed when the invocation ends.
With ``OBSERVABILITY_BACKEND=none`` this adds no callbacks and no I/O.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from functools import partial
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import AuditActor
from proofpath.db.repositories.analysis import AnalysisRepo, ClaimVerdictInput
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
from proofpath.observability.tracing import (
    ANALYSIS_TRACE_NAME,
    NullRunTracer,
    RunTracer,
    create_run_tracer,
)
from proofpath.observability.usage import collect_llm_usage
from proofpath.policy.ledger_validator import LedgerValidator
from proofpath.policy.output_guard import (
    OUTPUT_CONTRACT_REJECTED,
    OutputContractError,
    optional_output_guard,
    validate_final_output,
)
from proofpath.policy.tool_trace import collect_tool_calls
from proofpath.retrieval.embeddings import EmbeddingClient
from proofpath.retrieval.hybrid_search import HybridSearcher
from proofpath.schemas.evidence import EvidenceSnippet

MAX_GRAPH_STEPS = 100


def _retrieval_span_metadata(
    evidence: Mapping[str, Sequence[EvidenceSnippet]],
) -> dict[str, object]:
    """Summarize retrieval for the ``retrieve_evidence`` span (Sub-part 2.4.2a).

    Records the total retrieved chunk count, per-requirement counts, and the
    hybrid RRF similarity scores so they are queryable on the span without
    parsing its full output blob.
    """
    scores: list[float] = []
    count = 0
    per_requirement: dict[str, int] = {}
    for requirement_id, snippets in evidence.items():
        per_requirement[requirement_id] = len(snippets)
        count += len(snippets)
        scores.extend(round(s.score, 6) for s in snippets if s.score is not None)
    return {
        "retrieved_chunk_count": count,
        "requirements_searched": len(evidence),
        "chunks_per_requirement": per_requirement,
        "similarity_scores": scores,
    }


Node = Callable[[ProofPathState, RunnableConfig], Awaitable[dict[str, object]]]


class StepBudgetExceededError(RuntimeError):
    """Raised internally when a run has consumed its execution budget."""


class BoundedWorkflow:
    """Async graph facade that enforces a fixed thread, recursion limit and sync durability.

    Also the single tracing attachment point (Sub-part 2.4.1b): every execution
    enters the run tracer's invocation context, merges its callback handlers
    into the invocation config, and flushes buffered spans afterwards. Direct
    construction without a tracer (facade-level tests) is untraced;
    ``build_workflow`` always binds a run-scoped tracer.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[ProofPathState, None, ProofPathState, ProofPathState],
        thread_id: str,
        step_budget: int,
        tracer: RunTracer | None = None,
    ) -> None:
        self._graph = graph
        self._thread_id = thread_id
        self._step_budget = step_budget
        self._tracer: RunTracer = tracer if tracer is not None else NullRunTracer(trace_id="")

    @property
    def trace_id(self) -> str:
        """Unified observability trace ID for this run (empty when unbound)."""
        return self._tracer.trace_id

    def _config(
        self,
        config: RunnableConfig | None,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> RunnableConfig:
        result: RunnableConfig = dict(config or {})  # type: ignore[assignment]
        configurable = dict(result.get("configurable", {}))
        if configurable.get("thread_id", self._thread_id) != self._thread_id:
            raise ValueError("Graph invocation thread_id does not match analysis run")
        configurable["thread_id"] = self._thread_id
        result["configurable"] = configurable
        result["recursion_limit"] = self._step_budget + 4
        if callbacks:
            existing = result.get("callbacks")
            if existing is None:
                merged = list(callbacks)
            elif isinstance(existing, list):
                merged = [*existing, *callbacks]
            else:
                raise ValueError("Graph callbacks must be provided as a list of handlers")
            result["callbacks"] = merged
            result.setdefault("run_name", ANALYSIS_TRACE_NAME)
        return result

    async def ainvoke(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute with a fixed upper bound; caller overrides cannot relax it."""
        kwargs["durability"] = "sync"
        try:
            with self._tracer.invocation() as callbacks:
                return await self._graph.ainvoke(input, self._config(config, callbacks), **kwargs)
        finally:
            await self._tracer.flush()

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
        """Stream with the same bound, durability and tracing as invocation."""
        kwargs["durability"] = "sync"
        try:
            with self._tracer.invocation() as callbacks:
                async for item in self._graph.astream(
                    input, self._config(config, callbacks), **kwargs
                ):
                    yield item
        finally:
            await self._tracer.flush()

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
    tracer: RunTracer | None = None,
) -> BoundedWorkflow:
    """Build a graph tied to one persisted run and thread; no memory fallback.

    Calendar times are optional; if provided they must be timezone-aware.
    Invocation enforces a hard recursion limit.
    Reopen the saver before rebuilding
    a graph after process restart, with the same run, thread and dependencies.

    The run's observability tracer is resolved from settings unless explicitly
    supplied (Sub-slice 2.4.1); a misconfigured backend fails closed here,
    before any node runs. Rebuilding for an approval resume derives the *same*
    unified trace ID, so the resume joins the original trace.
    """
    if not thread_id.strip() or not 1 <= step_budget <= MAX_GRAPH_STEPS or not 1 <= top_k <= 50:
        raise ValueError("Invalid thread_id, step_budget or top_k")
    if not isinstance(checkpointer, AsyncPostgresSaver):
        raise TypeError("A live AsyncPostgresSaver is required")
    run_tracer = (
        tracer
        if tracer is not None
        else create_run_tracer(settings, analysis_run_id=analysis_run_id, thread_id=thread_id)
    )

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

    async def check_output(state: ProofPathState, session: AsyncSession) -> None:
        if not state.get("policy_verified") or state.get("policy_retry"):
            raise OutputContractError("The comparison policy has not verified these verdicts.")
        await validate_final_output(
            state.get("verdicts", []),
            state.get("tool_trace"),
            lookup=partial(EvidenceRepo(session).get_existing_chunk_ids, for_reference=True),
            extra_guard=optional_output_guard(settings),
        )

    async def publish_output(state: ProofPathState, session: AsyncSession) -> None:
        await check_output(state, session)
        await AnalysisRepo(session).upsert_verdicts(
            analysis_run_id,
            [
                ClaimVerdictInput(
                    requirement_id=v.requirement_id,
                    verdict=v.verdict,
                    claim_summary=v.claim_summary,
                    cited_chunk_ids=v.cited_chunk_ids,
                    policy_downgraded=v.policy_downgraded,
                    gap_notes=v.gap_notes,
                )
                for v in state.get("verdicts", [])
            ],
        )

    async def drafts(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        async with session_factory.begin() as session:
            updates = await create_action_drafts_node(
                state,
                session=session,
                analysis_run_id=analysis_run_id,
                calendar_start=calendar_start,
                calendar_end=calendar_end,
            )
            # The last tool-using node and the first publication boundary. Include
            # this node's actual calls, and roll back its pending drafts on rejection.
            await publish_output(state, session)
            return {
                "draft_actions": updates["draft_actions"],
                "step_count": updates["step_count"],
                "output_validated": True,
            }

    async def approval(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
        guard(state, config)
        if not state.get("output_validated"):
            raise OutputContractError("The analysis has no validated final output.")
        decision = await approval_gate_node(state, analysis_run_id=analysis_run_id)
        # Only reached on resume. Evidence may have disappeared during the human
        # pause; validate again before returning the terminal state. No new draft,
        # verdict or decision is committed by this read-only check.
        async with session_factory.begin() as session:
            await check_output(state, session)
        return decision

    def wrap_node(name: str, node: Node) -> Node:
        async def guarded(state: ProofPathState, config: RunnableConfig) -> dict[str, object]:
            # Thread mismatch is an authorization/configuration failure, not a graph error.
            if config.get("configurable", {}).get("thread_id") != thread_id:
                raise ValueError("Graph invocation thread_id does not match analysis run")
            working = state.copy()
            try:
                with (
                    collect_tool_calls(state.get("tool_trace")) as calls,
                    collect_llm_usage(state.get("llm_usage")) as usage,
                ):
                    working["tool_trace"] = calls
                    working["llm_usage"] = usage
                    updates = await node(working, config)
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
                if isinstance(exc, OutputContractError):
                    error = contract_error(name, exc)
                updates = {"pending_error": error}
            # Best-effort span enrichment; never affects the node's outcome.
            if name == "retrieve_evidence":
                evidence = updates.get("retrieved_evidence")
                if isinstance(evidence, dict):
                    run_tracer.annotate_current_span(metadata=_retrieval_span_metadata(evidence))
            # Only the dispatcher supplies these channels, including a failed node's calls.
            if "tool_trace" in working:
                updates["tool_trace"] = working["tool_trace"]
            if "llm_usage" in working:
                updates["llm_usage"] = working["llm_usage"]
            return updates

        return guarded

    def contract_error(node: str, exc: OutputContractError) -> GraphError:
        return {
            "node": node,
            "code": OUTPUT_CONTRACT_REJECTED,
            "message": str(exc),
            "retryable": False,
        }

    async def fallback_error_node(state: ProofPathState) -> dict[str, object]:
        error = state.get("pending_error")
        if error is None:
            raise ValueError("Fallback requires a pending error")
        errors = [*state.get("errors", []), error]
        # Preserve a useful partial report on unrelated failures, but only after
        # checking its actual citations and *complete* trace (including the failure).
        publish_partial = False
        if state.get("policy_verified") and error["code"] != OUTPUT_CONTRACT_REJECTED:
            try:
                async with session_factory.begin() as session:
                    await publish_output(state, session)
                publish_partial = True
            except OutputContractError as exc:
                errors.append(contract_error("final_output", exc))
            except Exception:
                errors.append(
                    contract_error(
                        "final_output", OutputContractError("Final validation unavailable.")
                    )
                )
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
            "output_validated": publish_partial,
        }
        if not publish_partial:
            updates.update(
                {
                    "verdicts": [],
                    "contribution_sprint": None,
                    "draft_actions": [],
                    "approval_request": None,
                    "approval_decision": None,
                    "policy_verified": False,
                }
            )
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
        ("policy_verifier", policy),
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
    builder.add_conditional_edges(
        "compare_evidence", route("policy_verifier"), ["policy_verifier", "fallback_error"]
    )
    builder.add_conditional_edges(
        "policy_verifier", route_policy, ["compare_evidence", "sprint", "fallback_error"]
    )
    builder.add_conditional_edges("sprint", route("drafts"), ["drafts", "fallback_error"])
    builder.add_conditional_edges(
        "drafts", route("approval_gate"), ["approval_gate", "fallback_error"]
    )
    builder.add_conditional_edges("approval_gate", route_approval, [END, "fallback_error"])
    builder.add_edge("fallback_error", END)
    return BoundedWorkflow(
        builder.compile(checkpointer=checkpointer), thread_id, step_budget, run_tracer
    )


def workflow_config(thread_id: str, *, step_budget: int = 50) -> RunnableConfig:
    """Create a thread configuration with a recursion limit matching the budget."""
    if not 1 <= step_budget <= MAX_GRAPH_STEPS:
        raise ValueError("step_budget is outside the allowed range")
    config = thread_config(thread_id)
    config["recursion_limit"] = step_budget + 4
    return config
