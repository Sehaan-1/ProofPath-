"""Approval decisions for action drafts (Sub-slice 2.1.2).

This is the service layer behind the approval API. One decision does two things,
in this order:

1. **Commit the human decision exactly once.** ``action_drafts`` moves from
   ``pending_approval`` to ``approved``/``rejected`` through a single
   compare-and-set (``DraftRepo.claim_decision``). A repeated request, a retried
   request, or two concurrent requests therefore produce at most one winner; the
   others observe the already-decided row. Draft rows are unique on
   ``idempotency_key`` (``proofpath:<run_id>:<action_type>``), which stays the
   stable key the approved-only executors (Sub-slice 2.1.3) must dedupe side
   effects on — approving twice can never create a second tracker entry or
   calendar block.
2. **Finish the paused run.** The approval gate opened in Sub-slice 2.1.1 is
   resumed with ``Command(resume={"decision": ..., "draft_id": ...})`` on the
   run's own ``thread_id``; that resume is itself claimed atomically.
3. **Materialize the approved side effect.** Only after the run is finished does
   the approved-only executor run, so the resume guard still sees the draft as
   ``approved`` and a resume failure can never strand a half-applied effect.

Retries are safe in both directions: if the decision committed but the resume did
not (process death in between), re-sending the same request completes the resume
without re-deciding. If the run was already finalised, the decision request still
returns the current state instead of failing.

Side effects are delegated to the approved-only executors (Sub-slice 2.1.3):
after an approval is committed, ``proofpath.execution`` advances the tracker or
materializes the calendar task and appends the immutable audit record. Rejections
never execute anything, and a replayed approval is deduped by the draft row itself
(``executed`` -> ``already_executed``, no second side effect).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import ActionDraftStatus, AnalysisRunStatus
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.draft import (
    DraftDecision,
    DraftRepo,
    decision_satisfied,
    draft_record,
)
from proofpath.execution import (
    DraftExecution,
    DraftExecutor,
    DraftNotExecutableError,
    execution_record,
)
from proofpath.execution import (
    DraftNotFoundError as ExecutionDraftNotFoundError,
)
from proofpath.graph.approval import ApprovalError, ApprovalStateError
from proofpath.graph.dependencies import GraphDependencies
from proofpath.graph.resume import resume_approval_run
from proofpath.schemas.actions import ActionDraftRecord, DraftDecisionResponse

logger = logging.getLogger(__name__)


class DraftNotFoundError(LookupError):
    """The draft (or its analysis run) does not exist."""


class DraftConflictError(RuntimeError):
    """The draft cannot accept this decision in its current state."""


@dataclass(frozen=True, slots=True)
class DraftDecisionResult:
    """Committed decision plus the run state it left behind."""

    draft: ActionDraftRecord
    decision: DraftDecision
    already_processed: bool
    run_status: AnalysisRunStatus
    thread_id: str
    execution: DraftExecution | None = None


async def _apply_decision(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    draft_id: uuid.UUID,
    decision: DraftDecision,
) -> tuple[bool, AnalysisRunStatus, str]:
    """Claim the decision atomically; return (already_processed, run status, thread)."""
    async with session_factory.begin() as session:
        repo = DraftRepo(session)
        draft = await repo.get_by_id(draft_id)
        if draft is None:
            raise DraftNotFoundError("Action draft does not exist.")
        run = await AnalysisRepo(session).get_run_by_id(draft.analysis_run_id)
        if run is None:
            raise DraftNotFoundError("Analysis run for this draft does not exist.")
        if draft.status is ActionDraftStatus.PENDING_APPROVAL:
            if await repo.claim_decision(draft_id, decision=decision):
                return False, run.status, run.thread_id
            # Lost a concurrent race: accept it only if it landed on our decision.
            await session.refresh(draft)
            if not decision_satisfied(draft.status, decision):
                raise DraftConflictError("Action draft was decided concurrently.")
            return True, run.status, run.thread_id
        if decision_satisfied(draft.status, decision):
            return True, run.status, run.thread_id
        raise DraftConflictError(f"Action draft is already '{draft.status.value}'.")


async def _execute_if_approved(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    draft_id: uuid.UUID,
    decision: DraftDecision,
) -> DraftExecution | None:
    """Run the approved-only side effect for an approved draft, at most once.

    Called on every approval request, including replays: the executor itself decides
    between `executed` and `already_executed`, so a retry after a crash between
    "decision committed" and "side effect written" heals instead of duplicating.
    """
    if decision != "approved":
        return None
    executor = DraftExecutor(session_factory=session_factory, settings=settings)
    try:
        return await executor.execute(draft_id)
    except (DraftNotExecutableError, ExecutionDraftNotFoundError):
        # Draft is not in an executable state (e.g. an earlier execution failed and
        # marked it `failed`); the committed decision still stands.
        logger.warning("Approved draft %s is not executable; no side effect written.", draft_id)
        return None


async def _resume_if_paused(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver | None,
    settings: Settings,
    dependencies: GraphDependencies | None,
    thread_id: str,
    draft_id: uuid.UUID,
    decision: DraftDecision,
    run_status: AnalysisRunStatus,
) -> None:
    """Resume a still-paused run; never fail the request on resume trouble."""
    if run_status is not AnalysisRunStatus.AWAITING_APPROVAL:
        return
    if checkpointer is None or dependencies is None:  # pragma: no cover - guarded above
        raise RuntimeError("A paused run cannot be resumed without a checkpointer.")
    try:
        await resume_approval_run(
            checkpointer=checkpointer,
            session_factory=session_factory,
            settings=settings,
            thread_id=thread_id,
            decision=decision,
            draft_id=draft_id,
            parser=dependencies.parser,
            embedder=dependencies.embedder,
            comparator=dependencies.comparator,
            sprint_planner=dependencies.sprint_planner,
        )
    except ApprovalStateError:
        # Another request resumed it first; the decision itself already stands.
        return
    except ApprovalError:
        # The decision is committed and the run is marked failed by the helper.
        logger.warning("Approval resume failed; decision remains committed.")


async def decide_draft(
    *,
    draft_id: uuid.UUID,
    decision: DraftDecision,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver | None = None,
    settings: Settings,
    dependencies: GraphDependencies | None = None,
) -> DraftDecisionResult:
    """Approve or reject one draft and finish its paused run, idempotently.

    Raises ``DraftNotFoundError`` for an unknown draft or run and
    ``DraftConflictError`` when the draft is already in a contradictory state.
    Container/model configuration must be validated by the caller before this
    runs, so a failure here cannot leave a half-applied decision.

    ``checkpointer``/``dependencies`` are only needed to finish a run that is
    still ``awaiting_approval`` (every API caller passes them); a decision on a
    finalised run works without them, which is what lets the MCP approval tool
    decide historical drafts on a key-less server.
    """
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'.")
    already_processed, run_status, thread_id = await _apply_decision(
        session_factory, draft_id=draft_id, decision=decision
    )
    if run_status is AnalysisRunStatus.AWAITING_APPROVAL and (
        checkpointer is None or dependencies is None
    ):
        raise ValueError(
            "Resuming a paused analysis run requires a checkpointer and graph dependencies."
        )
    await _resume_if_paused(
        session_factory=session_factory,
        checkpointer=checkpointer,
        settings=settings,
        dependencies=dependencies,
        thread_id=thread_id,
        draft_id=draft_id,
        decision=decision,
        run_status=run_status,
    )
    # Side effects run last: the graph resume guard requires the draft to still be
    # `approved`, and a failed resume must never leave the tracker half-updated.
    execution = await _execute_if_approved(
        session_factory=session_factory,
        settings=settings,
        draft_id=draft_id,
        decision=decision,
    )

    async with session_factory() as session:
        stored = await DraftRepo(session).get_by_id(draft_id)
        if stored is None:  # pragma: no cover - the draft existed moments ago
            raise DraftNotFoundError("Action draft does not exist.")
        run = await AnalysisRepo(session).get_run_by_id(stored.analysis_run_id)
        final_status = run.status if run is not None else AnalysisRunStatus.FAILED
        return DraftDecisionResult(
            draft=draft_record(stored),
            decision=decision,
            already_processed=already_processed,
            run_status=final_status,
            thread_id=thread_id,
            execution=execution,
        )


def decision_response(result: DraftDecisionResult) -> DraftDecisionResponse:
    """Map a committed decision (plus its execution) to its API/MCP response contract."""
    return DraftDecisionResponse(
        draft=result.draft,
        decision=result.decision,
        already_processed=result.already_processed,
        run_status=result.run_status,
        thread_id=result.thread_id,
        execution=execution_record(result.execution) if result.execution is not None else None,
    )
