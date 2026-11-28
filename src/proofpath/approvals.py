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

Retries are safe in both directions: if the decision committed but the resume did
not (process death in between), re-sending the same request completes the resume
without re-deciding. If the run was already finalised, the decision request still
returns the current state instead of failing.

Side effects remain out of scope: nothing here mutates ``application_tracker``,
calendars, or audit logs. Those are Sub-slice 2.1.3 executors, which run only for
drafts a human approved here.
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
from proofpath.graph.approval import ApprovalError, ApprovalStateError
from proofpath.graph.dependencies import GraphDependencies
from proofpath.graph.resume import resume_approval_run
from proofpath.schemas.actions import ActionDraftRecord

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


async def _resume_if_paused(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver,
    settings: Settings,
    dependencies: GraphDependencies,
    thread_id: str,
    draft_id: uuid.UUID,
    decision: DraftDecision,
    run_status: AnalysisRunStatus,
) -> None:
    """Resume a still-paused run; never fail the request on resume trouble."""
    if run_status is not AnalysisRunStatus.AWAITING_APPROVAL:
        return
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
    checkpointer: AsyncPostgresSaver,
    settings: Settings,
    dependencies: GraphDependencies,
) -> DraftDecisionResult:
    """Approve or reject one draft and finish its paused run, idempotently.

    Raises ``DraftNotFoundError`` for an unknown draft or run and
    ``DraftConflictError`` when the draft is already in a contradictory state.
    Container/model configuration must be validated by the caller before this
    runs, so a failure here cannot leave a half-applied decision.
    """
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'.")
    already_processed, run_status, thread_id = await _apply_decision(
        session_factory, draft_id=draft_id, decision=decision
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
        )
