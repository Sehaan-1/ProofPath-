"""Resume an approval-gated analysis run on its own `thread_id` (Sub-part 2.1.1b).

The run paused in PostgreSQL at the ``approval_gate`` node. Resuming rebuilds the
same graph for the *same* thread and invokes it with

    Command(resume={"decision": "approved" | "rejected", "draft_id": "<uuid>"})

The pause is consumed durably: the run is atomically claimed out of
``awaiting_approval`` before the resume is attempted, so two concurrent approvals
cannot both resume (and therefore cannot both execute later side effects).

This module never approves or executes a draft. If the final output fails
revalidation after the human pause, it retires the gate's still-actionable
drafts in the same transaction as the failed status. The approval service owns
human decisions; approved-only executors own tracker and calendar effects.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import ActionDraft, ActionDraftStatus, AnalysisRun, AnalysisRunStatus
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.draft import DraftRepo
from proofpath.graph.approval import (
    ApprovalError,
    ApprovalStateError,
    parse_resume_value,
    resume_value,
)
from proofpath.graph.nodes import EvidenceComparator, SprintPlanner
from proofpath.graph.state import ApprovalDecision
from proofpath.graph.workflow import build_workflow, workflow_config
from proofpath.ingestion.analyse_role import RoleParserBackend
from proofpath.policy.output_guard import OUTPUT_CONTRACT_REJECTED
from proofpath.retrieval.embeddings import EmbeddingClient


@dataclass(frozen=True, slots=True)
class ApprovalResumeResult:
    """Outcome of one resume attempt against a single paused run."""

    analysis_run_id: uuid.UUID
    thread_id: str
    decision: ApprovalDecision
    status: AnalysisRunStatus
    state: Mapping[str, Any]


def _eligible_statuses(decision: str) -> frozenset[ActionDraftStatus]:
    """Draft states a decision may resume from (2.1.2 may commit the draft first)."""
    decided = ActionDraftStatus.APPROVED if decision == "approved" else ActionDraftStatus.REJECTED
    return frozenset({ActionDraftStatus.PENDING_APPROVAL, decided})


def _validate_draft(
    run: AnalysisRun, draft: ActionDraft | None, decision: ApprovalDecision
) -> ActionDraft:
    if draft is None or draft.analysis_run_id != run.id:
        raise ApprovalStateError("Draft does not belong to this run.")
    if draft.status not in _eligible_statuses(decision.decision):
        raise ApprovalStateError("Draft is not awaiting approval for this decision.")
    return draft


async def _load_paused_run(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    thread_id: str,
    decision: ApprovalDecision,
) -> AnalysisRun:
    """Read the run and prove the target draft is resumable (no writes)."""
    async with session_factory() as session:
        run = await AnalysisRepo(session).get_run_by_thread_id(thread_id, include_drafts=True)
        if run is None:
            raise LookupError("Analysis run not found for this thread.")
        if run.status is not AnalysisRunStatus.AWAITING_APPROVAL:
            raise ApprovalStateError("Analysis run is not awaiting approval.")
        draft = next((item for item in run.action_drafts if item.id == decision.draft_id), None)
        _validate_draft(run, draft, decision)
        return run


async def _set_status(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
    status: AnalysisRunStatus,
    *,
    invalidate_drafts: Collection[UUID] = (),
) -> None:
    async with session_factory.begin() as session:
        if invalidate_drafts:
            await DraftRepo(session).invalidate_gate_drafts(run_id, invalidate_drafts)
        await AnalysisRepo(session).update_run_metrics(run_id, status=status)


async def resume_approval_run(
    *,
    checkpointer: AsyncPostgresSaver,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    thread_id: str,
    decision: str,
    draft_id: UUID,
    parser: RoleParserBackend,
    embedder: EmbeddingClient,
    comparator: EvidenceComparator,
    sprint_planner: SprintPlanner,
    calendar_start: datetime | None = None,
    calendar_end: datetime | None = None,
    step_budget: int = 50,
    top_k: int = 8,
) -> ApprovalResumeResult:
    """Resume one paused run with a human decision bound to its `thread_id`.

    Raises ``LookupError`` when no run owns the thread, ``ApprovalStateError`` when
    the run/draft is not resumable, and ``ApprovalPayloadError`` when the resume
    value is malformed. A node failure during resume is *not* raised: the returned
    result carries ``status=FAILED`` and the sanitized ``errors`` in graph state.
    """
    if not thread_id.strip():
        raise ValueError("thread_id must not be blank")
    parsed = parse_resume_value(decision=decision, draft_id=draft_id)

    run = await _load_paused_run(session_factory, thread_id=thread_id, decision=parsed)
    workflow = build_workflow(
        checkpointer=checkpointer,
        session_factory=session_factory,
        settings=settings,
        analysis_run_id=run.id,
        thread_id=thread_id,
        parser=parser,
        embedder=embedder,
        comparator=comparator,
        sprint_planner=sprint_planner,
        calendar_start=calendar_start,
        calendar_end=calendar_end,
        step_budget=step_budget,
        top_k=top_k,
    )
    paused = await workflow.apending_approval()
    if paused is None or paused.analysis_run_id != run.id:
        raise ApprovalStateError("Analysis run is not paused at the approval gate.")
    if parsed.draft_id not in paused.draft_ids:
        raise ApprovalStateError("Draft is not part of the pending approval request.")

    # Consume the pause: only one resume may proceed past this point.
    async with session_factory.begin() as session:
        claimed = await AnalysisRepo(session).claim_awaiting_approval(run.id)
    if not claimed:
        raise ApprovalStateError("Analysis run is no longer awaiting approval.")

    try:
        state = await workflow.aresume(resume_value(parsed), workflow_config(thread_id))
    except asyncio.CancelledError:
        await _set_status(session_factory, run.id, AnalysisRunStatus.FAILED)
        raise
    except Exception as exc:  # process-level failures are not node failures
        await _set_status(session_factory, run.id, AnalysisRunStatus.FAILED)
        raise ApprovalError("Approval resume could not be completed.") from exc

    errors = list(state.get("errors") or ())
    interrupted = bool(state.get("__interrupt__"))
    recorded = state.get("approval_decision")
    if recorded is not None:
        recorded = (
            recorded
            if isinstance(recorded, ApprovalDecision)
            else ApprovalDecision.model_validate(recorded)
        )
    if recorded is None and not errors and not interrupted:
        await _set_status(session_factory, run.id, AnalysisRunStatus.FAILED)
        raise ApprovalError("Approval gate resumed without recording a decision.")
    if recorded is not None and recorded != parsed:
        await _set_status(session_factory, run.id, AnalysisRunStatus.FAILED)
        raise ApprovalError("Approval gate recorded a decision that was not requested.")

    status = AnalysisRunStatus.FAILED if errors or interrupted else AnalysisRunStatus.COMPLETED
    rejected = any(error["code"] == OUTPUT_CONTRACT_REJECTED for error in errors)
    await _set_status(
        session_factory, run.id, status, invalidate_drafts=paused.draft_ids if rejected else ()
    )
    return ApprovalResumeResult(
        analysis_run_id=run.id,
        thread_id=thread_id,
        decision=parsed,
        status=status,
        state=state,
    )


__all__ = ["ApprovalResumeResult", "resume_approval_run"]
