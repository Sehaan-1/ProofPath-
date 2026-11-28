"""Approval API for pending action drafts (Sub-part 2.1.2a).

Listing is a read and rejecting never writes user data. Approving records the
decision, finishes the paused run, and then hands the approved draft to the
approved-only executors (Sub-slice 2.1.3), which materialize the tracker/calendar
side effect and the immutable audit record; the response reports that as
``execution``.

Approving is idempotent (Sub-part 2.1.2b): the same request re-sent after a
timeout, a retry, or a concurrent duplicate returns ``200`` with
``already_processed=true`` and the same ``approved_at``. It never applies the
decision twice, so it can never create a second tracker entry or calendar block.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request

from proofpath.api.routes import Session
from proofpath.approvals import (
    DraftConflictError,
    DraftDecisionResult,
    DraftNotFoundError,
    decide_draft,
    decision_response,
)
from proofpath.db.models import ActionDraftStatus
from proofpath.db.repositories.draft import DraftDecision, DraftRepo, draft_record
from proofpath.graph.dependencies import GraphDependenciesError, create_graph_dependencies
from proofpath.schemas.actions import ActionDraftRecord, DraftDecisionResponse

router = APIRouter(prefix="/api/v1", tags=["drafts"])


@router.get("/drafts", response_model=list[ActionDraftRecord])
async def list_drafts(
    session: Session,
    status: Annotated[ActionDraftStatus | None, Query()] = None,
    analysis_run_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ActionDraftRecord]:
    """List action drafts, typically `?status=pending_approval` for the review queue."""
    rows = await DraftRepo(session).list_drafts(
        status=status,
        analysis_run_id=analysis_run_id,
        limit=limit,
        offset=offset,
    )
    return [draft_record(row) for row in rows]


async def _decide(
    draft_id: uuid.UUID, *, decision: DraftDecision, request: Request
) -> DraftDecisionResult:
    """Shared approve/reject flow: validate wiring, then commit the decision."""
    settings = request.app.state.settings
    try:
        dependencies = create_graph_dependencies(settings)
    except GraphDependenciesError as exc:
        # Fail before writing anything: the caller can retry once configured.
        raise HTTPException(503, "Analysis model is not configured.") from exc
    try:
        return await decide_draft(
            draft_id=draft_id,
            decision=decision,
            session_factory=request.app.state.session_factory,
            checkpointer=request.app.state.checkpointer,
            settings=settings,
            dependencies=dependencies,
        )
    except DraftNotFoundError as exc:
        raise HTTPException(404, "Action draft not found.") from exc
    except DraftConflictError as exc:
        raise HTTPException(409, "Action draft was already decided differently.") from exc


@router.post("/drafts/{draft_id}/approve", response_model=DraftDecisionResponse)
async def approve_draft(draft_id: uuid.UUID, request: Request) -> DraftDecisionResponse:
    """Approve one pending draft and resume its run on the bound `thread_id`."""
    return decision_response(await _decide(draft_id, decision="approved", request=request))


@router.post("/drafts/{draft_id}/reject", response_model=DraftDecisionResponse)
async def reject_draft(draft_id: uuid.UUID, request: Request) -> DraftDecisionResponse:
    """Reject one pending draft; the run resumes and ends with no side effects."""
    return decision_response(await _decide(draft_id, decision="rejected", request=request))
