"""Durable human-approval gate for the opportunity analysis graph (Slice 2.1).

`approval_gate_node` runs after `create_action_drafts_node` and calls LangGraph's
native ``interrupt()``. The graph therefore stops with a checkpoint written to
PostgreSQL and no further node executes; the caller sees ``status =
"awaiting_approval"``. Resuming is a separate invocation of the same thread:

    Command(resume={"decision": "approved" | "rejected", "draft_id": "<uuid>"})

The gate only records the validated decision in state. It never mutates
``application_tracker``, calendars, or ``action_drafts``: the approval API
(Sub-slice 2.1.2) owns draft status transitions and approved-only executors
(Sub-slice 2.1.3) own every side effect.

Everything executed before ``interrupt()`` re-runs when the graph resumes, so the
approval request must be derived from state alone (no clock, randomness, or I/O).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Final

from langgraph.types import StateSnapshot, interrupt
from pydantic import ValidationError

from proofpath.db.models import ActionDraftStatus, ActionType
from proofpath.graph.state import (
    ApprovalDecision,
    ApprovalDraftSummary,
    ApprovalRequest,
    ProofPathState,
)
from proofpath.schemas.actions import (
    ActionDraftRecord,
    CalendarBlockPayload,
    ContributionSprintPayload,
    TrackerUpsertPayload,
)

APPROVAL_GATE_NODE: Final[str] = "approval_gate"
APPROVAL_DECISIONS: Final[frozenset[str]] = frozenset({"approved", "rejected"})
_RESUME_KEYS: Final[frozenset[str]] = frozenset({"decision", "draft_id"})
_SUMMARY_LIMIT: Final[int] = 120


class ApprovalError(RuntimeError):
    """Base class for approval-gate failures."""


class ApprovalPayloadError(ApprovalError, ValueError):
    """The resume value is not the documented ``{decision, draft_id}`` payload."""


class ApprovalStateError(ApprovalError):
    """The run or draft is not in a state that may be approved."""


def _summarize(draft: ActionDraftRecord) -> str:
    """Describe a draft for a human without copying personal draft payloads."""
    payload = draft.payload
    if isinstance(payload, TrackerUpsertPayload):
        summary = f"Update application tracker stage to '{payload.stage.value}'"
    elif isinstance(payload, CalendarBlockPayload):
        summary = f"Create calendar block '{payload.title}'"
    elif isinstance(payload, ContributionSprintPayload):
        summary = f"Propose contribution sprint '{payload.plan.title}'"
    else:  # pragma: no cover - the discriminated union is exhaustive
        raise ApprovalStateError("Unsupported action draft type")
    return summary[:_SUMMARY_LIMIT]


def build_approval_request(state: ProofPathState, *, analysis_run_id: uuid.UUID) -> ApprovalRequest:
    """Build the deterministic interrupt payload from checkpointed drafts.

    Fails closed when drafts are missing, duplicated, owned by another run, or no
    longer awaiting approval.
    """
    drafts: list[ActionDraftRecord] = list(state.get("draft_actions") or [])
    if not drafts:
        raise ApprovalStateError("Approval gate requires at least one action draft")
    draft_ids = [draft.id for draft in drafts]
    if len(draft_ids) != len(set(draft_ids)):
        raise ApprovalStateError("Action drafts must be unique")
    if any(draft.analysis_run_id != analysis_run_id for draft in drafts):
        raise ApprovalStateError("Action drafts do not belong to the analysis run")
    pending = [d for d in drafts if d.status is ActionDraftStatus.PENDING_APPROVAL]
    if not pending:
        raise ApprovalStateError("Approval gate requires at least one pending draft")
    summaries = [
        ApprovalDraftSummary(
            draft_id=draft.id,
            action_type=ActionType(draft.action_type),
            summary=_summarize(draft),
        )
        for draft in pending
    ]
    summaries.sort(key=lambda item: (item.action_type.value, str(item.draft_id)))
    return ApprovalRequest(analysis_run_id=analysis_run_id, drafts=summaries)


def parse_resume_value(*, decision: object, draft_id: object) -> ApprovalDecision:
    """Validate a decision/draft pair before any graph or database work."""
    return _parse_resume_shape({"decision": decision, "draft_id": draft_id})


def parse_approval_decision(
    payload: object, *, allowed_draft_ids: Iterable[uuid.UUID]
) -> ApprovalDecision:
    """Validate a resume value against the pending request without echoing it."""
    parsed = _parse_resume_shape(payload)
    if parsed.draft_id not in set(allowed_draft_ids):
        raise ApprovalPayloadError("Approval resume targets a draft outside this run.")
    return parsed


def _parse_resume_shape(payload: object) -> ApprovalDecision:
    if not isinstance(payload, Mapping):
        raise ApprovalPayloadError("Approval resume must be a mapping.")
    if set(payload) != _RESUME_KEYS:
        raise ApprovalPayloadError("Approval resume requires exactly 'decision' and 'draft_id'.")
    decision = payload.get("decision")
    if not isinstance(decision, str) or decision not in APPROVAL_DECISIONS:
        raise ApprovalPayloadError("Approval decision must be 'approved' or 'rejected'.")
    raw_draft_id = payload.get("draft_id")
    if not isinstance(raw_draft_id, (str, uuid.UUID)):
        raise ApprovalPayloadError("Approval resume requires a draft_id.")
    try:
        return ApprovalDecision.model_validate({"decision": decision, "draft_id": raw_draft_id})
    except ValidationError as exc:
        raise ApprovalPayloadError("Approval resume requires a valid draft_id.") from exc


def resume_value(decision: ApprovalDecision) -> dict[str, str]:
    """Return the canonical JSON-compatible `Command(resume=...)` payload."""
    return {"decision": decision.decision, "draft_id": str(decision.draft_id)}


def approval_request_from_interrupts(interrupts: Iterable[object] | None) -> ApprovalRequest | None:
    """Return the first interrupt value that is a valid approval request."""
    for item in interrupts or ():
        value = getattr(item, "value", item)
        try:
            return ApprovalRequest.model_validate(value)
        except ValidationError:
            continue
    return None


def pending_approval_request(snapshot: StateSnapshot) -> ApprovalRequest | None:
    """Return the approval request when `snapshot` is paused at the gate, else None."""
    if APPROVAL_GATE_NODE not in tuple(snapshot.next or ()):
        return None
    request = approval_request_from_interrupts(snapshot.interrupts)
    if request is not None:
        return request
    for task in snapshot.tasks or ():
        request = approval_request_from_interrupts(getattr(task, "interrupts", ()))
        if request is not None:
            return request
    return None


async def approval_gate_node(
    state: ProofPathState, *, analysis_run_id: uuid.UUID
) -> dict[str, object]:
    """Pause the graph until a human decision arrives for one pending draft.

    The first execution raises LangGraph's interrupt, which stops the run and is
    checkpointed by the caller's Postgres saver. On resume the node re-executes,
    ``interrupt()`` returns the resume value, and the validated decision is
    written to state before the graph finishes.
    """
    request = build_approval_request(state, analysis_run_id=analysis_run_id)
    decision = parse_approval_decision(
        interrupt(request.model_dump(mode="json")),
        allowed_draft_ids=request.draft_ids,
    )
    return {
        "approval_request": request,
        "approval_decision": decision,
        "step_count": state.get("step_count", 0) + 1,
    }
