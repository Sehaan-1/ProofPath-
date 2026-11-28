"""Approval-gated MCP action tools (Sub-slice 2.1.4).

Two tools, one of which can actually change the user's data:

* ``create_action_draft`` is **side-effect free**. It persists a ``pending_approval``
  ``action_drafts`` row (and nothing else) so a human can review it. Tracker and
  calendar writes stay in the executor layer and only run after an approval.
* ``approve_action`` is the single MCP path that executes side effects. It refuses to
  run unless the draft is ``pending_approval`` **and** the caller repeats the exact
  confirmation phrase ``"approve action"``. Once confirmed it delegates to
  ``proofpath.approvals.decide_draft``, the same service the HTTP API uses, so the
  decision compare-and-set, the paused-run resume, the approved-only executors and
  the dedupe guarantees (one tracker row / one calendar task / one audit record per
  draft, replays report ``already_processed``) are identical on both surfaces.

Model-backed wiring is only built when the draft's run is still ``awaiting_approval``
and therefore needs a resume; deciding a finalised run's draft works on a key-less
server. When the wiring cannot be built, the tool fails closed *before* writing
anything.
"""

from __future__ import annotations

import uuid
from contextlib import AsyncExitStack
from typing import Any, Final

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.approvals import (
    DraftConflictError,
    DraftNotFoundError,
    decide_draft,
    decision_response,
)
from proofpath.config import Settings
from proofpath.db.models import ActionDraftStatus, ActionType, AnalysisRunStatus
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.draft import DraftRepo, draft_record
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.graph.dependencies import (
    GraphDependencies,
    GraphDependenciesError,
    create_graph_dependencies,
)
from proofpath.policy.tool_allowlist import APPROVAL_CONFIRMATION_PHRASE
from proofpath.schemas.actions import (
    ActionDraftCreationResponse,
    ActionDraftPayload,
    DraftDecisionResponse,
    TrackerUpsertPayload,
    parse_action_draft_payload,
)

__all__ = [
    "APPROVABLE_STATUSES",
    "APPROVAL_CONFIRMATION_PHRASE",
    "approve_action",
    "canonical_idempotency_key",
    "create_action_draft",
    "normalize_confirmation_phrase",
    "require_confirmation_phrase",
]

#: Statuses an approval request may still act on. `pending_approval` is the only
#: state that can *become* approved; the other two already carry that decision, so a
#: repeat call is answered idempotently (and heals a crash between decision and
#: side effect) instead of executing twice.
APPROVABLE_STATUSES: Final[frozenset[ActionDraftStatus]] = frozenset(
    {
        ActionDraftStatus.PENDING_APPROVAL,
        ActionDraftStatus.APPROVED,
        ActionDraftStatus.EXECUTED,
    }
)


def canonical_idempotency_key(analysis_run_id: uuid.UUID, action_type: ActionType) -> str:
    """Return the canonical `proofpath:<run_id>:<action_type>` draft key."""
    return f"proofpath:{analysis_run_id}:{action_type.value}"


def normalize_confirmation_phrase(raw: str) -> str:
    """Normalize a confirmation phrase for comparison (whitespace and case)."""
    return " ".join(raw.split()).casefold()


def require_confirmation_phrase(*, draft_id: uuid.UUID, confirmation_phrase: str) -> None:
    """Fail closed unless the caller repeated the expected confirmation phrase."""
    if normalize_confirmation_phrase(confirmation_phrase) != normalize_confirmation_phrase(
        APPROVAL_CONFIRMATION_PHRASE
    ):
        raise ToolError(
            "Confirmation phrase does not match. Re-send approve_action with "
            f"confirmation_phrase={APPROVAL_CONFIRMATION_PHRASE!r} to approve draft "
            f"{draft_id}; nothing was executed."
        )


def _canonical_payload(payload: ActionDraftPayload | dict[str, Any]) -> ActionDraftPayload:
    """Re-validate a caller-supplied payload into its strict discriminated contract."""
    raw = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    try:
        return parse_action_draft_payload(raw)
    except ValidationError as exc:
        raise ToolError(f"Action draft payload is invalid: {exc.error_count()} error(s).") from exc


async def create_action_draft(
    session: AsyncSession,
    *,
    analysis_run_id: uuid.UUID,
    payload: ActionDraftPayload | dict[str, Any],
    idempotency_key: str,
) -> ActionDraftCreationResponse:
    """Persist a `pending_approval` action draft and commit; never executes it."""
    parsed = _canonical_payload(payload)
    action_type = ActionType(parsed.action_type)
    run = await AnalysisRepo(session).get_run_by_id(analysis_run_id)
    if run is None:
        raise ToolError(f"Analysis run {analysis_run_id} not found.")
    if run.status is AnalysisRunStatus.AWAITING_APPROVAL:
        # A paused run already exposes its own drafts to the review queue, and its
        # approval gate only accepts decisions for those drafts. Queueing an extra
        # draft here would make `approve_action` resume with a draft id the gate
        # rejects, so refuse it instead.
        raise ToolError(
            f"Analysis run {analysis_run_id} is paused at the approval gate; approve or "
            "reject its queued drafts first. No draft was created."
        )

    if isinstance(parsed, TrackerUpsertPayload) and parsed.opportunity_id != run.opportunity_id:
        raise ToolError("Tracker draft opportunity_id must match the analysis run's opportunity.")

    expected_key = canonical_idempotency_key(analysis_run_id, action_type)
    if idempotency_key.strip() != expected_key:
        raise ToolError(
            "idempotency_key must be exactly the canonical draft key "
            f"{expected_key!r} for this run and action type."
        )

    repo = DraftRepo(session)
    draft, created = await repo.create_or_get_draft(
        analysis_run_id=analysis_run_id,
        action_type=action_type,
        payload_json=parsed.model_dump(mode="json"),
        idempotency_key=expected_key,
    )
    if not created:
        try:
            stored = parse_action_draft_payload(draft.payload_json)
        except ValidationError as exc:  # pragma: no cover - stored rows are validated
            raise ToolError(f"Stored action draft {draft.id} has an invalid payload.") from exc
        if stored != parsed:
            raise ToolError(
                f"Action draft {draft.id} already exists for {expected_key!r} with a "
                "different payload; fetch it with get_action_draft instead."
            )

    response = ActionDraftCreationResponse(draft=draft_record(draft), created=created)
    await session.commit()
    return response


async def approve_action(
    *,
    draft_id: uuid.UUID,
    confirmation_phrase: str,
    session_maker: async_sessionmaker[AsyncSession] | None,
    settings: Settings,
) -> DraftDecisionResponse:
    """Approve one `pending_approval` draft after confirmation, then execute it once."""
    require_confirmation_phrase(draft_id=draft_id, confirmation_phrase=confirmation_phrase)
    if session_maker is None:
        raise ToolError(
            "Approval execution requires a database session factory; "
            "the MCP server is not wired to one."
        )

    status, run_status = await _read_draft_state(session_maker, draft_id)
    if status is ActionDraftStatus.REJECTED:
        raise ToolError(
            f"Action draft {draft_id} is 'rejected'; only pending_approval drafts "
            "can be approved. Nothing was executed."
        )
    if status is ActionDraftStatus.FAILED:
        raise ToolError(
            f"Action draft {draft_id} is 'failed' from an earlier execution attempt; "
            "it is never retried. Nothing was executed."
        )
    if status not in APPROVABLE_STATUSES:
        raise ToolError(
            f"Action draft {draft_id} is '{status.value}'; only pending_approval "
            "drafts can be approved. Nothing was executed."
        )

    async with AsyncExitStack() as stack:
        checkpointer = None
        dependencies: GraphDependencies | None = None
        if run_status is AnalysisRunStatus.AWAITING_APPROVAL:
            checkpointer, dependencies = await _resume_wiring(stack, settings)
        try:
            result = await decide_draft(
                draft_id=draft_id,
                decision="approved",
                session_factory=session_maker,
                checkpointer=checkpointer,
                settings=settings,
                dependencies=dependencies,
            )
        except DraftNotFoundError as exc:
            raise ToolError(f"Action draft {draft_id} not found.") from exc
        except DraftConflictError as exc:
            raise ToolError(
                f"Action draft {draft_id} was decided concurrently; nothing was executed."
            ) from exc
    return decision_response(result)


async def _read_draft_state(
    session_maker: async_sessionmaker[AsyncSession],
    draft_id: uuid.UUID,
) -> tuple[ActionDraftStatus, AnalysisRunStatus]:
    """Read the draft's status and its run's status without writing anything."""
    async with session_maker() as session:
        draft = await DraftRepo(session).get_by_id(draft_id)
        if draft is None:
            raise ToolError(f"Action draft {draft_id} not found.")
        run = await AnalysisRepo(session).get_run_by_id(draft.analysis_run_id)
        if run is None:  # pragma: no cover - FK guarantees the run exists
            raise ToolError(f"Analysis run for action draft {draft_id} not found.")
        return draft.status, run.status


async def _resume_wiring(
    stack: AsyncExitStack,
    settings: Settings,
) -> tuple[AsyncPostgresSaver, GraphDependencies]:
    """Open the checkpoint store and graph wiring needed to finish a paused run."""
    try:
        dependencies = create_graph_dependencies(settings)
    except GraphDependenciesError as exc:
        raise ToolError(
            "Analysis model is not configured, so the paused run cannot be resumed; "
            "nothing was executed."
        ) from exc
    try:
        checkpointer = await stack.enter_async_context(postgres_checkpointer(settings=settings))
    except Exception as exc:
        raise ToolError(
            "Could not open the LangGraph checkpoint store; nothing was executed."
        ) from exc
    return checkpointer, dependencies
