"""Approved-only side-effect executors (Sub-slice 2.1.3).

Nothing in this module ever runs from inside the analysis graph: an executor fires only
for a draft a human explicitly approved (`action_drafts.status = 'approved'` with a
non-null `approved_at`), and every execution is transactional:

* 2.1.3a `tracker_upsert` -> atomically advances `application_tracker` and appends an
  immutable `audit_logs` record carrying the human approval timestamp.
* 2.1.3b `calendar_block` -> writes a local `calendar_tasks` row served as a
  downloadable `.ics` invite, plus the same audit record. A live Google Calendar write
  happens only when `ENABLE_GOOGLE_CALENDAR=true` (explicit opt-in, credentials
  required) — otherwise ProofPath never touches the network.
* `contribution_sprint` drafts are advisory only; executing one fails closed with
  `unsupported_action_type` instead of inventing a side effect.

Dedupe (the reason 2.1.2 excluded these writes): one execution per draft. The draft row
is locked `FOR UPDATE` for the whole transaction, `calendar_tasks.action_draft_id` is
unique, the tracker upsert is keyed on `opportunity_id`, and the draft's unique
`idempotency_key` (`proofpath:<run_id>:<action_type>`) is copied into the audit record,
so a retry/replay is provably the same side effect rather than a second one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.calendar_ics import build_calendar_invite, calendar_uid
from proofpath.config import Settings, get_settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    AuditActor,
    AuditLog,
    CalendarTask,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.calendar_task import CalendarTaskRepo
from proofpath.db.repositories.draft import DraftRepo
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.schemas.actions import (
    CalendarBlockPayload,
    DraftExecutionResponse,
    TrackerUpsertPayload,
    parse_action_draft_payload,
)

__all__ = [
    "APPROVED_EXECUTION_EVENT",
    "FAILED_EXECUTION_EVENT",
    "DraftExecution",
    "DraftExecutionError",
    "DraftExecutor",
    "DraftNotExecutableError",
    "DraftNotFoundError",
    "ExecutionErrorCode",
    "ExecutionOutcome",
    "GoogleCalendarPublisher",
    "calendar_invite_for_task",
    "execute_approved_draft",
    "execution_record",
    "google_calendar_enabled",
]

APPROVED_EXECUTION_EVENT = "action_draft.executed"
FAILED_EXECUTION_EVENT = "action_draft.execution_failed"

_DEFAULT_GOOGLE_TIMEOUT = 10.0


class ExecutionOutcome(StrEnum):
    """Result of one executor run."""

    EXECUTED = "executed"
    ALREADY_EXECUTED = "already_executed"
    FAILED = "failed"


class ExecutionErrorCode(StrEnum):
    """Stable, loggable failure reasons recorded on the audit trail."""

    INVALID_PAYLOAD = "invalid_payload"
    MISSING_OPPORTUNITY = "missing_opportunity"
    UNSUPPORTED_ACTION_TYPE = "unsupported_action_type"
    CALENDAR_PUBLISH_FAILED = "calendar_publish_failed"
    UNEXPECTED_ERROR = "unexpected_error"


class DraftNotFoundError(LookupError):
    """Raised when the draft to execute does not exist."""


class DraftNotExecutableError(RuntimeError):
    """Raised when a draft is not human-approved (or already failed) and must not run."""


class CalendarPublishError(RuntimeError):
    """Raised when the opt-in Google Calendar push fails."""


@dataclass(frozen=True, slots=True)
class DraftExecution:
    """Outcome of executing (or observing) one approved action draft."""

    draft_id: uuid.UUID
    action_type: ActionType
    outcome: ExecutionOutcome
    approved_at: datetime | None = None
    executed_at: datetime | None = None
    audit_log_id: uuid.UUID | None = None
    calendar_task_id: uuid.UUID | None = None
    ics_uid: str | None = None
    error_code: ExecutionErrorCode | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the side effect is in place (freshly executed or already there)."""
        return self.outcome is not ExecutionOutcome.FAILED

    @property
    def did_execute(self) -> bool:
        """True only when *this* call performed the side effect."""
        return self.outcome is ExecutionOutcome.EXECUTED


class DraftExecutionError(Exception):
    """Handled executor failure carrying the audit error code it records."""

    def __init__(self, code: ExecutionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class _DraftSnapshot:
    """Detached copy of the draft fields the executor needs, taken under the row lock."""

    draft_id: uuid.UUID
    analysis_run_id: uuid.UUID
    action_type: ActionType
    status: ActionDraftStatus
    approved_at: datetime | None
    executed_at: datetime | None
    idempotency_key: str
    payload_json: dict[str, Any]


def google_calendar_enabled(settings: Settings | None = None) -> bool:
    """Return True only when the operator explicitly opted into live Calendar writes."""
    return (settings or get_settings()).enable_google_calendar


def calendar_invite_for_task(task: CalendarTask) -> str:
    """Render the deterministic `.ics` invite for a persisted calendar task."""
    return build_calendar_invite(
        draft_id=task.action_draft_id,
        title=task.title,
        start_at=task.start_at,
        end_at=task.end_at,
        description=task.description,
        timezone_name=task.timezone,
        stamp=task.created_at,
    )


def execution_record(execution: DraftExecution) -> DraftExecutionResponse:
    """Map an executor result to its API contract."""
    return DraftExecutionResponse(
        draft_id=execution.draft_id,
        action_type=execution.action_type,
        outcome=execution.outcome.value,
        approved_at=execution.approved_at,
        executed_at=execution.executed_at,
        audit_log_id=execution.audit_log_id,
        calendar_task_id=execution.calendar_task_id,
        error_code=execution.error_code.value if execution.error_code is not None else None,
        error_message=execution.error_message,
    )


class GoogleCalendarPublisher:
    """Push one approved block into Google Calendar (Calendar API v3 `events.import`).

    Only ever constructed when `ENABLE_GOOGLE_CALENDAR=true`; the local task and `.ics`
    invite are the source of truth and are always written first.
    """

    endpoint = "https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/import"

    def __init__(
        self,
        *,
        calendar_id: str,
        access_token: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = _DEFAULT_GOOGLE_TIMEOUT,
    ) -> None:
        self._calendar_id = calendar_id
        self._access_token = access_token
        self._external_client = client
        self._client = client
        self._timeout = timeout

    async def __aenter__(self) -> GoogleCalendarPublisher:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._external_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def publish(self, *, ics: str) -> str:
        """Import `ics` into the opted-in calendar and return the provider event id."""
        if self._client is None:
            raise CalendarPublishError("GoogleCalendarPublisher must be used as a context manager.")
        url = self.endpoint.format(calendar_id=self._calendar_id)
        try:
            response = await self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "text/calendar; charset=utf-8",
                    "Accept": "application/json",
                },
                content=ics.encode("utf-8"),
            )
        except httpx.HTTPError as exc:
            raise CalendarPublishError(f"Google Calendar request failed: {exc}") from exc
        if not response.is_success:
            raise CalendarPublishError(
                f"Google Calendar rejected the event (HTTP {response.status_code})."
            )
        event_id = response.json().get("id")
        if not isinstance(event_id, str) or not event_id:
            raise CalendarPublishError("Google Calendar returned no event id.")
        return event_id


class DraftExecutor:
    """Transactional executor for human-approved action drafts."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._http_client = http_client

    @property
    def settings(self) -> Settings:
        """Resolved settings (lazily loaded from the environment when not injected)."""
        return self._settings or get_settings()

    async def execute(self, draft_id: uuid.UUID) -> DraftExecution:
        """Execute the approved draft `draft_id`, exactly once, in one transaction.

        The whole execution commits atomically: either the side effect *and* its audit
        record and the `executed` status land together, or the savepoint is rolled back
        and only the recorded failure remains.
        """
        async with self._session_factory.begin() as session:
            draft = await DraftRepo(session).get_by_id_for_update(draft_id)
            if draft is None:
                raise DraftNotFoundError(f"Action draft {draft_id} not found.")
            snapshot = _DraftSnapshot(
                draft_id=draft.id,
                analysis_run_id=draft.analysis_run_id,
                action_type=draft.action_type,
                status=draft.status,
                approved_at=draft.approved_at,
                executed_at=draft.executed_at,
                idempotency_key=draft.idempotency_key,
                payload_json=dict(draft.payload_json),
            )
            if snapshot.status is ActionDraftStatus.EXECUTED:
                return await self._describe_executed(session, snapshot)
            if snapshot.status is not ActionDraftStatus.APPROVED or snapshot.approved_at is None:
                raise DraftNotExecutableError(
                    "Only human-approved action drafts execute side effects; "
                    f"draft is '{snapshot.status.value}'."
                )
            return await self._execute_locked(session, snapshot)

    # ------------------------------------------------------------------ internals

    async def _execute_locked(
        self,
        session: AsyncSession,
        snapshot: _DraftSnapshot,
    ) -> DraftExecution:
        executed_at = datetime.now(UTC)
        try:
            run = await AnalysisRepo(session).get_run_by_id(snapshot.analysis_run_id)
            if run is None:
                raise DraftExecutionError(
                    ExecutionErrorCode.MISSING_OPPORTUNITY,
                    "Analysis run for the draft no longer exists.",
                )
            # A concurrent approval may see a committed decision while another
            # caller still owns the resume. Do not race final output validation.
            if run.status is AnalysisRunStatus.RUNNING:
                raise DraftNotExecutableError("The analysis run has not finished validation.")
            opportunity_id = run.opportunity_id
            async with session.begin_nested():
                target = await self._apply_side_effect(
                    session,
                    snapshot=snapshot,
                    opportunity_id=opportunity_id,
                )
                await DraftRepo(session).mark_executed(snapshot.draft_id, executed_at=executed_at)
                entry = await AuditRepo(session).record(
                    actor=AuditActor.AGENT,
                    event_type=APPROVED_EXECUTION_EVENT,
                    resource_id=snapshot.draft_id,
                    details_json=_success_details(
                        snapshot,
                        executed_at=executed_at,
                        target=target,
                    ),
                )
        except DraftNotExecutableError:
            raise  # In-flight resume: leave the draft untouched for its owner.
        except DraftExecutionError as exc:
            entry = await self._record_failure(session, snapshot, exc)
            return DraftExecution(
                draft_id=snapshot.draft_id,
                action_type=snapshot.action_type,
                outcome=ExecutionOutcome.FAILED,
                approved_at=snapshot.approved_at,
                audit_log_id=entry.id,
                error_code=exc.code,
                error_message=exc.message,
            )
        except (
            Exception
        ) as exc:  # fail closed: any unexpected error is audited, never retried blindly
            failure = DraftExecutionError(
                ExecutionErrorCode.UNEXPECTED_ERROR,
                f"{type(exc).__name__}: {exc}",
            )
            entry = await self._record_failure(session, snapshot, failure)
            return DraftExecution(
                draft_id=snapshot.draft_id,
                action_type=snapshot.action_type,
                outcome=ExecutionOutcome.FAILED,
                approved_at=snapshot.approved_at,
                audit_log_id=entry.id,
                error_code=failure.code,
                error_message=failure.message,
            )
        return DraftExecution(
            draft_id=snapshot.draft_id,
            action_type=snapshot.action_type,
            outcome=ExecutionOutcome.EXECUTED,
            approved_at=snapshot.approved_at,
            executed_at=executed_at,
            audit_log_id=entry.id,
            calendar_task_id=_target_uuid(target, "calendar_task_id"),
            ics_uid=_target_str(target, "ics_uid"),
        )

    async def _apply_side_effect(
        self,
        session: AsyncSession,
        *,
        snapshot: _DraftSnapshot,
        opportunity_id: uuid.UUID,
    ) -> dict[str, Any]:
        try:
            payload = parse_action_draft_payload(snapshot.payload_json)
        except ValidationError as exc:
            raise DraftExecutionError(
                ExecutionErrorCode.INVALID_PAYLOAD,
                f"Draft payload does not satisfy its contract: {exc.error_count()} error(s).",
            ) from exc
        if payload.action_type is not snapshot.action_type:
            raise DraftExecutionError(
                ExecutionErrorCode.INVALID_PAYLOAD,
                "Draft payload action_type does not match the stored action_type.",
            )

        if isinstance(payload, TrackerUpsertPayload):
            return await self._apply_tracker(
                session,
                opportunity_id=opportunity_id,
                payload=payload,
            )
        if isinstance(payload, CalendarBlockPayload):
            return await self._apply_calendar(
                session,
                snapshot=snapshot,
                opportunity_id=opportunity_id,
                payload=payload,
            )
        raise DraftExecutionError(
            ExecutionErrorCode.UNSUPPORTED_ACTION_TYPE,
            f"No executor is defined for '{payload.action_type.value}' drafts.",
        )

    async def _apply_tracker(
        self,
        session: AsyncSession,
        *,
        opportunity_id: uuid.UUID,
        payload: TrackerUpsertPayload,
    ) -> dict[str, Any]:
        tracker = await TrackerRepo(session).advance(
            opportunity_id=opportunity_id,
            stage=payload.stage,
            notes=payload.notes,
        )
        return {
            "type": "application_tracker",
            "tracker_id": str(tracker.id),
            "opportunity_id": str(opportunity_id),
            "stage": tracker.stage.value,
            "notes_written": payload.notes is not None,
        }

    async def _apply_calendar(
        self,
        session: AsyncSession,
        *,
        snapshot: _DraftSnapshot,
        opportunity_id: uuid.UUID,
        payload: CalendarBlockPayload,
    ) -> dict[str, Any]:
        task, created = await CalendarTaskRepo(session).upsert_for_draft(
            action_draft_id=snapshot.draft_id,
            opportunity_id=opportunity_id,
            title=payload.title,
            description=payload.description,
            start_at=payload.start_at,
            end_at=payload.end_at,
            timezone=payload.timezone,
            ics_uid=calendar_uid(draft_id=snapshot.draft_id),
        )
        google_event_id = await self._maybe_publish_to_google(task)
        return {
            "type": "calendar_task",
            "calendar_task_id": str(task.id),
            "ics_uid": task.ics_uid,
            "start_at": task.start_at.isoformat(),
            "end_at": task.end_at.isoformat(),
            "timezone": task.timezone,
            "created": created,
            "google_calendar": {
                "opted_in": self.settings.enable_google_calendar,
                "pushed": google_event_id is not None,
                "event_id": google_event_id,
            },
        }

    async def _maybe_publish_to_google(self, task: CalendarTask) -> str | None:
        """Push to Google Calendar only when the operator opted in with credentials."""
        settings = self.settings
        if not settings.enable_google_calendar:
            return None
        if settings.google_calendar_id is None or settings.google_calendar_access_token is None:
            raise DraftExecutionError(
                ExecutionErrorCode.CALENDAR_PUBLISH_FAILED,
                "ENABLE_GOOGLE_CALENDAR is set but the calendar id/token are missing.",
            )
        invite = calendar_invite_for_task(task)
        publisher = GoogleCalendarPublisher(
            calendar_id=settings.google_calendar_id.get_secret_value(),
            access_token=settings.google_calendar_access_token.get_secret_value(),
            client=self._http_client,
        )
        try:
            async with publisher:
                return await publisher.publish(ics=invite)
        except CalendarPublishError as exc:
            raise DraftExecutionError(
                ExecutionErrorCode.CALENDAR_PUBLISH_FAILED,
                str(exc),
            ) from exc

    async def _record_failure(
        self,
        session: AsyncSession,
        snapshot: _DraftSnapshot,
        failure: DraftExecutionError,
    ) -> AuditLog:
        await DraftRepo(session).mark_failed(snapshot.draft_id)
        return await AuditRepo(session).record(
            actor=AuditActor.AGENT,
            event_type=FAILED_EXECUTION_EVENT,
            resource_id=snapshot.draft_id,
            details_json={
                "draft_id": str(snapshot.draft_id),
                "action_type": snapshot.action_type.value,
                "idempotency_key": snapshot.idempotency_key,
                "approved_at": _iso(snapshot.approved_at),
                "approved_by": "human",
                "error": {"code": failure.code.value, "message": failure.message},
            },
        )

    async def _describe_executed(
        self,
        session: AsyncSession,
        snapshot: _DraftSnapshot,
    ) -> DraftExecution:
        """Describe an already-executed draft (idempotent replays perform no side effect)."""
        audit = AuditRepo(session)
        events = await audit.list_events(
            event_type=APPROVED_EXECUTION_EVENT,
            resource_id=snapshot.draft_id,
            limit=1,
        )
        task = await CalendarTaskRepo(session).get_by_action_draft_id(snapshot.draft_id)
        return DraftExecution(
            draft_id=snapshot.draft_id,
            action_type=snapshot.action_type,
            outcome=ExecutionOutcome.ALREADY_EXECUTED,
            approved_at=snapshot.approved_at,
            executed_at=snapshot.executed_at,
            audit_log_id=events[0].id if events else None,
            calendar_task_id=task.id if task is not None else None,
            ics_uid=task.ics_uid if task is not None else None,
        )


def _success_details(
    snapshot: _DraftSnapshot,
    *,
    executed_at: datetime,
    target: dict[str, Any],
) -> dict[str, Any]:
    return {
        "draft_id": str(snapshot.draft_id),
        "action_type": snapshot.action_type.value,
        "idempotency_key": snapshot.idempotency_key,
        "approved_at": _iso(snapshot.approved_at),
        "approved_by": "human",
        "executed_at": executed_at.isoformat(),
        "target": target,
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _target_uuid(target: dict[str, Any], key: str) -> uuid.UUID | None:
    raw = target.get(key)
    return uuid.UUID(raw) if isinstance(raw, str) else None


def _target_str(target: dict[str, Any], key: str) -> str | None:
    raw = target.get(key)
    return raw if isinstance(raw, str) else None


async def execute_approved_draft(
    *,
    draft_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> DraftExecution:
    """Convenience wrapper around `DraftExecutor` for callers without a long-lived executor."""
    executor = DraftExecutor(
        session_factory=session_factory,
        settings=settings,
        http_client=http_client,
    )
    return await executor.execute(draft_id)
