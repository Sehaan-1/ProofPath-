"""Approved-only executor orchestration (Sub-slice 2.1.3).

The executor is driven against an in-memory world with fake repositories, so the
tests pin *behavior* (nothing runs without approval, one side effect per draft, the
audit record carries the human approval timestamp, failures are recorded and never
retried). Real PostgreSQL transaction/savepoint semantics are covered by
`tests/integration/test_executors_postgres.py`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from proofpath import execution as execution_module
from proofpath.calendar_ics import calendar_uid
from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
    AuditActor,
)
from proofpath.execution import (
    APPROVED_EXECUTION_EVENT,
    FAILED_EXECUTION_EVENT,
    CalendarPublishError,
    DraftExecution,
    DraftExecutor,
    DraftNotExecutableError,
    DraftNotFoundError,
    ExecutionErrorCode,
    ExecutionOutcome,
    GoogleCalendarPublisher,
    calendar_invite_for_task,
    execution_record,
    google_calendar_enabled,
)

pytestmark = pytest.mark.unit

START = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
END = START + timedelta(minutes=90)


class World:
    """One approved draft, its run, and recording fakes for every collaborator."""

    def __init__(
        self,
        *,
        status: ActionDraftStatus = ActionDraftStatus.APPROVED,
        action_type: ActionType = ActionType.TRACKER_UPSERT,
        payload: dict[str, Any] | None = None,
        tracker_error: Exception | None = None,
        existing_task: SimpleNamespace | None = None,
    ) -> None:
        self.opportunity_id = uuid.uuid4()
        self.draft = SimpleNamespace(
            id=uuid.uuid4(),
            analysis_run_id=uuid.uuid4(),
            action_type=action_type,
            status=status,
            payload_json=payload
            or {
                "action_type": "tracker_upsert",
                "opportunity_id": str(self.opportunity_id),
                "stage": ApplicationStage.PREPARING.value,
                "notes": "Tailored resume drafted.",
            },
            idempotency_key="proofpath:run-1:tracker_upsert",
            approved_at=datetime(2026, 11, 28, 10, 30, tzinfo=UTC),
            executed_at=None,
        )
        self.run = SimpleNamespace(
            id=self.draft.analysis_run_id,
            opportunity_id=self.opportunity_id,
            status=AnalysisRunStatus.COMPLETED,
        )
        self.tracker_error = tracker_error
        self.existing_task = existing_task
        self.audit: list[dict[str, Any]] = []
        self.tracker_calls: list[dict[str, Any]] = []
        self.calendar_calls: list[dict[str, Any]] = []
        self.savepoints = 0
        self.savepoint_rollbacks = 0
        self.marked: list[str] = []
        self.task: SimpleNamespace | None = existing_task
        self.session_factory = self._build_session_factory()

    # --- fakes ---------------------------------------------------------
    def _build_session_factory(self) -> Any:
        world = self

        class _Savepoint:
            async def __aenter__(self) -> _Savepoint:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                if exc and exc[0] is not None:
                    world.savepoint_rollbacks += 1
                return False

        class FakeSession:
            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *_exc: object) -> bool:
                return False

            def begin_nested(self) -> _Savepoint:
                world.savepoints += 1
                return _Savepoint()

            async def flush(self) -> None:
                return None

        def session() -> Any:
            return FakeSession()

        def begin() -> Any:
            return FakeSession()

        session.begin = begin  # type: ignore[attr-defined]
        return session

    def draft_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def get_by_id_for_update(self, draft_id: uuid.UUID) -> Any:
                return world.draft if draft_id == world.draft.id else None

            async def mark_executed(
                self, draft_id: uuid.UUID, *, executed_at: datetime | None = None
            ) -> Any:
                world.marked.append("executed")
                world.draft.status = ActionDraftStatus.EXECUTED
                world.draft.executed_at = executed_at or datetime.now(UTC)
                return world.draft

            async def mark_failed(self, draft_id: uuid.UUID) -> Any:
                world.marked.append("failed")
                world.draft.status = ActionDraftStatus.FAILED
                return world.draft

        return Repo

    def analysis_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def get_run_by_id(self, run_id: uuid.UUID, **_kwargs: object) -> Any:
                return world.run if run_id == world.run.id else None

        return Repo

    def tracker_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def advance(
                self,
                *,
                opportunity_id: uuid.UUID,
                stage: ApplicationStage,
                notes: str | None = None,
            ) -> Any:
                world.tracker_calls.append(
                    {"opportunity_id": opportunity_id, "stage": stage, "notes": notes}
                )
                if world.tracker_error is not None:
                    raise world.tracker_error
                return SimpleNamespace(
                    id=uuid.uuid4(),
                    opportunity_id=opportunity_id,
                    stage=stage,
                    notes=notes,
                )

        return Repo

    def calendar_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def upsert_for_draft(self, **kwargs: Any) -> tuple[Any, bool]:
                world.calendar_calls.append(kwargs)
                if world.task is None:
                    world.task = SimpleNamespace(
                        id=uuid.uuid4(),
                        action_draft_id=kwargs["action_draft_id"],
                        opportunity_id=kwargs["opportunity_id"],
                        title=kwargs["title"],
                        description=kwargs.get("description"),
                        start_at=kwargs["start_at"],
                        end_at=kwargs["end_at"],
                        timezone=kwargs["timezone"],
                        ics_uid=kwargs["ics_uid"],
                        created_at=datetime(2026, 11, 28, 11, 0, tzinfo=UTC),
                    )
                return world.task, True

            async def get_by_action_draft_id(self, draft_id: uuid.UUID) -> Any:
                if world.task is not None and world.task.action_draft_id == draft_id:
                    return world.task
                return None

        return Repo

    def audit_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def record(self, **kwargs: Any) -> Any:
                entry = SimpleNamespace(id=uuid.uuid4(), **kwargs)
                world.audit.append(kwargs)
                return entry

            async def list_events(
                self,
                *,
                actor: AuditActor | None = None,
                event_type: str | None = None,
                resource_id: uuid.UUID | None = None,
                limit: int = 100,
                offset: int = 0,
            ) -> list[Any]:
                rows = [
                    SimpleNamespace(id=uuid.uuid4(), **item)
                    for item in world.audit
                    if (event_type is None or item["event_type"] == event_type)
                    and (resource_id is None or item["resource_id"] == resource_id)
                ]
                return rows[:limit]

        return Repo


def _install(monkeypatch: pytest.MonkeyPatch, world: World) -> None:
    monkeypatch.setattr(execution_module, "DraftRepo", world.draft_repo())
    monkeypatch.setattr(execution_module, "AnalysisRepo", world.analysis_repo())
    monkeypatch.setattr(execution_module, "TrackerRepo", world.tracker_repo())
    monkeypatch.setattr(execution_module, "CalendarTaskRepo", world.calendar_repo())
    monkeypatch.setattr(execution_module, "AuditRepo", world.audit_repo())


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _executor(world: World, **kwargs: Any) -> DraftExecutor:
    return DraftExecutor(session_factory=world.session_factory, settings=_settings(), **kwargs)


def _calendar_world(**kwargs: Any) -> World:
    return World(
        action_type=ActionType.CALENDAR_BLOCK,
        payload={
            "action_type": "calendar_block",
            "title": "Interview prep block",
            "start_at": START.isoformat(),
            "end_at": END.isoformat(),
            "description": "Work through the gap list.",
            "timezone": "Asia/Kolkata",
        },
        **kwargs,
    )


# ---------------------------------------------------------------- tracker (2.1.3a)


async def test_approved_tracker_draft_upserts_tracker_and_audits_the_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.EXECUTED
    assert result.did_execute is True
    assert result.succeeded is True
    assert result.executed_at is not None
    assert world.tracker_calls == [
        {
            "opportunity_id": world.opportunity_id,
            "stage": ApplicationStage.PREPARING,
            "notes": "Tailored resume drafted.",
        }
    ]
    assert world.marked == ["executed"]

    assert len(world.audit) == 1
    entry = world.audit[0]
    assert entry["event_type"] == APPROVED_EXECUTION_EVENT
    assert entry["actor"] is AuditActor.AGENT
    assert entry["resource_id"] == world.draft.id
    details = entry["details_json"]
    assert details["approved_at"] == world.draft.approved_at.isoformat()
    assert details["approved_by"] == "human"
    assert details["executed_at"] == result.executed_at.isoformat()
    assert details["idempotency_key"] == world.draft.idempotency_key
    assert details["target"]["type"] == "application_tracker"
    assert details["target"]["stage"] == ApplicationStage.PREPARING.value
    assert result.audit_log_id is not None
    assert world.savepoint_rollbacks == 0


async def test_concurrent_approval_cannot_execute_during_output_validation(monkeypatch):
    world = World()
    world.run.status = AnalysisRunStatus.RUNNING
    _install(monkeypatch, world)
    with pytest.raises(DraftNotExecutableError, match="validation"):
        await _executor(world).execute(world.draft.id)
    assert world.tracker_calls == []
    assert world.marked == []


@pytest.mark.parametrize(
    "status",
    [
        ActionDraftStatus.PENDING_APPROVAL,
        ActionDraftStatus.REJECTED,
        ActionDraftStatus.FAILED,
    ],
)
async def test_drafts_that_are_not_approved_never_execute(
    monkeypatch: pytest.MonkeyPatch,
    status: ActionDraftStatus,
) -> None:
    world = World(status=status)
    _install(monkeypatch, world)

    with pytest.raises(DraftNotExecutableError):
        await _executor(world).execute(world.draft.id)

    assert world.tracker_calls == []
    assert world.calendar_calls == []
    assert world.audit == []
    assert world.marked == []


async def test_approved_draft_without_approval_timestamp_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    world.draft.approved_at = None
    _install(monkeypatch, world)

    with pytest.raises(DraftNotExecutableError):
        await _executor(world).execute(world.draft.id)

    assert world.tracker_calls == []


async def test_missing_draft_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World()
    _install(monkeypatch, world)

    with pytest.raises(DraftNotFoundError):
        await _executor(world).execute(uuid.uuid4())

    assert world.audit == []


async def test_already_executed_draft_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(status=ActionDraftStatus.EXECUTED)
    world.draft.executed_at = datetime(2026, 11, 28, 10, 31, tzinfo=UTC)
    world.audit.append(
        {
            "actor": AuditActor.AGENT,
            "event_type": APPROVED_EXECUTION_EVENT,
            "resource_id": world.draft.id,
            "details_json": {},
        }
    )
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.ALREADY_EXECUTED
    assert result.did_execute is False
    assert result.succeeded is True
    assert result.executed_at == world.draft.executed_at
    assert result.audit_log_id is not None
    assert world.tracker_calls == []
    assert world.calendar_calls == []
    assert world.marked == []


async def test_payload_that_violates_its_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(payload={"action_type": "tracker_upsert", "stage": "saved"})
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.FAILED
    assert result.error_code is ExecutionErrorCode.INVALID_PAYLOAD
    assert result.executed_at is None
    assert world.tracker_calls == []
    assert world.marked == ["failed"]
    assert world.savepoints == 1
    assert world.savepoint_rollbacks == 1
    assert world.audit[0]["event_type"] == FAILED_EXECUTION_EVENT
    assert world.audit[0]["details_json"]["error"]["code"] == "invalid_payload"


async def test_payload_action_type_must_match_the_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(action_type=ActionType.CALENDAR_BLOCK)
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.FAILED
    assert result.error_code is ExecutionErrorCode.INVALID_PAYLOAD
    assert world.tracker_calls == []


async def test_contribution_sprint_drafts_have_no_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        action_type=ActionType.CONTRIBUTION_SPRINT,
        payload={
            "action_type": "contribution_sprint",
            "plan": {
                "title": "Ship a small PR",
                "scope": "Add a focused benchmark to a popular repo.",
                "repo_or_topic": "owner/repo",
                "deliverables": ["One merged PR"],
                "estimate_days": 3,
                "target_gap_requirement_ids": [str(uuid.uuid4())],
                "rationale": "Proves the missing systems-depth gap.",
            },
        },
    )
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.FAILED
    assert result.error_code is ExecutionErrorCode.UNSUPPORTED_ACTION_TYPE
    assert world.marked == ["failed"]
    assert world.audit[0]["details_json"]["error"]["code"] == "unsupported_action_type"


async def test_unexpected_executor_errors_are_audited_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(tracker_error=RuntimeError("tracker write exploded"))
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.FAILED
    assert result.error_code is ExecutionErrorCode.UNEXPECTED_ERROR
    assert "tracker write exploded" in (result.error_message or "")
    assert world.marked == ["failed"]
    assert world.savepoints == 1
    # The half-applied side effect was rolled back to the savepoint.
    assert world.savepoint_rollbacks == 1
    assert world.audit[0]["event_type"] == FAILED_EXECUTION_EVENT


# --------------------------------------------------------------- calendar (2.1.3b)


async def test_approved_calendar_draft_writes_a_local_task_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _calendar_world()
    _install(monkeypatch, world)

    result = await _executor(world).execute(world.draft.id)

    assert result.outcome is ExecutionOutcome.EXECUTED
    assert result.calendar_task_id is not None
    assert result.ics_uid == calendar_uid(draft_id=world.draft.id)
    assert world.calendar_calls[0]["timezone"] == "Asia/Kolkata"
    assert world.calendar_calls[0]["opportunity_id"] == world.opportunity_id

    details = world.audit[0]["details_json"]
    assert details["target"]["type"] == "calendar_task"
    assert details["target"]["ics_uid"] == result.ics_uid
    assert details["target"]["google_calendar"] == {
        "opted_in": False,
        "pushed": False,
        "event_id": None,
    }
    assert details["approved_at"] == world.draft.approved_at.isoformat()


async def test_calendar_invite_round_trips_from_the_persisted_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _calendar_world()
    _install(monkeypatch, world)

    await _executor(world).execute(world.draft.id)

    invite = calendar_invite_for_task(world.task)
    assert invite.startswith("BEGIN:VCALENDAR\r\n")
    assert f"UID:{calendar_uid(draft_id=world.draft.id)}" in invite
    assert "SUMMARY:Interview prep block" in invite
    assert "DTSTART:20261201T090000Z" in invite


async def test_live_google_calendar_is_off_unless_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _calendar_world()
    _install(monkeypatch, world)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        requests.append(request)
        return httpx.Response(200, json={"id": "gcal-1"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        executor = DraftExecutor(
            session_factory=world.session_factory,
            settings=_settings(),
            http_client=client,
        )
        await executor.execute(world.draft.id)
    finally:
        await client.aclose()

    assert requests == []
    assert google_calendar_enabled(_settings()) is False


async def test_opted_in_google_calendar_push_uses_the_generated_invite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _calendar_world()
    _install(monkeypatch, world)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "gcal-event-1"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = _settings(
        enable_google_calendar=True,
        google_calendar_id="primary",
        google_calendar_access_token="token-123",
    )
    try:
        executor = DraftExecutor(
            session_factory=world.session_factory,
            settings=settings,
            http_client=client,
        )
        result = await executor.execute(world.draft.id)
    finally:
        await client.aclose()

    assert result.outcome is ExecutionOutcome.EXECUTED
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url).endswith("/calendars/primary/events/import")
    assert request.headers["Authorization"] == "Bearer token-123"
    assert request.headers["Content-Type"].startswith("text/calendar")
    assert request.content.startswith(b"BEGIN:VCALENDAR")
    assert world.audit[0]["details_json"]["target"]["google_calendar"] == {
        "opted_in": True,
        "pushed": True,
        "event_id": "gcal-event-1",
    }


async def test_google_calendar_push_failures_are_recorded_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _calendar_world()
    _install(monkeypatch, world)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "insufficient scope"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = _settings(
        enable_google_calendar=True,
        google_calendar_id="primary",
        google_calendar_access_token="token-123",
    )
    try:
        executor = DraftExecutor(
            session_factory=world.session_factory,
            settings=settings,
            http_client=client,
        )
        result = await executor.execute(world.draft.id)
    finally:
        await client.aclose()

    assert result.outcome is ExecutionOutcome.FAILED
    assert result.error_code is ExecutionErrorCode.CALENDAR_PUBLISH_FAILED
    assert world.savepoint_rollbacks == 1
    assert world.marked == ["failed"]


async def test_google_calendar_publisher_requires_a_context_manager() -> None:
    publisher = GoogleCalendarPublisher(calendar_id="primary", access_token="t")

    with pytest.raises(CalendarPublishError, match="context manager"):
        await publisher.publish(ics="BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")

    assert "/calendars/{calendar_id}/events/import" in publisher.endpoint


def test_google_opt_in_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="ENABLE_GOOGLE_CALENDAR"):
        _settings(enable_google_calendar=True)
    with pytest.raises(ValidationError, match="GOOGLE_CALENDAR_ACCESS_TOKEN"):
        _settings(enable_google_calendar=True, google_calendar_id="primary")


def test_execution_record_maps_the_response_contract() -> None:
    record = execution_record(
        DraftExecution(
            draft_id=uuid.uuid4(),
            action_type=ActionType.TRACKER_UPSERT,
            outcome=ExecutionOutcome.FAILED,
            approved_at=datetime(2026, 11, 28, 10, 30, tzinfo=UTC),
            error_code=ExecutionErrorCode.INVALID_PAYLOAD,
            error_message="bad payload",
        )
    )

    assert record.outcome == "failed"
    assert record.error_code == "invalid_payload"
    assert record.executed_at is None
