"""Approved-only executors against live PostgreSQL (Sub-slice 2.1.3, end to end).

Every case uses the real repositories and real transactions, so what is proven here
is the thing unit tests cannot prove:

* the tracker upsert is one atomic `ON CONFLICT` statement that never regresses a
  stage and never drops notes;
* the audit row is appended in the same transaction as the side effect, carries the
  human approval timestamp, and survives a replay unchanged (one row, one
  `approved_at`);
* the calendar task is unique per draft and its `.ics` invite is downloadable over
  HTTP;
* a rejected execution rolls the whole savepoint back, so a failed run never leaves
  a half-applied side effect;
* two concurrent executions of the same approved draft produce one side effect.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from pydantic import PostgresDsn, SecretStr
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.calendar_ics import calendar_uid
from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
    AuditActor,
    AuditLog,
    CalendarTask,
    Opportunity,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.calendar_task import CalendarTaskRepo
from proofpath.db.repositories.draft import DraftRepo
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.execution import (
    APPROVED_EXECUTION_EVENT,
    FAILED_EXECUTION_EVENT,
    DraftExecution,
    DraftExecutor,
    DraftNotExecutableError,
    ExecutionErrorCode,
    ExecutionOutcome,
    calendar_invite_for_task,
)
from proofpath.main import create_app

pytestmark = pytest.mark.integration

START = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
END = START + timedelta(minutes=90)


@dataclass
class SeededDraft:
    """Identifiers of one opportunity + run + draft triple to clean up afterwards."""

    opportunity_id: uuid.UUID
    run_id: uuid.UUID
    draft_id: uuid.UUID


@dataclass
class ExecutorHarness:
    """Real engine-backed session factory plus the drafts created by a test."""

    settings: Settings
    factory: async_sessionmaker[AsyncSession]
    seeded: list[SeededDraft] = field(default_factory=list)


@pytest.fixture
async def harness(
    db_engine: AsyncEngine,
    integration_database_url: str,
) -> AsyncIterator[ExecutorHarness]:
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        llm_api_key=SecretStr("test-key"),
    )
    world = ExecutorHarness(
        settings=settings,
        factory=async_sessionmaker(db_engine, expire_on_commit=False),
    )
    try:
        yield world
    finally:
        for item in world.seeded:
            async with world.factory.begin() as session:
                await session.execute(delete(AuditLog).where(AuditLog.resource_id == item.draft_id))
                await session.execute(
                    delete(Opportunity).where(Opportunity.id == item.opportunity_id)
                )


async def _seed(
    harness: ExecutorHarness,
    *,
    action_type: ActionType,
    payload: dict[str, Any],
    status: ActionDraftStatus = ActionDraftStatus.PENDING_APPROVAL,
    tracker_stage: ApplicationStage | None = None,
    tracker_notes: str | None = None,
) -> SeededDraft:
    """Create an opportunity, a run, and one draft in the requested state."""
    async with harness.factory.begin() as session:
        opportunity = await OpportunityRepo(session).create_opportunity(
            company="Example",
            role_title="Platform Engineer",
            raw_text="SAP S/4HANA required",
            sanitized_text="SAP S/4HANA required",
        )
        run = await AnalysisRepo(session).create_run(
            opportunity_id=opportunity.id,
            thread_id=f"thread-{uuid.uuid4()}",
            status=AnalysisRunStatus.AWAITING_APPROVAL,
        )
        payload = dict(payload)
        if action_type is ActionType.TRACKER_UPSERT:
            payload["opportunity_id"] = str(opportunity.id)
        if tracker_stage is not None:
            tracker = await TrackerRepo(session).upsert(
                opportunity_id=opportunity.id,
                stage=tracker_stage,
                notes=tracker_notes,
            )
            assert tracker.stage is tracker_stage
        draft, _ = await DraftRepo(session).create_or_get_draft(
            analysis_run_id=run.id,
            action_type=action_type,
            payload_json=payload,
            idempotency_key=f"proofpath:{run.id}:{action_type}",
        )
        if status is ActionDraftStatus.APPROVED:
            assert await DraftRepo(session).claim_decision(draft.id, decision="approved")
        seeded = SeededDraft(
            opportunity_id=opportunity.id,
            run_id=run.id,
            draft_id=draft.id,
        )
    if status is ActionDraftStatus.REJECTED:
        async with harness.factory.begin() as session:
            assert await DraftRepo(session).claim_decision(seeded.draft_id, decision="rejected")
    harness.seeded.append(seeded)
    return seeded


def _tracker_payload(
    *,
    stage: ApplicationStage = ApplicationStage.PREPARING,
    notes: str | None = "Tailored resume drafted.",
) -> dict[str, Any]:
    return {
        "action_type": "tracker_upsert",
        "opportunity_id": str(uuid.uuid4()),
        "stage": stage.value,
        "notes": notes,
    }


def _calendar_payload() -> dict[str, Any]:
    return {
        "action_type": "calendar_block",
        "title": "Interview prep block",
        "start_at": START.isoformat(),
        "end_at": END.isoformat(),
        "description": "Work through the gap list.",
        "timezone": "Asia/Kolkata",
    }


def _executor(harness: ExecutorHarness) -> DraftExecutor:
    return DraftExecutor(session_factory=harness.factory, settings=harness.settings)


async def _audits(harness: ExecutorHarness, draft_id: uuid.UUID) -> list[AuditLog]:
    async with harness.factory() as session:
        return await AuditRepo(session).list_events(resource_id=draft_id)


# ------------------------------------------------------------- tracker (2.1.3a)


async def test_approved_tracker_draft_advances_and_audits_the_approval(
    harness: ExecutorHarness,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.TRACKER_UPSERT,
        payload=_tracker_payload(),
        status=ActionDraftStatus.APPROVED,
    )

    result = await _executor(harness).execute(seeded.draft_id)

    assert result.outcome is ExecutionOutcome.EXECUTED
    assert result.executed_at is not None
    async with harness.factory() as session:
        draft = await DraftRepo(session).get_by_id(seeded.draft_id)
        assert draft is not None
        assert draft.status is ActionDraftStatus.EXECUTED
        assert draft.approved_at is not None
        assert draft.executed_at is not None

        tracker = await TrackerRepo(session).get_by_opportunity_id(seeded.opportunity_id)
        assert tracker is not None
        assert tracker.stage is ApplicationStage.PREPARING
        assert tracker.notes == "Tailored resume drafted."

    audits = await _audits(harness, seeded.draft_id)
    assert len(audits) == 1
    entry = audits[0]
    assert entry.id == result.audit_log_id
    assert entry.actor is AuditActor.AGENT
    assert entry.event_type == APPROVED_EXECUTION_EVENT
    assert entry.details_json["approved_at"] == draft.approved_at.isoformat()
    assert entry.details_json["approved_by"] == "human"
    assert entry.details_json["idempotency_key"] == (f"proofpath:{seeded.run_id}:tracker_upsert")
    assert entry.details_json["target"]["tracker_id"] == str(tracker.id)

    # Idempotent replay: same audit row, no second tracker write.
    replay = await _executor(harness).execute(seeded.draft_id)
    assert replay.outcome is ExecutionOutcome.ALREADY_EXECUTED
    assert replay.audit_log_id == result.audit_log_id
    assert replay.executed_at == result.executed_at
    assert len(await _audits(harness, seeded.draft_id)) == 1
    async with harness.factory() as session:
        # A tracker draft never materializes a calendar task.
        assert await CalendarTaskRepo(session).get_by_action_draft_id(seeded.draft_id) is None


async def test_tracker_execution_never_regresses_stage_or_drops_notes(
    harness: ExecutorHarness,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.TRACKER_UPSERT,
        payload=_tracker_payload(stage=ApplicationStage.PREPARING, notes=None),
        status=ActionDraftStatus.APPROVED,
        tracker_stage=ApplicationStage.APPLIED,
        tracker_notes="Recruiter said interviews start in March.",
    )

    result = await _executor(harness).execute(seeded.draft_id)

    assert result.outcome is ExecutionOutcome.EXECUTED
    async with harness.factory() as session:
        tracker = await TrackerRepo(session).get_by_opportunity_id(seeded.opportunity_id)
        assert tracker is not None
        # The human already moved it further along: an approved draft must not rewind it.
        assert tracker.stage is ApplicationStage.APPLIED
        assert tracker.notes == "Recruiter said interviews start in March."


async def test_tracker_execution_can_advance_and_replace_notes(
    harness: ExecutorHarness,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.TRACKER_UPSERT,
        payload=_tracker_payload(stage=ApplicationStage.READY_TO_APPLY, notes="Now ready."),
        status=ActionDraftStatus.APPROVED,
        tracker_stage=ApplicationStage.SAVED,
        tracker_notes="Old note",
    )

    await _executor(harness).execute(seeded.draft_id)

    async with harness.factory() as session:
        tracker = await TrackerRepo(session).get_by_opportunity_id(seeded.opportunity_id)
        assert tracker is not None
        assert tracker.stage is ApplicationStage.READY_TO_APPLY
        assert tracker.notes == "Now ready."


@pytest.mark.parametrize(
    "status",
    [ActionDraftStatus.PENDING_APPROVAL, ActionDraftStatus.REJECTED],
)
async def test_unapproved_drafts_never_execute(
    harness: ExecutorHarness,
    status: ActionDraftStatus,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.TRACKER_UPSERT,
        payload=_tracker_payload(),
        status=status,
    )

    with pytest.raises(DraftNotExecutableError):
        await _executor(harness).execute(seeded.draft_id)

    async with harness.factory() as session:
        assert await TrackerRepo(session).get_by_opportunity_id(seeded.opportunity_id) is None
        draft = await DraftRepo(session).get_by_id(seeded.draft_id)
        assert draft is not None
        assert draft.status is status
    assert await _audits(harness, seeded.draft_id) == []


async def test_failed_execution_is_rolled_back_and_recorded(
    harness: ExecutorHarness,
) -> None:
    """A `contribution_sprint` draft has no executor: it must fail closed and cleanly."""
    seeded = await _seed(
        harness,
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
        status=ActionDraftStatus.APPROVED,
    )

    result = await _executor(harness).execute(seeded.draft_id)

    assert result.outcome is ExecutionOutcome.FAILED
    assert result.error_code is ExecutionErrorCode.UNSUPPORTED_ACTION_TYPE
    assert result.audit_log_id is not None
    async with harness.factory() as session:
        draft = await DraftRepo(session).get_by_id(seeded.draft_id)
        assert draft is not None
        assert draft.status is ActionDraftStatus.FAILED
        assert draft.executed_at is None
        assert await TrackerRepo(session).get_by_opportunity_id(seeded.opportunity_id) is None

    audits = await _audits(harness, seeded.draft_id)
    assert len(audits) == 1
    assert audits[0].event_type == FAILED_EXECUTION_EVENT
    assert audits[0].details_json["error"]["code"] == "unsupported_action_type"
    # A failed draft is never silently retried by a replayed approval.
    with pytest.raises(DraftNotExecutableError):
        await _executor(harness).execute(seeded.draft_id)


# ------------------------------------------------------------ calendar (2.1.3b)


async def test_approved_calendar_draft_materializes_one_task_and_a_valid_invite(
    harness: ExecutorHarness,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.CALENDAR_BLOCK,
        payload=_calendar_payload(),
        status=ActionDraftStatus.APPROVED,
    )

    result = await _executor(harness).execute(seeded.draft_id)

    assert result.outcome is ExecutionOutcome.EXECUTED
    assert result.calendar_task_id is not None
    assert result.ics_uid == calendar_uid(draft_id=seeded.draft_id)
    async with harness.factory() as session:
        task = await CalendarTaskRepo(session).get_by_action_draft_id(seeded.draft_id)
        assert task is not None
        assert task.id == result.calendar_task_id
        assert task.opportunity_id == seeded.opportunity_id
        assert task.timezone == "Asia/Kolkata"
        assert task.ics_uid == calendar_uid(draft_id=seeded.draft_id)
        # Times land in PostgreSQL as timezone-aware UTC instants.
        assert task.start_at.astimezone(UTC) == START
        assert task.end_at.astimezone(UTC) == END

    invite = calendar_invite_for_task(task)
    assert invite.startswith("BEGIN:VCALENDAR\r\n")
    assert f"UID:{task.ics_uid}" in invite
    assert "SUMMARY:Interview prep block" in invite
    assert "DTSTART:20261201T090000Z" in invite
    assert all(len(line.encode()) <= 75 for line in invite.split("\r\n"))

    audits = await _audits(harness, seeded.draft_id)
    assert len(audits) == 1
    assert audits[0].details_json["target"]["google_calendar"] == {
        "opted_in": False,
        "pushed": False,
        "event_id": None,
    }

    # Replay: no second task, same audit row.
    replay = await _executor(harness).execute(seeded.draft_id)
    assert replay.outcome is ExecutionOutcome.ALREADY_EXECUTED
    assert replay.calendar_task_id == task.id
    async with harness.factory() as session:
        tasks = (
            (
                await session.execute(
                    select(CalendarTask).where(CalendarTask.opportunity_id == seeded.opportunity_id)
                )
            )
            .scalars()
            .all()
        )
    assert len(tasks) == 1
    assert len(await _audits(harness, seeded.draft_id)) == 1


async def test_calendar_task_is_unique_per_draft(harness: ExecutorHarness) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.CALENDAR_BLOCK,
        payload=_calendar_payload(),
        status=ActionDraftStatus.APPROVED,
    )

    async with harness.factory.begin() as session:
        repo = CalendarTaskRepo(session)
        first, created = await repo.upsert_for_draft(
            action_draft_id=seeded.draft_id,
            opportunity_id=seeded.opportunity_id,
            title="Interview prep block",
            start_at=START,
            end_at=END,
            ics_uid=calendar_uid(draft_id=seeded.draft_id),
            description=None,
            timezone="Asia/Kolkata",
        )
        await session.flush()
        second, again = await repo.upsert_for_draft(
            action_draft_id=seeded.draft_id,
            opportunity_id=seeded.opportunity_id,
            title="Something else entirely",
            start_at=START,
            end_at=END,
            ics_uid=calendar_uid(draft_id=seeded.draft_id),
            description=None,
            timezone="Asia/Kolkata",
        )

    assert created is True
    assert again is False
    assert second.id == first.id
    assert second.title == "Interview prep block"


async def test_calendar_invite_is_downloadable_over_http(
    harness: ExecutorHarness,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.CALENDAR_BLOCK,
        payload=_calendar_payload(),
        status=ActionDraftStatus.APPROVED,
    )
    result = await _executor(harness).execute(seeded.draft_id)
    assert result.calendar_task_id is not None

    app = create_app(settings=harness.settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        listing = await client.get("/api/v1/calendar-tasks")
        assert listing.status_code == 200
        ids = [item["id"] for item in listing.json()]
        assert str(result.calendar_task_id) in ids
        record = next(item for item in listing.json() if item["id"] == str(result.calendar_task_id))
        assert record["invite_url"] == (
            f"/api/v1/calendar-tasks/{result.calendar_task_id}/invite.ics"
        )

        download = await client.get(record["invite_url"])
        assert download.status_code == 200, download.text
        assert download.headers["content-type"].startswith("text/calendar")
        assert "attachment;" in download.headers["content-disposition"]
        body = download.text
        assert body.startswith("BEGIN:VCALENDAR\r\n")
        assert f"UID:{result.ics_uid}" in body
        assert "SUMMARY:Interview prep block" in body

        missing = await client.get(f"/api/v1/calendar-tasks/{uuid.uuid4()}/invite.ics")
        assert missing.status_code == 404


# --------------------------------------------------------------- concurrency


async def test_concurrent_execution_writes_exactly_one_side_effect(
    harness: ExecutorHarness,
) -> None:
    seeded = await _seed(
        harness,
        action_type=ActionType.TRACKER_UPSERT,
        payload=_tracker_payload(),
        status=ActionDraftStatus.APPROVED,
    )

    first, second = await asyncio.gather(
        _executor(harness).execute(seeded.draft_id),
        _executor(harness).execute(seeded.draft_id),
    )
    results: list[DraftExecution] = [first, second]

    outcomes = sorted(result.outcome.value for result in results)
    assert outcomes == ["already_executed", "executed"]
    executed = next(r for r in results if r.outcome is ExecutionOutcome.EXECUTED)
    replayed = next(r for r in results if r.outcome is ExecutionOutcome.ALREADY_EXECUTED)
    assert replayed.audit_log_id == executed.audit_log_id

    audits = await _audits(harness, seeded.draft_id)
    assert len(audits) == 1
    async with harness.factory() as session:
        tracker = await TrackerRepo(session).get_by_opportunity_id(seeded.opportunity_id)
    assert tracker is not None
    assert tracker.stage is ApplicationStage.PREPARING
