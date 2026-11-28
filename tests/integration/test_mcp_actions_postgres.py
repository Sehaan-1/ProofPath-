"""Approval-gated MCP tools against live PostgreSQL (Sub-slice 2.1.4, end to end).

A real MCP `Client` drives `create_action_draft` and `approve_action` against the
real repositories, so what is proven here is behaviour no fake can prove:

* `create_action_draft` writes exactly one `pending_approval` row and nothing else —
  no tracker entry, no calendar task, no audit record;
* `approve_action` refuses a wrong confirmation phrase and leaves the draft pending;
* a confirmed approval executes through the shared `proofpath.approvals` service:
  the tracker advances, one immutable audit row carries the human `approved_at`,
  and the draft ends `executed`;
* repeating the approval executes nothing a second time (one audit row, one tracker
  row, `already_executed`);
* rejected drafts are refused, and a duplicate idempotency key with a different
  payload is refused instead of silently overwriting the queued draft.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from mcp import Client
from pydantic import PostgresDsn, SecretStr
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraft,
    ActionDraftStatus,
    AnalysisRunStatus,
    ApplicationStage,
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
from proofpath.execution import APPROVED_EXECUTION_EVENT
from proofpath.mcp_server import create_mcp_server
from proofpath.policy.tool_allowlist import APPROVAL_CONFIRMATION_PHRASE

pytestmark = pytest.mark.integration

START = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
END = START + timedelta(minutes=90)


@dataclass
class MCPActionHarness:
    """Engine-backed MCP server plus the rows a test must clean up afterwards."""

    settings: Settings
    session_maker: async_sessionmaker[AsyncSession]
    server: Any
    opportunity_ids: list[uuid.UUID] = field(default_factory=list)
    draft_ids: list[uuid.UUID] = field(default_factory=list)


@pytest.fixture
async def harness(
    db_engine: AsyncEngine,
    integration_database_url: str,
) -> AsyncIterator[MCPActionHarness]:
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        embedding_model="hash",
        llm_api_key=SecretStr("test-key"),
    )
    session_maker = async_sessionmaker(db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def open_session() -> AsyncIterator[AsyncSession]:
        async with session_maker() as session:
            yield session

    server = create_mcp_server(
        settings=settings,
        session_factory=open_session,
        session_maker=session_maker,
    )
    world = MCPActionHarness(settings=settings, session_maker=session_maker, server=server)
    try:
        yield world
    finally:
        async with session_maker.begin() as session:
            if world.draft_ids:
                await session.execute(
                    delete(AuditLog).where(AuditLog.resource_id.in_(world.draft_ids))
                )
            for opportunity_id in world.opportunity_ids:
                await session.execute(delete(Opportunity).where(Opportunity.id == opportunity_id))


async def _seed_run(
    harness: MCPActionHarness,
    *,
    status: AnalysisRunStatus = AnalysisRunStatus.COMPLETED,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create one opportunity and analysis run; return their ids."""
    async with harness.session_maker.begin() as session:
        opportunity = await OpportunityRepo(session).create_opportunity(
            company="Example",
            role_title="Platform Engineer",
            raw_text="SAP S/4HANA required",
            sanitized_text="SAP S/4HANA required",
        )
        run = await AnalysisRepo(session).create_run(
            opportunity_id=opportunity.id,
            thread_id=f"thread-{uuid.uuid4()}",
            status=status,
        )
    harness.opportunity_ids.append(opportunity.id)
    return opportunity.id, run.id


def _tracker_payload(
    opportunity_id: uuid.UUID, *, notes: str | None = "Tailored resume."
) -> dict[str, Any]:
    return {
        "action_type": "tracker_upsert",
        "opportunity_id": str(opportunity_id),
        "stage": ApplicationStage.PREPARING.value,
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


async def _drafts_for_run(harness: MCPActionHarness, run_id: uuid.UUID) -> list[ActionDraft]:
    async with harness.session_maker() as session:
        return await DraftRepo(session).list_drafts(analysis_run_id=run_id)


async def test_create_and_approve_a_draft_executes_exactly_once(
    harness: MCPActionHarness,
) -> None:
    opportunity_id, run_id = await _seed_run(harness)
    key = f"proofpath:{run_id}:tracker_upsert"

    async with Client(harness.server) as client:
        created = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _tracker_payload(opportunity_id),
                "idempotency_key": key,
            },
        )
        assert created.is_error is False, created.content
        assert created.structured_content is not None
        assert created.structured_content["created"] is True
        assert created.structured_content["requires_approval"] is True
        draft_id = uuid.UUID(created.structured_content["draft"]["id"])
        harness.draft_ids.append(draft_id)
        assert created.structured_content["draft"]["status"] == "pending_approval"

        # Creating is side-effect free, and repeating it returns the same draft.
        async with harness.session_maker() as session:
            assert await TrackerRepo(session).get_by_opportunity_id(opportunity_id) is None
            assert await CalendarTaskRepo(session).get_by_action_draft_id(draft_id) is None
            assert await AuditRepo(session).list_events(resource_id=draft_id) == []

        repeated = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _tracker_payload(opportunity_id),
                "idempotency_key": key,
            },
        )
        assert repeated.is_error is False, repeated.content
        assert repeated.structured_content is not None
        assert repeated.structured_content["created"] is False
        assert repeated.structured_content["draft"]["id"] == str(draft_id)
        assert len(await _drafts_for_run(harness, run_id)) == 1

        # A wrong confirmation phrase executes nothing.
        refused = await client.call_tool(
            "approve_action",
            {"draft_id": str(draft_id), "confirmation_phrase": "approve"},
        )
        assert refused.is_error is True
        assert (await _drafts_for_run(harness, run_id))[0].status is (
            ActionDraftStatus.PENDING_APPROVAL
        )

        approved = await client.call_tool(
            "approve_action",
            {"draft_id": str(draft_id), "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE},
        )
        assert approved.is_error is False, approved.content
        assert approved.structured_content is not None
        assert approved.structured_content["draft"]["status"] == "executed"
        assert approved.structured_content["already_processed"] is False
        assert approved.structured_content["execution"]["outcome"] == "executed"
        audit_log_id = approved.structured_content["execution"]["audit_log_id"]

        replay = await client.call_tool(
            "approve_action",
            {"draft_id": str(draft_id), "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE},
        )
        assert replay.is_error is False, replay.content
        assert replay.structured_content is not None
        assert replay.structured_content["already_processed"] is True
        assert replay.structured_content["execution"]["outcome"] == "already_executed"
        assert replay.structured_content["execution"]["audit_log_id"] == audit_log_id

    async with harness.session_maker() as session:
        draft = await DraftRepo(session).get_by_id(draft_id)
        assert draft is not None
        assert draft.status is ActionDraftStatus.EXECUTED
        assert draft.approved_at is not None
        assert draft.executed_at is not None

        tracker = await TrackerRepo(session).get_by_opportunity_id(opportunity_id)
        assert tracker is not None
        assert tracker.stage is ApplicationStage.PREPARING
        assert tracker.notes == "Tailored resume."

        audits = await AuditRepo(session).list_events(resource_id=draft_id)
        assert len(audits) == 1
        assert str(audits[0].id) == audit_log_id
        assert audits[0].event_type == APPROVED_EXECUTION_EVENT
        assert audits[0].details_json["approved_at"] is not None
        assert audits[0].details_json["approved_by"] == "human"
        assert audits[0].details_json["target"]["type"] == "application_tracker"


async def test_calendar_draft_only_materializes_after_approval(
    harness: MCPActionHarness,
) -> None:
    opportunity_id, run_id = await _seed_run(harness)
    key = f"proofpath:{run_id}:calendar_block"

    async with Client(harness.server) as client:
        created = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _calendar_payload(),
                "idempotency_key": key,
            },
        )
        assert created.is_error is False, created.content
        assert created.structured_content is not None
        draft_id = uuid.UUID(created.structured_content["draft"]["id"])
        harness.draft_ids.append(draft_id)

        async with harness.session_maker() as session:
            assert await CalendarTaskRepo(session).get_by_action_draft_id(draft_id) is None

        approved = await client.call_tool(
            "approve_action",
            {"draft_id": str(draft_id), "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE},
        )
        assert approved.is_error is False, approved.content
        assert approved.structured_content is not None
        assert approved.structured_content["execution"]["calendar_task_id"] is not None

    async with harness.session_maker() as session:
        task = await CalendarTaskRepo(session).get_by_action_draft_id(draft_id)
        assert task is not None
        assert task.opportunity_id == opportunity_id
        assert task.title == "Interview prep block"
        tasks = (
            (
                await session.execute(
                    select(CalendarTask).where(CalendarTask.opportunity_id == opportunity_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(tasks) == 1


async def test_approve_action_refuses_a_rejected_draft(harness: MCPActionHarness) -> None:
    opportunity_id, run_id = await _seed_run(harness)
    async with harness.session_maker.begin() as session:
        draft, _ = await DraftRepo(session).create_or_get_draft(
            analysis_run_id=run_id,
            action_type="tracker_upsert",  # type: ignore[arg-type]
            payload_json=_tracker_payload(opportunity_id),
            idempotency_key=f"proofpath:{run_id}:tracker_upsert",
        )
        assert await DraftRepo(session).claim_decision(draft.id, decision="rejected")
    harness.draft_ids.append(draft.id)

    async with Client(harness.server) as client:
        result = await client.call_tool(
            "approve_action",
            {"draft_id": str(draft.id), "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE},
        )

    assert result.is_error is True
    assert "rejected" in str(result.content)
    async with harness.session_maker() as session:
        stored = await DraftRepo(session).get_by_id(draft.id)
        assert stored is not None
        assert stored.status is ActionDraftStatus.REJECTED
        assert await TrackerRepo(session).get_by_opportunity_id(opportunity_id) is None
        assert await AuditRepo(session).list_events(resource_id=draft.id) == []


async def test_create_action_draft_refuses_a_duplicate_key_with_new_content(
    harness: MCPActionHarness,
) -> None:
    opportunity_id, run_id = await _seed_run(harness)
    key = f"proofpath:{run_id}:tracker_upsert"

    async with Client(harness.server) as client:
        first = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _tracker_payload(opportunity_id, notes="First notes"),
                "idempotency_key": key,
            },
        )
        assert first.is_error is False, first.content
        assert first.structured_content is not None
        harness.draft_ids.append(uuid.UUID(first.structured_content["draft"]["id"]))

        conflicting = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _tracker_payload(opportunity_id, notes="Different notes"),
                "idempotency_key": key,
            },
        )

    assert conflicting.is_error is True
    assert "different payload" in str(conflicting.content)
    drafts = await _drafts_for_run(harness, run_id)
    assert len(drafts) == 1
    assert drafts[0].payload_json["notes"] == "First notes"


async def test_create_action_draft_refuses_a_run_paused_at_the_gate(
    harness: MCPActionHarness,
) -> None:
    opportunity_id, run_id = await _seed_run(harness, status=AnalysisRunStatus.AWAITING_APPROVAL)

    async with Client(harness.server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _tracker_payload(opportunity_id),
                "idempotency_key": f"proofpath:{run_id}:tracker_upsert",
            },
        )

    assert result.is_error is True
    assert "approval gate" in str(result.content)
    assert await _drafts_for_run(harness, run_id) == []


async def test_create_action_draft_rejects_a_non_canonical_key(
    harness: MCPActionHarness,
) -> None:
    opportunity_id, run_id = await _seed_run(harness)

    async with Client(harness.server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(run_id),
                "payload": _tracker_payload(opportunity_id),
                "idempotency_key": "proofpath:draft:1",
            },
        )

    assert result.is_error is True
    assert "canonical draft key" in str(result.content)
    assert await _drafts_for_run(harness, run_id) == []
