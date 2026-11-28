"""Approval API against live PostgreSQL (Sub-slice 2.1.2, end to end).

Runs the real workflow to the approval gate, then drives the HTTP API:

* `GET /api/v1/drafts?status=pending_approval` surfaces the review queue;
* `POST /api/v1/drafts/{id}/approve|reject` commits the decision and finishes the
  paused run on its own thread;
* repeating an identical decision is idempotent — one `approved_at`, one resume,
  no duplicate side effects — and a contradictory decision is a 409.

Nothing here writes to `application_tracker` or `audit_logs`: approved-only
executors arrive in Sub-slice 2.1.3.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete, select

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    AnalysisRunStatus,
    AuditLog,
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.graph import workflow as workflow_module
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.graph.dependencies import create_graph_dependencies
from proofpath.graph.state import initial_state
from proofpath.graph.workflow import build_workflow, workflow_config
from proofpath.main import create_app
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.opportunity import RequirementRecord

pytestmark = pytest.mark.integration


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    """Keep the real wiring, drafts node, and gate; stub the model calls."""

    async def analyse(state: dict, **_kwargs: object) -> dict[str, object]:
        async with factory() as session:
            rows = await OpportunityRepo(session).list_requirements(state["opportunity_id"])
            requirements = [
                RequirementRecord(
                    id=row.id,
                    opportunity_id=row.opportunity_id,
                    requirement_text=row.requirement_text,
                    entity=row.entity,
                    category=row.category,
                    priority=row.priority,
                )
                for row in rows
            ]
        return {"requirements": requirements, "step_count": state.get("step_count", 0) + 1}

    async def retrieve(state: dict, **_kwargs: object) -> dict[str, object]:
        return {
            "retrieved_evidence": {str(item.id): [] for item in state["requirements"]},
            "step_count": state.get("step_count", 0) + 1,
        }

    async def compare(state: dict, **_kwargs: object) -> dict[str, object]:
        return {
            "verdicts": [
                ClaimVerdictSchema(
                    requirement_id=item.id,
                    verdict=VerdictType.UNSUPPORTED,
                    claim_summary="Insufficient verified evidence for this requirement.",
                )
                for item in state["requirements"]
            ],
            "step_count": state.get("step_count", 0) + 1,
        }

    async def policy(state: dict, **_kwargs: object) -> dict[str, object]:
        return {
            "policy_violations": [],
            "policy_retry": False,
            "policy_verified": True,
            "step_count": state.get("step_count", 0) + 1,
        }

    async def sprint(state: dict, **_kwargs: object) -> dict[str, object]:
        return {"contribution_sprint": None, "step_count": state.get("step_count", 0) + 1}

    monkeypatch.setattr(workflow_module, "analyse_role_node", analyse)
    monkeypatch.setattr(workflow_module, "retrieve_evidence_node", retrieve)
    monkeypatch.setattr(workflow_module, "compare_evidence_node", compare)
    monkeypatch.setattr(workflow_module, "policy_verifier_node", policy)
    monkeypatch.setattr(workflow_module, "recommend_contribution_sprint_node", sprint)


async def _seed_and_pause(db_engine, settings: Settings, monkeypatch) -> dict[str, Any]:
    """Create a run, drive it to the approval gate, and mark it awaiting approval."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    _patch_pipeline(monkeypatch, factory)
    thread_id, run_id = str(uuid4()), uuid.uuid4()
    async with factory.begin() as session:
        opportunity = await OpportunityRepo(session).create_opportunity(
            company="Example",
            role_title="Platform Engineer",
            raw_text="SAP S/4HANA required",
            sanitized_text="SAP S/4HANA required",
        )
        await OpportunityRepo(session).add_requirements(
            opportunity.id,
            [
                RoleRequirementInput(
                    requirement_text="SAP S/4HANA",
                    entity="SAP",
                    category=RequirementCategory.DOMAIN,
                    priority=RequirementPriority.MUST_HAVE,
                )
            ],
        )
        await AnalysisRepo(session).create_run(
            opportunity_id=opportunity.id, thread_id=thread_id, run_id=run_id
        )
    async with postgres_checkpointer(settings=settings) as saver:
        graph = build_workflow(
            checkpointer=saver,
            session_factory=factory,
            settings=settings,
            analysis_run_id=run_id,
            thread_id=thread_id,
            parser=MagicMock(),
            embedder=MagicMock(),
            comparator=MagicMock(),
            sprint_planner=MagicMock(),
        )
        paused = await graph.ainvoke(
            initial_state(opportunity.id, settings=settings), workflow_config(thread_id)
        )
        assert paused["__interrupt__"]
        request = await graph.apending_approval()
        assert request is not None
        draft_id = next(iter(request.draft_ids))
        async with factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                run_id, status=AnalysisRunStatus.AWAITING_APPROVAL
            )
    return {
        "factory": factory,
        "thread_id": thread_id,
        "run_id": run_id,
        "opportunity_id": opportunity.id,
        "draft_id": draft_id,
    }


@pytest.fixture
async def approval_app(integration_database_url: str, monkeypatch: pytest.MonkeyPatch):
    """App wired to the integration database, with a dummy model key."""
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        llm_api_key="test-key",
    )
    create_graph_dependencies(settings)  # fail early if wiring cannot be built
    app = create_app(settings=settings)
    seeded: list[dict[str, Any]] = []
    try:
        yield app, settings, seeded
    finally:
        for item in seeded:
            async with item["factory"].begin() as session:
                await session.execute(
                    delete(Opportunity).where(Opportunity.id == item["opportunity_id"])
                )
            async with postgres_checkpointer(settings=settings) as saver:
                with suppress(Exception):
                    await saver.adelete_thread(item["thread_id"])


async def test_approve_is_idempotent_and_finishes_the_run(
    db_engine, integration_database_url: str, approval_app, monkeypatch
) -> None:
    app, settings, seeded = approval_app
    run = await _seed_and_pause(db_engine, settings, monkeypatch)
    seeded.append(run)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        queue = await client.get("/api/v1/drafts", params={"status": "pending_approval"})
        assert queue.status_code == 200
        assert [item["id"] for item in queue.json()] == [str(run["draft_id"])]
        assert queue.json()[0]["status"] == "pending_approval"

        first = await client.post(f"/api/v1/drafts/{run['draft_id']}/approve")
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["decision"] == "approved"
        assert body["already_processed"] is False
        assert body["run_status"] == "completed"
        assert body["thread_id"] == run["thread_id"]
        assert body["draft"]["status"] == "approved"
        approved_at = body["draft"]["approved_at"]
        assert approved_at is not None
        assert body["draft"]["idempotency_key"] == (f"proofpath:{run['run_id']}:tracker_upsert")

        again = await client.post(f"/api/v1/drafts/{run['draft_id']}/approve")
        assert again.status_code == 200, again.text
        assert again.json()["already_processed"] is True
        assert again.json()["draft"]["approved_at"] == approved_at
        assert again.json()["run_status"] == "completed"

        conflict = await client.post(f"/api/v1/drafts/{run['draft_id']}/reject")
        assert conflict.status_code == 409
        assert conflict.headers["content-type"].startswith("application/problem+json")

        approved = await client.get("/api/v1/drafts", params={"status": "approved"})
        assert [item["id"] for item in approved.json()] == [str(run["draft_id"])]

        missing = await client.post(f"/api/v1/drafts/{uuid.uuid4()}/approve")
        assert missing.status_code == 404

    async with run["factory"]() as session:
        stored = await AnalysisRepo(session).get_run_by_id(run["run_id"], include_drafts=True)
        assert stored is not None
        assert stored.status is AnalysisRunStatus.COMPLETED
        assert [draft.status for draft in stored.action_drafts] == [ActionDraftStatus.APPROVED]
        assert stored.action_drafts[0].approved_at is not None
        # Side effects are Sub-slice 2.1.3: nothing was tracked or audited here.
        assert await TrackerRepo(session).get_by_opportunity_id(run["opportunity_id"]) is None
        audits = (
            (await session.execute(select(AuditLog).where(AuditLog.resource_id == run["run_id"])))
            .scalars()
            .all()
        )
        assert audits == []


async def test_reject_completes_the_run_without_side_effects(
    db_engine, integration_database_url: str, approval_app, monkeypatch
) -> None:
    app, settings, seeded = approval_app
    run = await _seed_and_pause(db_engine, settings, monkeypatch)
    seeded.append(run)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post(f"/api/v1/drafts/{run['draft_id']}/reject")
        assert first.status_code == 200, first.text
        assert first.json()["draft"]["status"] == "rejected"
        assert first.json()["already_processed"] is False
        assert first.json()["draft"]["approved_at"] is None

        again = await client.post(f"/api/v1/drafts/{run['draft_id']}/reject")
        assert again.status_code == 200
        assert again.json()["already_processed"] is True

        approve_after_reject = await client.post(f"/api/v1/drafts/{run['draft_id']}/approve")
        assert approve_after_reject.status_code == 409

        pending = await client.get("/api/v1/drafts", params={"status": "pending_approval"})
        assert pending.json() == []

    async with run["factory"]() as session:
        stored = await AnalysisRepo(session).get_run_by_id(run["run_id"], include_drafts=True)
        assert stored is not None
        assert stored.status is AnalysisRunStatus.COMPLETED
        assert [draft.status for draft in stored.action_drafts] == [ActionDraftStatus.REJECTED]
        assert await TrackerRepo(session).get_by_opportunity_id(run["opportunity_id"]) is None
