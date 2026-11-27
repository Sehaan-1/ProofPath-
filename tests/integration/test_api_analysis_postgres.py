"""Analysis endpoint with live Postgres/checkpointer and deterministic model stubs."""

import uuid

import httpx
import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete

from proofpath.config import Settings
from proofpath.db.models import (
    AuditLog,
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.main import create_app
from proofpath.schemas.actions import ContributionSprintPlan
from proofpath.schemas.analysis import ClaimVerdictSchema

pytestmark = pytest.mark.integration


async def test_analysis_run_is_durable_and_only_creates_pending_drafts(
    integration_database_url, monkeypatch
):
    settings = Settings(
        _env_file=None, database_url=PostgresDsn(integration_database_url), embedding_model="hash"
    )
    app = create_app(settings=settings)

    class Backend:
        async def compare(self, requirement, evidence, feedback):
            return ClaimVerdictSchema(
                requirement_id=requirement.id,
                verdict=VerdictType.UNSUPPORTED,
                claim_summary="No verified evidence",
            )

        async def plan(self, requirement, verdict):
            return ContributionSprintPlan(
                title="Practice",
                scope="Build a demo",
                repo_or_topic="Python",
                deliverables=["Tests"],
                estimate_days=2,
                target_gap_requirement_ids=[requirement.id],
                rationale="Practice the gap",
            )

    monkeypatch.setattr(
        "proofpath.api.analysis_routes.create_role_parser", lambda _settings: object()
    )
    monkeypatch.setattr(
        "proofpath.api.analysis_routes.create_graph_backend", lambda _settings: Backend()
    )
    async with app.router.lifespan_context(app):
        async with app.state.session_factory.begin() as session:
            repo = OpportunityRepo(session)
            opp = await repo.create_opportunity(
                company="Co", role_title="Engineer", raw_text="Python", sanitized_text="Python"
            )
            opp_id = opp.id
            rows = await repo.add_requirements(
                opp_id,
                [
                    RoleRequirementInput(
                        requirement_text="Python",
                        entity="Python",
                        category=RequirementCategory.LANGUAGE,
                        priority=RequirementPriority.MUST_HAVE,
                    )
                ],
            )
            req_id = rows[0].id
        run_id = None
        thread = None
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/analyses/run", json={"opportunity_id": str(opp_id)}
                )
                assert response.status_code == 201, response.text
                data = response.json()
                run_id = uuid.UUID(data["analysis_run_id"])
                thread = data["thread_id"]
                assert data["status"] == "awaiting_approval"
                assert data["verdicts"][0]["requirement_id"] == str(req_id)
                assert data["verdicts"][0]["verdict"] == "unsupported"
                assert data["contribution_sprint"]["target_gap_requirement_ids"] == [str(req_id)]
                assert all(d["status"] == "pending_approval" for d in data["action_drafts"])
                report = await client.get(f"/api/v1/analyses/{run_id}")
                assert report.status_code == 200
                assert report.json()["verdicts"] == data["verdicts"]
                async with app.state.session_factory() as session:
                    assert await TrackerRepo(session).get_by_opportunity_id(opp_id) is None
        finally:
            if thread is not None:
                await app.state.checkpointer.adelete_thread(thread)
            async with app.state.session_factory.begin() as session:
                if run_id is not None:
                    await session.execute(delete(AuditLog).where(AuditLog.resource_id == run_id))
                await session.execute(delete(Opportunity).where(Opportunity.id == opp_id))
