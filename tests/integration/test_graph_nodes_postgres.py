"""Real-database coverage for parsing, tracker reads and draft replay."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraft,
    ActionDraftStatus,
    ApplicationStage,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.graph.nodes import (
    analyse_role_node,
    create_action_drafts_node,
    planner_route,
    planner_router,
)
from proofpath.graph.state import initial_state
from proofpath.schemas.actions import ContributionSprintPlan
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.opportunity import ParsedRole, RequirementItem

pytestmark = pytest.mark.integration


async def test_parse_existing_opportunity_and_draft_replay(db_session: AsyncSession) -> None:
    repo = OpportunityRepo(db_session)
    opp = await repo.create_opportunity(
        company="Example",
        role_title="Engineer",
        raw_text="Python required",
        sanitized_text="Python required",
    )
    run = await AnalysisRepo(db_session).create_run(
        opportunity_id=opp.id, thread_id=f"test-{opp.id}"
    )
    await TrackerRepo(db_session).upsert(
        opportunity_id=opp.id, stage=ApplicationStage.INTERVIEWING, notes="Keep these notes"
    )
    state = initial_state(opp.id, settings=Settings(_env_file=None))

    class Parser:
        calls = 0

        async def parse(self, prompt):
            self.calls += 1
            assert "<untrusted_job_description>" in prompt
            return ParsedRole(
                company="Example",
                role_title="Engineer",
                requirements=[
                    RequirementItem(
                        requirement_text="Python required",
                        entity="Python",
                        category=RequirementCategory.LANGUAGE,
                        priority=RequirementPriority.MUST_HAVE,
                    )
                ],
            )

    parser = Parser()
    state.update(await planner_router(state, session=db_session))
    assert state["tracker_stage"] == "interviewing"
    assert planner_route(state) == "analyse_role"
    state.update(await analyse_role_node(state, session=db_session, parser=parser))
    assert parser.calls == 1
    assert planner_route(state) == "retrieve_evidence"
    req = state["requirements"][0]
    # A fresh graph state after a crash must reuse the stored row and entity.
    fresh = initial_state(opp.id, settings=Settings(_env_file=None))
    fresh.update(await analyse_role_node(fresh, session=db_session, parser=parser))
    assert fresh["requirements"][0].id == req.id
    assert fresh["requirements"][0].entity == "Python"
    assert parser.calls == 1
    state["verdicts"] = [
        ClaimVerdictSchema(
            requirement_id=req.id, verdict=VerdictType.UNSUPPORTED, claim_summary="No evidence"
        )
    ]
    state["policy_verified"] = True
    state["contribution_sprint"] = ContributionSprintPlan(
        title="Practice",
        scope="Build a demo",
        repo_or_topic="Python",
        deliverables=["Tests"],
        estimate_days=2,
        target_gap_requirement_ids=[req.id],
        rationale="Practice",
    )
    start = datetime.now(UTC)
    kwargs = {
        "session": db_session,
        "analysis_run_id": run.id,
        "calendar_start": start,
        "calendar_end": start + timedelta(hours=1),
    }
    first = await create_action_drafts_node(state, **kwargs)
    second = await create_action_drafts_node(state, **kwargs)
    assert [r.id for r in first["draft_actions"]] == [r.id for r in second["draft_actions"]]
    assert all(r.status == ActionDraftStatus.PENDING_APPROVAL for r in first["draft_actions"])
    tracker = await TrackerRepo(db_session).get_by_opportunity_id(opp.id)
    assert tracker.stage == ApplicationStage.INTERVIEWING
    assert tracker.notes == "Keep these notes"
    assert (
        await db_session.execute(
            select(func.count())
            .select_from(ActionDraft)
            .where(ActionDraft.analysis_run_id == run.id)
        )
    ).scalar_one() == 2
