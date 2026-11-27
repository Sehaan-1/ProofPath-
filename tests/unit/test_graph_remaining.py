"""Unit coverage for planner, parsing, model adapters and draft safety."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from proofpath.config import LLMProvider, Settings
from proofpath.db.models import RequirementCategory, RequirementPriority, VerdictType
from proofpath.graph.backends import GraphModelError, create_graph_backend
from proofpath.graph.nodes import (
    analyse_role_node,
    create_action_drafts_node,
    planner_route,
    planner_router,
)
from proofpath.graph.state import initial_state
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.opportunity import ParsedRole, RequirementItem

pytestmark = pytest.mark.unit


def _opportunity(opp_id):
    return SimpleNamespace(id=opp_id, raw_text="Python required")


async def test_planner_validates_and_reads_status(monkeypatch):
    opp_id = uuid4()
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=_opportunity(opp_id))
    tracker = MagicMock()
    tracker.get_by_opportunity_id = AsyncMock(
        return_value=SimpleNamespace(stage=SimpleNamespace(value="saved"))
    )
    monkeypatch.setattr("proofpath.graph.nodes.OpportunityRepo", lambda _session: repo)
    monkeypatch.setattr("proofpath.graph.nodes.TrackerRepo", lambda _session: tracker)
    state = initial_state(opp_id, settings=Settings(_env_file=None))
    state.update(await planner_router(state, session=MagicMock(), step_budget=13))
    assert state["tracker_stage"] == "saved"
    assert state["step_budget"] == 13
    assert planner_route(state) == "analyse_role"
    repo.get_by_id.return_value = None
    with pytest.raises(ValueError, match="does not exist"):
        await planner_router(state, session=MagicMock())


async def test_parse_persists_once_and_reuses_entities(monkeypatch):
    opp_id, req_id = uuid4(), uuid4()
    opportunity = _opportunity(opp_id)
    result = MagicMock()
    result.scalar_one_or_none.return_value = opportunity
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    row = SimpleNamespace(
        id=req_id,
        opportunity_id=opp_id,
        entity="Python",
        requirement_text="Python required",
        category=RequirementCategory.LANGUAGE,
        priority=RequirementPriority.MUST_HAVE,
    )
    repo = MagicMock()
    repo.list_requirements = AsyncMock(side_effect=[[], [row]])
    repo.add_requirements = AsyncMock(return_value=[row])
    monkeypatch.setattr("proofpath.graph.nodes.OpportunityRepo", lambda _session: repo)
    parser = MagicMock()
    parser.parse = AsyncMock(
        return_value=ParsedRole(
            company="Co",
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
    )
    state = initial_state(opp_id, settings=Settings(_env_file=None))
    parsed = await analyse_role_node(state, session=session, parser=parser)
    assert parsed["requirements"][0].id == req_id
    assert repo.add_requirements.await_args.args[1][0].entity == "Python"
    again = await analyse_role_node(state, session=session, parser=parser)
    assert again["requirements"][0].id == req_id
    assert parser.parse.await_count == 1


async def test_drafts_require_verified_policy():
    state = initial_state(uuid4(), settings=Settings(_env_file=None))
    session = MagicMock()
    with pytest.raises(ValueError, match="Policy verification"):
        await create_action_drafts_node(state, session=session, analysis_run_id=uuid4())
    session.execute.assert_not_called()


async def test_openai_adapter_uses_only_supplied_data_and_handles_refusal():
    req_id = uuid4()
    from proofpath.schemas.opportunity import RequirementRecord

    req = RequirementRecord(
        id=req_id,
        opportunity_id=uuid4(),
        entity="Python",
        requirement_text="Python",
        category=RequirementCategory.LANGUAGE,
        priority=RequirementPriority.MUST_HAVE,
    )
    verdict = ClaimVerdictSchema(
        requirement_id=req_id, verdict=VerdictType.UNSUPPORTED, claim_summary="No evidence"
    )
    message = SimpleNamespace(refusal=None, parsed=verdict)
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)])
    )
    backend = create_graph_backend(Settings(_env_file=None), client=client)
    assert await backend.compare(req, [], []) == verdict
    kwargs = client.chat.completions.parse.await_args.kwargs
    assert "untrusted" in kwargs["messages"][0]["content"]
    assert '"evidence": []' in kwargs["messages"][1]["content"]
    message.refusal = "no"
    with pytest.raises(GraphModelError):
        await backend.compare(req, [], [])
    with pytest.raises(ValueError, match="only OpenAI"):
        create_graph_backend(
            Settings(_env_file=None, llm_provider=LLMProvider.OLLAMA), client=client
        )


async def test_drafts_are_replay_safe_and_preserve_tracker(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from proofpath.db.models import ActionDraftStatus, ApplicationStage
    from proofpath.schemas.actions import ContributionSprintPlan
    from proofpath.schemas.opportunity import RequirementRecord

    opp_id, req_id, run_id = uuid4(), uuid4(), uuid4()
    state = initial_state(opp_id, settings=Settings(_env_file=None))
    state["requirements"] = [
        RequirementRecord(
            id=req_id,
            opportunity_id=opp_id,
            entity="Python",
            requirement_text="Python",
            category=RequirementCategory.LANGUAGE,
            priority=RequirementPriority.MUST_HAVE,
        )
    ]
    state["verdicts"] = [
        ClaimVerdictSchema(
            requirement_id=req_id, verdict=VerdictType.UNSUPPORTED, claim_summary="No evidence"
        )
    ]
    state["policy_verified"] = True
    state["contribution_sprint"] = ContributionSprintPlan(
        title="Practice",
        scope="Prototype",
        repo_or_topic="Python",
        deliverables=["tests"],
        estimate_days=2,
        target_gap_requirement_ids=[req_id],
        rationale="Practice",
    )
    run = SimpleNamespace(opportunity_id=opp_id)
    result = MagicMock()
    result.scalar_one_or_none.return_value = run
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    tracker = SimpleNamespace(stage=ApplicationStage.INTERVIEWING, notes="keep")
    tracker_repo = MagicMock()
    tracker_repo.get_by_opportunity_id = AsyncMock(return_value=tracker)
    monkeypatch.setattr("proofpath.graph.nodes.TrackerRepo", lambda _session: tracker_repo)
    stored = {}

    class FakeDraftRepo:
        def __init__(self, session):
            self.session = session

        async def create_or_get_draft(
            self, *, analysis_run_id, action_type, payload_json, idempotency_key
        ):
            if idempotency_key not in stored:
                stored[idempotency_key] = SimpleNamespace(
                    id=uuid4(),
                    analysis_run_id=analysis_run_id,
                    action_type=action_type,
                    status=ActionDraftStatus.PENDING_APPROVAL,
                    payload_json=payload_json,
                    idempotency_key=idempotency_key,
                    approved_at=None,
                    executed_at=None,
                )
            return stored[idempotency_key], False

    monkeypatch.setattr("proofpath.graph.nodes.DraftRepo", FakeDraftRepo)
    with pytest.raises(ValueError, match="Explicit calendar"):
        await create_action_drafts_node(state, session=session, analysis_run_id=run_id)
    assert not stored
    start = datetime.now(UTC)
    kwargs = {
        "session": session,
        "analysis_run_id": run_id,
        "calendar_start": start,
        "calendar_end": start + timedelta(hours=1),
    }
    first = await create_action_drafts_node(state, **kwargs)
    second = await create_action_drafts_node(state, **kwargs)
    assert len(stored) == 2
    assert [item.id for item in first["draft_actions"]] == [
        item.id for item in second["draft_actions"]
    ]
    assert first["draft_actions"][0].payload.stage == ApplicationStage.INTERVIEWING
    assert first["draft_actions"][0].payload.notes == "keep"
    assert tracker.stage == ApplicationStage.INTERVIEWING
