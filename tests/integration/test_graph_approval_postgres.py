"""Durable approval pause/resume against live PostgreSQL (Sub-slice 2.1.1).

Proves the properties that only a real checkpointer can prove:

1. the analysis run halts at `approval_gate` with the checkpoint committed and a
   real `pending_approval` draft row, and no node after the gate executes;
2. the pause survives closing and reopening the Postgres saver (process restart);
3. `Command(resume={"decision", "draft_id"})` bound to the run's `thread_id`
   records the decision and completes the run exactly once;
4. nothing touches `application_tracker`, calendars, or draft status here, and a
   second resume cannot consume the same pause.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    AnalysisRunStatus,
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.graph import workflow as workflow_module
from proofpath.graph.approval import ApprovalStateError
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.graph.resume import resume_approval_run
from proofpath.graph.state import initial_state
from proofpath.graph.workflow import build_workflow, workflow_config
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.opportunity import RequirementRecord

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class SeededRun:
    """One opportunity + requirement + analysis run bound to a unique thread."""

    settings: Settings
    factory: async_sessionmaker[AsyncSession]
    thread_id: str
    run_id: uuid.UUID
    opportunity_id: uuid.UUID
    requirement_id: uuid.UUID

    def workflow_kwargs(self, checkpointer: object) -> dict[str, object]:
        return {
            "checkpointer": checkpointer,
            "session_factory": self.factory,
            "settings": self.settings,
            "analysis_run_id": self.run_id,
            "thread_id": self.thread_id,
            "parser": MagicMock(),
            "embedder": MagicMock(),
            "comparator": MagicMock(),
            "sprint_planner": MagicMock(),
        }

    def resume_kwargs(
        self, checkpointer: object, decision: str, draft_id: uuid.UUID
    ) -> dict[str, object]:
        kwargs = self.workflow_kwargs(checkpointer)
        # The run is resolved from the thread: resuming never invents a run id.
        kwargs.pop("analysis_run_id")
        return kwargs | {"decision": decision, "draft_id": draft_id}


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, factory: async_sessionmaker[AsyncSession]
) -> None:
    """Keep the real wiring, drafts node, and gate; stub only the model calls."""

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


@pytest.fixture
async def seeded_run(
    db_engine: AsyncEngine,
    integration_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[SeededRun]:
    """Create one opportunity, requirement, and analysis run; clean up afterwards."""
    settings = Settings(_env_file=None, database_url=PostgresDsn(integration_database_url))
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    thread_id, run_id = str(uuid4()), uuid.uuid4()
    async with factory.begin() as session:
        opportunity = await OpportunityRepo(session).create_opportunity(
            company="Example",
            role_title="Platform Engineer",
            raw_text="SAP S/4HANA required",
            sanitized_text="SAP S/4HANA required",
        )
        opportunity_id = opportunity.id
        rows = await OpportunityRepo(session).add_requirements(
            opportunity_id,
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
            opportunity_id=opportunity_id, thread_id=thread_id, run_id=run_id
        )
    _patch_pipeline(monkeypatch, factory)
    try:
        yield SeededRun(
            settings=settings,
            factory=factory,
            thread_id=thread_id,
            run_id=run_id,
            opportunity_id=opportunity_id,
            requirement_id=rows[0].id,
        )
    finally:
        async with factory.begin() as session:
            await session.execute(delete(Opportunity).where(Opportunity.id == opportunity_id))


async def test_pause_is_durable_and_resume_is_bound_to_the_thread(
    seeded_run: SeededRun,
) -> None:
    # --- 1. run to the approval gate and pause durably ---------------------
    async with postgres_checkpointer(settings=seeded_run.settings) as saver:
        graph = build_workflow(**seeded_run.workflow_kwargs(saver))
        paused = await graph.ainvoke(
            initial_state(seeded_run.opportunity_id, settings=seeded_run.settings),
            workflow_config(seeded_run.thread_id),
        )
        assert paused["__interrupt__"]
        request = await graph.apending_approval()
        assert request is not None
        assert request.analysis_run_id == seeded_run.run_id
        assert len(request.drafts) == 1
        draft_id = next(iter(request.draft_ids))
        assert request.drafts[0].action_type.value == "tracker_upsert"
        # The interrupt payload is checkpointed: no personal draft contents leak.
        assert "notes" not in str(paused["__interrupt__"])

        async with seeded_run.factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                seeded_run.run_id, status=AnalysisRunStatus.AWAITING_APPROVAL
            )

    # --- 2. the pause lives in Postgres, not in this process --------------
    async with postgres_checkpointer(settings=seeded_run.settings) as saver:
        rebuilt = build_workflow(**seeded_run.workflow_kwargs(saver))
        assert (await rebuilt.apending_approval()) == request
        snapshot = await rebuilt.aget_state()
        assert snapshot.next == ("approval_gate",)

    async with seeded_run.factory() as session:
        run = await AnalysisRepo(session).get_run_by_id(seeded_run.run_id, include_drafts=True)
        assert run is not None
        assert run.status is AnalysisRunStatus.AWAITING_APPROVAL
        assert [draft.status for draft in run.action_drafts] == [ActionDraftStatus.PENDING_APPROVAL]

    # --- 3. resume with `Command(resume=...)` bound to `thread_id` --------
    async with postgres_checkpointer(settings=seeded_run.settings) as saver:
        result = await resume_approval_run(**seeded_run.resume_kwargs(saver, "approved", draft_id))
        assert result.status is AnalysisRunStatus.COMPLETED
        assert result.decision.decision == "approved"
        assert result.decision.draft_id == draft_id
        assert result.state["errors"] == []
        assert result.state["approval_request"].draft_ids == request.draft_ids

        # The pause has been consumed: a second resume cannot run the side effects
        # a second time.
        with pytest.raises(ApprovalStateError):
            await resume_approval_run(**seeded_run.resume_kwargs(saver, "approved", draft_id))
        await saver.adelete_thread(seeded_run.thread_id)

    async with seeded_run.factory() as session:
        run = await AnalysisRepo(session).get_run_by_id(seeded_run.run_id, include_drafts=True)
        assert run is not None
        assert run.status is AnalysisRunStatus.COMPLETED
        # Sub-slices 2.1.2/2.1.3 own draft transitions and tracker writes.
        assert [draft.status for draft in run.action_drafts] == [ActionDraftStatus.PENDING_APPROVAL]
        assert run.action_drafts[0].approved_at is None
        assert await TrackerRepo(session).get_by_opportunity_id(seeded_run.opportunity_id) is None


async def test_resume_rejects_a_draft_from_another_run(seeded_run: SeededRun) -> None:
    async with postgres_checkpointer(settings=seeded_run.settings) as saver:
        graph = build_workflow(**seeded_run.workflow_kwargs(saver))
        await graph.ainvoke(
            initial_state(seeded_run.opportunity_id, settings=seeded_run.settings),
            workflow_config(seeded_run.thread_id),
        )
        async with seeded_run.factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                seeded_run.run_id, status=AnalysisRunStatus.AWAITING_APPROVAL
            )
        with pytest.raises(ApprovalStateError):
            await resume_approval_run(**seeded_run.resume_kwargs(saver, "approved", uuid.uuid4()))
        snapshot = await graph.aget_state()
        assert snapshot.next == ("approval_gate",)
        with suppress(Exception):
            await saver.adelete_thread(seeded_run.thread_id)

    async with seeded_run.factory() as session:
        run = await AnalysisRepo(session).get_run_by_id(seeded_run.run_id)
        assert run is not None
        assert run.status is AnalysisRunStatus.AWAITING_APPROVAL
        drafts = await AnalysisRepo(session).get_run_by_id(seeded_run.run_id, include_drafts=True)
        assert drafts is not None
        assert [draft.status for draft in drafts.action_drafts] == [
            ActionDraftStatus.PENDING_APPROVAL
        ]
