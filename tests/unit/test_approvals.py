"""Approval decision idempotency (Sub-slice 2.1.2).

The service is exercised against an in-memory world: a draft, its run, a fake
`DraftRepo.claim_decision` compare-and-set, and a fake resume. Real PostgreSQL
behaviour is covered by `tests/integration/test_api_approval_postgres.py`.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from proofpath import approvals as approvals_module
from proofpath.approvals import DraftConflictError, DraftNotFoundError, decide_draft
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
)
from proofpath.db.repositories.draft import decided_status
from proofpath.graph.approval import ApprovalError, ApprovalStateError
from proofpath.graph.dependencies import GraphDependencies

pytestmark = pytest.mark.unit

PENDING = ActionDraftStatus.PENDING_APPROVAL


class World:
    """One draft plus its analysis run, with fake persistence and resume."""

    def __init__(
        self,
        *,
        draft_status: ActionDraftStatus = PENDING,
        run_status: AnalysisRunStatus = AnalysisRunStatus.AWAITING_APPROVAL,
        resume_outcome: str = "completed",
        race_winner: ActionDraftStatus | None = None,
    ) -> None:
        self.draft = SimpleNamespace(
            id=uuid.uuid4(),
            analysis_run_id=uuid.uuid4(),
            action_type=ActionType.TRACKER_UPSERT,
            status=draft_status,
            payload_json={
                "action_type": "tracker_upsert",
                "opportunity_id": str(uuid.uuid4()),
                "stage": ApplicationStage.PREPARING.value,
            },
            idempotency_key="proofpath:run:tracker_upsert",
            approved_at=(
                datetime.now(UTC)
                if draft_status in {ActionDraftStatus.APPROVED, ActionDraftStatus.EXECUTED}
                else None
            ),
            executed_at=None,
        )
        self.run = SimpleNamespace(
            id=self.draft.analysis_run_id,
            thread_id="thread-1",
            status=run_status,
        )
        self.resume_outcome = resume_outcome
        self.race_winner = race_winner
        self.claims: list[str] = []
        self.resumes: list[dict[str, Any]] = []

    # --- fake repository behaviour -------------------------------------
    def draft_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def get_by_id(self, draft_id: uuid.UUID) -> Any:
                if draft_id != world.draft.id:
                    return None
                return world.draft

            async def claim_decision(
                self,
                draft_id: uuid.UUID,
                *,
                decision: str,
                decided_at: datetime | None = None,
            ) -> bool:
                world.claims.append(decision)
                if world.race_winner is not None:
                    world.draft.status = world.race_winner
                    return False
                if world.draft.status is not PENDING:
                    return False
                world.draft.status = decided_status(decision)  # type: ignore[arg-type]
                world.draft.approved_at = (
                    decided_at or datetime.now(UTC) if decision == "approved" else None
                )
                return True

        return Repo

    def analysis_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            @staticmethod
            async def get_run_by_id(run_id: uuid.UUID, **_kwargs: object) -> Any:
                return world.run if run_id == world.run.id else None

        return Repo

    async def resume(self, **kwargs: Any) -> None:
        self.resumes.append(kwargs)
        if self.resume_outcome == "state_error":
            raise ApprovalStateError("Analysis run is no longer awaiting approval.")
        if self.resume_outcome == "failed":
            self.run.status = AnalysisRunStatus.FAILED
            raise ApprovalError("Approval resume could not be completed.")
        self.run.status = AnalysisRunStatus.COMPLETED


def _install(monkeypatch: pytest.MonkeyPatch, world: World) -> tuple[MagicMock, MagicMock]:
    statuses: list[Any] = []

    class Session:
        def __init__(self) -> None:
            self.refresh = AsyncMock(side_effect=lambda draft: statuses.append(draft.status))

    session = Session()

    @asynccontextmanager
    async def context():
        yield session

    factory = MagicMock(side_effect=lambda: context())
    factory.begin = lambda: context()

    monkeypatch.setattr(approvals_module, "DraftRepo", world.draft_repo())
    monkeypatch.setattr(approvals_module, "AnalysisRepo", world.analysis_repo())
    monkeypatch.setattr(approvals_module, "resume_approval_run", world.resume)
    return factory, session


def _dependencies() -> GraphDependencies:
    return GraphDependencies(
        parser=MagicMock(),
        embedder=MagicMock(),
        comparator=MagicMock(),
        sprint_planner=MagicMock(),
    )


def _kwargs(world: World, factory: MagicMock, decision: str = "approved") -> dict[str, Any]:
    return {
        "draft_id": world.draft.id,
        "decision": decision,
        "session_factory": factory,
        "checkpointer": MagicMock(),
        "settings": MagicMock(),
        "dependencies": _dependencies(),
    }


async def test_approve_commits_once_and_finishes_the_run(monkeypatch) -> None:
    world = World()
    factory, _ = _install(monkeypatch, world)

    result = await decide_draft(**_kwargs(world, factory))

    assert result.decision == "approved"
    assert result.already_processed is False
    assert result.draft.status is ActionDraftStatus.APPROVED
    assert result.draft.approved_at is not None
    assert result.run_status is AnalysisRunStatus.COMPLETED
    assert result.thread_id == world.run.thread_id
    assert world.claims == ["approved"]
    assert len(world.resumes) == 1
    assert world.resumes[0]["thread_id"] == world.run.thread_id
    assert world.resumes[0]["decision"] == "approved"
    assert world.resumes[0]["draft_id"] == world.draft.id


async def test_approving_twice_never_reapplies_the_decision(monkeypatch) -> None:
    world = World()
    factory, _ = _install(monkeypatch, world)

    first = await decide_draft(**_kwargs(world, factory))
    approved_at = first.draft.approved_at
    second = await decide_draft(**_kwargs(world, factory))

    assert first.already_processed is False
    assert second.already_processed is True
    assert second.draft.approved_at == approved_at
    # The decision CAS ran exactly once; the run was not resumed a second time.
    assert world.claims == ["approved"]
    assert len(world.resumes) == 1


async def test_reject_is_idempotent_and_never_resumes_twice(monkeypatch) -> None:
    world = World()
    factory, _ = _install(monkeypatch, world)

    first = await decide_draft(**_kwargs(world, factory, "rejected"))
    second = await decide_draft(**_kwargs(world, factory, "rejected"))

    assert first.draft.status is ActionDraftStatus.REJECTED
    assert first.already_processed is False
    assert second.already_processed is True
    assert world.claims == ["rejected"]
    assert len(world.resumes) == 1
    assert world.resumes[0]["decision"] == "rejected"


@pytest.mark.parametrize(
    ("existing", "decision"),
    [
        (ActionDraftStatus.REJECTED, "approved"),
        (ActionDraftStatus.APPROVED, "rejected"),
        (ActionDraftStatus.EXECUTED, "rejected"),
        (ActionDraftStatus.FAILED, "rejected"),
    ],
)
async def test_contradictory_decisions_are_conflicts(monkeypatch, existing, decision) -> None:
    world = World(draft_status=existing)
    factory, _ = _install(monkeypatch, world)

    with pytest.raises(DraftConflictError):
        await decide_draft(**_kwargs(world, factory, decision))

    assert world.claims == []
    assert world.resumes == []
    assert world.draft.status is existing


@pytest.mark.parametrize("existing", [ActionDraftStatus.EXECUTED, ActionDraftStatus.FAILED])
async def test_approved_draft_reports_already_processed_after_execution(
    monkeypatch, existing
) -> None:
    world = World(draft_status=existing, run_status=AnalysisRunStatus.COMPLETED)
    factory, _ = _install(monkeypatch, world)

    result = await decide_draft(**_kwargs(world, factory))

    assert result.already_processed is True
    assert world.claims == []
    assert world.resumes == []


async def test_missing_draft_or_run_is_not_found(monkeypatch) -> None:
    world = World()
    factory, _ = _install(monkeypatch, world)

    with pytest.raises(DraftNotFoundError):
        await decide_draft(**(_kwargs(world, factory) | {"draft_id": uuid.uuid4()}))

    orphan = World()
    orphan.draft.analysis_run_id = uuid.uuid4()
    factory, _ = _install(monkeypatch, orphan)
    with pytest.raises(DraftNotFoundError):
        await decide_draft(**_kwargs(orphan, factory))
    assert orphan.resumes == []


async def test_losing_a_concurrent_race_to_the_same_decision_is_idempotent(monkeypatch) -> None:
    world = World(race_winner=ActionDraftStatus.APPROVED)
    factory, session = _install(monkeypatch, world)

    result = await decide_draft(**_kwargs(world, factory))

    assert result.already_processed is True
    assert session.refresh.await_count == 1
    # The run was still paused, so the retry is allowed to finish it.
    assert len(world.resumes) == 1


async def test_losing_a_concurrent_race_to_the_opposite_decision_is_a_conflict(monkeypatch) -> None:
    world = World(race_winner=ActionDraftStatus.REJECTED)
    factory, _ = _install(monkeypatch, world)

    with pytest.raises(DraftConflictError):
        await decide_draft(**_kwargs(world, factory))

    assert world.resumes == []


@pytest.mark.parametrize("run_status", [AnalysisRunStatus.COMPLETED, AnalysisRunStatus.FAILED])
async def test_decision_is_recorded_without_resuming_a_finalised_run(
    monkeypatch, run_status
) -> None:
    world = World(run_status=run_status)
    factory, _ = _install(monkeypatch, world)

    result = await decide_draft(**_kwargs(world, factory))

    assert result.draft.status is ActionDraftStatus.APPROVED
    assert result.run_status is run_status
    assert world.resumes == []


async def test_resume_race_is_not_reported_as_a_failure(monkeypatch) -> None:
    world = World(resume_outcome="state_error")
    factory, _ = _install(monkeypatch, world)

    result = await decide_draft(**_kwargs(world, factory))

    assert result.draft.status is ActionDraftStatus.APPROVED
    assert len(world.resumes) == 1


async def test_resume_failure_keeps_the_committed_decision(monkeypatch) -> None:
    world = World(resume_outcome="failed")
    factory, _ = _install(monkeypatch, world)

    result = await decide_draft(**_kwargs(world, factory))

    assert result.draft.status is ActionDraftStatus.APPROVED
    assert result.run_status is AnalysisRunStatus.FAILED


async def test_invalid_decision_is_rejected_before_any_work(monkeypatch) -> None:
    world = World()
    factory, _ = _install(monkeypatch, world)

    with pytest.raises(ValueError, match=r"approved.*rejected"):
        await decide_draft(**_kwargs(world, factory, "maybe"))

    assert world.claims == []
