"""Unit tests for repository invariants, guards, and async session configuration."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from proofpath.db import (
    EMBEDDING_DIM,
    ActionDraft,
    ActionDraftStatus,
    ActionType,
    AnalysisRepo,
    AnalysisRun,
    AnalysisRunStatus,
    AuditRepo,
    ClaimVerdictInput,
    DatabaseSessionManager,
    DraftRepo,
    EvidenceChunkInput,
    EvidenceRepo,
    OpportunityRepo,
    TrackerRepo,
    VerdictType,
    create_db_engine,
    create_session_factory,
)


def _mock_session_with_scalar(scalar_value: Any) -> AsyncMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_value
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()
    return session


@pytest.mark.unit
def test_session_factory_disables_expire_on_commit() -> None:
    engine = create_db_engine("postgresql+asyncpg://proofpath:proofpath@localhost:5432/proofpath")
    factory = create_session_factory(engine)
    assert factory.kw["expire_on_commit"] is False
    assert factory.kw["autoflush"] is False

    manager = DatabaseSessionManager(
        "postgresql+asyncpg://proofpath:proofpath@localhost:5432/proofpath"
    )
    assert manager.engine.url.drivername == "postgresql+asyncpg"
    assert manager.session_factory.kw["expire_on_commit"] is False


@pytest.mark.unit
async def test_evidence_repo_validates_embedding_dimensions_and_short_circuits_empty() -> None:
    session = AsyncMock()
    repo = EvidenceRepo(session)

    assert await repo.get_chunks_by_ids([]) == []
    assert await repo.get_existing_chunk_ids([]) == set()
    session.execute.assert_not_awaited()

    bad_chunk = EvidenceChunkInput(
        chunk_index=0,
        content="Python & FastAPI",
        token_count=4,
        embedding=[0.1] * (EMBEDDING_DIM - 1),
    )
    with pytest.raises(ValueError, match=str(EMBEDDING_DIM)):
        await repo.add_chunks(uuid.uuid4(), [bad_chunk])


@pytest.mark.unit
async def test_analysis_repo_enforces_citation_and_non_negative_metrics() -> None:
    run = AnalysisRun(
        id=uuid.uuid4(),
        opportunity_id=uuid.uuid4(),
        thread_id="thread-guard-test",
        status=AnalysisRunStatus.RUNNING,
    )
    session = _mock_session_with_scalar(run)
    repo = AnalysisRepo(session)

    assert await repo.upsert_verdicts(run.id, []) == []
    session.execute.assert_not_awaited()

    for verdict_type in (VerdictType.SUPPORTED, VerdictType.PARTIAL):
        with pytest.raises(ValueError, match="cited_chunk_id"):
            await repo.upsert_verdicts(
                run.id,
                [
                    ClaimVerdictInput(
                        requirement_id=uuid.uuid4(),
                        verdict=verdict_type,
                        claim_summary="Uncited claim",
                        cited_chunk_ids=[],
                    )
                ],
            )

    with pytest.raises(ValueError, match="total_tokens"):
        await repo.update_run_metrics(run.id, total_tokens=-1)

    with pytest.raises(ValueError, match="estimated_cost_usd"):
        await repo.update_run_metrics(run.id, estimated_cost_usd=Decimal("-0.01"))

    with pytest.raises(ValueError, match="latency_ms"):
        await repo.update_run_metrics(run.id, latency_ms=-5)


@pytest.mark.unit
async def test_analysis_repo_claim_is_a_single_compare_and_set() -> None:
    run_id = uuid.uuid4()

    claimed_session = _mock_session_with_scalar(run_id)
    assert await AnalysisRepo(claimed_session).claim_awaiting_approval(run_id) is True
    statement = claimed_session.execute.await_args.args[0]
    compiled = str(statement)
    assert "UPDATE analysis_runs" in compiled
    assert "analysis_runs.status = " in compiled
    assert "RETURNING analysis_runs.id" in compiled
    assert statement._execution_options["synchronize_session"] is False

    # A run that already left `awaiting_approval` returns no row: no second resume.
    lost_session = _mock_session_with_scalar(None)
    assert await AnalysisRepo(lost_session).claim_awaiting_approval(run_id) is False


@pytest.mark.unit
async def test_draft_repo_enforces_approval_before_execution() -> None:
    pending_draft = ActionDraft(
        id=uuid.uuid4(),
        analysis_run_id=uuid.uuid4(),
        action_type=ActionType.TRACKER_UPSERT,
        payload_json={"stage": "preparing"},
        status=ActionDraftStatus.PENDING_APPROVAL,
        idempotency_key="run-1:tracker_upsert",
        approved_at=None,
        executed_at=None,
    )
    session = _mock_session_with_scalar(pending_draft)
    repo = DraftRepo(session)

    with pytest.raises(ValueError, match="explicit human approval"):
        await repo.mark_executed(pending_draft.id)

    with pytest.raises(ValueError, match="expected 'approved'"):
        await repo.mark_failed(pending_draft.id)

    # Approve transitions pending_approval -> approved with UTC timestamp.
    approved = await repo.approve(pending_draft.id)
    assert approved is not None
    assert approved.status is ActionDraftStatus.APPROVED
    assert approved.approved_at is not None

    # Double-approve or reject after approval is rejected.
    with pytest.raises(ValueError, match="pending_approval"):
        await repo.approve(pending_draft.id)
    with pytest.raises(ValueError, match="pending_approval"):
        await repo.reject(pending_draft.id)

    # Now execution succeeds and stamps executed_at.
    executed = await repo.mark_executed(pending_draft.id)
    assert executed is not None
    assert executed.status is ActionDraftStatus.EXECUTED
    assert executed.executed_at is not None


@pytest.mark.unit
def test_repositories_and_append_only_audit_contract() -> None:
    assert EvidenceRepo is not None
    assert OpportunityRepo is not None
    assert AnalysisRepo is not None
    assert DraftRepo is not None
    assert TrackerRepo is not None

    # AuditRepo must be strictly append-only (no update/delete helpers).
    audit_methods = {name for name in dir(AuditRepo) if not name.startswith("_")}
    assert audit_methods == {"record", "list_events"}
