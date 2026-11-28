"""State defaults and checkpoint serialization contracts."""

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    ApplicationStage,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.graph.checkpointer import _STATE_TYPES, postgres_checkpointer, thread_config
from proofpath.graph.state import (
    ApprovalDecision,
    ApprovalDraftSummary,
    ApprovalRequest,
    PolicyViolation,
    initial_state,
)
from proofpath.policy.ledger_validator import DowngradeReason
from proofpath.schemas.actions import (
    ActionDraftRecord,
    ContributionSprintPlan,
    TrackerUpsertPayload,
)
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.evidence import EvidenceSnippet
from proofpath.schemas.opportunity import RequirementRecord

pytestmark = pytest.mark.unit


def test_initial_state_is_independent_and_uses_settings() -> None:
    settings = Settings(_env_file=None, max_agent_retries=4)
    opp = uuid.uuid4()
    first = initial_state(opp, settings=settings)
    second = initial_state(opp, settings=settings)
    assert first["max_retries"] == 4
    assert first["step_count"] == first["retry_count"] == 0
    first["errors"].append({"node": "n", "code": "x", "message": "failure", "retryable": False})
    assert second["errors"] == []
    assert first["retrieved_evidence"] == {}


def test_state_types_round_trip_with_strict_serializer() -> None:
    opp, req, chunk, source, run = (uuid.uuid4() for _ in range(5))
    state = initial_state(opp, settings=Settings(_env_file=None))
    state["requirements"] = [
        RequirementRecord(
            id=req,
            opportunity_id=opp,
            entity="Python",
            requirement_text="Python",
            category=RequirementCategory.LANGUAGE,
            priority=RequirementPriority.MUST_HAVE,
        )
    ]
    state["retrieved_evidence"] = {
        str(req): [
            EvidenceSnippet(
                chunk_id=chunk, source_id=source, canonical_uri="file:///test", content="Python"
            )
        ]
    }
    state["verdicts"] = [
        ClaimVerdictSchema(
            requirement_id=req,
            verdict=VerdictType.SUPPORTED,
            claim_summary="Python",
            cited_chunk_ids=[chunk],
        )
    ]
    state["policy_violations"] = [
        PolicyViolation(requirement_id=req, message="Unknown citation", unknown_chunk_ids=[chunk]),
        PolicyViolation(
            requirement_id=req,
            message="No citation",
            downgrade_reason=DowngradeReason.EMPTY_CITATIONS,
        ),
    ]
    state["contribution_sprint"] = ContributionSprintPlan(
        title="Sprint",
        scope="Build",
        repo_or_topic="Python",
        deliverables=["Tests"],
        estimate_days=2,
        target_gap_requirement_ids=[req],
        rationale="Practice",
    )
    state["draft_actions"] = [
        ActionDraftRecord(
            id=uuid.uuid4(),
            analysis_run_id=run,
            action_type=ActionType.TRACKER_UPSERT,
            status=ActionDraftStatus.PENDING_APPROVAL,
            payload=TrackerUpsertPayload(
                action_type=ActionType.TRACKER_UPSERT,
                opportunity_id=opp,
                stage=ApplicationStage.SAVED,
            ),
            idempotency_key="key",
            approved_at=datetime.now(UTC),
        )
    ]
    serde = JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)
    result = serde.loads_typed(serde.dumps_typed(state))
    assert result == state
    assert isinstance(result["requirements"][0], RequirementRecord)
    assert isinstance(result["retrieved_evidence"][str(req)][0], EvidenceSnippet)
    assert isinstance(result["policy_violations"][0], PolicyViolation)
    assert result["policy_violations"][1].downgrade_reason == DowngradeReason.EMPTY_CITATIONS
    assert isinstance(result["draft_actions"][0].payload, TrackerUpsertPayload)


def test_approval_state_types_round_trip_with_strict_serializer() -> None:
    run, draft = uuid.uuid4(), uuid.uuid4()
    request = ApprovalRequest(
        analysis_run_id=run,
        drafts=[
            ApprovalDraftSummary(
                draft_id=draft,
                action_type=ActionType.TRACKER_UPSERT,
                summary="Update application tracker stage to 'preparing'",
            )
        ],
    )
    state = initial_state(uuid.uuid4(), settings=Settings(_env_file=None))
    state["approval_request"] = request
    state["approval_decision"] = ApprovalDecision(decision="approved", draft_id=draft)

    serde = JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)
    result = serde.loads_typed(serde.dumps_typed(state))
    assert result == state
    assert isinstance(result["approval_request"], ApprovalRequest)
    assert result["approval_request"].draft_ids == {draft}
    assert result["approval_decision"] == ApprovalDecision(decision="approved", draft_id=draft)


def test_thread_config_rejects_blank_without_rewriting_id() -> None:
    for value in ("", "  "):
        with pytest.raises(ValueError):
            thread_config(value)
    assert thread_config(" run-1 ")["configurable"]["thread_id"] == " run-1 "


async def test_checkpointer_setup_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Saver:
        async def setup(self):
            calls.append("setup")

    @asynccontextmanager
    async def connect(dsn, *, serde):
        calls.append((dsn, serde))
        try:
            yield Saver()
        finally:
            calls.append("closed")

    monkeypatch.setattr("proofpath.graph.checkpointer.AsyncPostgresSaver.from_conn_string", connect)
    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="failed"):
        async with postgres_checkpointer(settings=settings):
            raise RuntimeError("failed")
    assert calls[0][0] == settings.psycopg_database_url
    assert calls[1] == "setup"
    assert calls[-1] == "closed"
