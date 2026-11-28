"""Approval API contracts (Sub-part 2.1.2a): listing, decisions, error mapping."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from proofpath.api import draft_routes, routes
from proofpath.approvals import (
    DraftConflictError,
    DraftDecisionResult,
    DraftNotFoundError,
)
from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
)
from proofpath.execution import DraftExecution, ExecutionOutcome
from proofpath.graph.dependencies import GraphDependenciesError
from proofpath.main import create_app
from proofpath.schemas.actions import ActionDraftRecord, TrackerUpsertPayload

pytestmark = pytest.mark.unit


def _draft_row(
    *,
    status: ActionDraftStatus = ActionDraftStatus.PENDING_APPROVAL,
    opportunity_id: uuid.UUID | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        analysis_run_id=uuid.uuid4(),
        action_type=ActionType.TRACKER_UPSERT,
        status=status,
        payload_json={
            "action_type": "tracker_upsert",
            "opportunity_id": str(opportunity_id or uuid.uuid4()),
            "stage": ApplicationStage.PREPARING.value,
        },
        idempotency_key=f"proofpath:{uuid.uuid4()}:tracker_upsert",
        approved_at=None,
        executed_at=None,
        total_tokens=0,
        estimated_cost_usd=Decimal(0),
        latency_ms=0,
        trace_id=None,
    )


@pytest.fixture
def app_services(monkeypatch):
    app = create_app(settings=Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None)
    app.state.checkpointer = MagicMock()
    session = MagicMock()

    @asynccontextmanager
    async def begin():
        yield session

    app.state.session_factory = MagicMock()
    app.state.session_factory.begin = begin

    drafts: list[SimpleNamespace] = []
    listing = {"rows": drafts, "calls": []}

    class Repo:
        def __init__(self, _session):
            pass

        async def list_drafts(self, *, status=None, analysis_run_id=None, limit=100, offset=0):
            listing["calls"].append(
                {
                    "status": status,
                    "analysis_run_id": analysis_run_id,
                    "limit": limit,
                    "offset": offset,
                }
            )
            rows = drafts
            if status is not None:
                rows = [row for row in rows if row.status is status]
            return rows[offset : offset + limit]

    monkeypatch.setattr(draft_routes, "DraftRepo", Repo)

    async def provide_session():
        yield session

    app.dependency_overrides[routes.session_dependency] = provide_session
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, listing, drafts, app
    finally:
        client.close()


def _execution(outcome: ExecutionOutcome = ExecutionOutcome.EXECUTED) -> DraftExecution:
    return DraftExecution(
        draft_id=uuid.uuid4(),
        action_type=ActionType.TRACKER_UPSERT,
        outcome=outcome,
        approved_at=datetime.now(UTC),
        executed_at=datetime.now(UTC) if outcome is ExecutionOutcome.EXECUTED else None,
        audit_log_id=uuid.uuid4() if outcome is not ExecutionOutcome.FAILED else None,
    )


def _result(
    *,
    decision: str = "approved",
    already_processed: bool = False,
    run_status: AnalysisRunStatus = AnalysisRunStatus.COMPLETED,
    execution: DraftExecution | None = None,
) -> DraftDecisionResult:
    row = _draft_row(
        status=(
            ActionDraftStatus.APPROVED if decision == "approved" else ActionDraftStatus.REJECTED
        )
    )
    row.approved_at = datetime.now(UTC) if decision == "approved" else None
    return DraftDecisionResult(
        draft=ActionDraftRecord(
            id=row.id,
            analysis_run_id=row.analysis_run_id,
            action_type=row.action_type,
            status=row.status,
            payload=TrackerUpsertPayload(
                action_type=ActionType.TRACKER_UPSERT,
                opportunity_id=uuid.uuid4(),
                stage=ApplicationStage.PREPARING,
            ),
            idempotency_key=row.idempotency_key,
            approved_at=row.approved_at,
            executed_at=None,
        ),
        decision=decision,  # type: ignore[arg-type]
        already_processed=already_processed,
        run_status=run_status,
        thread_id="thread-1",
        execution=execution,
    )


def test_list_drafts_defaults_to_all_and_forwards_filters(app_services):
    client, listing, drafts, _ = app_services
    drafts.extend([_draft_row(), _draft_row(status=ActionDraftStatus.APPROVED)])

    everything = client.get("/api/v1/drafts")
    assert everything.status_code == 200
    assert len(everything.json()) == 2

    pending = client.get(
        "/api/v1/drafts",
        params={"status": "pending_approval", "limit": 5, "offset": 0},
    )
    assert pending.status_code == 200
    body = pending.json()
    assert [item["status"] for item in body] == ["pending_approval"]
    assert body[0]["idempotency_key"].startswith("proofpath:")
    assert body[0]["payload"]["action_type"] == "tracker_upsert"
    assert listing["calls"][-1] == {
        "status": ActionDraftStatus.PENDING_APPROVAL,
        "analysis_run_id": None,
        "limit": 5,
        "offset": 0,
    }


def test_list_drafts_rejects_unknown_status_and_bad_paging(app_services):
    client, _, _, _ = app_services
    assert client.get("/api/v1/drafts", params={"status": "done"}).status_code == 422
    assert client.get("/api/v1/drafts", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/drafts", params={"offset": -1}).status_code == 422


def test_approve_returns_the_committed_decision(app_services, monkeypatch):
    client, _, _, _ = app_services
    draft_id = uuid.uuid4()
    calls = []

    async def fake_decide(**kwargs):
        calls.append(kwargs)
        return _result()

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", lambda _settings: object())
    monkeypatch.setattr(draft_routes, "decide_draft", fake_decide)

    response = client.post(f"/api/v1/drafts/{draft_id}/approve")
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "approved"
    assert body["already_processed"] is False
    assert body["run_status"] == "completed"
    assert body["thread_id"] == "thread-1"
    assert body["draft"]["status"] == "approved"
    assert body["execution"] is None
    assert calls[0]["draft_id"] == draft_id
    assert calls[0]["decision"] == "approved"


def test_approve_reports_the_materialized_side_effect(app_services, monkeypatch):
    client, _, _, _ = app_services
    execution = _execution()

    async def fake_decide(**kwargs):
        return _result(execution=execution)

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", lambda _settings: object())
    monkeypatch.setattr(draft_routes, "decide_draft", fake_decide)

    response = client.post(f"/api/v1/drafts/{execution.draft_id}/approve")

    assert response.status_code == 200
    body = response.json()["execution"]
    assert body["outcome"] == "executed"
    assert body["action_type"] == "tracker_upsert"
    assert body["audit_log_id"] == str(execution.audit_log_id)
    assert body["executed_at"] is not None
    assert body["error_code"] is None


def test_rejection_never_reports_an_execution(app_services, monkeypatch):
    client, _, _, _ = app_services

    async def fake_decide(**kwargs):
        return _result(decision="rejected")

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", lambda _settings: object())
    monkeypatch.setattr(draft_routes, "decide_draft", fake_decide)

    response = client.post(f"/api/v1/drafts/{uuid.uuid4()}/reject")

    assert response.status_code == 200
    assert response.json()["execution"] is None


def test_reject_uses_the_rejected_decision(app_services, monkeypatch):
    client, _, _, _ = app_services
    calls = []

    async def fake_decide(**kwargs):
        calls.append(kwargs)
        return _result(decision="rejected", run_status=AnalysisRunStatus.COMPLETED)

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", lambda _settings: object())
    monkeypatch.setattr(draft_routes, "decide_draft", fake_decide)

    response = client.post(f"/api/v1/drafts/{uuid.uuid4()}/reject")
    assert response.status_code == 200
    assert response.json()["decision"] == "rejected"
    assert response.json()["draft"]["status"] == "rejected"
    assert calls[0]["decision"] == "rejected"


def test_repeated_approval_is_reported_as_already_processed(app_services, monkeypatch):
    client, _, _, _ = app_services

    async def fake_decide(**_kwargs):
        return _result(already_processed=True)

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", lambda _settings: object())
    monkeypatch.setattr(draft_routes, "decide_draft", fake_decide)

    body = client.post(f"/api/v1/drafts/{uuid.uuid4()}/approve").json()
    assert body["already_processed"] is True
    assert body["draft"]["status"] == "approved"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DraftNotFoundError("missing"), 404),
        (DraftConflictError("conflict"), 409),
    ],
)
def test_decision_errors_map_to_problem_responses(app_services, monkeypatch, error, expected):
    client, _, _, _ = app_services

    async def fake_decide(**_kwargs):
        raise error

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", lambda _settings: object())
    monkeypatch.setattr(draft_routes, "decide_draft", fake_decide)

    response = client.post(f"/api/v1/drafts/{uuid.uuid4()}/approve")
    assert response.status_code == expected
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "conflict" not in response.text


def test_unconfigured_model_blocks_any_decision(app_services, monkeypatch):
    client, _, _, _ = app_services
    called = AsyncMock()

    def unavailable(_settings):
        raise GraphDependenciesError("Analysis model is not configured.")

    monkeypatch.setattr(draft_routes, "create_graph_dependencies", unavailable)
    monkeypatch.setattr(draft_routes, "decide_draft", called)

    for path in ("approve", "reject"):
        response = client.post(f"/api/v1/drafts/{uuid.uuid4()}/{path}")
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/problem+json")
    called.assert_not_awaited()


def test_invalid_draft_id_is_a_validation_error(app_services):
    client, _, _, _ = app_services
    assert client.post("/api/v1/drafts/not-a-uuid/approve").status_code == 422
