"""Synchronous and async analysis lifecycle, reporting, and failure semantics."""

from contextlib import asynccontextmanager
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Interrupt

from proofpath import jobs
from proofpath.api import analysis_routes, routes
from proofpath.config import Settings
from proofpath.db.models import (
    AnalysisRunStatus,
    JobStatus,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.graph.state import ApprovalDraftSummary, ApprovalRequest
from proofpath.main import create_app
from proofpath.schemas.analysis import ClaimVerdictSchema

pytestmark = pytest.mark.unit


@pytest.fixture
def app_services(monkeypatch):
    app = create_app(settings=Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None)
    app.state.checkpointer = MagicMock()
    app.state.checkpointer.aget_tuple = AsyncMock(return_value=None)
    session = MagicMock()

    @asynccontextmanager
    async def begin():
        yield session

    @asynccontextmanager
    async def current_session():
        yield session

    app.state.session_factory = MagicMock(side_effect=current_session)
    app.state.session_factory.begin = begin
    opp_id = uuid4()
    req_id = uuid4()
    req = SimpleNamespace(
        id=req_id,
        opportunity_id=opp_id,
        entity="Python",
        requirement_text="Python",
        category=RequirementCategory.LANGUAGE,
        priority=RequirementPriority.MUST_HAVE,
    )
    opp = SimpleNamespace(
        id=opp_id, company="Co", role_title="Engineer", source_url=None, requirements=[req]
    )
    opp_repo = MagicMock()
    opp_repo.get_by_id = AsyncMock(return_value=opp)
    monkeypatch.setattr(analysis_routes, "OpportunityRepo", lambda _session: opp_repo)
    runs = {}

    class Repo:
        def __init__(self, _session):
            pass

        async def create_run(self, *, opportunity_id, thread_id, run_id):
            runs[run_id] = SimpleNamespace(
                id=run_id,
                opportunity_id=opportunity_id,
                thread_id=thread_id,
                status=AnalysisRunStatus.RUNNING,
                claim_verdicts=[],
                action_drafts=[],
                total_tokens=0,
                estimated_cost_usd=Decimal(0),
                latency_ms=0,
                trace_id=None,
            )
            return runs[run_id]

        async def update_run_metrics(self, run_id, *, status, latency_ms):
            runs[run_id].status = status
            runs[run_id].latency_ms = latency_ms

        async def get_run_by_id(self, run_id, **kwargs):
            return runs.get(run_id)

        async def upsert_verdicts(self, run_id, verdicts):
            runs[run_id].claim_verdicts = [
                SimpleNamespace(
                    **{
                        name: getattr(v, name)
                        for name in (
                            "requirement_id",
                            "verdict",
                            "claim_summary",
                            "cited_chunk_ids",
                            "policy_downgraded",
                            "gap_notes",
                        )
                    }
                )
                for v in verdicts
            ]

    monkeypatch.setattr(analysis_routes, "AnalysisRepo", Repo)
    # `execute_analysis_graph` (in `proofpath.jobs`) resolves AnalysisRepo and
    # build_workflow from the jobs module, so patch there too.
    monkeypatch.setattr(jobs, "AnalysisRepo", Repo)

    class FakeDependencies:
        parser = embedder = comparator = sprint_planner = object()

    monkeypatch.setattr(
        analysis_routes, "create_graph_dependencies", lambda _settings: FakeDependencies()
    )
    approval_interrupt = Interrupt(
        value=ApprovalRequest(
            analysis_run_id=uuid4(),
            drafts=[
                ApprovalDraftSummary(
                    draft_id=uuid4(), action_type="tracker_upsert", summary="Update tracker"
                )
            ],
        ).model_dump(mode="json")
    )
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "policy_verified": True,
            "errors": [],
            "draft_actions": [object()],
            "__interrupt__": (approval_interrupt,),
            "verdicts": [
                ClaimVerdictSchema(
                    requirement_id=req_id,
                    verdict=VerdictType.SUPPORTED,
                    claim_summary="Grounded",
                    cited_chunk_ids=[uuid4()],
                )
            ],
        }
    )
    monkeypatch.setattr(jobs, "build_workflow", lambda **_kwargs: graph)

    # Async path: a job queue on app state and a mocked enqueue_job.
    app.state.job_queue = MagicMock()
    fake_job = SimpleNamespace(id=uuid4(), job_type="analysis_run", status=JobStatus.QUEUED)
    enqueue_mock = AsyncMock(return_value=fake_job)
    monkeypatch.setattr(analysis_routes, "enqueue_job", enqueue_mock)

    async def provide_session():
        yield session

    app.dependency_overrides[routes.session_dependency] = provide_session
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, graph, runs, opp_id, req_id, opp_repo
    finally:
        client.close()


def test_analysis_run_persists_only_verified_verdicts_and_report(app_services):
    client, graph, runs, opp_id, req_id, _ = app_services
    response = client.post("/api/v1/analyses/run", json={"opportunity_id": str(opp_id)})
    assert response.status_code == 201
    report = response.json()
    assert report["status"] == "awaiting_approval"
    assert report["verdicts"][0]["requirement_id"] == str(req_id)
    run_id = next(iter(runs))
    assert client.get(f"/api/v1/analyses/{run_id}").json()["verdicts"] == report["verdicts"]
    graph.ainvoke.assert_awaited_once()
    assert client.get(f"/api/v1/analyses/{uuid4()}").status_code == 404


def test_pause_outside_approval_gate_is_failed_not_approvable(app_services):
    client, graph, runs, opp_id, _, _ = app_services
    graph.ainvoke.return_value = {
        "policy_verified": True,
        "errors": [],
        "draft_actions": [object()],
        "__interrupt__": (Interrupt(value="continue?"),),
        "verdicts": [],
    }
    response = client.post("/api/v1/analyses/run", json={"opportunity_id": str(opp_id)})
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert next(iter(runs.values())).status == AnalysisRunStatus.FAILED


def test_graph_failure_is_failed_and_never_persists_unverified_verdicts(app_services):
    client, graph, runs, opp_id, _, _ = app_services
    graph.ainvoke.return_value = {
        "policy_verified": False,
        "errors": [
            {
                "node": "policy",
                "code": "node_failed",
                "message": "Policy failed.",
                "retryable": False,
            }
        ],
        "verdicts": [object()],
    }
    response = client.post("/api/v1/analyses/run", json={"opportunity_id": str(opp_id)})
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert response.json()["verdicts"] == []
    assert next(iter(runs.values())).status == AnalysisRunStatus.FAILED


def test_async_analysis_enqueues_job_and_returns_202(app_services):
    client, _, runs, opp_id, _, _ = app_services
    response = client.post(
        "/api/v1/analyses/run", json={"opportunity_id": str(opp_id), "async_run": True}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_type"] == "analysis_run"
    assert body["status"] == "queued"
    assert "job_id" in body
    # The async path creates a background job, not an analysis_run row.
    assert not runs


def test_query_async_alias_also_enqueues(app_services):
    client, _, _, opp_id, _, _ = app_services
    response = client.post(
        "/api/v1/analyses/run?async=true",
        json={"opportunity_id": str(opp_id)},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_invalid_calendar_rejected_before_creation(app_services):
    client, _, runs, opp_id, _, _ = app_services
    assert (
        client.post(
            "/api/v1/analyses/run",
            json={"opportunity_id": str(opp_id), "calendar_start": "2026-01-01T10:00:00Z"},
        ).status_code
        == 422
    )
    assert not runs


def test_exception_marks_run_failed_without_leaking_error(app_services):
    client, graph, runs, opp_id, _, _ = app_services
    graph.ainvoke.side_effect = RuntimeError("private secret")
    response = client.post("/api/v1/analyses/run", json={"opportunity_id": str(opp_id)})
    assert response.status_code == 503
    assert "private secret" not in response.text
    assert next(iter(runs.values())).status == AnalysisRunStatus.FAILED
