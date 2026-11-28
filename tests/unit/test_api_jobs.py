"""Async ingestion endpoint and job status endpoint contracts (Sub-slice 2.2.2)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from proofpath.api import jobs_routes, routes
from proofpath.config import Settings
from proofpath.db.models import JobStatus
from proofpath.main import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    session = MagicMock()
    app = create_app(settings=Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None, github_token=None)
    app.state.job_queue = MagicMock()
    app.state.session_factory = MagicMock()

    async def provide_session():
        yield session

    app.dependency_overrides[routes.session_dependency] = provide_session
    test_client = TestClient(app, raise_server_exceptions=False)
    try:
        yield test_client, session
    finally:
        test_client.close()


# ---------------------------------------------------------------------------
# POST /evidence/ingest/github?async=true
# ---------------------------------------------------------------------------


def test_async_github_ingest_returns_202_with_job_id(client, monkeypatch):
    test_client, _ = client
    job = SimpleNamespace(id=uuid.uuid4(), job_type="github_ingest", status=JobStatus.QUEUED)
    enqueue = AsyncMock(return_value=job)
    monkeypatch.setattr(routes, "enqueue_job", enqueue)

    response = test_client.post(
        "/api/v1/evidence/ingest/github?async=true",
        json={"owner": "example", "name": "repo"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == str(job.id)
    assert body["job_type"] == "github_ingest"
    assert body["status"] == "queued"
    enqueue.assert_awaited_once()
    assert enqueue.await_args.kwargs["job_type"] == "github_ingest"


def test_async_github_ingest_rejects_invalid_repo_before_enqueue(client, monkeypatch):
    test_client, _ = client
    enqueue = AsyncMock()
    monkeypatch.setattr(routes, "enqueue_job", enqueue)
    response = test_client.post(
        "/api/v1/evidence/ingest/github?async=true",
        json={"owner": "..", "name": "repo"},
    )
    assert response.status_code == 422
    enqueue.assert_not_awaited()


def test_async_github_ingest_503_when_queue_unavailable(client, monkeypatch):
    test_client, _ = client
    enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    monkeypatch.setattr(routes, "enqueue_job", enqueue)
    response = test_client.post(
        "/api/v1/evidence/ingest/github?async=true",
        json={"owner": "example", "name": "repo"},
    )
    assert response.status_code == 503
    enqueue.assert_awaited_once()


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}
# ---------------------------------------------------------------------------


def _job_row(*, status: JobStatus = JobStatus.SUCCEEDED) -> SimpleNamespace:
    now = datetime(2026, 11, 28, 9, 0, tzinfo=UTC)
    return SimpleNamespace(
        id=uuid.uuid4(),
        job_type="github_ingest",
        status=status,
        attempt=1,
        max_attempts=3,
        last_error=None,
        result_json={"source_id": str(uuid.uuid4()), "chunk_count": 4},
        started_at=now,
        finished_at=now,
        created_at=now,
        updated_at=now,
    )


def test_get_job_returns_trimmed_status(client, monkeypatch):
    test_client, _ = client
    row = _job_row()
    repo = MagicMock()
    repo.get = AsyncMock(return_value=row)
    monkeypatch.setattr(jobs_routes, "BackgroundJobRepo", lambda _session: repo)

    response = test_client.get(f"/api/v1/jobs/{row.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(row.id)
    assert body["status"] == "succeeded"
    assert body["result"]["chunk_count"] == 4
    # The raw payload and traceback are not exposed on the public status view.
    assert "payload" not in body
    assert "error_traceback" not in body


def test_get_job_404_when_unknown(client, monkeypatch):
    test_client, _ = client
    repo = MagicMock()
    repo.get = AsyncMock(return_value=None)
    monkeypatch.setattr(jobs_routes, "BackgroundJobRepo", lambda _session: repo)
    response = test_client.get(f"/api/v1/jobs/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_job_surfaces_failed_status_and_error(client, monkeypatch):
    test_client, _ = client
    row = _job_row(status=JobStatus.DEAD_LETTER)
    row.last_error = "github fetch failed: not found"
    row.result_json = None
    repo = MagicMock()
    repo.get = AsyncMock(return_value=row)
    monkeypatch.setattr(jobs_routes, "BackgroundJobRepo", lambda _session: repo)

    response = test_client.get(f"/api/v1/jobs/{row.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "dead_letter"
    assert body["last_error"] == "github fetch failed: not found"
    assert body["result"] is None
