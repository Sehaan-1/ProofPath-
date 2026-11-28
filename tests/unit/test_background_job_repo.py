"""Unit tests for the `BackgroundJobRepo` state machine and record mapping."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from proofpath.db.models import BackgroundJob, JobStatus
from proofpath.db.repositories.background_job import BackgroundJobRepo, background_job_record

JID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _mock_session_with_scalar(scalar_value: Any) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_value
    # Use a MagicMock (not AsyncMock) so `session.add(...)` stays synchronous —
    # AsyncMock would turn it into an un-awaited coroutine. `execute`/`flush`
    # remain async because the real AsyncSession methods are.
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()
    return session


@pytest.mark.unit
async def test_create_rejects_non_positive_max_attempts() -> None:
    repo = BackgroundJobRepo(MagicMock())
    with pytest.raises(ValueError, match="max_attempts"):
        await repo.create(job_type="noop", payload={}, max_attempts=0)


@pytest.mark.unit
async def test_create_persists_queued_job_with_defaults() -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    repo = BackgroundJobRepo(session)

    job = await repo.create(job_type="github_ingest", payload={"repo": "x"}, max_attempts=5)

    assert job.job_type == "github_ingest"
    assert job.status is JobStatus.QUEUED
    assert job.attempt == 0
    assert job.max_attempts == 5
    assert job.payload_json == {"repo": "x"}
    session.add.assert_called_once_with(job)
    session.flush.assert_awaited_once()


@pytest.mark.unit
async def test_mark_running_returns_none_when_job_missing() -> None:
    session = _mock_session_with_scalar(None)
    repo = BackgroundJobRepo(session)
    assert await repo.mark_running(uuid.uuid4()) is None


@pytest.mark.unit
async def test_mark_running_claims_queued_job_and_bumps_attempt() -> None:
    job = BackgroundJob(
        id=JID,
        job_type="noop",
        payload_json={},
        status=JobStatus.QUEUED,
        attempt=0,
        max_attempts=3,
    )
    session = _mock_session_with_scalar(job)
    repo = BackgroundJobRepo(session)

    claimed = await repo.mark_running(JID)

    assert claimed is job
    assert claimed.status is JobStatus.RUNNING
    assert claimed.attempt == 1
    assert claimed.started_at is not None
    session.flush.assert_awaited_once()


@pytest.mark.unit
async def test_mark_running_skips_already_terminal_job() -> None:
    job = BackgroundJob(
        id=JID,
        job_type="noop",
        payload_json={},
        status=JobStatus.SUCCEEDED,
        attempt=1,
        max_attempts=3,
    )
    session = _mock_session_with_scalar(job)
    repo = BackgroundJobRepo(session)

    assert await repo.mark_running(JID) is None
    assert job.status is JobStatus.SUCCEEDED  # unchanged
    session.flush.assert_not_awaited()


@pytest.mark.unit
async def test_mark_succeeded_sets_result_and_finished_at() -> None:
    job = BackgroundJob(
        id=JID,
        job_type="echo",
        payload_json={},
        status=JobStatus.RUNNING,
        attempt=1,
        max_attempts=3,
    )
    session = _mock_session_with_scalar(job)
    repo = BackgroundJobRepo(session)

    updated = await repo.mark_succeeded(JID, result={"ok": 1})

    assert updated is job
    assert updated.status is JobStatus.SUCCEEDED
    assert updated.result_json == {"ok": 1}
    assert updated.finished_at is not None


@pytest.mark.unit
async def test_mark_failed_and_dead_letter_record_errors() -> None:
    job = BackgroundJob(
        id=JID,
        job_type="noop",
        payload_json={},
        status=JobStatus.RUNNING,
        attempt=2,
        max_attempts=3,
    )
    session = _mock_session_with_scalar(job)
    repo = BackgroundJobRepo(session)

    failed = await repo.mark_failed(JID, error="boom", traceback="tb")
    assert failed.status is JobStatus.FAILED
    assert failed.last_error == "boom"
    assert failed.error_traceback == "tb"
    assert failed.finished_at is not None

    # Same row object reused; transition to dead_letter from failed.
    dead = await repo.mark_dead_letter(JID, error="exhausted", traceback="tb2")
    assert dead.status is JobStatus.DEAD_LETTER
    assert dead.last_error == "exhausted"


@pytest.mark.unit
def test_background_job_record_maps_all_fields() -> None:
    now = datetime(2026, 11, 28, 9, 0, tzinfo=UTC)
    started = datetime(2026, 11, 28, 9, 1, tzinfo=UTC)
    finished = datetime(2026, 11, 28, 9, 2, tzinfo=UTC)
    job = BackgroundJob(
        id=JID,
        job_type="github_ingest",
        payload_json={"repo": "sehaan-1/proofpath"},
        status=JobStatus.RUNNING,
        attempt=2,
        max_attempts=3,
        last_error="timeout",
        error_traceback="Traceback ...",
        result_json={"ingested": 12},
        started_at=started,
        finished_at=finished,
        created_at=now,
        updated_at=now,
    )

    record = background_job_record(job)

    assert record.id == JID
    assert record.job_type == "github_ingest"
    assert record.payload == {"repo": "sehaan-1/proofpath"}
    assert record.status is JobStatus.RUNNING
    assert record.attempt == 2
    assert record.max_attempts == 3
    assert record.last_error == "timeout"
    assert record.error_traceback == "Traceback ..."
    assert record.result == {"ingested": 12}
    assert record.started_at == started
    assert record.finished_at == finished
    assert record.created_at == now
    assert record.updated_at == now


@pytest.mark.unit
def test_background_job_record_nulls_result_when_absent() -> None:
    now = datetime(2026, 11, 28, 9, 0, tzinfo=UTC)
    job = BackgroundJob(
        id=JID,
        job_type="noop",
        payload_json={},
        status=JobStatus.QUEUED,
        attempt=0,
        max_attempts=3,
        created_at=now,
        updated_at=now,
    )
    record = background_job_record(job)
    assert record.result is None
    assert record.last_error is None
    assert record.started_at is None
    assert record.finished_at is None
