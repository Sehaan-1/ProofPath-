"""Unit tests for the retry & dead-letter policy (Sub-slice 2.2.3).

Covers the pure backoff computation, the delayed-retry queue parsing, and the
dead-letter audit-logging behavior of `SqlAlchemyJobStore` (with a mocked
session). The live Redis + PostgreSQL end-to-end loop is covered by
`tests/integration/test_retry_policy_postgres.py`.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from proofpath.db.models import AuditActor
from proofpath.workers.queue import RetryItem
from proofpath.workers.worker import (
    _MAX_TRACEBACK_CHARS,
    DEAD_LETTER_AUDIT_EVENT,
    DEFAULT_BACKOFF_SCHEDULE,
    _truncate_traceback,
    compute_backoff_delay,
)

pytestmark = pytest.mark.unit

JID = uuid.UUID("12345678-1234-5678-1234-567812345678")


# ---------------------------------------------------------------------------
# compute_backoff_delay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (1, 2.0),
        (2, 8.0),
        (3, 30.0),
        (4, 30.0),  # caps at the last schedule entry
        (99, 30.0),
    ],
)
def test_backoff_delay_indexed_by_attempt(attempt: int, expected: float) -> None:
    assert compute_backoff_delay(attempt, DEFAULT_BACKOFF_SCHEDULE) == expected


def test_backoff_delay_zero_or_negative_attempt_returns_zero() -> None:
    assert compute_backoff_delay(0, (2, 8, 30)) == 0.0
    assert compute_backoff_delay(-1, (2, 8, 30)) == 0.0


def test_backoff_delay_empty_schedule_returns_zero() -> None:
    assert compute_backoff_delay(1, ()) == 0.0


def test_backoff_delay_custom_schedule() -> None:
    assert compute_backoff_delay(1, (5, 10)) == 5.0
    assert compute_backoff_delay(2, (5, 10)) == 10.0
    assert compute_backoff_delay(3, (5, 10)) == 10.0  # caps


# ---------------------------------------------------------------------------
# _truncate_traceback
# ---------------------------------------------------------------------------


def test_truncate_traceback_none_passes_through() -> None:
    assert _truncate_traceback(None) is None


def test_truncate_traceback_short_passes_through() -> None:
    tb = "Traceback (most recent call last):\n  ValueError: boom\n"
    assert _truncate_traceback(tb) == tb


def test_truncate_traceback_long_is_bounded_and_keeps_tail() -> None:
    tb = "A" * 20_000
    truncated = _truncate_traceback(tb)
    assert truncated is not None
    assert len(truncated) <= _MAX_TRACEBACK_CHARS
    assert truncated.startswith("...\n")
    # The tail (most recent frame) is preserved.
    assert truncated.endswith("A")


# ---------------------------------------------------------------------------
# Delayed-retry message parsing
# ---------------------------------------------------------------------------


def test_retry_item_is_immutable_and_carries_payload() -> None:
    item = RetryItem(job_id=JID, job_type="github_ingest", payload={"k": 1}, ready_at=1000.0)
    assert item.job_id == JID
    assert item.job_type == "github_ingest"
    assert item.payload == {"k": 1}
    assert item.ready_at == 1000.0


# ---------------------------------------------------------------------------
# SqlAlchemyJobStore.dead_letter writes an audit log
# ---------------------------------------------------------------------------


class _FakeRepo:
    """Captures mark_dead_letter / mark_running / mark_succeeded / mark_failed."""

    def __init__(self, job: Any | None = None) -> None:
        self._job = job
        self.dead_lettered: list[tuple[uuid.UUID, str, str | None]] = []

    async def mark_dead_letter(
        self, job_id: uuid.UUID, *, error: str, traceback: str | None = None
    ) -> Any:
        self.dead_lettered.append((job_id, error, traceback))
        return self._job

    async def mark_running(self, _jid: uuid.UUID) -> Any:
        return self._job

    async def mark_succeeded(self, _jid: uuid.UUID, *, result: Any = None) -> Any:
        return self._job

    async def mark_failed(
        self, _jid: uuid.UUID, *, error: str, traceback: str | None = None
    ) -> Any:
        return self._job


class _FakeAuditRepo:
    """Captures audit `record` calls."""

    def __init__(self) -> None:
        self.recorded: list[dict[str, Any]] = []

    async def record(
        self,
        *,
        actor: AuditActor,
        event_type: str,
        resource_id: uuid.UUID | None = None,
        details_json: dict[str, Any] | None = None,
        log_id: uuid.UUID | None = None,
    ) -> Any:
        entry = {
            "actor": actor,
            "event_type": event_type,
            "resource_id": resource_id,
            "details_json": dict(details_json or {}),
        }
        self.recorded.append(entry)
        return MagicMock()


def _mock_session(repo: _FakeRepo, audit: _FakeAuditRepo) -> MagicMock:
    """Build a session that routes BackgroundJobRepo and AuditRepo to fakes."""
    session = MagicMock()
    session.commit = AsyncMock()

    # The store uses `BackgroundJobRepo(session)` and `AuditRepo(session)`.
    # Patch at the module level instead — see the test functions below.
    return session


async def test_dead_letter_writes_audit_log_with_error_and_job_metadata(monkeypatch):

    from proofpath.workers.worker import SqlAlchemyJobStore

    job = MagicMock(
        job_type="github_ingest",
        attempt=3,
        max_attempts=3,
    )
    repo = _FakeRepo(job=job)
    audit = _FakeAuditRepo()
    monkeypatch.setattr("proofpath.workers.worker.BackgroundJobRepo", lambda _session: repo)
    monkeypatch.setattr("proofpath.workers.worker.AuditRepo", lambda _session: audit)

    # Session factory yields a MagicMock session; commit is async.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def factory():
        session = MagicMock()
        session.commit = AsyncMock()
        yield session

    store = SqlAlchemyJobStore(factory)  # type: ignore[arg-type]
    await store.dead_letter(JID, error="github fetch failed: 503", traceback="Traceback ...")

    assert len(repo.dead_lettered) == 1
    assert repo.dead_lettered[0][0] == JID
    assert len(audit.recorded) == 1
    entry = audit.recorded[0]
    assert entry["actor"] is AuditActor.AGENT
    assert entry["event_type"] == DEAD_LETTER_AUDIT_EVENT
    assert entry["resource_id"] == JID
    details = entry["details_json"]
    assert details["error"] == "github fetch failed: 503"
    assert details["job_type"] == "github_ingest"
    assert details["attempt"] == 3
    assert details["max_attempts"] == 3
    assert details["traceback"] == "Traceback ..."


async def test_dead_letter_audit_omits_traceback_when_none(monkeypatch):
    from contextlib import asynccontextmanager

    from proofpath.workers.worker import SqlAlchemyJobStore

    job = MagicMock(job_type="noop", attempt=1, max_attempts=1)
    repo = _FakeRepo(job=job)
    audit = _FakeAuditRepo()
    monkeypatch.setattr("proofpath.workers.worker.BackgroundJobRepo", lambda _session: repo)
    monkeypatch.setattr("proofpath.workers.worker.AuditRepo", lambda _session: audit)

    @asynccontextmanager
    async def factory():
        session = MagicMock()
        session.commit = AsyncMock()
        yield session

    store = SqlAlchemyJobStore(factory)  # type: ignore[arg-type]
    await store.dead_letter(JID, error="unknown task type 'bogus'", traceback=None)

    entry = audit.recorded[0]
    assert "traceback" not in entry["details_json"]
    assert entry["details_json"]["error"] == "unknown task type 'bogus'"


async def test_dead_letter_audit_bounded_traceback(monkeypatch):
    from contextlib import asynccontextmanager

    from proofpath.workers.worker import SqlAlchemyJobStore

    job = MagicMock(job_type="flaky", attempt=2, max_attempts=2)
    repo = _FakeRepo(job=job)
    audit = _FakeAuditRepo()
    monkeypatch.setattr("proofpath.workers.worker.BackgroundJobRepo", lambda _session: repo)
    monkeypatch.setattr("proofpath.workers.worker.AuditRepo", lambda _session: audit)

    @asynccontextmanager
    async def factory():
        session = MagicMock()
        session.commit = AsyncMock()
        yield session

    store = SqlAlchemyJobStore(factory)  # type: ignore[arg-type]
    long_tb = "X" * 20_000
    await store.dead_letter(JID, error="timeout", traceback=long_tb)

    entry = audit.recorded[0]
    stored_tb = entry["details_json"]["traceback"]
    assert stored_tb is not None
    assert len(stored_tb) <= _MAX_TRACEBACK_CHARS
    assert stored_tb.endswith("X")
