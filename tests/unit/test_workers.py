"""Unit tests for the Redis job queue, task registry, and worker state machine.

These use in-memory fakes (no Redis, no PostgreSQL) so they exercise the worker's
branching logic — success, transient retry, exhaustion, dead-letter, skip —
deterministically and fast. The real Redis + PostgreSQL loop is covered by
`tests/integration/test_background_jobs_postgres.py`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from proofpath.db.models import JOB_TERMINAL_STATUSES, JobStatus
from proofpath.workers.queue import (
    DEFAULT_GROUP,
    DEFAULT_STREAM,
    JobMessage,
    JobQueue,
    _parse_message,
)
from proofpath.workers.tasks import (
    ECHO_TASK,
    NOOP_TASK,
    JobContext,
    JobResult,
    TaskRegistry,
    default_registry,
    echo_handler,
    noop_handler,
    validate_job_type,
)
from proofpath.workers.worker import ClaimedJob, ProcessOutcome, Worker

JID = uuid.UUID("12345678-1234-5678-1234-567812345678")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeJob:
    job_id: uuid.UUID
    job_type: str
    payload: dict[str, Any]
    status: JobStatus
    attempt: int = 0
    max_attempts: int = 3
    last_error: str | None = None
    traceback: str | None = None
    result: dict[str, Any] | None = None


class FakeJobStore:
    """In-memory `JobStore` implementing the same state machine as the repo."""

    def __init__(self, jobs: dict[uuid.UUID, FakeJob] | None = None) -> None:
        self.jobs: dict[uuid.UUID, FakeJob] = dict(jobs or {})
        self.claimed: list[uuid.UUID] = []
        self.succeeded: list[tuple[uuid.UUID, dict[str, Any] | None]] = []
        self.failed: list[tuple[uuid.UUID, str, str | None]] = []
        self.dead_lettered: list[tuple[uuid.UUID, str, str | None]] = []

    async def claim(self, job_id: uuid.UUID) -> ClaimedJob | None:
        job = self.jobs.get(job_id)
        if job is None or job.status in JOB_TERMINAL_STATUSES:
            return None
        job.status = JobStatus.RUNNING
        job.attempt += 1
        self.claimed.append(job_id)
        return ClaimedJob(
            job_id=job.job_id,
            job_type=job.job_type,
            payload=dict(job.payload),
            attempt=job.attempt,
            max_attempts=job.max_attempts,
        )

    async def succeed(self, job_id: uuid.UUID, *, result: dict[str, Any] | None = None) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.SUCCEEDED
        job.result = result
        self.succeeded.append((job_id, result))

    async def fail(self, job_id: uuid.UUID, *, error: str, traceback: str | None = None) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.FAILED
        job.last_error = error
        job.traceback = traceback
        self.failed.append((job_id, error, traceback))

    async def dead_letter(
        self, job_id: uuid.UUID, *, error: str, traceback: str | None = None
    ) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.DEAD_LETTER
        job.last_error = error
        job.traceback = traceback
        self.dead_lettered.append((job_id, error, traceback))


@dataclass
class FakeQueue:
    messages: list[JobMessage] = field(default_factory=list)
    reclaim_messages: list[JobMessage] = field(default_factory=list)
    acked: list[str] = field(default_factory=list)
    enqueued: list[tuple[uuid.UUID, str, dict[str, Any]]] = field(default_factory=list)
    scheduled_retries: list[tuple[uuid.UUID, str, dict[str, Any], float]] = field(
        default_factory=list
    )
    ready_retries: list[JobMessage] = field(default_factory=list)
    drained: bool = False
    consumer: str = "fake-worker"
    _ensured_group: bool = False

    async def ensure_group(self) -> None:
        self._ensured_group = True

    async def dequeue(self, *, count: int = 1, block_ms: int = 0) -> list[JobMessage]:
        out = self.messages[:count]
        self.messages = self.messages[count:]
        return out

    async def ack(self, *message_ids: str) -> int:
        self.acked.extend(message_ids)
        return len(message_ids)

    async def enqueue(self, job_id: uuid.UUID, *, job_type: str, payload: dict[str, Any]) -> str:
        mid = f"re-{len(self.enqueued)}"
        self.enqueued.append((job_id, job_type, payload))
        return mid

    async def schedule_retry(
        self,
        job_id: uuid.UUID,
        *,
        job_type: str,
        payload: dict[str, Any],
        delay_seconds: float,
    ) -> float:
        self.scheduled_retries.append((job_id, job_type, payload, delay_seconds))
        return delay_seconds

    async def drain_ready_retries(self, *, now: float | None = None) -> list:
        self.drained = True
        out = list(self.ready_retries)
        self.ready_retries = []
        return out

    async def reclaim(self, *, min_idle_ms: int = 60_000, count: int = 10) -> list[JobMessage]:
        out = list(self.reclaim_messages)
        self.reclaim_messages = []
        return out


def _msg(job_id: uuid.UUID | None, job_type: str, *, message_id: str = "m1") -> JobMessage:
    return JobMessage(message_id=message_id, job_id=job_id, job_type=job_type, payload={})


async def _always_fail(_ctx: JobContext) -> JobResult:
    return JobResult(ok=False, error="boom", transient=True)


async def _raises(_ctx: JobContext) -> JobResult:
    raise ValueError("nope")


# ---------------------------------------------------------------------------
# Task registry & handlers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_job_type_accepts_snake_case_rejects_invalid() -> None:
    assert validate_job_type("github_ingest") == "github_ingest"
    for bad in ("Bad", "", "1num", "x" * 65, "has-dash", "UPPER"):
        with pytest.raises(ValueError, match="job_type"):
            validate_job_type(bad)


@pytest.mark.unit
def test_registry_register_get_contains_and_known_types() -> None:
    registry = TaskRegistry()

    async def handler(_ctx: JobContext) -> JobResult:
        return JobResult(ok=True)

    registry.register("my_task", handler)
    assert "my_task" in registry
    assert registry.get("my_task") is handler
    assert registry.known_types() == frozenset({"my_task"})
    assert registry.get("missing") is None


@pytest.mark.unit
def test_registry_rejects_duplicate_and_invalid_bindings() -> None:
    registry = TaskRegistry()
    registry.register("my_task", _always_fail)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("my_task", _always_fail)
    with pytest.raises(ValueError, match="job_type"):
        registry.register("BadName", _always_fail)


@pytest.mark.unit
def test_default_registry_has_noop_and_echo() -> None:
    registry = default_registry()
    assert {NOOP_TASK, ECHO_TASK}.issubset(registry.known_types())


@pytest.mark.unit
async def test_noop_and_echo_handler_results() -> None:
    noop_ctx = JobContext(job_id=uuid.uuid4(), job_type=NOOP_TASK, payload={}, attempt=1)
    assert (await noop_handler(noop_ctx)) == JobResult(ok=True)

    echo_ctx = JobContext(job_id=uuid.uuid4(), job_type=ECHO_TASK, payload={"k": 1}, attempt=1)
    result = await echo_handler(echo_ctx)
    assert result.ok is True
    assert result.result == {"job_type": ECHO_TASK, "payload": {"k": 1}}


@pytest.mark.unit
def test_job_result_defaults_to_transient_retry() -> None:
    result = JobResult(ok=False, error="x")
    assert result.transient is True
    assert result.traceback is None
    assert result.result is None


# ---------------------------------------------------------------------------
# Stream message parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_message_decodes_valid_entry() -> None:
    msg = _parse_message(
        "1-0",
        {"job_id": str(JID), "job_type": "github_ingest", "payload": '{"repo": "x"}'},
    )
    assert msg.message_id == "1-0"
    assert msg.job_id == JID
    assert msg.job_type == "github_ingest"
    assert msg.payload == {"repo": "x"}


@pytest.mark.unit
def test_parse_message_missing_job_id_is_malformed() -> None:
    msg = _parse_message("1-0", {"job_type": "noop"})
    assert msg.job_id is None
    assert msg.job_type == ""
    assert msg.payload == {}


@pytest.mark.unit
def test_parse_message_bad_uuid_is_malformed() -> None:
    msg = _parse_message("1-0", {"job_id": "not-a-uuid", "job_type": "noop"})
    assert msg.job_id is None


@pytest.mark.unit
def test_parse_message_bad_json_payload_defaults_to_empty() -> None:
    msg = _parse_message("1-0", {"job_id": str(JID), "job_type": "noop", "payload": "{not json"})
    assert msg.job_id == JID
    assert msg.payload == {}


@pytest.mark.unit
def test_job_queue_from_url_sets_defaults() -> None:
    queue = JobQueue.from_url("redis://localhost:6379/0")
    assert queue.stream == DEFAULT_STREAM
    assert queue.group == DEFAULT_GROUP
    assert queue.consumer == "worker-1"


# ---------------------------------------------------------------------------
# Worker state machine
# ---------------------------------------------------------------------------


def _worker(store: FakeJobStore, queue: FakeQueue) -> Worker:
    return Worker(queue=queue, store=store, registry=default_registry())


@pytest.mark.unit
async def test_successful_handler_marks_succeeded_and_acks() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "echo", {"k": "v"}, JobStatus.QUEUED)})
    queue = FakeQueue(messages=[_msg(JID, "echo")])
    worker = _worker(store, queue)

    outcome = await worker.process_once()

    assert outcome.outcome is ProcessOutcome.PROCESSED
    assert outcome.succeeded == 1
    assert store.jobs[JID].status is JobStatus.SUCCEEDED
    assert store.jobs[JID].result == {"job_type": "echo", "payload": {"k": "v"}}
    assert queue.acked == ["m1"]
    assert queue.enqueued == []
    assert store.claimed == [JID]


@pytest.mark.unit
async def test_unknown_task_type_goes_straight_to_dead_letter() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "bogus", {}, JobStatus.QUEUED)})
    queue = FakeQueue(messages=[_msg(JID, "bogus")])
    worker = _worker(store, queue)

    outcome = await worker.process_once()

    assert outcome.dead_lettered == 1
    assert store.jobs[JID].status is JobStatus.DEAD_LETTER
    assert queue.acked == ["m1"]
    assert queue.enqueued == []  # permanent error: no re-enqueue
    assert "unknown task type" in (store.jobs[JID].last_error or "")


@pytest.mark.unit
async def test_transient_failure_schedules_delayed_retry_then_dead_letter() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "flaky", {}, JobStatus.QUEUED, max_attempts=2)})
    registry = TaskRegistry()
    registry.register("flaky", _always_fail)
    worker = Worker(queue=FakeQueue(), store=store, registry=registry, backoff_schedule=(2, 8, 30))

    # First delivery: attempt 1 (< max 2) -> failed + delayed retry scheduled.
    worker._queue.messages = [_msg(JID, "flaky", message_id="m1")]  # type: ignore[attr-defined]
    out1 = await worker.process_once()
    assert out1.failed == 1
    assert store.jobs[JID].status is JobStatus.FAILED
    assert store.jobs[JID].attempt == 1
    # Retry is delayed (scheduled_retry), not immediately enqueued.
    assert worker._queue.enqueued == []  # type: ignore[attr-defined]
    assert len(worker._queue.scheduled_retries) == 1  # type: ignore[attr-defined]
    scheduled = worker._queue.scheduled_retries[0]  # type: ignore[attr-defined]
    assert scheduled[0] == JID
    assert scheduled[3] == 2.0  # backoff delay for attempt 1 is 2s
    assert worker._queue.acked == ["m1"]  # type: ignore[attr-defined]

    # Second delivery: attempt 2 (>= max 2) -> dead_letter, no retry scheduled.
    worker._queue.messages = [_msg(JID, "flaky", message_id="m2")]  # type: ignore[attr-defined]
    out2 = await worker.process_once()
    assert out2.dead_lettered == 1
    assert store.jobs[JID].status is JobStatus.DEAD_LETTER
    assert store.jobs[JID].attempt == 2
    assert len(worker._queue.scheduled_retries) == 1  # type: ignore[attr-defined]
    assert worker._queue.acked == ["m1", "m2"]  # type: ignore[attr-defined]


@pytest.mark.unit
async def test_max_attempts_one_exhausts_on_first_failure() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "flaky", {}, JobStatus.QUEUED, max_attempts=1)})
    registry = TaskRegistry()
    registry.register("flaky", _always_fail)
    worker = Worker(
        queue=FakeQueue(messages=[_msg(JID, "flaky")]),
        store=store,
        registry=registry,
        backoff_schedule=(2, 8, 30),
    )

    outcome = await worker.process_once()

    assert outcome.dead_lettered == 1
    assert store.jobs[JID].status is JobStatus.DEAD_LETTER
    assert store.jobs[JID].attempt == 1
    assert worker._queue.scheduled_retries == []  # type: ignore[attr-defined]
    assert worker._queue.enqueued == []  # type: ignore[attr-defined]


@pytest.mark.unit
async def test_unhandled_handler_exception_is_transient_and_recorded() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "raisey", {}, JobStatus.QUEUED, max_attempts=1)})
    registry = TaskRegistry()
    registry.register("raisey", _raises)
    worker = Worker(queue=FakeQueue(messages=[_msg(JID, "raisey")]), store=store, registry=registry)

    outcome = await worker.process_once()

    assert outcome.dead_lettered == 1  # max_attempts=1 -> exhausted immediately
    job = store.jobs[JID]
    assert job.status is JobStatus.DEAD_LETTER
    assert "ValueError" in (job.last_error or "")
    assert job.traceback is not None


@pytest.mark.unit
async def test_already_terminal_job_is_skipped_and_acked() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "noop", {}, JobStatus.SUCCEEDED, attempt=1)})
    queue = FakeQueue(messages=[_msg(JID, "noop")])
    worker = _worker(store, queue)

    outcome = await worker.process_once()

    assert outcome.outcome is ProcessOutcome.SKIPPED
    assert outcome.skipped == 1
    assert store.jobs[JID].status is JobStatus.SUCCEEDED  # unchanged
    assert store.claimed == []  # claim returned None for terminal
    assert queue.acked == ["m1"]


@pytest.mark.unit
async def test_missing_job_row_is_skipped_and_acked() -> None:
    queue = FakeQueue(messages=[_msg(uuid.uuid4(), "noop")])
    worker = _worker(FakeJobStore(), queue)

    outcome = await worker.process_once()

    assert outcome.outcome is ProcessOutcome.SKIPPED
    assert outcome.skipped == 1
    assert queue.acked == ["m1"]


@pytest.mark.unit
async def test_malformed_message_is_skipped_and_acked() -> None:
    queue = FakeQueue(messages=[JobMessage(message_id="bad", job_id=None, job_type="", payload={})])
    worker = _worker(FakeJobStore(), queue)

    outcome = await worker.process_once()

    assert outcome.skipped == 1
    assert queue.acked == ["bad"]


@pytest.mark.unit
async def test_empty_dequeue_is_idle() -> None:
    worker = _worker(FakeJobStore(), FakeQueue(messages=[]))

    outcome = await worker.process_once()

    assert outcome.outcome is ProcessOutcome.IDLE
    assert outcome.total == 0


@pytest.mark.unit
async def test_reclaim_stale_processes_reclaimed_messages() -> None:
    store = FakeJobStore({JID: FakeJob(JID, "noop", {}, JobStatus.QUEUED)})
    queue = FakeQueue(reclaim_messages=[_msg(JID, "noop")])
    worker = _worker(store, queue)

    outcome = await worker.reclaim_stale()

    assert outcome.succeeded == 1
    assert store.jobs[JID].status is JobStatus.SUCCEEDED
    assert queue.acked == ["m1"]


@pytest.mark.unit
async def test_run_stops_after_max_iterations_and_ensures_group() -> None:
    worker = _worker(FakeJobStore(), FakeQueue(messages=[]))
    queue = worker._queue  # type: ignore[attr-defined]

    await worker.run(max_iterations=2, idle_sleep_s=0)

    assert queue._ensured_group is True  # type: ignore[attr-defined]
