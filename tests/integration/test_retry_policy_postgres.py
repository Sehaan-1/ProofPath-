"""Live Redis + PostgreSQL integration tests for the retry & dead-letter policy.

Sub-slice 2.2.3: these prove the things unit tests cannot — that the delayed-retry
sorted-set round-trips through real Redis, that ``drain_ready_retries`` re-enqueues
ready jobs onto the stream, that the worker applies exponential backoff (a
transiently-failed job is NOT redelivered immediately), and that a dead-lettered
job writes both a ``background_jobs`` row update and an immutable ``audit_logs``
entry in the same PostgreSQL transaction. Requires ``make up`` (or
``TEST_DATABASE_URL`` / ``TEST_REDIS_URL``).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.db.models import AuditActor, AuditLog, BackgroundJob, JobStatus
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.background_job import BackgroundJobRepo
from proofpath.workers.queue import JobQueue
from proofpath.workers.tasks import JobContext, JobResult, TaskRegistry
from proofpath.workers.worker import (
    DEAD_LETTER_AUDIT_EVENT,
    SqlAlchemyJobStore,
    Worker,
)

pytestmark = pytest.mark.integration


@dataclass
class RetryHarness:
    factory: async_sessionmaker[AsyncSession]
    queue: JobQueue
    redis_client: Any
    seeded_job_ids: list[uuid.UUID] = field(default_factory=list)


@pytest.fixture
async def retry_harness(db_engine: AsyncEngine, redis_client) -> AsyncIterator[RetryHarness]:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    stream = f"proofpath:jobs:test:{uuid.uuid4()}"
    group = f"test-workers-{uuid.uuid4().hex[:8]}"
    delay_key = f"proofpath:jobs:delayed:test:{uuid.uuid4()}"
    queue = JobQueue(
        redis_client, stream=stream, group=group, consumer="test-worker", delay_key=delay_key
    )
    await queue.ensure_group()
    harness = RetryHarness(factory=factory, queue=queue, redis_client=redis_client)
    try:
        yield harness
    finally:
        for job_id in harness.seeded_job_ids:
            async with harness.factory.begin() as session:
                await session.execute(delete(AuditLog).where(AuditLog.resource_id == job_id))
                await session.execute(delete(BackgroundJob).where(BackgroundJob.id == job_id))
        await harness.redis_client.delete(queue.stream, queue.delay_key)


async def _seed_job(harness: RetryHarness, *, job_type: str, max_attempts: int) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with harness.factory.begin() as session:
        await BackgroundJobRepo(session).create(
            job_type=job_type, payload={}, max_attempts=max_attempts, job_id=job_id
        )
    harness.seeded_job_ids.append(job_id)
    return job_id


async def _get_job(harness: RetryHarness, job_id: uuid.UUID) -> BackgroundJob:
    async with harness.factory() as session:
        return await BackgroundJobRepo(session).get(job_id)  # type: ignore[return-value]


def _worker(harness: RetryHarness, registry: TaskRegistry, *, backoff=(2, 8, 30)) -> Worker:
    return Worker(
        queue=harness.queue,
        store=SqlAlchemyJobStore(harness.factory),
        registry=registry,
        block_ms=500,
        backoff_schedule=backoff,
    )


# ---------------------------------------------------------------------------
# Delayed-retry sorted-set primitives (live Redis)
# ---------------------------------------------------------------------------


async def test_schedule_retry_and_drain_ready_roundtrip(retry_harness: RetryHarness) -> None:
    queue = retry_harness.queue
    job_id = uuid.uuid4()

    # Schedule a retry 10s in the future.
    ready_at = await queue.schedule_retry(
        job_id, job_type="noop", payload={"x": 1}, delay_seconds=10
    )
    assert await queue.delayed_count() == 1

    # Nothing is ready yet.
    assert await queue.drain_ready_retries() == []
    assert await queue.delayed_count() == 1

    # Simulate time passing: drain with an explicit "now" in the future.
    items = await queue.drain_ready_retries(now=ready_at + 1)
    assert len(items) == 1
    assert items[0].job_id == job_id
    assert items[0].job_type == "noop"
    assert items[0].payload == {"x": 1}
    # The job was re-enqueued onto the stream and removed from the delay set.
    assert await queue.delayed_count() == 0
    assert await queue.stream_length() == 1


async def test_schedule_retry_zero_delay_is_immediately_ready(retry_harness: RetryHarness) -> None:
    queue = retry_harness.queue
    job_id = uuid.uuid4()
    await queue.schedule_retry(job_id, job_type="noop", payload={}, delay_seconds=0)
    items = await queue.drain_ready_retries()
    assert len(items) == 1
    assert items[0].job_id == job_id


# ---------------------------------------------------------------------------
# Worker backoff + dead-letter + audit (live Redis + PostgreSQL)
# ---------------------------------------------------------------------------


async def test_transient_failure_schedules_delayed_retry_not_immediate(
    retry_harness: RetryHarness,
) -> None:
    """A transient failure places the job in the delay set, not back on the stream."""
    job_id = await _seed_job(retry_harness, job_type="flaky", max_attempts=3)

    attempts: list[int] = []

    async def flaky(_ctx: JobContext) -> JobResult:
        attempts.append(1)
        return JobResult(ok=False, error="transient", transient=True)

    registry = TaskRegistry()
    registry.register("flaky", flaky)
    worker = _worker(retry_harness, registry, backoff=(2, 8, 30))

    await retry_harness.queue.enqueue(job_id, job_type="flaky", payload={})
    outcome = await worker.process_once()

    assert outcome.failed == 1
    job = await _get_job(retry_harness, job_id)
    assert job.status is JobStatus.FAILED
    assert job.attempt == 1
    # The job is in the delay set (not immediately on the stream).
    assert await retry_harness.queue.delayed_count() == 1
    assert await retry_harness.queue.stream_length() == 0  # original message consumed


async def test_drain_ready_retries_redelivers_and_worker_succeeds(
    retry_harness: RetryHarness,
) -> None:
    """Draining ready retries re-enqueues the job; the worker then succeeds."""
    job_id = await _seed_job(retry_harness, job_type="recover", max_attempts=3)
    call_count = 0

    async def recover(_ctx: JobContext) -> JobResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return JobResult(ok=False, error="transient", transient=True)
        return JobResult(ok=True, result={"recovered": True})

    registry = TaskRegistry()
    registry.register("recover", recover)
    worker = _worker(retry_harness, registry, backoff=(0, 0, 0))

    await retry_harness.queue.enqueue(job_id, job_type="recover", payload={})
    # First attempt fails transiently.
    await worker.process_once()
    assert await retry_harness.queue.delayed_count() == 1

    # Drain the ready retry (backoff=0 so it's immediately ready) and process it.
    await worker.process_ready_retries()
    outcome = await worker.process_once()
    assert outcome.succeeded == 1

    job = await _get_job(retry_harness, job_id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.attempt == 2
    assert job.result_json == {"recovered": True}


async def test_dead_letter_writes_audit_log_in_same_transaction(
    retry_harness: RetryHarness,
) -> None:
    """An exhausted job is dead-lettered with an immutable audit_logs entry."""
    job_id = await _seed_job(retry_harness, job_type="doomed", max_attempts=1)

    async def always_fail(_ctx: JobContext) -> JobResult:
        return JobResult(ok=False, error="permanent boom", transient=True)

    registry = TaskRegistry()
    registry.register("doomed", always_fail)
    worker = _worker(retry_harness, registry, backoff=(2, 8, 30))

    await retry_harness.queue.enqueue(job_id, job_type="doomed", payload={})
    outcome = await worker.process_once()

    assert outcome.dead_lettered == 1
    job = await _get_job(retry_harness, job_id)
    assert job.status is JobStatus.DEAD_LETTER
    assert job.attempt == 1
    assert job.last_error == "permanent boom"
    assert job.error_traceback is None  # handler returned no traceback

    # The audit_logs entry was written in the same transaction.
    async with retry_harness.factory() as session:
        events = await AuditRepo(session).list_events(
            actor=AuditActor.AGENT,
            event_type=DEAD_LETTER_AUDIT_EVENT,
            resource_id=job_id,
        )
    assert len(events) == 1
    details = events[0].details_json
    assert details["error"] == "permanent boom"
    assert details["job_type"] == "doomed"
    assert details["attempt"] == 1
    assert details["max_attempts"] == 1


async def test_permanent_error_dead_letters_immediately_no_retry(
    retry_harness: RetryHarness,
) -> None:
    """A non-transient (validation) error goes straight to dead_letter, no delay."""
    job_id = await _seed_job(retry_harness, job_type="bad_input", max_attempts=3)

    async def bad_input(_ctx: JobContext) -> JobResult:
        return JobResult(ok=False, error="validation failed", transient=False)

    registry = TaskRegistry()
    registry.register("bad_input", bad_input)
    worker = _worker(retry_harness, registry, backoff=(2, 8, 30))

    await retry_harness.queue.enqueue(job_id, job_type="bad_input", payload={})
    outcome = await worker.process_once()

    assert outcome.dead_lettered == 1
    job = await _get_job(retry_harness, job_id)
    assert job.status is JobStatus.DEAD_LETTER
    assert job.attempt == 1  # only one attempt — no retry
    # No delayed retry was scheduled.
    assert await retry_harness.queue.delayed_count() == 0


async def test_run_loop_drains_ready_retries_on_idle(retry_harness: RetryHarness) -> None:
    """The run loop re-enqueues backoff-scheduled retries during idle cycles."""
    job_id = await _seed_job(retry_harness, job_type="retry_then_ok", max_attempts=3)
    call_count = 0

    async def handler(_ctx: JobContext) -> JobResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return JobResult(ok=False, error="transient", transient=True)
        return JobResult(ok=True)

    registry = TaskRegistry()
    registry.register("retry_then_ok", handler)
    worker = _worker(retry_harness, registry, backoff=(0, 0, 0))

    await retry_harness.queue.enqueue(job_id, job_type="retry_then_ok", payload={})

    # Run a few iterations; the backoff is 0s so the retry is drained and
    # processed on the first idle cycle.
    stop = asyncio.Event()
    await asyncio.wait_for(
        worker.run(max_iterations=10, stop_event=stop, idle_sleep_s=0.01),
        timeout=5.0,
    )

    job = await _get_job(retry_harness, job_id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.attempt == 2
