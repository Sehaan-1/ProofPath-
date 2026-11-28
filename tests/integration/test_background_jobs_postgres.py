"""Live Redis + PostgreSQL integration tests for the background job queue.

Sub-slice 2.2.1: these prove the things unit tests cannot — that the Redis
Stream consumer group round-trips real messages, that `BackgroundJobRepo` drives
the status state machine against real PostgreSQL with `SELECT ... FOR UPDATE`,
and that the `Worker` end-to-end drains the queue and finalizes the ledger
(succeed, transient retry + re-enqueue, dead-letter on exhaustion, duplicate
skip). Requires `make up` (or `TEST_DATABASE_URL` / `TEST_REDIS_URL`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from redis.asyncio import Redis
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.db.models import BackgroundJob, JobStatus
from proofpath.db.repositories.background_job import BackgroundJobRepo
from proofpath.workers.queue import JobQueue
from proofpath.workers.tasks import JobContext, JobResult, TaskRegistry, default_registry
from proofpath.workers.worker import ProcessOutcome, SqlAlchemyJobStore, Worker

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Repository state machine against real PostgreSQL (transactional, rolled back)
# ---------------------------------------------------------------------------


async def test_background_job_repo_state_machine_against_postgres(
    db_session: AsyncSession,
) -> None:
    repo = BackgroundJobRepo(db_session)

    job = await repo.create(job_type="github_ingest", payload={"repo": "x"}, max_attempts=2)
    assert job.status is JobStatus.QUEUED
    assert job.attempt == 0

    claimed = await repo.mark_running(job.id)
    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING
    assert claimed.attempt == 1
    assert claimed.started_at is not None

    # A redelivery (crash recovery / reclaim) re-claims and bumps attempt again.
    claimed_again = await repo.mark_running(job.id)
    assert claimed_again is not None
    assert claimed_again.attempt == 2

    succeeded = await repo.mark_succeeded(job.id, result={"ingested": 3})
    assert succeeded.status is JobStatus.SUCCEEDED
    assert succeeded.result_json == {"ingested": 3}
    assert succeeded.finished_at is not None

    # Terminal rows are never re-claimed or re-finalized (duplicate-delivery guard).
    assert await repo.mark_running(job.id) is None
    assert await repo.mark_failed(job.id, error="late") is None
    assert await repo.mark_dead_letter(job.id, error="late") is None

    listed = await repo.list_jobs(status=JobStatus.SUCCEEDED)
    assert any(item.id == job.id for item in listed)


async def test_background_job_repo_transient_to_dead_letter_against_postgres(
    db_session: AsyncSession,
) -> None:
    repo = BackgroundJobRepo(db_session)
    job = await repo.create(job_type="flaky", payload={}, max_attempts=2)

    await repo.mark_running(job.id)  # attempt 1
    failed = await repo.mark_failed(job.id, error="timeout", traceback="tb-1")
    assert failed.status is JobStatus.FAILED
    assert failed.last_error == "timeout"
    assert failed.error_traceback == "tb-1"
    assert failed.finished_at is not None

    # A failed row is still claimable (retry), then dead-lettered on exhaustion.
    await repo.mark_running(job.id)  # attempt 2
    dead = await repo.mark_dead_letter(job.id, error="exhausted", traceback="tb-2")
    assert dead.status is JobStatus.DEAD_LETTER
    assert dead.last_error == "exhausted"

    assert await repo.mark_running(job.id) is None  # terminal -> skip


# ---------------------------------------------------------------------------
# Redis Stream queue primitives (live Redis)
# ---------------------------------------------------------------------------


async def test_job_queue_enqueue_dequeue_ack_roundtrip(redis_client: Redis) -> None:
    stream = f"proofpath:jobs:test:{uuid.uuid4()}"
    group = f"test-workers-{uuid.uuid4().hex[:8]}"
    queue = JobQueue(redis_client, stream=stream, group=group, consumer="test-q")
    await queue.ensure_group()
    try:
        assert await queue.stream_length() == 0

        job_id = uuid.uuid4()
        message_id = await queue.enqueue(job_id, job_type="noop", payload={"x": 1})
        assert await queue.stream_length() == 1

        messages = await queue.dequeue(count=1, block_ms=1000)
        assert len(messages) == 1
        msg = messages[0]
        assert msg.message_id == message_id
        assert msg.job_id == job_id
        assert msg.job_type == "noop"
        assert msg.payload == {"x": 1}

        # Delivered but not yet acked -> pending entries list grows.
        assert await queue.pending_count() == 1
        assert await queue.ack(message_id) == 1
        assert await queue.pending_count() == 0

        # No further new messages for the group.
        assert await queue.dequeue(count=1, block_ms=200) == []
    finally:
        await redis_client.delete(stream)


async def test_job_queue_ensure_group_is_idempotent(redis_client: Redis) -> None:
    stream = f"proofpath:jobs:test:{uuid.uuid4()}"
    group = f"test-workers-{uuid.uuid4().hex[:8]}"
    queue = JobQueue(redis_client, stream=stream, group=group, consumer="test-q")
    try:
        await queue.ensure_group()
        await queue.ensure_group()  # second call must not raise BUSYGROUP
    finally:
        await redis_client.delete(stream)


# ---------------------------------------------------------------------------
# Worker end-to-end (live Redis + PostgreSQL)
# ---------------------------------------------------------------------------


@dataclass
class WorkerHarness:
    factory: async_sessionmaker[AsyncSession]
    queue: JobQueue
    redis: Redis
    seeded_job_ids: list[uuid.UUID] = field(default_factory=list)


@pytest.fixture
async def worker_harness(
    db_engine: AsyncEngine, redis_client: Redis
) -> AsyncIterator[WorkerHarness]:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    stream = f"proofpath:jobs:test:{uuid.uuid4()}"
    group = f"test-workers-{uuid.uuid4().hex[:8]}"
    queue = JobQueue(redis_client, stream=stream, group=group, consumer="test-worker")
    await queue.ensure_group()
    harness = WorkerHarness(factory=factory, queue=queue, redis=redis_client)
    try:
        yield harness
    finally:
        for job_id in harness.seeded_job_ids:
            async with harness.factory.begin() as session:
                await session.execute(delete(BackgroundJob).where(BackgroundJob.id == job_id))
        await harness.redis.delete(harness.queue.stream)


async def _seed_job(
    harness: WorkerHarness,
    *,
    job_type: str = "noop",
    payload: dict[str, object] | None = None,
    max_attempts: int = 3,
    status: JobStatus = JobStatus.QUEUED,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with harness.factory.begin() as session:
        repo = BackgroundJobRepo(session)
        await repo.create(
            job_type=job_type,
            payload=dict(payload or {}),
            max_attempts=max_attempts,
            job_id=job_id,
        )
        if status is not JobStatus.QUEUED:
            await session.execute(
                update(BackgroundJob).where(BackgroundJob.id == job_id).values(status=status.value)
            )
    harness.seeded_job_ids.append(job_id)
    return job_id


async def _fetch_job(harness: WorkerHarness, job_id: uuid.UUID) -> BackgroundJob:
    async with harness.factory() as session:
        return await BackgroundJobRepo(session).get(job_id)  # type: ignore[return-value]


def _worker(harness: WorkerHarness, registry: TaskRegistry) -> Worker:
    return Worker(
        queue=harness.queue,
        store=SqlAlchemyJobStore(harness.factory),
        registry=registry,
        block_ms=1000,
    )


async def test_worker_processes_noop_job_end_to_end(worker_harness: WorkerHarness) -> None:
    job_id = await _seed_job(worker_harness, job_type="noop")
    await worker_harness.queue.enqueue(job_id, job_type="noop", payload={})

    outcome = await _worker(worker_harness, default_registry()).process_once()

    assert outcome.outcome is ProcessOutcome.PROCESSED
    assert outcome.succeeded == 1
    job = await _fetch_job(worker_harness, job_id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.attempt == 1
    assert await worker_harness.queue.pending_count() == 0  # acked


async def test_worker_persists_handler_result(worker_harness: WorkerHarness) -> None:
    job_id = await _seed_job(worker_harness, job_type="echo", payload={"k": "v"})
    await worker_harness.queue.enqueue(job_id, job_type="echo", payload={"k": "v"})

    await _worker(worker_harness, default_registry()).process_once()

    job = await _fetch_job(worker_harness, job_id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.result_json == {"job_type": "echo", "payload": {"k": "v"}}


async def _always_fail(_ctx: JobContext) -> JobResult:
    return JobResult(ok=False, error="boom", transient=True)


async def test_worker_retries_then_dead_letters_transient_failures(
    worker_harness: WorkerHarness,
) -> None:
    registry = TaskRegistry()
    registry.register("flaky", _always_fail)
    job_id = await _seed_job(worker_harness, job_type="flaky", max_attempts=2)
    await worker_harness.queue.enqueue(job_id, job_type="flaky", payload={})
    worker = _worker(worker_harness, registry)

    first = await worker.process_once()
    assert first.failed == 1  # attempt 1 < max 2 -> transient, re-enqueued
    job = await _fetch_job(worker_harness, job_id)
    assert job.status is JobStatus.FAILED
    assert job.attempt == 1
    assert job.last_error == "boom"

    # The re-enqueued message is a new stream entry -> delivered on the next read.
    second = await worker.process_once()
    assert second.dead_lettered == 1  # attempt 2 >= max 2 -> dead_letter
    job = await _fetch_job(worker_harness, job_id)
    assert job.status is JobStatus.DEAD_LETTER
    assert job.attempt == 2
    assert job.last_error == "boom"
    assert await worker_harness.queue.pending_count() == 0  # final ack, no re-enqueue


async def test_worker_dead_letters_unknown_task_type(worker_harness: WorkerHarness) -> None:
    job_id = await _seed_job(worker_harness, job_type="bogus")
    await worker_harness.queue.enqueue(job_id, job_type="bogus", payload={})

    outcome = await _worker(worker_harness, default_registry()).process_once()

    assert outcome.dead_lettered == 1
    job = await _fetch_job(worker_harness, job_id)
    assert job.status is JobStatus.DEAD_LETTER
    assert "unknown task type" in (job.last_error or "")


async def test_worker_skips_already_succeeded_duplicate_delivery(
    worker_harness: WorkerHarness,
) -> None:
    job_id = await _seed_job(worker_harness, job_type="noop", status=JobStatus.SUCCEEDED)
    await worker_harness.queue.enqueue(job_id, job_type="noop", payload={})

    outcome = await _worker(worker_harness, default_registry()).process_once()

    assert outcome.outcome is ProcessOutcome.SKIPPED
    job = await _fetch_job(worker_harness, job_id)
    assert job.status is JobStatus.SUCCEEDED  # unchanged
    assert job.attempt == 0  # never claimed
    assert await worker_harness.queue.pending_count() == 0  # acked-and-skipped
