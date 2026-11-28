"""Background worker that drains the Redis job queue and drives the job ledger.

Sub-part 2.2.1b: the worker is the bridge between the Redis Stream (transport)
and the ``background_jobs`` PostgreSQL table (state). For each message it:

1. claims the job row and atomically moves it ``queued/failed -> running`` — a
   duplicate delivery can never re-run a finished job;
2. dispatches a registered handler **outside** the DB transaction so long-running
   LLM/ingestion work never holds a connection;
3. finalizes the row — ``succeeded`` on success, ``failed`` (then re-enqueued) on
   a transient error, or ``dead_letter`` when retries are exhausted or a handler
   reports a permanent error — and acks the Redis message.

The DB interactions live behind a :class:`JobStore` protocol so the orchestration
is unit-testable with an in-memory fake; :class:`SqlAlchemyJobStore` is the
production implementation wrapping :class:`BackgroundJobRepo`.

Exponential backoff timing and a dead-letter replay CLI are Sub-slice 2.2.3;
today a transient failure re-enqueues immediately and the row caps at
``max_attempts`` before moving to ``dead_letter``.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.db.repositories.background_job import BackgroundJobRepo
from proofpath.workers.queue import JobMessage, JobQueue
from proofpath.workers.tasks import JobContext, JobResult, TaskRegistry

logger = logging.getLogger(__name__)


class ProcessOutcome(StrEnum):
    """Coarse result of one `Worker.process_once` batch for the run loop."""

    IDLE = "idle"  # no messages were available
    PROCESSED = "processed"  # at least one handler was dispatched
    SKIPPED = "skipped"  # messages seen but none dispatched (all terminal/orphan/malformed)


class _MessageResult(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """Per-batch tally returned by `Worker.process_once` / `reclaim_stale`."""

    outcome: ProcessOutcome
    dispatched: int = 0
    succeeded: int = 0
    failed: int = 0
    dead_lettered: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.dispatched + self.skipped


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    """A job claimed for one processing attempt, with its dispatch context."""

    job_id: uuid.UUID
    job_type: str
    payload: dict[str, Any]
    attempt: int
    max_attempts: int


class JobStore(Protocol):
    """Persistence seam for the worker so its logic is unit-testable without DB."""

    async def claim(self, job_id: uuid.UUID) -> ClaimedJob | None:
        """Atomically move ``queued/failed`` -> ``running`` and bump ``attempt``.

        Returns the claimed context, or ``None`` when the row is missing or
        already terminal (``succeeded`` / ``dead_letter``).
        """
        ...

    async def succeed(self, job_id: uuid.UUID, *, result: dict[str, Any] | None = None) -> None:
        """Terminal success transition."""
        ...

    async def fail(self, job_id: uuid.UUID, *, error: str, traceback: str | None = None) -> None:
        """Transient failure: record the error, pending retry."""
        ...

    async def dead_letter(
        self, job_id: uuid.UUID, *, error: str, traceback: str | None = None
    ) -> None:
        """Terminal failure: retries exhausted or a permanent error."""
        ...


class SqlAlchemyJobStore:
    """Production `JobStore` backed by `BackgroundJobRepo` and async sessions.

    Each state transition opens its own short-lived session so a handler's
    long-running work (run between :meth:`claim` and the finalize calls) never
    holds a database connection.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def claim(self, job_id: uuid.UUID) -> ClaimedJob | None:
        async with self._session() as session:
            repo = BackgroundJobRepo(session)
            job = await repo.mark_running(job_id)
            # Commit to release the FOR UPDATE lock and persist the running state
            # before dispatch, so a crash mid-handler leaves a durable `running` row
            # rather than a phantom queued one.
            await session.commit()
            if job is None:
                return None
            return ClaimedJob(
                job_id=job.id,
                job_type=job.job_type,
                payload=dict(job.payload_json),
                attempt=job.attempt,
                max_attempts=job.max_attempts,
            )

    async def succeed(self, job_id: uuid.UUID, *, result: dict[str, Any] | None = None) -> None:
        async with self._session() as session:
            await BackgroundJobRepo(session).mark_succeeded(job_id, result=result)
            await session.commit()

    async def fail(self, job_id: uuid.UUID, *, error: str, traceback: str | None = None) -> None:
        async with self._session() as session:
            await BackgroundJobRepo(session).mark_failed(job_id, error=error, traceback=traceback)
            await session.commit()

    async def dead_letter(
        self, job_id: uuid.UUID, *, error: str, traceback: str | None = None
    ) -> None:
        async with self._session() as session:
            await BackgroundJobRepo(session).mark_dead_letter(
                job_id, error=error, traceback=traceback
            )
            await session.commit()


class Worker:
    """Drains a `JobQueue`, dispatches handlers, and updates the job ledger."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        store: JobStore,
        registry: TaskRegistry,
        block_ms: int = 5_000,
        batch_size: int = 1,
        reclaim_min_idle_ms: int = 60_000,
        reclaim_batch_size: int = 10,
    ) -> None:
        self._queue = queue
        self._store = store
        self._registry = registry
        self._block_ms = block_ms
        self._batch_size = batch_size
        self._reclaim_min_idle_ms = reclaim_min_idle_ms
        self._reclaim_batch_size = reclaim_batch_size

    async def process_once(self) -> BatchOutcome:
        """Block on the queue for one batch, process every message, return a tally."""
        messages = await self._queue.dequeue(count=self._batch_size, block_ms=self._block_ms)
        return await self._process_messages(messages)

    async def reclaim_stale(self) -> BatchOutcome:
        """Claim and process messages stranded in the PEL by a stalled consumer."""
        messages = await self._queue.reclaim(
            min_idle_ms=self._reclaim_min_idle_ms,
            count=self._reclaim_batch_size,
        )
        return await self._process_messages(messages)

    async def _process_messages(self, messages: list[JobMessage]) -> BatchOutcome:
        dispatched = succeeded = failed = dead_lettered = skipped = 0
        for message in messages:
            result = await self._process_message(message)
            if result is _MessageResult.SUCCEEDED:
                succeeded += 1
                dispatched += 1
            elif result is _MessageResult.FAILED:
                failed += 1
                dispatched += 1
            elif result is _MessageResult.DEAD_LETTERED:
                dead_lettered += 1
                dispatched += 1
            else:
                skipped += 1
        if not messages:
            outcome = ProcessOutcome.IDLE
        elif dispatched > 0:
            outcome = ProcessOutcome.PROCESSED
        else:
            outcome = ProcessOutcome.SKIPPED
        return BatchOutcome(
            outcome=outcome,
            dispatched=dispatched,
            succeeded=succeeded,
            failed=failed,
            dead_lettered=dead_lettered,
            skipped=skipped,
        )

    async def _process_message(self, message: JobMessage) -> _MessageResult:
        """Claim, dispatch, and finalize one job; always resolves the Redis message."""
        if message.job_id is None:
            # Malformed stream entry — no job row to update; drop and ack.
            await self._queue.ack(message.message_id)
            return _MessageResult.SKIPPED

        claimed = await self._store.claim(message.job_id)
        if claimed is None:
            # Missing row, or already terminal (succeeded/dead_letter) — drop and ack.
            await self._queue.ack(message.message_id)
            return _MessageResult.SKIPPED

        ctx = JobContext(
            job_id=claimed.job_id,
            job_type=claimed.job_type,
            payload=claimed.payload,
            attempt=claimed.attempt,
        )
        result = await self._dispatch(ctx)
        return await self._finalize(claimed, message, result)

    async def _dispatch(self, ctx: JobContext) -> JobResult:
        """Invoke the registered handler, mapping unhandled exceptions to a retry."""
        handler = self._registry.get(ctx.job_type)
        if handler is None:
            return JobResult(
                ok=False,
                error=f"unknown task type {ctx.job_type!r}",
                transient=False,
            )
        try:
            return await handler(ctx)
        except Exception as exc:  # worker must not crash on a bad handler
            logger.exception(
                "Handler for task %s raised on job %s (attempt %d).",
                ctx.job_type,
                ctx.job_id,
                ctx.attempt,
            )
            return JobResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
                transient=True,
            )

    async def _finalize(
        self,
        claimed: ClaimedJob,
        message: JobMessage,
        result: JobResult,
    ) -> _MessageResult:
        """Persist the terminal/transient outcome and resolve the Redis message."""
        error = result.error or "unknown error"
        if result.ok:
            await self._store.succeed(claimed.job_id, result=result.result)
            await self._queue.ack(message.message_id)
            return _MessageResult.SUCCEEDED

        exhausted = (not result.transient) or claimed.attempt >= claimed.max_attempts
        if exhausted:
            await self._store.dead_letter(claimed.job_id, error=error, traceback=result.traceback)
            await self._queue.ack(message.message_id)
            return _MessageResult.DEAD_LETTERED

        # Transient failure with retries remaining: record it, re-enqueue, ack.
        await self._store.fail(claimed.job_id, error=error, traceback=result.traceback)
        await self._queue.enqueue(
            claimed.job_id,
            job_type=claimed.job_type,
            payload=claimed.payload,
        )
        await self._queue.ack(message.message_id)
        return _MessageResult.FAILED

    async def run(
        self,
        *,
        max_iterations: int | None = None,
        stop_event: asyncio.Event | None = None,
        idle_sleep_s: float = 0.1,
    ) -> None:
        """Run the consume loop until cancelled, stopped, or ``max_iterations`` reached.

        ``process_once`` blocks on Redis for ``block_ms``, so the loop only spins
        when the queue is actually idle. On idle iterations it reclaims stranded
        PEL messages (crashed-consumer recovery) before a short sleep.
        """
        await self._queue.ensure_group()
        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                if stop_event is not None and stop_event.is_set():
                    break
                outcome = await self.process_once()
                iterations += 1
                if outcome.outcome is ProcessOutcome.IDLE:
                    await self.reclaim_stale()
                    if stop_event is not None and stop_event.is_set():
                        break
                    await asyncio.sleep(idle_sleep_s)
        except asyncio.CancelledError:
            logger.info("Worker '%s' cancelled; shutting down cleanly.", self._queue.consumer)
            raise
        logger.info("Worker '%s' stopped after %d iterations.", self._queue.consumer, iterations)
