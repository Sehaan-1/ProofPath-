"""Typed async repository for the Redis-backed `background_jobs` status ledger.

Sub-part 2.2.1b: Redis Streams carry job messages; this repository owns the
authoritative PostgreSQL status row keyed by the same job id. Every state
transition is a compare-and-set under a ``SELECT ... FOR UPDATE`` row lock so a
duplicate or concurrent delivery (reclaim, re-enqueue, or a crashed worker
restart) can never double-process a finished job or clobber a terminal state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import JOB_TERMINAL_STATUSES, BackgroundJob, JobStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from proofpath.schemas.jobs import BackgroundJobRecord

_DEFAULT_MAX_ATTEMPTS = 3


def background_job_record(job: BackgroundJob) -> BackgroundJobRecord:
    """Map a persisted `BackgroundJob` row to its API/state contract.

    Imported lazily: `proofpath.schemas.jobs` reads `proofpath.db.models`, which
    initializes the `proofpath.db` package and therefore this module.
    """
    from proofpath.schemas.jobs import BackgroundJobRecord

    return BackgroundJobRecord(
        id=job.id,
        job_type=job.job_type,
        payload=dict(job.payload_json),
        status=job.status,
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        last_error=job.last_error,
        error_traceback=job.error_traceback,
        result=dict(job.result_json) if job.result_json is not None else None,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


class BackgroundJobRepo:
    """Async state-machine repository for `background_jobs`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        job_type: str,
        payload: dict[str, Any],
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        job_id: uuid.UUID | None = None,
    ) -> BackgroundJob:
        """Persist a new `queued` background job row (does not touch Redis)."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        job = BackgroundJob(
            id=job_id or uuid.uuid4(),
            job_type=job_type,
            payload_json=dict(payload),
            status=JobStatus.QUEUED,
            attempt=0,
            max_attempts=max_attempts,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(self, job_id: uuid.UUID) -> BackgroundJob | None:
        """Fetch a `BackgroundJob` by UUID."""
        stmt = select(BackgroundJob).where(BackgroundJob.id == job_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_for_update(self, job_id: uuid.UUID) -> BackgroundJob | None:
        """Fetch a `BackgroundJob` by UUID holding a row lock (`SELECT ... FOR UPDATE`).

        Every state transition takes this lock so concurrent deliveries of the
        same job serialize: the loser sees the committed terminal/running state
        and performs no second side effect.
        """
        stmt = select(BackgroundJob).where(BackgroundJob.id == job_id).with_for_update()
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        job_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BackgroundJob]:
        """List background jobs, filtered by status and/or type, newest-first."""
        stmt = select(BackgroundJob)
        if status is not None:
            stmt = stmt.where(BackgroundJob.status == status)
        if job_type is not None:
            stmt = stmt.where(BackgroundJob.job_type == job_type)
        stmt = stmt.order_by(BackgroundJob.created_at.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_running(self, job_id: uuid.UUID) -> BackgroundJob | None:
        """Claim a job for processing: ``queued/failed`` -> ``running``, attempt+1.

        Returns the row when claimed, or ``None`` when the job is missing or
        already terminal (``succeeded`` / ``dead_letter``) — in both cases the
        caller should ack the Redis message and skip it.
        """
        job = await self.get_for_update(job_id)
        if job is None or job.status in JOB_TERMINAL_STATUSES:
            return None
        job.status = JobStatus.RUNNING
        job.attempt = job.attempt + 1
        job.started_at = datetime.now(UTC)
        await self._session.flush()
        return job

    async def mark_succeeded(
        self,
        job_id: uuid.UUID,
        *,
        result: dict[str, Any] | None = None,
    ) -> BackgroundJob | None:
        """Terminal success transition: any non-terminal state -> ``succeeded``."""
        job = await self.get_for_update(job_id)
        if job is None or job.status in JOB_TERMINAL_STATUSES:
            return None
        job.status = JobStatus.SUCCEEDED
        job.result_json = dict(result) if result is not None else None
        job.finished_at = datetime.now(UTC)
        await self._session.flush()
        return job

    async def mark_failed(
        self,
        job_id: uuid.UUID,
        *,
        error: str,
        traceback: str | None = None,
    ) -> BackgroundJob | None:
        """Transient failure: record the error and move to ``failed`` (pending retry).

        The worker re-enqueues the job in Redis after committing this transition;
        the next pickup re-claims the row and bumps ``attempt`` again.
        """
        job = await self.get_for_update(job_id)
        if job is None or job.status in JOB_TERMINAL_STATUSES:
            return None
        job.status = JobStatus.FAILED
        job.last_error = error
        job.error_traceback = traceback
        job.finished_at = datetime.now(UTC)
        await self._session.flush()
        return job

    async def mark_dead_letter(
        self,
        job_id: uuid.UUID,
        *,
        error: str,
        traceback: str | None = None,
    ) -> BackgroundJob | None:
        """Terminal failure: retries exhausted (or a permanent error) -> ``dead_letter``.

        The row stays in the table for replay/inspection (Sub-slice 2.2.3b);
        the Redis message is acked so it is not redelivered.
        """
        job = await self.get_for_update(job_id)
        if job is None or job.status is JobStatus.SUCCEEDED:
            return None
        job.status = JobStatus.DEAD_LETTER
        job.last_error = error
        job.error_traceback = traceback
        job.finished_at = datetime.now(UTC)
        await self._session.flush()
        return job
