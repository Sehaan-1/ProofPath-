"""Background job status endpoint (Sub-slice 2.2.2).

``GET /api/v1/jobs/{job_id}`` reads the durable ``background_jobs`` ledger row so a
client can poll the outcome of an async ingestion or analysis job accepted with
``202``. The row is the source of truth (Redis is only the transport), so this
read works even after the worker has acked and dropped the stream message.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from proofpath.api.routes import Session
from proofpath.db.repositories.background_job import BackgroundJobRepo
from proofpath.schemas.jobs import JobStatusResponse

router = APIRouter(prefix="/api/v1", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: uuid.UUID, session: Session) -> JobStatusResponse:
    """Return the trimmed status of one background job, or 404 when unknown."""
    job = await BackgroundJobRepo(session).get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return JobStatusResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        last_error=job.last_error,
        result=dict(job.result_json) if job.result_json is not None else None,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
