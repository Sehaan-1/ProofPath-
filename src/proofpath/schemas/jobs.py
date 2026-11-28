"""Background job ledger contracts (Sub-slice 2.2.1 / 2.2.2)."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import AwareDatetime, Field

from proofpath.db.models import JobStatus
from proofpath.schemas.base import StrictModel

_MAX_JOB_TYPE = 128


class BackgroundJobRecord(StrictModel):
    """Internal/state contract for a `background_jobs` ledger row.

    Redis is the transport; this record is the durable view of a job's lifecycle
    used by the worker and repositories. The public API surface
    (`GET /api/v1/jobs/{job_id}`) uses the trimmed `JobStatusResponse`.
    """

    id: uuid.UUID
    job_type: str = Field(min_length=1, max_length=_MAX_JOB_TYPE)
    payload: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    last_error: str | None = None
    error_traceback: str | None = None
    result: dict[str, Any] | None = None
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class JobAcceptedResponse(StrictModel):
    """`202 Accepted` body returned when a job is enqueued (Sub-slice 2.2.2)."""

    job_id: uuid.UUID
    job_type: str = Field(min_length=1, max_length=_MAX_JOB_TYPE)
    status: JobStatus = Field(description="Always `queued` at acceptance time.")


class JobStatusResponse(StrictModel):
    """Trimmed public view of a `background_jobs` row (`GET /api/v1/jobs/{job_id}`).

    Omits the raw payload (which may carry repo references or calendar times) and
    the full Python traceback (kept internal to `BackgroundJobRecord`), exposing
    only what a client polling a job needs: status, attempt budget, the short
    error message, and the structured result.
    """

    id: uuid.UUID
    job_type: str = Field(min_length=1, max_length=_MAX_JOB_TYPE)
    status: JobStatus
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    last_error: str | None = None
    result: dict[str, Any] | None = None
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
