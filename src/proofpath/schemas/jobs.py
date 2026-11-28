"""Background job ledger contracts (Sub-slice 2.2.1)."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import AwareDatetime, Field

from proofpath.db.models import JobStatus
from proofpath.schemas.base import StrictModel

_MAX_JOB_TYPE = 128


class BackgroundJobRecord(StrictModel):
    """API/state contract for a `background_jobs` ledger row.

    Redis is the transport; this record is the queryable, durable view of a job's
    lifecycle. The ``GET /api/v1/jobs/{job_id}`` endpoint (Sub-slice 2.2.2)
    returns this shape; the worker and repositories use it internally today.
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
