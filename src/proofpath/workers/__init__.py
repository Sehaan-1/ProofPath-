"""Redis-backed background workers for ProofPath (Sub-slice 2.2.1).

The worker service (``docker-compose.yml`` → ``worker``) runs
``python -m proofpath.workers``, which builds a `JobQueue` against Redis, a
`SqlAlchemyJobStore` against PostgreSQL, and a `Worker` that drains the queue and
drives the ``background_jobs`` ledger. See :mod:`proofpath.workers.worker`.
"""

from proofpath.workers.queue import DEFAULT_GROUP, DEFAULT_STREAM, JobMessage, JobQueue
from proofpath.workers.tasks import (
    ECHO_TASK,
    NOOP_TASK,
    JobContext,
    JobResult,
    TaskHandler,
    TaskRegistry,
    default_registry,
    echo_handler,
    noop_handler,
    validate_job_type,
)
from proofpath.workers.worker import (
    BatchOutcome,
    ClaimedJob,
    JobStore,
    ProcessOutcome,
    SqlAlchemyJobStore,
    Worker,
)

__all__ = [
    "DEFAULT_GROUP",
    "DEFAULT_STREAM",
    "ECHO_TASK",
    "NOOP_TASK",
    "BatchOutcome",
    "ClaimedJob",
    "JobContext",
    "JobMessage",
    "JobQueue",
    "JobResult",
    "JobStore",
    "ProcessOutcome",
    "SqlAlchemyJobStore",
    "TaskHandler",
    "TaskRegistry",
    "Worker",
    "default_registry",
    "echo_handler",
    "noop_handler",
    "validate_job_type",
]
