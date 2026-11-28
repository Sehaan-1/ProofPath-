"""Redis-backed background workers for ProofPath (Sub-slices 2.2.1-2.2.3).

The worker service (``docker-compose.yml`` → ``worker``) runs
``python -m proofpath.workers``, which builds a `JobQueue` against Redis, a
`SqlAlchemyJobStore` against PostgreSQL, and a `Worker` that drains the queue and
drives the ``background_jobs`` ledger with exponential-backoff retries and a
dead-letter audit trail. See :mod:`proofpath.workers.worker`.
"""

from proofpath.workers.queue import (
    DEFAULT_DELAY_KEY,
    DEFAULT_GROUP,
    DEFAULT_STREAM,
    JobMessage,
    JobQueue,
    RetryItem,
)
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
    DEAD_LETTER_AUDIT_EVENT,
    DEFAULT_BACKOFF_SCHEDULE,
    BatchOutcome,
    ClaimedJob,
    JobStore,
    ProcessOutcome,
    SqlAlchemyJobStore,
    Worker,
    compute_backoff_delay,
)

__all__ = [
    "DEAD_LETTER_AUDIT_EVENT",
    "DEFAULT_BACKOFF_SCHEDULE",
    "DEFAULT_DELAY_KEY",
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
    "RetryItem",
    "SqlAlchemyJobStore",
    "TaskHandler",
    "TaskRegistry",
    "Worker",
    "compute_backoff_delay",
    "default_registry",
    "echo_handler",
    "noop_handler",
    "validate_job_type",
]
