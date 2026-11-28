"""ProofPath background worker entrypoint.

Run with ``python -m proofpath.workers`` (the ``worker`` Docker Compose service
does exactly this). It wires Redis (queue), PostgreSQL (job ledger), and the
Postgres checkpointer (required for analysis runs), then registers the real
async-work handlers from :mod:`proofpath.jobs` and runs the consume loop until
SIGINT/SIGTERM.

GitHub ingestion handlers need only settings + a session factory; the analysis
handler additionally needs a live ``AsyncPostgresSaver``. If the checkpointer
cannot be opened, the worker still starts with the ``github_ingest`` and
smoke-test tasks registered — an enqueued analysis job is then dead-lettered as
an unknown task type rather than crashing the worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from typing import TYPE_CHECKING

from proofpath.config import Settings, get_settings
from proofpath.db.session import create_db_engine, create_session_factory
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.jobs import create_task_registry
from proofpath.observability.tracing import shutdown_tracing
from proofpath.workers.queue import JobQueue
from proofpath.workers.worker import SqlAlchemyJobStore, Worker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


async def _run_worker(
    *,
    queue: JobQueue,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver | None,
) -> None:
    registry = create_task_registry(
        settings=settings,
        session_factory=session_factory,
        checkpointer=checkpointer,
    )
    worker = Worker(
        queue=queue,
        store=SqlAlchemyJobStore(session_factory),
        registry=registry,
        backoff_schedule=settings.job_retry_backoff_seconds,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows lacks add_signal_handler; suppress rather than crash on startup.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)

    logger.info(
        "Starting ProofPath worker: stream=%s group=%s consumer=%s tasks=%s",
        queue.stream,
        queue.group,
        queue.consumer,
        ", ".join(sorted(registry.known_types())),
    )
    try:
        await worker.run(stop_event=stop_event)
    finally:
        logger.info("ProofPath worker stopped.")


async def main() -> None:
    """Bootstrap and run the ProofPath background worker until stopped."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.value,
        format="%(asctime)s %(levelname)-5.5s [%(name)s] %(message)s",
    )

    queue = JobQueue.from_url(str(settings.redis_url))
    engine = create_db_engine(settings.asyncpg_database_url)
    session_factory = create_session_factory(engine)
    try:
        # Open the checkpointer for the analysis handler. If it cannot be opened
        # (DDL unavailable, connectivity), fall back to a registry without the
        # analysis task so github_ingest / smoke-test jobs still process.
        try:
            async with postgres_checkpointer(settings=settings) as checkpointer:
                await _run_worker(
                    queue=queue,
                    settings=settings,
                    session_factory=session_factory,
                    checkpointer=checkpointer,
                )
        except Exception:
            logger.exception(
                "Postgres checkpointer unavailable; starting worker without the "
                "analysis_run task (those jobs will be dead-lettered)."
            )
            await _run_worker(
                queue=queue,
                settings=settings,
                session_factory=session_factory,
                checkpointer=None,
            )
    finally:
        # Flush/release the tracing backend before closing infrastructure.
        await shutdown_tracing()
        await engine.dispose()
        await queue.close()
        logger.info("ProofPath worker shut down.")


if __name__ == "__main__":
    asyncio.run(main())
