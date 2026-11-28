"""ProofPath background worker entrypoint.

Run with ``python -m proofpath.workers`` (the ``worker`` Docker Compose service
does exactly this). It wires Redis (queue), PostgreSQL (job ledger), and the
task registry, then runs the consume loop until SIGINT/SIGTERM.

Sub-slice 2.2.1 registers the built-in ``noop`` / ``echo`` smoke-test tasks;
Sub-slice 2.2.2 will register the real ``github_ingest`` and ``analysis_run``
handlers here (or via a factory) without changing this bootstrap.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from proofpath.config import get_settings
from proofpath.db.session import create_db_engine, create_session_factory
from proofpath.workers.queue import JobQueue
from proofpath.workers.tasks import default_registry
from proofpath.workers.worker import SqlAlchemyJobStore, Worker

logger = logging.getLogger(__name__)


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
    store = SqlAlchemyJobStore(session_factory)
    registry = default_registry()
    worker = Worker(queue=queue, store=store, registry=registry)

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
        await engine.dispose()
        await queue.close()
        logger.info("ProofPath worker shut down.")


if __name__ == "__main__":
    asyncio.run(main())
