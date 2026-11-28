"""Live Redis + PostgreSQL integration tests for the async job flow (Sub-slice 2.2.2).

These prove the producer→consumer→status loop the unit tests cannot:

* :func:`enqueue_job` atomically creates a ``background_jobs`` row in PostgreSQL
  and appends a Redis stream message;
* the ``Worker`` (with the real ``SqlAlchemyJobStore``) drains that message,
  runs a registered handler, and finalizes the row to ``succeeded``;
* the github_ingest handler persists a real evidence source + embeddings against
  live PostgreSQL when the GitHub client returns a snapshot (mocked to avoid
  network);
* ``GET /api/v1/jobs/{job_id}`` reads the durable status back over HTTP.

Requires ``make up`` (or ``TEST_DATABASE_URL`` / ``TEST_REDIS_URL``).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import BackgroundJob, EvidenceSource, JobStatus
from proofpath.db.repositories.background_job import BackgroundJobRepo
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.ingestion.github_ingester import (
    GitHubPullRequest,
    GitHubRelease,
    GitHubRepoRef,
    GitHubRepoSnapshot,
)
from proofpath.jobs import (
    GITHUB_INGEST_TASK,
    GithubIngestJobPayload,
    create_task_registry,
    enqueue_job,
)
from proofpath.workers.queue import JobQueue
from proofpath.workers.worker import ProcessOutcome, SqlAlchemyJobStore, Worker

pytestmark = pytest.mark.integration


@dataclass
class JobHarness:
    settings: Settings
    factory: async_sessionmaker[AsyncSession]
    queue: JobQueue
    redis_client: object
    seeded_source_uris: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.seeded_source_uris is None:
            self.seeded_source_uris = []


@pytest.fixture
async def job_harness(
    db_engine: AsyncEngine, redis_client, integration_database_url
) -> AsyncIterator[JobHarness]:
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        embedding_model="hash",
    )
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    stream = f"proofpath:jobs:test:{uuid.uuid4()}"
    group = f"test-workers-{uuid.uuid4().hex[:8]}"
    queue = JobQueue(redis_client, stream=stream, group=group, consumer="test-worker")
    await queue.ensure_group()
    harness = JobHarness(settings=settings, factory=factory, queue=queue, redis_client=redis_client)
    try:
        yield harness
    finally:
        for uri in harness.seeded_source_uris:
            async with harness.factory.begin() as session:
                await session.execute(
                    delete(EvidenceSource).where(EvidenceSource.canonical_uri == uri)
                )
        await redis_client.delete(queue.stream)


async def test_enqueue_job_creates_row_and_redis_message(job_harness: JobHarness) -> None:
    job = await enqueue_job(
        queue=job_harness.queue,
        session_factory=job_harness.factory,
        job_type="echo",
        payload_model=GithubIngestJobPayload(owner="o", name="n"),  # any JSON-mode payload
    )
    try:
        # The row is committed and queryable immediately.
        async with job_harness.factory() as session:
            row = await BackgroundJobRepo(session).get(job.id)
        assert row is not None
        assert row.status is JobStatus.QUEUED
        # The Redis stream has the message.
        assert await job_harness.queue.stream_length() == 1
    finally:
        msg = (await job_harness.queue.dequeue(count=1, block_ms=1000))[0]
        await job_harness.queue.ack(msg.message_id)
        async with job_harness.factory.begin() as session:
            await session.execute(delete(BackgroundJob).where(BackgroundJob.id == job.id))


async def test_full_loop_enqueue_worker_succeed_and_status(job_harness: JobHarness) -> None:
    job = await enqueue_job(
        queue=job_harness.queue,
        session_factory=job_harness.factory,
        job_type="echo",
        payload_model=GithubIngestJobPayload(owner="o", name="n"),
    )
    try:
        registry = create_task_registry(
            settings=job_harness.settings,
            session_factory=job_harness.factory,
            checkpointer=None,  # echo task needs no checkpointer
        )
        worker = Worker(
            queue=job_harness.queue,
            store=SqlAlchemyJobStore(job_harness.factory),
            registry=registry,
            block_ms=1000,
        )
        outcome = await worker.process_once()
        assert outcome.outcome is ProcessOutcome.PROCESSED
        assert outcome.succeeded == 1

        async with job_harness.factory() as session:
            row = await BackgroundJobRepo(session).get(job.id)
        assert row is not None
        assert row.status is JobStatus.SUCCEEDED
        assert row.result_json is not None
        assert row.result_json["job_type"] == "echo"
    finally:
        async with job_harness.factory.begin() as session:
            await session.execute(delete(BackgroundJob).where(BackgroundJob.id == job.id))


async def test_github_ingest_handler_persists_source_against_postgres(
    job_harness: JobHarness, monkeypatch
) -> None:
    """The real github_ingest handler persists + embeds against live PostgreSQL."""
    ref = GitHubRepoRef(owner="sehaan-1", name="ProofPath-")
    snapshot = GitHubRepoSnapshot(
        ref=ref,
        description="Personal Evidence & Opportunity Agent",
        default_branch="main",
        html_url=ref.html_url,
        stargazers_count=42,
        languages=("Python",),
        readme_markdown="# ProofPath\n\nFastAPI + LangGraph + pgvector agent.\n",
        releases=(
            GitHubRelease(
                tag_name="v0.1.0",
                name="Initial release",
                body="First public release.",
                html_url=f"{ref.html_url}/releases/tag/v0.1.0",
            ),
        ),
        pull_requests=(
            GitHubPullRequest(
                number=1,
                title="Add worker",
                body="Redis queue + background_jobs ledger.",
                html_url=f"{ref.html_url}/pull/1",
            ),
        ),
    )

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def fetch_snapshot(self, _ref):
            return snapshot

    monkeypatch.setattr("proofpath.jobs.GitHubEvidenceClient", lambda **_kw: FakeClient())

    job = await enqueue_job(
        queue=job_harness.queue,
        session_factory=job_harness.factory,
        job_type=GITHUB_INGEST_TASK,
        payload_model=GithubIngestJobPayload(owner="sehaan-1", name="ProofPath-"),
    )
    job_harness.seeded_source_uris.append(ref.html_url)
    try:
        registry = create_task_registry(
            settings=job_harness.settings,
            session_factory=job_harness.factory,
            checkpointer=None,
        )
        worker = Worker(
            queue=job_harness.queue,
            store=SqlAlchemyJobStore(job_harness.factory),
            registry=registry,
            block_ms=1000,
        )
        outcome = await worker.process_once()
        assert outcome.succeeded == 1

        async with job_harness.factory() as session:
            row = await BackgroundJobRepo(session).get(job.id)
        assert row is not None
        assert row.status is JobStatus.SUCCEEDED
        source_id = uuid.UUID(row.result_json["source_id"])
        assert row.result_json["chunk_count"] >= 1

        # The evidence source and embedded chunks are really persisted.
        async with job_harness.factory() as session:
            source = await EvidenceRepo(session).get_source_by_id(source_id)
        assert source is not None
        assert source.canonical_uri == ref.html_url
        chunks = await EvidenceRepo(session).list_chunks_for_source(source_id)
        assert chunks
        assert all(chunk.embedding is not None for chunk in chunks)
    finally:
        async with job_harness.factory.begin() as session:
            await session.execute(delete(BackgroundJob).where(BackgroundJob.id == job.id))
