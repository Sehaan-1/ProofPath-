"""Async job payloads, enqueue helper, and worker task handlers (Sub-slice 2.2.2).

This is the service layer that bridges the API (producer) and the worker
(consumer) introduced in Sub-slice 2.2.1:

* **Payload contracts** (`GithubIngestJobPayload`, `AnalysisRunJobPayload`) are
  Pydantic models dumped to JSON-mode dicts so a job id, UUID, and datetime
  survive a Redis round-trip and reload identically in the handler.
* :func:`enqueue_job` durably creates the `background_jobs` ledger row (committed)
  and only then appends the Redis stream message, marking the row
  ``dead_letter`` if the enqueue itself fails so no phantom ``queued`` row is left.
* :func:`execute_analysis_graph` is the single graph-execution routine shared by
  the synchronous ``POST /analyses/run`` and the async ``analysis_run`` handler,
  so the two paths can never diverge in verdict persistence or metric recording.
* :func:`create_task_registry` registers the real ``github_ingest`` and
  ``analysis_run`` handlers (plus the 2.2.1 smoke-test tasks) on the worker,
  closing over the settings, session factory, and (for analysis) the checkpointer.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Self

from pydantic import AwareDatetime, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proofpath.config import LLMProvider, Settings
from proofpath.db.models import AnalysisRunStatus, BackgroundJob
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.background_job import BackgroundJobRepo
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.graph.approval import approval_request_from_interrupts
from proofpath.graph.checkpointer import thread_config
from proofpath.graph.dependencies import (
    GraphDependencies,
    GraphDependenciesError,
    create_graph_dependencies,
)
from proofpath.graph.state import initial_state
from proofpath.graph.workflow import build_workflow
from proofpath.ingestion.github_ingester import (
    GitHubEvidenceClient,
    GitHubIngestError,
    GitHubRepoRef,
    snapshot_to_document,
)
from proofpath.ingestion.markdown_ingester import persist_parsed_document
from proofpath.ingestion.resume_parser import ParsedDocument
from proofpath.retrieval.embeddings import create_embedding_client
from proofpath.schemas.base import StrictModel
from proofpath.workers.queue import JobQueue
from proofpath.workers.tasks import (
    JobContext,
    JobResult,
    TaskHandler,
    TaskRegistry,
    default_registry,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Job type identifiers registered on the worker and carried in stream messages.
GITHUB_INGEST_TASK = "github_ingest"
ANALYSIS_RUN_TASK = "analysis_run"

#: Default max attempts when the caller does not override; the worker bootstrap
#: passes ``settings.job_max_attempts`` so the policy is centrally configured.
_DEFAULT_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Payload contracts
# ---------------------------------------------------------------------------


class GithubIngestJobPayload(StrictModel):
    """Payload for an async GitHub repo ingestion & batch embedding job."""

    owner: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)


class AnalysisRunJobPayload(StrictModel):
    """Payload for an async full opportunity analysis job.

    The ``analysis_run_id`` / ``thread_id`` are created by the worker when it
    picks the job up (not by the API), so a queued job carries only the request
    inputs. The handler records the created run id in the job ``result``.
    """

    opportunity_id: uuid.UUID
    calendar_start: AwareDatetime | None = None
    calendar_end: AwareDatetime | None = None

    @model_validator(mode="after")
    def _validate_calendar(self) -> Self:
        if (self.calendar_start is None) != (self.calendar_end is None):
            raise ValueError("Provide both calendar_start and calendar_end.")
        if (
            self.calendar_start is not None
            and self.calendar_end is not None
            and self.calendar_end <= self.calendar_start
        ):
            raise ValueError("calendar_end must be after calendar_start.")
        return self


# ---------------------------------------------------------------------------
# Enqueue helper (producer side)
# ---------------------------------------------------------------------------


async def enqueue_job(
    *,
    queue: JobQueue,
    session_factory: async_sessionmaker[AsyncSession],
    job_type: str,
    payload_model: StrictModel,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> BackgroundJob:
    """Durably enqueue a background job: commit the ledger row, then append to Redis.

    The row is committed *before* the Redis ``XADD`` so a worker can never receive
    a message whose status row is still uncommitted (which would make it skip the
    job and orphan the row). If the Redis append fails, the row is moved to
    ``dead_letter`` so it is visible and replayable rather than stuck as a phantom
    ``queued`` entry; the caller surfaces the failure as a 503.
    """
    payload = payload_model.model_dump(mode="json")
    async with session_factory.begin() as session:
        job = await BackgroundJobRepo(session).create(
            job_type=job_type,
            payload=payload,
            max_attempts=max_attempts,
        )
    try:
        await queue.enqueue(job.id, job_type=job_type, payload=payload)
    except Exception:
        async with session_factory.begin() as session:
            await BackgroundJobRepo(session).mark_dead_letter(
                job.id, error="Redis enqueue failed", traceback=None
            )
        raise
    return job


# ---------------------------------------------------------------------------
# Shared analysis graph execution (sync endpoint + async handler)
# ---------------------------------------------------------------------------


async def execute_analysis_graph(
    *,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver,
    dependencies: GraphDependencies,
    opportunity_id: uuid.UUID,
    run_id: uuid.UUID,
    thread_id: str,
    calendar_start: AwareDatetime | None = None,
    calendar_end: AwareDatetime | None = None,
) -> AnalysisRunStatus:
    """Run one LangGraph analysis run to a terminal state and persist the outcome.

    Shared by the synchronous API path and the ``analysis_run`` worker handler so
    metric recording and status derivation stay identical. The graph validates and
    persists verdicts atomically with the pending drafts, *before* its interrupt.
    The caller builds (and validates) ``dependencies`` so wiring failures surface
    before any run row is created. On an unhandled exception the run is marked
    ``failed`` and the exception re-raises (the API maps that to 503; the handler
    treats a recorded failure as job success to avoid duplicate runs on retry).
    """
    started = time.monotonic()
    try:
        graph = build_workflow(
            checkpointer=checkpointer,
            session_factory=session_factory,
            settings=settings,
            analysis_run_id=run_id,
            thread_id=thread_id,
            parser=dependencies.parser,
            embedder=dependencies.embedder,
            comparator=dependencies.comparator,
            sprint_planner=dependencies.sprint_planner,
            calendar_start=calendar_start,
            calendar_end=calendar_end,
        )
        state = await graph.ainvoke(
            initial_state(opportunity_id, settings=settings), thread_config(thread_id)
        )
        errors = state.get("errors", [])
        verified = state.get("output_validated") is True and state.get("policy_verified") is True
        interrupts = state.get("__interrupt__") or ()
        paused_for_approval = approval_request_from_interrupts(interrupts) is not None and bool(
            state.get("draft_actions")
        )
        if errors or not verified or (interrupts and not paused_for_approval):
            status = AnalysisRunStatus.FAILED
        elif paused_for_approval:
            status = AnalysisRunStatus.AWAITING_APPROVAL
        else:
            status = AnalysisRunStatus.COMPLETED
        async with session_factory.begin() as session:
            repo = AnalysisRepo(session)
            await repo.update_run_metrics(
                run_id, status=status, latency_ms=int((time.monotonic() - started) * 1000)
            )
        return status
    except asyncio.CancelledError:
        async with session_factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                run_id,
                status=AnalysisRunStatus.FAILED,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        raise
    except Exception:
        async with session_factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                run_id,
                status=AnalysisRunStatus.FAILED,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        raise


# ---------------------------------------------------------------------------
# Task handlers (consumer side)
# ---------------------------------------------------------------------------


async def _persist_and_embed(
    document: ParsedDocument,
    settings: Settings,
    session: AsyncSession,
) -> tuple[uuid.UUID, int, str, str, str]:
    """Insert/replace a parsed document and embed any new chunks in one transaction.

    Returns ``(source_id, chunk_count, content_sha256, source_type, canonical_uri)``.
    Raises ``ValueError`` on a URI/content conflict (a permanent error for the job).
    """
    repo = EvidenceRepo(session)
    result = await persist_parsed_document(repo, document, replace_existing=False)
    chunks = await repo.list_chunks_for_source(result.source_id)
    unembedded = [chunk for chunk in chunks if chunk.embedding is None]
    if unembedded:
        embedder = create_embedding_client(settings)
        vectors = await embedder.embed([chunk.content for chunk in unembedded])
        for chunk, vector in zip(unembedded, vectors, strict=True):
            chunk.embedding = vector
        await session.flush()
    source = await repo.get_source_by_id(result.source_id)
    if source is None:  # pragma: no cover - defensive
        raise RuntimeError("Ingested evidence source disappeared")
    return (
        result.source_id,
        result.chunk_count,
        result.content_sha256,
        source.source_type.value,
        source.canonical_uri,
    )


def _make_github_ingest_handler(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> TaskHandler:
    """Build the ``github_ingest`` task handler closing over its dependencies."""

    async def handler(ctx: JobContext) -> JobResult:
        payload = GithubIngestJobPayload(**ctx.payload)
        ref = GitHubRepoRef(owner=payload.owner, name=payload.name)
        token = settings.github_token.get_secret_value() if settings.github_token else None
        try:
            async with GitHubEvidenceClient(token=token) as client:
                snapshot = await client.fetch_snapshot(ref)
        except GitHubIngestError as exc:
            # 404/422 = bad repo (permanent); 429/5xx/network = retryable.
            transient = exc.status_code is None or exc.status_code >= 500 or exc.status_code == 429
            return JobResult(ok=False, error=f"github fetch failed: {exc}", transient=transient)
        document = snapshot_to_document(snapshot)
        try:
            async with session_factory.begin() as session:
                source_id, chunk_count, sha, source_type, canonical_uri = await _persist_and_embed(
                    document, settings, session
                )
        except ValueError as exc:
            # URI/content conflict or bad content — retrying will not help.
            return JobResult(ok=False, error=f"ingest conflict: {exc}", transient=False)
        return JobResult(
            ok=True,
            result={
                "source_id": str(source_id),
                "chunk_count": chunk_count,
                "content_sha256": sha,
                "source_type": source_type,
                "canonical_uri": canonical_uri,
            },
        )

    return handler


def _make_analysis_run_handler(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver,
) -> TaskHandler:
    """Build the ``analysis_run`` task handler closing over its dependencies."""

    async def handler(ctx: JobContext) -> JobResult:
        payload = AnalysisRunJobPayload(**ctx.payload)
        # A missing opportunity is a permanent (input) error, not a retry.
        async with session_factory() as session:
            opportunity = await OpportunityRepo(session).get_by_id(payload.opportunity_id)
        if opportunity is None:
            return JobResult(ok=False, error="opportunity not found", transient=False)
        # Unconfigured model wiring is permanent; retrying cannot fix it.
        if settings.llm_provider is not LLMProvider.OPENAI:
            return JobResult(
                ok=False, error="analysis requires the openai provider", transient=False
            )
        try:
            dependencies = create_graph_dependencies(settings)
        except GraphDependenciesError:
            return JobResult(ok=False, error="analysis model is not configured", transient=False)
        thread_id = str(uuid.uuid4())
        run_id = uuid.uuid4()
        async with session_factory.begin() as session:
            await AnalysisRepo(session).create_run(
                opportunity_id=payload.opportunity_id,
                thread_id=thread_id,
                run_id=run_id,
            )
        try:
            status = await execute_analysis_graph(
                settings=settings,
                session_factory=session_factory,
                checkpointer=checkpointer,
                dependencies=dependencies,
                opportunity_id=payload.opportunity_id,
                run_id=run_id,
                thread_id=thread_id,
                calendar_start=payload.calendar_start,
                calendar_end=payload.calendar_end,
            )
        except Exception:
            # The run was already marked failed by execute_analysis_graph; the job
            # fulfilled its purpose (the analysis reached a recorded terminal state),
            # so report success with status=failed rather than retrying into a
            # duplicate run.
            return JobResult(
                ok=True,
                result={
                    "analysis_run_id": str(run_id),
                    "thread_id": thread_id,
                    "status": AnalysisRunStatus.FAILED.value,
                },
            )
        return JobResult(
            ok=True,
            result={
                "analysis_run_id": str(run_id),
                "thread_id": thread_id,
                "status": status.value,
            },
        )

    return handler


def create_task_registry(
    *,
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: AsyncPostgresSaver | None = None,
) -> TaskRegistry:
    """Register the real async-work handlers on top of the smoke-test tasks.

    The ``analysis_run`` handler needs a live Postgres checkpointer; when none is
    provided (e.g. the worker could not open one) it is simply not registered, so
    an enqueued analysis job is dead-lettered as an unknown task type rather than
    crashing the worker.
    """
    registry = default_registry()
    registry.register(GITHUB_INGEST_TASK, _make_github_ingest_handler(settings, session_factory))
    if checkpointer is not None:
        registry.register(
            ANALYSIS_RUN_TASK,
            _make_analysis_run_handler(settings, session_factory, checkpointer),
        )
    return registry


__all__ = [
    "ANALYSIS_RUN_TASK",
    "GITHUB_INGEST_TASK",
    "AnalysisRunJobPayload",
    "GithubIngestJobPayload",
    "create_task_registry",
    "enqueue_job",
    "execute_analysis_graph",
]
