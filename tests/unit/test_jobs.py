"""Unit tests for the async job service layer (`proofpath.jobs`)."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from proofpath import jobs
from proofpath.db.models import AnalysisRunStatus, JobStatus
from proofpath.jobs import (
    ANALYSIS_RUN_TASK,
    GITHUB_INGEST_TASK,
    AnalysisRunJobPayload,
    GithubIngestJobPayload,
    create_task_registry,
    enqueue_job,
    execute_analysis_graph,
)
from proofpath.workers.tasks import JobContext

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Payload contracts
# ---------------------------------------------------------------------------


def test_github_ingest_payload_round_trips_through_json_mode() -> None:
    payload = GithubIngestJobPayload(owner="sehaan-1", name="ProofPath-")
    dumped = payload.model_dump(mode="json")
    assert dumped == {"owner": "sehaan-1", "name": "ProofPath-"}
    # Reloads identically (simulating a Redis -> handler round-trip).
    assert GithubIngestJobPayload(**dumped) == payload


def test_analysis_run_payload_validates_calendar_pair() -> None:
    opp_id = uuid.uuid4()
    start = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
    end = datetime(2026, 12, 1, 10, 0, tzinfo=UTC)
    payload = AnalysisRunJobPayload(opportunity_id=opp_id, calendar_start=start, calendar_end=end)
    dumped = payload.model_dump(mode="json")
    reloaded = AnalysisRunJobPayload(**dumped)
    assert reloaded.opportunity_id == opp_id
    assert reloaded.calendar_start == start
    assert reloaded.calendar_end == end


def test_analysis_run_payload_rejects_partial_calendar() -> None:
    with pytest.raises(ValueError, match="calendar"):
        AnalysisRunJobPayload(
            opportunity_id=uuid.uuid4(),
            calendar_start=datetime(2026, 12, 1, 9, 0, tzinfo=UTC),
        )


def test_analysis_run_payload_rejects_reversed_calendar() -> None:
    start = datetime(2026, 12, 1, 10, 0, tzinfo=UTC)
    end = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="after"):
        AnalysisRunJobPayload(opportunity_id=uuid.uuid4(), calendar_start=start, calendar_end=end)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


def _make_factory() -> MagicMock:
    """A session factory whose `.begin()` yields a throwaway session."""

    factory = MagicMock()

    @asynccontextmanager
    async def begin():
        yield MagicMock()

    factory.begin = begin
    return factory


class _FakeJobRepo:
    def __init__(self) -> None:
        self.created: list[SimpleNamespace] = []
        self.dead_lettered: list[uuid.UUID] = []

    async def create(self, *, job_type, payload, max_attempts, job_id=None) -> SimpleNamespace:
        job = SimpleNamespace(
            id=job_id or uuid.uuid4(),
            job_type=job_type,
            payload_json=dict(payload),
            status=JobStatus.QUEUED,
            attempt=0,
            max_attempts=max_attempts,
        )
        self.created.append(job)
        return job

    async def mark_dead_letter(self, job_id, *, error, traceback=None) -> None:
        self.dead_lettered.append(job_id)


# ---------------------------------------------------------------------------
# enqueue_job
# ---------------------------------------------------------------------------


async def test_enqueue_job_creates_row_then_enqueues(monkeypatch) -> None:
    repo = _FakeJobRepo()
    monkeypatch.setattr(jobs, "BackgroundJobRepo", lambda _session: repo)
    queue = MagicMock()
    queue.enqueue = AsyncMock(return_value="1-0")
    factory = _make_factory()

    job = await enqueue_job(
        queue=queue,
        session_factory=factory,
        job_type=GITHUB_INGEST_TASK,
        payload_model=GithubIngestJobPayload(owner="o", name="n"),
    )

    assert job.status is JobStatus.QUEUED
    assert len(repo.created) == 1
    queue.enqueue.assert_awaited_once()
    enqueued = queue.enqueue.await_args
    assert enqueued.kwargs["job_type"] == GITHUB_INGEST_TASK
    assert enqueued.kwargs["payload"] == {"owner": "o", "name": "n"}
    assert enqueued.args[0] == job.id
    assert repo.dead_lettered == []


async def test_enqueue_job_dead_letters_on_redis_failure(monkeypatch) -> None:
    repo = _FakeJobRepo()
    monkeypatch.setattr(jobs, "BackgroundJobRepo", lambda _session: repo)
    queue = MagicMock()
    queue.enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    factory = _make_factory()

    with pytest.raises(RuntimeError, match="redis down"):
        await enqueue_job(
            queue=queue,
            session_factory=factory,
            job_type=ANALYSIS_RUN_TASK,
            payload_model=AnalysisRunJobPayload(opportunity_id=uuid.uuid4()),
        )

    assert len(repo.created) == 1
    # Row was committed, then enqueue failed -> dead-lettered (not stuck as queued).
    assert repo.dead_lettered == [repo.created[0].id]


# ---------------------------------------------------------------------------
# execute_analysis_graph
# ---------------------------------------------------------------------------


def _fake_dependencies() -> object:
    return SimpleNamespace(
        parser=object(), embedder=object(), comparator=object(), sprint_planner=object()
    )


@pytest.mark.parametrize(
    ("validated", "expected"),
    [(True, AnalysisRunStatus.COMPLETED), (False, AnalysisRunStatus.FAILED)],
)
async def test_execute_analysis_graph_requires_final_contract_not_only_policy(
    monkeypatch, validated, expected
) -> None:
    run_id = uuid.uuid4()
    opp_id = uuid.uuid4()
    runs = {run_id: SimpleNamespace(status=AnalysisRunStatus.RUNNING)}

    class Repo:
        # Verdicts now commit in the graph's validated draft transaction.
        async def update_run_metrics(self, _run_id, *, status, latency_ms):
            runs[_run_id].status = status
            runs[_run_id].latency_ms = latency_ms

    monkeypatch.setattr(jobs, "AnalysisRepo", lambda _session: Repo())
    monkeypatch.setattr(
        jobs,
        "build_workflow",
        lambda **_kw: SimpleNamespace(
            ainvoke=AsyncMock(
                return_value={
                    "policy_verified": True,
                    "errors": [],
                    "draft_actions": [],
                    "__interrupt__": (),
                    "output_validated": validated,
                }
            )
        ),
    )

    factory = _make_factory()
    status = await execute_analysis_graph(
        settings=MagicMock(),
        session_factory=factory,
        checkpointer=MagicMock(),
        dependencies=_fake_dependencies(),
        opportunity_id=opp_id,
        run_id=run_id,
        thread_id="t-1",
    )
    assert status is expected
    assert runs[run_id].status is expected


async def test_execute_analysis_graph_marks_failed_on_exception(monkeypatch) -> None:
    run_id = uuid.uuid4()
    state = {"status": AnalysisRunStatus.RUNNING}

    class Repo:
        async def upsert_verdicts(self, *_a, **_kw):
            pass

        async def update_run_metrics(self, _run_id, *, status, latency_ms):
            state["status"] = status

    monkeypatch.setattr(jobs, "AnalysisRepo", lambda _session: Repo())
    monkeypatch.setattr(
        jobs,
        "build_workflow",
        lambda **_kw: SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom"))),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await execute_analysis_graph(
            settings=MagicMock(),
            session_factory=_make_factory(),
            checkpointer=MagicMock(),
            dependencies=_fake_dependencies(),
            opportunity_id=uuid.uuid4(),
            run_id=run_id,
            thread_id="t-1",
        )
    assert state["status"] is AnalysisRunStatus.FAILED


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------


def test_create_task_registry_registers_github_and_analysis() -> None:
    registry = create_task_registry(
        settings=MagicMock(),
        session_factory=MagicMock(),
        checkpointer=MagicMock(),
    )
    assert GITHUB_INGEST_TASK in registry
    assert ANALYSIS_RUN_TASK in registry
    assert "noop" in registry  # smoke-test tasks retained


def test_create_task_registry_without_checkpointer_omits_analysis() -> None:
    registry = create_task_registry(
        settings=MagicMock(),
        session_factory=MagicMock(),
        checkpointer=None,
    )
    assert GITHUB_INGEST_TASK in registry
    assert ANALYSIS_RUN_TASK not in registry


async def test_github_ingest_handler_success(monkeypatch) -> None:
    settings = MagicMock()
    settings.github_token = None

    snapshot = object()
    document = MagicMock()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def fetch_snapshot(self, _ref):
            return snapshot

    monkeypatch.setattr(jobs, "GitHubEvidenceClient", lambda **_kw: FakeClient())
    monkeypatch.setattr(jobs, "snapshot_to_document", lambda _snap: document)

    captured: dict = {}

    async def fake_persist_and_embed(doc, _settings, _session):
        captured["doc"] = doc
        return uuid.uuid4(), 3, "a" * 64, "github_repo", "https://github.com/o/n"

    monkeypatch.setattr(jobs, "_persist_and_embed", fake_persist_and_embed)

    factory = _make_factory()
    handler = jobs._make_github_ingest_handler(settings, factory)
    result = await handler(
        JobContext(
            job_id=uuid.uuid4(),
            job_type=GITHUB_INGEST_TASK,
            payload={"owner": "o", "name": "n"},
            attempt=1,
        )
    )
    assert result.ok is True
    assert result.result["chunk_count"] == 3
    assert captured["doc"] is document


async def test_github_ingest_handler_transient_on_5xx(monkeypatch) -> None:
    from proofpath.ingestion.github_ingester import GitHubIngestError

    settings = MagicMock()
    settings.github_token = None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def fetch_snapshot(self, _ref):
            raise GitHubIngestError("server error", status_code=503)

    monkeypatch.setattr(jobs, "GitHubEvidenceClient", lambda **_kw: FakeClient())
    handler = jobs._make_github_ingest_handler(settings, _make_factory())
    result = await handler(
        JobContext(
            job_id=uuid.uuid4(),
            job_type=GITHUB_INGEST_TASK,
            payload={"owner": "o", "name": "n"},
            attempt=1,
        )
    )
    assert result.ok is False
    assert result.transient is True


async def test_github_ingest_handler_permanent_on_404(monkeypatch) -> None:
    from proofpath.ingestion.github_ingester import GitHubIngestError

    settings = MagicMock()
    settings.github_token = None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def fetch_snapshot(self, _ref):
            raise GitHubIngestError("not found", status_code=404)

    monkeypatch.setattr(jobs, "GitHubEvidenceClient", lambda **_kw: FakeClient())
    handler = jobs._make_github_ingest_handler(settings, _make_factory())
    result = await handler(
        JobContext(
            job_id=uuid.uuid4(),
            job_type=GITHUB_INGEST_TASK,
            payload={"owner": "o", "name": "n"},
            attempt=1,
        )
    )
    assert result.ok is False
    assert result.transient is False


async def test_analysis_run_handler_missing_opportunity_is_permanent(monkeypatch) -> None:
    settings = MagicMock()
    settings.llm_provider = MagicMock()
    settings.llm_provider.value = "openai"

    class OppRepo:
        async def get_by_id(self, _opp_id):
            return None

    monkeypatch.setattr(jobs, "OpportunityRepo", lambda _session: OppRepo())
    handler = jobs._make_analysis_run_handler(settings, _make_factory(), MagicMock())
    result = await handler(
        JobContext(
            job_id=uuid.uuid4(),
            job_type=ANALYSIS_RUN_TASK,
            payload={"opportunity_id": str(uuid.uuid4())},
            attempt=1,
        )
    )
    assert result.ok is False
    assert result.transient is False
    assert "opportunity" in (result.error or "")


async def test_analysis_run_handler_success_records_run_id(monkeypatch) -> None:
    from proofpath.config import LLMProvider

    settings = MagicMock()
    settings.llm_provider = LLMProvider.OPENAI
    opp_id = uuid.uuid4()

    class OppRepo:
        async def get_by_id(self, _opp_id):
            return SimpleNamespace(id=opp_id)

    class RunRepo:
        async def create_run(self, *, opportunity_id, thread_id, run_id):
            return SimpleNamespace(id=run_id, opportunity_id=opportunity_id, thread_id=thread_id)

    monkeypatch.setattr(jobs, "OpportunityRepo", lambda _session: OppRepo())
    monkeypatch.setattr(jobs, "AnalysisRepo", lambda _session: RunRepo())
    monkeypatch.setattr(jobs, "create_graph_dependencies", lambda _settings: _fake_dependencies())
    monkeypatch.setattr(
        jobs,
        "execute_analysis_graph",
        AsyncMock(return_value=AnalysisRunStatus.AWAITING_APPROVAL),
    )

    handler = jobs._make_analysis_run_handler(settings, _make_factory(), MagicMock())
    result = await handler(
        JobContext(
            job_id=uuid.uuid4(),
            job_type=ANALYSIS_RUN_TASK,
            payload={"opportunity_id": str(opp_id)},
            attempt=1,
        )
    )
    assert result.ok is True
    assert result.result["status"] == "awaiting_approval"
    assert "analysis_run_id" in result.result
    assert "thread_id" in result.result


async def test_analysis_run_handler_graph_exception_is_job_success_with_failed(monkeypatch) -> None:
    from proofpath.config import LLMProvider

    settings = MagicMock()
    settings.llm_provider = LLMProvider.OPENAI
    opp_id = uuid.uuid4()

    class OppRepo:
        async def get_by_id(self, _opp_id):
            return SimpleNamespace(id=opp_id)

    class RunRepo:
        async def create_run(self, *, opportunity_id, thread_id, run_id):
            return SimpleNamespace(id=run_id)

    monkeypatch.setattr(jobs, "OpportunityRepo", lambda _session: OppRepo())
    monkeypatch.setattr(jobs, "AnalysisRepo", lambda _session: RunRepo())
    monkeypatch.setattr(jobs, "create_graph_dependencies", lambda _settings: _fake_dependencies())
    monkeypatch.setattr(
        jobs,
        "execute_analysis_graph",
        AsyncMock(side_effect=RuntimeError("graph blew up")),
    )

    handler = jobs._make_analysis_run_handler(settings, _make_factory(), MagicMock())
    result = await handler(
        JobContext(
            job_id=uuid.uuid4(),
            job_type=ANALYSIS_RUN_TASK,
            payload={"opportunity_id": str(opp_id)},
            attempt=1,
        )
    )
    # The run was already marked failed inside execute_analysis_graph; the job
    # reports success (no retry -> no duplicate run) with status=failed.
    assert result.ok is True
    assert result.result["status"] == "failed"
