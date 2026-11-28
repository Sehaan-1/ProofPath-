"""Unit tests for single-backend observability wiring (Sub-slice 2.4.1).

Covers the backend choice (2.4.1a: Langfuse only, fail-closed otherwise) and the
unified trace ID attachment (2.4.1b). Real-SDK tests run the actual Langfuse v3
OTel pipeline against a loopback OTLP capture server and decode the exported
protobuf — no mock-only tracing. They skip unless ``proofpath[observability]``
is installed, mirroring the guardrails-extra convention.
"""

from __future__ import annotations

import builtins
import gzip
import json
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from typing import Any, TypedDict
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from proofpath.config import Settings
from proofpath.db.models import ActionDraftStatus, ActionType, ApplicationStage
from proofpath.graph import workflow
from proofpath.graph.checkpointer import thread_config
from proofpath.graph.state import initial_state
from proofpath.observability import langfuse_tracing
from proofpath.observability.tracing import (
    ANALYSIS_TRACE_NAME,
    NullRunTracer,
    TracingUnavailableError,
    analysis_trace_id,
    create_run_tracer,
    shutdown_tracing,
)
from proofpath.schemas.actions import ActionDraftRecord, TrackerUpsertPayload
from proofpath.schemas.evidence import EvidenceSnippet

pytestmark = pytest.mark.unit

_HAS_SDK = find_spec("langfuse") is not None and find_spec("langchain") is not None
requires_sdk = pytest.mark.skipif(
    not _HAS_SDK, reason="Install .[observability] to test the real SDK"
)


@pytest.fixture(autouse=True)
def _reset_langfuse_cache() -> Iterator[None]:
    """Isolate the SDK client cache and the global OTel provider between tests.

    The Langfuse SDK installs its TracerProvider as the process-global OTel
    provider once and reuses it for later clients; without restoring the unset
    state, shut-down processors from earlier tests would accumulate on the
    shared provider and retry exports against dead ports. Production runs one
    fixed backend config per process, so this is purely test isolation.
    """
    yield
    langfuse_tracing.shutdown_langfuse()
    if _HAS_SDK:
        from opentelemetry import trace as otel_trace

        if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
            otel_trace._TRACER_PROVIDER = None
            otel_trace._TRACER_PROVIDER_SET_ONCE._done = False


def _langfuse_settings(host: str = "http://127.0.0.1:9") -> Settings:
    return Settings(
        _env_file=None,
        observability_backend="langfuse",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_host=host,
    )


# ---------------------------------------------------------------------------
# Unified trace ID derivation (Sub-part 2.4.1b)
# ---------------------------------------------------------------------------


def test_analysis_trace_id_is_deterministic_and_well_formed() -> None:
    run_id = uuid4()
    trace_id = analysis_trace_id(run_id)
    assert trace_id == analysis_trace_id(run_id)
    assert len(trace_id) == 32
    assert all(char in "0123456789abcdef" for char in trace_id)
    assert int(trace_id, 16) != 0  # OTel rejects the all-zero trace ID
    assert analysis_trace_id(uuid4()) != trace_id


@requires_sdk
def test_analysis_trace_id_matches_langfuse_sdk_derivation() -> None:
    from langfuse import Langfuse

    run_id = uuid4()
    assert analysis_trace_id(run_id) == Langfuse.create_trace_id(seed=str(run_id))


# ---------------------------------------------------------------------------
# Backend choice (Sub-part 2.4.1a): exactly one backend, fail closed
# ---------------------------------------------------------------------------


async def test_none_backend_returns_null_tracer_without_touching_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode() -> Any:
        raise AssertionError("backend=none must never import the Langfuse SDK")

    monkeypatch.setattr(langfuse_tracing, "_import_sdk", explode)
    run_id = uuid4()
    tracer = create_run_tracer(Settings(_env_file=None), analysis_run_id=run_id, thread_id="thread")
    assert isinstance(tracer, NullRunTracer)
    assert tracer.trace_id == analysis_trace_id(run_id)
    with tracer.invocation() as callbacks:
        assert tuple(callbacks) == ()
    await tracer.flush()  # no-op, no I/O


def test_openllmetry_backend_fails_closed_single_backend_rule() -> None:
    settings = Settings(_env_file=None, observability_backend="openllmetry")
    with pytest.raises(TracingUnavailableError, match="not wired in the MVP"):
        create_run_tracer(settings, analysis_run_id=uuid4(), thread_id="thread")


def test_langfuse_without_extra_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse" or name.startswith("langfuse."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    settings = _langfuse_settings()
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(TracingUnavailableError, match=r"proofpath\[observability\]"):
        create_run_tracer(settings, analysis_run_id=uuid4(), thread_id="thread")


async def test_shutdown_tracing_is_noop_without_backend() -> None:
    await shutdown_tracing()
    assert langfuse_tracing._client is None


# ---------------------------------------------------------------------------
# Callback attachment at the BoundedWorkflow choke point (Sub-part 2.4.1b)
# ---------------------------------------------------------------------------


class RecordingTracer:
    """Fake RunTracer capturing invocation lifecycle and yielding one handler."""

    def __init__(self, trace_id: str = "unified-trace-id") -> None:
        self._trace_id = trace_id
        self.events: list[str] = []
        self.handler = BaseCallbackHandler()

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @contextmanager
    def invocation(self) -> Iterator[tuple[BaseCallbackHandler, ...]]:
        self.events.append("enter")
        try:
            yield (self.handler,)
        finally:
            self.events.append("exit")

    async def flush(self) -> None:
        self.events.append("flush")


def _fake_graph() -> tuple[MagicMock, dict[str, Any]]:
    graph = MagicMock()
    captured: dict[str, Any] = {}

    async def ainvoke(_input: Any, config: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(config=config, kwargs=kwargs)
        return {"ok": True}

    async def astream(_input: Any, config: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        captured.update(config=config, kwargs=kwargs)
        yield {"ok": True}

    graph.ainvoke = ainvoke
    graph.astream = astream
    return graph, captured


async def test_ainvoke_attaches_tracer_callbacks_and_flushes() -> None:
    graph, captured = _fake_graph()
    tracer = RecordingTracer()
    caller_handler = BaseCallbackHandler()
    bounded = workflow.BoundedWorkflow(graph, "thread-1", 7, tracer)

    result = await bounded.ainvoke(
        {},
        {
            **thread_config("thread-1"),
            "callbacks": [caller_handler],
            "recursion_limit": 9999,
        },
    )

    assert result == {"ok": True}
    # Caller callbacks are preserved; the tracer handler is appended.
    assert captured["config"]["callbacks"] == [caller_handler, tracer.handler]
    assert captured["config"]["run_name"] == ANALYSIS_TRACE_NAME
    # Bounds are still enforced with tracing attached.
    assert captured["config"]["recursion_limit"] == 11
    assert captured["config"]["configurable"]["thread_id"] == "thread-1"
    assert captured["kwargs"]["durability"] == "sync"
    # The invocation context is exited before spans are flushed.
    assert tracer.events == ["enter", "exit", "flush"]
    assert bounded.trace_id == "unified-trace-id"


async def test_aresume_attaches_the_same_tracer() -> None:
    graph, captured = _fake_graph()
    tracer = RecordingTracer()
    bounded = workflow.BoundedWorkflow(graph, "thread-1", 7, tracer)

    await bounded.aresume({"decision": "approved", "draft_id": "x"}, thread_config("thread-1"))

    assert captured["config"]["callbacks"] == [tracer.handler]
    assert tracer.events == ["enter", "exit", "flush"]


async def test_astream_attaches_callbacks_and_flushes() -> None:
    graph, captured = _fake_graph()
    tracer = RecordingTracer()
    bounded = workflow.BoundedWorkflow(graph, "thread-1", 7, tracer)

    events = [item async for item in bounded.astream({}, thread_config("thread-1"))]

    assert events == [{"ok": True}]
    assert captured["config"]["callbacks"] == [tracer.handler]
    assert captured["config"]["recursion_limit"] == 11
    assert captured["kwargs"]["durability"] == "sync"
    assert tracer.events == ["enter", "exit", "flush"]


async def test_flush_runs_even_when_the_graph_raises() -> None:
    graph = MagicMock()

    async def boom(_input: Any, _config: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("graph blew up")

    graph.ainvoke = boom
    tracer = RecordingTracer()
    bounded = workflow.BoundedWorkflow(graph, "thread-1", 5, tracer)

    with pytest.raises(RuntimeError, match="graph blew up"):
        await bounded.ainvoke({}, thread_config("thread-1"))
    assert tracer.events == ["enter", "exit", "flush"]


async def test_default_facade_without_tracer_is_untraced() -> None:
    graph, captured = _fake_graph()
    bounded = workflow.BoundedWorkflow(graph, "thread-1", 5)

    await bounded.ainvoke({}, thread_config("thread-1"))

    assert "callbacks" not in captured["config"]
    assert "run_name" not in captured["config"]
    assert bounded.trace_id == ""


async def test_non_list_callbacks_are_rejected() -> None:
    from langchain_core.callbacks.manager import CallbackManager

    graph, _captured = _fake_graph()
    bounded = workflow.BoundedWorkflow(graph, "thread-1", 5, RecordingTracer())
    manager = CallbackManager(handlers=[], inheritable_handlers=[])

    with pytest.raises(ValueError, match="list of handlers"):
        await bounded.ainvoke({}, {**thread_config("thread-1"), "callbacks": manager})


def _build_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "checkpointer": InMemorySaver(),
        "session_factory": MagicMock(),
        "settings": Settings(_env_file=None),
        "analysis_run_id": uuid4(),
        "thread_id": str(uuid4()),
        "parser": MagicMock(),
        "embedder": MagicMock(),
        "comparator": MagicMock(),
        "sprint_planner": MagicMock(),
    }
    kwargs.update(overrides)
    return kwargs


async def test_build_workflow_binds_unified_trace_id_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "AsyncPostgresSaver", InMemorySaver)
    run_id = uuid4()
    graph = workflow.build_workflow(**_build_kwargs(analysis_run_id=run_id))
    assert graph.trace_id == analysis_trace_id(run_id)


async def test_build_workflow_fails_closed_for_unwired_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "AsyncPostgresSaver", InMemorySaver)
    settings = Settings(_env_file=None, observability_backend="openllmetry")
    with pytest.raises(TracingUnavailableError, match="not wired in the MVP"):
        workflow.build_workflow(**_build_kwargs(settings=settings))


async def test_build_workflow_accepts_explicit_tracer_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "AsyncPostgresSaver", InMemorySaver)
    tracer = RecordingTracer("explicit-trace")
    graph = workflow.build_workflow(**_build_kwargs(tracer=tracer))
    assert graph.trace_id == "explicit-trace"


# ---------------------------------------------------------------------------
# Real SDK: actual OTel export pipeline against a loopback OTLP capture server
# ---------------------------------------------------------------------------


class _OtlpCaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        headers = {key.lower(): value for key, value in self.headers.items()}
        self.server.captures.append((self.path, headers, body))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()
        self.wfile.write(b"")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def otlp_capture() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OtlpCaptureHandler)
    server.captures = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        # Flush the SDK while the capture server is still listening so teardown
        # exports never hit a closed port (the autouse reset is then a no-op).
        langfuse_tracing.shutdown_langfuse()
        server.shutdown()
        server.server_close()


def _exported_spans(server: ThreadingHTTPServer) -> list[Any]:
    """Decode the real OTLP protobuf payloads the SDK shipped to the backend."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    spans: list[Any] = []
    captures = server.captures  # type: ignore[attr-defined]
    assert captures, "the SDK exported no spans to the OTLP endpoint"
    for path, headers, body in captures:
        assert path == "/api/public/otel/v1/traces"
        assert headers.get("authorization", "").startswith("Basic ")
        if headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                spans.extend(scope_spans.spans)
    return spans


def _string_attrs(span: Any) -> dict[str, str]:
    return {
        kv.key: kv.value.string_value for kv in span.attributes if kv.value.HasField("string_value")
    }


def _span_attrs(span: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for kv in span.attributes:
        if kv.value.HasField("string_value"):
            attrs[kv.key] = kv.value.string_value
        elif kv.value.HasField("int_value"):
            attrs[kv.key] = kv.value.int_value
    return attrs


@requires_sdk
async def test_langfuse_tracer_caches_client_and_binds_trace_context() -> None:
    settings = _langfuse_settings()
    run_id, thread = uuid4(), str(uuid4())
    tracer = create_run_tracer(settings, analysis_run_id=run_id, thread_id=thread)
    other = create_run_tracer(settings, analysis_run_id=uuid4(), thread_id=str(uuid4()))
    assert tracer._client is other._client  # one SDK client per process
    assert tracer.trace_id == analysis_trace_id(run_id)

    from langfuse.langchain import CallbackHandler

    with tracer.invocation() as callbacks:
        handler = callbacks[0]
        assert isinstance(handler, CallbackHandler)
        assert handler.trace_context == {"trace_id": analysis_trace_id(run_id)}
        assert handler.update_trace is True
    with tracer.invocation() as again:
        # Handlers carry per-invocation state: never reuse one across runs.
        assert again[0] is not handler


@requires_sdk
async def test_flush_swallows_export_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = create_run_tracer(_langfuse_settings(), analysis_run_id=uuid4(), thread_id="t")

    class BrokenClient:
        def flush(self) -> None:
            raise RuntimeError("backend unreachable")

    monkeypatch.setattr(tracer, "_client", BrokenClient())
    await tracer.flush()  # tracing failures must never fail the run


@requires_sdk
async def test_shutdown_tracing_clears_cache_and_allows_restart() -> None:
    settings = _langfuse_settings()
    create_run_tracer(settings, analysis_run_id=uuid4(), thread_id="t")
    assert langfuse_tracing._client is not None

    await shutdown_tracing()
    assert langfuse_tracing._client is None

    # An in-process restart rebuilds a live client instead of reusing dead threads.
    tracer = create_run_tracer(settings, analysis_run_id=uuid4(), thread_id="t")
    assert tracer._client is not None


class _CountState(TypedDict):
    count: int


def _count_graph() -> Any:
    async def alpha(state: _CountState, config: RunnableConfig) -> dict[str, int]:
        return {"count": state.get("count", 0) + 1}

    async def beta(state: _CountState, config: RunnableConfig) -> dict[str, int]:
        return {"count": state["count"] + 1}

    builder = StateGraph(_CountState)
    builder.add_node("alpha", alpha)
    builder.add_node("beta", beta)
    builder.add_edge(START, "alpha")
    builder.add_edge("alpha", "beta")
    builder.add_edge("beta", END)
    return builder.compile(checkpointer=InMemorySaver())


@requires_sdk
async def test_real_sdk_exports_unified_trace_across_invocations(
    otlp_capture: ThreadingHTTPServer,
) -> None:
    """Two invocations of one run (start + later resume) land in ONE trace."""
    settings = _langfuse_settings(host=f"http://127.0.0.1:{otlp_capture.server_port}")
    run_id, thread = uuid4(), str(uuid4())
    expected = analysis_trace_id(run_id)
    compiled = _count_graph()

    first = create_run_tracer(settings, analysis_run_id=run_id, thread_id=thread)
    await workflow.BoundedWorkflow(compiled, thread, 10, first).ainvoke(
        {"count": 0}, thread_config(thread)
    )

    # Simulate the approval resume in a fresh process: new tracer, new handler,
    # same run id — the deterministic trace id must join the original trace.
    second = create_run_tracer(settings, analysis_run_id=run_id, thread_id=thread)
    await workflow.BoundedWorkflow(compiled, thread, 10, second).ainvoke(
        {"count": 5}, thread_config(thread)
    )

    spans = _exported_spans(otlp_capture)
    assert {span.trace_id.hex() for span in spans} == {expected}
    names = {span.name for span in spans}
    assert {"alpha", "beta", ANALYSIS_TRACE_NAME} <= names
    for span in spans:
        attrs = _string_attrs(span)
        assert attrs.get("session.id") == thread
        assert attrs.get("langfuse.trace.name") == ANALYSIS_TRACE_NAME
        assert attrs.get("langfuse.trace.metadata.analysis_run_id") == str(run_id)
        assert attrs.get("langfuse.trace.metadata.thread_id") == thread


@requires_sdk
async def test_full_graph_interrupt_and_resume_share_one_trace(
    monkeypatch: pytest.MonkeyPatch, otlp_capture: ThreadingHTTPServer
) -> None:
    """The real ProofPath graph, paused at the approval gate and resumed,
    emits one unified trace ID across both invocations (Sub-part 2.4.1b)."""
    monkeypatch.setattr(workflow, "AsyncPostgresSaver", InMemorySaver)
    thread, opp, run_id = str(uuid4()), uuid4(), uuid4()
    session = MagicMock()

    @asynccontextmanager
    async def transaction() -> AsyncIterator[MagicMock]:
        yield session

    factory = MagicMock()
    factory.begin = transaction
    repo = MagicMock()
    repo.upsert_verdicts = AsyncMock()
    repo.get_run_by_id = AsyncMock(return_value=MagicMock(thread_id=thread, opportunity_id=opp))
    monkeypatch.setattr(workflow, "AnalysisRepo", lambda _session: repo)

    async def planner(state: Any, **kwargs: Any) -> dict[str, Any]:
        return {"step_budget": 50, "step_count": 1}

    async def bump(state: Any, **kwargs: Any) -> dict[str, Any]:
        return {"step_count": state["step_count"] + 1}

    async def policy(state: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "policy_verified": True,
            "policy_retry": False,
            "step_count": state["step_count"] + 1,
        }

    draft_id = uuid4()

    async def drafts(state: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "draft_actions": [
                ActionDraftRecord(
                    id=draft_id,
                    analysis_run_id=run_id,
                    action_type=ActionType.TRACKER_UPSERT,
                    status=ActionDraftStatus.PENDING_APPROVAL,
                    payload=TrackerUpsertPayload(
                        action_type=ActionType.TRACKER_UPSERT,
                        opportunity_id=opp,
                        stage=ApplicationStage.PREPARING,
                    ),
                    idempotency_key=f"proofpath:{run_id}:tracker_upsert",
                )
            ],
            "step_count": state["step_count"] + 1,
        }

    async def retrieve(state: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "retrieved_evidence": {
                str(opp): [
                    EvidenceSnippet(
                        chunk_id=uuid4(),
                        source_id=uuid4(),
                        canonical_uri="https://github.com/example/repo#readme",
                        content="Verified evidence text.",
                        score=0.016393,
                    )
                ]
            },
            "step_count": state["step_count"] + 1,
        }

    for name, fn in [
        ("planner_router", planner),
        ("analyse_role_node", bump),
        ("retrieve_evidence_node", retrieve),
        ("compare_evidence_node", bump),
        ("policy_verifier_node", policy),
        ("recommend_contribution_sprint_node", bump),
        ("create_action_drafts_node", drafts),
    ]:
        monkeypatch.setattr(workflow, name, fn)

    settings = _langfuse_settings(host=f"http://127.0.0.1:{otlp_capture.server_port}")
    graph = workflow.build_workflow(
        **_build_kwargs(
            settings=settings,
            analysis_run_id=run_id,
            thread_id=thread,
            session_factory=factory,
        )
    )
    assert graph.trace_id == analysis_trace_id(run_id)

    state = await graph.ainvoke(initial_state(opp, settings=settings), thread_config(thread))
    assert state.get("__interrupt__"), "graph must pause at the approval gate"

    resumed = await graph.aresume(
        {"decision": "approved", "draft_id": str(draft_id)}, thread_config(thread)
    )
    assert resumed.get("approval_decision") is not None
    assert not resumed.get("errors")

    spans = _exported_spans(otlp_capture)
    assert {span.trace_id.hex() for span in spans} == {analysis_trace_id(run_id)}
    names = {span.name for span in spans}
    assert ANALYSIS_TRACE_NAME in names
    # Sub-part 2.4.2a: the canonical parent-child node spans, incl. policy_verifier.
    assert {
        "planner",
        "analyse_role",
        "retrieve_evidence",
        "compare_evidence",
        "policy_verifier",
        "approval_gate",
    } <= names
    for span in spans:
        assert _string_attrs(span).get("session.id") == thread

    # retrieve_evidence carries chunk count + similarity scores as span metadata.
    retrieve_span = next(s for s in spans if s.name == "retrieve_evidence")
    meta = _span_attrs(retrieve_span)
    assert meta.get("langfuse.observation.metadata.retrieved_chunk_count") == 1
    assert meta.get("langfuse.observation.metadata.requirements_searched") == 1
    assert json.loads(meta["langfuse.observation.metadata.similarity_scores"]) == [0.016393]
