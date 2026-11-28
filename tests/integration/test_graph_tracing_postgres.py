"""Unified run tracing against live PostgreSQL + the real Langfuse SDK (2.4.1).

Proves the properties that only the real service path can prove:

1. ``execute_analysis_graph`` (shared by the sync API and the async worker) and
   ``resume_approval_run`` both attach the configured backend's callback handler
   without any caller-side tracing wiring;
2. the initial run, the durable approval pause, and a resume after reopening the
   Postgres saver (process restart) all export spans under ONE unified trace ID
   derived from the analysis run id;
3. the spans actually leave the process: the real Langfuse v3 OTel pipeline
   ships OTLP protobuf to a loopback capture endpoint, which is decoded and
   asserted (no mock-only tracing).

Skips without live PostgreSQL (like the other integration tests) and without
the ``proofpath[observability]`` extra.
"""

from __future__ import annotations

import gzip
import json
import threading
import uuid
from collections.abc import Iterator
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import (
    AnalysisRunStatus,
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.graph import workflow as workflow_module
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.graph.dependencies import GraphDependencies
from proofpath.graph.resume import resume_approval_run
from proofpath.jobs import execute_analysis_graph
from proofpath.observability import langfuse_tracing
from proofpath.observability.tracing import ANALYSIS_TRACE_NAME, analysis_trace_id
from proofpath.observability.usage import record_llm_usage
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.evidence import EvidenceSnippet
from proofpath.schemas.opportunity import RequirementRecord

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("langfuse") is None or find_spec("langchain") is None,
        reason="Install .[observability] to test the real SDK",
    ),
]


@pytest.fixture(autouse=True)
def _reset_langfuse_cache() -> Iterator[None]:
    """Isolate the SDK client cache and global OTel provider between tests."""
    yield
    langfuse_tracing.shutdown_langfuse()
    from opentelemetry import trace as otel_trace

    if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        otel_trace._TRACER_PROVIDER = None
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False


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
        langfuse_tracing.shutdown_langfuse()
        server.shutdown()
        server.server_close()


def _span_attrs(span: Any) -> dict[str, Any]:
    """Decode OTel span attributes (string and int values) into a flat dict."""
    attrs: dict[str, Any] = {}
    for kv in span.attributes:
        if kv.value.HasField("string_value"):
            attrs[kv.key] = kv.value.string_value
        elif kv.value.HasField("int_value"):
            attrs[kv.key] = kv.value.int_value
    return attrs


def _exported_spans(server: ThreadingHTTPServer) -> list[Any]:
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


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, factory: async_sessionmaker[AsyncSession]
) -> None:
    """Keep the real wiring, drafts node, and gate; stub only the model calls."""

    async def analyse(state: dict, **_kwargs: object) -> dict[str, object]:
        async with factory() as session:
            rows = await OpportunityRepo(session).list_requirements(state["opportunity_id"])
            requirements = [
                RequirementRecord(
                    id=row.id,
                    opportunity_id=row.opportunity_id,
                    requirement_text=row.requirement_text,
                    entity=row.entity,
                    category=row.category,
                    priority=row.priority,
                )
                for row in rows
            ]
        return {"requirements": requirements, "step_count": state.get("step_count", 0) + 1}

    async def retrieve(state: dict, **_kwargs: object) -> dict[str, object]:
        # One scored snippet per requirement so the span carries real counts/scores.
        return {
            "retrieved_evidence": {
                str(item.id): [
                    EvidenceSnippet(
                        chunk_id=uuid.uuid4(),
                        source_id=uuid.uuid4(),
                        canonical_uri="https://github.com/example/repo#readme",
                        content="Verified evidence text.",
                        score=0.016393,
                    )
                ]
                for item in state["requirements"]
            },
            "step_count": state.get("step_count", 0) + 1,
        }

    async def compare(state: dict, **_kwargs: object) -> dict[str, object]:
        # Simulate a model call's token usage (no live LLM in integration tests).
        record_llm_usage(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        return {
            "verdicts": [
                ClaimVerdictSchema(
                    requirement_id=item.id,
                    verdict=VerdictType.UNSUPPORTED,
                    claim_summary="Insufficient verified evidence for this requirement.",
                )
                for item in state["requirements"]
            ],
            "step_count": state.get("step_count", 0) + 1,
        }

    async def policy(state: dict, **_kwargs: object) -> dict[str, object]:
        return {
            "policy_violations": [],
            "policy_retry": False,
            "policy_verified": True,
            "step_count": state.get("step_count", 0) + 1,
        }

    async def sprint(state: dict, **_kwargs: object) -> dict[str, object]:
        return {"contribution_sprint": None, "step_count": state.get("step_count", 0) + 1}

    monkeypatch.setattr(workflow_module, "analyse_role_node", analyse)
    monkeypatch.setattr(workflow_module, "retrieve_evidence_node", retrieve)
    monkeypatch.setattr(workflow_module, "compare_evidence_node", compare)
    monkeypatch.setattr(workflow_module, "policy_verifier_node", policy)
    monkeypatch.setattr(workflow_module, "recommend_contribution_sprint_node", sprint)


async def test_run_and_approval_resume_export_one_unified_trace(
    db_engine: AsyncEngine,
    integration_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    otlp_capture: ThreadingHTTPServer,
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn(integration_database_url),
        observability_backend="langfuse",
        langfuse_public_key="pk-lf-integration",
        langfuse_secret_key="sk-lf-integration",
        langfuse_host=f"http://127.0.0.1:{otlp_capture.server_port}",
    )
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    thread_id, run_id = str(uuid4()), uuid.uuid4()
    async with factory.begin() as session:
        opportunity = await OpportunityRepo(session).create_opportunity(
            company="Example",
            role_title="Platform Engineer",
            raw_text="SAP S/4HANA required",
            sanitized_text="SAP S/4HANA required",
        )
        opportunity_id = opportunity.id
        await OpportunityRepo(session).add_requirements(
            opportunity_id,
            [
                RoleRequirementInput(
                    requirement_text="SAP S/4HANA",
                    entity="SAP",
                    category=RequirementCategory.DOMAIN,
                    priority=RequirementPriority.MUST_HAVE,
                )
            ],
        )
        await AnalysisRepo(session).create_run(
            opportunity_id=opportunity_id, thread_id=thread_id, run_id=run_id
        )
    _patch_pipeline(monkeypatch, factory)
    dependencies = GraphDependencies(
        parser=MagicMock(), embedder=MagicMock(), comparator=MagicMock(), sprint_planner=MagicMock()
    )
    expected_trace = analysis_trace_id(run_id)

    try:
        # --- 1. the shared execution routine traces without caller wiring ----
        async with postgres_checkpointer(settings=settings) as saver:
            status = await execute_analysis_graph(
                settings=settings,
                session_factory=factory,
                checkpointer=saver,
                dependencies=dependencies,
                opportunity_id=opportunity_id,
                run_id=run_id,
                thread_id=thread_id,
            )
            assert status is AnalysisRunStatus.AWAITING_APPROVAL

        # --- 2. resume after reopening the saver (process restart) -----------
        async with postgres_checkpointer(settings=settings) as saver:
            async with factory() as session:
                run = await AnalysisRepo(session).get_run_by_thread_id(
                    thread_id, include_drafts=True
                )
            assert run is not None
            assert run.action_drafts
            draft_id = run.action_drafts[0].id
            result = await resume_approval_run(
                checkpointer=saver,
                session_factory=factory,
                settings=settings,
                thread_id=thread_id,
                decision="approved",
                draft_id=draft_id,
                parser=dependencies.parser,
                embedder=dependencies.embedder,
                comparator=dependencies.comparator,
                sprint_planner=dependencies.sprint_planner,
            )
            assert result.status is AnalysisRunStatus.COMPLETED
            await saver.adelete_thread(thread_id)

        # --- 2b. metrics persisted on the run row (Sub-part 2.4.2b) ----------
        async with factory() as session:
            run = await AnalysisRepo(session).get_run_by_id(run_id)
        assert run is not None
        assert run.trace_id == expected_trace
        assert run.total_tokens == 1500  # one simulated compare call
        assert run.estimated_cost_usd == Decimal("0.000450")
        assert run.latency_ms >= 0

        # --- 3. everything landed in ONE trace, exported over real OTLP ------
        spans = _exported_spans(otlp_capture)
        assert {span.trace_id.hex() for span in spans} == {expected_trace}
        names = {span.name for span in spans}
        assert ANALYSIS_TRACE_NAME in names
        assert {
            "planner",
            "analyse_role",
            "retrieve_evidence",
            "compare_evidence",
            "policy_verifier",
            "sprint",
            "drafts",
            "approval_gate",
        } <= names
        for span in spans:
            attrs = _span_attrs(span)
            assert attrs.get("session.id") == thread_id
            assert attrs.get("langfuse.trace.name") == ANALYSIS_TRACE_NAME
            assert attrs.get("langfuse.trace.metadata.analysis_run_id") == str(run_id)

        # --- 3a. retrieve_evidence span carries chunk count + scores ---------
        retrieve_spans = [s for s in spans if s.name == "retrieve_evidence"]
        assert retrieve_spans
        meta = _span_attrs(retrieve_spans[0])
        assert meta.get("langfuse.observation.metadata.retrieved_chunk_count") == 1
        assert meta.get("langfuse.observation.metadata.requirements_searched") == 1
        assert json.loads(meta["langfuse.observation.metadata.similarity_scores"]) == [0.016393]
    finally:
        async with factory.begin() as session:
            await session.execute(delete(Opportunity).where(Opportunity.id == opportunity_id))
