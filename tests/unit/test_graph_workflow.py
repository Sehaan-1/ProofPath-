"""Graph wiring and bounded policy retry with an in-memory test saver."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from proofpath.config import Settings
from proofpath.graph import workflow
from proofpath.graph.checkpointer import thread_config
from proofpath.graph.state import initial_state

pytestmark = pytest.mark.unit


async def test_graph_routes_retry_and_requires_thread(monkeypatch):
    thread, opp, run_id = str(uuid4()), uuid4(), uuid4()
    calls = []
    session = MagicMock()

    @asynccontextmanager
    async def transaction():
        yield session

    factory = MagicMock()
    factory.begin = transaction
    repo = MagicMock()
    repo.get_run_by_id = AsyncMock(return_value=MagicMock(thread_id=thread, opportunity_id=opp))
    monkeypatch.setattr(workflow, "AnalysisRepo", lambda _session: repo)
    monkeypatch.setattr(workflow, "AsyncPostgresSaver", InMemorySaver)

    async def planner(state, **kwargs):
        calls.append("planner")
        return {"step_budget": 50, "step_count": 1}

    async def analyse(state, **kwargs):
        calls.append("analyse")
        return {"step_count": state["step_count"] + 1}

    async def retrieve(state, **kwargs):
        calls.append("retrieve")
        return {"step_count": state["step_count"] + 1}

    async def compare(state, **kwargs):
        calls.append("compare")
        return {"step_count": state["step_count"] + 1}

    async def policy(state, **kwargs):
        calls.append("policy")
        return {"policy_retry": calls.count("policy") == 1, "step_count": state["step_count"] + 1}

    async def sprint(state, **kwargs):
        calls.append("sprint")
        return {"step_count": state["step_count"] + 1}

    async def drafts(state, **kwargs):
        calls.append("drafts")
        return {"step_count": state["step_count"] + 1}

    for name, fn in [
        ("planner_router", planner),
        ("analyse_role_node", analyse),
        ("retrieve_evidence_node", retrieve),
        ("compare_evidence_node", compare),
        ("policy_verifier_node", policy),
        ("recommend_contribution_sprint_node", sprint),
        ("create_action_drafts_node", drafts),
    ]:
        monkeypatch.setattr(workflow, name, fn)
    graph = workflow.build_workflow(
        checkpointer=InMemorySaver(),
        session_factory=factory,
        settings=Settings(_env_file=None),
        analysis_run_id=run_id,
        thread_id=thread,
        parser=MagicMock(),
        embedder=MagicMock(),
        comparator=MagicMock(),
        sprint_planner=MagicMock(),
    )
    with pytest.raises(ValueError, match="thread_id"):
        await graph.ainvoke(
            initial_state(opp, settings=Settings(_env_file=None)), thread_config("wrong")
        )
    output = await graph.ainvoke(
        initial_state(opp, settings=Settings(_env_file=None)), thread_config(thread)
    )
    assert calls == [
        "planner",
        "analyse",
        "retrieve",
        "compare",
        "policy",
        "compare",
        "policy",
        "sprint",
        "drafts",
    ]
    assert output["step_count"] == 9


def _failure_graph(monkeypatch, *, step_budget=50, audit_error=None):
    thread, opp, run_id = str(uuid4()), uuid4(), uuid4()
    session = MagicMock()

    @asynccontextmanager
    async def transaction():
        yield session

    factory = MagicMock()
    factory.begin = transaction
    repo = MagicMock()
    repo.get_run_by_id = AsyncMock(return_value=MagicMock(thread_id=thread, opportunity_id=opp))
    audit = MagicMock()
    audit.record = AsyncMock(side_effect=audit_error)
    monkeypatch.setattr(workflow, "AnalysisRepo", lambda _session: repo)
    monkeypatch.setattr(workflow, "AuditRepo", lambda _session: audit)
    monkeypatch.setattr(workflow, "AsyncPostgresSaver", InMemorySaver)
    graph = workflow.build_workflow(
        checkpointer=InMemorySaver(),
        session_factory=factory,
        settings=Settings(_env_file=None),
        analysis_run_id=run_id,
        thread_id=thread,
        parser=MagicMock(),
        embedder=MagicMock(),
        comparator=MagicMock(),
        sprint_planner=MagicMock(),
        step_budget=step_budget,
    )
    return graph, audit, thread, opp, run_id


async def test_failure_audited_and_unverified_verdicts_removed(monkeypatch):
    from proofpath.db.models import VerdictType
    from proofpath.schemas.analysis import ClaimVerdictSchema

    async def fake_planner(state, **kwargs):
        return {"step_count": 1}

    async def fake_analyse(state, **kwargs):
        raise TimeoutError("secret token and private content")

    monkeypatch.setattr(workflow, "planner_router", fake_planner)
    monkeypatch.setattr(workflow, "analyse_role_node", fake_analyse)
    graph, audit, thread, opp, run_id = _failure_graph(monkeypatch)
    state = initial_state(opp, settings=Settings(_env_file=None))
    state["verdicts"] = [
        ClaimVerdictSchema(
            requirement_id=uuid4(), verdict=VerdictType.SUPPORTED, claim_summary="Unverified"
        )
    ]
    result = await graph.ainvoke(
        state, {**thread_config(thread), "recursion_limit": 999999}, durability="async"
    )
    assert result["verdicts"] == []
    assert result["errors"][0]["node"] == "analyse_role"
    assert result["errors"][0]["retryable"] is True
    assert "secret" not in str(result["errors"])
    assert audit.record.await_count == 1
    kwargs = audit.record.await_args.kwargs
    assert kwargs["event_type"] == "graph.node_failed"
    assert kwargs["resource_id"] == run_id
    assert kwargs["details_json"]["thread_id"] == thread
    assert "secret" not in str(kwargs)
    snapshot = await graph.aget_state(thread_config(thread))
    assert snapshot.next == ()
    assert snapshot.values["errors"] == result["errors"]


async def test_audit_failure_is_reported_and_verified_results_preserved(monkeypatch):
    from proofpath.db.models import VerdictType
    from proofpath.schemas.analysis import ClaimVerdictSchema

    async def fake_planner(state, **kwargs):
        return {"step_count": 1}

    async def fail(state, **kwargs):
        raise ValueError("unsafe data")

    monkeypatch.setattr(workflow, "planner_router", fake_planner)
    monkeypatch.setattr(workflow, "analyse_role_node", fail)
    graph, audit, thread, opp, _ = _failure_graph(
        monkeypatch, audit_error=RuntimeError("database offline")
    )
    state = initial_state(opp, settings=Settings(_env_file=None))
    verdict = ClaimVerdictSchema(
        requirement_id=uuid4(), verdict=VerdictType.SUPPORTED, claim_summary="Verified"
    )
    state["verdicts"] = [verdict]
    state["policy_verified"] = True
    result = await graph.ainvoke(state, thread_config(thread))
    assert result["verdicts"] == [verdict]
    assert [error["code"] for error in result["errors"]] == ["node_failed", "audit_write_failed"]
    assert audit.record.await_count == 1


async def test_step_budget_routes_to_fallback_even_when_limit_overridden(monkeypatch):
    async def fake_planner(state, **kwargs):
        return {"step_count": 1, "step_budget": 1}

    monkeypatch.setattr(workflow, "planner_router", fake_planner)
    graph, audit, thread, opp, _ = _failure_graph(monkeypatch, step_budget=1)
    result = await graph.ainvoke(
        initial_state(opp, settings=Settings(_env_file=None)),
        {**thread_config(thread), "recursion_limit": 9999},
    )
    assert result["errors"][0]["code"] == "step_budget_exhausted"
    assert result["errors"][0]["node"] == "analyse_role"
    assert audit.record.await_count == 1


def test_config_rejects_unbounded_limits():
    with pytest.raises(ValueError):
        workflow.workflow_config("thread", step_budget=workflow.MAX_GRAPH_STEPS + 1)


async def test_interrupt_is_not_treated_as_failure(monkeypatch):
    from langgraph.types import Command, interrupt

    async def fake_planner(state, **kwargs):
        interrupt("approve?")
        return {"step_count": 1}

    async def fail(state, **kwargs):
        raise ValueError("after resume")

    monkeypatch.setattr(workflow, "planner_router", fake_planner)
    monkeypatch.setattr(workflow, "analyse_role_node", fail)
    graph, audit, thread, opp, _ = _failure_graph(monkeypatch)
    result = await graph.ainvoke(
        initial_state(opp, settings=Settings(_env_file=None)), thread_config(thread)
    )
    assert result["__interrupt__"]
    assert result["errors"] == []
    audit.record.assert_not_awaited()
    resumed = await graph.ainvoke(Command(resume=True), thread_config(thread))
    assert resumed["errors"][0]["node"] == "analyse_role"
    assert audit.record.await_count == 1


async def test_stream_uses_fixed_config_and_durability(monkeypatch):
    fake_graph = MagicMock()
    captured = {}

    async def stream(_input, config, **kwargs):
        captured.update(config=config, kwargs=kwargs)
        yield {"ok": True}

    fake_graph.astream = stream
    graph = workflow.BoundedWorkflow(fake_graph, "run", 7)
    events = [
        item
        async for item in graph.astream(
            {}, {**thread_config("run"), "recursion_limit": 10000}, durability="exit"
        )
    ]
    assert events == [{"ok": True}]
    assert captured["config"]["recursion_limit"] == 11
    assert captured["kwargs"]["durability"] == "sync"
