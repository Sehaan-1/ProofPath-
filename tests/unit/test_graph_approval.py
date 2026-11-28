"""Approval-gate interrupt/resume contracts (Sub-slice 2.1.1).

These tests exercise LangGraph's native pause/resume semantics with an in-memory
saver: the durable PostgreSQL path is covered by
`tests/integration/test_graph_approval_postgres.py`.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
)
from proofpath.graph import resume as resume_module
from proofpath.graph import workflow
from proofpath.graph.approval import (
    ApprovalPayloadError,
    ApprovalStateError,
    approval_request_from_interrupts,
    build_approval_request,
    parse_approval_decision,
    parse_resume_value,
    pending_approval_request,
)
from proofpath.graph.checkpointer import _STATE_TYPES, thread_config
from proofpath.graph.state import (
    ApprovalDecision,
    ApprovalDraftSummary,
    ApprovalRequest,
    ProofPathState,
    initial_state,
)
from proofpath.schemas.actions import (
    ActionDraftRecord,
    CalendarBlockPayload,
    ContributionSprintPayload,
    ContributionSprintPlan,
    TrackerUpsertPayload,
)

pytestmark = pytest.mark.unit


def _saver() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES))


def _draft(
    *,
    run_id: uuid.UUID,
    opportunity_id: uuid.UUID,
    status: ActionDraftStatus = ActionDraftStatus.PENDING_APPROVAL,
    action_type: ActionType = ActionType.TRACKER_UPSERT,
    notes: str | None = None,
) -> ActionDraftRecord:
    if action_type is ActionType.CALENDAR_BLOCK:
        payload: Any = CalendarBlockPayload(
            action_type=ActionType.CALENDAR_BLOCK,
            title="Sprint: hybrid RAG",
            start_at=datetime(2026, 1, 1, 9, tzinfo=UTC),
            end_at=datetime(2026, 1, 2, 9, tzinfo=UTC),
        )
    elif action_type is ActionType.CONTRIBUTION_SPRINT:
        payload = ContributionSprintPayload(
            action_type=ActionType.CONTRIBUTION_SPRINT,
            plan=ContributionSprintPlan(
                title="Hybrid RAG sprint",
                scope="Build a small hybrid retrieval demo.",
                repo_or_topic="pgvector",
                deliverables=["Demo"],
                estimate_days=3,
                target_gap_requirement_ids=[uuid.uuid4()],
                rationale="Highest-impact gap.",
            ),
        )
    else:
        payload = TrackerUpsertPayload(
            action_type=ActionType.TRACKER_UPSERT,
            opportunity_id=opportunity_id,
            stage=ApplicationStage.PREPARING,
            notes=notes,
        )
    return ActionDraftRecord(
        id=uuid.uuid4(),
        analysis_run_id=run_id,
        action_type=action_type,
        status=status,
        payload=payload,
        idempotency_key=f"proofpath:{run_id}:{action_type.value}",
    )


def _gate_state(run_id: uuid.UUID, opportunity_id: uuid.UUID, **kwargs: Any) -> ProofPathState:
    state = initial_state(opportunity_id)
    state["draft_actions"] = [_draft(run_id=run_id, opportunity_id=opportunity_id, **kwargs)]
    state["step_count"] = 7
    return state


def _gate_graph(saver: InMemorySaver, run_id: uuid.UUID):
    builder = StateGraph(ProofPathState)
    builder.add_node("approval_gate", partial(_guarded_gate, analysis_run_id=run_id))
    builder.add_edge(START, "approval_gate")
    builder.add_edge("approval_gate", END)
    return builder.compile(checkpointer=saver)


async def _guarded_gate(state: ProofPathState, *, analysis_run_id: uuid.UUID):
    from proofpath.graph.approval import approval_gate_node

    return await approval_gate_node(state, analysis_run_id=analysis_run_id)


async def test_gate_pauses_without_side_effects_and_without_draft_contents() -> None:
    run_id, opportunity_id = uuid.uuid4(), uuid.uuid4()
    saver = _saver()
    graph = _gate_graph(saver, run_id)
    config = thread_config("thread-pause")

    paused = await graph.ainvoke(
        _gate_state(run_id, opportunity_id, notes="personal phone +91-99999-00000"),
        config,
        durability="sync",
    )

    assert paused["__interrupt__"]
    assert paused["approval_decision"] is None
    assert paused["approval_request"] is None
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("approval_gate",)
    request = pending_approval_request(snapshot)
    assert request is not None
    assert request.analysis_run_id == run_id
    assert request.draft_ids == {paused["draft_actions"][0].id}
    # The interrupt payload is checkpointed, so it must not copy personal notes.
    assert "99999" not in str(snapshot.interrupts)
    assert "personal phone" not in str(snapshot.interrupts)
    # Nothing has been approved or executed by pausing.
    assert snapshot.values["draft_actions"][0].status is ActionDraftStatus.PENDING_APPROVAL


@pytest.mark.parametrize("decision", ["approved", "rejected"])
async def test_resume_on_rebuilt_graph_records_decision(decision: str) -> None:
    run_id, opportunity_id = uuid.uuid4(), uuid.uuid4()
    saver = _saver()
    config = thread_config(f"thread-{decision}")
    draft_id = uuid.uuid4()
    state = _gate_state(run_id, opportunity_id)
    state["draft_actions"][0] = state["draft_actions"][0].model_copy(update={"id": draft_id})

    await _gate_graph(saver, run_id).ainvoke(state, config, durability="sync")
    # A second compiled graph over the same saver models a process restart: the
    # pause lives in the checkpoint, not in memory.
    restarted = _gate_graph(saver, run_id)
    assert pending_approval_request(await restarted.aget_state(config)) is not None

    resumed = await restarted.ainvoke(
        Command(resume={"decision": decision, "draft_id": str(draft_id)}),
        config,
        durability="sync",
    )

    assert not resumed.get("__interrupt__")
    assert resumed["approval_decision"] is not None
    assert resumed["approval_decision"].decision == decision
    assert resumed["approval_decision"].draft_id == draft_id
    assert resumed["step_count"] == 8
    assert resumed["errors"] == []
    # Resuming records the decision only; the draft row is untouched here.
    assert resumed["draft_actions"][0].status is ActionDraftStatus.PENDING_APPROVAL


@pytest.mark.parametrize(
    "value",
    [
        {"decision": "approved", "draft_id": "__other__"},
        {"decision": "maybe", "draft_id": "__target__"},
        {"decision": "approved"},
        {"decision": "approved", "draft_id": "__target__", "force": True},
        {"decision": "approved", "draft_id": "not-a-uuid"},
        "approved",
    ],
)
async def test_invalid_resume_values_are_rejected(value: object) -> None:
    run_id, opportunity_id = uuid.uuid4(), uuid.uuid4()
    saver = _saver()
    config = thread_config("thread-invalid")
    state = _gate_state(run_id, opportunity_id)
    target = state["draft_actions"][0].id

    await _gate_graph(saver, run_id).ainvoke(state, config, durability="sync")
    if isinstance(value, dict):
        value = {
            key: (
                str(uuid.uuid4())
                if item == "__other__"
                else str(target)
                if item == "__target__"
                else item
            )
            for key, item in value.items()
        }
    with pytest.raises(ValueError):
        await _gate_graph(saver, run_id).ainvoke(Command(resume=value), config, durability="sync")
    # The pause survives a rejected resume and can still be resumed correctly.
    snapshot = await _gate_graph(saver, run_id).aget_state(config)
    assert pending_approval_request(snapshot) is not None


def test_parse_resume_value_validates_only_the_payload_shape() -> None:
    draft_id = uuid.uuid4()
    # Shape validation happens before any database or graph work.
    assert parse_resume_value(decision="rejected", draft_id=str(draft_id)) == ApprovalDecision(
        decision="rejected", draft_id=draft_id
    )
    for decision, target in (
        ("approved", "not-a-uuid"),
        ("Approved", uuid.uuid4()),
        (None, draft_id),
    ):
        with pytest.raises(ApprovalPayloadError):
            parse_resume_value(decision=decision, draft_id=target)


def test_parse_approval_decision_rejects_unknown_and_foreign_targets() -> None:
    draft_id = uuid.uuid4()
    parsed = parse_approval_decision(
        {"decision": "approved", "draft_id": str(draft_id)}, allowed_draft_ids={draft_id}
    )
    assert parsed == ApprovalDecision(decision="approved", draft_id=draft_id)
    assert set(
        parse_approval_decision(
            {"decision": "rejected", "draft_id": draft_id}, allowed_draft_ids={draft_id}
        ).model_dump()
    ) == {"decision", "draft_id"}
    with pytest.raises(ApprovalPayloadError, match="outside this run"):
        parse_approval_decision(
            {"decision": "approved", "draft_id": str(uuid.uuid4())}, allowed_draft_ids={draft_id}
        )
    with pytest.raises(ApprovalPayloadError, match="exactly"):
        parse_approval_decision(
            {"decision": "approved", "draft_id": str(draft_id), "extra": 1},
            allowed_draft_ids={draft_id},
        )
    with pytest.raises(ApprovalPayloadError, match="approved"):
        parse_approval_decision(
            {"decision": "APPROVED", "draft_id": str(draft_id)}, allowed_draft_ids={draft_id}
        )


@pytest.mark.parametrize(
    ("drafts", "match"),
    [
        ([], "at least one action draft"),
        ("duplicate", "unique"),
        ("foreign", "do not belong"),
        ("decided", "at least one pending"),
    ],
)
def test_build_approval_request_fails_closed(drafts: object, match: str) -> None:
    run_id, opportunity_id = uuid.uuid4(), uuid.uuid4()
    state = initial_state(opportunity_id)
    if drafts == "duplicate":
        record = _draft(run_id=run_id, opportunity_id=opportunity_id)
        state["draft_actions"] = [record, record.model_copy(deep=True)]
    elif drafts == "foreign":
        state["draft_actions"] = [_draft(run_id=uuid.uuid4(), opportunity_id=opportunity_id)]
    elif drafts == "decided":
        state["draft_actions"] = [
            _draft(
                run_id=run_id,
                opportunity_id=opportunity_id,
                status=ActionDraftStatus.APPROVED,
            )
        ]
    else:
        state["draft_actions"] = []
    with pytest.raises(ApprovalStateError, match=match):
        build_approval_request(state, analysis_run_id=run_id)


def test_approval_request_summarizes_each_action_type() -> None:
    run_id, opportunity_id = uuid.uuid4(), uuid.uuid4()
    state = initial_state(opportunity_id)
    state["draft_actions"] = [
        _draft(run_id=run_id, opportunity_id=opportunity_id),
        _draft(
            run_id=run_id,
            opportunity_id=opportunity_id,
            action_type=ActionType.CALENDAR_BLOCK,
        ),
        _draft(
            run_id=run_id,
            opportunity_id=opportunity_id,
            action_type=ActionType.CONTRIBUTION_SPRINT,
        ),
    ]
    request = build_approval_request(state, analysis_run_id=run_id)
    assert [item.action_type for item in request.drafts] == [
        ActionType.CALENDAR_BLOCK,
        ActionType.CONTRIBUTION_SPRINT,
        ActionType.TRACKER_UPSERT,
    ]
    assert request.drafts[0].summary == "Create calendar block 'Sprint: hybrid RAG'"
    assert request.drafts[2].summary == "Update application tracker stage to 'preparing'"


def test_approval_request_from_interrupts_ignores_foreign_interrupts() -> None:
    assert approval_request_from_interrupts(None) is None
    assert approval_request_from_interrupts([]) is None
    assert approval_request_from_interrupts([{"value": {"question": "continue?"}}]) is None
    request = ApprovalRequest(
        analysis_run_id=uuid.uuid4(),
        drafts=[
            ApprovalDraftSummary(
                draft_id=uuid.uuid4(), action_type=ActionType.TRACKER_UPSERT, summary="Track"
            )
        ],
    )
    assert approval_request_from_interrupts([request]) == request


async def test_bounded_workflow_aresume_binds_thread_and_durability() -> None:
    fake_graph = MagicMock()
    captured: dict[str, Any] = {}

    async def ainvoke(_input, config, **kwargs):
        captured.update(input=_input, config=config, kwargs=kwargs)
        return {"ok": True}

    fake_graph.ainvoke = ainvoke
    bounded = workflow.BoundedWorkflow(fake_graph, "run-thread", 7)
    value = {"decision": "approved", "draft_id": str(uuid.uuid4())}

    result = await bounded.aresume(value, {**thread_config("run-thread"), "recursion_limit": 999})
    assert result == {"ok": True}
    assert isinstance(captured["input"], Command)
    assert captured["input"].resume == value
    assert captured["config"]["recursion_limit"] == 11
    assert captured["kwargs"]["durability"] == "sync"

    with pytest.raises(ValueError, match="thread_id"):
        await bounded.aresume(value, thread_config("other-thread"))


class _FakeWorkflow:
    """Stand-in for `BoundedWorkflow` inside `resume_approval_run` tests."""

    def __init__(self, request: ApprovalRequest | None, state: dict[str, Any] | Exception):
        self.request = request
        self.state = state
        self.resumed_with: list[object] = []

    async def apending_approval(self) -> ApprovalRequest | None:
        return self.request

    async def aresume(self, value: object, _config: object = None) -> dict[str, Any]:
        self.resumed_with.append(value)
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


@dataclass(frozen=True)
class _Harness:
    """Everything a `resume_approval_run` test needs to inspect or assert on."""

    thread_id: str
    draft_id: uuid.UUID
    run: SimpleNamespace
    fake_workflow: _FakeWorkflow
    statuses: list[AnalysisRunStatus]
    built: list[dict[str, Any]]
    claims: list[uuid.UUID]
    factory: MagicMock

    def kwargs(self) -> dict[str, Any]:
        return {
            "checkpointer": MagicMock(),
            "session_factory": self.factory,
            "settings": Settings(_env_file=None),
            "thread_id": self.thread_id,
            "decision": "approved",
            "draft_id": self.draft_id,
            "parser": MagicMock(),
            "embedder": MagicMock(),
            "comparator": MagicMock(),
            "sprint_planner": MagicMock(),
        }


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_status: AnalysisRunStatus = AnalysisRunStatus.AWAITING_APPROVAL,
    draft_status: ActionDraftStatus = ActionDraftStatus.PENDING_APPROVAL,
    draft_run_id: uuid.UUID | None = None,
    claimed: bool = True,
    paused: bool = True,
    resumed_state: dict[str, Any] | Exception | None = None,
) -> _Harness:
    thread_id = str(uuid.uuid4())
    run_id, opportunity_id = uuid.uuid4(), uuid.uuid4()
    draft_id = uuid.uuid4()
    run = SimpleNamespace(
        id=run_id,
        thread_id=thread_id,
        opportunity_id=opportunity_id,
        status=run_status,
        action_drafts=[
            SimpleNamespace(
                id=draft_id,
                analysis_run_id=draft_run_id if draft_run_id is not None else run_id,
                status=draft_status,
            )
        ],
        claim_verdicts=[],
        total_tokens=0,
        estimated_cost_usd=Decimal(0),
        latency_ms=0,
        trace_id=None,
    )
    request = (
        ApprovalRequest(
            analysis_run_id=run_id,
            drafts=[
                ApprovalDraftSummary(
                    draft_id=draft_id,
                    action_type=ActionType.TRACKER_UPSERT,
                    summary="Track",
                )
            ],
        )
        if paused
        else None
    )
    if resumed_state is None:
        resumed_state = {
            "errors": [],
            "approval_decision": ApprovalDecision(decision="approved", draft_id=draft_id),
        }
    fake_workflow = _FakeWorkflow(request, resumed_state)
    statuses: list[AnalysisRunStatus] = []
    built: list[dict[str, Any]] = []
    claims: list[uuid.UUID] = []

    class Repo:
        def __init__(self, _session: object) -> None:
            pass

        async def get_run_by_thread_id(self, _thread_id: str, include_drafts: bool = False):
            return run

        async def claim_awaiting_approval(self, target: uuid.UUID) -> bool:
            claims.append(target)
            return claimed

        async def update_run_metrics(self, target: uuid.UUID, *, status: AnalysisRunStatus):
            statuses.append(status)
            run.status = status
            return run

    def build(**kwargs: Any) -> _FakeWorkflow:
        built.append(kwargs)
        return fake_workflow

    @asynccontextmanager
    async def session_context():
        yield MagicMock()

    factory = MagicMock(side_effect=lambda: session_context())
    factory.begin = lambda: session_context()

    monkeypatch.setattr(resume_module, "AnalysisRepo", Repo)
    monkeypatch.setattr(resume_module, "build_workflow", build)
    return _Harness(thread_id, draft_id, run, fake_workflow, statuses, built, claims, factory)


async def test_resume_approval_run_completes_run_on_the_same_thread(monkeypatch) -> None:
    harness = _harness(monkeypatch)

    result = await resume_module.resume_approval_run(**harness.kwargs())

    assert result.status is AnalysisRunStatus.COMPLETED
    assert result.analysis_run_id == harness.run.id
    assert result.thread_id == harness.thread_id
    assert result.decision == ApprovalDecision(decision="approved", draft_id=harness.draft_id)
    assert harness.statuses == [AnalysisRunStatus.COMPLETED]
    assert harness.claims == [harness.run.id]
    assert len(harness.built) == 1
    assert harness.built[0]["thread_id"] == harness.thread_id
    assert harness.built[0]["analysis_run_id"] == harness.run.id
    assert harness.fake_workflow.resumed_with == [
        {"decision": "approved", "draft_id": str(harness.draft_id)}
    ]
    # No draft mutation: approval transitions belong to Sub-slice 2.1.2.
    assert harness.run.action_drafts[0].status is ActionDraftStatus.PENDING_APPROVAL


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_status": AnalysisRunStatus.RUNNING},
        {"run_status": AnalysisRunStatus.COMPLETED},
        {"draft_status": ActionDraftStatus.REJECTED},
        {"draft_status": ActionDraftStatus.EXECUTED},
        {"draft_run_id": uuid.uuid4()},
    ],
)
async def test_resume_approval_run_refuses_ineligible_runs(monkeypatch, overrides) -> None:
    harness = _harness(monkeypatch, **overrides)

    with pytest.raises(ApprovalStateError):
        await resume_module.resume_approval_run(**harness.kwargs())

    assert harness.built == []
    assert harness.claims == []
    assert harness.statuses == []
    assert harness.fake_workflow.resumed_with == []


async def test_resume_approval_run_requires_a_gate_pause(monkeypatch) -> None:
    harness = _harness(monkeypatch, paused=False)

    with pytest.raises(ApprovalStateError, match="not paused"):
        await resume_module.resume_approval_run(**harness.kwargs())

    assert harness.claims == []
    assert harness.fake_workflow.resumed_with == []


async def test_resume_approval_run_rejects_draft_outside_the_pending_request(monkeypatch) -> None:
    harness = _harness(monkeypatch)
    assert harness.fake_workflow.request is not None
    harness.fake_workflow.request = harness.fake_workflow.request.model_copy(
        update={
            "drafts": [
                ApprovalDraftSummary(
                    draft_id=uuid.uuid4(),
                    action_type=ActionType.TRACKER_UPSERT,
                    summary="Other",
                )
            ]
        }
    )

    with pytest.raises(ApprovalStateError, match="pending approval request"):
        await resume_module.resume_approval_run(**harness.kwargs())

    assert harness.claims == []


async def test_resume_approval_run_refuses_a_lost_claim(monkeypatch) -> None:
    harness = _harness(monkeypatch, claimed=False)

    with pytest.raises(ApprovalStateError, match="no longer awaiting approval"):
        await resume_module.resume_approval_run(**harness.kwargs())

    assert harness.fake_workflow.resumed_with == []
    assert harness.statuses == []


async def test_resume_approval_run_reports_node_failure_as_failed(monkeypatch) -> None:
    harness = _harness(
        monkeypatch,
        resumed_state={
            "errors": [
                {"node": "approval_gate", "code": "node_failed", "message": "x", "retryable": False}
            ]
        },
    )

    result = await resume_module.resume_approval_run(**harness.kwargs())

    assert result.status is AnalysisRunStatus.FAILED
    assert result.state["errors"]
    assert harness.statuses == [AnalysisRunStatus.FAILED]


async def test_resume_approval_run_sanitizes_process_failures(monkeypatch) -> None:
    harness = _harness(monkeypatch, resumed_state=RuntimeError("private secret"))

    with pytest.raises(resume_module.ApprovalError, match="could not be completed") as excinfo:
        await resume_module.resume_approval_run(**harness.kwargs())

    assert "private secret" not in str(excinfo.value)
    assert harness.statuses == [AnalysisRunStatus.FAILED]


async def test_resume_approval_run_rejects_missing_or_mismatched_decisions(monkeypatch) -> None:
    for state, message in (
        ({"errors": []}, "without recording a decision"),
        (
            {
                "errors": [],
                "approval_decision": ApprovalDecision(decision="rejected", draft_id=uuid.uuid4()),
            },
            "not requested",
        ),
    ):
        harness = _harness(monkeypatch, resumed_state=state)
        with pytest.raises(resume_module.ApprovalError, match=message):
            await resume_module.resume_approval_run(**harness.kwargs())
        assert harness.statuses == [AnalysisRunStatus.FAILED]


@pytest.mark.parametrize(
    ("thread_id", "draft_id"),
    [("", uuid.uuid4()), ("thread", "not-a-uuid")],
)
async def test_resume_approval_run_validates_its_arguments(
    monkeypatch, thread_id, draft_id
) -> None:
    harness = _harness(monkeypatch)
    kwargs = harness.kwargs() | {"thread_id": thread_id, "draft_id": draft_id}

    with pytest.raises(ValueError):
        await resume_module.resume_approval_run(**kwargs)

    assert harness.built == []
