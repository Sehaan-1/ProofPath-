"""Approval-gated MCP tools (Sub-slice 2.1.4): `create_action_draft` and `approve_action`.

The tools are driven through a real MCP `Client` so the wire contract (strict
`extra='forbid'` schemas, structured output, `is_error`) is exercised, with the
repositories and the shared approval service replaced by recording fakes. The gate
itself is what these tests pin: nothing executes without a `pending_approval` draft
*and* the exact confirmation phrase, and `create_action_draft` never executes at all.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import Client

from proofpath import mcp_server
from proofpath.approvals import DraftConflictError, DraftDecisionResult, DraftNotFoundError
from proofpath.config import Settings
from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
)
from proofpath.execution import DraftExecution, ExecutionOutcome
from proofpath.graph.dependencies import GraphDependenciesError
from proofpath.mcp_server import action_tools, create_mcp_server
from proofpath.policy.tool_allowlist import APPROVAL_CONFIRMATION_PHRASE
from proofpath.schemas.actions import ActionDraftRecord, TrackerUpsertPayload

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)


class World:
    """Fake persistence plus every call the tools make, recorded."""

    def __init__(
        self,
        *,
        status: ActionDraftStatus = ActionDraftStatus.PENDING_APPROVAL,
        run_status: AnalysisRunStatus = AnalysisRunStatus.COMPLETED,
        existing_draft: SimpleNamespace | None = None,
        created: bool = True,
        decide_error: Exception | None = None,
        already_processed: bool = False,
    ) -> None:
        self.run_id = uuid.uuid4()
        self.opportunity_id = uuid.uuid4()
        self.draft_id = uuid.uuid4()
        self.run = SimpleNamespace(
            id=self.run_id,
            opportunity_id=self.opportunity_id,
            thread_id="thread-1",
            status=run_status,
        )
        self.draft = SimpleNamespace(
            id=self.draft_id,
            analysis_run_id=self.run_id,
            action_type=ActionType.TRACKER_UPSERT,
            status=status,
            payload_json={
                "action_type": "tracker_upsert",
                "opportunity_id": str(self.opportunity_id),
                "stage": ApplicationStage.PREPARING.value,
                "notes": "Tailored resume drafted.",
            },
            idempotency_key=f"proofpath:{self.run_id}:tracker_upsert",
            approved_at=(
                NOW if status in {ActionDraftStatus.APPROVED, ActionDraftStatus.EXECUTED} else None
            ),
            executed_at=None,
        )
        self.existing_draft = existing_draft
        self.created = created
        self.decide_error = decide_error
        self.already_processed = already_processed
        self.commits = 0
        self.decide_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    # --- fake repositories --------------------------------------------
    def draft_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def get_by_id(self, draft_id: uuid.UUID) -> Any:
                return world.draft if draft_id == world.draft_id else None

            async def create_or_get_draft(self, **kwargs: Any) -> tuple[Any, bool]:
                world.create_calls.append(kwargs)
                if not world.created and world.existing_draft is not None:
                    return world.existing_draft, False
                return world.draft, world.created

        return Repo

    def analysis_repo(self) -> Any:
        world = self

        class Repo:
            def __init__(self, _session: object) -> None:
                pass

            async def get_run_by_id(self, run_id: uuid.UUID, **_kwargs: object) -> Any:
                return world.run if run_id == world.run_id else None

        return Repo

    # --- fake services -------------------------------------------------
    async def decide(self, **kwargs: Any) -> DraftDecisionResult:
        self.decide_calls.append(kwargs)
        if self.decide_error is not None:
            raise self.decide_error
        draft = ActionDraftRecord(
            id=self.draft_id,
            analysis_run_id=self.run_id,
            action_type=ActionType.TRACKER_UPSERT,
            status=ActionDraftStatus.EXECUTED,
            payload=TrackerUpsertPayload(
                action_type=ActionType.TRACKER_UPSERT,
                opportunity_id=self.opportunity_id,
                stage=ApplicationStage.PREPARING,
            ),
            idempotency_key=f"proofpath:{self.run_id}:tracker_upsert",
            approved_at=NOW,
            executed_at=NOW,
        )
        return DraftDecisionResult(
            draft=draft,
            decision="approved",
            already_processed=self.already_processed,
            run_status=AnalysisRunStatus.COMPLETED,
            thread_id="thread-1",
            execution=DraftExecution(
                draft_id=self.draft_id,
                action_type=ActionType.TRACKER_UPSERT,
                outcome=(
                    ExecutionOutcome.ALREADY_EXECUTED
                    if self.already_processed
                    else ExecutionOutcome.EXECUTED
                ),
                approved_at=NOW,
                executed_at=NOW,
                audit_log_id=uuid.uuid4(),
            ),
        )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> Any:
    def build(**kwargs: Any) -> World:
        instance = World(**kwargs)
        monkeypatch.setattr(action_tools, "DraftRepo", instance.draft_repo())
        monkeypatch.setattr(action_tools, "AnalysisRepo", instance.analysis_repo())
        monkeypatch.setattr(action_tools, "decide_draft", instance.decide)
        return instance

    return build


def _session() -> SimpleNamespace:
    async def commit() -> None:
        return None

    return SimpleNamespace(commit=commit)


def _server(world: World, *, session_maker: Any = "default") -> Any:
    session = _session()

    @asynccontextmanager
    async def open_session():
        yield session

    @asynccontextmanager
    async def open_transaction():
        yield session

    if session_maker == "default":
        maker: Any = open_transaction
    elif session_maker == "none":
        maker = None
    else:
        maker = session_maker
    return create_mcp_server(
        settings=Settings(_env_file=None, embedding_model="hash", llm_api_key=None),
        session_factory=open_session,
        session_maker=maker,
    )


def _payload(world: World, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action_type": "tracker_upsert",
        "opportunity_id": str(world.opportunity_id),
        "stage": ApplicationStage.PREPARING.value,
        "notes": "Tailored resume drafted.",
    }
    payload.update(overrides)
    return payload


def _key(world: World, action_type: str = "tracker_upsert") -> str:
    return f"proofpath:{world.run_id}:{action_type}"


# ------------------------------------------------------- create_action_draft


async def test_create_action_draft_persists_a_pending_draft_and_nothing_else(
    world: Any,
) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(active.run_id),
                "payload": _payload(active),
                "idempotency_key": _key(active),
            },
        )

    assert result.is_error is False, result.content
    body = result.structured_content
    assert body is not None
    assert body["created"] is True
    assert body["requires_approval"] is True
    assert body["draft"]["status"] == "pending_approval"
    assert body["draft"]["id"] == str(active.draft_id)
    assert body["draft"]["approved_at"] is None
    # The draft is stored under the canonical key for its run and action type.
    assert active.create_calls[0]["idempotency_key"] == _key(active)
    assert active.create_calls[0]["action_type"] is ActionType.TRACKER_UPSERT
    assert active.create_calls[0]["payload_json"]["stage"] == "preparing"
    # Creating a draft never decides or executes anything.
    assert active.decide_calls == []


async def test_create_action_draft_returns_the_existing_draft_unchanged(world: Any) -> None:
    active = world(created=False)
    existing = SimpleNamespace(
        id=uuid.uuid4(),
        analysis_run_id=active.run_id,
        action_type=ActionType.TRACKER_UPSERT,
        status=ActionDraftStatus.PENDING_APPROVAL,
        payload_json=_payload(active),
        idempotency_key=_key(active),
        approved_at=None,
        executed_at=None,
    )
    active.existing_draft = existing
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(active.run_id),
                "payload": _payload(active),
                "idempotency_key": _key(active),
            },
        )

    assert result.is_error is False, result.content
    assert result.structured_content is not None
    assert result.structured_content["created"] is False
    assert result.structured_content["draft"]["id"] == str(existing.id)
    assert active.decide_calls == []


async def test_create_action_draft_refuses_to_reuse_a_key_for_a_different_payload(
    world: Any,
) -> None:
    active = world(created=False)
    active.existing_draft = SimpleNamespace(
        id=uuid.uuid4(),
        analysis_run_id=active.run_id,
        action_type=ActionType.TRACKER_UPSERT,
        status=ActionDraftStatus.PENDING_APPROVAL,
        payload_json=_payload(active, stage=ApplicationStage.READY_TO_APPLY.value),
        idempotency_key=_key(active),
        approved_at=None,
        executed_at=None,
    )
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(active.run_id),
                "payload": _payload(active),
                "idempotency_key": _key(active),
            },
        )

    assert result.is_error is True
    assert "different payload" in str(result.content)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"analysis_run_id": "00000000-0000-4000-8000-000000000000"}, "not found"),
        ({"idempotency_key": "proofpath:custom-key"}, "canonical draft key"),
        ({"payload": {"action_type": "tracker_upsert", "stage": "preparing"}}, None),
    ],
)
async def test_create_action_draft_validates_run_key_and_payload(
    world: Any,
    overrides: dict[str, Any],
    expected: str | None,
) -> None:
    active = world()
    server = _server(active)
    args: dict[str, Any] = {
        "analysis_run_id": str(active.run_id),
        "payload": _payload(active),
        "idempotency_key": _key(active),
    }
    args.update(overrides)

    async with Client(server) as client:
        result = await client.call_tool("create_action_draft", args)

    assert result.is_error is True
    if expected is not None:
        assert expected in str(result.content).lower()
    assert active.create_calls == []


async def test_create_action_draft_refuses_a_run_paused_at_the_gate(world: Any) -> None:
    active = world(run_status=AnalysisRunStatus.AWAITING_APPROVAL)
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(active.run_id),
                "payload": _payload(active),
                "idempotency_key": _key(active),
            },
        )

    assert result.is_error is True
    assert "approval gate" in str(result.content)
    assert active.create_calls == []


async def test_create_action_draft_rejects_a_tracker_payload_for_another_opportunity(
    world: Any,
) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(active.run_id),
                "payload": _payload(active, opportunity_id=str(uuid.uuid4())),
                "idempotency_key": _key(active),
            },
        )

    assert result.is_error is True
    assert "must match" in str(result.content)
    assert active.create_calls == []


async def test_create_action_draft_rejects_hallucinated_arguments(world: Any) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "create_action_draft",
            {
                "analysis_run_id": str(active.run_id),
                "payload": _payload(active),
                "idempotency_key": _key(active),
                "execute_now": True,
            },
        )

    assert result.is_error is True
    assert active.create_calls == []


# ------------------------------------------------------------ approve_action


async def test_approve_action_requires_the_exact_confirmation_phrase(world: Any) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        for phrase in ("", "   ", "approve", "approve all", f"approve {active.draft_id}"):
            result = await client.call_tool(
                "approve_action",
                {"draft_id": str(active.draft_id), "confirmation_phrase": phrase},
            )
            if phrase.strip() == "":
                assert result.is_error is True  # schema-level minimum length
                continue
            assert result.is_error is True
            assert "confirmation phrase" in str(result.content).lower()

    assert active.decide_calls == []


async def test_approve_action_executes_a_pending_draft_once(world: Any) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is False, result.content
    body = result.structured_content
    assert body is not None
    assert body["decision"] == "approved"
    assert body["already_processed"] is False
    assert body["draft"]["status"] == "executed"
    assert body["execution"]["outcome"] == "executed"

    assert len(active.decide_calls) == 1
    call = active.decide_calls[0]
    assert call["draft_id"] == active.draft_id
    assert call["decision"] == "approved"
    # A finalised run needs no resume wiring, so a key-less server can approve it.
    assert call["checkpointer"] is None
    assert call["dependencies"] is None


async def test_approve_action_normalizes_whitespace_and_case(world: Any) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {"draft_id": str(active.draft_id), "confirmation_phrase": "  Approve   ACTION "},
        )

    assert result.is_error is False, result.content
    assert len(active.decide_calls) == 1


async def test_repeated_approval_reports_no_second_execution(world: Any) -> None:
    active = world(status=ActionDraftStatus.EXECUTED, already_processed=True)
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is False, result.content
    assert result.structured_content is not None
    assert result.structured_content["already_processed"] is True
    assert result.structured_content["execution"]["outcome"] == "already_executed"


@pytest.mark.parametrize("status", [ActionDraftStatus.REJECTED, ActionDraftStatus.FAILED])
async def test_approve_action_rejects_drafts_that_are_not_approvable(
    world: Any,
    status: ActionDraftStatus,
) -> None:
    active = world(status=status)
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is True
    assert status.value in str(result.content)
    assert active.decide_calls == []


async def test_approve_action_rejects_an_unknown_draft(world: Any) -> None:
    active = world()
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(uuid.uuid4()),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is True
    assert "not found" in str(result.content)
    assert active.decide_calls == []


async def test_approve_action_fails_closed_when_the_model_is_unconfigured(
    world: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = world(run_status=AnalysisRunStatus.AWAITING_APPROVAL)
    server = _server(active)

    def unconfigured(_settings: Settings) -> Any:
        raise GraphDependenciesError("Analysis model is not configured.")

    monkeypatch.setattr(action_tools, "create_graph_dependencies", unconfigured)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is True
    assert "not configured" in str(result.content)
    # The decision is never written when the paused run cannot be finished.
    assert active.decide_calls == []


async def test_approve_action_resumes_a_paused_run_with_graph_wiring(
    world: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = world(run_status=AnalysisRunStatus.AWAITING_APPROVAL)
    server = _server(active)
    dependencies = SimpleNamespace(parser=object(), embedder=object())
    opened: list[str] = []

    @asynccontextmanager
    async def fake_checkpointer(**_kwargs: Any):
        opened.append("enter")
        try:
            yield "checkpointer"
        finally:
            opened.append("exit")

    monkeypatch.setattr(action_tools, "create_graph_dependencies", lambda _s: dependencies)
    monkeypatch.setattr(action_tools, "postgres_checkpointer", fake_checkpointer)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is False, result.content
    assert opened == ["enter", "exit"]
    call = active.decide_calls[0]
    assert call["checkpointer"] == "checkpointer"
    assert call["dependencies"] is dependencies


async def test_approve_action_requires_a_session_maker(world: Any) -> None:
    active = world()
    # A server wired with only a per-call session factory cannot run a decision
    # (it spans transactions), so the approval tool must refuse up front.
    server = _server(active, session_maker="none")

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is True
    assert "session factory" in str(result.content)
    assert active.decide_calls == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DraftNotFoundError("gone"), "not found"),
        (DraftConflictError("raced"), "concurrently"),
    ],
)
async def test_approve_action_maps_service_errors_to_tool_errors(
    world: Any,
    error: Exception,
    expected: str,
) -> None:
    active = world(decide_error=error)
    server = _server(active)

    async with Client(server) as client:
        result = await client.call_tool(
            "approve_action",
            {
                "draft_id": str(active.draft_id),
                "confirmation_phrase": APPROVAL_CONFIRMATION_PHRASE,
            },
        )

    assert result.is_error is True
    assert expected in str(result.content)
    assert len(active.decide_calls) == 1


# ------------------------------------------------------------------ helpers


def test_canonical_idempotency_key_matches_the_draft_convention() -> None:
    run_id = uuid.uuid4()
    assert (
        action_tools.canonical_idempotency_key(run_id, ActionType.CALENDAR_BLOCK)
        == f"proofpath:{run_id}:calendar_block"
    )


def test_confirmation_phrase_normalization_only_ignores_case_and_spacing() -> None:
    assert action_tools.normalize_confirmation_phrase("  Approve \n action ") == "approve action"
    assert action_tools.normalize_confirmation_phrase("approve-action") != "approve action"
    assert APPROVAL_CONFIRMATION_PHRASE == "approve action"


def test_mcp_server_package_is_importable_without_wiring() -> None:
    assert mcp_server.create_mcp_server is create_mcp_server
