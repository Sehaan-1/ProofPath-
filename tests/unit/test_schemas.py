"""Unit tests for strict Pydantic v2 domain and tool-input schemas."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from proofpath.db.models import (
    ActionDraftStatus,
    ActionType,
    AnalysisRunStatus,
    ApplicationStage,
    EvidenceSourceType,
    RequirementCategory,
    RequirementPriority,
    VerdictType,
)
from proofpath.schemas import (
    TOOL_INPUT_REGISTRY,
    ActionDraftRecord,
    AnalysisReport,
    AnalysisRunCreateRequest,
    ApproveActionInput,
    CalendarBlockPayload,
    CalendarTaskRecord,
    ClaimVerdictSchema,
    ContributionSprintPayload,
    ContributionSprintPlan,
    CreateActionDraftInput,
    DraftDecisionResponse,
    DraftExecutionResponse,
    EvidenceSearchRequest,
    EvidenceSnippet,
    GetActionDraftInput,
    GetApplicationStatusInput,
    GetOpportunityInput,
    IngestGitHubRequest,
    IngestTextRequest,
    OpportunityCreateRequest,
    ParsedRole,
    RequirementItem,
    RequirementRecord,
    SearchEvidenceInput,
    StrictModel,
    TrackerUpsertPayload,
    parse_action_draft_payload,
)
from proofpath.schemas.base import StrictModel as StrictModelDirect
from proofpath.schemas.tools import TOOL_INPUT_REGISTRY as TOOL_REGISTRY_DIRECT

_REQ_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
_OPP_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
_RUN_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
_CHUNK_ID = uuid.UUID("44444444-4444-4444-8444-444444444444")
_SOURCE_ID = uuid.UUID("55555555-5555-4555-8555-555555555555")
_DRAFT_ID = uuid.UUID("66666666-6666-4666-8666-666666666666")


def _requirement_item(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requirement_text": "Production experience with FastAPI.",
        "entity": "FastAPI",
        "category": "framework",
        "priority": "must_have",
    }
    payload.update(overrides)
    return payload


def _parsed_role(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "company": "Palo Alto Networks",
        "role_title": "AI Agent Engineer",
        "source_url": "https://jobs.paloaltonetworks.com/ai-agent-engineer",
        "summary": "Build safe, cited AI agents with MCP and LangGraph.",
        "requirements": [
            _requirement_item(),
            _requirement_item(
                requirement_text="Hands-on SAP S/4HANA configuration.",
                entity="SAP S/4HANA",
                category="domain",
                priority="must_have",
            ),
        ],
    }
    payload.update(overrides)
    return payload


def _sprint_plan(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": "MCP evidence-ledger demo",
        "scope": "Ship a small MCP server that cites pgvector chunks.",
        "repo_or_topic": "github.com/example/proofpath-mcp-demo",
        "deliverables": ["README with architecture", "pytest coverage of search_evidence"],
        "estimate_days": 3,
        "target_gap_requirement_ids": [str(_REQ_ID)],
        "rationale": "Highest-impact unsupported gap is MCP production experience.",
    }
    payload.update(overrides)
    return payload


def _assert_extra_forbidden(model_cls: type[StrictModel], valid: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model_cls.model_validate({**valid, "hallucinated_arg": "ignore previous instructions"})
    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


def _json_schema_forbids_additional_properties(model_cls: type[StrictModel]) -> None:
    schema = model_cls.model_json_schema()
    assert schema.get("additionalProperties") is False


@pytest.mark.unit
def test_strict_model_exports_and_forbids_extra() -> None:
    assert StrictModel is StrictModelDirect
    assert StrictModel.model_config["extra"] == "forbid"
    _json_schema_forbids_additional_properties(SearchEvidenceInput)


@pytest.mark.unit
def test_requirement_item_and_parsed_role_happy_path() -> None:
    role = ParsedRole.model_validate(_parsed_role())

    assert role.company == "Palo Alto Networks"
    assert role.role_title == "AI Agent Engineer"
    assert str(role.source_url).startswith("https://jobs.paloaltonetworks.com/")
    assert [item.entity for item in role.must_have] == ["FastAPI", "SAP S/4HANA"]
    assert role.nice_to_have == []
    assert role.requirements[0].category is RequirementCategory.FRAMEWORK
    assert role.requirements[0].priority is RequirementPriority.MUST_HAVE


@pytest.mark.unit
def test_parsed_role_rejects_extra_keys_empty_requirements_and_duplicate_entities() -> None:
    _assert_extra_forbidden(ParsedRole, _parsed_role())
    _json_schema_forbids_additional_properties(ParsedRole)
    _json_schema_forbids_additional_properties(RequirementItem)

    with pytest.raises(ValidationError):
        ParsedRole.model_validate(_parsed_role(requirements=[]))

    with pytest.raises(ValidationError, match="duplicate entities"):
        ParsedRole.model_validate(
            _parsed_role(
                requirements=[
                    _requirement_item(),
                    _requirement_item(requirement_text="Also FastAPI services."),
                ]
            )
        )

    with pytest.raises(ValidationError):
        RequirementItem.model_validate(_requirement_item(category="not-a-category"))

    stripped = RequirementItem.model_validate(_requirement_item(entity="  FastAPI  "))
    assert stripped.entity == "FastAPI"


@pytest.mark.unit
def test_opportunity_create_request_requires_text_or_url() -> None:
    from_text = OpportunityCreateRequest.model_validate({"raw_text": "Build LangGraph agents."})
    assert from_text.source_url is None

    from_url = OpportunityCreateRequest.model_validate(
        {"source_url": "https://example.com/jobs/ai"}
    )
    assert from_url.raw_text is None

    with pytest.raises(ValidationError, match="raw_text or source_url"):
        OpportunityCreateRequest.model_validate({"company": "Acme"})

    _assert_extra_forbidden(OpportunityCreateRequest, {"raw_text": "JD text"})


@pytest.mark.unit
def test_evidence_snippet_and_ingest_requests() -> None:
    snippet = EvidenceSnippet.model_validate(
        {
            "chunk_id": str(_CHUNK_ID),
            "source_id": str(_SOURCE_ID),
            "canonical_uri": "https://github.com/sehaan/proofpath#readme",
            "content": "Built a FastAPI + pgvector evidence ledger.",
            "section_label": "Projects",
            "source_type": "github_repo",
            "score": 0.81,
        }
    )
    assert snippet.chunk_id == _CHUNK_ID
    assert snippet.source_type is EvidenceSourceType.GITHUB_REPO
    _assert_extra_forbidden(
        EvidenceSnippet,
        {
            "chunk_id": str(_CHUNK_ID),
            "source_id": str(_SOURCE_ID),
            "canonical_uri": "file:///tmp/resume.md",
            "content": "Python, FastAPI, MCP.",
        },
    )

    text_req = IngestTextRequest.model_validate(
        {
            "title": "Resume",
            "content": "# Experience\nPython, FastAPI.",
            "canonical_uri": "file:///home/user/resume.md",
        }
    )
    assert text_req.source_type is EvidenceSourceType.EXPERIENCE_NOTE

    github_req = IngestGitHubRequest.model_validate({"owner": "sehaan", "name": "ProofPath-"})
    assert github_req.repo_url is None

    with pytest.raises(ValidationError, match="repo_url or both owner and name"):
        IngestGitHubRequest.model_validate({})

    search = EvidenceSearchRequest.model_validate({"query": "MCP FastAPI"})
    assert search.top_k == 8
    with pytest.raises(ValidationError):
        EvidenceSearchRequest.model_validate({"query": "MCP", "top_k": 0})


@pytest.mark.unit
def test_claim_verdict_schema_allows_unsupported_without_citations() -> None:
    unsupported = ClaimVerdictSchema.model_validate(
        {
            "requirement_id": str(_REQ_ID),
            "verdict": "unsupported",
            "claim_summary": "No SAP evidence exists in the ledger.",
        }
    )
    assert unsupported.cited_chunk_ids == []
    assert unsupported.policy_downgraded is False
    assert unsupported.needs_citation_downgrade() is False

    supported_without_cites = ClaimVerdictSchema.model_validate(
        {
            "requirement_id": str(_REQ_ID),
            "verdict": "supported",
            "claim_summary": "Hallucinated support with no citations.",
        }
    )
    assert supported_without_cites.needs_citation_downgrade() is True

    supported = ClaimVerdictSchema.model_validate(
        {
            "requirement_id": str(_REQ_ID),
            "verdict": "supported",
            "claim_summary": "Resume project section cites FastAPI production use.",
            "cited_chunk_ids": [str(_CHUNK_ID), str(_CHUNK_ID)],
        }
    )
    assert supported.cited_chunk_ids == [_CHUNK_ID]
    assert supported.needs_citation_downgrade() is False
    assert supported.verdict is VerdictType.SUPPORTED
    _assert_extra_forbidden(
        ClaimVerdictSchema,
        {
            "requirement_id": str(_REQ_ID),
            "verdict": "partial",
            "claim_summary": "Some overlap.",
            "cited_chunk_ids": [str(_CHUNK_ID)],
        },
    )


@pytest.mark.unit
def test_contribution_sprint_plan_bounds_and_deliverables() -> None:
    plan = ContributionSprintPlan.model_validate(_sprint_plan())
    assert plan.estimate_days == 3
    assert plan.target_gap_requirement_ids == [_REQ_ID]

    with pytest.raises(ValidationError):
        ContributionSprintPlan.model_validate(_sprint_plan(estimate_days=1))
    with pytest.raises(ValidationError):
        ContributionSprintPlan.model_validate(_sprint_plan(estimate_days=6))
    with pytest.raises(ValidationError, match="blank items"):
        ContributionSprintPlan.model_validate(_sprint_plan(deliverables=["README", "  "]))
    with pytest.raises(ValidationError, match="unique"):
        ContributionSprintPlan.model_validate(
            _sprint_plan(target_gap_requirement_ids=[str(_REQ_ID), str(_REQ_ID)])
        )
    _assert_extra_forbidden(ContributionSprintPlan, _sprint_plan())


@pytest.mark.unit
def test_action_draft_payload_discriminated_union() -> None:
    tracker = parse_action_draft_payload(
        {
            "action_type": "tracker_upsert",
            "opportunity_id": str(_OPP_ID),
            "stage": "preparing",
            "notes": "Drafting MCP sprint first.",
        }
    )
    assert isinstance(tracker, TrackerUpsertPayload)
    assert tracker.action_type == ActionType.TRACKER_UPSERT
    assert tracker.stage is ApplicationStage.PREPARING

    start = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
    calendar = parse_action_draft_payload(
        {
            "action_type": "calendar_block",
            "title": "Contribution sprint",
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=3)).isoformat(),
        }
    )
    assert isinstance(calendar, CalendarBlockPayload)
    assert calendar.timezone == "UTC"

    sprint = parse_action_draft_payload(
        {
            "action_type": "contribution_sprint",
            "plan": _sprint_plan(),
        }
    )
    assert isinstance(sprint, ContributionSprintPayload)
    assert sprint.plan.title == "MCP evidence-ledger demo"

    with pytest.raises(ValidationError):
        parse_action_draft_payload(
            {
                "action_type": "tracker_upsert",
                "opportunity_id": str(_OPP_ID),
                "stage": "preparing",
                "execute_now": True,
            }
        )

    with pytest.raises(ValidationError):
        parse_action_draft_payload(
            {
                "action_type": "submit_application",
                "opportunity_id": str(_OPP_ID),
            }
        )

    naive = start.replace(tzinfo=None)
    with pytest.raises(ValidationError):
        CalendarBlockPayload.model_validate(
            {
                "action_type": "calendar_block",
                "title": "Sprint",
                "start_at": naive.isoformat(),
                "end_at": (naive + timedelta(hours=1)).isoformat(),
            }
        )

    with pytest.raises(ValidationError, match="end_at"):
        parse_action_draft_payload(
            {
                "action_type": "calendar_block",
                "title": "Sprint",
                "start_at": start.isoformat(),
                "end_at": start.isoformat(),
            }
        )


@pytest.mark.unit
def test_analysis_report_groups_verdicts_and_metrics() -> None:
    requirement = RequirementRecord.model_validate(
        {
            **_requirement_item(),
            "id": str(_REQ_ID),
            "opportunity_id": str(_OPP_ID),
        }
    )
    report = AnalysisReport.model_validate(
        {
            "analysis_run_id": str(_RUN_ID),
            "opportunity_id": str(_OPP_ID),
            "thread_id": "thread-panw-001",
            "status": "awaiting_approval",
            "role": _parsed_role(),
            "requirements": [requirement.model_dump(mode="json")],
            "verdicts": [
                {
                    "requirement_id": str(_REQ_ID),
                    "verdict": "supported",
                    "claim_summary": "FastAPI is cited in the resume projects section.",
                    "cited_chunk_ids": [str(_CHUNK_ID)],
                },
                {
                    "requirement_id": str(uuid.uuid4()),
                    "verdict": "unsupported",
                    "claim_summary": "No SAP evidence.",
                    "policy_downgraded": True,
                    "gap_notes": "Ledger has no SAP S/4HANA artifacts.",
                },
                {
                    "requirement_id": str(uuid.uuid4()),
                    "verdict": "partial",
                    "claim_summary": "MCP mentioned in a README only.",
                    "cited_chunk_ids": [str(_CHUNK_ID)],
                },
            ],
            "contribution_sprint": _sprint_plan(),
            "action_drafts": [
                {
                    "id": str(_DRAFT_ID),
                    "analysis_run_id": str(_RUN_ID),
                    "action_type": "tracker_upsert",
                    "status": "pending_approval",
                    "payload": {
                        "action_type": "tracker_upsert",
                        "opportunity_id": str(_OPP_ID),
                        "stage": "preparing",
                    },
                    "idempotency_key": "thread-panw-001:tracker_upsert",
                }
            ],
            "total_tokens": 1200,
            "estimated_cost_usd": "0.042000",
            "latency_ms": 1500,
            "trace_id": "trace-abc",
        }
    )

    assert report.status is AnalysisRunStatus.AWAITING_APPROVAL
    assert len(report.supported) == 1
    assert len(report.partial) == 1
    assert len(report.unsupported) == 1
    assert report.policy_downgraded_count == 1
    assert report.estimated_cost_usd == Decimal("0.042000")
    assert report.contribution_sprint is not None
    assert report.action_drafts[0].status is ActionDraftStatus.PENDING_APPROVAL
    assert isinstance(report.action_drafts[0], ActionDraftRecord)
    _assert_extra_forbidden(
        AnalysisReport,
        {
            "analysis_run_id": str(_RUN_ID),
            "opportunity_id": str(_OPP_ID),
            "thread_id": "thread-1",
            "status": "completed",
            "role": _parsed_role(),
        },
    )
    # Silence unused start if I didn't use it — actually I didn't use start. That's fine.


@pytest.mark.unit
def test_analysis_run_create_request_defaults() -> None:
    request = AnalysisRunCreateRequest.model_validate({"opportunity_id": str(_OPP_ID)})
    assert request.async_run is False
    _assert_extra_forbidden(AnalysisRunCreateRequest, {"opportunity_id": str(_OPP_ID)})


@pytest.mark.unit
def test_draft_decision_response_contract() -> None:
    payload = {
        "draft": {
            "id": str(_DRAFT_ID),
            "analysis_run_id": str(_RUN_ID),
            "action_type": "tracker_upsert",
            "status": "approved",
            "payload": {
                "action_type": "tracker_upsert",
                "opportunity_id": str(_OPP_ID),
                "stage": "preparing",
            },
            "idempotency_key": f"proofpath:{_RUN_ID}:tracker_upsert",
            "approved_at": "2026-01-01T00:00:00Z",
        },
        "decision": "approved",
        "already_processed": True,
        "run_status": "completed",
        "thread_id": "thread-1",
    }
    response = DraftDecisionResponse.model_validate(payload)
    assert response.already_processed is True
    assert response.run_status is AnalysisRunStatus.COMPLETED
    assert response.draft.id == _DRAFT_ID

    for decision in ("approved", "rejected"):
        assert (
            DraftDecisionResponse.model_validate(payload | {"decision": decision}).decision
            == decision
        )
    with pytest.raises(ValidationError):
        DraftDecisionResponse.model_validate(payload | {"decision": "maybe"})
    with pytest.raises(ValidationError):
        DraftDecisionResponse.model_validate(payload | {"run_status": "pending"})
    with pytest.raises(ValidationError):
        DraftDecisionResponse.model_validate(payload | {"thread_id": ""})
    _assert_extra_forbidden(DraftDecisionResponse, payload)


@pytest.mark.unit
def test_draft_decision_response_accepts_an_execution_report() -> None:
    payload = {
        "draft": {
            "id": str(_DRAFT_ID),
            "analysis_run_id": str(_RUN_ID),
            "action_type": "tracker_upsert",
            "status": "executed",
            "payload": {
                "action_type": "tracker_upsert",
                "opportunity_id": str(_OPP_ID),
                "stage": "preparing",
            },
            "idempotency_key": f"proofpath:{_RUN_ID}:tracker_upsert",
            "approved_at": "2026-01-01T00:00:00Z",
            "executed_at": "2026-01-01T00:00:05Z",
        },
        "decision": "approved",
        "already_processed": False,
        "run_status": "completed",
        "thread_id": "thread-1",
        "execution": {
            "draft_id": str(_DRAFT_ID),
            "action_type": "tracker_upsert",
            "outcome": "executed",
            "approved_at": "2026-01-01T00:00:00Z",
            "executed_at": "2026-01-01T00:00:05Z",
            "audit_log_id": str(uuid.uuid4()),
        },
    }

    response = DraftDecisionResponse.model_validate(payload)

    assert response.execution is not None
    assert response.execution.outcome == "executed"
    assert response.execution.error_code is None
    assert DraftDecisionResponse.model_validate(payload | {"execution": None}).execution is None


@pytest.mark.unit
def test_draft_execution_response_validates_its_outcome_contract() -> None:
    base = {
        "draft_id": str(_DRAFT_ID),
        "action_type": "calendar_block",
        "outcome": "executed",
        "approved_at": "2026-01-01T00:00:00Z",
        "executed_at": "2026-01-01T00:00:05Z",
        "calendar_task_id": str(uuid.uuid4()),
    }
    record = DraftExecutionResponse.model_validate(base)
    assert record.outcome == "executed"

    # `already_executed` replays carry no new executed_at but stay valid.
    replay = DraftExecutionResponse.model_validate(
        base | {"outcome": "already_executed", "executed_at": None}
    )
    assert replay.outcome == "already_executed"

    # Failures must name a reason and must not pretend anything executed.
    failed = DraftExecutionResponse.model_validate(
        base | {"outcome": "failed", "executed_at": None, "error_code": "invalid_payload"}
    )
    assert failed.error_code == "invalid_payload"
    with pytest.raises(ValidationError, match="error_code"):
        DraftExecutionResponse.model_validate(base | {"outcome": "failed", "executed_at": None})
    with pytest.raises(ValidationError, match="executed_at"):
        DraftExecutionResponse.model_validate(base | {"outcome": "failed", "error_code": "x"})
    with pytest.raises(ValidationError, match="error_code"):
        DraftExecutionResponse.model_validate(base | {"error_code": "x"})
    with pytest.raises(ValidationError):
        DraftExecutionResponse.model_validate(base | {"outcome": "succeeded"})
    _assert_extra_forbidden(DraftExecutionResponse, base)


@pytest.mark.unit
def test_calendar_task_record_contract() -> None:
    payload = {
        "id": str(uuid.uuid4()),
        "action_draft_id": str(_DRAFT_ID),
        "opportunity_id": str(_OPP_ID),
        "title": "Interview prep block",
        "description": None,
        "start_at": "2026-12-01T09:00:00Z",
        "end_at": "2026-12-01T10:30:00Z",
        "timezone": "Asia/Kolkata",
        "ics_uid": f"proofpath-{_DRAFT_ID}@proofpath.local",
        "created_at": "2026-11-28T11:00:00Z",
        "invite_url": f"/api/v1/calendar-tasks/{_DRAFT_ID}/invite.ics",
    }
    record = CalendarTaskRecord.model_validate(payload)
    assert record.timezone == "Asia/Kolkata"
    assert record.ics_uid.startswith("proofpath-")
    _assert_extra_forbidden(CalendarTaskRecord, payload)


@pytest.mark.unit
def test_tool_input_registry_forbids_hallucinated_arguments() -> None:
    assert TOOL_INPUT_REGISTRY is TOOL_REGISTRY_DIRECT
    expected_tools = {
        "search_evidence",
        "get_opportunity",
        "get_application_status",
        "get_action_draft",
        "create_action_draft",
        "approve_action",
    }
    assert set(TOOL_INPUT_REGISTRY) == expected_tools

    valid_payloads: dict[str, dict[str, Any]] = {
        "search_evidence": {"query": "FastAPI MCP pgvector"},
        "get_opportunity": {"opportunity_id": str(_OPP_ID)},
        "get_application_status": {"opportunity_id": str(_OPP_ID)},
        "get_action_draft": {"draft_id": str(_DRAFT_ID)},
        "create_action_draft": {
            "analysis_run_id": str(_RUN_ID),
            "idempotency_key": "thread-1:tracker_upsert",
            "payload": {
                "action_type": "tracker_upsert",
                "opportunity_id": str(_OPP_ID),
                "stage": "saved",
            },
        },
        "approve_action": {
            "draft_id": str(_DRAFT_ID),
            "confirmation_phrase": "APPROVE",
        },
    }

    for tool_name, model_cls in TOOL_INPUT_REGISTRY.items():
        assert model_cls.model_config["extra"] == "forbid"
        _json_schema_forbids_additional_properties(model_cls)
        valid = valid_payloads[tool_name]
        model_cls.model_validate(valid)
        _assert_extra_forbidden(model_cls, valid)

    # Forbidden mutating tools must not appear in the input registry.
    for banned in ("submit_application", "send_email", "delete_tracker"):
        assert banned not in TOOL_INPUT_REGISTRY


@pytest.mark.unit
def test_search_evidence_and_approve_action_field_bounds() -> None:
    search = SearchEvidenceInput.model_validate({"query": "  LangGraph interrupts  "})
    assert search.query == "LangGraph interrupts"
    assert search.top_k == 8

    with pytest.raises(ValidationError):
        SearchEvidenceInput.model_validate({"query": "x", "top_k": 51})

    approve = ApproveActionInput.model_validate(
        {"draft_id": str(_DRAFT_ID), "confirmation_phrase": "APPROVE"}
    )
    assert approve.draft_id == _DRAFT_ID

    with pytest.raises(ValidationError):
        ApproveActionInput.model_validate(
            {"draft_id": str(_DRAFT_ID), "confirmation_phrase": "   "}
        )

    GetOpportunityInput.model_validate({"opportunity_id": str(_OPP_ID)})
    GetApplicationStatusInput.model_validate({"opportunity_id": str(_OPP_ID)})
    GetActionDraftInput.model_validate({"draft_id": str(_DRAFT_ID)})


@pytest.mark.unit
def test_create_action_draft_input_rejects_unknown_payload_fields() -> None:
    payload = {
        "analysis_run_id": str(_RUN_ID),
        "idempotency_key": "thread-1:calendar_block",
        "payload": {
            "action_type": "calendar_block",
            "title": "Sprint block",
            "start_at": "2026-12-01T09:00:00+00:00",
            "end_at": "2026-12-04T09:00:00+00:00",
            "send_invite_now": True,
        },
    }
    with pytest.raises(ValidationError) as exc_info:
        CreateActionDraftInput.model_validate(payload)
    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


@pytest.mark.unit
def test_json_roundtrip_for_core_llm_contracts() -> None:
    role = ParsedRole.model_validate(_parsed_role())
    restored_role = ParsedRole.model_validate_json(role.model_dump_json())
    assert restored_role == role

    verdict = ClaimVerdictSchema(
        requirement_id=_REQ_ID,
        verdict=VerdictType.PARTIAL,
        claim_summary="README mentions MCP without production proof.",
        cited_chunk_ids=[_CHUNK_ID],
        gap_notes="Need a shipped MCP server.",
    )
    restored_verdict = ClaimVerdictSchema.model_validate_json(verdict.model_dump_json())
    assert restored_verdict == verdict

    plan = ContributionSprintPlan.model_validate(_sprint_plan())
    restored_plan = ContributionSprintPlan.model_validate_json(plan.model_dump_json())
    assert restored_plan == plan

    dumped = json.loads(SearchEvidenceInput(query="pgvector HNSW").model_dump_json())
    assert dumped == {"query": "pgvector HNSW", "top_k": 8}
