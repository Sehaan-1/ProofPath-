"""Unit tests for SQLAlchemy 2.0 ORM models (`proofpath.db.models`)."""

from __future__ import annotations

import hashlib
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import create_mock_engine
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.schema import CreateIndex, CreateTable

from proofpath.config import Settings
from proofpath.db.models import (
    EMBEDDING_DIM,
    ActionDraft,
    ActionDraftStatus,
    ActionType,
    AnalysisRun,
    AnalysisRunStatus,
    ApplicationStage,
    ApplicationTracker,
    AuditActor,
    AuditLog,
    Base,
    ClaimVerdict,
    EvidenceChunk,
    EvidenceSource,
    EvidenceSourceType,
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    RoleRequirement,
    VerdictType,
)


def _compile_table_ddl(table_name: str) -> str:
    pg = create_mock_engine("postgresql+asyncpg://", lambda *_args, **_kwargs: None).dialect
    table = Base.metadata.tables[table_name]
    parts = [str(CreateTable(table).compile(dialect=pg))]
    parts.extend(str(CreateIndex(idx).compile(dialect=pg)) for idx in table.indexes)
    return "\n".join(parts)


@pytest.mark.unit
def test_evidence_source_table_schema_and_constraints() -> None:
    assert "evidence_sources" in Base.metadata.tables
    table = Base.metadata.tables["evidence_sources"]

    expected_columns = {
        "id",
        "source_type",
        "title",
        "canonical_uri",
        "content_sha256",
        "ingested_at",
        "metadata_json",
    }
    assert set(table.columns.keys()) == expected_columns

    expected_source_types = {
        "resume",
        "readme",
        "github_repo",
        "certificate",
        "experience_note",
    }
    assert {member.value for member in EvidenceSourceType} == expected_source_types

    ddl = _compile_table_ddl("evidence_sources")
    assert "id UUID DEFAULT gen_random_uuid() NOT NULL" in ddl
    assert "CONSTRAINT uq_evidence_sources_content_sha256 UNIQUE (content_sha256)" in ddl
    assert "CONSTRAINT ck_evidence_sources_content_sha256_len" in ddl
    assert "TIMESTAMP WITH TIME ZONE" in ddl
    assert "metadata_json JSONB" in ddl
    for source_type in expected_source_types:
        assert f"'{source_type}'" in ddl


@pytest.mark.unit
def test_evidence_chunk_table_schema_fk_vector_and_indexes() -> None:
    assert "evidence_chunks" in Base.metadata.tables
    table = Base.metadata.tables["evidence_chunks"]

    expected_columns = {
        "id",
        "source_id",
        "chunk_index",
        "section_label",
        "content",
        "tsv",
        "embedding",
        "token_count",
    }
    assert set(table.columns.keys()) == expected_columns

    # Vector dimension must align with default `Settings.embedding_dimensions`.
    assert Settings(_env_file=None).embedding_dimensions == EMBEDDING_DIM

    ddl = _compile_table_ddl("evidence_chunks")
    assert "REFERENCES evidence_sources (id) ON DELETE CASCADE" in ddl
    assert "UNIQUE (source_id, chunk_index)" in ddl
    assert "tsv TSVECTOR GENERATED ALWAYS AS" in ddl
    assert "STORED" in ddl
    assert f"embedding VECTOR({EMBEDDING_DIM})" in ddl
    assert "USING gin (tsv)" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl


@pytest.mark.unit
def test_evidence_source_chunks_relationship_and_instantiation() -> None:
    rel = EvidenceSource.__mapper__.relationships["chunks"]
    assert isinstance(rel, RelationshipProperty)
    assert rel.passive_deletes is True
    assert "delete-orphan" in rel.cascade

    digest = hashlib.sha256(b"sample resume content").hexdigest()
    source_id = uuid.uuid4()
    source = EvidenceSource(
        id=source_id,
        source_type=EvidenceSourceType.RESUME,
        title="Staff Engineer Resume",
        canonical_uri="file:///resumes/resume.md#experience",
        content_sha256=digest,
        metadata_json={"format": "markdown"},
    )
    chunk = EvidenceChunk(
        source=source,
        chunk_index=0,
        section_label="Experience",
        content="Built pgvector hybrid retrieval with FastAPI and LangGraph.",
        embedding=[0.01] * EMBEDDING_DIM,
        token_count=12,
    )

    assert chunk.source is source
    assert source.chunks == [chunk]
    assert len(chunk.embedding or []) == EMBEDDING_DIM


@pytest.mark.unit
def test_opportunity_and_role_requirement_schema_and_constraints() -> None:
    assert "opportunities" in Base.metadata.tables
    assert "role_requirements" in Base.metadata.tables

    opp_table = Base.metadata.tables["opportunities"]
    assert set(opp_table.columns.keys()) == {
        "id",
        "company",
        "role_title",
        "source_url",
        "raw_text",
        "sanitized_text",
        "created_at",
    }
    assert opp_table.columns["source_url"].nullable is True

    req_table = Base.metadata.tables["role_requirements"]
    assert set(req_table.columns.keys()) == {
        "id",
        "opportunity_id",
        "requirement_text",
        "entity",
        "category",
        "priority",
    }

    expected_categories = {"language", "framework", "cloud", "ai_agent", "process", "domain"}
    expected_priorities = {"must_have", "nice_to_have"}
    assert {m.value for m in RequirementCategory} == expected_categories
    assert {m.value for m in RequirementPriority} == expected_priorities

    opp_ddl = _compile_table_ddl("opportunities")
    assert "id UUID DEFAULT gen_random_uuid() NOT NULL" in opp_ddl
    assert "created_at TIMESTAMP WITH TIME ZONE" in opp_ddl

    req_ddl = _compile_table_ddl("role_requirements")
    assert "REFERENCES opportunities (id) ON DELETE CASCADE" in req_ddl
    assert "CONSTRAINT ck_role_requirements_requirement_category" in req_ddl
    assert "CONSTRAINT ck_role_requirements_requirement_priority" in req_ddl
    for category in expected_categories:
        assert f"'{category}'" in req_ddl
    for priority in expected_priorities:
        assert f"'{priority}'" in req_ddl


@pytest.mark.unit
def test_opportunity_requirements_relationship_and_instantiation() -> None:
    rel = Opportunity.__mapper__.relationships["requirements"]
    assert isinstance(rel, RelationshipProperty)
    assert rel.passive_deletes is True
    assert "delete-orphan" in rel.cascade

    opp = Opportunity(
        company="Palo Alto Networks",
        role_title="AI Agent Engineer",
        source_url="https://jobs.example.com/panw-ai-agent",
        raw_text="Build MCP + LangGraph agents with Python and pgvector.",
        sanitized_text="Build MCP + LangGraph agents with Python and pgvector.",
    )
    req = RoleRequirement(
        opportunity=opp,
        requirement_text="Production experience with LangGraph and MCP",
        category=RequirementCategory.AI_AGENT,
        priority=RequirementPriority.MUST_HAVE,
    )

    assert req.opportunity is opp
    assert opp.requirements == [req]


@pytest.mark.unit
def test_analysis_run_and_claim_verdict_schema_and_constraints() -> None:
    assert "analysis_runs" in Base.metadata.tables
    assert "claim_verdicts" in Base.metadata.tables

    run_table = Base.metadata.tables["analysis_runs"]
    assert set(run_table.columns.keys()) == {
        "id",
        "opportunity_id",
        "thread_id",
        "status",
        "total_tokens",
        "estimated_cost_usd",
        "latency_ms",
        "trace_id",
        "created_at",
    }
    assert run_table.columns["trace_id"].nullable is True

    verdict_table = Base.metadata.tables["claim_verdicts"]
    assert set(verdict_table.columns.keys()) == {
        "id",
        "analysis_run_id",
        "requirement_id",
        "verdict",
        "claim_summary",
        "cited_chunk_ids",
        "policy_downgraded",
        "gap_notes",
    }
    assert verdict_table.columns["gap_notes"].nullable is True

    expected_statuses = {"running", "awaiting_approval", "completed", "failed"}
    expected_verdicts = {"supported", "partial", "unsupported"}
    assert {m.value for m in AnalysisRunStatus} == expected_statuses
    assert {m.value for m in VerdictType} == expected_verdicts

    run_ddl = _compile_table_ddl("analysis_runs")
    assert "REFERENCES opportunities (id) ON DELETE CASCADE" in run_ddl
    assert "UNIQUE (thread_id)" in run_ddl
    assert "NUMERIC(12, 6)" in run_ddl
    assert "CONSTRAINT ck_analysis_runs_analysis_run_status" in run_ddl
    for status in expected_statuses:
        assert f"'{status}'" in run_ddl

    verdict_ddl = _compile_table_ddl("claim_verdicts")
    assert "REFERENCES analysis_runs (id) ON DELETE CASCADE" in verdict_ddl
    assert "REFERENCES role_requirements (id) ON DELETE CASCADE" in verdict_ddl
    assert "UNIQUE (analysis_run_id, requirement_id)" in verdict_ddl
    assert "UUID[]" in verdict_ddl
    assert "BOOLEAN" in verdict_ddl
    assert "ck_claim_verdicts_supported_requires_citations" in verdict_ddl
    for verdict in expected_verdicts:
        assert f"'{verdict}'" in verdict_ddl


@pytest.mark.unit
def test_analysis_run_verdicts_relationship_and_instantiation() -> None:
    opp = Opportunity(
        company="Palo Alto Networks",
        role_title="AI Agent Engineer",
        raw_text="Requires Python, LangGraph, and SAP S/4HANA.",
        sanitized_text="Requires Python, LangGraph, and SAP S/4HANA.",
    )
    req = RoleRequirement(
        opportunity=opp,
        requirement_text="SAP S/4HANA administration",
        category=RequirementCategory.DOMAIN,
        priority=RequirementPriority.MUST_HAVE,
    )
    run = AnalysisRun(
        opportunity=opp,
        thread_id="thread-001",
        status=AnalysisRunStatus.AWAITING_APPROVAL,
        total_tokens=1420,
        estimated_cost_usd=Decimal("0.002150"),
        latency_ms=1850,
        trace_id="trace-abc",
    )
    verdict = ClaimVerdict(
        analysis_run=run,
        requirement=req,
        verdict=VerdictType.UNSUPPORTED,
        claim_summary="No verified SAP S/4HANA evidence found in ledger.",
        cited_chunk_ids=[],
        policy_downgraded=True,
        gap_notes="Missing enterprise SAP S/4HANA proof.",
    )

    assert run.opportunity is opp
    assert opp.analysis_runs == [run]
    assert run.claim_verdicts == [verdict]
    assert req.claim_verdicts == [verdict]
    assert verdict.policy_downgraded is True


@pytest.mark.unit
def test_all_sub_slice_1_1_1_tables_are_registered() -> None:
    expected_tables = {
        "evidence_sources",
        "evidence_chunks",
        "opportunities",
        "role_requirements",
        "analysis_runs",
        "claim_verdicts",
        "application_tracker",
        "action_drafts",
        "audit_logs",
    }
    assert set(Base.metadata.tables.keys()) == expected_tables


@pytest.mark.unit
def test_tracker_action_draft_and_audit_log_schemas_and_constraints() -> None:
    tracker_table = Base.metadata.tables["application_tracker"]
    assert set(tracker_table.columns.keys()) == {
        "id",
        "opportunity_id",
        "stage",
        "notes",
        "updated_at",
    }
    assert tracker_table.columns["notes"].nullable is True

    draft_table = Base.metadata.tables["action_drafts"]
    assert set(draft_table.columns.keys()) == {
        "id",
        "analysis_run_id",
        "action_type",
        "payload_json",
        "status",
        "idempotency_key",
        "approved_at",
        "executed_at",
    }
    assert draft_table.columns["approved_at"].nullable is True
    assert draft_table.columns["executed_at"].nullable is True

    audit_table = Base.metadata.tables["audit_logs"]
    assert set(audit_table.columns.keys()) == {
        "id",
        "actor",
        "event_type",
        "resource_id",
        "details_json",
        "created_at",
    }
    assert audit_table.columns["resource_id"].nullable is True
    # Audit logs must not cascade-delete with domain rows.
    assert len(audit_table.foreign_keys) == 0

    expected_stages = {
        "saved",
        "preparing",
        "ready_to_apply",
        "applied",
        "interviewing",
        "archived",
    }
    expected_action_types = {"tracker_upsert", "calendar_block", "contribution_sprint"}
    expected_draft_statuses = {
        "pending_approval",
        "approved",
        "rejected",
        "executed",
        "failed",
    }
    expected_actors = {"agent", "human"}

    assert {m.value for m in ApplicationStage} == expected_stages
    assert {m.value for m in ActionType} == expected_action_types
    assert {m.value for m in ActionDraftStatus} == expected_draft_statuses
    assert {m.value for m in AuditActor} == expected_actors

    tracker_ddl = _compile_table_ddl("application_tracker")
    assert "REFERENCES opportunities (id) ON DELETE CASCADE" in tracker_ddl
    assert "UNIQUE (opportunity_id)" in tracker_ddl
    assert "updated_at TIMESTAMP WITH TIME ZONE" in tracker_ddl
    for stage in expected_stages:
        assert f"'{stage}'" in tracker_ddl

    draft_ddl = _compile_table_ddl("action_drafts")
    assert "REFERENCES analysis_runs (id) ON DELETE CASCADE" in draft_ddl
    assert "UNIQUE (idempotency_key)" in draft_ddl
    assert "payload_json JSONB" in draft_ddl
    assert "ck_action_drafts_execution_requires_approval" in draft_ddl
    assert "ck_action_drafts_executed_status_requires_timestamps" in draft_ddl
    for action_type in expected_action_types:
        assert f"'{action_type}'" in draft_ddl
    for status in expected_draft_statuses:
        assert f"'{status}'" in draft_ddl

    audit_ddl = _compile_table_ddl("audit_logs")
    assert "details_json JSONB" in audit_ddl
    assert "created_at TIMESTAMP WITH TIME ZONE" in audit_ddl
    for actor in expected_actors:
        assert f"'{actor}'" in audit_ddl


@pytest.mark.unit
def test_tracker_action_draft_and_audit_log_relationships_and_instantiation() -> None:
    opp = Opportunity(
        company="Palo Alto Networks",
        role_title="AI Agent Engineer",
        raw_text="Build safe MCP + LangGraph agents.",
        sanitized_text="Build safe MCP + LangGraph agents.",
    )
    tracker = ApplicationTracker(
        opportunity=opp,
        stage=ApplicationStage.PREPARING,
        notes="Working on MCP contribution sprint before applying.",
    )
    run = AnalysisRun(
        opportunity=opp,
        thread_id="thread-002",
        status=AnalysisRunStatus.AWAITING_APPROVAL,
    )
    draft = ActionDraft(
        analysis_run=run,
        action_type=ActionType.TRACKER_UPSERT,
        payload_json={"stage": "preparing"},
        status=ActionDraftStatus.PENDING_APPROVAL,
        idempotency_key="run-002:tracker_upsert",
    )
    audit = AuditLog(
        actor=AuditActor.AGENT,
        event_type="action_draft_created",
        resource_id=draft.id,
        details_json={"action_type": ActionType.TRACKER_UPSERT.value},
    )

    assert opp.tracker_entry is tracker
    assert tracker.opportunity is opp
    assert run.action_drafts == [draft]
    assert draft.analysis_run is run
    assert draft.approved_at is None
    assert draft.executed_at is None
    assert audit.actor is AuditActor.AGENT
