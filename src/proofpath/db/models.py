"""SQLAlchemy 2.0 async ORM models for ProofPath."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any, Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY as PGARRAY
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint naming convention required for reliable Alembic migrations.
NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Default vector dimensionality matching `Settings.embedding_dimensions` (`text-embedding-3-small`).
EMBEDDING_DIM: Final[int] = 1536


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    """Persist StrEnum `.value` strings instead of member names."""
    return [str(member.value) for member in enum_cls]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Base(AsyncAttrs, DeclarativeBase):
    """Shared SQLAlchemy 2.0 async declarative base for all ProofPath ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ---------------------------------------------------------------------------
# Sub-part 1.1.1a — Evidence Ledger tables
# ---------------------------------------------------------------------------


class EvidenceSourceType(StrEnum):
    """Permitted provenance categories for user-owned evidence sources."""

    RESUME = "resume"
    README = "readme"
    GITHUB_REPO = "github_repo"
    CERTIFICATE = "certificate"
    EXPERIENCE_NOTE = "experience_note"


class EvidenceSource(Base):
    """Canonical record of an ingested user-owned evidence artifact."""

    __tablename__ = "evidence_sources"
    __table_args__ = (
        CheckConstraint("char_length(content_sha256) = 64", name="content_sha256_len"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    source_type: Mapped[EvidenceSourceType] = mapped_column(
        SAEnum(
            EvidenceSourceType,
            name="evidence_source_type",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    canonical_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    chunks: Mapped[list[EvidenceChunk]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="EvidenceChunk.chunk_index",
    )


class EvidenceChunk(Base):
    """Chunked, full-text + vector indexed slice of an `EvidenceSource`."""

    __tablename__ = "evidence_chunks"
    __table_args__ = (
        UniqueConstraint("source_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("token_count >= 0", name="token_count_non_negative"),
        Index("ix_evidence_chunks_source_id", "source_id"),
        Index("ix_evidence_chunks_tsv_gin", "tsv", postgresql_using="gin"),
        Index(
            "ix_evidence_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(section_label, '') || ' ' || content)",
            persisted=True,
        ),
        nullable=True,
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=True,
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    source: Mapped[EvidenceSource] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# Sub-part 1.1.1b — Opportunity tables
# ---------------------------------------------------------------------------


class RequirementCategory(StrEnum):
    """Normalized skill/requirement categories extracted from a job description."""

    LANGUAGE = "language"
    FRAMEWORK = "framework"
    CLOUD = "cloud"
    AI_AGENT = "ai_agent"
    PROCESS = "process"
    DOMAIN = "domain"


class RequirementPriority(StrEnum):
    """Importance tier for a parsed role requirement."""

    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"


class Opportunity(Base):
    """Target job opportunity ingested from a public posting URL or pasted JD text."""

    __tablename__ = "opportunities"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role_title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    sanitized_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    requirements: Mapped[list[RoleRequirement]] = relationship(
        back_populates="opportunity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(
        back_populates="opportunity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tracker_entry: Mapped[ApplicationTracker | None] = relationship(
        back_populates="opportunity",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RoleRequirement(Base):
    """Structured requirement extracted from an `Opportunity`."""

    __tablename__ = "role_requirements"
    __table_args__ = (
        Index("ix_role_requirements_opportunity_id", "opportunity_id"),
        Index("ix_role_requirements_opportunity_priority", "opportunity_id", "priority"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        nullable=False,
    )
    requirement_text: Mapped[str] = mapped_column(Text, nullable=False)
    entity: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category: Mapped[RequirementCategory] = mapped_column(
        SAEnum(
            RequirementCategory,
            name="requirement_category",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
    )
    priority: Mapped[RequirementPriority] = mapped_column(
        SAEnum(
            RequirementPriority,
            name="requirement_priority",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
    )

    opportunity: Mapped[Opportunity] = relationship(back_populates="requirements")
    claim_verdicts: Mapped[list[ClaimVerdict]] = relationship(
        back_populates="requirement",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ---------------------------------------------------------------------------
# Sub-part 1.1.1c — Analysis & Claim Verdict tables
# ---------------------------------------------------------------------------


class AnalysisRunStatus(StrEnum):
    """Lifecycle state of a LangGraph opportunity analysis run."""

    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class VerdictType(StrEnum):
    """Evidence-backed claim verdict for a single role requirement."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class AnalysisRun(Base):
    """Persisted execution record, checkpoint thread, and cost metrics for a graph run."""

    __tablename__ = "analysis_runs"
    __table_args__ = (
        CheckConstraint("total_tokens >= 0", name="total_tokens_non_negative"),
        CheckConstraint("estimated_cost_usd >= 0", name="estimated_cost_usd_non_negative"),
        CheckConstraint("latency_ms >= 0", name="latency_ms_non_negative"),
        Index("ix_analysis_runs_opportunity_id", "opportunity_id"),
        Index("ix_analysis_runs_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        nullable=False,
    )
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    status: Mapped[AnalysisRunStatus] = mapped_column(
        SAEnum(
            AnalysisRunStatus,
            name="analysis_run_status",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
        default=AnalysisRunStatus.RUNNING,
        server_default=text("'running'"),
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        default=Decimal("0"),
        server_default=text("0"),
    )
    latency_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    opportunity: Mapped[Opportunity] = relationship(back_populates="analysis_runs")
    claim_verdicts: Mapped[list[ClaimVerdict]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    action_drafts: Mapped[list[ActionDraft]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ClaimVerdict(Base):
    """Per-requirement evidence verdict enforcing cited provenance for supported claims."""

    __tablename__ = "claim_verdicts"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", "requirement_id"),
        CheckConstraint(
            "verdict = 'unsupported' OR cardinality(cited_chunk_ids) > 0",
            name="supported_requires_citations",
        ),
        Index("ix_claim_verdicts_analysis_run_id", "analysis_run_id"),
        Index("ix_claim_verdicts_requirement_id", "requirement_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("role_requirements.id", ondelete="CASCADE"),
        nullable=False,
    )
    verdict: Mapped[VerdictType] = mapped_column(
        SAEnum(
            VerdictType,
            name="claim_verdict_type",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
    )
    claim_summary: Mapped[str] = mapped_column(Text, nullable=False)
    cited_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        PGARRAY(PGUUID(as_uuid=True)),
        nullable=False,
        default=list,
        server_default=text("'{}'::uuid[]"),
    )
    policy_downgraded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    gap_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="claim_verdicts")
    requirement: Mapped[RoleRequirement] = relationship(back_populates="claim_verdicts")


# ---------------------------------------------------------------------------
# Sub-part 1.1.1d — Personal Tracker, Action Draft & Audit Log tables
# ---------------------------------------------------------------------------


class ApplicationStage(StrEnum):
    """Pipeline stage in the user's personal opportunity tracker."""

    SAVED = "saved"
    PREPARING = "preparing"
    READY_TO_APPLY = "ready_to_apply"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    ARCHIVED = "archived"


class ActionType(StrEnum):
    """Allowlisted approval-gated action draft categories."""

    TRACKER_UPSERT = "tracker_upsert"
    CALENDAR_BLOCK = "calendar_block"
    CONTRIBUTION_SPRINT = "contribution_sprint"


class ActionDraftStatus(StrEnum):
    """Approval and execution lifecycle state of an action draft."""

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


class AuditActor(StrEnum):
    """Actor category recorded on immutable audit log entries."""

    AGENT = "agent"
    HUMAN = "human"


class ApplicationTracker(Base):
    """User's personal application stage and notes for a tracked `Opportunity`."""

    __tablename__ = "application_tracker"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    stage: Mapped[ApplicationStage] = mapped_column(
        SAEnum(
            ApplicationStage,
            name="application_stage",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
        default=ApplicationStage.SAVED,
        server_default=text("'saved'"),
        index=True,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=func.now(),
    )

    opportunity: Mapped[Opportunity] = relationship(back_populates="tracker_entry")


class ActionDraft(Base):
    """Pending or approved side-effect draft gated by explicit human approval."""

    __tablename__ = "action_drafts"
    __table_args__ = (
        CheckConstraint(
            "executed_at IS NULL OR approved_at IS NOT NULL",
            name="execution_requires_approval",
        ),
        CheckConstraint(
            "status <> 'executed' OR (approved_at IS NOT NULL AND executed_at IS NOT NULL)",
            name="executed_status_requires_timestamps",
        ),
        Index("ix_action_drafts_analysis_run_id", "analysis_run_id"),
        Index("ix_action_drafts_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_type: Mapped[ActionType] = mapped_column(
        SAEnum(
            ActionType,
            name="action_type",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
    )
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    status: Mapped[ActionDraftStatus] = mapped_column(
        SAEnum(
            ActionDraftStatus,
            name="action_draft_status",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
        default=ActionDraftStatus.PENDING_APPROVAL,
        server_default=text("'pending_approval'"),
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="action_drafts")


class CalendarTask(Base):
    """Local calendar/task entry materialized from one approved `calendar_block` draft.

    Sub-part 2.1.3b: written only by the approved-only executor, never by the
    analysis graph. `action_draft_id` is unique, so executing the same draft twice
    cannot create a second task; `ics_uid` is deterministic per draft, which keeps
    re-generated `.ics` invites identical.
    """

    __tablename__ = "calendar_tasks"
    __table_args__ = (
        CheckConstraint("end_at > start_at", name="end_after_start"),
        UniqueConstraint("action_draft_id"),
        UniqueConstraint("ics_uid"),
        Index("ix_calendar_tasks_opportunity_id", "opportunity_id"),
        Index("ix_calendar_tasks_start_at", "start_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    action_draft_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("action_drafts.id", ondelete="CASCADE"),
        nullable=False,
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="UTC",
        server_default=text("'UTC'"),
    )
    ics_uid: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )


class AuditLog(Base):
    """Immutable append-only audit log for agent actions, policy events, and approvals."""

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_actor_created_at", "actor", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    actor: Mapped[AuditActor] = mapped_column(
        SAEnum(
            AuditActor,
            name="audit_actor",
            native_enum=False,
            create_constraint=True,
            length=16,
            values_callable=_enum_values,
            validate_strings=True,
        ),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    details_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )
