"""Initial ProofPath relational and vector schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-27 00:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create pgvector extension and all initial ProofPath tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. evidence_sources
    op.create_table(
        "evidence_sources",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("canonical_uri", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_sources")),
        sa.UniqueConstraint("content_sha256", name=op.f("uq_evidence_sources_content_sha256")),
        sa.CheckConstraint(
            "char_length(content_sha256) = 64",
            name=op.f("ck_evidence_sources_content_sha256_len"),
        ),
        sa.CheckConstraint(
            "source_type IN ('resume', 'readme', 'github_repo', 'certificate', 'experience_note')",
            name=op.f("ck_evidence_sources_evidence_source_type"),
        ),
    )
    op.create_index("ix_evidence_sources_source_type", "evidence_sources", ["source_type"])

    # 2. evidence_chunks (vector & full-text indexes added in Sub-part 1.1.2b)
    op.create_table(
        "evidence_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section_label", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', coalesce(section_label, '') || ' ' || content)",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1536), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_chunks")),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["evidence_sources.id"],
            name=op.f("fk_evidence_chunks_source_id_evidence_sources"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "source_id",
            "chunk_index",
            name=op.f("uq_evidence_chunks_source_id_chunk_index"),
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name=op.f("ck_evidence_chunks_chunk_index_non_negative"),
        ),
        sa.CheckConstraint(
            "token_count >= 0",
            name=op.f("ck_evidence_chunks_token_count_non_negative"),
        ),
    )
    op.create_index("ix_evidence_chunks_source_id", "evidence_chunks", ["source_id"])

    # 3. opportunities
    op.create_table(
        "opportunities",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("company", sa.String(length=255), nullable=False),
        sa.Column("role_title", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("sanitized_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_opportunities")),
    )
    op.create_index("ix_opportunities_company", "opportunities", ["company"])

    # 4. role_requirements
    op.create_table(
        "role_requirements",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("opportunity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requirement_text", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_role_requirements")),
        sa.ForeignKeyConstraint(
            ["opportunity_id"],
            ["opportunities.id"],
            name=op.f("fk_role_requirements_opportunity_id_opportunities"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "category IN ('language', 'framework', 'cloud', 'ai_agent', 'process', 'domain')",
            name=op.f("ck_role_requirements_requirement_category"),
        ),
        sa.CheckConstraint(
            "priority IN ('must_have', 'nice_to_have')",
            name=op.f("ck_role_requirements_requirement_priority"),
        ),
    )
    op.create_index(
        "ix_role_requirements_opportunity_id",
        "role_requirements",
        ["opportunity_id"],
    )
    op.create_index(
        "ix_role_requirements_opportunity_priority",
        "role_requirements",
        ["opportunity_id", "priority"],
    )

    # 5. analysis_runs
    op.create_table(
        "analysis_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("opportunity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'running'"),
            nullable=False,
        ),
        sa.Column(
            "total_tokens",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "estimated_cost_usd",
            sa.Numeric(precision=12, scale=6),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "latency_ms",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analysis_runs")),
        sa.ForeignKeyConstraint(
            ["opportunity_id"],
            ["opportunities.id"],
            name=op.f("fk_analysis_runs_opportunity_id_opportunities"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("thread_id", name=op.f("uq_analysis_runs_thread_id")),
        sa.CheckConstraint(
            "status IN ('running', 'awaiting_approval', 'completed', 'failed')",
            name=op.f("ck_analysis_runs_analysis_run_status"),
        ),
        sa.CheckConstraint(
            "total_tokens >= 0",
            name=op.f("ck_analysis_runs_total_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "estimated_cost_usd >= 0",
            name=op.f("ck_analysis_runs_estimated_cost_usd_non_negative"),
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name=op.f("ck_analysis_runs_latency_ms_non_negative"),
        ),
    )
    op.create_index("ix_analysis_runs_opportunity_id", "analysis_runs", ["opportunity_id"])
    op.create_index("ix_analysis_runs_status", "analysis_runs", ["status"])
    op.create_index("ix_analysis_runs_trace_id", "analysis_runs", ["trace_id"])

    # 6. claim_verdicts
    op.create_table(
        "claim_verdicts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("analysis_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requirement_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("claim_summary", sa.Text(), nullable=False),
        sa.Column(
            "cited_chunk_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            server_default=sa.text("'{}'::uuid[]"),
            nullable=False,
        ),
        sa.Column(
            "policy_downgraded",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("gap_notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_claim_verdicts")),
        sa.ForeignKeyConstraint(
            ["analysis_run_id"],
            ["analysis_runs.id"],
            name=op.f("fk_claim_verdicts_analysis_run_id_analysis_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["requirement_id"],
            ["role_requirements.id"],
            name=op.f("fk_claim_verdicts_requirement_id_role_requirements"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "analysis_run_id",
            "requirement_id",
            name=op.f("uq_claim_verdicts_analysis_run_id_requirement_id"),
        ),
        sa.CheckConstraint(
            "verdict IN ('supported', 'partial', 'unsupported')",
            name=op.f("ck_claim_verdicts_claim_verdict_type"),
        ),
        sa.CheckConstraint(
            "verdict = 'unsupported' OR cardinality(cited_chunk_ids) > 0",
            name=op.f("ck_claim_verdicts_supported_requires_citations"),
        ),
    )
    op.create_index("ix_claim_verdicts_analysis_run_id", "claim_verdicts", ["analysis_run_id"])
    op.create_index("ix_claim_verdicts_requirement_id", "claim_verdicts", ["requirement_id"])

    # 7. application_tracker
    op.create_table(
        "application_tracker",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("opportunity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "stage",
            sa.String(length=32),
            server_default=sa.text("'saved'"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_application_tracker")),
        sa.ForeignKeyConstraint(
            ["opportunity_id"],
            ["opportunities.id"],
            name=op.f("fk_application_tracker_opportunity_id_opportunities"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("opportunity_id", name=op.f("uq_application_tracker_opportunity_id")),
        sa.CheckConstraint(
            "stage IN ('saved', 'preparing', 'ready_to_apply', 'applied', 'interviewing', 'archived')",
            name=op.f("ck_application_tracker_application_stage"),
        ),
    )
    op.create_index("ix_application_tracker_stage", "application_tracker", ["stage"])

    # 8. action_drafts
    op.create_table(
        "action_drafts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("analysis_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column(
            "payload_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'pending_approval'"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_action_drafts")),
        sa.ForeignKeyConstraint(
            ["analysis_run_id"],
            ["analysis_runs.id"],
            name=op.f("fk_action_drafts_analysis_run_id_analysis_runs"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_action_drafts_idempotency_key")),
        sa.CheckConstraint(
            "action_type IN ('tracker_upsert', 'calendar_block', 'contribution_sprint')",
            name=op.f("ck_action_drafts_action_type"),
        ),
        sa.CheckConstraint(
            "status IN ('pending_approval', 'approved', 'rejected', 'executed', 'failed')",
            name=op.f("ck_action_drafts_action_draft_status"),
        ),
        sa.CheckConstraint(
            "executed_at IS NULL OR approved_at IS NOT NULL",
            name=op.f("ck_action_drafts_execution_requires_approval"),
        ),
        sa.CheckConstraint(
            "status <> 'executed' OR (approved_at IS NOT NULL AND executed_at IS NOT NULL)",
            name=op.f("ck_action_drafts_executed_status_requires_timestamps"),
        ),
    )
    op.create_index("ix_action_drafts_analysis_run_id", "action_drafts", ["analysis_run_id"])
    op.create_index("ix_action_drafts_status", "action_drafts", ["status"])

    # 9. audit_logs
    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("actor", sa.String(length=16), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "details_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
        sa.CheckConstraint(
            "actor IN ('agent', 'human')",
            name=op.f("ck_audit_logs_audit_actor"),
        ),
    )
    op.create_index("ix_audit_logs_event_type", "audit_logs", ["event_type"])
    op.create_index("ix_audit_logs_resource_id", "audit_logs", ["resource_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index("ix_audit_logs_actor_created_at", "audit_logs", ["actor", "created_at"])


def downgrade() -> None:
    """Drop all ProofPath tables in reverse dependency order."""
    op.drop_index("ix_audit_logs_actor_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_resource_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_event_type", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_action_drafts_status", table_name="action_drafts")
    op.drop_index("ix_action_drafts_analysis_run_id", table_name="action_drafts")
    op.drop_table("action_drafts")

    op.drop_index("ix_application_tracker_stage", table_name="application_tracker")
    op.drop_table("application_tracker")

    op.drop_index("ix_claim_verdicts_requirement_id", table_name="claim_verdicts")
    op.drop_index("ix_claim_verdicts_analysis_run_id", table_name="claim_verdicts")
    op.drop_table("claim_verdicts")

    op.drop_index("ix_analysis_runs_trace_id", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_status", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_opportunity_id", table_name="analysis_runs")
    op.drop_table("analysis_runs")

    op.drop_index("ix_role_requirements_opportunity_priority", table_name="role_requirements")
    op.drop_index("ix_role_requirements_opportunity_id", table_name="role_requirements")
    op.drop_table("role_requirements")

    op.drop_index("ix_opportunities_company", table_name="opportunities")
    op.drop_table("opportunities")

    op.drop_index("ix_evidence_chunks_source_id", table_name="evidence_chunks")
    op.drop_table("evidence_chunks")

    op.drop_index("ix_evidence_sources_source_type", table_name="evidence_sources")
    op.drop_table("evidence_sources")
