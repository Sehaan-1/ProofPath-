"""Add the background job status ledger table for Redis-queued workers.

Revision ID: 0005_background_jobs
Revises: 0004_calendar_tasks

Sub-part 2.2.1b: a Redis Stream is the job *transport*; this table is the
authoritative, queryable status ledger (`queued | running | succeeded |
failed | dead_letter`). The worker correlates stream messages to rows here by
`id` and drives the state machine, while acking/re-enqueueing purely against
Redis.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_background_jobs"
down_revision: str | Sequence[str] | None = "0004_calendar_tasks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "background_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("job_type", sa.String(length=128), nullable=False),
        sa.Column(
            "payload_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'queued'"),
            nullable=False,
        ),
        sa.Column(
            "attempt",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            server_default=sa.text("3"),
            nullable=False,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("error_traceback", sa.Text(), nullable=True),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_background_jobs")),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'dead_letter')",
            name=op.f("ck_background_jobs_job_status"),
        ),
        sa.CheckConstraint(
            "attempt >= 0",
            name=op.f("ck_background_jobs_attempt_non_negative"),
        ),
        sa.CheckConstraint(
            "max_attempts >= 1",
            name=op.f("ck_background_jobs_max_attempts_at_least_one"),
        ),
    )
    op.create_index("ix_background_jobs_status", "background_jobs", ["status"])
    op.create_index("ix_background_jobs_job_type", "background_jobs", ["job_type"])
    op.create_index(
        "ix_background_jobs_status_created_at",
        "background_jobs",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_background_jobs_status_created_at", table_name="background_jobs")
    op.drop_index("ix_background_jobs_job_type", table_name="background_jobs")
    op.drop_index("ix_background_jobs_status", table_name="background_jobs")
    op.drop_table("background_jobs")
