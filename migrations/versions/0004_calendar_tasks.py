"""Add the local calendar task table used by the approved-only executor.

Revision ID: 0004_calendar_tasks
Revises: 0003_requirement_entity
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_calendar_tasks"
down_revision: str | Sequence[str] | None = "0003_requirement_entity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "calendar_tasks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("action_draft_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("opportunity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "timezone",
            sa.String(length=64),
            server_default=sa.text("'UTC'"),
            nullable=False,
        ),
        sa.Column("ics_uid", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_calendar_tasks")),
        sa.ForeignKeyConstraint(
            ["action_draft_id"],
            ["action_drafts.id"],
            name=op.f("fk_calendar_tasks_action_draft_id_action_drafts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["opportunity_id"],
            ["opportunities.id"],
            name=op.f("fk_calendar_tasks_opportunity_id_opportunities"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("action_draft_id", name=op.f("uq_calendar_tasks_action_draft_id")),
        sa.UniqueConstraint("ics_uid", name=op.f("uq_calendar_tasks_ics_uid")),
        sa.CheckConstraint("end_at > start_at", name=op.f("ck_calendar_tasks_end_after_start")),
    )
    op.create_index("ix_calendar_tasks_opportunity_id", "calendar_tasks", ["opportunity_id"])
    op.create_index("ix_calendar_tasks_start_at", "calendar_tasks", ["start_at"])


def downgrade() -> None:
    op.drop_index("ix_calendar_tasks_start_at", table_name="calendar_tasks")
    op.drop_index("ix_calendar_tasks_opportunity_id", table_name="calendar_tasks")
    op.drop_table("calendar_tasks")
