"""Preserve the entity extracted for deterministic evidence checks.

Revision ID: 0003_requirement_entity
Revises: 0002_add_vector_and_fts_indexes
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_requirement_entity"
down_revision: str | Sequence[str] | None = "0002_add_vector_and_fts_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("role_requirements", sa.Column("entity", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("role_requirements", "entity")
