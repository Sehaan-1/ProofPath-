"""Add HNSW cosine vector index and GIN full-text index on evidence_chunks.

Revision ID: 0002_add_vector_and_fts_indexes
Revises: 0001_initial_schema
Create Date: 2026-11-27 00:05:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_add_vector_and_fts_indexes"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create HNSW cosine index on embeddings and GIN index on tsvector."""
    op.create_index(
        "ix_evidence_chunks_embedding_hnsw",
        "evidence_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_evidence_chunks_tsv_gin",
        "evidence_chunks",
        ["tsv"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Drop GIN full-text index and HNSW cosine vector index."""
    op.drop_index("ix_evidence_chunks_tsv_gin", table_name="evidence_chunks")
    op.drop_index("ix_evidence_chunks_embedding_hnsw", table_name="evidence_chunks")
