"""Unit tests for async Alembic configuration and schema migrations."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from proofpath.db.models import EMBEDDING_DIM, Base

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI_PATH = REPO_ROOT / "alembic.ini"
MIGRATIONS_ENV_PATH = REPO_ROOT / "migrations" / "env.py"


def _make_alembic_config(output_buffer: io.StringIO | None = None) -> Config:
    cfg = Config(str(ALEMBIC_INI_PATH), output_buffer=output_buffer)
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return cfg


@pytest.mark.unit
def test_alembic_config_is_present_without_hardcoded_credentials() -> None:
    assert ALEMBIC_INI_PATH.exists()
    assert MIGRATIONS_ENV_PATH.exists()

    cfg = _make_alembic_config()
    assert cfg.get_main_option("sqlalchemy.url") in (None, "")

    env_source = MIGRATIONS_ENV_PATH.read_text(encoding="utf-8")
    assert "async_engine_from_config" in env_source
    assert "get_settings().asyncpg_database_url" in env_source
    assert "checkpoints" in env_source


@pytest.mark.unit
def test_migration_history_has_single_linear_head() -> None:
    cfg = _make_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert heads == ["0003_requirement_entity"]
    revisions = {rev.revision for rev in script.walk_revisions()}
    assert {"0001_initial_schema", "0002_add_vector_and_fts_indexes"}.issubset(revisions)


@pytest.mark.unit
def test_offline_upgrade_emits_all_tables_and_constraints() -> None:
    buffer = io.StringIO()
    cfg = _make_alembic_config(buffer)

    command.upgrade(cfg, "head", sql=True)
    sql = buffer.getvalue()

    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    for table_name in Base.metadata.tables:
        assert f"CREATE TABLE {table_name} (" in sql

    assert f"VECTOR({EMBEDDING_DIM})" in sql
    assert "TSVECTOR GENERATED ALWAYS AS" in sql
    assert (
        "CREATE INDEX ix_evidence_chunks_embedding_hnsw ON evidence_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    ) in sql
    assert "CREATE INDEX ix_evidence_chunks_tsv_gin ON evidence_chunks USING gin (tsv)" in sql
    assert "ck_claim_verdicts_supported_requires_citations" in sql
    assert "ck_action_drafts_execution_requires_approval" in sql
    assert "ck_action_drafts_executed_status_requires_timestamps" in sql
    assert "_ck_" not in sql


@pytest.mark.unit
def test_offline_downgrade_drops_all_tables_cleanly() -> None:
    buffer = io.StringIO()
    cfg = _make_alembic_config(buffer)
    head_rev = ScriptDirectory.from_config(cfg).get_current_head()
    assert head_rev is not None

    command.downgrade(cfg, f"{head_rev}:base", sql=True)
    sql = buffer.getvalue()

    assert "DROP INDEX ix_evidence_chunks_tsv_gin" in sql
    assert "DROP INDEX ix_evidence_chunks_embedding_hnsw" in sql
    for table_name in Base.metadata.tables:
        assert f"DROP TABLE {table_name}" in sql
