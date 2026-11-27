"""Async Alembic migration environment for ProofPath."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from proofpath.config import get_settings
from proofpath.db.models import Base

# Alembic Config object providing access to values within `alembic.ini`.
config = context.config

# Configure Python logging when running via CLI with an ini file.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# LangGraph's `AsyncPostgresSaver` manages its own checkpoint tables inside the
# same database; exclude them from Alembic autogenerate diffs.
LANGGRAPH_CHECKPOINT_TABLES: frozenset[str] = frozenset(
    {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_migrations",
        "checkpoint_writes",
    }
)


def _get_database_url() -> str:
    """Resolve the async PostgreSQL URL from config overrides or `Settings`."""
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    return get_settings().asyncpg_database_url


def include_object(
    obj: Any,
    name: str | None,
    type_: str,
    reflected: bool,  # noqa: ARG001
    compare_to: Any,  # noqa: ARG001
) -> bool:
    """Exclude externally managed LangGraph checkpointer tables from autogenerate."""
    if type_ == "table" and name in LANGGRAPH_CHECKPOINT_TABLES:
        return False
    table = getattr(obj, "table", None)
    return not (table is not None and getattr(table, "name", None) in LANGGRAPH_CHECKPOINT_TABLES)


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render pgvector `Vector(dim)` columns cleanly during autogenerate."""
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        if obj.dim is not None:
            return f"pgvector.sqlalchemy.Vector({obj.dim})"
        return "pgvector.sqlalchemy.Vector()"
    return False


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits PostgreSQL DDL without a live DB)."""
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        output_buffer=config.output_buffer,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations synchronously within an async connection's `run_sync`."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async SQLAlchemy engine and run migrations inside a transaction."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _get_database_url()

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode, supporting injected test connections."""
    connectable = config.attributes.get("connection")
    if isinstance(connectable, Connection):
        do_run_migrations(connectable)
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
