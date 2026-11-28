"""Transactional PyTest integration fixtures targeting Docker PostgreSQL + pgvector."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from proofpath.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI_PATH = REPO_ROOT / "alembic.ini"


def _resolve_integration_database_url() -> str:
    """Resolve the async PostgreSQL URL for integration tests."""
    override = os.getenv("TEST_DATABASE_URL")
    if override:
        return override
    return get_settings().asyncpg_database_url


def _require_live_db() -> bool:
    """Fail instead of skipping in CI or when `REQUIRE_LIVE_DB=1` is set."""
    return os.getenv("CI", "").lower() in {"1", "true", "yes"} or os.getenv(
        "REQUIRE_LIVE_DB", ""
    ).lower() in {"1", "true", "yes"}


async def _probe_and_migrate(database_url: str) -> None:
    """Verify PostgreSQL reachability and run Alembic migrations up to `head`."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        alembic_cfg = Config(str(ALEMBIC_INI_PATH))
        alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        async with engine.begin() as conn:

            def _run_upgrade(sync_conn: object) -> None:
                alembic_cfg.attributes["connection"] = sync_conn
                command.upgrade(alembic_cfg, "head")

            await conn.run_sync(_run_upgrade)
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def integration_database_url() -> str:
    """Ensure the Docker PostgreSQL + pgvector service is reachable and migrated."""
    database_url = _resolve_integration_database_url()
    try:
        asyncio.run(_probe_and_migrate(database_url))
    except Exception as exc:
        if _require_live_db():
            raise
        pytest.skip(
            f"Live PostgreSQL service unavailable at {database_url!r} ({exc}). "
            "Start local containers with `make up` or set TEST_DATABASE_URL."
        )
    return database_url


@pytest.fixture
async def db_engine(integration_database_url: str) -> AsyncIterator[AsyncEngine]:
    """Yield a `NullPool` async engine scoped to the test's event loop."""
    engine = create_async_engine(integration_database_url, poolclass=NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Yield a transactional `AsyncSession` that always rolls back after each test.

    Uses `join_transaction_mode='create_savepoint'` so nested savepoints and
    integrity-error tests stay isolated without dirtying PostgreSQL state.
    """
    async with db_engine.connect() as connection:
        outer_tx = await connection.begin()
        session = AsyncSession(
            bind=connection,
            join_transaction_mode="create_savepoint",
            autoflush=False,
            expire_on_commit=False,
        )
        try:
            yield session
        finally:
            await session.close()
            await outer_tx.rollback()


def _resolve_integration_redis_url() -> str:
    """Resolve the Redis URL for integration tests."""
    override = os.getenv("TEST_REDIS_URL")
    if override:
        return override
    return str(get_settings().redis_url)


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    """Yield a live Redis client, skipping when the service is unavailable."""
    url = _resolve_integration_redis_url()
    client = Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        if _require_live_db():
            raise
        pytest.skip(
            f"Live Redis service unavailable at {url!r} ({exc}). "
            "Start local containers with `make up` or set TEST_REDIS_URL."
        )
    try:
        yield client
    finally:
        await client.aclose()
