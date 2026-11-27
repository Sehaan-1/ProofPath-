"""Async SQLAlchemy engine, session factory, and health check helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from proofpath.config import Settings, get_settings


def create_db_engine(
    database_url: str | None = None,
    *,
    settings: Settings | None = None,
    echo: bool = False,
    pool_pre_ping: bool = True,
) -> AsyncEngine:
    """Build an async SQLAlchemy engine targeting PostgreSQL via asyncpg."""
    resolved_url = database_url
    if resolved_url is None:
        active_settings = settings if settings is not None else get_settings()
        resolved_url = active_settings.asyncpg_database_url
    return create_async_engine(
        resolved_url,
        echo=echo,
        pool_pre_ping=pool_pre_ping,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory with `expire_on_commit=False` for async safety."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )


class DatabaseSessionManager:
    """Lifecycle-aware container for an async SQLAlchemy engine and session factory."""

    def __init__(self, database_url: str | None = None, *, echo: bool = False) -> None:
        self._database_url = database_url
        self._echo = echo
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_db_engine(self._database_url, echo=self._echo)
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = create_session_factory(self.engine)
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional `AsyncSession`, rolling back on unhandled errors."""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Dispose of pooled database connections cleanly on shutdown."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None


async def verify_pgvector_extension(session: AsyncSession) -> bool:
    """Confirm PostgreSQL is reachable and the `vector` extension is installed."""
    result = await session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
    )
    return result.scalar_one_or_none() == 1
