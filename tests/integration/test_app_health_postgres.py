"""Live PostgreSQL and pgvector check through the real app lifespan."""

import httpx
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import PostgresDsn

from proofpath.config import Settings
from proofpath.main import create_app

pytestmark = pytest.mark.integration


async def test_app_lifespan_and_health_with_postgres(integration_database_url: str) -> None:
    settings = Settings(_env_file=None, database_url=PostgresDsn(integration_database_url))
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.checkpointer, AsyncPostgresSaver)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "postgres": "ok", "pgvector": "ok"}
    assert not hasattr(app.state, "checkpointer")
    assert not hasattr(app.state, "db_engine")
