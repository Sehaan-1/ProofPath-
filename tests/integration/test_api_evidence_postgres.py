"""Real ledger ingestion and hybrid search through the API."""

import uuid

import httpx
import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete

from proofpath.config import Settings
from proofpath.db.models import EvidenceSource
from proofpath.main import create_app

pytestmark = pytest.mark.integration


async def test_text_ingestion_dedup_search_and_conflict(integration_database_url: str) -> None:
    settings = Settings(
        _env_file=None, database_url=PostgresDsn(integration_database_url), embedding_model="hash"
    )
    app = create_app(settings=settings)
    uri = f"file:///proofpath-test/{uuid.uuid4()}"
    token = f"skill{uuid.uuid4().hex}"
    payload = {
        "title": "Test note",
        "content": f"Experience using {token} in projects.",
        "canonical_uri": uri,
    }
    async with app.router.lifespan_context(app):
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = await client.post("/api/v1/evidence/ingest/text", json=payload)
                assert first.status_code == 201
                source_id = first.json()["source_id"]
                second = await client.post("/api/v1/evidence/ingest/text", json=payload)
                assert second.status_code == 201
                assert second.json()["created"] is False
                assert second.json()["source_id"] == source_id
                search = await client.get("/api/v1/evidence/search", params={"query": token})
                assert search.status_code == 200
                assert any(item["source_id"] == source_id for item in search.json()["results"])
                conflict = await client.post(
                    "/api/v1/evidence/ingest/text", json={**payload, "content": "Different content"}
                )
                assert conflict.status_code == 409
                assert conflict.headers["content-type"] == "application/problem+json"
        finally:
            async with app.state.session_factory.begin() as session:
                await session.execute(
                    delete(EvidenceSource).where(EvidenceSource.canonical_uri == uri)
                )
