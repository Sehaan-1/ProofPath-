"""Live PostgreSQL tests for hybrid pgvector + ts_rank RRF search."""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import EMBEDDING_DIM, EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceChunkInput, EvidenceRepo
from proofpath.retrieval import EmbeddingClient, HashEmbeddingBackend, HybridSearcher

pytestmark = pytest.mark.integration


def _digest(label: str) -> str:
    return hashlib.sha256(f"hybrid-{label}-{uuid.uuid4()}".encode()).hexdigest()


async def test_hybrid_search_surfaces_acronym_fts_hits(
    db_session: AsyncSession,
) -> None:
    repo = EvidenceRepo(db_session)
    source = await repo.create_source(
        source_type=EvidenceSourceType.RESUME,
        title="Hybrid retrieval resume",
        canonical_uri="file:///local/hybrid.md",
        content_sha256=_digest("resume"),
        metadata_json={
            "chunk_anchors": [
                {
                    "chunk_index": 0,
                    "canonical_uri": "file:///local/hybrid.md#Experience",
                }
            ]
        },
    )
    embedder = EmbeddingClient(
        HashEmbeddingBackend(dimensions=EMBEDDING_DIM),
        model="hash",
        dimensions=EMBEDDING_DIM,
    )
    sap_text = "Configured SAP S/4HANA finance modules in a prior role."
    fastapi_text = "Built FastAPI services with pgvector, MCP tools, and Helm charts."
    other_text = "Wrote documentation for internal onboarding workshops."
    sap_vec, fastapi_vec, other_vec, _query_vec = await embedder.embed(
        [sap_text, fastapi_text, other_text, "SAP S/4HANA"]
    )
    await repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Experience",
                content=sap_text,
                token_count=12,
                embedding=sap_vec,
            ),
            EvidenceChunkInput(
                chunk_index=1,
                section_label="Projects",
                content=fastapi_text,
                token_count=14,
                embedding=fastapi_vec,
            ),
            EvidenceChunkInput(
                chunk_index=2,
                section_label="Other",
                content=other_text,
                token_count=8,
                embedding=other_vec,
            ),
        ],
    )
    await db_session.flush()

    searcher = HybridSearcher(db_session, embedder)
    sap_hits = await searcher.search("SAP S/4HANA", top_k=3)
    assert sap_hits
    assert any("SAP" in hit.content for hit in sap_hits)
    assert sap_hits[0].canonical_uri.startswith("file:///local/hybrid.md")
    assert sap_hits[0].source_title == "Hybrid retrieval resume"
    assert sap_hits[0].score is not None
    assert sap_hits[0].score > 0

    stack_hits = await searcher.search("MCP FastAPI Helm pgvector", top_k=3)
    assert stack_hits
    assert any("FastAPI" in hit.content and "MCP" in hit.content for hit in stack_hits)
