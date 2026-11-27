"""Hybrid pgvector + full-text retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Final

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import EMBEDDING_DIM, EvidenceChunk, EvidenceSource
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.retrieval.embeddings import EmbeddingClient, EmbeddingDimensionError
from proofpath.schemas.evidence import EvidenceSnippet

DEFAULT_TOP_K: Final[int] = 8
DEFAULT_VECTOR_K: Final[int] = 20
DEFAULT_FTS_K: Final[int] = 20
DEFAULT_RRF_K: Final[int] = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[uuid.UUID]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_TOP_K,
) -> list[tuple[uuid.UUID, float]]:
    """Fuse rank lists with RRF: ``score(d) = Σ 1 / (k + rank_r(d))``.

    Exact lexical hits (``SAP``, ``MCP``, ``pgvector``, ``FastAPI``, ``Helm``)
    survive even when they rank poorly in pure cosine search.
    """
    if k < 1:
        raise ValueError("RRF k must be at least 1.")
    if limit < 1:
        return []
    scores: dict[uuid.UUID, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    return ordered[:limit]


class HybridSearcher:
    """Embed a query and search the Evidence Ledger with hybrid RRF (1.3.3b)."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbeddingClient,
        *,
        vector_k: int = DEFAULT_VECTOR_K,
        fts_k: int = DEFAULT_FTS_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._repo = EvidenceRepo(session)
        self.vector_k = vector_k
        self.fts_k = fts_k
        self.rrf_k = rrf_k

    async def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[EvidenceSnippet]:
        """Return fused, provenance-bearing evidence snippets for `query`."""
        cleaned = query.strip()
        if not cleaned or top_k < 1:
            return []
        query_vector = await self._embedder.embed_query(cleaned)
        return await hybrid_search(
            self._session,
            cleaned,
            query_vector,
            top_k=top_k,
            vector_k=self.vector_k,
            fts_k=self.fts_k,
            rrf_k=self.rrf_k,
            repo=self._repo,
        )


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_embedding: Sequence[float],
    *,
    top_k: int = DEFAULT_TOP_K,
    vector_k: int = DEFAULT_VECTOR_K,
    fts_k: int = DEFAULT_FTS_K,
    rrf_k: int = DEFAULT_RRF_K,
    repo: EvidenceRepo | None = None,
) -> list[EvidenceSnippet]:
    """Run pgvector cosine top-K and ts_rank top-K, then fuse with RRF."""
    if len(query_embedding) != EMBEDDING_DIM:
        raise EmbeddingDimensionError(
            f"Query embedding has dimension {len(query_embedding)}, expected {EMBEDDING_DIM}."
        )
    vector_ids = await _vector_top_k(session, query_embedding, limit=vector_k)
    fts_ids = await _fts_top_k(session, query, limit=fts_k)
    fused = reciprocal_rank_fusion([vector_ids, fts_ids], k=rrf_k, limit=top_k)
    if not fused:
        return []

    evidence_repo = repo if repo is not None else EvidenceRepo(session)
    chunks = await evidence_repo.get_chunks_by_ids([chunk_id for chunk_id, _score in fused])
    by_id = {chunk.id: chunk for chunk in chunks}
    snippets: list[EvidenceSnippet] = []
    for chunk_id, score in fused:
        chunk = by_id.get(chunk_id)
        if chunk is None or chunk.source is None:
            continue
        snippets.append(_to_snippet(chunk, score))
    return snippets


async def _vector_top_k(
    session: AsyncSession,
    query_embedding: Sequence[float],
    *,
    limit: int,
) -> list[uuid.UUID]:
    if limit < 1:
        return []
    distance = EvidenceChunk.embedding.cosine_distance(list(query_embedding))
    stmt = (
        select(EvidenceChunk.id)
        .where(EvidenceChunk.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _fts_top_k(session: AsyncSession, query: str, *, limit: int) -> list[uuid.UUID]:
    if limit < 1 or not query.strip():
        return []
    english_query = func.plainto_tsquery("english", query)
    simple_query = func.plainto_tsquery("simple", query)
    simple_vector = func.to_tsvector(
        "simple",
        func.concat(func.coalesce(EvidenceChunk.section_label, ""), " ", EvidenceChunk.content),
    )
    rank = func.greatest(
        func.ts_rank(EvidenceChunk.tsv, english_query),
        func.ts_rank(simple_vector, simple_query),
    )
    matched = or_(
        EvidenceChunk.tsv.op("@@")(english_query),
        simple_vector.op("@@")(simple_query),
    )
    stmt = select(EvidenceChunk.id).where(matched).order_by(rank.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _to_snippet(chunk: EvidenceChunk, score: float) -> EvidenceSnippet:
    source: EvidenceSource = chunk.source
    return EvidenceSnippet(
        chunk_id=chunk.id,
        source_id=chunk.source_id,
        canonical_uri=_canonical_uri(source, chunk),
        content=chunk.content,
        section_label=chunk.section_label,
        source_type=source.source_type,
        source_title=source.title,
        chunk_index=chunk.chunk_index,
        score=score,
        token_count=chunk.token_count,
    )


def _canonical_uri(source: EvidenceSource, chunk: EvidenceChunk) -> str:
    metadata = source.metadata_json or {}
    anchors = metadata.get("chunk_anchors")
    if isinstance(anchors, list):
        for item in anchors:
            if (
                isinstance(item, dict)
                and item.get("chunk_index") == chunk.chunk_index
                and isinstance(item.get("canonical_uri"), str)
                and item["canonical_uri"]
            ):
                return str(item["canonical_uri"])
    if chunk.section_label and "#" not in source.canonical_uri:
        return f"{source.canonical_uri}#{chunk.section_label}"
    return source.canonical_uri
