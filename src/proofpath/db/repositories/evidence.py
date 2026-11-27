"""Typed async repository for Evidence Ledger sources and chunks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from proofpath.db.models import (
    EMBEDDING_DIM,
    EvidenceChunk,
    EvidenceSource,
    EvidenceSourceType,
)


@dataclass(frozen=True, slots=True)
class EvidenceChunkInput:
    """Typed input payload for inserting an `EvidenceChunk`."""

    chunk_index: int
    content: str
    token_count: int
    section_label: str | None = None
    embedding: Sequence[float] | None = None
    chunk_id: uuid.UUID | None = None


class EvidenceRepo:
    """Async CRUD repository for `evidence_sources` and `evidence_chunks`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_source(
        self,
        *,
        source_type: EvidenceSourceType,
        title: str,
        canonical_uri: str,
        content_sha256: str,
        metadata_json: dict[str, Any] | None = None,
        source_id: uuid.UUID | None = None,
    ) -> EvidenceSource:
        """Insert a new `EvidenceSource` and flush (raises IntegrityError on duplicate SHA-256)."""
        source = EvidenceSource(
            id=source_id or uuid.uuid4(),
            source_type=source_type,
            title=title,
            canonical_uri=canonical_uri,
            content_sha256=content_sha256,
            metadata_json=dict(metadata_json) if metadata_json is not None else {},
        )
        self._session.add(source)
        await self._session.flush()
        return source

    async def get_or_create_source(
        self,
        *,
        source_type: EvidenceSourceType,
        title: str,
        canonical_uri: str,
        content_sha256: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> tuple[EvidenceSource, bool]:
        """Return `(source, created)` deduplicating by `content_sha256`."""
        existing = await self.get_source_by_sha256(content_sha256)
        if existing is not None:
            return existing, False
        created = await self.create_source(
            source_type=source_type,
            title=title,
            canonical_uri=canonical_uri,
            content_sha256=content_sha256,
            metadata_json=metadata_json,
        )
        return created, True

    async def get_source_by_id(
        self,
        source_id: uuid.UUID,
        *,
        include_chunks: bool = False,
    ) -> EvidenceSource | None:
        """Fetch an `EvidenceSource` by UUID, optionally eager-loading ordered chunks."""
        stmt = select(EvidenceSource).where(EvidenceSource.id == source_id)
        if include_chunks:
            stmt = stmt.options(selectinload(EvidenceSource.chunks))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_source_by_sha256(
        self,
        content_sha256: str,
        *,
        include_chunks: bool = False,
    ) -> EvidenceSource | None:
        """Look up an `EvidenceSource` by its unique SHA-256 content digest."""
        stmt = select(EvidenceSource).where(EvidenceSource.content_sha256 == content_sha256)
        if include_chunks:
            stmt = stmt.options(selectinload(EvidenceSource.chunks))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_source_by_uri(
        self,
        canonical_uri: str,
        *,
        include_chunks: bool = False,
    ) -> EvidenceSource | None:
        """Look up the most recent `EvidenceSource` matching a canonical URI."""
        stmt = (
            select(EvidenceSource)
            .where(EvidenceSource.canonical_uri == canonical_uri)
            .order_by(EvidenceSource.ingested_at.desc())
            .limit(1)
        )
        if include_chunks:
            stmt = stmt.options(selectinload(EvidenceSource.chunks))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sources(
        self,
        *,
        source_type: EvidenceSourceType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EvidenceSource]:
        """List ingested evidence sources ordered newest-first."""
        stmt = select(EvidenceSource).order_by(EvidenceSource.ingested_at.desc())
        if source_type is not None:
            stmt = stmt.where(EvidenceSource.source_type == source_type)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_source(self, source_id: uuid.UUID) -> bool:
        """Delete an `EvidenceSource` (cascades to child `evidence_chunks`)."""
        result = await self._session.execute(
            delete(EvidenceSource)
            .where(EvidenceSource.id == source_id)
            .returning(EvidenceSource.id)
        )
        return result.scalar_one_or_none() is not None

    async def add_chunks(
        self,
        source_id: uuid.UUID,
        chunks: Sequence[EvidenceChunkInput],
    ) -> list[EvidenceChunk]:
        """Insert a batch of `EvidenceChunk` rows for an existing source."""
        models: list[EvidenceChunk] = []
        for item in chunks:
            embedding_list: list[float] | None = None
            if item.embedding is not None:
                if len(item.embedding) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Expected embedding dimension {EMBEDDING_DIM}, got {len(item.embedding)}."
                    )
                embedding_list = [float(v) for v in item.embedding]
            models.append(
                EvidenceChunk(
                    id=item.chunk_id or uuid.uuid4(),
                    source_id=source_id,
                    chunk_index=item.chunk_index,
                    section_label=item.section_label,
                    content=item.content,
                    embedding=embedding_list,
                    token_count=item.token_count,
                )
            )
        self._session.add_all(models)
        await self._session.flush()
        return models

    async def replace_chunks(
        self,
        source_id: uuid.UUID,
        chunks: Sequence[EvidenceChunkInput],
    ) -> list[EvidenceChunk]:
        """Atomically replace all chunks belonging to `source_id` within the active transaction."""
        await self._session.execute(
            delete(EvidenceChunk).where(EvidenceChunk.source_id == source_id)
        )
        await self._session.flush()
        return await self.add_chunks(source_id, chunks)

    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> EvidenceChunk | None:
        """Fetch a single `EvidenceChunk` with its parent `EvidenceSource` loaded."""
        stmt = (
            select(EvidenceChunk)
            .where(EvidenceChunk.id == chunk_id)
            .options(selectinload(EvidenceChunk.source))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_chunks_by_ids(self, chunk_ids: Sequence[uuid.UUID]) -> list[EvidenceChunk]:
        """Fetch multiple `EvidenceChunk` rows with their parent sources loaded."""
        if not chunk_ids:
            return []
        stmt = (
            select(EvidenceChunk)
            .where(EvidenceChunk.id.in_(chunk_ids))
            .options(selectinload(EvidenceChunk.source))
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_existing_chunk_ids(self, chunk_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        """Return the subset of `chunk_ids` that actually exist in the Evidence Ledger."""
        if not chunk_ids:
            return set()
        stmt = select(EvidenceChunk.id).where(EvidenceChunk.id.in_(chunk_ids))
        result = await self._session.execute(stmt)
        return set(result.scalars().all())

    async def list_chunks_for_source(self, source_id: uuid.UUID) -> list[EvidenceChunk]:
        """Return all chunks for a source ordered by `chunk_index`."""
        stmt = (
            select(EvidenceChunk)
            .where(EvidenceChunk.source_id == source_id)
            .order_by(EvidenceChunk.chunk_index.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
