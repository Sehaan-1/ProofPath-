"""Local README / experience-note folder ingester (Sub-part 1.3.1b)."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final
from uuid import UUID

from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceChunkInput, EvidenceRepo
from proofpath.ingestion.resume_parser import (
    SUPPORTED_SUFFIXES,
    ParsedDocument,
    parse_resume_file,
)

SKIP_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".nox",
    }
)

NOTE_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".markdown", ".txt"})


class IngestStatus(StrEnum):
    """Outcome of ingesting one local file into the Evidence Ledger."""

    CREATED = "created"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    REINDEXED = "reindexed"


@dataclass(frozen=True, slots=True)
class IngestFileResult:
    """Per-file ingest outcome including SHA-256 skip/re-index status."""

    path: Path
    canonical_uri: str
    content_sha256: str
    status: IngestStatus
    source_id: UUID
    chunk_count: int
    source_type: EvidenceSourceType


@dataclass(frozen=True, slots=True)
class FolderIngestResult:
    """Aggregate outcome of walking a local notes/README folder."""

    root: Path
    files: tuple[IngestFileResult, ...]

    @property
    def created(self) -> int:
        return sum(1 for item in self.files if item.status is IngestStatus.CREATED)

    @property
    def skipped(self) -> int:
        return sum(1 for item in self.files if item.status is IngestStatus.SKIPPED_UNCHANGED)

    @property
    def reindexed(self) -> int:
        return sum(1 for item in self.files if item.status is IngestStatus.REINDEXED)


class LocalEvidenceIngester:
    """Persist parsed local documents, skipping unchanged files by SHA-256."""

    def __init__(self, evidence_repo: EvidenceRepo) -> None:
        self._repo = evidence_repo

    async def ingest_file(
        self,
        path: Path | str,
        *,
        source_type: EvidenceSourceType | None = None,
    ) -> IngestFileResult:
        """Parse one local file and upsert it into the Evidence Ledger."""
        document = parse_resume_file(path, source_type=source_type)
        return await self._persist(document)

    async def ingest_folder(
        self,
        root: Path | str,
        *,
        suffixes: Sequence[str] | None = None,
    ) -> FolderIngestResult:
        """Walk `root` for READMEs and notes; skip unchanged, re-index updated files."""
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"Evidence folder not found: {resolved}")

        allowed = (
            {item.casefold() for item in suffixes} if suffixes is not None else set(NOTE_SUFFIXES)
        )
        results: list[IngestFileResult] = []
        for file_path in iter_local_notes(resolved, suffixes=allowed):
            results.append(await self.ingest_file(file_path))
        return FolderIngestResult(root=resolved, files=tuple(results))

    async def _persist(self, document: ParsedDocument) -> IngestFileResult:
        existing_sha = await self._repo.get_source_by_sha256(document.content_sha256)
        if existing_sha is not None:
            chunk_count = len(await self._repo.list_chunks_for_source(existing_sha.id))
            return IngestFileResult(
                path=document.path or Path(document.canonical_uri),
                canonical_uri=document.canonical_uri,
                content_sha256=document.content_sha256,
                status=IngestStatus.SKIPPED_UNCHANGED,
                source_id=existing_sha.id,
                chunk_count=chunk_count,
                source_type=existing_sha.source_type,
            )

        status = IngestStatus.CREATED
        existing_uri = await self._repo.get_source_by_uri(document.canonical_uri)
        if existing_uri is not None:
            await self._repo.delete_source(existing_uri.id)
            status = IngestStatus.REINDEXED

        source = await self._repo.create_source(
            source_type=document.source_type,
            title=document.title,
            canonical_uri=document.canonical_uri,
            content_sha256=document.content_sha256,
            metadata_json=dict(document.metadata),
        )
        chunk_inputs = [
            EvidenceChunkInput(
                chunk_index=index,
                content=chunk.content,
                token_count=chunk.token_count,
                section_label=chunk.section_label,
            )
            for index, chunk in enumerate(document.chunks)
        ]
        stored = await self._repo.add_chunks(source.id, chunk_inputs)
        return IngestFileResult(
            path=document.path or Path(document.canonical_uri),
            canonical_uri=document.canonical_uri,
            content_sha256=document.content_sha256,
            status=status,
            source_id=source.id,
            chunk_count=len(stored),
            source_type=source.source_type,
        )


def iter_local_notes(
    root: Path,
    *,
    suffixes: set[str] | None = None,
) -> Iterator[Path]:
    """Yield note/README files under `root`, skipping VCS and toolchain directories."""
    allowed = suffixes if suffixes is not None else set(NOTE_SUFFIXES)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in SKIP_DIR_NAMES and not name.startswith(".")
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.suffix.casefold() in allowed:
                yield path


def is_supported_local_file(path: Path) -> bool:
    """Return True when `path` is a Markdown, text, or PDF evidence file."""
    return path.suffix.casefold() in SUPPORTED_SUFFIXES


# Re-export for callers that only import the folder ingester.
__all__ = [
    "NOTE_SUFFIXES",
    "SKIP_DIR_NAMES",
    "FolderIngestResult",
    "IngestFileResult",
    "IngestStatus",
    "LocalEvidenceIngester",
    "is_supported_local_file",
    "iter_local_notes",
]
