"""Unit tests for Sub-part 1.3.1b (README/notes folder ingester)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceChunkInput
from proofpath.ingestion import (
    IngestStatus,
    LocalEvidenceIngester,
    infer_source_type,
    iter_local_notes,
)


class _FakeSource:
    def __init__(
        self,
        *,
        source_type: EvidenceSourceType,
        title: str,
        canonical_uri: str,
        content_sha256: str,
        metadata_json: dict[str, Any],
    ) -> None:
        self.id = uuid.uuid4()
        self.source_type = source_type
        self.title = title
        self.canonical_uri = canonical_uri
        self.content_sha256 = content_sha256
        self.metadata_json = metadata_json
        self.chunks: list[EvidenceChunkInput] = []


class _FakeEvidenceRepo:
    def __init__(self) -> None:
        self.sources: list[_FakeSource] = []

    def _by_sha(self, digest: str) -> _FakeSource | None:
        for source in self.sources:
            if source.content_sha256 == digest:
                return source
        return None

    def _by_uri(self, uri: str) -> _FakeSource | None:
        matches = [source for source in self.sources if source.canonical_uri == uri]
        return matches[-1] if matches else None

    async def get_source_by_sha256(self, content_sha256: str) -> _FakeSource | None:
        return self._by_sha(content_sha256)

    async def get_source_by_uri(self, canonical_uri: str) -> _FakeSource | None:
        return self._by_uri(canonical_uri)

    async def delete_source(self, source_id: uuid.UUID) -> bool:
        before = len(self.sources)
        self.sources = [source for source in self.sources if source.id != source_id]
        return len(self.sources) != before

    async def create_source(
        self,
        *,
        source_type: EvidenceSourceType,
        title: str,
        canonical_uri: str,
        content_sha256: str,
        metadata_json: dict[str, Any] | None = None,
        source_id: uuid.UUID | None = None,
    ) -> _FakeSource:
        source = _FakeSource(
            source_type=source_type,
            title=title,
            canonical_uri=canonical_uri,
            content_sha256=content_sha256,
            metadata_json=dict(metadata_json) if metadata_json is not None else {},
        )
        if source_id is not None:
            source.id = source_id
        self.sources.append(source)
        return source

    async def add_chunks(
        self,
        source_id: uuid.UUID,
        chunks: list[EvidenceChunkInput],
    ) -> list[EvidenceChunkInput]:
        source = next(item for item in self.sources if item.id == source_id)
        source.chunks = list(chunks)
        return list(chunks)

    async def list_chunks_for_source(self, source_id: uuid.UUID) -> list[EvidenceChunkInput]:
        source = next(item for item in self.sources if item.id == source_id)
        return list(source.chunks)


_README = """\
# ProofPath

Hybrid pgvector search with FastAPI.

## Usage

Run make up.
"""

_NOTE = """\
# Experience note

Shipped MCP tools against a cited evidence ledger.
"""


@pytest.mark.unit
def test_iter_local_notes_skips_hidden_and_toolchain_dirs(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(_README, encoding="utf-8")
    (tmp_path / "notes.md").write_text(_NOTE, encoding="utf-8")
    (tmp_path / "ignore.py").write_text("print('nope')\n", encoding="utf-8")
    nested = tmp_path / "projects" / "demo"
    nested.mkdir(parents=True)
    (nested / "README.md").write_text("# Demo\n", encoding="utf-8")
    skipped = tmp_path / ".venv" / "lib"
    skipped.mkdir(parents=True)
    (skipped / "README.md").write_text("# should skip\n", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "README.md").write_text("# git\n", encoding="utf-8")

    found = [path.name for path in iter_local_notes(tmp_path)]
    assert found.count("README.md") == 2
    assert "notes.md" in found
    assert "ignore.py" not in found


@pytest.mark.unit
async def test_ingest_folder_skips_unchanged_and_reindexes_updated_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(_README, encoding="utf-8")
    notes = tmp_path / "experience.md"
    notes.write_text(_NOTE, encoding="utf-8")

    repo = _FakeEvidenceRepo()
    ingester = LocalEvidenceIngester(repo)  # type: ignore[arg-type]

    first = await ingester.ingest_folder(tmp_path)
    assert first.created == 2
    assert first.skipped == 0
    assert first.reindexed == 0
    assert {item.source_type for item in first.files} == {
        EvidenceSourceType.README,
        EvidenceSourceType.EXPERIENCE_NOTE,
    }
    assert all(item.chunk_count >= 1 for item in first.files)
    original_ids = {item.canonical_uri: item.source_id for item in first.files}

    second = await ingester.ingest_folder(tmp_path)
    assert second.created == 0
    assert second.skipped == 2
    assert second.reindexed == 0
    assert {item.source_id for item in second.files} == set(original_ids.values())

    notes.write_text(_NOTE + "\n\nAdded a FastAPI follow-up.\n", encoding="utf-8")
    third = await ingester.ingest_folder(tmp_path)
    by_path = {item.path.name: item for item in third.files}
    assert by_path["README.md"].status is IngestStatus.SKIPPED_UNCHANGED
    assert by_path["experience.md"].status is IngestStatus.REINDEXED
    assert (
        by_path["experience.md"].source_id != original_ids[by_path["experience.md"].canonical_uri]
    )
    assert third.reindexed == 1
    assert third.skipped == 1
    assert len(repo.sources) == 2


@pytest.mark.unit
async def test_ingest_file_persists_chunk_anchors_on_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text(_README, encoding="utf-8")
    repo = _FakeEvidenceRepo()
    ingester = LocalEvidenceIngester(repo)  # type: ignore[arg-type]
    result = await ingester.ingest_file(path)

    assert result.status is IngestStatus.CREATED
    assert infer_source_type(path) is EvidenceSourceType.README
    source = repo.sources[0]
    anchors = source.metadata_json["chunk_anchors"]
    assert isinstance(anchors, list)
    assert anchors
    assert "start_line" in anchors[0]
    assert "anchor" in anchors[0]
    assert source.chunks[0].section_label in {"ProofPath", "Usage", None, "Summary"}
