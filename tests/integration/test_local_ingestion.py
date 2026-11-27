"""Live PostgreSQL tests for local resume/notes ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.ingestion import IngestStatus, LocalEvidenceIngester

pytestmark = pytest.mark.integration

_CV = """\
# Experience

Built FastAPI + pgvector search.

# Skills

Python, MCP
"""


async def test_ingest_folder_dedupes_sha_and_reindexes_in_postgres(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("# Ledger\n\nMCP tools.\n", encoding="utf-8")
    note = tmp_path / "experience.md"
    note.write_text(_CV, encoding="utf-8")

    repo = EvidenceRepo(db_session)
    ingester = LocalEvidenceIngester(repo)

    first = await ingester.ingest_folder(tmp_path)
    await db_session.flush()
    assert first.created == 2
    assert first.skipped == 0

    sources = await repo.list_sources()
    assert len(sources) == 2
    readme = next(item for item in sources if item.source_type is EvidenceSourceType.README)
    chunks = await repo.list_chunks_for_source(readme.id)
    assert chunks
    assert chunks[0].content
    assert readme.metadata_json["chunk_anchors"][0]["start_line"] >= 1

    second = await ingester.ingest_folder(tmp_path)
    await db_session.flush()
    assert second.skipped == 2
    assert len(await repo.list_sources()) == 2

    note.write_text(_CV + "\n\nAdded Helm.\n", encoding="utf-8")
    third = await ingester.ingest_folder(tmp_path)
    await db_session.flush()
    updated = next(item for item in third.files if item.path.name == "experience.md")
    assert updated.status is IngestStatus.REINDEXED
    remaining = await repo.list_sources()
    assert len(remaining) == 2
    note_source = await repo.get_source_by_id(updated.source_id, include_chunks=True)
    assert note_source is not None
    assert "Helm" in " ".join(chunk.content for chunk in note_source.chunks)
