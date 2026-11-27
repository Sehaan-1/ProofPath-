"""Live PostgreSQL tests for GitHub evidence persistence."""

from __future__ import annotations

import httpx
import pytest
import respx
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.ingestion import GitHubEvidenceClient, GitHubEvidenceIngester, IngestStatus

pytestmark = pytest.mark.integration


def _mock_repo(router: respx.MockRouter) -> None:
    base = "https://api.github.com/repos/octo/ledger"
    router.get(base).mock(
        return_value=httpx.Response(
            200,
            json={
                "full_name": "octo/ledger",
                "html_url": "https://github.com/octo/ledger",
                "description": "Evidence ledger",
                "default_branch": "main",
                "stargazers_count": 3,
            },
        )
    )
    router.get(f"{base}/languages").mock(return_value=httpx.Response(200, json={"Python": 1}))
    router.get(f"{base}/readme").mock(
        return_value=httpx.Response(200, text="# Ledger\n\nFastAPI + MCP.\n")
    )
    router.get(f"{base}/releases").mock(return_value=httpx.Response(200, json=[]))
    router.get(f"{base}/pulls").mock(return_value=httpx.Response(200, json=[]))


async def test_github_ingest_persists_canonical_uris_in_postgres(
    db_session: AsyncSession,
) -> None:
    repo = EvidenceRepo(db_session)
    with respx.mock(assert_all_called=True) as router:
        _mock_repo(router)
        async with GitHubEvidenceClient(token="t") as client:
            result = await GitHubEvidenceIngester(repo, client).ingest("octo/ledger")
        await db_session.flush()

    assert result.status is IngestStatus.CREATED
    assert result.canonical_uri == "https://github.com/octo/ledger"
    source = await repo.get_source_by_id(result.source_id, include_chunks=True)
    assert source is not None
    assert source.source_type is EvidenceSourceType.GITHUB_REPO
    assert source.canonical_uri == "https://github.com/octo/ledger"
    uris = [item["canonical_uri"] for item in source.metadata_json["chunk_anchors"]]
    assert any(str(uri).startswith("https://github.com/octo/ledger") for uri in uris)
    assert source.chunks
    assert any("FastAPI" in chunk.content or "Python" in chunk.content for chunk in source.chunks)
