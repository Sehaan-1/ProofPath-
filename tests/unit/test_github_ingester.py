"""Unit tests for Sub-slice 1.3.2 (GitHub REST evidence ingester)."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import pytest
import respx

from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceChunkInput
from proofpath.ingestion import (
    GitHubEvidenceClient,
    GitHubEvidenceIngester,
    GitHubIngestError,
    GitHubRepoRef,
    IngestStatus,
    canonical_github_uri,
)
from proofpath.schemas.evidence import IngestGitHubRequest


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

    async def get_source_by_sha256(self, content_sha256: str) -> _FakeSource | None:
        for source in self.sources:
            if source.content_sha256 == content_sha256:
                return source
        return None

    async def get_source_by_uri(self, canonical_uri: str) -> _FakeSource | None:
        matches = [source for source in self.sources if source.canonical_uri == canonical_uri]
        return matches[-1] if matches else None

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


def _mock_github_repo(router: respx.MockRouter, *, owner: str = "octo", name: str = "demo") -> None:
    base = f"https://api.github.com/repos/{owner}/{name}"
    router.get(base).mock(
        return_value=httpx.Response(
            200,
            json={
                "full_name": f"{owner}/{name}",
                "html_url": f"https://github.com/{owner}/{name}",
                "description": "MCP + FastAPI evidence agent.",
                "default_branch": "main",
                "stargazers_count": 12,
            },
        )
    )
    router.get(f"{base}/languages").mock(
        return_value=httpx.Response(200, json={"Python": 9000, "TypeScript": 100})
    )
    router.get(f"{base}/readme").mock(
        return_value=httpx.Response(
            200,
            text="# Demo\n\nShipped MCP tools over pgvector.\n",
            headers={"Content-Type": "text/plain"},
        )
    )
    router.get(f"{base}/releases").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "tag_name": "v1.0.0",
                    "name": "v1.0.0",
                    "body": "First public MCP release.",
                    "html_url": f"https://github.com/{owner}/{name}/releases/tag/v1.0.0",
                }
            ],
        )
    )
    router.get(f"{base}/pulls").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "number": 7,
                    "title": "Add hybrid search",
                    "body": "Uses pgvector cosine + ts_rank.",
                    "html_url": f"https://github.com/{owner}/{name}/pull/7",
                }
            ],
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "owner", "name"),
    [
        ("https://github.com/Sehaan-1/ProofPath-", "Sehaan-1", "ProofPath-"),
        ("https://github.com/Sehaan-1/ProofPath-.git", "Sehaan-1", "ProofPath-"),
        ("git@github.com:Sehaan-1/ProofPath-.git", "Sehaan-1", "ProofPath-"),
        ("Sehaan-1/ProofPath-", "Sehaan-1", "ProofPath-"),
    ],
)
def test_parse_github_repo_ref(value: str, owner: str, name: str) -> None:
    ref = GitHubRepoRef.parse(value)
    assert ref.owner == owner
    assert ref.name == name
    assert ref.html_url == f"https://github.com/{owner}/{name}"
    assert "api.github.com" not in ref.html_url


@pytest.mark.unit
def test_repo_ref_from_ingest_request_and_canonical_uri() -> None:
    from_url = GitHubRepoRef.from_request(
        IngestGitHubRequest.model_validate({"repo_url": "https://github.com/octo/demo"})
    )
    assert from_url == GitHubRepoRef("octo", "demo")
    from_parts = GitHubRepoRef.from_request(
        IngestGitHubRequest.model_validate({"owner": "octo", "name": "demo"})
    )
    assert from_parts.html_url == "https://github.com/octo/demo"
    assert canonical_github_uri("octo", "demo", "Languages") == (
        "https://github.com/octo/demo#languages"
    )
    with pytest.raises(ValueError, match="Invalid GitHub"):
        GitHubRepoRef.parse("https://gitlab.com/octo/demo")


@pytest.mark.unit
async def test_client_fetches_repo_languages_readme_releases_and_pulls() -> None:
    with respx.mock(assert_all_called=True) as router:
        _mock_github_repo(router)
        async with GitHubEvidenceClient(token="ghp_test_token_123") as client:
            snapshot = await client.fetch_snapshot(GitHubRepoRef("octo", "demo"))
        auth = router.calls[0].request.headers["Authorization"]

    assert snapshot.html_url == "https://github.com/octo/demo"
    assert snapshot.languages == ("Python", "TypeScript")
    assert "MCP tools" in snapshot.readme_markdown
    assert snapshot.releases[0].tag_name == "v1.0.0"
    assert snapshot.pull_requests[0].number == 7
    assert auth == "Bearer ghp_test_token_123"
    assert "ghp_test_token_123" not in str(snapshot)


@pytest.mark.unit
async def test_client_omits_authorization_without_token_and_maps_404() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.github.com/repos/octo/missing").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        async with GitHubEvidenceClient() as client:
            with pytest.raises(GitHubIngestError, match="not found") as exc_info:
                await client.fetch_snapshot(GitHubRepoRef("octo", "missing"))
        assert exc_info.value.status_code == 404
        assert "Authorization" not in router.calls[0].request.headers


@pytest.mark.unit
async def test_ingester_persists_canonical_github_chunk_uris() -> None:
    repo = _FakeEvidenceRepo()
    with respx.mock(assert_all_called=True) as router:
        _mock_github_repo(router)
        async with GitHubEvidenceClient(token="t") as client:
            ingester = GitHubEvidenceIngester(repo, client)  # type: ignore[arg-type]
            result = await ingester.ingest("https://github.com/octo/demo")

    assert result.status is IngestStatus.CREATED
    assert result.canonical_uri == "https://github.com/octo/demo"
    assert result.chunk_count >= 3
    source = repo.sources[0]
    assert source.source_type is EvidenceSourceType.GITHUB_REPO
    assert source.canonical_uri == "https://github.com/octo/demo"
    uris = [str(item["canonical_uri"]) for item in source.metadata_json["chunk_anchors"]]
    assert "https://github.com/octo/demo#languages" in uris
    assert "https://github.com/octo/demo/releases/tag/v1.0.0" in uris
    assert "https://github.com/octo/demo/pull/7" in uris
    assert all(uri.startswith("https://github.com/") for uri in uris)
    assert all("api.github.com" not in uri for uri in uris)

    with respx.mock(assert_all_called=True) as router:
        _mock_github_repo(router)
        async with GitHubEvidenceClient(token="t") as client:
            ingester = GitHubEvidenceIngester(repo, client)  # type: ignore[arg-type]
            skipped = await ingester.ingest("octo/demo")
    assert skipped.status is IngestStatus.SKIPPED_UNCHANGED
    assert skipped.source_id == result.source_id
    assert len(repo.sources) == 1


@pytest.mark.live
async def test_live_github_public_repo_snapshot() -> None:
    if os.getenv("LIVE_GITHUB", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set LIVE_GITHUB=1 to hit the real GitHub REST API.")
    token = os.getenv("GITHUB_TOKEN") or None
    async with GitHubEvidenceClient(token=token) as client:
        snapshot = await client.fetch_snapshot(GitHubRepoRef("octocat", "Hello-World"))
    assert snapshot.html_url.startswith("https://github.com/octocat/Hello-World")
    assert snapshot.ref.full_name == "octocat/Hello-World"
