"""Endpoint contracts and dependency wiring without external services."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from proofpath.api import routes
from proofpath.config import Settings
from proofpath.db.models import (
    ApplicationStage,
    EvidenceSourceType,
    RequirementCategory,
    RequirementPriority,
)
from proofpath.ingestion.markdown_ingester import IngestStatus
from proofpath.main import create_app
from proofpath.schemas.evidence import EvidenceSnippet

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    session = MagicMock()
    session.flush = AsyncMock()
    app = create_app(settings=Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None, embedding_model="hash", github_token=None)

    async def provide_session():
        yield session

    app.dependency_overrides[routes.session_dependency] = provide_session
    test_client = TestClient(app)
    try:
        yield test_client, session
    finally:
        test_client.close()


@pytest.fixture
def fake_ingest(monkeypatch):
    source_id = uuid4()
    source = SimpleNamespace(
        id=source_id, source_type=EvidenceSourceType.RESUME, canonical_uri="file:///resume"
    )
    chunk = SimpleNamespace(content="Python projects", embedding=None)
    repo = MagicMock()
    repo.get_source_by_sha256 = AsyncMock(return_value=None)
    repo.get_source_by_uri = AsyncMock(return_value=None)
    repo.list_chunks_for_source = AsyncMock(return_value=[chunk])
    repo.get_source_by_id = AsyncMock(return_value=source)
    monkeypatch.setattr(routes, "EvidenceRepo", lambda _session: repo)
    persist = AsyncMock(
        return_value=SimpleNamespace(
            source_id=source_id, chunk_count=1, content_sha256="a" * 64, status=IngestStatus.CREATED
        )
    )
    monkeypatch.setattr(routes, "persist_parsed_document", persist)
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.2] * 1536])
    monkeypatch.setattr(routes, "create_embedding_client", lambda _settings: embedder)
    return repo, persist, chunk, source


def test_text_and_file_ingestion_embed_chunks(client, fake_ingest):
    test_client, session = client
    _, persist, chunk, source = fake_ingest
    response = test_client.post(
        "/api/v1/evidence/ingest/text",
        json={
            "title": "Resume",
            "content": "Python projects",
            "canonical_uri": "file:///resume",
            "source_type": "resume",
        },
    )
    assert response.status_code == 201
    assert response.json()["source_id"] == str(source.id)
    assert chunk.embedding == [0.2] * 1536
    assert persist.await_args.args[1].text == "Python projects"
    assert session.flush.await_count == 1
    response = test_client.post(
        "/api/v1/evidence/ingest/file",
        files={"file": ("resume.txt", b"Python projects", "text/plain")},
        data={"canonical_uri": "file:///resume"},
    )
    assert response.status_code == 201
    assert persist.await_args.args[1].canonical_uri == "file:///resume"


def test_ingestion_rejects_conflicting_uri_and_invalid_file(client, fake_ingest):
    test_client, _ = client
    repo, persist, _, _ = fake_ingest
    repo.get_source_by_uri.return_value = object()
    response = test_client.post(
        "/api/v1/evidence/ingest/text",
        json={"title": "Resume", "content": "Python", "canonical_uri": "file:///resume"},
    )
    assert response.status_code == 409
    assert response.headers["content-type"] == "application/problem+json"
    persist.assert_not_awaited()
    response = test_client.post(
        "/api/v1/evidence/ingest/file",
        files={"file": ("x.exe", b"123", "application/octet-stream")},
        data={"canonical_uri": "file:///x"},
    )
    assert response.status_code == 422


def test_evidence_search_uses_hybrid_search_and_validates(client, monkeypatch):
    test_client, _ = client
    snippet = EvidenceSnippet(
        chunk_id=uuid4(), source_id=uuid4(), content="Python", canonical_uri="file:///resume"
    )
    searcher = MagicMock()
    searcher.search = AsyncMock(return_value=[snippet])
    monkeypatch.setattr(routes, "HybridSearcher", lambda _session, _embedder: searcher)
    monkeypatch.setattr(routes, "create_embedding_client", lambda _settings: object())
    response = test_client.get("/api/v1/evidence/search", params={"query": " Python ", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["results"][0]["chunk_id"] == str(snippet.chunk_id)
    searcher.search.assert_awaited_once_with("Python", top_k=3)
    assert test_client.get("/api/v1/evidence/search", params={"query": "  "}).status_code == 422
    assert (
        test_client.get(
            "/api/v1/evidence/search", params={"query": "Python", "top_k": 51}
        ).status_code
        == 422
    )


def test_opportunity_and_tracker_reads(client, monkeypatch):
    test_client, _ = client
    opp_id = uuid4()
    requirement = SimpleNamespace(
        id=uuid4(),
        opportunity_id=opp_id,
        entity="Python",
        requirement_text="Python",
        category=RequirementCategory.LANGUAGE,
        priority=RequirementPriority.MUST_HAVE,
    )
    opp = SimpleNamespace(
        id=opp_id,
        company="Company",
        role_title="Engineer",
        source_url=None,
        created_at=datetime.now(UTC),
        requirements=[requirement],
    )
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=opp)
    monkeypatch.setattr(routes, "OpportunityRepo", lambda _session: repo)
    response = test_client.get(f"/api/v1/opportunities/{opp_id}")
    assert response.status_code == 200
    assert response.json()["requirements"][0]["entity"] == "Python"
    requirement.entity = None
    assert test_client.get(f"/api/v1/opportunities/{opp_id}").status_code == 409
    repo.get_by_id.return_value = None
    assert test_client.get(f"/api/v1/opportunities/{opp_id}").status_code == 404
    tracker = MagicMock()
    tracker.list_entries = AsyncMock(
        return_value=[
            SimpleNamespace(
                id=uuid4(),
                opportunity_id=opp_id,
                stage=ApplicationStage.SAVED,
                notes="keep",
                updated_at=datetime.now(UTC),
            )
        ]
    )
    monkeypatch.setattr(routes, "TrackerRepo", lambda _session: tracker)
    response = test_client.get("/api/v1/tracker?limit=5&offset=1")
    assert response.status_code == 200
    assert response.json()[0]["notes"] == "keep"
    tracker.list_entries.assert_awaited_once_with(limit=5, offset=1)


def test_opportunity_url_only_is_not_fetched(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/opportunities", json={"source_url": "https://example.com/job"}
    )
    assert response.status_code == 422
    assert "URL-only" in response.json()["detail"]


async def test_request_session_rolls_back_on_failure():
    events = []

    @asynccontextmanager
    async def begin():
        try:
            yield object()
        except Exception:
            events.append("rollback")
            raise
        else:
            events.append("commit")

    request = MagicMock()
    request.app.state.session_factory.begin = begin
    generator = routes.session_dependency(request)
    await anext(generator)
    with pytest.raises(RuntimeError):
        await generator.athrow(RuntimeError("failed"))
    assert events == ["rollback"]


def test_github_ingestion_fetches_github_snapshot(client, fake_ingest, monkeypatch):
    test_client, _ = client
    _, persist, _, _ = fake_ingest
    doc = routes.parse_resume_text(
        "README Python",
        title="repo",
        canonical_uri="https://github.com/example/repo",
        source_type=EvidenceSourceType.GITHUB_REPO,
    )
    snapshot = object()
    calls = []

    class FakeClient:
        def __init__(self, token=None):
            calls.append(token)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def fetch_snapshot(self, ref):
            calls.append(ref.full_name)
            return snapshot

    monkeypatch.setattr(routes, "GitHubEvidenceClient", FakeClient)
    monkeypatch.setattr(routes, "snapshot_to_document", lambda _snapshot: doc)
    response = test_client.post(
        "/api/v1/evidence/ingest/github", json={"owner": "example", "name": "repo"}
    )
    assert response.status_code == 201
    assert calls == [None, "example/repo"]
    assert persist.await_args.args[1] is doc


def test_opportunity_creation_uses_parser_and_response(client, monkeypatch):
    from proofpath.schemas.opportunity import OpportunityResponse

    test_client, _ = client
    client_settings = Settings(_env_file=None)
    test_client.app.state.settings = client_settings
    output = OpportunityResponse(
        id=uuid4(),
        company="Example",
        role_title="Engineer",
        source_url=None,
        created_at=datetime.now(UTC),
        requirements=[],
    )
    extractor = MagicMock()
    extractor.analyse = AsyncMock(return_value=SimpleNamespace(to_response=lambda: output))
    monkeypatch.setattr(routes, "create_role_parser", lambda _settings: object())
    monkeypatch.setattr(routes, "AnalyseRoleExtractor", lambda _repo, _parser: extractor)
    response = test_client.post("/api/v1/opportunities", json={"raw_text": "Python engineer"})
    assert response.status_code == 201
    assert response.json()["id"] == str(output.id)
    assert extractor.analyse.await_args.args[0].raw_text == "Python engineer"


def test_upload_size_limit(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 3)
    response = test_client.post(
        "/api/v1/evidence/ingest/file",
        files={"file": ("note.txt", b"four", "text/plain")},
        data={"canonical_uri": "file:///note"},
    )
    assert response.status_code == 413
