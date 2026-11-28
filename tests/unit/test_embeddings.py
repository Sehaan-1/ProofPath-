"""Unit tests for Sub-part 1.3.3a (embedding pipeline)."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest
import respx
from pydantic import SecretStr

from proofpath.config import LLMProvider, Settings
from proofpath.db.models import EMBEDDING_DIM
from proofpath.db.repositories.evidence import EvidenceChunkInput
from proofpath.retrieval import (
    EmbeddingClient,
    EmbeddingDimensionError,
    GeminiEmbeddingBackend,
    HashEmbeddingBackend,
    OllamaEmbeddingBackend,
    attach_embeddings,
    create_embedding_client,
)


class _FakeBackend:
    def __init__(self, vectors: list[list[float]] | None = None, *, fail_times: int = 0) -> None:
        self.vectors = vectors
        self.fail_times = fail_times
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise httpx.ConnectError("transient embedding failure")
        if self.vectors is not None:
            return self.vectors
        return [[0.1] * EMBEDDING_DIM for _ in texts]


@pytest.mark.unit
async def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    backend = HashEmbeddingBackend(dimensions=EMBEDDING_DIM)
    client = EmbeddingClient(backend, model="hash", dimensions=EMBEDDING_DIM)
    first = await client.embed(["FastAPI", "SAP S/4HANA"])
    second = await client.embed(["FastAPI", "SAP S/4HANA"])
    assert first == second
    assert len(first) == 2
    assert all(len(vector) == EMBEDDING_DIM for vector in first)
    assert first[0] != first[1]
    norm = sum(component * component for component in first[0]) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)
    query = await client.embed_query("FastAPI")
    assert query == first[0]


@pytest.mark.unit
async def test_embedding_client_batches_and_retries_transient_errors() -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    backend = _FakeBackend(fail_times=2)
    client = EmbeddingClient(
        backend,
        model="hash",
        dimensions=EMBEDDING_DIM,
        batch_size=2,
        max_retries=3,
        base_delay_seconds=0.5,
        sleeper=fake_sleep,
    )
    texts = ["MCP", "pgvector", "Helm"]
    vectors = await client.embed(texts)
    assert len(vectors) == 3
    assert backend.calls[0] == ["MCP", "pgvector"]
    assert backend.calls[-1] == ["Helm"]
    assert delays == [0.5, 1.0]


@pytest.mark.unit
async def test_dimension_mismatch_fails_fast() -> None:
    backend = _FakeBackend(vectors=[[0.1, 0.2, 0.3]])
    client = EmbeddingClient(backend, model="test", dimensions=EMBEDDING_DIM, max_retries=0)
    with pytest.raises(EmbeddingDimensionError, match=str(EMBEDDING_DIM)):
        await client.embed(["SAP"])


@pytest.mark.unit
async def test_attach_embeddings_preserves_chunk_order() -> None:
    chunks = [
        EvidenceChunkInput(chunk_index=0, content="MCP", token_count=1),
        EvidenceChunkInput(chunk_index=1, content="Helm", token_count=1),
    ]
    vectors = [[0.0] * EMBEDDING_DIM, [1.0] + [0.0] * (EMBEDDING_DIM - 1)]
    attached = attach_embeddings(chunks, vectors)
    assert attached[0].embedding is not None
    assert attached[1].embedding is not None
    assert attached[0].content == "MCP"
    assert attached[1].chunk_index == 1
    with pytest.raises(EmbeddingDimensionError):
        attach_embeddings(chunks, vectors[:1])


@pytest.mark.unit
async def test_gemini_and_ollama_backends_post_expected_payloads() -> None:
    dummy = [0.0] * 8
    with respx.mock(assert_all_called=True) as router:
        router.post(
            "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents"
        ).mock(return_value=httpx.Response(200, json={"embeddings": [{"values": dummy}]}))
        gemini = GeminiEmbeddingBackend(api_key="gk-test", model="text-embedding-004")
        gemini_vectors = await gemini.embed_batch(["MCP"])
        assert gemini_vectors == [dummy]

        router.post("http://127.0.0.1:11434/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [dummy]})
        )
        ollama = OllamaEmbeddingBackend(model="nomic-embed-text")
        ollama_vectors = await ollama.embed_batch(["MCP"])
        assert ollama_vectors == [dummy]


@pytest.mark.unit
async def test_factory_uses_hash_backend_for_local_model_name() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LLMProvider.OPENAI,
        embedding_model="local-hash",
        embedding_dimensions=EMBEDDING_DIM,
        redact_pii=True,
    )
    client = create_embedding_client(settings)
    assert isinstance(client._backend, HashEmbeddingBackend)
    assert client.dimensions == EMBEDDING_DIM
    local = ["CV: 12 Main St. Tel: 415-555-0173"]
    assert await client.embed(local) == await HashEmbeddingBackend().embed_batch(local)


@pytest.mark.unit
@pytest.mark.parametrize("redact", [True, False])
@pytest.mark.parametrize("provider", [LLMProvider.OPENAI, LLMProvider.GEMINI])
async def test_hosted_embedding_requests_respect_pii_switch(provider, redact):
    source = "CV: 42 Main Street; call 415-555-0123."
    settings = Settings(
        _env_file=None,
        llm_provider=provider,
        llm_api_key=SecretStr("test-key"),
        embedding_model="test-model",
        embedding_dimensions=64,
        redact_pii=redact,
    )
    vector = [0.0] * 64
    if provider == LLMProvider.OPENAI:
        url = "https://api.openai.com/v1/embeddings"
        response = {"data": [{"index": 0, "embedding": vector}]}
    else:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/test-model:batchEmbedContents"
        )
        response = {"embeddings": [{"values": vector}]}
    with respx.mock(assert_all_called=True) as router:
        route = router.post(url).mock(return_value=httpx.Response(200, json=response))
        assert await create_embedding_client(settings).embed_query(source) == vector
        data = json.loads(route.calls.last.request.content)
    sent = (
        data["input"][0]
        if provider == LLMProvider.OPENAI
        else data["requests"][0]["content"]["parts"][0]["text"]
    )
    assert sent == ("CV: [REDACTED ADDRESS]; call [REDACTED PHONE]." if redact else source)
