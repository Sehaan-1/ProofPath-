"""Embedding client with batching, dimension checks, and exponential backoff."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Final, Protocol

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from proofpath.config import LLMProvider, Settings
from proofpath.db.models import EMBEDDING_DIM
from proofpath.db.repositories.evidence import EvidenceChunkInput
from proofpath.policy.injection_guard import mask_pii

DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BASE_DELAY_SECONDS: Final[float] = 0.5
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 529})
Sleeper = Callable[[float], Awaitable[None]]


class EmbeddingDimensionError(ValueError):
    """Raised when a provider returns a vector of the wrong dimensionality."""


class EmbeddingBackend(Protocol):
    """Provider-specific batch embedding call (no retry/batching of its own)."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingClient:
    """Provider-agnostic embedding pipeline (Sub-part 1.3.3a)."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        model: str,
        dimensions: int = EMBEDDING_DIM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
        sleeper: Sleeper | None = None,
    ) -> None:
        if dimensions < 1:
            raise EmbeddingDimensionError("embedding dimensions must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self._backend = backend
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self._sleeper: Sleeper = sleeper if sleeper is not None else asyncio.sleep

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed `texts` in batches with retries and a hard dimension check."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [_normalize_input(text) for text in texts[start : start + self.batch_size]]
            embedded = await self._embed_batch_with_retry(batch)
            self._assert_dimensions(embedded, expected_count=len(batch))
            vectors.extend(embedded)
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        vectors = await self.embed([query])
        return vectors[0]

    async def _embed_batch_with_retry(self, batch: Sequence[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return await self._backend.embed_batch(batch)
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable_embedding_error(exc):
                    raise
                delay = self.base_delay_seconds * (2**attempt)
                await self._sleeper(delay)
                attempt += 1

    def _assert_dimensions(
        self, vectors: Sequence[Sequence[float]], *, expected_count: int
    ) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingDimensionError(
                f"Embedding backend returned {len(vectors)} vectors, expected {expected_count}."
            )
        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Embedding {index} has dimension {len(vector)}, expected {self.dimensions}."
                )


class HashEmbeddingBackend:
    """Deterministic local embeddings (no network) for tests and offline use."""

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        self.dimensions = dimensions

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text, self.dimensions) for text in texts]


class OpenAIEmbeddingBackend:
    """OpenAI embeddings API (`text-embedding-3-*`)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        *,
        client: AsyncOpenAI | None = None,
        redact_pii: bool = False,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._redact_pii = redact_pii

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [mask_pii(text) for text in texts] if self._redact_pii else list(texts)
        kwargs: dict[str, object] = {"model": self._model, "input": inputs}
        if self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions
        response = await self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


class GeminiEmbeddingBackend:
    """Google Gemini embeddings via the Generative Language REST API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        redact_pii: bool = False,
    ) -> None:
        self._api_key = api_key
        self._redact_pii = redact_pii
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._base_url = base_url.rstrip("/")
        self._http = http_client

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {
            "requests": [
                {
                    "model": self._model,
                    "content": {"parts": [{"text": mask_pii(text) if self._redact_pii else text}]},
                }
                for text in texts
            ]
        }
        url = f"{self._base_url}/{self._model}:batchEmbedContents"
        client = self._http or httpx.AsyncClient(timeout=30.0)
        owns_client = self._http is None
        try:
            response = await client.post(url, params={"key": self._api_key}, json=payload)
            if response.status_code in _RETRYABLE_STATUS:
                response.raise_for_status()
            response.raise_for_status()
            body = response.json()
        finally:
            if owns_client:
                await client.aclose()
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingDimensionError("Gemini embedding response missing 'embeddings'.")
        values: list[list[float]] = []
        for item in embeddings:
            if not isinstance(item, dict) or not isinstance(item.get("values"), list):
                raise EmbeddingDimensionError("Gemini embedding item missing 'values'.")
            values.append([float(v) for v in item["values"]])
        return values


class OllamaEmbeddingBackend:
    """Local Ollama embeddings (`POST /api/embed`)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http = http_client

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._http or httpx.AsyncClient(timeout=60.0)
        owns_client = self._http is None
        try:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": list(texts)},
            )
            response.raise_for_status()
            body = response.json()
        finally:
            if owns_client:
                await client.aclose()
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingDimensionError("Ollama embedding response missing 'embeddings'.")
        return [[float(v) for v in vector] for vector in embeddings]


def create_embedding_client(
    settings: Settings,
    *,
    backend: EmbeddingBackend | None = None,
    sleeper: Sleeper | None = None,
) -> EmbeddingClient:
    """Build an `EmbeddingClient` for the configured LLM/embedding provider."""
    resolved = backend if backend is not None else _backend_from_settings(settings)
    return EmbeddingClient(
        resolved,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        sleeper=sleeper,
    )


def attach_embeddings(
    chunks: Sequence[EvidenceChunkInput],
    vectors: Sequence[Sequence[float]],
) -> list[EvidenceChunkInput]:
    """Return copies of `chunks` with embeddings attached (index-aligned)."""
    if len(chunks) != len(vectors):
        raise EmbeddingDimensionError(
            f"Cannot attach {len(vectors)} embeddings to {len(chunks)} chunks."
        )
    attached: list[EvidenceChunkInput] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        attached.append(
            EvidenceChunkInput(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                section_label=chunk.section_label,
                embedding=list(vector),
                chunk_id=chunk.chunk_id,
            )
        )
    return attached


def is_retryable_embedding_error(exc: BaseException) -> bool:
    """Return True for transient HTTP/provider failures worth retrying."""
    if isinstance(exc, RateLimitError | APIConnectionError | APITimeoutError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in _RETRYABLE_STATUS:
        return True
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError | httpx.RemoteProtocolError):
        return True
    return bool(
        isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRYABLE_STATUS
    )


def _backend_from_settings(settings: Settings) -> EmbeddingBackend:
    if settings.llm_provider is LLMProvider.GEMINI:
        return GeminiEmbeddingBackend(
            redact_pii=settings.redact_pii,
            api_key=settings.require_llm_api_key(),
            model=settings.embedding_model,
        )
    if settings.llm_provider is LLMProvider.OLLAMA:
        return OllamaEmbeddingBackend(model=settings.embedding_model)
    if settings.embedding_model in {"hash", "local-hash", "deterministic"}:
        return HashEmbeddingBackend(dimensions=settings.embedding_dimensions)
    return OpenAIEmbeddingBackend(
        redact_pii=settings.redact_pii,
        api_key=settings.require_llm_api_key(),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


def _normalize_input(text: str) -> str:
    stripped = text.strip()
    return stripped if stripped else "."


def _hash_vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw: list[int] = []
    counter = 0
    while len(raw) < dimensions:
        block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        raw.extend(block)
        counter += 1
    values = [(byte / 127.5) - 1.0 for byte in raw[:dimensions]]
    norm = math.sqrt(sum(component * component for component in values)) or 1.0
    return [component / norm for component in values]
