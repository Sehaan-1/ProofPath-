"""Embeddings and hybrid Evidence Ledger retrieval."""

from proofpath.retrieval.embeddings import (
    EmbeddingBackend,
    EmbeddingClient,
    EmbeddingDimensionError,
    GeminiEmbeddingBackend,
    HashEmbeddingBackend,
    OllamaEmbeddingBackend,
    OpenAIEmbeddingBackend,
    attach_embeddings,
    create_embedding_client,
    is_retryable_embedding_error,
)
from proofpath.retrieval.hybrid_search import (
    HybridSearcher,
    hybrid_search,
    reciprocal_rank_fusion,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingClient",
    "EmbeddingDimensionError",
    "GeminiEmbeddingBackend",
    "HashEmbeddingBackend",
    "HybridSearcher",
    "OllamaEmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "attach_embeddings",
    "create_embedding_client",
    "hybrid_search",
    "is_retryable_embedding_error",
    "reciprocal_rank_fusion",
]
