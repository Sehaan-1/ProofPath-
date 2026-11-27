-- Executed automatically by the official PostgreSQL entrypoint on first boot.
-- Enables pgvector for semantic embedding storage and HNSW cosine search.
CREATE EXTENSION IF NOT EXISTS vector;
