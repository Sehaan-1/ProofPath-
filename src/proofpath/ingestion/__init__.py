"""Local and remote evidence ingestion pipelines."""

from proofpath.ingestion.chunking import ParsedChunk, chunk_document
from proofpath.ingestion.github_ingester import (
    GitHubEvidenceClient,
    GitHubEvidenceIngester,
    GitHubIngestError,
    GitHubIngestResult,
    GitHubRepoRef,
    canonical_github_uri,
    snapshot_to_document,
)
from proofpath.ingestion.markdown_ingester import (
    FolderIngestResult,
    IngestFileResult,
    IngestStatus,
    LocalEvidenceIngester,
    iter_local_notes,
    persist_parsed_document,
)
from proofpath.ingestion.resume_parser import (
    EmptyDocumentError,
    ParsedDocument,
    UnsupportedDocumentError,
    infer_source_type,
    parse_resume_file,
    parse_resume_text,
)

__all__ = [
    "EmptyDocumentError",
    "FolderIngestResult",
    "GitHubEvidenceClient",
    "GitHubEvidenceIngester",
    "GitHubIngestError",
    "GitHubIngestResult",
    "GitHubRepoRef",
    "IngestFileResult",
    "IngestStatus",
    "LocalEvidenceIngester",
    "ParsedChunk",
    "ParsedDocument",
    "UnsupportedDocumentError",
    "canonical_github_uri",
    "chunk_document",
    "infer_source_type",
    "iter_local_notes",
    "parse_resume_file",
    "parse_resume_text",
    "persist_parsed_document",
    "snapshot_to_document",
]
