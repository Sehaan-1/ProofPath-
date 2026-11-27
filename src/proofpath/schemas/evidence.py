"""Evidence Ledger request/response and retrieval snippet schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Self

from pydantic import AnyHttpUrl, Field, field_validator, model_validator

from proofpath.db.models import EvidenceSourceType
from proofpath.schemas.base import StrictModel

_MAX_TITLE = 512
_MAX_URI = 2048
_MAX_CONTENT = 500_000
_MAX_SECTION = 255
_SHA256_LEN = 64


class EvidenceSnippet(StrictModel):
    """Cited, user-owned evidence chunk returned by hybrid retrieval."""

    chunk_id: uuid.UUID = Field(description="Primary key of the `evidence_chunks` row.")
    source_id: uuid.UUID = Field(description="Parent `evidence_sources` row.")
    canonical_uri: str = Field(
        min_length=1,
        max_length=_MAX_URI,
        description="Provenance URI (file path, GitHub URL, or certificate locator).",
    )
    content: str = Field(
        min_length=1,
        max_length=_MAX_CONTENT,
        description="Exact chunk text that may be cited in a claim verdict.",
    )
    section_label: str | None = Field(
        default=None,
        max_length=_MAX_SECTION,
        description="Optional section heading (Experience, Projects, Skills, Education).",
    )
    source_type: EvidenceSourceType | None = Field(
        default=None,
        description="Provenance category of the parent evidence source.",
    )
    source_title: str | None = Field(
        default=None,
        max_length=_MAX_TITLE,
        description="Human-readable title of the parent evidence source.",
    )
    chunk_index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based index of this chunk within its source.",
    )
    score: float | None = Field(
        default=None,
        description="Retrieval rank score (RRF fusion or similarity); higher is better.",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Approximate token count of `content` when known.",
    )

    @field_validator("section_label", "source_title", mode="before")
    @classmethod
    def _blank_optional_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class IngestTextRequest(StrictModel):
    """API request to ingest a local resume, README, note, or certificate as text."""

    title: str = Field(min_length=1, max_length=_MAX_TITLE)
    content: str = Field(min_length=1, max_length=_MAX_CONTENT)
    canonical_uri: str = Field(
        min_length=1,
        max_length=_MAX_URI,
        description="Stable provenance locator stored on `evidence_sources.canonical_uri`.",
    )
    source_type: EvidenceSourceType = Field(
        default=EvidenceSourceType.EXPERIENCE_NOTE,
        description="Provenance category for the ingested artifact.",
    )


class IngestGitHubRequest(StrictModel):
    """API request to ingest a GitHub repository as evidence (Slice 1.3.2)."""

    repo_url: AnyHttpUrl | None = Field(
        default=None,
        description="Full GitHub repository URL (https://github.com/<owner>/<repo>).",
    )
    owner: str | None = Field(default=None, min_length=1, max_length=255)
    name: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("owner", "name", mode="before")
    @classmethod
    def _blank_optional_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_url_or_owner_name(self) -> Self:
        if self.repo_url is None and (self.owner is None or self.name is None):
            raise ValueError("Provide repo_url or both owner and name.")
        return self


class IngestResponse(StrictModel):
    """API response after ingesting (or SHA-256-deduplicating) an evidence source."""

    source_id: uuid.UUID
    created: bool = Field(description="False when an identical SHA-256 digest already existed.")
    chunk_count: int = Field(ge=0)
    content_sha256: str = Field(min_length=_SHA256_LEN, max_length=_SHA256_LEN)
    source_type: EvidenceSourceType
    canonical_uri: str = Field(min_length=1, max_length=_MAX_URI)


class EvidenceSearchRequest(StrictModel):
    """API request body for hybrid evidence search."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)


class EvidenceSearchResponse(StrictModel):
    """Ranked evidence snippets for a search query."""

    query: str = Field(min_length=1, max_length=2000)
    results: list[EvidenceSnippet] = Field(default_factory=list)


class EvidenceSourceResponse(StrictModel):
    """API response describing a persisted evidence source without chunk bodies."""

    id: uuid.UUID
    source_type: EvidenceSourceType
    title: str
    canonical_uri: str
    content_sha256: str = Field(min_length=_SHA256_LEN, max_length=_SHA256_LEN)
    ingested_at: datetime
    chunk_count: int = Field(ge=0)
