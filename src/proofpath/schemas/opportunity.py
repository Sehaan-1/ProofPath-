"""Opportunity request/response and LLM structured-output schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Self

from pydantic import AnyHttpUrl, Field, field_validator, model_validator

from proofpath.db.models import RequirementCategory, RequirementPriority
from proofpath.schemas.base import StrictModel

_MAX_REQUIREMENT_TEXT = 4000
_MAX_ENTITY = 255
_MAX_COMPANY = 255
_MAX_ROLE_TITLE = 255
_MAX_JD_TEXT = 200_000


class RequirementItem(StrictModel):
    """Normalized skill/requirement extracted from a job description (LLM output)."""

    requirement_text: str = Field(
        min_length=1,
        max_length=_MAX_REQUIREMENT_TEXT,
        description="Verbatim or lightly normalized requirement sentence from the JD.",
    )
    entity: str = Field(
        min_length=1,
        max_length=_MAX_ENTITY,
        description="Canonical skill, tool, or domain name (e.g. 'FastAPI', 'SAP S/4HANA').",
    )
    category: RequirementCategory = Field(
        description="Normalized requirement category.",
    )
    priority: RequirementPriority = Field(
        description="Whether the JD treats this as must-have or nice-to-have.",
    )


class RequirementRecord(RequirementItem):
    """Persisted `RequirementItem` with ledger identifiers for API responses."""

    id: uuid.UUID = Field(description="Primary key of the `role_requirements` row.")
    opportunity_id: uuid.UUID = Field(
        description="Parent opportunity this requirement belongs to.",
    )


class ParsedRole(StrictModel):
    """LLM structured output for `analyse_role` (Slice 1.4.2)."""

    company: str = Field(
        min_length=1,
        max_length=_MAX_COMPANY,
        description="Hiring company or organization name.",
    )
    role_title: str = Field(
        min_length=1,
        max_length=_MAX_ROLE_TITLE,
        description="Normalized job title.",
    )
    source_url: AnyHttpUrl | None = Field(
        default=None,
        description="Public job posting URL when the JD was fetched rather than pasted.",
    )
    summary: str | None = Field(
        default=None,
        max_length=4000,
        description="One-paragraph role summary. Must not invent facts absent from the JD.",
    )
    requirements: list[RequirementItem] = Field(
        min_length=1,
        max_length=100,
        description="Normalized must-have and nice-to-have requirements.",
    )

    @field_validator("summary", mode="before")
    @classmethod
    def _blank_summary_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _unique_entities(self) -> Self:
        entities = [item.entity.casefold() for item in self.requirements]
        if len(entities) != len(set(entities)):
            raise ValueError("ParsedRole.requirements must not contain duplicate entities.")
        return self

    @property
    def must_have(self) -> list[RequirementItem]:
        """Requirements marked `must_have`."""
        return [
            item for item in self.requirements if item.priority is RequirementPriority.MUST_HAVE
        ]

    @property
    def nice_to_have(self) -> list[RequirementItem]:
        """Requirements marked `nice_to_have`."""
        return [
            item for item in self.requirements if item.priority is RequirementPriority.NICE_TO_HAVE
        ]


class OpportunityCreateRequest(StrictModel):
    """API request to ingest a target opportunity from a URL and/or pasted JD text."""

    raw_text: str | None = Field(
        default=None,
        max_length=_MAX_JD_TEXT,
        description="Pasted job-description text. Required when `source_url` is omitted.",
    )
    source_url: AnyHttpUrl | None = Field(
        default=None,
        description="Public job posting URL to fetch when `raw_text` is omitted.",
    )
    company: str | None = Field(default=None, max_length=_MAX_COMPANY)
    role_title: str | None = Field(default=None, max_length=_MAX_ROLE_TITLE)

    @field_validator("raw_text", "company", "role_title", mode="before")
    @classmethod
    def _blank_optional_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_text_or_url(self) -> Self:
        if self.raw_text is None and self.source_url is None:
            raise ValueError("Provide raw_text or source_url (or both).")
        return self


class OpportunityResponse(StrictModel):
    """API response for a persisted opportunity and its structured requirements."""

    id: uuid.UUID
    company: str
    role_title: str
    source_url: str | None = None
    created_at: datetime
    requirements: list[RequirementRecord] = Field(default_factory=list)
