"""Analysis-run, claim-verdict, and report schemas."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from proofpath.db.models import AnalysisRunStatus, VerdictType
from proofpath.schemas.actions import ActionDraftRecord, ContributionSprintPlan
from proofpath.schemas.base import StrictModel
from proofpath.schemas.opportunity import ParsedRole, RequirementRecord

_MAX_SUMMARY = 4000


class ClaimVerdictSchema(StrictModel):
    """Per-requirement evidence verdict (LLM structured output + report contract).

    Supported/partial verdicts *should* cite chunk IDs, but empty citations are
    accepted here so Slice 1.2.2 can deterministically downgrade them.
    """

    requirement_id: uuid.UUID = Field(
        description="Persisted `role_requirements.id` this verdict evaluates.",
    )
    verdict: VerdictType
    claim_summary: str = Field(
        min_length=1,
        max_length=_MAX_SUMMARY,
        description="Honest one- or two-sentence justification grounded in cited evidence.",
    )
    cited_chunk_ids: list[uuid.UUID] = Field(
        default_factory=list,
        max_length=32,
        description="Evidence Ledger chunk IDs that substantiate the claim.",
    )
    policy_downgraded: bool = Field(
        default=False,
        description="True when the ledger validator rewrote this verdict to unsupported.",
    )
    gap_notes: str | None = Field(
        default=None,
        max_length=_MAX_SUMMARY,
        description="What is missing when the verdict is partial or unsupported.",
    )
    requirement_text: str | None = Field(
        default=None,
        max_length=_MAX_SUMMARY,
        description="Optional copy of the requirement text for display in reports.",
    )

    @field_validator("gap_notes", "requirement_text", mode="before")
    @classmethod
    def _blank_optional_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cited_chunk_ids")
    @classmethod
    def _dedupe_cited_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        seen: set[uuid.UUID] = set()
        unique: list[uuid.UUID] = []
        for chunk_id in value:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk_id)
        return unique

    def needs_citation_downgrade(self) -> bool:
        """Return True when a supported/partial verdict cites no chunk IDs."""
        return self.verdict is not VerdictType.UNSUPPORTED and not self.cited_chunk_ids


class AnalysisRunCreateRequest(StrictModel):
    """API request to start a LangGraph opportunity analysis run."""

    opportunity_id: uuid.UUID
    async_run: bool = Field(
        default=False,
        description="When true, enqueue the run and return 202 Accepted (Slice 2.2).",
    )

    calendar_start: AwareDatetime | None = None
    calendar_end: AwareDatetime | None = None

    @model_validator(mode="after")
    def _validate_calendar(self) -> Self:
        if (self.calendar_start is None) != (self.calendar_end is None):
            raise ValueError("Provide both calendar_start and calendar_end.")
        if (
            self.calendar_start is not None
            and self.calendar_end is not None
            and self.calendar_end <= self.calendar_start
        ):
            raise ValueError("calendar_end must be after calendar_start.")
        return self


class AnalysisError(StrictModel):
    """Sanitized failure reported by a graph node."""

    node: str
    code: str
    message: str
    retryable: bool


class AnalysisReport(StrictModel):
    """End-to-end analysis report returned by `GET /api/v1/analyses/{run_id}`."""

    analysis_run_id: uuid.UUID
    opportunity_id: uuid.UUID
    thread_id: str = Field(
        min_length=1,
        max_length=128,
        description="LangGraph checkpoint thread id for resume/approval.",
    )
    status: AnalysisRunStatus
    role: ParsedRole | None = None
    requirements: list[RequirementRecord] = Field(default_factory=list)
    verdicts: list[ClaimVerdictSchema] = Field(default_factory=list)
    contribution_sprint: ContributionSprintPlan | None = None
    action_drafts: list[ActionDraftRecord] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: Decimal = Field(default=Decimal("0"), ge=0, max_digits=12, decimal_places=6)
    latency_ms: int = Field(default=0, ge=0)
    trace_id: str | None = Field(default=None, max_length=128)
    errors: list[AnalysisError] = Field(default_factory=list)

    @field_validator("trace_id", mode="before")
    @classmethod
    def _blank_trace_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def supported(self) -> list[ClaimVerdictSchema]:
        """Verdicts marked `supported`."""
        return [item for item in self.verdicts if item.verdict is VerdictType.SUPPORTED]

    @property
    def partial(self) -> list[ClaimVerdictSchema]:
        """Verdicts marked `partial`."""
        return [item for item in self.verdicts if item.verdict is VerdictType.PARTIAL]

    @property
    def unsupported(self) -> list[ClaimVerdictSchema]:
        """Verdicts marked `unsupported`."""
        return [item for item in self.verdicts if item.verdict is VerdictType.UNSUPPORTED]

    @property
    def policy_downgraded_count(self) -> int:
        """Number of verdicts rewritten by the Evidence Ledger policy."""
        return sum(1 for item in self.verdicts if item.policy_downgraded)
