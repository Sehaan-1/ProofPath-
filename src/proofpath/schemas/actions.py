"""Action-draft, tracker, and contribution-sprint contracts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import AwareDatetime, Field, TypeAdapter, field_validator, model_validator

from proofpath.db.models import ActionDraftStatus, ActionType, ApplicationStage
from proofpath.schemas.base import StrictModel

_MAX_NOTES = 8000
_MAX_TITLE = 255
_MAX_DESCRIPTION = 4000
_MAX_IDEMPOTENCY_KEY = 128


class ContributionSprintPlan(StrictModel):
    """LLM structured output for `recommend_contribution_sprint` (Slice 1.5.2f)."""

    title: str = Field(
        min_length=1,
        max_length=_MAX_TITLE,
        description="Short name of the proposed sprint.",
    )
    scope: str = Field(
        min_length=1,
        max_length=_MAX_DESCRIPTION,
        description="Realistic, small-scope description of what will be built.",
    )
    repo_or_topic: str = Field(
        min_length=1,
        max_length=512,
        description="Target open-source repo, personal project, or topic area.",
    )
    deliverables: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Concrete artifacts to complete (PR, README, demo, tests).",
    )
    estimate_days: int = Field(
        ge=2,
        le=5,
        description="Calendar-day estimate; contribution sprints are bounded to 2-5 days.",
    )
    target_gap_requirement_ids: list[uuid.UUID] = Field(
        min_length=1,
        max_length=20,
        description="Unsupported or partial requirement IDs this sprint is meant to address.",
    )
    rationale: str = Field(
        min_length=1,
        max_length=_MAX_DESCRIPTION,
        description="Why this sprint is the highest-impact next step for the gap.",
    )

    @field_validator("deliverables")
    @classmethod
    def _non_empty_deliverables(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("deliverables must not contain blank items.")
        return cleaned

    @field_validator("target_gap_requirement_ids")
    @classmethod
    def _unique_gap_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(value) != len(set(value)):
            raise ValueError("target_gap_requirement_ids must be unique.")
        return value


class TrackerUpsertPayload(StrictModel):
    """Payload for an approval-gated `tracker_upsert` action draft."""

    action_type: Literal[ActionType.TRACKER_UPSERT] = Field(
        description="Discriminator; must be 'tracker_upsert'.",
    )
    opportunity_id: uuid.UUID
    stage: ApplicationStage
    notes: str | None = Field(default=None, max_length=_MAX_NOTES)

    @field_validator("notes", mode="before")
    @classmethod
    def _blank_notes_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class CalendarBlockPayload(StrictModel):
    """Payload for an approval-gated `calendar_block` action draft (.ics / local task)."""

    action_type: Literal[ActionType.CALENDAR_BLOCK] = Field(
        description="Discriminator; must be 'calendar_block'.",
    )
    title: str = Field(min_length=1, max_length=_MAX_TITLE)
    start_at: AwareDatetime
    end_at: AwareDatetime
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION)
    timezone: str = Field(
        default="UTC",
        min_length=1,
        max_length=64,
        description="IANA timezone name used when materializing the calendar block.",
    )

    @field_validator("description", mode="before")
    @classmethod
    def _blank_description_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be strictly after start_at.")
        return self


class ContributionSprintPayload(StrictModel):
    """Payload for an approval-gated `contribution_sprint` action draft."""

    action_type: Literal[ActionType.CONTRIBUTION_SPRINT] = Field(
        description="Discriminator; must be 'contribution_sprint'.",
    )
    plan: ContributionSprintPlan


ActionDraftPayload: TypeAlias = Annotated[
    TrackerUpsertPayload | CalendarBlockPayload | ContributionSprintPayload,
    Field(discriminator="action_type"),
]

ACTION_DRAFT_PAYLOAD_ADAPTER: TypeAdapter[ActionDraftPayload] = TypeAdapter(ActionDraftPayload)


def parse_action_draft_payload(data: object) -> ActionDraftPayload:
    """Validate a JSON/dict payload as a discriminated `ActionDraftPayload`."""
    return ACTION_DRAFT_PAYLOAD_ADAPTER.validate_python(data)


class ActionDraftRecord(StrictModel):
    """API/MCP response for a persisted, never-auto-executed action draft."""

    id: uuid.UUID
    analysis_run_id: uuid.UUID
    action_type: ActionType
    status: ActionDraftStatus
    payload: ActionDraftPayload
    idempotency_key: str = Field(min_length=1, max_length=_MAX_IDEMPOTENCY_KEY)
    approved_at: AwareDatetime | None = None
    executed_at: AwareDatetime | None = None


class TrackerRecord(StrictModel):
    """API response for a personal application-tracker entry."""

    id: uuid.UUID
    opportunity_id: uuid.UUID
    stage: ApplicationStage
    notes: str | None = None
    updated_at: datetime


class ApplicationStatusResponse(StrictModel):
    """API/MCP response for the personal application-tracker status of an opportunity."""

    opportunity_id: uuid.UUID
    tracked: bool = Field(
        description="True when an `application_tracker` entry exists for this opportunity.",
    )
    tracker_id: uuid.UUID | None = None
    stage: ApplicationStage | None = None
    notes: str | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_tracked_fields(self) -> Self:
        if self.tracked:
            if self.tracker_id is None or self.stage is None or self.updated_at is None:
                raise ValueError(
                    "tracked=True requires tracker_id, stage, and updated_at to be set."
                )
        elif any(
            value is not None
            for value in (self.tracker_id, self.stage, self.notes, self.updated_at)
        ):
            raise ValueError(
                "tracked=False requires tracker_id, stage, notes, and updated_at to be None."
            )
        return self
