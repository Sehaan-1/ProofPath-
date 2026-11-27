"""Shared state contract for the checkpointed opportunity analysis graph.

Nodes return partial updates; fields use LangGraph's default replacement semantics.
In particular, a node updating retrieved_evidence must return the complete mapping,
not just one entry. Evidence keys are canonical string representations of
requirement UUIDs (UUID map keys are not preserved by the checkpoint serializer).
Use `initial_state` for a fresh run, not for a resume.
"""

from __future__ import annotations

import uuid
from typing import NotRequired, TypedDict

from pydantic import Field

from proofpath.config import Settings, get_settings
from proofpath.policy.ledger_validator import DowngradeReason
from proofpath.schemas.actions import ActionDraftRecord, ContributionSprintPlan
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.base import StrictModel
from proofpath.schemas.evidence import EvidenceSnippet
from proofpath.schemas.opportunity import RequirementRecord


class PolicyViolation(StrictModel):
    """Citation or downgrade feedback for a requirement."""

    requirement_id: uuid.UUID
    message: str
    unknown_chunk_ids: list[uuid.UUID] = Field(default_factory=list)
    out_of_scope_chunk_ids: list[uuid.UUID] = Field(default_factory=list)
    downgrade_reason: DowngradeReason | None = None


class GraphError(TypedDict):
    """A failure suitable for recording in state and eventually in the audit log."""

    node: str
    code: str
    message: str
    retryable: bool


class ProofPathState(TypedDict):
    """Analysis state. Keys other than the opportunity may be absent in partial updates."""

    opportunity_id: uuid.UUID
    requirements: NotRequired[list[RequirementRecord]]
    retrieved_evidence: NotRequired[dict[str, list[EvidenceSnippet]]]
    verdicts: NotRequired[list[ClaimVerdictSchema]]
    policy_violations: NotRequired[list[PolicyViolation]]
    policy_retry: NotRequired[bool]
    policy_verified: NotRequired[bool]
    contribution_sprint: NotRequired[ContributionSprintPlan | None]
    draft_actions: NotRequired[list[ActionDraftRecord]]
    tracker_stage: NotRequired[str | None]
    step_budget: NotRequired[int]
    step_count: NotRequired[int]
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    errors: NotRequired[list[GraphError]]
    pending_error: NotRequired[GraphError | None]


def initial_state(opportunity_id: uuid.UUID, *, settings: Settings | None = None) -> ProofPathState:
    """Create independent initial collections and capture the configured retry budget."""
    active_settings = settings if settings is not None else get_settings()
    return {
        "opportunity_id": opportunity_id,
        "requirements": [],
        "retrieved_evidence": {},
        "verdicts": [],
        "policy_violations": [],
        "policy_retry": False,
        "policy_verified": False,
        "contribution_sprint": None,
        "draft_actions": [],
        "step_count": 0,
        "retry_count": 0,
        "max_retries": active_settings.max_agent_retries,
        "errors": [],
        "pending_error": None,
    }
