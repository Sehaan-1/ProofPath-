"""MCP / LangGraph tool-input schemas.

Sub-part 1.2.1b: every model inherits `StrictModel` (`extra="forbid"`) so
unexpected LLM-hallucinated arguments fail fast at validation time.
"""

from __future__ import annotations

import uuid
from typing import Final

from pydantic import Field

from proofpath.schemas.actions import ActionDraftPayload
from proofpath.schemas.base import StrictModel

_MAX_QUERY = 2000
_MAX_CONFIRMATION = 128
_MAX_IDEMPOTENCY_KEY = 128


class SearchEvidenceInput(StrictModel):
    """Input for the read-only `search_evidence` tool."""

    query: str = Field(min_length=1, max_length=_MAX_QUERY)
    top_k: int = Field(default=8, ge=1, le=50)


class GetOpportunityInput(StrictModel):
    """Input for the read-only `get_opportunity` tool."""

    opportunity_id: uuid.UUID


class GetApplicationStatusInput(StrictModel):
    """Input for the read-only `get_application_status` tool."""

    opportunity_id: uuid.UUID


class GetActionDraftInput(StrictModel):
    """Input for the read-only `get_action_draft` tool."""

    draft_id: uuid.UUID


class CreateActionDraftInput(StrictModel):
    """Input for the side-effect-free `create_action_draft` tool.

    Persists a `pending_approval` row only; never executes tracker or calendar writes.
    """

    analysis_run_id: uuid.UUID
    payload: ActionDraftPayload
    idempotency_key: str = Field(min_length=1, max_length=_MAX_IDEMPOTENCY_KEY)


class ApproveActionInput(StrictModel):
    """Input for the approval-gated `approve_action` tool (Slice 2.1.4)."""

    draft_id: uuid.UUID
    confirmation_phrase: str = Field(min_length=1, max_length=_MAX_CONFIRMATION)


# Explicit registry so MCP wiring and unit tests can assert extra="forbid"
# on every tool-input contract in one place.
TOOL_INPUT_REGISTRY: Final[dict[str, type[StrictModel]]] = {
    "search_evidence": SearchEvidenceInput,
    "get_opportunity": GetOpportunityInput,
    "get_application_status": GetApplicationStatusInput,
    "get_action_draft": GetActionDraftInput,
    "create_action_draft": CreateActionDraftInput,
    "approve_action": ApproveActionInput,
}
