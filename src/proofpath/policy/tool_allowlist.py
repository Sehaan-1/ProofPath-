"""Explicit MCP / LangGraph tool allowlist.

Sub-part 1.2.2c: only registered read-only and approval-gated write tools may
be invoked. Forbidden mutating tools (`submit_application`, `send_email`,
`delete_tracker`) are not in the registry and fail closed if requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from proofpath.schemas.base import StrictModel
from proofpath.schemas.tools import TOOL_INPUT_REGISTRY

# The exact phrase a human (or their client) must pass to `approve_action`.
# Confirmation is deliberately explicit: the tool refuses to execute anything
# without it, even when the draft is a perfectly valid `pending_approval` row.
APPROVAL_CONFIRMATION_PHRASE: Final[str] = "approve action"


class ToolAccess(StrEnum):
    """How a registered tool is allowed to run."""

    READ_ONLY = "read_only"
    APPROVAL_GATED = "approval_gated"


class ToolName(StrEnum):
    """Allowlisted tool names. Forbidden tools are intentionally absent."""

    SEARCH_EVIDENCE = "search_evidence"
    GET_OPPORTUNITY = "get_opportunity"
    GET_APPLICATION_STATUS = "get_application_status"
    GET_ACTION_DRAFT = "get_action_draft"
    CREATE_ACTION_DRAFT = "create_action_draft"
    APPROVE_ACTION = "approve_action"


class ToolNotAllowedError(PermissionError):
    """Raised when a tool name is unknown or explicitly forbidden."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name!r} is not on the ProofPath allowlist.")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One allowlisted tool and the gate it must pass before execution."""

    name: ToolName
    access: ToolAccess
    description: str
    input_schema: type[StrictModel]
    requires_confirmation: bool = False

    @property
    def tool_name(self) -> str:
        """Wire-level tool name used by MCP / LangGraph."""
        return self.name.value

    @property
    def is_read_only(self) -> bool:
        return self.access is ToolAccess.READ_ONLY

    @property
    def is_approval_gated(self) -> bool:
        return self.access is ToolAccess.APPROVAL_GATED


def _spec(
    name: ToolName,
    access: ToolAccess,
    description: str,
    *,
    requires_confirmation: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        access=access,
        description=description,
        input_schema=TOOL_INPUT_REGISTRY[name.value],
        requires_confirmation=requires_confirmation,
    )


# Names that must never be registered or implemented as callable tools.
FORBIDDEN_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "submit_application",
        "send_email",
        "delete_tracker",
    }
)

READ_ONLY_TOOLS: Final[tuple[ToolSpec, ...]] = (
    _spec(
        ToolName.SEARCH_EVIDENCE,
        ToolAccess.READ_ONLY,
        "Hybrid search over the user's Evidence Ledger.",
    ),
    _spec(
        ToolName.GET_OPPORTUNITY,
        ToolAccess.READ_ONLY,
        "Fetch a persisted opportunity and its structured requirements.",
    ),
    _spec(
        ToolName.GET_APPLICATION_STATUS,
        ToolAccess.READ_ONLY,
        "Read the personal tracker stage for an opportunity.",
    ),
    _spec(
        ToolName.GET_ACTION_DRAFT,
        ToolAccess.READ_ONLY,
        "Read a pending or historical action draft without executing it.",
    ),
)

APPROVAL_GATED_TOOLS: Final[tuple[ToolSpec, ...]] = (
    _spec(
        ToolName.CREATE_ACTION_DRAFT,
        ToolAccess.APPROVAL_GATED,
        "Persist a pending_approval action draft; never executes tracker or calendar writes.",
    ),
    _spec(
        ToolName.APPROVE_ACTION,
        ToolAccess.APPROVAL_GATED,
        "Execute one pending_approval action draft (tracker/calendar side effect plus "
        f"audit record) only after the explicit confirmation phrase {APPROVAL_CONFIRMATION_PHRASE!r}.",
        requires_confirmation=True,
    ),
)

TOOL_REGISTRY: Final[dict[str, ToolSpec]] = {
    spec.tool_name: spec for spec in (*READ_ONLY_TOOLS, *APPROVAL_GATED_TOOLS)
}

READ_ONLY_TOOL_NAMES: Final[frozenset[str]] = frozenset(spec.tool_name for spec in READ_ONLY_TOOLS)
APPROVAL_GATED_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    spec.tool_name for spec in APPROVAL_GATED_TOOLS
)
# Every tool the MCP server must expose, and nothing else.
ALL_TOOL_NAMES: Final[frozenset[str]] = frozenset(TOOL_REGISTRY)


def is_allowed(tool_name: str) -> bool:
    """Return True when `tool_name` is on the allowlist."""
    return tool_name in TOOL_REGISTRY


def is_read_only(tool_name: str) -> bool:
    """Return True when `tool_name` is a registered read-only tool."""
    return tool_name in READ_ONLY_TOOL_NAMES


def is_approval_gated(tool_name: str) -> bool:
    """Return True when `tool_name` is a registered approval-gated write tool."""
    return tool_name in APPROVAL_GATED_TOOL_NAMES


def is_forbidden(tool_name: str) -> bool:
    """Return True when `tool_name` is explicitly banned from the registry."""
    return tool_name in FORBIDDEN_TOOL_NAMES


def require_allowed(tool_name: str) -> ToolSpec:
    """Return the spec for an allowlisted tool, or fail closed."""
    if is_forbidden(tool_name) or tool_name not in TOOL_REGISTRY:
        raise ToolNotAllowedError(tool_name)
    return TOOL_REGISTRY[tool_name]
