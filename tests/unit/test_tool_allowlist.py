"""Unit tests for Sub-part 1.2.2c (tool allowlist registry)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from proofpath.policy import (
    APPROVAL_GATED_TOOL_NAMES,
    FORBIDDEN_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    TOOL_REGISTRY,
    ToolAccess,
    ToolName,
    ToolNotAllowedError,
    is_allowed,
    is_approval_gated,
    is_forbidden,
    is_read_only,
    require_allowed,
)
from proofpath.policy.tool_allowlist import APPROVAL_GATED_TOOLS, READ_ONLY_TOOLS
from proofpath.schemas.tools import TOOL_INPUT_REGISTRY

REPO_SRC = Path(__file__).resolve().parents[2] / "src" / "proofpath"

_FORBIDDEN = ("submit_application", "send_email", "delete_tracker")


@pytest.mark.unit
def test_forbidden_tools_are_absent_from_every_tool_registry() -> None:
    assert frozenset(_FORBIDDEN) == FORBIDDEN_TOOL_NAMES
    for name in _FORBIDDEN:
        assert name not in TOOL_REGISTRY
        assert name not in TOOL_INPUT_REGISTRY
        assert name not in READ_ONLY_TOOL_NAMES
        assert name not in APPROVAL_GATED_TOOL_NAMES
        assert name not in {member.value for member in ToolName}
        assert is_forbidden(name) is True
        assert is_allowed(name) is False
        assert is_read_only(name) is False
        assert is_approval_gated(name) is False
        with pytest.raises(ToolNotAllowedError, match=name):
            require_allowed(name)


@pytest.mark.unit
def test_read_only_and_approval_gated_sets_partition_the_registry() -> None:
    read_only = set(READ_ONLY_TOOL_NAMES)
    gated = set(APPROVAL_GATED_TOOL_NAMES)

    assert read_only.isdisjoint(gated)
    assert read_only | gated == set(TOOL_REGISTRY)
    assert read_only == {
        "search_evidence",
        "get_opportunity",
        "get_application_status",
        "get_action_draft",
    }
    assert gated == {"create_action_draft", "approve_action"}

    for spec in READ_ONLY_TOOLS:
        assert spec.access is ToolAccess.READ_ONLY
        assert spec.is_read_only is True
        assert spec.requires_confirmation is False
        assert is_read_only(spec.tool_name) is True
        assert is_approval_gated(spec.tool_name) is False

    for spec in APPROVAL_GATED_TOOLS:
        assert spec.access is ToolAccess.APPROVAL_GATED
        assert spec.is_approval_gated is True
        assert is_read_only(spec.tool_name) is False

    approve = require_allowed("approve_action")
    assert approve.requires_confirmation is True
    create_draft = require_allowed("create_action_draft")
    assert create_draft.requires_confirmation is False


@pytest.mark.unit
def test_allowlist_is_aligned_with_strict_tool_input_schemas() -> None:
    assert set(TOOL_REGISTRY) == set(TOOL_INPUT_REGISTRY)
    for name, spec in TOOL_REGISTRY.items():
        assert spec.input_schema is TOOL_INPUT_REGISTRY[name]
        assert spec.input_schema.model_config["extra"] == "forbid"
        assert require_allowed(name) is spec


@pytest.mark.unit
def test_unknown_tools_fail_closed() -> None:
    with pytest.raises(ToolNotAllowedError, match="not_a_real_tool"):
        require_allowed("not_a_real_tool")
    assert is_allowed("not_a_real_tool") is False
    assert is_forbidden("not_a_real_tool") is False


@pytest.mark.unit
def test_forbidden_tools_are_not_defined_as_callables_in_src() -> None:
    """Banned mutating tools must not be implemented anywhere under src/proofpath."""
    defined: set[str] = set()
    for path in REPO_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in _FORBIDDEN
            ):
                defined.add(f"{path}:{node.name}")
    assert defined == set()
