"""Record analysis tool attempts at dispatch, not by searching prompts or node names.

The graph uses local service operations (not a remote MCP client). A node-local
collector lets the workflow checkpoint names even on failure, without mixing
concurrent runs or retaining arguments. Human/MCP approvals are outside this trace.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import ParamSpec, TypeVar

from proofpath.policy.output_guard import OutputContractError
from proofpath.policy.tool_allowlist import (
    ALL_TOOL_NAMES,
    ANALYSIS_TOOL_NAMES,
    FORBIDDEN_TOOL_NAMES,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")
_node_calls: ContextVar[list[str] | None] = ContextVar("proofpath_node_calls", default=None)
_KNOWN_NAMES = ALL_TOOL_NAMES | FORBIDDEN_TOOL_NAMES


@contextmanager
def collect_tool_calls(prior: list[str] | None) -> Iterator[list[str]]:
    if not isinstance(prior, list):
        raise OutputContractError("Analysis tool trace is unavailable.")
    calls = prior.copy()  # Never mutate a previous checkpoint's list.
    token = _node_calls.set(calls)
    try:
        yield calls
    finally:
        _node_calls.reset(token)


def _record(name: object) -> None:
    calls = _node_calls.get()
    if calls is not None:
        # Unknown provider-supplied strings can themselves contain private data.
        calls.append(str(name) if isinstance(name, str) and name in _KNOWN_NAMES else "<unknown>")


async def call_analysis_tool(
    name: str, operation: Callable[_P, Awaitable[_T]], /, *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    _record(name)
    if name not in ANALYSIS_TOOL_NAMES:
        raise OutputContractError("Tool is not allowed during analysis.")
    return await operation(*args, **kwargs)


def reject_model_tool_calls(message: object) -> None:
    """These providers offer structured output only. Never dispatch requested calls."""
    calls = getattr(message, "tool_calls", None) or ()
    legacy_call = getattr(message, "function_call", None)
    if not calls and legacy_call is None:
        return
    for call in calls:
        target = getattr(call, "function", None) or getattr(call, "custom", None)
        _record(getattr(target, "name", None))
    if legacy_call is not None:
        _record(getattr(legacy_call, "name", None))
    raise OutputContractError("Structured output unexpectedly requested a tool call.")
