"""Names are collected at dispatch and scoped to one node invocation, even on failure."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proofpath.policy.output_guard import OutputContractError
from proofpath.policy.tool_trace import (
    call_analysis_tool,
    collect_tool_calls,
    reject_model_tool_calls,
)

pytestmark = pytest.mark.unit


async def test_concurrent_traces_do_not_mix_or_mutate_previous_checkpoint():
    prior = ["get_opportunity"]
    ready = asyncio.Barrier(2)

    async def run(name):
        with collect_tool_calls(prior) as trace:
            await ready.wait()
            await call_analysis_tool(name, AsyncMock(), "private argument")
            return trace

    first, second = await asyncio.gather(run("search_evidence"), run("get_application_status"))
    assert first == ["get_opportunity", "search_evidence"]
    assert second == ["get_opportunity", "get_application_status"]
    assert prior == ["get_opportunity"]


async def test_denied_dispatch_is_traced_but_operation_never_runs():
    operation = AsyncMock()
    with collect_tool_calls([]) as trace, pytest.raises(OutputContractError, match="not allowed"):
        await call_analysis_tool("delete_tracker", operation, token="private")
    operation.assert_not_called()
    assert trace == ["delete_tracker"]


def test_structured_provider_requests_are_never_executed_or_copied_as_text():
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name="send_email", arguments="private")),
            SimpleNamespace(custom=SimpleNamespace(name="unknown private name")),
        ],
        function_call=SimpleNamespace(name="approve_action", arguments="private"),
    )
    with collect_tool_calls([]) as trace, pytest.raises(OutputContractError):
        reject_model_tool_calls(message)
    assert trace == ["send_email", "<unknown>", "approve_action"]
