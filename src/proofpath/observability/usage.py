"""LLM token-usage capture and cost estimation for analysis runs (Sub-part 2.4.2b).

The model backends call OpenAI structured output directly (not via a LangChain
chat model), so token usage is recorded explicitly at each call site into a
node-local collector — the same pattern as :mod:`proofpath.policy.tool_trace`.
The workflow checkpoints the accumulated usage on ``ProofPathState.llm_usage``
so totals survive the approval pause and are queryable via the API even outside
the trace UI.

This module has **no SDK dependency**: it is pure accounting. Cost is an
*estimate* from a small public-pricing table (USD per 1M tokens); unknown models
estimate zero cost while still counting tokens. Nothing here ever raises into a
run — missing usage simply records nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import ROUND_HALF_UP, Decimal

from pydantic import Field

from proofpath.schemas.base import StrictModel

_COST_QUANTUM = Decimal("0.000001")
_TOKENS_PER_MTOK = Decimal("1000000")


class LLMUsage(StrictModel):
    """One model call's token usage (checkpoint-safe, non-sensitive)."""

    model: str = Field(min_length=1, max_length=128)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_llm_usage: ContextVar[list[LLMUsage] | None] = ContextVar("proofpath_llm_usage", default=None)


@contextmanager
def collect_llm_usage(prior: Sequence[LLMUsage] | None) -> Iterator[list[LLMUsage]]:
    """Accumulate usage for one node, continuing from the checkpointed total.

    Mirrors ``collect_tool_calls``: copy the prior list so a previous
    checkpoint is never mutated, install it for the node, then restore.
    """
    usage = list(prior) if prior is not None else []
    token = _llm_usage.set(usage)
    try:
        yield usage
    finally:
        _llm_usage.reset(token)


def record_llm_usage(*, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record one call's usage into the active collector (no-op outside a run)."""
    sink = _llm_usage.get()
    if sink is None:
        return
    sink.append(
        LLMUsage(
            model=model,
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
        )
    )


def record_response_usage(model: str, response: object) -> None:
    """Extract ``usage`` from an OpenAI completion response and record it.

    Tolerates a missing/None ``usage`` (e.g. a provider that omits it) by
    recording nothing rather than failing the run.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_llm_usage(
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def total_tokens(usage: Sequence[LLMUsage]) -> int:
    """Sum prompt + completion tokens across all recorded calls."""
    return sum(item.total_tokens for item in usage)


# ---------------------------------------------------------------------------
# Cost estimation (USD per 1M tokens). Public list prices, treated as estimates.
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4.1": (Decimal("2.00"), Decimal("8.00")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("1.60")),
    "gpt-4.1-nano": (Decimal("0.10"), Decimal("0.40")),
    "o3-mini": (Decimal("1.10"), Decimal("4.40")),
    "o4-mini": (Decimal("1.10"), Decimal("4.40")),
}

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$")


def _rates(model: str) -> tuple[Decimal, Decimal]:
    """Resolve (input, output) USD-per-1M rates; unknown models estimate zero."""
    name = model.lower().strip()
    if name in _PRICING:
        return _PRICING[name]
    base = _DATE_SUFFIX.sub("", name)
    if base in _PRICING:
        return _PRICING[base]
    # Longest known prefix wins (so gpt-4o-mini beats gpt-4o).
    for key in sorted(_PRICING, key=len, reverse=True):
        if name.startswith(key):
            return _PRICING[key]
    return (Decimal("0"), Decimal("0"))


def estimate_cost_usd(usage: Sequence[LLMUsage]) -> Decimal:
    """Estimate total USD cost, quantized to the ``Numeric(12, 6)`` column."""
    total = Decimal("0")
    for item in usage:
        input_rate, output_rate = _rates(item.model)
        total += (
            Decimal(item.prompt_tokens) * input_rate + Decimal(item.completion_tokens) * output_rate
        ) / _TOKENS_PER_MTOK
    return total.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)


__all__ = [
    "LLMUsage",
    "collect_llm_usage",
    "estimate_cost_usd",
    "record_llm_usage",
    "record_response_usage",
    "total_tokens",
]
