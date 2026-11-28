"""Unit tests for LLM token accounting and cost estimation (Sub-part 2.4.2b)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from proofpath.graph.workflow import _retrieval_span_metadata
from proofpath.observability.usage import (
    LLMUsage,
    collect_llm_usage,
    estimate_cost_usd,
    record_llm_usage,
    record_response_usage,
    total_tokens,
)
from proofpath.schemas.evidence import EvidenceSnippet

pytestmark = pytest.mark.unit


def test_llm_usage_total_tokens() -> None:
    assert LLMUsage(model="m", prompt_tokens=10, completion_tokens=4).total_tokens == 14


def test_record_outside_a_collector_is_a_noop() -> None:
    # No active run: recording must not raise or leak state.
    record_llm_usage(model="gpt-4o-mini", prompt_tokens=5, completion_tokens=5)
    with collect_llm_usage(None) as usage:
        assert usage == []


def test_collector_accumulates_and_continues_from_prior() -> None:
    prior = [LLMUsage(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=20)]
    with collect_llm_usage(prior) as usage:
        record_llm_usage(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
        assert len(usage) == 2
        assert total_tokens(usage) == 135
    # The prior checkpoint list is never mutated.
    assert len(prior) == 1
    # The collector is restored afterwards.
    with collect_llm_usage(None) as fresh:
        assert fresh == []


def test_collector_isolates_concurrent_runs() -> None:
    with collect_llm_usage(None) as outer:
        record_llm_usage(model="m", prompt_tokens=1, completion_tokens=0)
        with collect_llm_usage(None) as inner:
            record_llm_usage(model="m", prompt_tokens=2, completion_tokens=0)
            assert total_tokens(inner) == 2
        assert total_tokens(outer) == 1


def test_record_clamps_negative_tokens() -> None:
    with collect_llm_usage(None) as usage:
        record_llm_usage(model="m", prompt_tokens=-5, completion_tokens=3)
        assert usage[0].prompt_tokens == 0
        assert usage[0].completion_tokens == 3


def test_record_response_usage_extracts_openai_usage() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40))
    with collect_llm_usage(None) as usage:
        record_response_usage("gpt-4o-mini", response)
        assert usage == [LLMUsage(model="gpt-4o-mini", prompt_tokens=120, completion_tokens=40)]


@pytest.mark.parametrize("response", [SimpleNamespace(usage=None), SimpleNamespace()])
def test_record_response_usage_tolerates_missing_usage(response: object) -> None:
    with collect_llm_usage(None) as usage:
        record_response_usage("gpt-4o-mini", response)
        assert usage == []


def test_estimate_cost_known_model() -> None:
    usage = [LLMUsage(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)]
    # 1000 * 0.15/1e6 + 500 * 0.60/1e6 = 0.00015 + 0.0003 = 0.00045
    assert estimate_cost_usd(usage) == Decimal("0.000450")


def test_estimate_cost_strips_date_suffix() -> None:
    usage = [LLMUsage(model="gpt-4o-mini-2024-07-18", prompt_tokens=1_000_000, completion_tokens=0)]
    assert estimate_cost_usd(usage) == Decimal("0.150000")


def test_estimate_cost_longest_prefix_wins() -> None:
    # gpt-4o-mini must not be priced as gpt-4o.
    mini = [LLMUsage(model="gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)]
    full = [LLMUsage(model="gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)]
    assert estimate_cost_usd(mini) == Decimal("0.150000")
    assert estimate_cost_usd(full) == Decimal("2.500000")


def test_estimate_cost_unknown_model_is_zero_but_tokens_count() -> None:
    usage = [LLMUsage(model="some-local-model", prompt_tokens=999, completion_tokens=1)]
    assert estimate_cost_usd(usage) == Decimal("0.000000")
    assert total_tokens(usage) == 1000


def test_estimate_cost_empty_and_quantization() -> None:
    assert estimate_cost_usd([]) == Decimal("0.000000")
    usage = [LLMUsage(model="gpt-4o", prompt_tokens=1, completion_tokens=1)]
    cost = estimate_cost_usd(usage)
    assert cost == cost.quantize(Decimal("0.000001"))
    assert cost >= 0


def _snippet(score: float | None) -> EvidenceSnippet:
    return EvidenceSnippet(
        chunk_id="00000000-0000-0000-0000-000000000001",
        source_id="00000000-0000-0000-0000-000000000002",
        canonical_uri="file:///resume.md#skills",
        content="Python",
        score=score,
    )


def test_retrieval_span_metadata_counts_and_scores() -> None:
    evidence = {"req-a": [_snippet(0.016393), _snippet(0.01)], "req-b": [_snippet(None)]}
    meta = _retrieval_span_metadata(evidence)
    assert meta["retrieved_chunk_count"] == 3
    assert meta["requirements_searched"] == 2
    assert meta["chunks_per_requirement"] == {"req-a": 2, "req-b": 1}
    # None scores are omitted; values are rounded for compact span metadata.
    assert meta["similarity_scores"] == [0.016393, 0.01]


def test_retrieval_span_metadata_empty() -> None:
    meta = _retrieval_span_metadata({})
    assert meta["retrieved_chunk_count"] == 0
    assert meta["requirements_searched"] == 0
    assert meta["similarity_scores"] == []
