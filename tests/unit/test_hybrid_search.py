"""Unit tests for Sub-part 1.3.3b (hybrid RRF retrieval)."""

from __future__ import annotations

import uuid

import pytest

from proofpath.retrieval.hybrid_search import reciprocal_rank_fusion

_SAP = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_SEMANTIC = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_OTHER = uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_MCP = uuid.UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")


@pytest.mark.unit
def test_rrf_keeps_lexical_acronym_hits_that_rank_poorly_in_vector_search() -> None:
    """Pure cosine ranks a vague semantic neighbor first; FTS ranks SAP first."""
    vector_ranks = [_SEMANTIC, _OTHER, _SAP]
    fts_ranks = [_SAP, _MCP]
    fused = reciprocal_rank_fusion([vector_ranks, fts_ranks], k=60, limit=3)
    ids = [chunk_id for chunk_id, _score in fused]
    assert _SAP in ids
    sap_score = dict(fused)[_SAP]
    semantic_score = dict(fused)[_SEMANTIC]
    # SAP appears in both lists, so it should outrank the vector-only neighbor.
    assert sap_score > semantic_score
    assert ids[0] == _SAP


@pytest.mark.unit
def test_rrf_scores_follow_one_over_k_plus_rank() -> None:
    only = [_MCP]
    fused = reciprocal_rank_fusion([only], k=60, limit=8)
    assert fused == [(_MCP, 1.0 / 61.0)]


@pytest.mark.unit
def test_rrf_limit_and_empty_lists() -> None:
    assert reciprocal_rank_fusion([], k=60, limit=8) == []
    fused = reciprocal_rank_fusion([[_SAP, _MCP, _OTHER]], k=60, limit=2)
    assert [chunk_id for chunk_id, _ in fused] == [_SAP, _MCP]
    with pytest.raises(ValueError, match="RRF k"):
        reciprocal_rank_fusion([[_SAP]], k=0, limit=1)
