"""Unit tests for Sub-part 1.2.2a (citation existence check)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock

import pytest

from proofpath.db.models import AuditActor, VerdictType
from proofpath.db.repositories.audit import AuditRepo
from proofpath.policy import (
    CITATION_MISSING_EVENT,
    CitationCheckResult,
    LedgerValidator,
    check_citation_existence,
)
from proofpath.schemas import ClaimVerdictSchema

_LEDGER_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_LEDGER_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_HALLUCINATED = uuid.UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
_REQ_1 = uuid.UUID("11111111-1111-4111-8111-111111111111")
_REQ_2 = uuid.UUID("22222222-2222-4222-8222-222222222222")
_RUN_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")


def _verdict(
    *,
    requirement_id: uuid.UUID = _REQ_1,
    cited: Sequence[uuid.UUID] = (),
    verdict: VerdictType = VerdictType.SUPPORTED,
) -> ClaimVerdictSchema:
    return ClaimVerdictSchema(
        requirement_id=requirement_id,
        verdict=verdict,
        claim_summary="LLM claim under citation review.",
        cited_chunk_ids=list(cited),
    )


class _FakeChunkLookup:
    def __init__(self, existing: set[uuid.UUID]) -> None:
        self.existing = existing
        self.calls: list[list[uuid.UUID]] = []

    async def get_existing_chunk_ids(self, chunk_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        self.calls.append(list(chunk_ids))
        return {chunk_id for chunk_id in chunk_ids if chunk_id in self.existing}


@pytest.mark.unit
def test_all_cited_ids_in_ledger_pass_without_violations() -> None:
    original = _verdict(cited=[_LEDGER_A, _LEDGER_B])
    result = check_citation_existence([original], {_LEDGER_A, _LEDGER_B})

    assert result.ok is True
    assert result.violations == ()
    assert result.unknown_chunk_ids == frozenset()
    assert result.missing_chunk_ids == frozenset()
    assert result.valid_chunk_ids == frozenset({_LEDGER_A, _LEDGER_B})
    assert result.verdicts[0].cited_chunk_ids == [_LEDGER_A, _LEDGER_B]
    assert result.verdicts[0].requirement_id == original.requirement_id
    # Returned verdicts are copies; callers cannot mutate the input via the result.
    assert result.verdicts[0] is not original


@pytest.mark.unit
def test_unknown_cited_ids_are_stripped_and_reported() -> None:
    result = check_citation_existence(
        [
            _verdict(
                requirement_id=_REQ_1,
                cited=[_LEDGER_A, _HALLUCINATED],
            )
        ],
        {_LEDGER_A, _LEDGER_B},
    )

    assert result.ok is False
    assert result.unknown_chunk_ids == frozenset({_HALLUCINATED})
    assert result.valid_chunk_ids == frozenset({_LEDGER_A})
    assert result.verdicts[0].cited_chunk_ids == [_LEDGER_A]
    assert result.verdicts[0].verdict is VerdictType.SUPPORTED
    assert result.verdicts[0].policy_downgraded is False

    violation = result.violations[0]
    assert violation.requirement_id == _REQ_1
    assert violation.claimed_chunk_ids == (_LEDGER_A, _HALLUCINATED)
    assert violation.valid_chunk_ids == (_LEDGER_A,)
    assert violation.unknown_chunk_ids == (_HALLUCINATED,)
    assert violation.out_of_scope_chunk_ids == ()


@pytest.mark.unit
def test_empty_citations_are_not_existence_violations() -> None:
    """Empty lists are a 1.2.2b concern; 1.2.2a only flags IDs absent from the ledger."""
    result = check_citation_existence(
        [_verdict(cited=[], verdict=VerdictType.SUPPORTED)],
        {_LEDGER_A},
    )
    assert result.ok is True
    assert result.verdicts[0].cited_chunk_ids == []
    assert result.verdicts[0].needs_citation_downgrade() is True


@pytest.mark.unit
def test_all_citations_unknown_strips_every_id() -> None:
    result = check_citation_existence(
        [_verdict(cited=[_HALLUCINATED], verdict=VerdictType.PARTIAL)],
        {_LEDGER_A},
    )
    assert result.ok is False
    assert result.verdicts[0].cited_chunk_ids == []
    assert result.unknown_chunk_ids == frozenset({_HALLUCINATED})
    assert result.valid_chunk_ids == frozenset()


@pytest.mark.unit
def test_allowed_set_rejects_in_ledger_but_out_of_scope_ids() -> None:
    result = check_citation_existence(
        [_verdict(cited=[_LEDGER_A, _LEDGER_B])],
        {_LEDGER_A, _LEDGER_B},
        allowed_chunk_ids={_LEDGER_A},
    )
    assert result.ok is False
    assert result.out_of_scope_chunk_ids == frozenset({_LEDGER_B})
    assert result.unknown_chunk_ids == frozenset()
    assert result.missing_chunk_ids == frozenset({_LEDGER_B})
    assert result.verdicts[0].cited_chunk_ids == [_LEDGER_A]
    assert result.violations[0].out_of_scope_chunk_ids == (_LEDGER_B,)
    assert result.violations[0].unknown_chunk_ids == ()


@pytest.mark.unit
def test_stale_allowed_id_missing_from_ledger_is_unknown_not_out_of_scope() -> None:
    result = check_citation_existence(
        [_verdict(cited=[_HALLUCINATED])],
        {_LEDGER_A},
        allowed_chunk_ids={_HALLUCINATED, _LEDGER_A},
    )
    assert result.unknown_chunk_ids == frozenset({_HALLUCINATED})
    assert result.out_of_scope_chunk_ids == frozenset()
    assert result.verdicts[0].cited_chunk_ids == []


@pytest.mark.unit
def test_multiple_verdicts_are_checked_independently() -> None:
    result = check_citation_existence(
        [
            _verdict(requirement_id=_REQ_1, cited=[_LEDGER_A]),
            _verdict(
                requirement_id=_REQ_2,
                cited=[_HALLUCINATED],
                verdict=VerdictType.UNSUPPORTED,
            ),
        ],
        {_LEDGER_A, _LEDGER_B},
    )
    assert result.ok is False
    assert len(result.violations) == 1
    assert result.violations[0].requirement_id == _REQ_2
    assert result.verdicts[0].cited_chunk_ids == [_LEDGER_A]
    assert result.verdicts[1].cited_chunk_ids == []
    assert result.verdicts[1].verdict is VerdictType.UNSUPPORTED


@pytest.mark.unit
def test_citation_violation_audit_payload_uses_strings() -> None:
    result = check_citation_existence(
        [_verdict(cited=[_LEDGER_A, _HALLUCINATED])],
        {_LEDGER_A},
    )
    payload = result.violations[0].as_audit_details()
    assert payload["requirement_id"] == str(_REQ_1)
    assert payload["unknown_chunk_ids"] == [str(_HALLUCINATED)]
    assert payload["valid_chunk_ids"] == [str(_LEDGER_A)]
    assert all(isinstance(value, (str, list)) for value in payload.values())


@pytest.mark.unit
async def test_ledger_validator_batches_lookup_and_strips_unknown_ids() -> None:
    lookup = _FakeChunkLookup({_LEDGER_A, _LEDGER_B})
    validator = LedgerValidator(lookup)
    result = await validator.check_citation_existence(
        [
            _verdict(requirement_id=_REQ_1, cited=[_LEDGER_A, _HALLUCINATED]),
            _verdict(requirement_id=_REQ_2, cited=[_LEDGER_B, _LEDGER_A]),
        ]
    )

    assert lookup.calls == [[_LEDGER_A, _HALLUCINATED, _LEDGER_B]]
    assert result.ok is False
    assert result.unknown_chunk_ids == frozenset({_HALLUCINATED})
    assert result.verdicts[0].cited_chunk_ids == [_LEDGER_A]
    assert result.verdicts[1].cited_chunk_ids == [_LEDGER_B, _LEDGER_A]


@pytest.mark.unit
async def test_ledger_validator_short_circuits_when_no_citations() -> None:
    lookup = _FakeChunkLookup({_LEDGER_A})
    validator = LedgerValidator(lookup)
    result = await validator.check_citation_existence(
        [_verdict(cited=[], verdict=VerdictType.UNSUPPORTED)]
    )
    assert result.ok is True
    assert lookup.calls == [[]]


@pytest.mark.unit
async def test_ledger_validator_records_audit_events_for_violations() -> None:
    lookup = _FakeChunkLookup({_LEDGER_A})
    audit = AsyncMock(spec=AuditRepo)
    validator = LedgerValidator(lookup, audit_repo=audit)

    result = await validator.check_citation_existence(
        [_verdict(requirement_id=_REQ_1, cited=[_HALLUCINATED])],
        analysis_run_id=_RUN_ID,
    )
    assert result.ok is False
    audit.record.assert_awaited_once()
    kwargs: dict[str, Any] = audit.record.await_args.kwargs
    assert kwargs["actor"] is AuditActor.AGENT
    assert kwargs["event_type"] == CITATION_MISSING_EVENT
    assert kwargs["resource_id"] == _REQ_1
    assert kwargs["details_json"]["unknown_chunk_ids"] == [str(_HALLUCINATED)]
    assert kwargs["details_json"]["analysis_run_id"] == str(_RUN_ID)


@pytest.mark.unit
async def test_ledger_validator_does_not_audit_when_citations_exist() -> None:
    lookup = _FakeChunkLookup({_LEDGER_A})
    audit = AsyncMock(spec=AuditRepo)
    validator = LedgerValidator(lookup, audit_repo=audit)
    result = await validator.check_citation_existence([_verdict(cited=[_LEDGER_A])])
    assert result.ok is True
    audit.record.assert_not_awaited()


@pytest.mark.unit
def test_citation_check_result_is_exported_as_frozen_dataclass() -> None:
    result = check_citation_existence([], set())
    assert isinstance(result, CitationCheckResult)
    with pytest.raises(FrozenInstanceError):
        result.valid_chunk_ids = frozenset()  # type: ignore[misc]
