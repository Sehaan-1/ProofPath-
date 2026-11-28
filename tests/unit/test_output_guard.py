"""Final output is checked against independent ledger and run facts, never prose."""

from importlib.util import find_spec
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from proofpath.config import Settings
from proofpath.policy.output_guard import (
    OutputContractError,
    OutputFacts,
    OutputGuardUnavailableError,
    optional_output_guard,
    validate_final_output,
)
from proofpath.policy.tool_allowlist import FORBIDDEN_TOOL_NAMES
from proofpath.schemas.analysis import ClaimVerdictSchema

pytestmark = pytest.mark.unit


def _claim(verdict, *ids):
    return ClaimVerdictSchema(
        requirement_id=uuid4(),
        verdict=verdict,
        cited_chunk_ids=list(ids),
        claim_summary="Documented how to disable send_email. This is not a tool call.",
    )


async def test_batches_unique_ids_and_only_passes_facts_to_extra_guard():
    a, b = uuid4(), uuid4()
    ledger = Mock(get_existing_chunk_ids=AsyncMock(return_value={a, b}))
    extra = Mock()
    await validate_final_output(
        [_claim("supported", a, b), _claim("partial", a), _claim("unsupported")],
        ["search_evidence", "create_action_draft"],
        lookup=ledger.get_existing_chunk_ids,
        extra_guard=extra,
    )
    ledger.get_existing_chunk_ids.assert_awaited_once_with([a, b])
    extra.validate.assert_called_once_with(
        OutputFacts(((a, b), (a,)), frozenset({a, b}), ("search_evidence", "create_action_draft"))
    )


@pytest.mark.parametrize("verdict", ["supported", "partial"])
async def test_no_citations_or_mixed_real_and_fabricated_citations_fail_closed(verdict):
    real, fabricated = uuid4(), uuid4()
    ledger = Mock(get_existing_chunk_ids=AsyncMock(return_value={real}))
    extra = Mock()
    with pytest.raises(OutputContractError, match="no citation"):
        await validate_final_output(
            [_claim(verdict)], [], lookup=ledger.get_existing_chunk_ids, extra_guard=extra
        )
    ledger.get_existing_chunk_ids.assert_not_awaited()
    with pytest.raises(OutputContractError, match="missing from the Evidence Ledger"):
        await validate_final_output(
            [_claim(verdict, real, fabricated)],
            [],
            lookup=ledger.get_existing_chunk_ids,
            extra_guard=extra,
        )
    extra.validate.assert_not_called()  # An adapter can veto, never override mandatory policy.


@pytest.mark.parametrize(
    "name", [*sorted(FORBIDDEN_TOOL_NAMES), "approve_action", "other_tool", {}]
)
async def test_rejects_disallowed_trace_even_if_verdicts_need_no_citations(name):
    with pytest.raises(OutputContractError, match="disallowed tool"):
        await validate_final_output(
            [_claim("unsupported")], ["search_evidence", name], lookup=AsyncMock()
        )


@pytest.mark.parametrize("missing_or_malformed", [None, "search_evidence", {}])
async def test_requires_a_trace_not_just_an_absence_of_forbidden_text(missing_or_malformed):
    with pytest.raises(OutputContractError, match="trace is unavailable"):
        await validate_final_output([], missing_or_malformed, lookup=AsyncMock())


async def test_lookup_and_validator_failures_are_sanitized():
    ledger = Mock(get_existing_chunk_ids=AsyncMock(side_effect=RuntimeError("private URI")))
    with pytest.raises(OutputGuardUnavailableError, match="Ledger unavailable") as err:
        await validate_final_output(
            [_claim("supported", uuid4())], [], lookup=ledger.get_existing_chunk_ids
        )
    assert "private" not in str(err.value)
    extra = Mock()
    extra.validate.side_effect = ValueError("secret validator diagnostics")
    with pytest.raises(OutputContractError, match="additional output validator") as err:
        await validate_final_output(
            [_claim("unsupported")], [], lookup=ledger.get_existing_chunk_ids, extra_guard=extra
        )
    assert "secret" not in str(err.value)


async def test_guardrails_opt_in_is_lazy_and_never_falls_back_if_missing(monkeypatch):
    load = Mock(side_effect=ModuleNotFoundError("guardrails"))
    monkeypatch.setattr("proofpath.policy.guardrails_adapter.import_module", load)
    assert optional_output_guard(Settings(_env_file=None, enable_guardrails=False)) is None
    load.assert_not_called()
    extra = optional_output_guard(Settings(_env_file=None, enable_guardrails=True))
    with pytest.raises(OutputGuardUnavailableError, match=r"proofpath\[guardrails\]"):
        await validate_final_output([], [], lookup=AsyncMock(), extra_guard=extra)
    load.assert_called_once_with("guardrails.schema.validator")


@pytest.mark.skipif(
    find_spec("guardrails") is None, reason="Install .[guardrails] to test the real SDK"
)
def test_guardrails_sdk_validates_real_payload_not_just_presence_of_extra():
    from guardrails.schema.validator import SchemaValidationError

    guard = optional_output_guard(Settings(_env_file=None, enable_guardrails=True))
    cid = uuid4()
    known = frozenset({cid})
    guard.validate(OutputFacts(((cid,),), known, ("search_evidence",)))
    guard.validate(OutputFacts((), frozenset(), ()))
    for bad in (
        OutputFacts(((),), known, ()),
        OutputFacts(((uuid4(),),), known, ()),
        OutputFacts(((cid,),), known, ("send_email",)),
    ):
        with pytest.raises(SchemaValidationError):
            guard.validate(bad)
