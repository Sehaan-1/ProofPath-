"""Final analysis output contract. Unlike the comparison policy, this never repairs output."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from proofpath.config import Settings
from proofpath.db.models import VerdictType
from proofpath.policy.tool_allowlist import ANALYSIS_TOOL_NAMES
from proofpath.schemas.analysis import ClaimVerdictSchema

OUTPUT_CONTRACT_REJECTED = "output_contract_rejected"


class OutputContractError(ValueError):
    """Output must not be published. Messages contain no model text or tool arguments."""


class OutputGuardUnavailableError(OutputContractError):
    """A required source of validation was unavailable; do not silently skip it."""


@dataclass(frozen=True, slots=True)
class OutputFacts:
    """Only the facts an additional validator needs; no JD, evidence text or credentials."""

    grounded_citations: tuple[tuple[UUID, ...], ...]  # one group per supported/partial verdict
    existing_chunk_ids: frozenset[UUID]
    tool_trace: tuple[str, ...]


class OutputGuard(Protocol):
    """An additive, local validator. Raise on rejection; never rewrite the answer."""

    def validate(self, facts: OutputFacts) -> None: ...


def optional_output_guard(settings: Settings) -> OutputGuard | None:
    if not settings.enable_guardrails:
        return None
    from proofpath.policy.guardrails_adapter import GuardrailsAIOutputGuard

    return GuardrailsAIOutputGuard()


async def validate_final_output(
    verdicts: Sequence[ClaimVerdictSchema],
    tool_trace: Sequence[str] | None,
    *,
    lookup: Callable[[Sequence[UUID]], Awaitable[set[UUID]]],
    extra_guard: OutputGuard | None = None,
) -> None:
    """Check the actual run trace and batch-check citations against the *live* ledger.

    An empty trace is valid; a missing/malformed trace is not evidence of zero calls.
    The model schema intentionally allows empty citations for intermediate retries;
    the final contract does not. A mix of real and fabricated IDs is also invalid.
    """
    if not isinstance(tool_trace, (list, tuple)):
        raise OutputContractError("Analysis tool trace is unavailable.")
    if any(not isinstance(name, str) or name not in ANALYSIS_TOOL_NAMES for name in tool_trace):
        raise OutputContractError("Analysis trace contains a disallowed tool call.")
    grounded = tuple(
        tuple(v.cited_chunk_ids) for v in verdicts if v.verdict != VerdictType.UNSUPPORTED
    )
    if any(not group for group in grounded):
        raise OutputContractError("A supported or partial claim has no citation.")
    cited = list(dict.fromkeys(cid for v in verdicts for cid in v.cited_chunk_ids))
    try:
        existing = await lookup(cited) if cited else set()
    except Exception as exc:
        raise OutputGuardUnavailableError(
            "Evidence Ledger unavailable for final validation."
        ) from exc
    if not existing.issuperset(cited):
        raise OutputContractError("A cited chunk is missing from the Evidence Ledger.")
    if extra_guard is not None:
        facts = OutputFacts(grounded, frozenset(existing), tuple(tool_trace))
        try:
            await asyncio.to_thread(extra_guard.validate, facts)
        except OutputGuardUnavailableError:
            raise
        except Exception as exc:
            raise OutputContractError(
                "The additional output validator rejected the result."
            ) from exc
