"""Deterministic Evidence Ledger validator.

Sub-part 1.2.2a: every `cited_chunk_id` returned by the LLM must exist in
`evidence_chunks` (the user's ledger). Unknown or out-of-scope IDs are stripped
from the verdict so later policy (1.2.2b) can downgrade unsupported claims.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from proofpath.db.models import AuditActor
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.schemas.analysis import ClaimVerdictSchema

CITATION_MISSING_EVENT: Final[str] = "policy.citation_missing"


class ChunkIdLookup(Protocol):
    """Minimal async lookup used by `LedgerValidator` (satisfied by `EvidenceRepo`)."""

    async def get_existing_chunk_ids(self, chunk_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]: ...


@dataclass(frozen=True, slots=True)
class CitationViolation:
    """One requirement whose LLM citations were not fully present in the ledger."""

    requirement_id: uuid.UUID
    claimed_chunk_ids: tuple[uuid.UUID, ...]
    valid_chunk_ids: tuple[uuid.UUID, ...]
    unknown_chunk_ids: tuple[uuid.UUID, ...]
    out_of_scope_chunk_ids: tuple[uuid.UUID, ...]

    def as_audit_details(self) -> dict[str, object]:
        """JSON-serializable payload for `audit_logs.details_json`."""
        return {
            "requirement_id": str(self.requirement_id),
            "claimed_chunk_ids": [str(item) for item in self.claimed_chunk_ids],
            "valid_chunk_ids": [str(item) for item in self.valid_chunk_ids],
            "unknown_chunk_ids": [str(item) for item in self.unknown_chunk_ids],
            "out_of_scope_chunk_ids": [str(item) for item in self.out_of_scope_chunk_ids],
        }


@dataclass(frozen=True, slots=True)
class CitationCheckResult:
    """Outcome of Sub-part 1.2.2a (citation existence check)."""

    verdicts: tuple[ClaimVerdictSchema, ...]
    violations: tuple[CitationViolation, ...]
    unknown_chunk_ids: frozenset[uuid.UUID]
    out_of_scope_chunk_ids: frozenset[uuid.UUID]
    valid_chunk_ids: frozenset[uuid.UUID]

    @property
    def ok(self) -> bool:
        """True when every cited chunk ID exists in the ledger (and allowed set)."""
        return not self.violations

    @property
    def missing_chunk_ids(self) -> frozenset[uuid.UUID]:
        """Union of unknown (not in ledger) and out-of-scope citation IDs."""
        return self.unknown_chunk_ids | self.out_of_scope_chunk_ids


def check_citation_existence(
    verdicts: Sequence[ClaimVerdictSchema],
    existing_chunk_ids: Collection[uuid.UUID],
    *,
    allowed_chunk_ids: Collection[uuid.UUID] | None = None,
) -> CitationCheckResult:
    """Pure citation existence check against a known set of ledger chunk IDs.

    Invalid IDs are dropped from each returned verdict's `cited_chunk_ids`.
    Verdicts themselves are not downgraded here (that is Sub-part 1.2.2b).
    Empty citation lists are not violations; they are handled by 1.2.2b.
    """
    sanitized: list[ClaimVerdictSchema] = []
    violations: list[CitationViolation] = []
    all_unknown: set[uuid.UUID] = set()
    all_out_of_scope: set[uuid.UUID] = set()
    all_valid: set[uuid.UUID] = set()

    for verdict in verdicts:
        valid: list[uuid.UUID] = []
        unknown: list[uuid.UUID] = []
        out_of_scope: list[uuid.UUID] = []
        for chunk_id in verdict.cited_chunk_ids:
            if chunk_id not in existing_chunk_ids:
                unknown.append(chunk_id)
            elif allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                out_of_scope.append(chunk_id)
            else:
                valid.append(chunk_id)

        all_unknown.update(unknown)
        all_out_of_scope.update(out_of_scope)
        all_valid.update(valid)

        sanitized.append(verdict.model_copy(update={"cited_chunk_ids": list(valid)}))
        if unknown or out_of_scope:
            violations.append(
                CitationViolation(
                    requirement_id=verdict.requirement_id,
                    claimed_chunk_ids=tuple(verdict.cited_chunk_ids),
                    valid_chunk_ids=tuple(valid),
                    unknown_chunk_ids=tuple(unknown),
                    out_of_scope_chunk_ids=tuple(out_of_scope),
                )
            )

    return CitationCheckResult(
        verdicts=tuple(sanitized),
        violations=tuple(violations),
        unknown_chunk_ids=frozenset(all_unknown),
        out_of_scope_chunk_ids=frozenset(all_out_of_scope),
        valid_chunk_ids=frozenset(all_valid),
    )


class LedgerValidator:
    """Async Evidence Ledger policy checks backed by `EvidenceRepo`."""

    def __init__(
        self,
        evidence_repo: EvidenceRepo | ChunkIdLookup,
        audit_repo: AuditRepo | None = None,
    ) -> None:
        self._evidence_repo = evidence_repo
        self._audit_repo = audit_repo

    async def check_citation_existence(
        self,
        verdicts: Sequence[ClaimVerdictSchema],
        *,
        allowed_chunk_ids: Collection[uuid.UUID] | None = None,
        analysis_run_id: uuid.UUID | None = None,
    ) -> CitationCheckResult:
        """Load existing chunk IDs from the ledger and verify LLM citations.

        Performs a single batched `get_existing_chunk_ids` round-trip.
        When `audit_repo` was provided, each violating requirement is recorded
        as an immutable `policy.citation_missing` audit event.
        """
        claimed = _unique_cited_ids(verdicts)
        existing = await self._evidence_repo.get_existing_chunk_ids(claimed)
        result = check_citation_existence(
            verdicts,
            existing,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        if self._audit_repo is not None and result.violations:
            await self._record_violations(result, analysis_run_id=analysis_run_id)
        return result

    async def _record_violations(
        self,
        result: CitationCheckResult,
        *,
        analysis_run_id: uuid.UUID | None,
    ) -> None:
        audit_repo = self._audit_repo
        if audit_repo is None:  # pragma: no cover - guarded by caller
            return
        for violation in result.violations:
            details = violation.as_audit_details()
            if analysis_run_id is not None:
                details["analysis_run_id"] = str(analysis_run_id)
            await audit_repo.record(
                actor=AuditActor.AGENT,
                event_type=CITATION_MISSING_EVENT,
                resource_id=violation.requirement_id,
                details_json=details,
            )


def _unique_cited_ids(verdicts: Sequence[ClaimVerdictSchema]) -> list[uuid.UUID]:
    seen: set[uuid.UUID] = set()
    ordered: list[uuid.UUID] = []
    for verdict in verdicts:
        for chunk_id in verdict.cited_chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(chunk_id)
    return ordered
