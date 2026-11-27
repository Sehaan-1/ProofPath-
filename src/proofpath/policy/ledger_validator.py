"""Deterministic Evidence Ledger validator.

Sub-part 1.2.2a: every `cited_chunk_id` returned by the LLM must exist in
`evidence_chunks` (the user's ledger). Unknown or out-of-scope IDs are stripped
from the verdict so later policy (1.2.2b) can downgrade unsupported claims.

Sub-part 1.2.2b: if a `supported`/`partial` claim has zero valid citations, or
if none of the cited chunk texts mention the required entity/tool, rewrite the
verdict to `unsupported`, set `policy_downgraded=True`, and log a policy event.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

from proofpath.db.models import AuditActor, VerdictType
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.schemas.analysis import ClaimVerdictSchema

CITATION_MISSING_EVENT: Final[str] = "policy.citation_missing"
CLAIM_DOWNGRADED_EVENT: Final[str] = "policy.claim_downgraded"

_EMPTY_CITATION_NOTE: Final[str] = (
    "Policy: supported/partial claim had zero valid evidence citations."
)
_ENTITY_NOT_IN_EVIDENCE_NOTE: Final[str] = (
    "Policy: cited evidence does not mention required entity {entity!r}."
)

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MAX_GAP_NOTES = 4000


class ChunkIdLookup(Protocol):
    """Minimal async lookup used by citation existence checks."""

    async def get_existing_chunk_ids(self, chunk_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]: ...


class ChunkContent(Protocol):
    """Minimal chunk body used by the entity-support check."""

    id: uuid.UUID
    content: str


class EvidenceLedgerLookup(Protocol):
    """Ledger access required to enforce 1.2.2a + 1.2.2b together."""

    async def get_existing_chunk_ids(self, chunk_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]: ...

    async def get_chunks_by_ids(self, chunk_ids: Sequence[uuid.UUID]) -> Sequence[ChunkContent]: ...


class DowngradeReason(StrEnum):
    """Why a supported/partial claim was rewritten to unsupported."""

    EMPTY_CITATIONS = "empty_citations"
    ENTITY_NOT_IN_EVIDENCE = "entity_not_in_evidence"


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


@dataclass(frozen=True, slots=True)
class DowngradeEvent:
    """One requirement whose verdict was deterministically rewritten to unsupported."""

    requirement_id: uuid.UUID
    original_verdict: VerdictType
    reason: DowngradeReason
    entity: str | None
    cited_chunk_ids: tuple[uuid.UUID, ...]

    def as_audit_details(self) -> dict[str, object]:
        """JSON-serializable payload for `audit_logs.details_json`."""
        return {
            "requirement_id": str(self.requirement_id),
            "original_verdict": self.original_verdict.value,
            "rewritten_verdict": VerdictType.UNSUPPORTED.value,
            "reason": self.reason.value,
            "entity": self.entity,
            "cited_chunk_ids": [str(item) for item in self.cited_chunk_ids],
        }


@dataclass(frozen=True, slots=True)
class DowngradeResult:
    """Outcome of Sub-part 1.2.2b (unsupported-claim downgrade)."""

    verdicts: tuple[ClaimVerdictSchema, ...]
    downgrades: tuple[DowngradeEvent, ...]

    @property
    def ok(self) -> bool:
        """True when no supported/partial claim required a policy rewrite."""
        return not self.downgrades


@dataclass(frozen=True, slots=True)
class LedgerPolicyResult:
    """Combined 1.2.2a + 1.2.2b enforcement result for the policy verifier node."""

    verdicts: tuple[ClaimVerdictSchema, ...]
    citation_check: CitationCheckResult
    downgrades: tuple[DowngradeEvent, ...]

    @property
    def ok(self) -> bool:
        """True when citations exist in the ledger and no claim was downgraded."""
        return self.citation_check.ok and not self.downgrades

    @property
    def policy_downgraded_count(self) -> int:
        """Number of claims rewritten to `unsupported` by 1.2.2b."""
        return len(self.downgrades)


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


def chunk_supports_entity(chunk_text: str, entity: str) -> bool:
    """Return True when `chunk_text` deterministically mentions `entity`.

    Matching is conservative and LLM-free: case-insensitive phrase/token
    presence, plus a compact alphanumeric form for punctuated names
    (e.g. ``SAP S/4HANA`` vs ``SAPS4HANA``). Short tokens use word boundaries
    so ``Go`` does not match ``going``.
    """
    entity = entity.strip()
    if not entity:
        return False

    folded_chunk = _fold(chunk_text)
    folded_entity = _fold(entity)
    if not folded_entity or not folded_chunk:
        return False

    if _contains_phrase(folded_chunk, folded_entity):
        return True

    if _has_significant_punctuation(entity):
        compact_entity = _compact(entity)
        compact_chunk = _compact(chunk_text)
        if len(compact_entity) >= 4 and compact_entity in compact_chunk:
            return True
    return False


def apply_unsupported_downgrade(
    verdicts: Sequence[ClaimVerdictSchema],
    chunk_texts: Mapping[uuid.UUID, str],
    *,
    entities_by_requirement: Mapping[uuid.UUID, str] | None = None,
) -> DowngradeResult:
    """Rewrite supported/partial claims that lack grounding to `unsupported`.

    A claim is downgraded when:
    - it is `supported` or `partial` and `cited_chunk_ids` is empty, or
    - an entity is supplied for the requirement and no cited chunk text
      mentions that entity/tool.
    Already-`unsupported` claims are left unchanged.
    """
    entities = entities_by_requirement or {}
    rewritten: list[ClaimVerdictSchema] = []
    events: list[DowngradeEvent] = []

    for verdict in verdicts:
        reason = _downgrade_reason(verdict, chunk_texts, entities)
        if reason is None:
            rewritten.append(verdict)
            continue

        entity = entities.get(verdict.requirement_id)
        note = (
            _EMPTY_CITATION_NOTE
            if reason is DowngradeReason.EMPTY_CITATIONS
            else _ENTITY_NOT_IN_EVIDENCE_NOTE.format(entity=entity)
        )
        rewritten.append(
            verdict.model_copy(
                update={
                    "verdict": VerdictType.UNSUPPORTED,
                    "policy_downgraded": True,
                    "gap_notes": _with_gap_note(verdict.gap_notes, note),
                }
            )
        )
        events.append(
            DowngradeEvent(
                requirement_id=verdict.requirement_id,
                original_verdict=verdict.verdict,
                reason=reason,
                entity=entity,
                cited_chunk_ids=tuple(verdict.cited_chunk_ids),
            )
        )

    return DowngradeResult(verdicts=tuple(rewritten), downgrades=tuple(events))


class LedgerValidator:
    """Async Evidence Ledger policy checks backed by `EvidenceRepo`."""

    def __init__(
        self,
        evidence_repo: EvidenceRepo | EvidenceLedgerLookup | ChunkIdLookup,
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
            await self._record_citation_violations(result, analysis_run_id=analysis_run_id)
        return result

    async def apply_unsupported_downgrade(
        self,
        verdicts: Sequence[ClaimVerdictSchema],
        *,
        entities_by_requirement: Mapping[uuid.UUID, str] | None = None,
        analysis_run_id: uuid.UUID | None = None,
    ) -> DowngradeResult:
        """Load cited chunk text from the ledger and apply the 1.2.2b downgrade rule."""
        chunk_texts = await self._load_chunk_texts(_unique_cited_ids(verdicts))
        result = apply_unsupported_downgrade(
            verdicts,
            chunk_texts,
            entities_by_requirement=entities_by_requirement,
        )
        if self._audit_repo is not None and result.downgrades:
            await self._record_downgrades(result, analysis_run_id=analysis_run_id)
        return result

    async def enforce(
        self,
        verdicts: Sequence[ClaimVerdictSchema],
        *,
        entities_by_requirement: Mapping[uuid.UUID, str] | None = None,
        allowed_chunk_ids: Collection[uuid.UUID] | None = None,
        analysis_run_id: uuid.UUID | None = None,
    ) -> LedgerPolicyResult:
        """Run 1.2.2a then 1.2.2b against the Evidence Ledger."""
        citation_check = await self.check_citation_existence(
            verdicts,
            allowed_chunk_ids=allowed_chunk_ids,
            analysis_run_id=analysis_run_id,
        )
        downgrade = await self.apply_unsupported_downgrade(
            citation_check.verdicts,
            entities_by_requirement=entities_by_requirement,
            analysis_run_id=analysis_run_id,
        )
        return LedgerPolicyResult(
            verdicts=downgrade.verdicts,
            citation_check=citation_check,
            downgrades=downgrade.downgrades,
        )

    async def _load_chunk_texts(self, chunk_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, str]:
        loader = getattr(self._evidence_repo, "get_chunks_by_ids", None)
        if loader is None:
            raise TypeError(
                "Evidence repo must implement get_chunks_by_ids for unsupported-claim downgrades."
            )
        chunks = await loader(chunk_ids)
        texts: dict[uuid.UUID, str] = {}
        for chunk in chunks:
            section = getattr(chunk, "section_label", None)
            content = chunk.content
            texts[chunk.id] = f"{section}\n{content}" if section else content
        return texts

    async def _record_citation_violations(
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

    async def _record_downgrades(
        self,
        result: DowngradeResult,
        *,
        analysis_run_id: uuid.UUID | None,
    ) -> None:
        audit_repo = self._audit_repo
        if audit_repo is None:  # pragma: no cover - guarded by caller
            return
        for event in result.downgrades:
            details = event.as_audit_details()
            if analysis_run_id is not None:
                details["analysis_run_id"] = str(analysis_run_id)
            await audit_repo.record(
                actor=AuditActor.AGENT,
                event_type=CLAIM_DOWNGRADED_EVENT,
                resource_id=event.requirement_id,
                details_json=details,
            )


def _downgrade_reason(
    verdict: ClaimVerdictSchema,
    chunk_texts: Mapping[uuid.UUID, str],
    entities: Mapping[uuid.UUID, str],
) -> DowngradeReason | None:
    if verdict.verdict is VerdictType.UNSUPPORTED:
        return None
    if not verdict.cited_chunk_ids:
        return DowngradeReason.EMPTY_CITATIONS

    entity = entities.get(verdict.requirement_id)
    if entity is None or not entity.strip():
        return None

    for chunk_id in verdict.cited_chunk_ids:
        text = chunk_texts.get(chunk_id)
        if text is not None and chunk_supports_entity(text, entity):
            return None
    return DowngradeReason.ENTITY_NOT_IN_EVIDENCE


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("/", " ").replace("-", " ").replace("_", " ")
    return _WHITESPACE.sub(" ", normalized).strip()


def _compact(text: str) -> str:
    return _NON_ALNUM.sub("", _fold(text))


def _has_significant_punctuation(entity: str) -> bool:
    return any(char in entity for char in "/.+#")


def _contains_phrase(haystack: str, needle: str) -> bool:
    if " " in needle:
        return needle in haystack
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _with_gap_note(existing: str | None, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    combined = f"{existing.rstrip()} {note}"
    return combined[:_MAX_GAP_NOTES]


def _unique_cited_ids(verdicts: Sequence[ClaimVerdictSchema]) -> list[uuid.UUID]:
    seen: set[uuid.UUID] = set()
    ordered: list[uuid.UUID] = []
    for verdict in verdicts:
        for chunk_id in verdict.cited_chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(chunk_id)
    return ordered
