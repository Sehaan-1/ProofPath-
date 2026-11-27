"""Live PostgreSQL tests for Evidence Ledger citation existence checks."""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import AuditActor, EvidenceSourceType, VerdictType
from proofpath.db.repositories.audit import AuditRepo
from proofpath.db.repositories.evidence import EvidenceChunkInput, EvidenceRepo
from proofpath.policy import (
    CITATION_MISSING_EVENT,
    CLAIM_DOWNGRADED_EVENT,
    DowngradeReason,
    LedgerValidator,
)
from proofpath.schemas import ClaimVerdictSchema

pytestmark = pytest.mark.integration


async def test_citation_existence_against_live_evidence_chunks(
    db_session: AsyncSession,
) -> None:
    evidence_repo = EvidenceRepo(db_session)
    audit_repo = AuditRepo(db_session)
    digest = hashlib.sha256(f"ledger-validator-{uuid.uuid4()}".encode()).hexdigest()

    source = await evidence_repo.create_source(
        source_type=EvidenceSourceType.RESUME,
        title="Citation existence resume",
        canonical_uri="file:///local/citation-check.md",
        content_sha256=digest,
    )
    chunks = await evidence_repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Projects",
                content="Shipped a FastAPI + pgvector evidence ledger with MCP tools.",
                token_count=14,
            )
        ],
    )
    real_chunk_id = chunks[0].id
    hallucinated_id = uuid.uuid4()
    requirement_id = uuid.uuid4()
    analysis_run_id = uuid.uuid4()

    validator = LedgerValidator(evidence_repo, audit_repo=audit_repo)
    result = await validator.check_citation_existence(
        [
            ClaimVerdictSchema(
                requirement_id=requirement_id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="FastAPI production experience is cited.",
                cited_chunk_ids=[real_chunk_id, hallucinated_id],
            )
        ],
        analysis_run_id=analysis_run_id,
    )

    assert result.ok is False
    assert result.unknown_chunk_ids == frozenset({hallucinated_id})
    assert result.valid_chunk_ids == frozenset({real_chunk_id})
    assert result.verdicts[0].cited_chunk_ids == [real_chunk_id]
    assert result.verdicts[0].verdict is VerdictType.SUPPORTED
    assert result.verdicts[0].policy_downgraded is False

    events = await audit_repo.list_events(
        actor=AuditActor.AGENT,
        event_type=CITATION_MISSING_EVENT,
        resource_id=requirement_id,
    )
    assert len(events) == 1
    assert events[0].details_json["unknown_chunk_ids"] == [str(hallucinated_id)]
    assert events[0].details_json["valid_chunk_ids"] == [str(real_chunk_id)]
    assert events[0].details_json["analysis_run_id"] == str(analysis_run_id)

    clean = await validator.check_citation_existence(
        [
            ClaimVerdictSchema(
                requirement_id=uuid.uuid4(),
                verdict=VerdictType.SUPPORTED,
                claim_summary="Only the real ledger chunk is cited.",
                cited_chunk_ids=[real_chunk_id],
            )
        ]
    )
    assert clean.ok is True
    assert clean.verdicts[0].cited_chunk_ids == [real_chunk_id]


async def test_enforce_downgrades_unsupported_entity_against_live_chunks(
    db_session: AsyncSession,
) -> None:
    evidence_repo = EvidenceRepo(db_session)
    audit_repo = AuditRepo(db_session)
    digest = hashlib.sha256(f"ledger-downgrade-{uuid.uuid4()}".encode()).hexdigest()

    source = await evidence_repo.create_source(
        source_type=EvidenceSourceType.RESUME,
        title="Downgrade rule resume",
        canonical_uri="file:///local/downgrade-check.md",
        content_sha256=digest,
    )
    chunks = await evidence_repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Projects",
                content="Shipped a FastAPI + pgvector evidence ledger with MCP tools.",
                token_count=14,
            )
        ],
    )
    real_chunk_id = chunks[0].id
    sap_requirement_id = uuid.uuid4()
    fastapi_requirement_id = uuid.uuid4()
    empty_requirement_id = uuid.uuid4()
    analysis_run_id = uuid.uuid4()

    validator = LedgerValidator(evidence_repo, audit_repo=audit_repo)
    result = await validator.enforce(
        [
            ClaimVerdictSchema(
                requirement_id=sap_requirement_id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="Hallucinated SAP experience.",
                cited_chunk_ids=[real_chunk_id],
            ),
            ClaimVerdictSchema(
                requirement_id=fastapi_requirement_id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="FastAPI is cited in the projects section.",
                cited_chunk_ids=[real_chunk_id],
            ),
            ClaimVerdictSchema(
                requirement_id=empty_requirement_id,
                verdict=VerdictType.PARTIAL,
                claim_summary="Claimed Helm experience with no citations.",
            ),
        ],
        entities_by_requirement={
            sap_requirement_id: "SAP S/4HANA",
            fastapi_requirement_id: "FastAPI",
            empty_requirement_id: "Helm",
        },
        analysis_run_id=analysis_run_id,
    )

    by_requirement = {item.requirement_id: item for item in result.verdicts}

    sap = by_requirement[sap_requirement_id]
    assert sap.verdict is VerdictType.UNSUPPORTED
    assert sap.policy_downgraded is True
    assert sap.cited_chunk_ids == [real_chunk_id]

    fastapi = by_requirement[fastapi_requirement_id]
    assert fastapi.verdict is VerdictType.SUPPORTED
    assert fastapi.policy_downgraded is False

    empty = by_requirement[empty_requirement_id]
    assert empty.verdict is VerdictType.UNSUPPORTED
    assert empty.policy_downgraded is True
    assert empty.cited_chunk_ids == []

    assert result.policy_downgraded_count == 2
    reasons = {event.requirement_id: event.reason for event in result.downgrades}
    assert reasons[sap_requirement_id] is DowngradeReason.ENTITY_NOT_IN_EVIDENCE
    assert reasons[empty_requirement_id] is DowngradeReason.EMPTY_CITATIONS

    sap_events = await audit_repo.list_events(
        event_type=CLAIM_DOWNGRADED_EVENT,
        resource_id=sap_requirement_id,
    )
    assert len(sap_events) == 1
    assert sap_events[0].actor is AuditActor.AGENT
    assert sap_events[0].details_json["entity"] == "SAP S/4HANA"
    assert sap_events[0].details_json["analysis_run_id"] == str(analysis_run_id)
