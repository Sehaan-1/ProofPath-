"""Transactional PostgreSQL + pgvector integration tests for ProofPath repositories."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db import (
    EMBEDDING_DIM,
    ActionDraft,
    ActionDraftStatus,
    ActionType,
    AnalysisRepo,
    AnalysisRunStatus,
    ApplicationStage,
    AuditActor,
    AuditRepo,
    ClaimVerdict,
    ClaimVerdictInput,
    DraftRepo,
    EvidenceChunk,
    EvidenceChunkInput,
    EvidenceRepo,
    EvidenceSourceType,
    OpportunityRepo,
    RequirementCategory,
    RequirementPriority,
    RoleRequirementInput,
    TrackerRepo,
    VerdictType,
    verify_pgvector_extension,
)

pytestmark = pytest.mark.integration


def _basis_vector(hot_index: int) -> list[float]:
    """Create a unit vector of length `EMBEDDING_DIM` with 1.0 at `hot_index`."""
    vec = [0.0] * EMBEDDING_DIM
    vec[hot_index] = 1.0
    return vec


async def test_pgvector_extension_and_indexes_exist(db_session: AsyncSession) -> None:
    assert await verify_pgvector_extension(db_session) is True

    result = await db_session.execute(
        text(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'evidence_chunks'
            """
        )
    )
    indexes = {row.indexname: row.indexdef for row in result}

    assert "ix_evidence_chunks_embedding_hnsw" in indexes
    assert "USING hnsw" in indexes["ix_evidence_chunks_embedding_hnsw"]
    assert "vector_cosine_ops" in indexes["ix_evidence_chunks_embedding_hnsw"]

    assert "ix_evidence_chunks_tsv_gin" in indexes
    assert "USING gin" in indexes["ix_evidence_chunks_tsv_gin"]


async def test_vector_insertion_cosine_similarity_and_generated_tsv(
    db_session: AsyncSession,
) -> None:
    repo = EvidenceRepo(db_session)
    digest = hashlib.sha256(f"resume-vector-{uuid.uuid4()}".encode()).hexdigest()

    source = await repo.create_source(
        source_type=EvidenceSourceType.RESUME,
        title="Backend & AI Systems Resume",
        canonical_uri="file:///local/resume.md",
        content_sha256=digest,
        metadata_json={"pages": 1},
    )

    vec_pgvector = _basis_vector(0)
    vec_kubernetes = _basis_vector(1)

    chunks = await repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Projects",
                content="Implemented hybrid pgvector cosine search with FastAPI and SQLAlchemy.",
                token_count=12,
                embedding=vec_pgvector,
            ),
            EvidenceChunkInput(
                chunk_index=1,
                section_label="Infrastructure",
                content="Deployed Kubernetes clusters with Helm charts and Terraform.",
                token_count=10,
                embedding=vec_kubernetes,
            ),
        ],
    )
    assert len(chunks) == 2

    # 1. Verify pgvector cosine distance ordering returns the matching chunk first.
    cosine_expr = EvidenceChunk.embedding.cosine_distance(vec_pgvector)
    nearest_result = await db_session.execute(
        select(EvidenceChunk.id, cosine_expr.label("distance"))
        .where(EvidenceChunk.source_id == source.id)
        .order_by(cosine_expr.asc())
    )
    ranked = nearest_result.all()
    assert ranked[0].id == chunks[0].id
    assert ranked[0].distance == pytest.approx(0.0, abs=1e-6)
    assert ranked[1].id == chunks[1].id
    assert ranked[1].distance == pytest.approx(1.0, abs=1e-6)

    # 2. Verify stored generated `tsv` column supports full-text matching.
    fts_result = await db_session.execute(
        select(EvidenceChunk.id).where(
            EvidenceChunk.source_id == source.id,
            EvidenceChunk.tsv.op("@@")(func.plainto_tsquery("english", "pgvector")),
        )
    )
    assert list(fts_result.scalars().all()) == [chunks[0].id]

    # 3. Verify citation-existence lookup filters out nonexistent chunk UUIDs.
    fabricated_id = uuid.uuid4()
    existing_ids = await repo.get_existing_chunk_ids([chunks[0].id, fabricated_id])
    assert existing_ids == {chunks[0].id}


async def test_sha256_deduplication_and_unique_constraint(db_session: AsyncSession) -> None:
    repo = EvidenceRepo(db_session)
    digest = hashlib.sha256(f"dedup-source-{uuid.uuid4()}".encode()).hexdigest()

    first, created_first = await repo.get_or_create_source(
        source_type=EvidenceSourceType.GITHUB_REPO,
        title="proofpath-repo",
        canonical_uri="https://github.com/example/proofpath#readme",
        content_sha256=digest,
    )
    second, created_second = await repo.get_or_create_source(
        source_type=EvidenceSourceType.GITHUB_REPO,
        title="proofpath-repo-duplicate",
        canonical_uri="https://github.com/example/proofpath#readme",
        content_sha256=digest,
    )

    assert created_first is True
    assert created_second is False
    assert second.id == first.id

    # Direct insert with identical SHA-256 must trigger PostgreSQL unique constraint.
    async with db_session.begin_nested() as savepoint:
        with pytest.raises(IntegrityError):
            await repo.create_source(
                source_type=EvidenceSourceType.README,
                title="duplicate-raw-insert",
                canonical_uri="file:///local/README.md",
                content_sha256=digest,
            )
        await savepoint.rollback()

    # Outer transaction stays healthy and original record remains intact.
    fetched = await repo.get_source_by_sha256(digest)
    assert fetched is not None
    assert fetched.id == first.id


async def test_evidence_source_delete_cascades_to_chunks(db_session: AsyncSession) -> None:
    repo = EvidenceRepo(db_session)
    digest = hashlib.sha256(f"cascade-source-{uuid.uuid4()}".encode()).hexdigest()

    source = await repo.create_source(
        source_type=EvidenceSourceType.EXPERIENCE_NOTE,
        title="MCP Server Notes",
        canonical_uri="file:///notes/mcp.md",
        content_sha256=digest,
    )
    created_chunks = await repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Overview",
                content="Built read-only FastMCP tools backed by PostgreSQL.",
                token_count=9,
                embedding=_basis_vector(2),
            ),
            EvidenceChunkInput(
                chunk_index=1,
                section_label="Testing",
                content="Verified JSON schemas via automated MCP client tests.",
                token_count=9,
                embedding=_basis_vector(3),
            ),
        ],
    )
    chunk_ids = [chunk.id for chunk in created_chunks]
    assert await repo.get_existing_chunk_ids(chunk_ids) == set(chunk_ids)

    deleted = await repo.delete_source(source.id)
    assert deleted is True
    assert await repo.get_source_by_id(source.id) is None
    assert await repo.list_chunks_for_source(source.id) == []
    assert await repo.get_existing_chunk_ids(chunk_ids) == set()


async def test_opportunity_graph_upserts_db_invariants_and_cascades(
    db_session: AsyncSession,
) -> None:
    evidence_repo = EvidenceRepo(db_session)
    opp_repo = OpportunityRepo(db_session)
    analysis_repo = AnalysisRepo(db_session)
    draft_repo = DraftRepo(db_session)
    tracker_repo = TrackerRepo(db_session)
    audit_repo = AuditRepo(db_session)

    # Seed one verified evidence chunk to cite.
    source = await evidence_repo.create_source(
        source_type=EvidenceSourceType.RESUME,
        title="Verified Resume",
        canonical_uri="file:///resumes/verified.md",
        content_sha256=hashlib.sha256(f"verified-{uuid.uuid4()}".encode()).hexdigest(),
    )
    [chunk] = await evidence_repo.add_chunks(
        source.id,
        [
            EvidenceChunkInput(
                chunk_index=0,
                section_label="Skills",
                content="Python, FastAPI, LangGraph, PostgreSQL, pgvector",
                token_count=8,
                embedding=_basis_vector(4),
            )
        ],
    )

    # Create opportunity + requirements.
    opportunity = await opp_repo.create_opportunity(
        company="Palo Alto Networks",
        role_title="AI Agent Engineer",
        source_url="https://jobs.example.com/panw-ai",
        raw_text="Python + LangGraph required. SAP S/4HANA nice to have.",
        sanitized_text="Python + LangGraph required. SAP S/4HANA nice to have.",
    )
    reqs = await opp_repo.add_requirements(
        opportunity.id,
        [
            RoleRequirementInput(
                requirement_text="Python and LangGraph agent orchestration",
                category=RequirementCategory.AI_AGENT,
                priority=RequirementPriority.MUST_HAVE,
            ),
            RoleRequirementInput(
                requirement_text="SAP S/4HANA administration",
                category=RequirementCategory.DOMAIN,
                priority=RequirementPriority.NICE_TO_HAVE,
            ),
        ],
    )

    # Create analysis run + update metrics + upsert verdicts twice (idempotent retry).
    run = await analysis_repo.create_run(
        opportunity_id=opportunity.id,
        thread_id=f"thread-{uuid.uuid4()}",
        status=AnalysisRunStatus.RUNNING,
    )
    updated_run = await analysis_repo.update_run_metrics(
        run.id,
        status=AnalysisRunStatus.AWAITING_APPROVAL,
        total_tokens=950,
        estimated_cost_usd=Decimal("0.001425"),
        latency_ms=1120,
        trace_id="trace-int-001",
    )
    assert updated_run is not None
    assert updated_run.status is AnalysisRunStatus.AWAITING_APPROVAL

    verdicts = await analysis_repo.upsert_verdicts(
        run.id,
        [
            ClaimVerdictInput(
                requirement_id=reqs[0].id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="Verified Python and LangGraph skills.",
                cited_chunk_ids=[chunk.id],
            ),
            ClaimVerdictInput(
                requirement_id=reqs[1].id,
                verdict=VerdictType.UNSUPPORTED,
                claim_summary="No SAP S/4HANA evidence in ledger.",
                cited_chunk_ids=[],
                policy_downgraded=True,
                gap_notes="Missing SAP S/4HANA proof.",
            ),
        ],
    )
    assert len(verdicts) == 2

    # Re-upserting same (run, requirement) updates cleanly without duplicates.
    updated_verdicts = await analysis_repo.upsert_verdicts(
        run.id,
        [
            ClaimVerdictInput(
                requirement_id=reqs[0].id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="Updated verified Python & LangGraph summary.",
                cited_chunk_ids=[chunk.id],
            )
        ],
    )
    assert len(updated_verdicts) == 1
    assert len(await analysis_repo.list_verdicts(run.id)) == 2

    # DB-level invariant: raw ORM insert of uncited 'supported' verdict fails CHECK constraint.
    async with db_session.begin_nested() as savepoint:
        extra_run = await analysis_repo.create_run(
            opportunity_id=opportunity.id,
            thread_id=f"thread-bad-{uuid.uuid4()}",
        )
        db_session.add(
            ClaimVerdict(
                analysis_run_id=extra_run.id,
                requirement_id=reqs[0].id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="Uncited hallucination",
                cited_chunk_ids=[],
            )
        )
        with pytest.raises(IntegrityError):
            await db_session.flush()
        await savepoint.rollback()

    # Tracker atomic upsert updates existing row rather than duplicating.
    t1 = await tracker_repo.upsert(
        opportunity_id=opportunity.id,
        stage=ApplicationStage.SAVED,
        notes="Initial save",
    )
    t2 = await tracker_repo.upsert(
        opportunity_id=opportunity.id,
        stage=ApplicationStage.PREPARING,
        notes="Building sprint deliverable",
    )
    assert t2.id == t1.id
    assert t2.stage is ApplicationStage.PREPARING

    # Action draft idempotency + approval-gate execution + DB-level approval check.
    idem_key = f"run-{run.id}:tracker_upsert"
    draft, created_1 = await draft_repo.create_or_get_draft(
        analysis_run_id=run.id,
        action_type=ActionType.TRACKER_UPSERT,
        payload_json={"stage": "ready_to_apply"},
        idempotency_key=idem_key,
    )
    draft_dup, created_2 = await draft_repo.create_or_get_draft(
        analysis_run_id=run.id,
        action_type=ActionType.TRACKER_UPSERT,
        payload_json={"stage": "ready_to_apply"},
        idempotency_key=idem_key,
    )
    assert created_1 is True
    assert created_2 is False
    assert draft_dup.id == draft.id

    async with db_session.begin_nested() as savepoint:
        db_session.add(
            ActionDraft(
                analysis_run_id=run.id,
                action_type=ActionType.CALENDAR_BLOCK,
                payload_json={},
                status=ActionDraftStatus.EXECUTED,
                idempotency_key=f"bad-exec-{uuid.uuid4()}",
                approved_at=None,
                executed_at=datetime.now(UTC),
            )
        )
        with pytest.raises(IntegrityError):
            await db_session.flush()
        await savepoint.rollback()

    approved = await draft_repo.approve(draft.id)
    assert approved is not None
    executed = await draft_repo.mark_executed(draft.id)
    assert executed is not None
    assert executed.status is ActionDraftStatus.EXECUTED

    # Append an immutable audit log entry linked by resource_id.
    audit_entry = await audit_repo.record(
        actor=AuditActor.HUMAN,
        event_type="action_draft_approved_and_executed",
        resource_id=draft.id,
        details_json={"opportunity_id": str(opportunity.id)},
    )

    # Deleting Opportunity cascades across requirements, runs, verdicts, tracker, and drafts...
    assert await opp_repo.delete_opportunity(opportunity.id) is True
    assert await opp_repo.list_requirements(opportunity.id) == []
    assert await analysis_repo.get_run_by_id(run.id) is None
    assert await analysis_repo.list_verdicts(run.id) == []
    assert await tracker_repo.get_by_opportunity_id(opportunity.id) is None
    assert await draft_repo.get_by_id(draft.id) is None

    # ...while immutable audit logs remain preserved.
    surviving_logs = await audit_repo.list_events(resource_id=draft.id)
    assert [entry.id for entry in surviving_logs] == [audit_entry.id]
