"""Typed async repository for analysis runs and evidence-backed claim verdicts."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from proofpath.db.models import AnalysisRun, AnalysisRunStatus, ClaimVerdict, VerdictType


@dataclass(frozen=True, slots=True)
class ClaimVerdictInput:
    """Typed input payload for persisting a `ClaimVerdict`."""

    requirement_id: uuid.UUID
    verdict: VerdictType
    claim_summary: str
    cited_chunk_ids: Sequence[uuid.UUID] = ()
    policy_downgraded: bool = False
    gap_notes: str | None = None


class AnalysisRepo:
    """Async CRUD repository for `analysis_runs` and `claim_verdicts`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(
        self,
        *,
        opportunity_id: uuid.UUID,
        thread_id: str,
        status: AnalysisRunStatus = AnalysisRunStatus.RUNNING,
        trace_id: str | None = None,
        run_id: uuid.UUID | None = None,
    ) -> AnalysisRun:
        """Create a new `AnalysisRun` record bound to a LangGraph `thread_id`."""
        run = AnalysisRun(
            id=run_id or uuid.uuid4(),
            opportunity_id=opportunity_id,
            thread_id=thread_id,
            status=status,
            trace_id=trace_id,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def get_run_by_id(
        self,
        run_id: uuid.UUID,
        *,
        include_verdicts: bool = False,
        include_drafts: bool = False,
    ) -> AnalysisRun | None:
        """Fetch an `AnalysisRun` by UUID with optional eager-loaded relations."""
        stmt = select(AnalysisRun).where(AnalysisRun.id == run_id)
        if include_verdicts:
            stmt = stmt.options(selectinload(AnalysisRun.claim_verdicts))
        if include_drafts:
            stmt = stmt.options(selectinload(AnalysisRun.action_drafts))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_run_by_thread_id(
        self,
        thread_id: str,
        *,
        include_verdicts: bool = False,
        include_drafts: bool = False,
    ) -> AnalysisRun | None:
        """Fetch an `AnalysisRun` by its unique LangGraph checkpoint `thread_id`."""
        stmt = select(AnalysisRun).where(AnalysisRun.thread_id == thread_id)
        if include_verdicts:
            stmt = stmt.options(selectinload(AnalysisRun.claim_verdicts))
        if include_drafts:
            stmt = stmt.options(selectinload(AnalysisRun.action_drafts))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_runs_for_opportunity(
        self,
        opportunity_id: uuid.UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AnalysisRun]:
        """List analysis runs for a target opportunity ordered newest-first."""
        stmt = (
            select(AnalysisRun)
            .where(AnalysisRun.opportunity_id == opportunity_id)
            .order_by(AnalysisRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_run_metrics(
        self,
        run_id: uuid.UUID,
        *,
        status: AnalysisRunStatus | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: Decimal | None = None,
        latency_ms: int | None = None,
        trace_id: str | None = None,
    ) -> AnalysisRun | None:
        """Update status and observability metrics on an existing `AnalysisRun`."""
        run = await self.get_run_by_id(run_id)
        if run is None:
            return None
        if status is not None:
            run.status = status
        if total_tokens is not None:
            if total_tokens < 0:
                raise ValueError("total_tokens must be non-negative.")
            run.total_tokens = total_tokens
        if estimated_cost_usd is not None:
            if estimated_cost_usd < 0:
                raise ValueError("estimated_cost_usd must be non-negative.")
            run.estimated_cost_usd = estimated_cost_usd
        if latency_ms is not None:
            if latency_ms < 0:
                raise ValueError("latency_ms must be non-negative.")
            run.latency_ms = latency_ms
        if trace_id is not None:
            run.trace_id = trace_id
        await self._session.flush()
        return run

    async def delete_run(self, run_id: uuid.UUID) -> bool:
        """Delete an `AnalysisRun` (cascades to claim verdicts and action drafts)."""
        result = await self._session.execute(
            delete(AnalysisRun).where(AnalysisRun.id == run_id).returning(AnalysisRun.id)
        )
        return result.scalar_one_or_none() is not None

    async def upsert_verdicts(
        self,
        analysis_run_id: uuid.UUID,
        verdicts: Sequence[ClaimVerdictInput],
    ) -> list[ClaimVerdict]:
        """Atomically upsert `ClaimVerdict` rows for `(analysis_run_id, requirement_id)`."""
        if not verdicts:
            return []

        rows: list[dict[str, object]] = []
        for item in verdicts:
            cited_ids = list(item.cited_chunk_ids)
            if item.verdict is not VerdictType.UNSUPPORTED and not cited_ids:
                raise ValueError(
                    f"Verdict '{item.verdict.value}' requires at least one cited_chunk_id."
                )
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "analysis_run_id": analysis_run_id,
                    "requirement_id": item.requirement_id,
                    "verdict": item.verdict,
                    "claim_summary": item.claim_summary,
                    "cited_chunk_ids": cited_ids,
                    "policy_downgraded": item.policy_downgraded,
                    "gap_notes": item.gap_notes,
                }
            )

        insert_stmt = pg_insert(ClaimVerdict).values(rows)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[ClaimVerdict.analysis_run_id, ClaimVerdict.requirement_id],
            set_={
                "verdict": insert_stmt.excluded.verdict,
                "claim_summary": insert_stmt.excluded.claim_summary,
                "cited_chunk_ids": insert_stmt.excluded.cited_chunk_ids,
                "policy_downgraded": insert_stmt.excluded.policy_downgraded,
                "gap_notes": insert_stmt.excluded.gap_notes,
            },
        ).returning(ClaimVerdict)

        result = await self._session.execute(upsert_stmt)
        await self._session.flush()
        return list(result.scalars().all())

    async def list_verdicts(self, analysis_run_id: uuid.UUID) -> list[ClaimVerdict]:
        """List all claim verdicts recorded for an analysis run."""
        stmt = select(ClaimVerdict).where(ClaimVerdict.analysis_run_id == analysis_run_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
