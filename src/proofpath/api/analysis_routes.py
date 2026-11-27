"""Synchronous analysis execution and checkpoint-backed report endpoints."""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import AnyHttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.api.routes import Session
from proofpath.config import LLMProvider
from proofpath.db.models import AnalysisRunStatus
from proofpath.db.repositories.analysis import AnalysisRepo, ClaimVerdictInput
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.graph.backends import create_graph_backend
from proofpath.graph.checkpointer import thread_config
from proofpath.graph.state import initial_state
from proofpath.graph.workflow import build_workflow
from proofpath.ingestion.analyse_role import create_role_parser
from proofpath.retrieval.embeddings import create_embedding_client
from proofpath.schemas.actions import ActionDraftRecord, parse_action_draft_payload
from proofpath.schemas.analysis import AnalysisReport, AnalysisRunCreateRequest, ClaimVerdictSchema
from proofpath.schemas.opportunity import ParsedRole, RequirementItem, RequirementRecord

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyses/run", response_model=AnalysisReport, status_code=201)
async def run_analysis(payload: AnalysisRunCreateRequest, request: Request) -> AnalysisReport:
    """Run synchronously. Async queue execution is reserved for Slice 2.2."""
    if payload.async_run:
        raise HTTPException(501, "Queued analyses are not implemented.")
    settings = request.app.state.settings
    if settings.llm_provider is not LLMProvider.OPENAI:
        raise HTTPException(501, "Analysis currently requires the OpenAI model provider.")
    try:
        parser = create_role_parser(settings)
        embedder = create_embedding_client(settings)
        backend = create_graph_backend(settings)
    except ValueError as exc:
        raise HTTPException(503, "Analysis model is not configured.") from exc
    factory = request.app.state.session_factory
    thread_id = str(uuid.uuid4())
    run_id = uuid.uuid4()
    async with factory.begin() as session:
        if await OpportunityRepo(session).get_by_id(payload.opportunity_id) is None:
            raise HTTPException(404, "Opportunity not found.")
        await AnalysisRepo(session).create_run(
            opportunity_id=payload.opportunity_id, thread_id=thread_id, run_id=run_id
        )
    started = time.monotonic()
    try:
        graph = build_workflow(
            checkpointer=request.app.state.checkpointer,
            session_factory=factory,
            settings=settings,
            analysis_run_id=run_id,
            thread_id=thread_id,
            parser=parser,
            embedder=embedder,
            comparator=backend,
            sprint_planner=backend,
            calendar_start=payload.calendar_start,
            calendar_end=payload.calendar_end,
        )
        state = await graph.ainvoke(
            initial_state(payload.opportunity_id, settings=settings), thread_config(thread_id)
        )
        errors = state.get("errors", [])
        verified = state.get("policy_verified", False)
        status = (
            AnalysisRunStatus.FAILED
            if errors or not verified
            else (
                AnalysisRunStatus.AWAITING_APPROVAL
                if state.get("draft_actions")
                else AnalysisRunStatus.COMPLETED
            )
        )
        async with factory.begin() as session:
            repo = AnalysisRepo(session)
            if verified:
                await repo.upsert_verdicts(
                    run_id,
                    [
                        ClaimVerdictInput(
                            requirement_id=v.requirement_id,
                            verdict=v.verdict,
                            claim_summary=v.claim_summary,
                            cited_chunk_ids=v.cited_chunk_ids,
                            policy_downgraded=v.policy_downgraded,
                            gap_notes=v.gap_notes,
                        )
                        for v in state.get("verdicts", [])
                    ],
                )
            await repo.update_run_metrics(
                run_id, status=status, latency_ms=int((time.monotonic() - started) * 1000)
            )
    except asyncio.CancelledError:
        async with factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                run_id,
                status=AnalysisRunStatus.FAILED,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        raise
    except Exception as exc:
        async with factory.begin() as session:
            await AnalysisRepo(session).update_run_metrics(
                run_id,
                status=AnalysisRunStatus.FAILED,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        raise HTTPException(503, "Analysis could not be completed.") from exc
    async with factory() as session:
        return await _get_report(run_id, request, session)


async def _get_report(run_id: uuid.UUID, request: Request, session: AsyncSession) -> AnalysisReport:
    run = await AnalysisRepo(session).get_run_by_id(
        run_id, include_verdicts=True, include_drafts=True
    )
    if run is None:
        raise HTTPException(404, "Analysis run not found.")
    opportunity = await OpportunityRepo(session).get_by_id(
        run.opportunity_id, include_requirements=True
    )
    if opportunity is None:
        raise HTTPException(404, "Opportunity not found.")
    requirements = [
        RequirementRecord(
            id=r.id,
            opportunity_id=r.opportunity_id,
            requirement_text=r.requirement_text,
            entity=r.entity,
            category=r.category,
            priority=r.priority,
        )
        for r in opportunity.requirements
        if r.entity
    ]
    role = None
    if requirements:
        role = ParsedRole(
            company=opportunity.company,
            role_title=opportunity.role_title,
            source_url=AnyHttpUrl(opportunity.source_url) if opportunity.source_url else None,
            requirements=[
                RequirementItem(
                    requirement_text=r.requirement_text,
                    entity=r.entity,
                    category=r.category,
                    priority=r.priority,
                )
                for r in requirements
            ],
        )
    checkpoint = await request.app.state.checkpointer.aget_tuple(thread_config(run.thread_id))
    state = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
    errors = list(state.get("errors", []))
    if run.status is AnalysisRunStatus.FAILED and not errors:
        errors = [
            {
                "node": "workflow",
                "code": "run_failed",
                "message": "Analysis did not complete.",
                "retryable": False,
            }
        ]
    drafts = [
        ActionDraftRecord(
            id=d.id,
            analysis_run_id=d.analysis_run_id,
            action_type=d.action_type,
            status=d.status,
            payload=parse_action_draft_payload(d.payload_json),
            idempotency_key=d.idempotency_key,
            approved_at=d.approved_at,
            executed_at=d.executed_at,
        )
        for d in run.action_drafts
    ]
    verdicts = [ClaimVerdictSchema.model_validate(v) for v in run.claim_verdicts]
    return AnalysisReport(
        analysis_run_id=run.id,
        opportunity_id=run.opportunity_id,
        thread_id=run.thread_id,
        status=run.status,
        role=role,
        requirements=requirements,
        verdicts=verdicts,
        contribution_sprint=state.get("contribution_sprint")
        if state.get("policy_verified")
        else None,
        action_drafts=drafts,
        total_tokens=run.total_tokens,
        estimated_cost_usd=run.estimated_cost_usd,
        latency_ms=run.latency_ms,
        trace_id=run.trace_id,
        errors=errors,
    )


@router.get("/analyses/{run_id}", response_model=AnalysisReport)
async def get_analysis(run_id: uuid.UUID, request: Request, session: Session) -> AnalysisReport:
    """Return persisted verdicts/drafts and checkpointed partial failure details."""
    return await _get_report(run_id, request, session)
