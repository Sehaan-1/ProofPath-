"""Synchronous and async analysis execution and checkpoint-backed report endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.api.routes import Session, _job_queue
from proofpath.config import LLMProvider
from proofpath.db.models import AnalysisRunStatus, JobStatus
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.draft import draft_record
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.graph.checkpointer import thread_config
from proofpath.graph.dependencies import GraphDependenciesError, create_graph_dependencies
from proofpath.jobs import (
    ANALYSIS_RUN_TASK,
    AnalysisRunJobPayload,
    enqueue_job,
    execute_analysis_graph,
)
from proofpath.schemas.analysis import AnalysisReport, AnalysisRunCreateRequest, ClaimVerdictSchema
from proofpath.schemas.jobs import JobAcceptedResponse
from proofpath.schemas.opportunity import ParsedRole, RequirementItem, RequirementRecord

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post(
    "/analyses/run",
    response_model=AnalysisReport,
    status_code=201,
    responses={202: {"description": "Analysis job enqueued", "model": JobAcceptedResponse}},
)
async def run_analysis(
    payload: AnalysisRunCreateRequest,
    request: Request,
    async_run: Annotated[bool, Query(alias="async")] = False,
) -> AnalysisReport | JSONResponse:
    """Run synchronously, or enqueue a background job when `?async=true` (or body flag)."""
    if payload.async_run or async_run:
        return await _enqueue_analysis(payload, request)
    settings = request.app.state.settings
    if settings.llm_provider is not LLMProvider.OPENAI:
        raise HTTPException(501, "Analysis currently requires the OpenAI model provider.")
    try:
        dependencies = create_graph_dependencies(settings)
    except GraphDependenciesError as exc:
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
    try:
        await execute_analysis_graph(
            settings=settings,
            session_factory=factory,
            checkpointer=request.app.state.checkpointer,
            dependencies=dependencies,
            opportunity_id=payload.opportunity_id,
            run_id=run_id,
            thread_id=thread_id,
            calendar_start=payload.calendar_start,
            calendar_end=payload.calendar_end,
        )
    except Exception as exc:
        raise HTTPException(503, "Analysis could not be completed.") from exc
    async with factory() as session:
        return await _get_report(run_id, request, session)


async def _enqueue_analysis(payload: AnalysisRunCreateRequest, request: Request) -> JSONResponse:
    """Validate inputs and enqueue an async analysis job; return 202 + job_id."""
    settings = request.app.state.settings
    if settings.llm_provider is not LLMProvider.OPENAI:
        raise HTTPException(501, "Analysis currently requires the OpenAI model provider.")
    try:
        create_graph_dependencies(settings)
    except GraphDependenciesError as exc:
        raise HTTPException(503, "Analysis model is not configured.") from exc
    factory = request.app.state.session_factory
    async with factory.begin() as session:
        if await OpportunityRepo(session).get_by_id(payload.opportunity_id) is None:
            raise HTTPException(404, "Opportunity not found.")
    try:
        job = await enqueue_job(
            queue=_job_queue(request),
            session_factory=factory,
            job_type=ANALYSIS_RUN_TASK,
            payload_model=AnalysisRunJobPayload(
                opportunity_id=payload.opportunity_id,
                calendar_start=payload.calendar_start,
                calendar_end=payload.calendar_end,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Job queue is unavailable.") from exc
    return JSONResponse(
        status_code=202,
        content=JobAcceptedResponse(
            job_id=job.id, job_type=ANALYSIS_RUN_TASK, status=JobStatus.QUEUED
        ).model_dump(mode="json"),
    )


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
    drafts = [draft_record(d) for d in run.action_drafts]
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
