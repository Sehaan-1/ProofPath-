"""Retrieval, comparison and deterministic policy primitives for the graph.

These nodes accept explicit dependencies; the future workflow is responsible for
binding them and for routing with ``policy_route``. No node executes a user action.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.config import Settings, get_settings
from proofpath.db.models import (
    ActionType,
    AnalysisRun,
    ApplicationStage,
    Opportunity,
    RequirementPriority,
    VerdictType,
)
from proofpath.db.repositories.draft import DraftRepo, draft_record
from proofpath.db.repositories.opportunity import OpportunityRepo, RoleRequirementInput
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.graph.state import PolicyViolation, ProofPathState
from proofpath.ingestion.analyse_role import RoleParserBackend
from proofpath.ingestion.job_fetcher import build_parser_prompt
from proofpath.policy.injection_guard import contains_jd_injection
from proofpath.policy.ledger_validator import LedgerValidator
from proofpath.policy.tool_allowlist import ToolName
from proofpath.policy.tool_trace import call_analysis_tool
from proofpath.schemas.actions import (
    ActionDraftRecord,
    CalendarBlockPayload,
    ContributionSprintPlan,
    TrackerUpsertPayload,
)
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.evidence import EvidenceSnippet
from proofpath.schemas.opportunity import RequirementRecord


class EvidenceSearcher(Protocol):
    """The interface implemented by HybridSearcher."""

    async def search(self, query: str, *, top_k: int = 8) -> list[EvidenceSnippet]: ...


class EvidenceComparator(Protocol):
    """Structured comparison backend; all evidence is scoped to one requirement."""

    async def compare(
        self,
        requirement: RequirementRecord,
        evidence: Sequence[EvidenceSnippet],
        feedback: Sequence[PolicyViolation],
    ) -> ClaimVerdictSchema: ...


def _requirements(state: ProofPathState) -> list[RequirementRecord]:
    requirements = state.get("requirements", [])
    if not requirements:
        raise ValueError("Graph requires at least one parsed requirement")
    ids = [item.id for item in requirements]
    if len(ids) != len(set(ids)) or any(
        item.opportunity_id != state["opportunity_id"] for item in requirements
    ):
        raise ValueError("Requirements must be unique and belong to the opportunity")
    return requirements


async def retrieve_evidence_node(
    state: ProofPathState, *, searcher: EvidenceSearcher, top_k: int = 8
) -> dict[str, object]:
    """Search separately for each requirement, retaining complete provenance."""
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    evidence: dict[str, list[EvidenceSnippet]] = {}
    for requirement in _requirements(state):
        evidence[str(requirement.id)] = await call_analysis_tool(
            ToolName.SEARCH_EVIDENCE, searcher.search, requirement.requirement_text, top_k=top_k
        )
    return {
        "retrieved_evidence": evidence,
        "policy_verified": False,
        "step_count": state.get("step_count", 0) + 1,
    }


async def compare_evidence_node(
    state: ProofPathState, *, comparator: EvidenceComparator
) -> dict[str, object]:
    """Compare each requirement using only its own retrieved evidence and feedback."""
    evidence = state.get("retrieved_evidence", {})
    verdicts: list[ClaimVerdictSchema] = []
    for requirement in _requirements(state):
        key = str(requirement.id)
        if key not in evidence:
            raise ValueError(f"No retrieval result for requirement {key}")
        feedback = [
            v for v in state.get("policy_violations", []) if v.requirement_id == requirement.id
        ]
        verdict = await comparator.compare(requirement, evidence[key], feedback)
        if verdict.requirement_id != requirement.id:
            raise ValueError("Comparator returned a verdict for the wrong requirement")
        verdicts.append(verdict)
    return {
        "verdicts": verdicts,
        "policy_verified": False,
        "step_count": state.get("step_count", 0) + 1,
    }


async def policy_verifier_node(
    state: ProofPathState, *, validator: LedgerValidator, analysis_run_id: uuid.UUID | None = None
) -> dict[str, object]:
    """Check ledger and per-requirement retrieval scope; bound policy retries.

    All violations trigger a retry, even if a remaining valid citation exists.
    On exhaustion, affected verdicts are forced to unsupported rather than retaining
    an ungrounded claim. No verdicts may be missing or duplicated.
    """
    requirements = _requirements(state)
    verdicts = state.get("verdicts", [])
    ids = [verdict.requirement_id for verdict in verdicts]
    if len(ids) != len(set(ids)) or set(ids) != {r.id for r in requirements}:
        raise ValueError("Expected exactly one verdict per requirement")
    evidence = state.get("retrieved_evidence", {})
    sanitized: list[ClaimVerdictSchema] = []
    violations: list[PolicyViolation] = []
    for requirement, verdict in (
        (r, next(v for v in verdicts if v.requirement_id == r.id)) for r in requirements
    ):
        if str(requirement.id) not in evidence:
            raise ValueError("Missing per-requirement retrieval result")
        result = await validator.enforce(
            [verdict],
            entities_by_requirement={requirement.id: requirement.entity},
            allowed_chunk_ids={chunk.chunk_id for chunk in evidence[str(requirement.id)]},
            analysis_run_id=analysis_run_id,
        )
        sanitized.append(result.verdicts[0])
        for violation in result.citation_check.violations:
            violations.append(
                PolicyViolation(
                    requirement_id=requirement.id,
                    message="Invalid or out-of-scope citations",
                    unknown_chunk_ids=list(violation.unknown_chunk_ids),
                    out_of_scope_chunk_ids=list(violation.out_of_scope_chunk_ids),
                )
            )
        for event in result.downgrades:
            violations.append(
                PolicyViolation(
                    requirement_id=requirement.id,
                    message=f"Evidence policy: {event.reason.value}",
                    downgrade_reason=event.reason,
                )
            )
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 0)
    if retry_count < 0 or max_retries < 0:
        raise ValueError("Retry counts must be non-negative")
    if violations and retry_count < max_retries:
        retry_count += 1
        retry = True
        output = verdicts
    else:
        retry = False
        affected = {v.requirement_id for v in violations}
        output = [
            v.model_copy(
                update={
                    "verdict": VerdictType.UNSUPPORTED,
                    "policy_downgraded": True,
                    "claim_summary": "Insufficient verified evidence for this requirement.",
                }
            )
            if v.requirement_id in affected
            else v
            for v in sanitized
        ]
    return {
        "verdicts": output,
        "policy_violations": violations,
        "retry_count": retry_count,
        "policy_retry": retry,
        "policy_verified": not retry,
        "step_count": state.get("step_count", 0) + 1,
    }


def policy_route(state: ProofPathState) -> Literal["compare_evidence", "continue"]:
    """Conditional edge for the workflow's policy retry loop."""
    return "compare_evidence" if state.get("policy_retry", False) else "continue"


class SprintPlanner(Protocol):
    """Structured sprint generator for a single prioritized gap."""

    async def plan(
        self, requirement: RequirementRecord, verdict: ClaimVerdictSchema
    ) -> ContributionSprintPlan: ...


async def recommend_contribution_sprint_node(
    state: ProofPathState, *, planner: SprintPlanner
) -> dict[str, object]:
    """Pick a must-have gap first, then an unsupported gap, and request a plan."""
    requirements = _requirements(state)
    verdicts = state.get("verdicts", [])
    ids = [item.requirement_id for item in verdicts]
    if len(ids) != len(set(ids)) or set(ids) != {r.id for r in requirements}:
        raise ValueError("Expected exactly one verdict per requirement")
    by_id = {v.requirement_id: v for v in verdicts}
    gaps = [r for r in requirements if by_id[r.id].verdict != VerdictType.SUPPORTED]
    if not gaps:
        return {"contribution_sprint": None, "step_count": state.get("step_count", 0) + 1}
    chosen = min(
        gaps,
        key=lambda r: (
            r.priority != RequirementPriority.MUST_HAVE,
            by_id[r.id].verdict != VerdictType.UNSUPPORTED,
            str(r.id),
        ),
    )
    plan = await planner.plan(chosen, by_id[chosen.id])
    if plan.target_gap_requirement_ids != [chosen.id]:
        raise ValueError("Sprint must target exactly the selected gap")
    return {"contribution_sprint": plan, "step_count": state.get("step_count", 0) + 1}


async def planner_router(
    state: ProofPathState,
    *,
    session: AsyncSession,
    settings: Settings | None = None,
    step_budget: int = 50,
) -> dict[str, object]:
    """Validate the opportunity, read tracker status, and initialize the step budget.

    Route using ``planner_route``; this function does not mutate the tracker.
    The caller owns the session and must commit writes made by subsequent nodes.
    """
    opportunity_id = state.get("opportunity_id")
    if not isinstance(opportunity_id, uuid.UUID):
        raise ValueError("opportunity_id must be a UUID")
    if step_budget < 1:
        raise ValueError("step_budget must be positive")
    opportunity = await call_analysis_tool(
        ToolName.GET_OPPORTUNITY, OpportunityRepo(session).get_by_id, opportunity_id
    )
    if opportunity is None:
        raise ValueError("Opportunity does not exist")
    if state.get("requirements"):
        _requirements(state)
    tracker = await call_analysis_tool(
        ToolName.GET_APPLICATION_STATUS, TrackerRepo(session).get_by_opportunity_id, opportunity_id
    )
    retries = state.get("max_retries")
    if retries is None:
        retries = (settings or get_settings()).max_agent_retries
    if retries < 0 or retries > (settings or get_settings()).max_agent_retries:
        raise ValueError("max_retries exceeds the configured retry limit")
    return {
        # Diagnose the source even when cached requirements bypass role parsing.
        "injection_detected": contains_jd_injection(opportunity.raw_text),
        "tracker_stage": tracker.stage.value if tracker else None,
        "step_budget": step_budget,
        "step_count": state.get("step_count", 0) + 1,
        "retry_count": state.get("retry_count", 0),
        "max_retries": retries,
    }


def planner_route(state: ProofPathState) -> Literal["analyse_role", "retrieve_evidence"]:
    """Select parsing only when structured requirements are not in state."""
    return "retrieve_evidence" if state.get("requirements") else "analyse_role"


async def analyse_role_node(
    state: ProofPathState, *, session: AsyncSession, parser: RoleParserBackend
) -> dict[str, object]:
    """Reuse stored requirements or parse and persist them for the existing opportunity.

    Legacy rows without entities are matched exactly to a fresh parse; ambiguous or
    changed output fails closed, rather than silently using a wrong entity or IDs.
    Row locking serializes concurrent parsers until the caller commits/rolls back.
    """
    if state.get("requirements"):
        return {"requirements": _requirements(state), "step_count": state.get("step_count", 0) + 1}
    result = await session.execute(
        select(Opportunity).where(Opportunity.id == state["opportunity_id"]).with_for_update()
    )
    opportunity = result.scalar_one_or_none()
    if opportunity is None:
        raise ValueError("Opportunity does not exist")
    repo = OpportunityRepo(session)
    rows = await repo.list_requirements(opportunity.id)
    if not rows or any(not row.entity for row in rows):
        parsed = await parser.parse(build_parser_prompt(opportunity.raw_text))
        if not rows:
            rows = await repo.add_requirements(
                opportunity.id,
                [
                    RoleRequirementInput(
                        requirement_text=item.requirement_text,
                        entity=item.entity,
                        category=item.category,
                        priority=item.priority,
                    )
                    for item in parsed.requirements
                ],
            )
        else:
            candidates = {
                (item.requirement_text, item.category, item.priority): item.entity
                for item in parsed.requirements
            }
            keys = [(row.requirement_text, row.category, row.priority) for row in rows]
            if (
                len(candidates) != len(parsed.requirements)
                or len(set(keys)) != len(keys)
                or set(keys) != set(candidates)
            ):
                raise ValueError("Cannot safely recover entities for legacy requirements")
            for row, key in zip(rows, keys, strict=True):
                if row.entity and row.entity != candidates[key]:
                    raise ValueError("Parsed entity conflicts with stored requirement")
                row.entity = candidates[key]
            await session.flush()
    requirements = [
        RequirementRecord(
            id=row.id,
            opportunity_id=row.opportunity_id,
            requirement_text=row.requirement_text,
            entity=row.entity,
            category=row.category,
            priority=row.priority,
        )
        for row in rows
        if row.entity
    ]
    if len(requirements) != len(rows) or not requirements:
        raise ValueError("Requirements must have extracted entities")
    return {"requirements": requirements, "step_count": state.get("step_count", 0) + 1}


async def create_action_drafts_node(
    state: ProofPathState,
    *,
    session: AsyncSession,
    analysis_run_id: uuid.UUID,
    calendar_start: datetime | None = None,
    calendar_end: datetime | None = None,
) -> dict[str, object]:
    """Persist replay-safe, approval-gated drafts in the caller's transaction.

    A calendar draft is created only with explicit aware start/end times;
    scheduling is never invented. The caller commits before checkpointing.
    A run row lock serializes concurrent replays. Existing draft status is retained;
    no tracker/calendar side effect or approval transition is performed.
    """
    if not state.get("policy_verified") or state.get("policy_retry"):
        raise ValueError("Policy verification must complete before creating drafts")
    requirements = _requirements(state)
    verdicts = state.get("verdicts", [])
    ids = [v.requirement_id for v in verdicts]
    if len(ids) != len(set(ids)) or set(ids) != {r.id for r in requirements}:
        raise ValueError("Expected exactly one verified verdict per requirement")
    sprint = state.get("contribution_sprint")
    if sprint is not None and any(
        rid not in {v.requirement_id for v in verdicts if v.verdict != VerdictType.SUPPORTED}
        for rid in sprint.target_gap_requirement_ids
    ):
        raise ValueError("Sprint targets must be unresolved gaps")
    result = await session.execute(
        select(AnalysisRun).where(AnalysisRun.id == analysis_run_id).with_for_update()
    )
    run = result.scalar_one_or_none()
    if run is None or run.opportunity_id != state["opportunity_id"]:
        raise ValueError("Analysis run does not belong to opportunity")
    tracker = await call_analysis_tool(
        ToolName.GET_APPLICATION_STATUS,
        TrackerRepo(session).get_by_opportunity_id,
        state["opportunity_id"],
    )
    # Do not propose a backwards transition or discard existing notes.
    stage = tracker.stage if tracker else ApplicationStage.SAVED
    if stage is ApplicationStage.SAVED:
        stage = ApplicationStage.PREPARING
    tracker_payload = TrackerUpsertPayload(
        action_type=ActionType.TRACKER_UPSERT,
        opportunity_id=state["opportunity_id"],
        stage=stage,
        notes=tracker.notes if tracker else None,
    )
    payloads: list[TrackerUpsertPayload | CalendarBlockPayload] = [tracker_payload]
    if sprint is not None and (calendar_start is not None or calendar_end is not None):
        if calendar_start is None or calendar_end is None:
            raise ValueError("Explicit calendar start and end are required for a sprint draft")
        if calendar_start.utcoffset() is None or calendar_end.utcoffset() is None:
            raise ValueError("Calendar times must include a timezone")
        payloads.append(
            CalendarBlockPayload(
                action_type=ActionType.CALENDAR_BLOCK,
                title=sprint.title,
                start_at=calendar_start.astimezone(UTC),
                end_at=calendar_end.astimezone(UTC),
                description=f"{sprint.scope}\n\nDeliverables: {', '.join(sprint.deliverables)}"[
                    :4000
                ],
            )
        )
    repo = DraftRepo(session)
    records: list[ActionDraftRecord] = []
    for payload in payloads:
        key = f"proofpath:{analysis_run_id}:{payload.action_type.value}"
        row, _ = await call_analysis_tool(
            ToolName.CREATE_ACTION_DRAFT,
            repo.create_or_get_draft,
            analysis_run_id=analysis_run_id,
            action_type=payload.action_type,
            payload_json=payload.model_dump(mode="json"),
            idempotency_key=key,
        )
        if row.analysis_run_id != analysis_run_id or row.action_type != payload.action_type:
            raise ValueError("Draft idempotency key conflict")
        records.append(draft_record(row))
    return {"draft_actions": records, "step_count": state.get("step_count", 0) + 1}
