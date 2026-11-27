"""Scoped retrieval and policy retry tests."""

import uuid

import pytest

from proofpath.config import Settings
from proofpath.db.models import RequirementCategory, RequirementPriority, VerdictType
from proofpath.graph.nodes import (
    compare_evidence_node,
    policy_route,
    policy_verifier_node,
    retrieve_evidence_node,
)
from proofpath.graph.state import initial_state
from proofpath.policy.ledger_validator import LedgerValidator
from proofpath.schemas.analysis import ClaimVerdictSchema
from proofpath.schemas.evidence import EvidenceSnippet
from proofpath.schemas.opportunity import RequirementRecord

pytestmark = pytest.mark.unit


class Ledger:
    def __init__(self, chunks):
        self.chunks = chunks

    async def get_existing_chunk_ids(self, ids):
        return set(ids) & set(self.chunks)

    async def get_chunks_by_ids(self, ids):
        return [self.chunks[i] for i in ids if i in self.chunks]


class Chunk:
    def __init__(self, chunk_id, content):
        self.id = chunk_id
        self.content = content


async def test_per_requirement_scope_and_bounded_retry() -> None:
    opp, a, b, chunk = [uuid.uuid4() for _ in range(4)]
    state = initial_state(opp, settings=Settings(_env_file=None, max_agent_retries=1))
    state["requirements"] = [
        RequirementRecord(
            id=req_id,
            opportunity_id=opp,
            entity=entity,
            requirement_text=entity,
            category=RequirementCategory.LANGUAGE,
            priority=RequirementPriority.MUST_HAVE,
        )
        for req_id, entity in [(a, "Python"), (b, "Go")]
    ]
    snippet = EvidenceSnippet(
        chunk_id=chunk, source_id=uuid.uuid4(), content="Python Go", canonical_uri="file:///proof"
    )

    class Searcher:
        async def search(self, query, *, top_k):
            return [snippet] if query == "Python" else []

    class Comparator:
        async def compare(self, requirement, evidence, feedback):
            return ClaimVerdictSchema(
                requirement_id=requirement.id,
                verdict=VerdictType.SUPPORTED,
                claim_summary="Claim",
                cited_chunk_ids=[chunk],
            )

    state.update(await retrieve_evidence_node(state, searcher=Searcher()))
    state.update(await compare_evidence_node(state, comparator=Comparator()))
    validator = LedgerValidator(Ledger({chunk: Chunk(chunk, "Python Go")}))
    state.update(await policy_verifier_node(state, validator=validator))
    assert policy_route(state) == "compare_evidence"
    assert state["retry_count"] == 1
    assert state["policy_violations"][0].requirement_id == b
    assert state["policy_violations"][0].out_of_scope_chunk_ids == [chunk]
    state.update(await compare_evidence_node(state, comparator=Comparator()))
    state.update(await policy_verifier_node(state, validator=validator))
    assert policy_route(state) == "continue"
    first, second = state["verdicts"]
    assert first.verdict == VerdictType.SUPPORTED
    assert second.verdict == VerdictType.UNSUPPORTED
    assert second.policy_downgraded
    assert second.cited_chunk_ids == []


async def test_missing_verdict_rejected() -> None:
    opp = uuid.uuid4()
    state = initial_state(opp, settings=Settings(_env_file=None))
    state["requirements"] = [
        RequirementRecord(
            id=uuid.uuid4(),
            opportunity_id=opp,
            entity="Python",
            requirement_text="Python",
            category=RequirementCategory.LANGUAGE,
            priority=RequirementPriority.MUST_HAVE,
        )
    ]
    with pytest.raises(ValueError, match="exactly one verdict"):
        await policy_verifier_node(state, validator=LedgerValidator(Ledger({})))


async def test_sprint_prioritizes_must_have_and_targets_only_gap() -> None:
    from proofpath.graph.nodes import recommend_contribution_sprint_node
    from proofpath.schemas.actions import ContributionSprintPlan

    opp = uuid.uuid4()
    nice, must = uuid.uuid4(), uuid.uuid4()
    state = initial_state(opp, settings=Settings(_env_file=None))
    state["requirements"] = [
        RequirementRecord(
            id=nice,
            opportunity_id=opp,
            entity="Go",
            requirement_text="Go",
            category=RequirementCategory.LANGUAGE,
            priority=RequirementPriority.NICE_TO_HAVE,
        ),
        RequirementRecord(
            id=must,
            opportunity_id=opp,
            entity="Python",
            requirement_text="Python",
            category=RequirementCategory.LANGUAGE,
            priority=RequirementPriority.MUST_HAVE,
        ),
    ]
    state["verdicts"] = [
        ClaimVerdictSchema(
            requirement_id=nice, verdict=VerdictType.UNSUPPORTED, claim_summary="Gap"
        ),
        ClaimVerdictSchema(requirement_id=must, verdict=VerdictType.PARTIAL, claim_summary="Gap"),
    ]

    class Planner:
        async def plan(self, requirement, verdict):
            assert requirement.id == must
            return ContributionSprintPlan(
                title="Build",
                scope="Prototype",
                repo_or_topic="Python",
                deliverables=["tests"],
                estimate_days=2,
                target_gap_requirement_ids=[requirement.id],
                rationale="Practice",
            )

    result = await recommend_contribution_sprint_node(state, planner=Planner())
    assert result["contribution_sprint"].target_gap_requirement_ids == [must]
