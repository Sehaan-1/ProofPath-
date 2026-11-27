"""Exercise durable interrupt/resume across separate PostgreSQL connections."""

import uuid

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import PostgresDsn

from proofpath.config import Settings
from proofpath.db.models import RequirementCategory, RequirementPriority
from proofpath.graph.checkpointer import postgres_checkpointer, thread_config
from proofpath.graph.state import ProofPathState, initial_state
from proofpath.schemas.evidence import EvidenceSnippet
from proofpath.schemas.opportunity import RequirementRecord

pytestmark = pytest.mark.integration


def _workflow():
    async def collect(state: ProofPathState):
        req, chunk, source = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        return {
            "requirements": [
                RequirementRecord(
                    id=req,
                    opportunity_id=state["opportunity_id"],
                    entity="Python",
                    requirement_text="Python",
                    category=RequirementCategory.LANGUAGE,
                    priority=RequirementPriority.MUST_HAVE,
                )
            ],
            "retrieved_evidence": {
                str(req): [
                    EvidenceSnippet(
                        chunk_id=chunk,
                        source_id=source,
                        canonical_uri="file:///evidence",
                        content="Python",
                    )
                ]
            },
            "step_count": 1,
        }

    async def approval(state: ProofPathState):
        accepted = interrupt("continue?")
        return {
            "step_count": state["step_count"] + 1,
            "errors": []
            if accepted
            else [
                {"node": "approval", "code": "rejected", "message": "rejected", "retryable": False}
            ],
        }

    graph = StateGraph(ProofPathState)
    graph.add_node("collect", collect)
    graph.add_node("approval", approval)
    graph.add_edge(START, "collect")
    graph.add_edge("collect", "approval")
    graph.add_edge("approval", END)
    return graph


async def test_resume_with_new_saver_and_isolated_threads(integration_database_url: str) -> None:
    settings = Settings(_env_file=None, database_url=PostgresDsn(integration_database_url))
    thread_a, thread_b = str(uuid.uuid4()), str(uuid.uuid4())
    config_a, config_b = thread_config(thread_a), thread_config(thread_b)
    opp_a, opp_b = uuid.uuid4(), uuid.uuid4()
    try:
        async with postgres_checkpointer(settings=settings) as saver:
            graph = _workflow().compile(checkpointer=saver)
            await graph.ainvoke(
                initial_state(opp_a, settings=settings), config_a, durability="sync"
            )
            await graph.ainvoke(
                initial_state(opp_b, settings=settings), config_b, durability="sync"
            )
        # A newly opened connection and compiled graph must restore the suspended run.
        async with postgres_checkpointer(settings=settings) as saver:
            graph = _workflow().compile(checkpointer=saver)
            before = await graph.aget_state(config_a)
            assert before.next == ("approval",)
            assert before.values["opportunity_id"] == opp_a
            req = before.values["requirements"][0]
            assert isinstance(req, RequirementRecord)
            assert isinstance(before.values["retrieved_evidence"][str(req.id)][0], EvidenceSnippet)
            await graph.ainvoke(Command(resume=True), config_a, durability="sync")
            after = await graph.aget_state(config_a)
            other = await graph.aget_state(config_b)
            assert after.next == ()
            assert after.values["step_count"] == 2
            assert other.next == ("approval",)
            assert other.values["opportunity_id"] == opp_b
            assert other.values["step_count"] == 1
    finally:
        async with postgres_checkpointer(settings=settings) as saver:
            await saver.adelete_thread(thread_a)
            await saver.adelete_thread(thread_b)
