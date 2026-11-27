"""Live audit + checkpoint test for a failed model-backed node."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import PostgresDsn
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from proofpath.config import Settings
from proofpath.db.models import AuditLog, Opportunity
from proofpath.db.repositories.analysis import AnalysisRepo
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.graph.state import initial_state
from proofpath.graph.workflow import build_workflow, workflow_config

pytestmark = pytest.mark.integration


async def test_model_failure_is_audited_and_checkpointed(db_engine, integration_database_url):
    settings = Settings(_env_file=None, database_url=PostgresDsn(integration_database_url))
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    thread_id = str(uuid4())
    async with factory.begin() as session:
        opp = await OpportunityRepo(session).create_opportunity(
            company="Example",
            role_title="Engineer",
            raw_text="Python required",
            sanitized_text="Python required",
        )
        opp_id = opp.id
        run = await AnalysisRepo(session).create_run(opportunity_id=opp_id, thread_id=thread_id)
        run_id = run.id

    class FailingParser:
        async def parse(self, prompt):
            raise TimeoutError("sensitive provider response")

    try:
        async with postgres_checkpointer(settings=settings) as saver:
            graph = build_workflow(
                checkpointer=saver,
                session_factory=factory,
                settings=settings,
                analysis_run_id=run_id,
                thread_id=thread_id,
                parser=FailingParser(),
                embedder=MagicMock(),
                comparator=MagicMock(),
                sprint_planner=MagicMock(),
            )
            result = await graph.ainvoke(
                initial_state(opp_id, settings=settings), workflow_config(thread_id)
            )
            assert result["errors"][0]["node"] == "analyse_role"
            assert result["errors"][0]["retryable"] is True
            assert "sensitive" not in str(result["errors"])
        async with postgres_checkpointer(settings=settings) as saver:
            saved = await saver.aget_tuple(workflow_config(thread_id))
            assert saved is not None
            assert saved.checkpoint["channel_values"]["errors"] == result["errors"]
            await saver.adelete_thread(thread_id)
        async with factory() as session:
            logs = (
                (
                    await session.execute(
                        select(AuditLog).where(
                            AuditLog.resource_id == run_id,
                            AuditLog.event_type == "graph.node_failed",
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(logs) == 1
            assert logs[0].details_json["node"] == "analyse_role"
            assert "sensitive" not in str(logs[0].details_json)
    finally:
        async with factory.begin() as session:
            await session.execute(delete(AuditLog).where(AuditLog.resource_id == run_id))
            await session.execute(delete(Opportunity).where(Opportunity.id == opp_id))
