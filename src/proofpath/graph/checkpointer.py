"""PostgreSQL checkpoint lifecycle for asynchronous LangGraph workflows.

Keep the context open for the entire lifetime of a compiled graph. Compile with
`checkpointer=saver` and invoke/resume with the same non-empty thread ID, e.g.
`config={"configurable": {"thread_id": thread_id}}`. Use `durability="sync"`
when each step must be persisted before the next starts. The future workflow
must use this saver rather than compiling without one.

`setup()` creates/upgrades LangGraph's own tables and needs DDL privileges on
first use. Run startup/setup serially across processes when migrations are needed.
Checkpoint contents may contain personal evidence; restrict database access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from proofpath.config import Settings, get_settings

# Explicit allowlist keeps deserialization compatible with LANGGRAPH_STRICT_MSGPACK
# without permitting arbitrary checkpoint-specified constructors.
_STATE_TYPES = (
    ("proofpath.schemas.opportunity", "RequirementRecord"),
    ("proofpath.schemas.evidence", "EvidenceSnippet"),
    ("proofpath.schemas.analysis", "ClaimVerdictSchema"),
    ("proofpath.schemas.actions", "ContributionSprintPlan"),
    ("proofpath.schemas.actions", "ActionDraftRecord"),
    ("proofpath.schemas.actions", "TrackerUpsertPayload"),
    ("proofpath.schemas.actions", "CalendarBlockPayload"),
    ("proofpath.schemas.actions", "ContributionSprintPayload"),
    ("proofpath.graph.state", "PolicyViolation"),
    # Asyncpg returns UUID columns as `pgproto.UUID`; without this entry those
    # checkpointed identifiers are silently restored as bare hex strings.
    ("asyncpg.pgproto.pgproto", "UUID"),
    ("proofpath.graph.state", "ApprovalRequest"),
    ("proofpath.graph.state", "ApprovalDraftSummary"),
    ("proofpath.graph.state", "ApprovalDecision"),
    ("proofpath.policy.ledger_validator", "DowngradeReason"),
    ("proofpath.db.models", "RequirementCategory"),
    ("proofpath.db.models", "RequirementPriority"),
    ("proofpath.db.models", "EvidenceSourceType"),
    ("proofpath.db.models", "VerdictType"),
    ("proofpath.db.models", "ActionType"),
    ("proofpath.db.models", "ActionDraftStatus"),
    ("proofpath.db.models", "ApplicationStage"),
)


@asynccontextmanager
async def postgres_checkpointer(
    *, settings: Settings | None = None
) -> AsyncIterator[AsyncPostgresSaver]:
    """Open, initialize and close a saver using the configured psycopg DSN.

    Errors propagate; there is deliberately no in-memory fallback. Setup must
    complete before yielding, so a graph cannot accidentally run uncheckpointed.
    """
    active_settings = settings if settings is not None else get_settings()
    serde = JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)
    async with AsyncPostgresSaver.from_conn_string(
        active_settings.psycopg_database_url, serde=serde
    ) as saver:
        await saver.setup()
        yield saver


def thread_config(thread_id: str) -> RunnableConfig:
    """Build the LangGraph configuration identifying a run for resume."""
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id must not be blank")
    return {"configurable": {"thread_id": thread_id}}
