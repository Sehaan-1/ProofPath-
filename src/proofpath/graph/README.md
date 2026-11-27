# Running the analysis graph

`build_workflow` binds one existing analysis run and its `thread_id` to a live
`AsyncPostgresSaver`. Open the saver with `postgres_checkpointer()` and keep its
context open while invoking, streaming, or reading graph state. Construct the
workflow with a SQLAlchemy async session factory, settings, role parser,
embedding client, comparison backend, and sprint planner. `OpenAIGraphBackend`
implements both model-backed interfaces; `create_role_parser` provides the role
parser. Apply Alembic migrations before using the workflow.

Call `await graph.ainvoke(initial_state(opportunity_id, settings=settings),
workflow_config(thread_id))`. Use the same run and thread when reopening the
saver to resume. `graph.astream` and `graph.aget_state` use the same bound
thread. The facade enforces synchronous checkpoint writes and a fixed recursion
limit regardless of caller-provided settings. `step_budget` is capped at 100;
LangGraph's recursion limit includes additional steps for error handling.
Calendar start/end must be explicitly supplied when a sprint is generated.

A node failure is routed to `fallback_error` after its transaction rolls back.
The result retains completed state, includes a sanitized `errors` entry, and
omits unverified verdicts. The fallback attempts an independent
`graph.node_failed` audit entry. If auditing itself fails, the returned errors
include `audit_write_failed`. Audit records and checkpoints can contain sensitive
identifiers and must be access-controlled. Failures in the checkpointer itself,
invalid thread configuration, cancellation, and other process-level failures
cannot be converted into a checkpointed fallback result and propagate to the
caller. Audit writes can be repeated if a process fails between audit commit and
checkpointing the fallback result.
