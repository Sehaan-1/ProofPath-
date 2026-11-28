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
A calendar draft is created only when explicit start/end times are supplied
for a generated sprint.

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

## Approval gate (Slice 2.1.1)

`create_action_drafts` is followed by `approval_gate`, which calls LangGraph's
native `interrupt()`. The first invocation stops there: the checkpoint (including
the interrupt payload) is written to PostgreSQL, no later node runs, and
`BoundedWorkflow.ainvoke` returns the state values plus `__interrupt__`. Persist
`status="awaiting_approval"` on the analysis run at that point; nothing has been
approved, executed, or written to `application_tracker`/calendars.

The interrupt payload is an `ApprovalRequest`: the run id plus one
`ApprovalDraftSummary` (draft id, action type, short summary) per pending draft.
It never copies draft payloads, so personal notes stay out of checkpoints.

Resume the **same** `thread_id`, rebuilding the graph against the same run and
dependencies (reopen the saver first after a restart):

```py
resumed = await graph.aresume(
    {"decision": "approved", "draft_id": str(draft_id)}, workflow_config(thread_id)
)
```

`{"decision": "approved" | "rejected", "draft_id": "<uuid>"}` is validated inside
the node on replay: exactly those two keys, an exact decision string, and a draft
id from the pending request. Anything else raises and the run routes to the
fallback error node. The gate only records an `ApprovalDecision` in state;
committing approvals to `action_drafts` (Slice 2.1.2) and approved-only side
effects (Slice 2.1.3) happen outside the graph.

`proofpath.approvals.decide_draft` (exposed as `POST /api/v1/drafts/{id}/approve|reject`)
commits the decision with one compare-and-set on the draft row and then resumes
the run; a repeated request returns `already_processed` without re-deciding, so an
approval can never be applied twice.

`proofpath.graph.resume.resume_approval_run` is the service layer for that flow:
it resolves the run from the thread, requires `awaiting_approval` and a real gate
pause, atomically claims the run out of `awaiting_approval` (so only one resume
can ever proceed), invokes `Command(resume=...)`, then marks the run `completed`
or `failed`. A node failure during resume is returned as `status=FAILED` with the
sanitized errors rather than raised. `BoundedWorkflow.apending_approval()` returns
the pending request (or `None`) without resuming.
