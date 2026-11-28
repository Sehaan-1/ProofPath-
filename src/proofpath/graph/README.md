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
for a generated sprint. In the last tool-using node the final citation/trace
check, verdict upsert and draft writes share one transaction. Only a checked
result gets `output_validated=True`. `tool_trace` records names only, and is
independent of both the JD and any external observability backend.

## Run tracing (Sub-slice 2.4.1)

`build_workflow` resolves the single configured observability backend
(`proofpath.observability.tracing`) and binds it to the run; a misconfigured
backend fails closed at build time, before any node runs. Every `ainvoke`,
`aresume`, and `astream` enters the tracer's invocation context, attaches its
LangGraph callback handlers to the invocation config (caller callbacks are
preserved), and flushes buffered spans afterwards — even when the invocation
raises. The unified trace ID is `sha256(str(analysis_run_id))[:16].hex()`
(`BoundedWorkflow.trace_id`), so rebuilding the graph for an approval resume,
possibly in another process, joins the original trace. With
`OBSERVABILITY_BACKEND=none` no callbacks are attached and no SDK is imported.
The MVP backend is Langfuse (`proofpath[observability]` extra); OpenLLMetry is
not wired.

Span hierarchy (2.4.2a): the callback handler emits a parent `analysis_run`
span with child node spans (`analyse_role`, `retrieve_evidence`,
`compare_evidence`, `policy_verifier`, `approval_gate`, …). `wrap_node`
annotates the `retrieve_evidence` span with retrieved chunk count and RRF
similarity scores via `RunTracer.annotate_current_span`.

Run metrics (2.4.2b): `wrap_node` also collects each node's LLM token usage
(`proofpath.observability.usage`) onto `ProofPathState.llm_usage`, checkpointed
across the approval pause. `execute_analysis_graph` sums it into `total_tokens`
and an `estimated_cost_usd`, persisting both with `latency_ms` and the unified
`trace_id` on `analysis_runs` (the trace id even on failure); the resume path
records the same `trace_id`. `GET /api/v1/analyses/{run_id}` returns all four.

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
fallback error node. After `interrupt()` returns, the gate rechecks the ledger and
trace before recording an `ApprovalDecision` in state;
committing approvals to `action_drafts` (Slice 2.1.2) and approved-only side
effects (Slice 2.1.3) happen outside the graph.

`proofpath.approvals.decide_draft` (exposed as `POST /api/v1/drafts/{id}/approve|reject`)
commits the decision with one compare-and-set on the draft row and then resumes
the run; a repeated request returns `already_processed` without re-deciding, so an
approval can never be applied twice.

Approved-only side effects live entirely outside the graph, in
`proofpath.execution.DraftExecutor`: once the resume finishes, the executor runs for
`approved` drafts only, advances `application_tracker`, writes the local calendar
task (served as a downloadable `.ics`), and appends the immutable `audit_logs`
record that carries the approval timestamp. No graph node ever performs those
writes, and no draft is executed without a committed human approval.

The MCP surface shares that service: `approve_action` (Sub-slice 2.1.4) checks the
draft is `pending_approval` and the confirmation phrase, then calls
`proofpath.approvals.decide_draft` with the same checkpointer/dependency wiring, so
an approval from an IDE client finishes the paused run exactly like the HTTP API.

`proofpath.graph.resume.resume_approval_run` is the service layer for that flow:
it resolves the run from the thread, requires `awaiting_approval` and a real gate
pause, atomically claims the run out of `awaiting_approval` (so only one resume
can ever proceed), invokes `Command(resume=...)`, then marks the run `completed`
or `failed`. A node failure during resume is returned as `status=FAILED` with the
sanitized errors rather than raised. `BoundedWorkflow.apending_approval()` returns
the pending request (or `None`) without resuming.
