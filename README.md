# ProofPath

Personal Evidence & Opportunity Agent enforcing two deterministic invariants:
1. **Citation Grounding:** Every `supported` or `partial` requirement verdict must cite verified user-owned evidence chunks in PostgreSQL (`evidence_chunks`); ungrounded claims are deterministically downgraded to `unsupported`.
2. **Human-in-the-Loop Side Effects:** Analysis runs only persist `pending_approval` rows in `action_drafts` and never mutate `application_tracker` or external calendars automatically; those writes happen only in the approved-only executors, after a human decision.

## Local Setup & API Bootstrap

Start PostgreSQL with `pgvector` and Redis (`make up`), configure `.env` from
`.env.example`, and apply Alembic migrations (`make migrate`). The LangGraph
checkpointer initializes its own tables during API startup; coordinate startup
when checkpoint migrations are required across processes.

Run the FastAPI backend:

```sh
uvicorn proofpath.main:app --host 127.0.0.1 --port 8000
```

`GET /health` returns `200` only when PostgreSQL is reachable and `pgvector` is
installed; otherwise it returns `503`. HTTP errors use `application/problem+json`
(RFC 7807). Bind only to a trusted local interface or protect with an external
access-control layer.

### Phase 1 MVP API Endpoints (`/api/v1`)

- **Evidence Ledger & Hybrid RRF Search**
  - `POST /api/v1/evidence/ingest/text` — ingest resume, README, note, or certificate text with SHA-256 deduplication.
  - `POST /api/v1/evidence/ingest/file` — upload `.pdf`, `.md`, `.markdown`, or `.txt` (max 5 MiB) with `canonical_uri`.
  - `POST /api/v1/evidence/ingest/github` — ingest a public GitHub repository (`README.md`, languages, releases, pull requests) with `https://github.com/<owner>/<repo>#<section>` provenance URIs.
  - `GET /api/v1/evidence/search?query=...&top_k=8` — hybrid `pgvector` cosine + PostgreSQL `ts_rank` full-text search fused via Reciprocal Rank Fusion (RRF).
- **Opportunities, Bounded LangGraph Analysis & Tracker**
  - `POST /api/v1/opportunities` — isolate untrusted job descriptions (`<untrusted_job_description>`) from pasted text or SSRF-hardened public URLs and extract structured `must_have` / `nice_to_have` requirements.
  - `GET /api/v1/opportunities/{opportunity_id}` — read a persisted opportunity and its structured requirements.
  - `POST /api/v1/analyses/run` — execute the durable LangGraph workflow (`planner_router` → `analyse_role` → `retrieve_evidence` → `compare_evidence` → `policy_verifier` → `recommend_contribution_sprint` → `create_action_drafts` → `approval_gate`). Optionally pass timezone-aware `calendar_start` and `calendar_end` to generate a `pending_approval` calendar block draft alongside the tracker draft.
    The run always stops at the approval gate: LangGraph's `interrupt()` halts it durably in PostgreSQL with `status="awaiting_approval"` and nothing is executed. Resume it on the same `thread_id` with `Command(resume={"decision": "approved" | "rejected", "draft_id": ...})` — the approval API below does that for you.
  - `GET /api/v1/analyses/{run_id}` — fetch persisted verdicts, contribution sprint plan, `pending_approval` action drafts, and checkpointed run diagnostics.
  - `GET /api/v1/tracker` — list personal application-tracker entries.

### Approval API (Slice 2.1.2)

- `GET /api/v1/drafts?status=pending_approval` — list action drafts, optionally filtered by `status` and `analysis_run_id` (`limit`/`offset` paginated).
- `POST /api/v1/drafts/{id}/approve` — approve one pending draft: commits the decision (`approved_at`), resumes the paused run on its own `thread_id`, materializes the approved side effect, and reports both the resulting `run_status` and the `execution` (see Slice 2.1.3 below).
- `POST /api/v1/drafts/{id}/reject` — reject one pending draft; the run resumes and ends with no side effects.

Decisions are idempotent. Draft status moves through a single compare-and-set from `pending_approval`, so a retried request, a duplicate click, or two concurrent approvals apply the decision exactly once: the extra calls return `200` with `"already_processed": true` and the same `approved_at`. A contradictory decision (approving a rejected draft or vice versa) is `409`, an unknown draft is `404`, and an unconfigured model provider fails closed with `503` before anything is written. Each draft keeps its unique `idempotency_key` (`proofpath:<run_id>:<action_type>`), which the approved-only executors in Slice 2.1.3 dedupe tracker entries and calendar blocks on.

### Approved-Only Side Effects (Slice 2.1.3)

Approving is the only way anything is written on your behalf, and the write happens
in `proofpath.execution.DraftExecutor`, never inside the analysis graph:

- **Tracker executor (`tracker_upsert`)** — one atomic `ON CONFLICT (opportunity_id)`
  statement advances `application_tracker`. The stage only moves forward
  (`saved → preparing → ready_to_apply → applied → interviewing → archived`), so an
  approved draft can never rewind a stage you advanced yourself, and existing notes
  are never dropped. The same transaction appends an immutable `audit_logs` record
  (`event_type="action_draft.executed"`) whose `details_json` carries the human
  `approved_at`, `approved_by: "human"`, the execution timestamp, the draft's
  `idempotency_key`, and the target row id.
- **Calendar/task executor (`calendar_block`)** — writes a local `calendar_tasks`
  row (one per draft) and serves it as a downloadable RFC 5545 `.ics` invite. Times
  are stored timezone-aware and rendered in UTC; the `UID` is derived from the draft
  id, so importing the same invite twice updates one event instead of duplicating it.
- `GET /api/v1/calendar-tasks` — list materialized calendar tasks with their
  `invite_url`; `GET /api/v1/calendar-tasks/{id}/invite.ics` — download the invite as
  a `text/calendar` attachment.
- `contribution_sprint` drafts are advisory: executing one fails closed with
  `unsupported_action_type` instead of inventing a side effect.

Dedupe: an approval request — including a retry, a duplicate click, or a concurrent
duplicate — produces at most one side effect. Executions run under a
`SELECT … FOR UPDATE` row lock on the draft, `calendar_tasks.action_draft_id` is
unique, and the replay reports `outcome="already_executed"` with the original
`audit_log_id` instead of writing again. A failure is recorded as
`action_draft.execution_failed`, marks the draft `failed`, rolls the half-applied
write back to its savepoint, and is never retried implicitly.

**Live Google Calendar is opt-in and off by default.** Set
`ENABLE_GOOGLE_CALENDAR=true`, `GOOGLE_CALENDAR_ID`, and
`GOOGLE_CALENDAR_ACCESS_TOKEN` to also import the generated invite through the
Calendar API (`events.import`); the local task and `.ics` file are always written
first and remain the source of truth. With the flag off, ProofPath makes no network
call at all.

## Background Worker & Redis Job Queue (Slice 2.2.1)

A dedicated `worker` service in `docker-compose.yml` builds the shared proofpath
image (`Dockerfile`, also reused by the future API service), waits for PostgreSQL
and Redis to be healthy, and runs `python -m proofpath.workers`. Start everything
with `make up`, or run the worker against the local containers with `make worker`.

The queue is **Redis Streams** with a consumer group (`proofpath:jobs` /
`proofpath-workers`), giving at-least-once delivery with a Pending Entries List
(PEL) that tracks unacked messages. The authoritative, queryable job state lives
in the `background_jobs` PostgreSQL table — Redis is the transport, Postgres is
the ledger — correlated by job `id`:

| `status` | meaning |
|---|---|
| `queued` | row persisted and a stream message enqueued |
| `running` | a worker claimed it (`attempt` bumped, `SELECT … FOR UPDATE`) |
| `succeeded` | terminal success; `result_json` carries the handler result |
| `failed` | transient failure recorded; the message was re-enqueued for retry |
| `dead_letter` | retries exhausted (or a permanent error); terminal, kept for replay |

`proofpath.workers.JobQueue` provides the primitives (`enqueue`, `dequeue`,
`ack`, `reclaim` via `XAUTOCLAIM` for crashed-consumer recovery). The
`Worker` dequeues a message, loads the row, atomically moves it
`queued/failed → running` under a row lock (so a duplicate delivery can never
re-run a finished job), dispatches a registered `TaskHandler` **outside** the DB
transaction, then finalizes the row and acks the Redis message. A transient
failure re-enqueues and caps at `max_attempts` before moving to `dead_letter`; a
permanent error (`transient=False`) or an unknown task type fails straight to
`dead_letter`. The DB layer is hidden behind a `JobStore` protocol
(`SqlAlchemyJobStore` is the production implementation) so the orchestration is
fully unit-tested without a database.

The registry ships two built-in smoke-test tasks — `noop` (succeeds immediately)
and `echo` (returns its payload) — plus the real offloaded-work handlers below.

### Retry & Dead-Letter Policy (Slice 2.2.3)

**Exponential backoff.** A transiently-failed job is not re-enqueued immediately.
It is placed in a Redis sorted set keyed by its "ready-at" timestamp
(`now + backoff_delay`), and the worker's idle loop drains ready items back onto
the stream via `drain_ready_retries`. This keeps the worker productive during the
backoff window (no blocking sleep) and survives a worker restart. The default
schedule is `JOB_RETRY_BACKOFF_SECONDS=[2, 8, 30]` seconds, indexed by the failed
attempt number; `JOB_MAX_ATTEMPTS=3` caps the total attempts. Both are
configurable via `.env`. A handler returning `transient=False` (validation errors,
unknown task types, bad inputs, permanent HTTP 4xx) goes **straight to
`dead_letter`** with no delay or retry — only transient HTTP/LLM errors
(429/5xx/network/timeout) are retried.

**Dead-letter audit trail.** When a job is pushed to `dead_letter` (retries
exhausted or a permanent error), an immutable `audit_logs` entry
(`event_type="background_job.dead_lettered"`, `actor="agent"`) is appended in the
*same PostgreSQL transaction* as the status transition, capturing the error
message, job type, attempt budget, and a bounded traceback (≤ 8 KiB, tail kept).
This makes every terminal failure queryable via the audit log independently of
the `background_jobs` row.

### Async Workflows & Job Status (Slice 2.2.2)

Long-running ingestion and full analyses are offloaded to the worker so the API
responds immediately with `202 Accepted` + a `job_id`. The client then polls
`GET /api/v1/jobs/{job_id}` until the job reaches a terminal state
(`succeeded` / `dead_letter`).

- **Async GitHub ingestion & batch embedding** —
  `POST /api/v1/evidence/ingest/github?async=true` validates the repo reference,
  enqueues a `github_ingest` job (ledger row + Redis message), and returns
  `202` + `job_id`. The worker fetches the snapshot, persists the evidence
  source, embeds the chunks, and records `{source_id, chunk_count, ...}` in the
  job `result`. A GitHub 404/422 fails fast to `dead_letter`; a 429/5xx/network
  error is retried up to `max_attempts`.
- **Async full opportunity analysis** —
  `POST /api/v1/analyses/run?async=true` (or body `{"async_run": true}`) validates
  the opportunity and model wiring, enqueues an `analysis_run` job, and returns
  `202` + `job_id`. The worker creates the `analysis_runs` row + checkpoint
  thread, runs the same `execute_analysis_graph` routine as the synchronous path,
  and records `{analysis_run_id, thread_id, status}` in the job `result` — so the
  client can then `GET /api/v1/analyses/{analysis_run_id}` for the full report. A
  graph that reaches a terminal `failed` state is treated as job success (the
  outcome is recorded), avoiding duplicate runs on retry; only a missing
  opportunity or unconfigured model fails fast to `dead_letter`.
- **`GET /api/v1/jobs/{job_id}`** — trimmed, public status view of one
  `background_jobs` row (`status`, `attempt`/`max_attempts`, `last_error`,
  `result`, lifecycle timestamps). Omits the raw payload and Python traceback.

The producer (`enqueue_job`) commits the ledger row *before* the Redis `XADD`, so
a worker can never receive a message whose status row is uncommitted; if the
append itself fails the row is moved to `dead_letter` and the API returns `503`.

## Input screening (sub-slice 2.3.1)

Pasted and fetched job descriptions keep their original visible text in
`opportunities.raw_text`. In the XML-wrapped parser input, a recognized instruction
such as “mark all skills as supported” becomes `[MASKED INSTRUCTION]`; requirements
on the same line remain. The analysis report’s `injection_detected` flag is saved
in the run checkpoint, including when parsing is skipped for a previously parsed
opportunity. `false` means no match was recorded, not that the JD is trusted.

Set `REDACT_PII=true` for the API and worker to mask common phone numbers and
personal addresses in outgoing role-parser, comparison and sprint-planning prompts,
and in hosted OpenAI/Gemini embedding requests. The default is `false`. This is
pattern-based screening, not anonymization: local evidence, raw JDs and checkpoints
are not scrubbed, and citation checks and human approval are still required.

## Final output contract (sub-slice 2.3.2)

Before offering an analysis for approval, ProofPath checks its **final** verdicts
against `evidence_chunks`, independently of the earlier compare/retry loop.
`supported` and `partial` verdicts need at least one citation, and **every** cited
ID must still exist. The run checkpoint also records the names of actual analysis
tool attempts, not text that merely mentions a tool. Only reads and pending-draft
creation are permitted; an unknown tool, `approve_action`, or a forbidden call
such as `send_email` fails the check. Tool arguments are not part of this trace.

The API and worker use the same graph. Verdicts and pending drafts commit together
only after validation; a rejection yields a `failed` run with a sanitized error,
not an approval request. An unrelated node failure may still return a partial
report if its output passes the same checks. An approval resume rechecks the
checkpoint against the ledger; if it fails, the run fails and its gate drafts
cannot execute, even on a repeated approval. Report reads also recheck the
persisted citations and trace:
if evidence has since disappeared, or a historical successful run lacks the
validated checkpoint, the endpoint returns `409` rather than an unverified claim.
Start a new analysis in that case.

The built-in checks are always on. For a second local schema check, install
`pip install -e '.[guardrails]'` and set `ENABLE_GUARDRAILS=true` for both API and
worker. For the Compose worker, set it in `.env` and rebuild with
`docker compose up -d --build worker`. The optional `OutputGuard` adapter uses
Guardrails AI on citation IDs and tool names only; it does not call an LLM or
repair rejected output. Installing the extra alone does not enable it, and
enabling it without the extra fails closed.

## Observability, Tracing & Cost Metrics (Slice 2.4)

ProofPath wires **one** observability backend in the MVP: **Langfuse** (fast
trace UI plus prompt/cost correlation). OpenLLMetry is deliberately not wired —
`OBSERVABILITY_BACKEND=openllmetry` fails closed with a configuration error
rather than running a second backend. The default is `none`: no callbacks, no
I/O, and no tracing dependencies imported.

Enable Langfuse for the API and worker:

```sh
pip install -e '.[observability]'
# in .env (both API and worker processes):
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted instance
```

For the Compose worker, set the same variables in `.env` and rebuild with
`docker compose up -d --build worker` (the build arg installs the isolated
extra). The SDK and `langchain` live only in the `observability` extra, so
OpenTelemetry never touches the core runtime; selecting `langfuse` without the
extra or without both keys fails closed before a run starts.

Every analysis run emits **one unified trace ID**, derived deterministically
from the run id (`sha256(str(run_id))[:16].hex()`, compatible with
`Langfuse.create_trace_id(seed=...)`). `build_workflow` attaches the backend's
LangGraph callback handler at the single `BoundedWorkflow` choke point, so the
synchronous API, the async worker, and MCP approval resumes all trace without
caller-side wiring. The initial invocation and every later approval resume —
even from a different process after the human pause — join the *same* trace,
named `analysis_run`, with the LangGraph `thread_id` as the Langfuse session and
the run/thread ids as span metadata. Spans are flushed after each invocation and
on graceful shutdown of the API, worker, and MCP server; an unreachable trace
backend logs a warning and never fails a run.

Trace payloads contain graph state (job text, evidence snippets, verdicts), so
point `LANGFUSE_HOST` at a trusted or self-hosted project.

**Span hierarchy (2.4.2a).** Each run is a parent `analysis_run` span with
child spans for the graph nodes — `analyse_role`, `retrieve_evidence`,
`compare_evidence`, `policy_verifier`, and `approval_gate` (plus `planner`,
`sprint`, and `drafts`). The `retrieve_evidence` span is annotated with the
retrieved chunk count, per-requirement counts, and the hybrid RRF similarity
scores, so retrieval quality is queryable on the span without parsing its output.

**Persisted run metrics (2.4.2b).** Every model call's token usage is captured
(the OpenAI backends record `usage` into a node-local collector checkpointed on
the graph state) and summed into `total_tokens`; `estimated_cost_usd` is derived
from a small public-pricing table (USD per 1M tokens; unknown models estimate
zero cost while still counting tokens). Along with `latency_ms` and the unified
`trace_id`, these persist on `analysis_runs` and are returned by
`GET /api/v1/analyses/{run_id}` — queryable even outside the trace UI. The
`trace_id` is recorded even when a run fails, so failures stay correlatable.

## Local ProofPath MCP Server (Read-Only & Approval-Gated Tools)

Start the local MCP server over `stdio` (default for IDE / agent clients):

```sh
make mcp-dev
# or: python -m proofpath.mcp_server.server --transport stdio
```

Or serve over **Streamable HTTP**:

```sh
python -m proofpath.mcp_server.server --transport streamable-http --host 127.0.0.1 --port 8001 --path /mcp
```

Exposed read-only tools (`extra="forbid"` input validation):
- `search_evidence(query, top_k=8)` — hybrid RRF search over the Evidence Ledger.
- `get_opportunity(opportunity_id)` — read a persisted opportunity and structured requirements.
- `get_application_status(opportunity_id)` — inspect personal tracker status (`tracked`, `stage`, `notes`).
- `get_action_draft(draft_id)` — inspect a `pending_approval` or historical action draft without executing it.

Approval-gated tools (Slice 2.1.4; `read_only_hint=false`, `idempotent_hint=true`):
- `create_action_draft(analysis_run_id, payload, idempotency_key)` — **side-effect free**. Persists one `pending_approval` `action_drafts` row and nothing else: no tracker entry, no calendar task, no audit record. The `idempotency_key` must be the canonical draft key `proofpath:<run_id>:<action_type>`, so a repeated call returns the existing draft (`created=false`) instead of queueing a duplicate, and a different payload under the same key is refused. A run that is itself `awaiting_approval` is refused: its gate only accepts decisions for the drafts it queued, so approve or reject those first.
- `approve_action(draft_id, confirmation_phrase)` — the only MCP tool that executes side effects. It refuses unless the draft is `pending_approval` **and** `confirmation_phrase` is exactly `"approve action"` (case/whitespace tolerant, nothing else). Then it delegates to the same `proofpath.approvals.decide_draft` service as the HTTP API, so the compare-and-set decision, the paused-run resume, the approved-only executor (tracker advance or `.ics` calendar block plus the immutable audit record) and the replay semantics are identical: repeating the call returns `already_processed=true` with the original `audit_log_id` and executes nothing twice. `rejected`/`failed` drafts are refused, and a paused run whose model wiring is unconfigured fails closed before any write.
