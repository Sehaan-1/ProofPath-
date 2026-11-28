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

## Local ProofPath MCP Server (Read-Only Tools)

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
