# ProofPath

Personal Evidence & Opportunity Agent enforcing two deterministic invariants:
1. **Citation Grounding:** Every `supported` or `partial` requirement verdict must cite verified user-owned evidence chunks in PostgreSQL (`evidence_chunks`); ungrounded claims are deterministically downgraded to `unsupported`.
2. **Human-in-the-Loop Side Effects:** Analysis runs only persist `pending_approval` rows in `action_drafts` and never mutate `application_tracker` or external calendars automatically.

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
  - `POST /api/v1/analyses/run` — execute the durable LangGraph workflow (`planner_router` → `analyse_role` → `retrieve_evidence` → `compare_evidence` → `policy_verifier` → `recommend_contribution_sprint` → `create_action_drafts`). Optionally pass timezone-aware `calendar_start` and `calendar_end` to generate a `pending_approval` calendar block draft alongside the tracker draft.
  - `GET /api/v1/analyses/{run_id}` — fetch persisted verdicts, contribution sprint plan, `pending_approval` action drafts, and checkpointed run diagnostics.
  - `GET /api/v1/tracker` — list personal application-tracker entries.

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
