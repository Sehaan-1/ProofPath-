# ProofPath

## API bootstrap

Start PostgreSQL with pgvector (`make up`), configure `DATABASE_URL` using
`.env.example`, and apply the application schema with `make migrate`. The
LangGraph checkpointer initializes its own tables during API startup; the
configured database user needs permission to create or upgrade those tables.
Coordinate startup when checkpoint migrations are required across processes.

Run the API with:

```sh
uvicorn proofpath.main:app --host 127.0.0.1 --port 8000
```

`GET /health` returns 200 only when PostgreSQL is reachable and pgvector is
installed; otherwise it returns 503. Startup fails if the checkpointer cannot
be initialized. HTTP errors use `application/problem+json` with RFC 7807 fields.
The engine and checkpointer are closed on shutdown. The API has no authentication: bind only to a trusted local interface or protect it
with a separate access-control layer. Evidence ingestion sends content to the
configured embedding provider. GitHub ingestion fetches public repository data;
users are responsible for supplying repositories they own.

The API exposes text, file, and GitHub ingestion, evidence search, opportunity
creation from pasted text, opportunity lookup, and tracker listing under
`/api/v1`. File uploads require a `canonical_uri` form field and are limited to
5 MiB. Existing source URIs with changed content return 409 rather than deleting
chunks that may already be cited. URL-only opportunity fetching is disabled
until DNS and redirect SSRF protections are in place. Analysis run and report
endpoints have not yet been implemented.
