SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# Prefer active virtualenv, then repo-local .venv, then system python3.
PYTHON ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))
COMPOSE ?= docker compose

.PHONY: help up down ps psql check-pgvector migrate lint format typecheck test-unit test-int mcp-dev eval

help: ## Show available ProofPath targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start PostgreSQL (pgvector) and Redis services and wait for healthchecks
	$(COMPOSE) up -d --wait

down: ## Stop local Docker Compose services
	$(COMPOSE) down

ps: ## Show status and health of local Docker Compose services
	$(COMPOSE) ps

psql: ## Open an interactive psql shell against the local PostgreSQL service
	$(COMPOSE) exec postgres psql -U "$${POSTGRES_USER:-proofpath}" -d "$${POSTGRES_DB:-proofpath}"

check-pgvector: ## Verify the pgvector extension is active in PostgreSQL
	$(COMPOSE) exec -T postgres psql -U "$${POSTGRES_USER:-proofpath}" -d "$${POSTGRES_DB:-proofpath}" \
		-c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

migrate: ## Apply all Alembic database migrations to head
	$(PYTHON) -m alembic upgrade head

lint: ## Run Ruff linter and format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Auto-format code and apply safe Ruff fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck: ## Run strict mypy type checks on src/
	$(PYTHON) -m mypy src/

test-unit: ## Run fast isolated unit tests
	$(PYTHON) -m pytest tests/unit

test-int: ## Run live service-backed integration tests (Postgres/pgvector, Redis, MCP)
	$(PYTHON) -m pytest tests/integration

mcp-dev: ## Start the local ProofPath MCP server
	$(PYTHON) -m proofpath.mcp_server.server

eval: ## Run the ProofPath golden evaluation suite
	$(PYTHON) evals/run_evals.py
