"""Unit tests validating local infrastructure configuration (`docker-compose.yml` & init SQL)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from proofpath.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
POSTGRES_INIT_SQL_PATH = REPO_ROOT / "infra" / "postgres" / "init.sql"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"


def _load_compose() -> dict[str, Any]:
    yaml_mod = importlib.import_module("yaml")
    loaded: Any = yaml_mod.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.unit
def test_postgres_service_uses_pgvector_image_volume_and_healthcheck() -> None:
    assert COMPOSE_PATH.exists()
    compose = _load_compose()

    services = compose["services"]
    assert isinstance(services, dict)

    postgres = services["postgres"]
    assert isinstance(postgres, dict)
    assert postgres["image"] == "pgvector/pgvector:pg16"

    # Persistent named volume must be mounted to PGDATA and declared at top level.
    volumes = postgres["volumes"]
    assert isinstance(volumes, list)
    assert "postgres_data:/var/lib/postgresql/data" in volumes

    top_volumes = compose["volumes"]
    assert isinstance(top_volumes, dict)
    assert "postgres_data" in top_volumes

    # Init SQL must be mounted read-only into Postgres's first-boot init directory.
    assert any(
        isinstance(mount, str)
        and mount.startswith("./infra/postgres/init.sql:/docker-entrypoint-initdb.d/")
        and mount.endswith(":ro")
        for mount in volumes
    )

    # Healthcheck must invoke pg_isready.
    healthcheck = postgres["healthcheck"]
    assert isinstance(healthcheck, dict)
    test_cmd = healthcheck["test"]
    assert isinstance(test_cmd, list)
    assert any("pg_isready" in str(part) for part in test_cmd)


@pytest.mark.unit
def test_postgres_compose_defaults_match_settings_database_url() -> None:
    compose = _load_compose()
    postgres = compose["services"]["postgres"]
    env = postgres["environment"]
    assert isinstance(env, dict)

    settings = Settings(_env_file=None)
    parsed = urlparse(str(settings.database_url))

    assert parsed.username is not None
    assert parsed.password is not None
    assert parsed.port is not None
    db_name = parsed.path.lstrip("/")
    assert db_name

    assert f":-{parsed.username}" in str(env["POSTGRES_USER"])
    assert f":-{parsed.password}" in str(env["POSTGRES_PASSWORD"])
    assert f":-{db_name}" in str(env["POSTGRES_DB"])

    ports = postgres["ports"]
    assert isinstance(ports, list)
    assert any(f":-{parsed.port}}}:5432" in str(mapping) for mapping in ports)


@pytest.mark.unit
def test_postgres_init_sql_enables_pgvector_extension() -> None:
    assert POSTGRES_INIT_SQL_PATH.exists()
    sql = POSTGRES_INIT_SQL_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+vector\s*;",
        sql,
        flags=re.IGNORECASE,
    )


@pytest.mark.unit
def test_redis_service_uses_alpine_image_healthcheck_and_matches_settings() -> None:
    compose = _load_compose()
    redis = compose["services"]["redis"]
    assert isinstance(redis, dict)
    assert redis["image"] == "redis:7-alpine"

    # Persistent volume must be declared at top level and mounted to /data.
    volumes = redis["volumes"]
    assert isinstance(volumes, list)
    assert "redis_data:/data" in volumes
    assert "redis_data" in compose["volumes"]

    # Healthcheck must run `redis-cli ping`.
    healthcheck = redis["healthcheck"]
    assert isinstance(healthcheck, dict)
    test_cmd = " ".join(str(part) for part in healthcheck["test"])
    assert "redis-cli" in test_cmd
    assert "ping" in test_cmd

    # Port default must match Settings.redis_url.
    settings = Settings(_env_file=None)
    parsed = urlparse(str(settings.redis_url))
    assert parsed.port is not None
    ports = redis["ports"]
    assert isinstance(ports, list)
    assert any(f":-{parsed.port}}}:6379" in str(mapping) for mapping in ports)


@pytest.mark.unit
def test_dockerfile_builds_proofpath_image_and_runs_worker() -> None:
    """Sub-part 2.2.1a: a shared image installs the codebase and runs the worker."""
    assert DOCKERFILE_PATH.exists()
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    # The proofpath codebase is copied in and installed (shared across services).
    assert "COPY pyproject.toml" in dockerfile
    assert "COPY src/ ./src/" in dockerfile
    assert "pip install ." in dockerfile
    # Non-root runtime user.
    assert "useradd" in dockerfile
    assert "proofpath" in dockerfile
    assert "USER proofpath" in dockerfile
    # Default command runs the background worker entrypoint.
    assert '"python", "-m", "proofpath.workers"' in dockerfile


@pytest.mark.unit
def test_worker_service_shares_codebase_and_wires_postgres_and_redis() -> None:
    """Sub-part 2.2.1a: the worker builds the shared image and connects to both."""
    compose = _load_compose()
    worker = compose["services"]["worker"]
    assert isinstance(worker, dict)

    # Builds the shared proofpath image from the repo Dockerfile.
    build = worker["build"]
    assert isinstance(build, dict)
    assert build["context"] == "."
    assert build["dockerfile"] == "Dockerfile"

    # Waits for both dependencies to be healthy before starting.
    depends_on = worker["depends_on"]
    assert isinstance(depends_on, dict)
    assert depends_on["postgres"]["condition"] == "service_healthy"
    assert depends_on["redis"]["condition"] == "service_healthy"

    # Connects to Postgres and Redis by compose service hostname (not localhost).
    env = worker["environment"]
    assert isinstance(env, dict)
    assert "@postgres:5432" in str(env["DATABASE_URL"])
    assert str(env["REDIS_URL"]) == "redis://redis:6379/0"

    # Runs the worker entrypoint and restarts on failure.
    command = worker["command"]
    assert isinstance(command, list)
    assert command == ["python", "-m", "proofpath.workers"]
    assert worker["restart"] == "unless-stopped"


@pytest.mark.unit
def test_worker_service_database_url_uses_settings_credentials_and_db() -> None:
    """The worker's DATABASE_URL carries the same defaults as Settings."""
    compose = _load_compose()
    env = compose["services"]["worker"]["environment"]
    settings = Settings(_env_file=None)
    parsed = urlparse(str(settings.database_url))

    assert parsed.username is not None
    assert parsed.password is not None
    db_name = parsed.path.lstrip("/")
    assert db_name
    db_url = str(env["DATABASE_URL"])
    assert f":-{parsed.username}" in db_url
    assert f":-{parsed.password}" in db_url
    assert f":-{db_name}" in db_url
