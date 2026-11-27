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
