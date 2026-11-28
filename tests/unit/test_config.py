"""Unit tests for typed configuration (`config.py`), `.env.example`, and `.gitignore`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import PostgresDsn, ValidationError

from proofpath.config import (
    AppEnv,
    LLMProvider,
    LogLevel,
    ObservabilityBackend,
    Settings,
    get_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

_MANAGED_ENV_VARS = (
    "APP_ENV",
    "LOG_LEVEL",
    "DATABASE_URL",
    "REDIS_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "GITHUB_TOKEN",
    "ENABLE_GOOGLE_CALENDAR",
    "GOOGLE_CALENDAR_ID",
    "GOOGLE_CALENDAR_ACCESS_TOKEN",
    "OBSERVABILITY_BACKEND",
    "MAX_AGENT_RETRIES",
    "REDACT_PII",
    "JOB_MAX_ATTEMPTS",
    "JOB_RETRY_BACKOFF_SECONDS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate config tests from ambient shell variables and lru_cache state."""
    for key in _MANAGED_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env is AppEnv.LOCAL
    assert settings.log_level is LogLevel.INFO
    assert settings.llm_provider is LLMProvider.OPENAI
    assert settings.llm_api_key is None
    assert settings.github_token is None
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.observability_backend is ObservabilityBackend.NONE
    assert settings.max_agent_retries == 2
    assert settings.redact_pii is False
    assert settings.job_max_attempts == 3
    assert settings.job_retry_backoff_seconds == [2, 8, 30]
    # Live Google Calendar writes are opt-in only, never on by default.
    assert settings.enable_google_calendar is False
    assert settings.google_calendar_id is None
    assert settings.google_calendar_access_token is None
    assert settings.asyncpg_database_url.startswith("postgresql+asyncpg://")
    assert settings.psycopg_database_url.startswith("postgresql://")


@pytest.mark.unit
def test_settings_loads_all_env_vars_and_masks_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm_key = "test-llm-key-value-987654"
    fake_gh_token = "test-gh-token-value-123456"

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db.internal:5432/proofpath")
    monkeypatch.setenv("REDIS_URL", "rediss://cache.internal:6379/2")
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    monkeypatch.setenv("LLM_API_KEY", fake_llm_key)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("GITHUB_TOKEN", fake_gh_token)
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "LangFuse")
    monkeypatch.setenv("MAX_AGENT_RETRIES", "3")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("REDACT_PII", "true")

    settings = Settings(_env_file=None)

    assert settings.asyncpg_database_url == (
        "postgresql+asyncpg://user:pass@db.internal:5432/proofpath"
    )
    assert settings.psycopg_database_url == "postgresql://user:pass@db.internal:5432/proofpath"
    assert str(settings.redis_url) == "rediss://cache.internal:6379/2"
    assert settings.llm_provider is LLMProvider.OPENAI
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.embedding_dimensions == 3072
    assert settings.observability_backend is ObservabilityBackend.LANGFUSE
    assert settings.max_agent_retries == 3
    assert settings.log_level is LogLevel.DEBUG
    assert settings.redact_pii is True

    # Secrets must never appear in repr/str/model_dump output.
    assert fake_llm_key not in repr(settings)
    assert fake_llm_key not in str(settings)
    assert fake_gh_token not in repr(settings)
    assert fake_gh_token not in str(settings)
    assert fake_llm_key not in str(settings.model_dump())
    assert fake_gh_token not in str(settings.model_dump())

    # Explicit unwrapping helpers return the secret values.
    assert settings.require_llm_api_key() == fake_llm_key
    assert settings.require_github_token() == fake_gh_token


@pytest.mark.unit
def test_blank_secrets_normalize_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "   ")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("GOOGLE_CALENDAR_ID", " ")
    monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.github_token is None
    assert settings.google_calendar_id is None
    assert settings.google_calendar_access_token is None

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        settings.require_llm_api_key()

    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        settings.require_github_token()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_key", "bad_value"),
    [
        ("DATABASE_URL", "sqlite:///local.db"),
        ("DATABASE_URL", "not-a-database-url"),
        ("REDIS_URL", "http://localhost:6379/0"),
        ("LLM_PROVIDER", "unsupported-provider"),
        ("OBSERVABILITY_BACKEND", "both"),
        ("MAX_AGENT_RETRIES", "-1"),
        ("MAX_AGENT_RETRIES", "10"),
        ("EMBEDDING_MODEL", "   "),
        ("EMBEDDING_DIMENSIONS", "16"),
    ],
)
def test_invalid_settings_fail_validation(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    bad_value: str,
) -> None:
    monkeypatch.setenv(env_key, bad_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.unit
def test_google_calendar_opt_in_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_GOOGLE_CALENDAR", "true")

    with pytest.raises(ValidationError, match="GOOGLE_CALENDAR_ID"):
        Settings(_env_file=None)

    monkeypatch.setenv("GOOGLE_CALENDAR_ID", "primary")
    with pytest.raises(ValidationError, match="GOOGLE_CALENDAR_ACCESS_TOKEN"):
        Settings(_env_file=None)

    monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "token")
    settings = Settings(_env_file=None)
    assert settings.enable_google_calendar is True
    assert settings.google_calendar_id is not None
    assert settings.google_calendar_access_token is not None
    assert settings.google_calendar_access_token.get_secret_value() == "token"
    # Secrets never render in logs/reprs.
    assert "secret-value" not in repr(settings)
    assert "SecretStr('**********')" in repr(settings)


@pytest.mark.unit
def test_production_requires_llm_api_key_for_hosted_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        Settings(_env_file=None)

    # Local Ollama in production does not require an external LLM API key.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    ollama_settings = Settings(_env_file=None)
    assert ollama_settings.app_env is AppEnv.PRODUCTION
    assert ollama_settings.llm_provider is LLMProvider.OLLAMA


@pytest.mark.unit
def test_database_url_driver_conversion() -> None:
    settings = Settings(
        _env_file=None,
        database_url=PostgresDsn("postgresql://user:pass@localhost:5432/proofpath"),
    )
    assert settings.asyncpg_database_url == (
        "postgresql+asyncpg://user:pass@localhost:5432/proofpath"
    )
    assert settings.psycopg_database_url == "postgresql://user:pass@localhost:5432/proofpath"


@pytest.mark.unit
def test_get_settings_returns_cached_singleton() -> None:
    first = get_settings()
    second = get_settings()
    assert first is second


@pytest.mark.unit
def test_env_example_documents_all_settings_without_real_keys() -> None:
    assert ENV_EXAMPLE_PATH.exists()

    parsed_pairs: dict[str, str] = {}
    for raw_line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        assert sep == "=", f"Malformed line in .env.example: {raw_line!r}"
        parsed_pairs[key.strip()] = value.strip()

    expected_keys = {field_name.upper() for field_name in Settings.model_fields}
    assert expected_keys == set(parsed_pairs)

    # Secret slots must remain blank in `.env.example`.
    assert parsed_pairs["LLM_API_KEY"] == ""
    assert parsed_pairs["GITHUB_TOKEN"] == ""

    # `.env.example` itself must parse cleanly into `Settings`.
    loaded = Settings(_env_file=ENV_EXAMPLE_PATH)
    assert loaded.max_agent_retries == 2
    assert loaded.llm_api_key is None
    assert loaded.github_token is None


@pytest.mark.unit
def test_gitignore_protects_secrets_resumes_and_dumps() -> None:
    assert GITIGNORE_PATH.exists()

    ignored_paths = [
        ".env",
        ".env.local",
        "my_resume.pdf",
        "resumes/profile.txt",
        "data/resumes/cv.md",
        "local_checkpoint.sqlite3",
        "proofpath.db",
        "traces/run_01.json",
        "debug.trace.jsonl",
    ]
    for path in ignored_paths:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"Expected .gitignore to ignore {path!r}"

    # `.env.example` must remain trackable.
    example_check = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert example_check.returncode == 1
