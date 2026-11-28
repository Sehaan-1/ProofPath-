"""Typed application configuration and secrets hygiene for ProofPath."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    """Runtime environment profile."""

    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Standard structured logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(StrEnum):
    """Supported LLM and embedding providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class ObservabilityBackend(StrEnum):
    """Single active observability backend (never run multiple backends in MVP)."""

    NONE = "none"
    LANGFUSE = "langfuse"
    OPENLLMETRY = "openllmetry"


class Settings(BaseSettings):
    """Validated runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # Application profile
    app_env: AppEnv = AppEnv.LOCAL
    log_level: LogLevel = LogLevel.INFO

    # Core infrastructure
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://proofpath:proofpath@localhost:5432/proofpath"),
        description="Async SQLAlchemy PostgreSQL + pgvector connection string.",
    )
    redis_url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"),
        description="Redis connection string for background workers and queues.",
    )

    # LLM & embeddings
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Primary LLM provider identifier.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Default chat/reasoning model identifier.",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the configured LLM provider (masked in logs/reprs).",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="Embedding model used for pgvector indexing and retrieval.",
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=64,
        le=4096,
        description="Vector dimensionality matching EMBEDDING_MODEL.",
    )

    # External ingestion
    github_token: SecretStr | None = Field(
        default=None,
        description="Optional GitHub PAT for higher-rate-limit REST API evidence ingestion.",
    )

    # Approved-only side effects (Sub-part 2.1.3b)
    enable_google_calendar: bool = Field(
        default=False,
        description=(
            "Explicit opt-in for live Google Calendar writes. Defaults to False: an "
            "approved calendar_block draft then only writes the local calendar task and "
            "the downloadable .ics invite, and ProofPath makes no network call."
        ),
    )
    google_calendar_id: SecretStr | None = Field(
        default=None,
        description="Target Google Calendar id (for example 'primary') when opted in.",
    )
    google_calendar_access_token: SecretStr | None = Field(
        default=None,
        description="OAuth access token used for the opt-in Google Calendar import call.",
    )

    # Safety & observability
    observability_backend: ObservabilityBackend = Field(
        default=ObservabilityBackend.NONE,
        description=(
            "Active tracing backend: none, langfuse, or openllmetry. The MVP wires "
            "only Langfuse (Sub-part 2.4.1a); openllmetry fails closed if selected."
        ),
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        description="Langfuse project public key; required when OBSERVABILITY_BACKEND=langfuse.",
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        description="Langfuse project secret key (masked in logs/reprs).",
    )
    langfuse_host: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://cloud.langfuse.com"),
        description=(
            "Langfuse base URL (cloud or self-hosted). Trace payloads contain run "
            "state, so use a trusted project."
        ),
    )
    max_agent_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Hard upper bound on LangGraph policy/tool retry loops.",
    )
    enable_guardrails: bool = Field(
        default=False,
        description="Also check final output with the local Guardrails AI validator (guardrails extra).",
    )
    redact_pii: bool = Field(
        default=False,
        description="Mask personal phone numbers/addresses before external LLM calls.",
    )

    # Background worker retry & dead-letter policy (Sub-slice 2.2.3)
    job_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum processing attempts for a transiently-failing background job "
            "before it is moved to the dead-letter state."
        ),
    )
    job_retry_backoff_seconds: list[int] = Field(
        default=[2, 8, 30],
        description=(
            "Exponential backoff delays (seconds) applied before re-enqueuing a "
            "transiently-failed job, indexed by the failed attempt number. "
            "Parsed from the environment as a JSON array, e.g. [2, 8, 30]."
        ),
    )

    @field_validator(
        "llm_api_key",
        "github_token",
        "google_calendar_id",
        "google_calendar_access_token",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _empty_secret_to_none(cls, value: Any) -> Any:
        """Treat blank environment values (e.g. `GITHUB_TOKEN=`) as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("app_env", "llm_provider", "observability_backend", mode="before")
    @classmethod
    def _normalize_lowercase_enum(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_uppercase_enum(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> Settings:
        if (
            self.app_env is AppEnv.PRODUCTION
            and self.llm_provider is not LLMProvider.OLLAMA
            and self.llm_api_key is None
        ):
            raise ValueError(
                "LLM_API_KEY must be set when APP_ENV=production for hosted LLM providers."
            )
        return self

    @model_validator(mode="after")
    def _validate_langfuse_opt_in(self) -> Settings:
        if self.observability_backend is ObservabilityBackend.LANGFUSE and (
            self.langfuse_public_key is None or self.langfuse_secret_key is None
        ):
            raise ValueError(
                "OBSERVABILITY_BACKEND=langfuse requires LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY."
            )
        return self

    @model_validator(mode="after")
    def _validate_google_calendar_opt_in(self) -> Settings:
        if self.enable_google_calendar and (
            self.google_calendar_id is None or self.google_calendar_access_token is None
        ):
            raise ValueError(
                "ENABLE_GOOGLE_CALENDAR=true requires GOOGLE_CALENDAR_ID and "
                "GOOGLE_CALENDAR_ACCESS_TOKEN."
            )
        return self

    @property
    def asyncpg_database_url(self) -> str:
        """Return the PostgreSQL URI formatted for SQLAlchemy + asyncpg."""
        raw = str(self.database_url)
        if raw.startswith("postgresql+asyncpg://"):
            return raw
        if raw.startswith("postgresql+psycopg://"):
            return raw.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgresql://"):
            return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgres://"):
            return raw.replace("postgres://", "postgresql+asyncpg://", 1)
        return raw

    @property
    def psycopg_database_url(self) -> str:
        """Return a standard `postgresql://` URI for psycopg / LangGraph checkpointer."""
        raw = str(self.database_url)
        for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://", "postgres://"):
            if raw.startswith(prefix):
                return raw.replace(prefix, "postgresql://", 1)
        return raw

    def require_llm_api_key(self) -> str:
        """Return the unwrapped LLM API key or fail clearly if unconfigured."""
        if self.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required for live LLM and embedding requests.")
        return self.llm_api_key.get_secret_value()

    def require_github_token(self) -> str:
        """Return the unwrapped GitHub token or fail clearly if unconfigured."""
        if self.github_token is None:
            raise ValueError("GITHUB_TOKEN is required for authenticated GitHub ingestion.")
        return self.github_token.get_secret_value()

    def require_langfuse_keys(self) -> tuple[str, str]:
        """Return the unwrapped Langfuse (public, secret) keys or fail clearly."""
        if self.langfuse_public_key is None or self.langfuse_secret_key is None:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for Langfuse tracing."
            )
        return (
            self.langfuse_public_key.get_secret_value(),
            self.langfuse_secret_key.get_secret_value(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance for dependency injection."""
    return Settings()
