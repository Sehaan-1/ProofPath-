"""Task handler protocol and registry for the ProofPath background worker.

Sub-part 2.2.1b: the worker dequeues a Redis stream message, correlates it to a
`background_jobs` row, and dispatches by ``job_type`` to a registered handler.
Handlers are plain async callables that close over their own dependencies (a
session factory, settings, an LLM client), so the worker stays decoupled from
domain concerns. Real task registrations (``github_ingest``, ``analysis_run``)
arrive in Sub-slice 2.2.2; this module provides the protocol, registry, and two
built-in smoke-test tasks.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final


#: A handler returns ``ok`` with an optional structured ``result`` (persisted onto
#: the job row), or an ``error`` with ``transient`` indicating whether the worker
#: should retry (``True``) or fail straight to ``dead_letter`` (``False``). An
#: optional ``traceback`` is captured for unhandled handler exceptions so the
#: dead-letter row carries enough to diagnose the failure.
@dataclass(frozen=True, slots=True)
class JobResult:
    """Outcome of a single task-handler invocation."""

    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    transient: bool = True
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class JobContext:
    """Everything a task handler needs to do its work for one attempt."""

    job_id: uuid.UUID
    job_type: str
    payload: dict[str, Any]
    attempt: int


TaskHandler = Callable[[JobContext], Awaitable[JobResult]]

#: Lowercase snake-case identifiers, 1..64 chars. Enforced so a hallucinated or
#: mistyped job type fails fast instead of silently enqueuing undeliverable work.
_JOB_TYPE_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

#: Built-in job type names kept here so callers and tests can reference them.
NOOP_TASK: Final[str] = "noop"
ECHO_TASK: Final[str] = "echo"


def validate_job_type(job_type: str) -> str:
    """Return a normalized, valid job type or raise ``ValueError``."""
    if not _JOB_TYPE_RE.fullmatch(job_type):
        raise ValueError(f"job_type must match '^[a-z][a-z0-9_]{{0,63}}$' (got {job_type!r}).")
    return job_type


class TaskRegistry:
    """Maps job type strings to async handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, job_type: str, handler: TaskHandler) -> None:
        """Register a handler for a job type, rejecting duplicate bindings."""
        name = validate_job_type(job_type)
        if name in self._handlers:
            raise ValueError(f"job_type {name!r} is already registered.")
        self._handlers[name] = handler

    def get(self, job_type: str) -> TaskHandler | None:
        """Return the handler for a job type, or ``None`` if unregistered."""
        return self._handlers.get(job_type)

    def __contains__(self, job_type: object) -> bool:
        return isinstance(job_type, str) and job_type in self._handlers

    def known_types(self) -> frozenset[str]:
        """Return all registered job types."""
        return frozenset(self._handlers)


async def noop_handler(_ctx: JobContext) -> JobResult:
    """Built-in health-check task: succeeds immediately with no result.

    Useful for smoke-testing the full queue -> worker -> ledger loop without
    depending on any external service or LLM.
    """
    return JobResult(ok=True)


async def echo_handler(ctx: JobContext) -> JobResult:
    """Built-in smoke-test task: succeeds and echoes the payload as the result."""
    return JobResult(ok=True, result={"job_type": ctx.job_type, "payload": ctx.payload})


def default_registry() -> TaskRegistry:
    """Return a registry pre-loaded with the built-in smoke-test tasks.

    Sub-slice 2.2.2 extends this (or builds its own) with the real
    ``github_ingest`` and ``analysis_run`` handlers.
    """
    registry = TaskRegistry()
    registry.register(NOOP_TASK, noop_handler)
    registry.register(ECHO_TASK, echo_handler)
    return registry
