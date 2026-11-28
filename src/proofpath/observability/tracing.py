"""Single observability backend wiring for analysis runs (Sub-slice 2.4.1).

**Backend choice (Sub-part 2.4.1a):** Langfuse is the one MVP backend — fast
trace UI plus prompt/cost correlation. ``OBSERVABILITY_BACKEND=openllmetry``
remains a recognized configuration value but is intentionally *not* wired:
selecting it fails closed with a clear error instead of silently running a
second (or missing) backend. Never run both simultaneously in MVP.

**Unified trace ID (Sub-part 2.4.1b):** every analysis run derives one
deterministic trace ID from its ``analysis_run_id``. The initial graph
invocation and every later approval resume on the same run attach a fresh
LangGraph callback handler bound to that same ID, so the whole run — across
processes and the human approval pause — lands in one trace.

The Langfuse SDK lives in the isolated ``proofpath[observability]`` extra and is
imported lazily by :mod:`proofpath.observability.langfuse_tracing`, mirroring the
Guardrails adapter, so OpenTelemetry dependencies never touch the core runtime.
With ``OBSERVABILITY_BACKEND=none`` (the default) this module adds no callbacks,
performs no I/O, and never imports the SDK.

Span hierarchy detail and persisted token/cost metrics are Sub-slice 2.4.2 and
are deliberately out of scope here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from proofpath.config import ObservabilityBackend, Settings

#: Trace/root-span name for one analysis run (the future 2.4.2 parent span).
ANALYSIS_TRACE_NAME = "analysis_run"


class TracingUnavailableError(RuntimeError):
    """The configured observability backend cannot be wired.

    Messages are sanitized: they name the configuration fix, never credentials.
    """


def analysis_trace_id(analysis_run_id: UUID) -> str:
    """Return the unified, deterministic trace ID for one analysis run.

    The derivation is byte-for-byte compatible with
    ``Langfuse.create_trace_id(seed=str(analysis_run_id))`` (SHA-256, first 16
    bytes, lowercase hex — a valid 32-character W3C/OTel trace ID). Computing it
    here keeps the core runtime free of the Langfuse import while guaranteeing
    that the ID we hand the callback handler is exactly the ID Langfuse shows.
    The same run always maps to the same trace, so an approval resumed in a
    different process still joins the original trace.
    """
    return sha256(str(analysis_run_id).encode("utf-8")).digest()[:16].hex()


class RunTracer(Protocol):
    """One observability backend bound to one analysis run.

    Implementations must be safe to construct per graph build: the underlying
    SDK client is a cached singleton, while each ``invocation()`` yields fresh
    callback handlers (handlers carry per-invocation state and must never be
    shared across concurrent runs).
    """

    @property
    def trace_id(self) -> str:
        """Unified trace ID shared by the initial invocation and every resume."""
        ...

    def invocation(self) -> AbstractContextManager[Sequence[BaseCallbackHandler]]:
        """Wrap one graph invocation, yielding the callbacks to attach to it.

        The context manager activates any backend-side trace attributes for the
        duration of the invocation and must be exited before ``flush()``.
        """
        ...

    def annotate_current_span(self, *, metadata: Mapping[str, object]) -> None:
        """Attach metadata to the currently active node span (Sub-part 2.4.2a).

        Called from within a node's execution (e.g. retrieval chunk counts and
        similarity scores on ``retrieve_evidence``). A no-op when untraced or no
        span is active; never raises into the run.
        """
        ...

    async def flush(self) -> None:
        """Export buffered spans for the finished invocation; never raises.

        Tracing is auxiliary: an export failure must not fail or mask the result
        of an analysis run (configuration failures, by contrast, fail closed at
        tracer construction time).
        """
        ...


class NullRunTracer:
    """No-op tracer for ``OBSERVABILITY_BACKEND=none`` (the default).

    The unified trace ID is still derived so callers can correlate a run later
    (Sub-slice 2.4.2 persists it); no callbacks are attached and no I/O happens.
    """

    def __init__(self, *, trace_id: str) -> None:
        self._trace_id = trace_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def invocation(self) -> AbstractContextManager[Sequence[BaseCallbackHandler]]:
        return nullcontext(())

    def annotate_current_span(self, *, metadata: Mapping[str, object]) -> None:  # noqa: ARG002
        return None

    async def flush(self) -> None:
        return None


def create_run_tracer(
    settings: Settings,
    *,
    analysis_run_id: UUID,
    thread_id: str,
) -> RunTracer:
    """Build the single configured tracer for one run, or fail closed.

    ``none`` returns the no-op tracer without importing any SDK. ``langfuse``
    lazily loads the isolated adapter (requires the ``proofpath[observability]``
    extra and ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``). ``openllmetry``
    is rejected: the MVP wires exactly one backend and that backend is Langfuse.
    """
    backend = settings.observability_backend
    if backend is ObservabilityBackend.NONE:
        return NullRunTracer(trace_id=analysis_trace_id(analysis_run_id))
    if backend is ObservabilityBackend.LANGFUSE:
        from proofpath.observability.langfuse_tracing import LangfuseRunTracer

        return LangfuseRunTracer(
            settings=settings, analysis_run_id=analysis_run_id, thread_id=thread_id
        )
    raise TracingUnavailableError(
        "OBSERVABILITY_BACKEND=openllmetry is not wired in the MVP. "
        "Set OBSERVABILITY_BACKEND=langfuse (recommended) or none."
    )


async def shutdown_tracing() -> None:
    """Flush and release the tracing backend at process/app shutdown.

    Safe to call when tracing was never enabled or the SDK was never imported:
    the adapter module itself has no import-time SDK dependency.
    """
    from proofpath.observability.langfuse_tracing import shutdown_langfuse

    await asyncio.to_thread(shutdown_langfuse)
