"""Opt-in Langfuse adapter (Sub-part 2.4.1a). No import-time SDK dependency.

The Langfuse Python SDK (v3, OpenTelemetry-based) and ``langchain`` ship only in
the isolated ``proofpath[observability]`` extra. Everything is imported lazily so
the core runtime never pulls OpenTelemetry, mirroring the Guardrails adapter.

One cached SDK client per process (the SDK's resource manager is itself a
singleton keyed by public key); one fresh ``CallbackHandler`` per invocation,
bound to the run's unified trace ID via ``trace_context``. Trace-level
attributes (name, session, run metadata) are propagated with the SDK's
``propagate_attributes`` context manager so every span in the run carries them.

Trace payloads contain graph state (job text, evidence snippets, verdicts), so
point ``LANGFUSE_HOST`` at a trusted or self-hosted Langfuse project.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from proofpath import __version__
from proofpath.config import Settings
from proofpath.observability.tracing import (
    ANALYSIS_TRACE_NAME,
    TracingUnavailableError,
    analysis_trace_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# Process-level client cache. Settings are fixed per process in production; the
# cache key lets tests (or a reconfiguration) rebuild cleanly instead of reusing
# a stale SDK singleton.
_client: Langfuse | None = None
_client_key: tuple[str, str, str] | None = None
_lock = threading.Lock()


def _import_sdk() -> tuple[
    type[Langfuse], type[CallbackHandler], Callable[..., AbstractContextManager[Any]]
]:
    """Import the extra lazily; a missing SDK is a clear configuration error.

    ``langfuse.langchain`` also requires ``langchain`` itself; both are in the
    extra, and either absence surfaces as the same sanitized error.
    """
    try:
        from langfuse import Langfuse, propagate_attributes
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise TracingUnavailableError(
            "OBSERVABILITY_BACKEND=langfuse requires the proofpath[observability] extra."
        ) from exc
    return Langfuse, CallbackHandler, propagate_attributes


def _reset_sdk_resources() -> None:
    """Shut down *and unregister* the SDK's singleton resources.

    ``Langfuse.shutdown()`` alone leaves the resource manager registered under
    its public key, so a later client with the same key would reuse stopped
    export threads. ``LangfuseResourceManager.reset()`` (verified against the
    pinned 3.x SDK) flushes, stops and clears every instance; if that internal
    API ever moves, fall back to the public per-client shutdown.
    """
    try:
        from langfuse._client.resource_manager import LangfuseResourceManager
    except ImportError:  # pragma: no cover - defensive against SDK drift
        if _client is not None:
            _client.shutdown()
        return
    LangfuseResourceManager.reset()


def _get_or_create_client(settings: Settings, langfuse_cls: type[Langfuse]) -> Langfuse:
    """Return the cached client, constructing it on first use or config change."""
    global _client, _client_key
    public_key, secret_key = settings.require_langfuse_keys()
    host = str(settings.langfuse_host).rstrip("/")
    environment = settings.app_env.value
    key = (public_key, host, environment)
    with _lock:
        if _client is not None and _client_key == key:
            return _client
        if _client is not None:
            _reset_sdk_resources()
            # Never leave a shut-down client reachable if reconstruction fails.
            _client = None
            _client_key = None
        _client = langfuse_cls(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=environment,
            release=__version__,
        )
        _client_key = key
        return _client


class LangfuseRunTracer:
    """Langfuse backend bound to one analysis run and its LangGraph thread.

    Construction is cheap and offline (no auth/network call): it validates the
    extra and keys, reuses the cached SDK client, and derives the unified trace
    ID. Export happens in the background and via ``flush()`` after each
    invocation, so a run's spans are durable even in a short-lived process.
    """

    def __init__(self, *, settings: Settings, analysis_run_id: UUID, thread_id: str) -> None:
        langfuse_cls, handler_cls, propagate_attributes = _import_sdk()
        self._client = _get_or_create_client(settings, langfuse_cls)
        self._handler_cls = handler_cls
        self._propagate_attributes = propagate_attributes
        self._analysis_run_id = analysis_run_id
        self._thread_id = thread_id
        self._trace_id = analysis_trace_id(analysis_run_id)

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @contextmanager
    def invocation(self) -> Iterator[Sequence[BaseCallbackHandler]]:
        """Yield a fresh handler bound to the unified trace for one invocation.

        ``trace_context`` makes the handler's root span join the run's trace;
        on an approval resume a second root span joins the *same* trace.
        ``update_trace`` copies the root run's input/output onto the trace, and
        ``propagate_attributes`` stamps every span with the trace name, the
        LangGraph thread as session, and non-secret run identifiers.
        """
        handler = self._handler_cls(
            trace_context={"trace_id": self._trace_id},
            update_trace=True,
        )
        with self._propagate_attributes(
            trace_name=ANALYSIS_TRACE_NAME,
            session_id=self._thread_id,
            metadata={
                "analysis_run_id": str(self._analysis_run_id),
                "thread_id": self._thread_id,
            },
            tags=["proofpath"],
        ):
            yield (handler,)

    def annotate_current_span(self, *, metadata: Mapping[str, object]) -> None:
        """Stamp the active node span (the callback handler's span) with metadata.

        During a node's execution the LangChain handler has attached that node's
        span to the OTel context, so ``update_current_span`` targets it. Failures
        are logged, never raised: observability must not fail a run.
        """
        try:
            self._client.update_current_span(metadata=dict(metadata))
        except Exception as exc:
            logger.warning("Langfuse span annotation failed (%s).", type(exc).__name__)

    async def flush(self) -> None:
        """Force-export buffered spans off the event loop; never raises."""
        try:
            await asyncio.to_thread(self._client.flush)
        except Exception as exc:
            logger.warning("Langfuse flush failed (%s); run unaffected.", type(exc).__name__)


def shutdown_langfuse() -> None:
    """Flush, stop export threads, and clear the cache (called at shutdown).

    No-op when Langfuse was never used. Failures are logged, never raised: app
    teardown must complete even if the trace backend is unreachable.
    """
    global _client, _client_key
    with _lock:
        if _client is None:
            return
        try:
            _reset_sdk_resources()
        except Exception as exc:
            logger.warning("Langfuse shutdown failed (%s).", type(exc).__name__)
        finally:
            _client = None
            _client_key = None


__all__ = ["LangfuseRunTracer", "shutdown_langfuse"]
