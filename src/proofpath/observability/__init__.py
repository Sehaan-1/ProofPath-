"""Single-backend observability wiring for ProofPath (Sub-slice 2.4.1).

Sub-part 2.4.1a picks **Langfuse** as the one MVP tracing backend; OpenLLMetry
is deliberately not wired so the two never run simultaneously. Sub-part 2.4.1b
attaches a LangGraph callback handler through :mod:`proofpath.observability.tracing`
so every analysis run — including approval resumes — emits one unified trace ID.
"""

from proofpath.observability.tracing import (
    ANALYSIS_TRACE_NAME,
    NullRunTracer,
    RunTracer,
    TracingUnavailableError,
    analysis_trace_id,
    create_run_tracer,
    shutdown_tracing,
)

__all__ = [
    "ANALYSIS_TRACE_NAME",
    "NullRunTracer",
    "RunTracer",
    "TracingUnavailableError",
    "analysis_trace_id",
    "create_run_tracer",
    "shutdown_tracing",
]
