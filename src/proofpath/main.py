"""FastAPI application bootstrap, health checks, and problem responses.

The checkpointer owns a PostgreSQL connection for the entire application
lifespan. The application must be started only after database migrations have
been applied. Startup fails closed if checkpoint initialization fails.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from proofpath.api.analysis_routes import router as analysis_router
from proofpath.api.calendar_routes import router as calendar_router
from proofpath.api.draft_routes import router as draft_router
from proofpath.api.jobs_routes import router as jobs_router
from proofpath.api.routes import router
from proofpath.config import Settings, get_settings
from proofpath.db.session import create_db_engine, create_session_factory, verify_pgvector_extension
from proofpath.graph.checkpointer import postgres_checkpointer
from proofpath.observability.tracing import shutdown_tracing
from proofpath.workers.queue import JobQueue

logger = logging.getLogger(__name__)
_HEALTH_TIMEOUT_SECONDS = 5.0


def _problem(
    request: Request,
    status: int,
    detail: str,
    *,
    headers: Mapping[str, str] | None = None,
    extensions: dict[str, Any] | None = None,
) -> JSONResponse:
    """Render RFC-7807 problem details without including query-string secrets."""
    try:
        title = HTTPStatus(status).phrase
    except ValueError:
        title = "HTTP Error"
    content: dict[str, Any] = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
    }
    if extensions:
        content.update(extensions)
    return JSONResponse(
        status_code=status,
        content=content,
        media_type="application/problem+json",
        headers=headers,
    )


def create_app(*, settings: Settings | None = None) -> FastAPI:
    """Create an isolated application; infrastructure is initialized on startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_settings = settings if settings is not None else get_settings()
        engine = create_db_engine(settings=active_settings)
        # Redis client is lazy: `from_url` does not connect until a command runs,
        # so the API starts even when Redis is down; only the async enqueue paths
        # (which need it) will surface a 503 when Redis is unreachable.
        job_queue = JobQueue.from_url(str(active_settings.redis_url))
        try:
            session_factory = create_session_factory(engine)
            async with postgres_checkpointer(settings=active_settings) as saver:
                application.state.settings = active_settings
                application.state.db_engine = engine
                application.state.session_factory = session_factory
                application.state.checkpointer = saver
                application.state.job_queue = job_queue
                try:
                    yield
                finally:
                    await job_queue.close()
                    # Flush/release the tracing backend (no-op unless enabled).
                    await shutdown_tracing()
                    del application.state.job_queue
                    del application.state.checkpointer
                    del application.state.session_factory
                    del application.state.db_engine
                    del application.state.settings
        finally:
            await engine.dispose()

    application = FastAPI(title="ProofPath", version="0.1.0", lifespan=lifespan)

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code in (204, 304) or 100 <= exc.status_code < 200:
            return Response(status_code=exc.status_code, headers=exc.headers)
        detail = exc.detail if isinstance(exc.detail, str) and exc.status_code < 500 else None
        return _problem(
            request,
            exc.status_code,
            detail
            or (
                "The request could not be completed."
                if exc.status_code < 500
                else "Service unavailable."
            ),
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo submitted values, validator context, or custom messages.
        errors = [
            {"location": list(item.get("loc", ())), "code": item.get("type", "invalid")}
            for item in exc.errors()
        ]
        return _problem(request, 422, "Request validation failed.", extensions={"errors": errors})

    @application.exception_handler(ResponseValidationError)
    async def response_validation_handler(
        request: Request, _exc: ResponseValidationError
    ) -> JSONResponse:
        logger.error("Response validation failed")
        return _problem(request, 500, "An internal server error occurred.")

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Avoid logging raw exception messages, which can contain credentials or PII.
        logger.error("Unhandled request failure: %s", type(exc).__name__)
        return _problem(request, 500, "An internal server error occurred.")

    @application.get(
        "/health",
        tags=["health"],
        response_model=None,
        responses={503: {"description": "Database or pgvector unavailable"}},
    )
    async def health(request: Request) -> dict[str, str] | JSONResponse:
        """Check database reachability and the PostgreSQL vector extension."""
        factory = getattr(request.app.state, "session_factory", None)
        if factory is None:
            return _problem(request, 503, "Application infrastructure is not ready.")
        try:
            async with asyncio.timeout(_HEALTH_TIMEOUT_SECONDS), factory() as session:
                has_vector = await verify_pgvector_extension(session)
        except Exception:
            return _problem(request, 503, "PostgreSQL health check failed.")
        if not has_vector:
            return _problem(request, 503, "PostgreSQL pgvector extension is unavailable.")
        return {"status": "ok", "postgres": "ok", "pgvector": "ok"}

    application.include_router(router)
    application.include_router(analysis_router)
    application.include_router(draft_router)
    application.include_router(calendar_router)
    application.include_router(jobs_router)
    return application


app = create_app()
