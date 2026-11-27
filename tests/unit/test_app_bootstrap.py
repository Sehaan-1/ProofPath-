"""FastAPI lifecycle, health and problem-details response tests."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from proofpath import main
from proofpath.config import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def services(monkeypatch):
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = MagicMock()
    factory = MagicMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = session_context
    saver = object()
    events = []

    @asynccontextmanager
    async def fake_checkpointer(*, settings):
        events.append(("start", settings))
        try:
            yield saver
        finally:
            events.append(("stop", settings))

    monkeypatch.setattr(main, "create_db_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(main, "create_session_factory", lambda _engine: factory)
    monkeypatch.setattr(main, "postgres_checkpointer", fake_checkpointer)
    monkeypatch.setattr(main, "verify_pgvector_extension", AsyncMock(return_value=True))
    return engine, factory, session, saver, events


def assert_problem(response, status):
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"] == "about:blank"
    assert body["status"] == status
    assert body["title"]
    assert body["detail"]
    return body


def test_lifespan_initializes_health_and_closes_resources(services):
    engine, factory, session, saver, events = services
    settings = Settings(_env_file=None)
    app = main.create_app(settings=settings)
    with TestClient(app) as client:
        assert app.state.db_engine is engine
        assert app.state.session_factory is factory
        assert app.state.checkpointer is saver
        assert app.state.settings is settings
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "postgres": "ok", "pgvector": "ok"}
        main.verify_pgvector_extension.assert_awaited_once_with(session)
        engine.dispose.assert_not_awaited()
    engine.dispose.assert_awaited_once()
    assert events == [("start", settings), ("stop", settings)]
    assert not hasattr(app.state, "checkpointer")
    assert not hasattr(app.state, "db_engine")


def test_startup_failure_disposes_engine(monkeypatch, services):
    engine, _, _, _, _ = services

    @asynccontextmanager
    async def fail_checkpointer(*, settings):
        raise RuntimeError("database unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(main, "postgres_checkpointer", fail_checkpointer)
    app = main.create_app(settings=Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="database unavailable"), TestClient(app):
        pass
    engine.dispose.assert_awaited_once()
    assert not hasattr(app.state, "checkpointer")


def test_health_failure_and_missing_extension_are_problem_details(monkeypatch, services):
    app = main.create_app(settings=Settings(_env_file=None))
    with TestClient(app) as client:
        monkeypatch.setattr(main, "verify_pgvector_extension", AsyncMock(return_value=False))
        body = assert_problem(client.get("/health?token=secret"), 503)
        assert body["detail"] == "PostgreSQL pgvector extension is unavailable."
        assert body["instance"] == "/health"
        assert "secret" not in str(body)
        monkeypatch.setattr(
            main,
            "verify_pgvector_extension",
            AsyncMock(side_effect=RuntimeError("postgres://password@host")),
        )
        body = assert_problem(client.get("/health"), 503)
        assert body["detail"] == "PostgreSQL health check failed."
        assert "password" not in str(body)


def test_http_validation_and_unhandled_errors_are_problems(services):
    app = main.create_app(settings=Settings(_env_file=None))

    @app.get("/protected")
    async def protected():
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.get("/validate")
    async def validate(number: int):
        return {"number": number}

    @app.get("/failure")
    async def failure():
        raise RuntimeError("private secret")

    @app.get("/invalid-response", response_model=int)
    async def invalid_response():
        return "private secret"

    with TestClient(app, raise_server_exceptions=False) as client:
        body = assert_problem(client.get("/missing?password=secret"), 404)
        assert body["instance"] == "/missing"
        assert "secret" not in str(body)
        response = client.get("/protected")
        assert assert_problem(response, 401)["detail"] == "Authentication required"
        assert response.headers["www-authenticate"] == "Bearer"
        body = assert_problem(client.get("/validate?number=private-secret"), 422)
        assert body["errors"][0]["location"] == ["query", "number"]
        assert "private-secret" not in str(body)
        for path in ("/failure", "/invalid-response"):
            body = assert_problem(client.get(path), 500)
            assert body["detail"] == "An internal server error occurred."
            assert "secret" not in str(body)


def test_health_without_lifespan_reports_not_ready():
    app = main.create_app(settings=Settings(_env_file=None))
    # Deliberately avoid entering TestClient's context, so lifespan is not run.
    client = TestClient(app)
    body = assert_problem(client.get("/health"), 503)
    assert body["detail"] == "Application infrastructure is not ready."
    client.close()


def test_shutdown_disposes_engine_even_when_saver_close_fails(monkeypatch, services):
    engine, _, _, _, _ = services

    @asynccontextmanager
    async def broken_close(*, settings):
        yield object()
        raise RuntimeError("failed to close")

    monkeypatch.setattr(main, "postgres_checkpointer", broken_close)
    app = main.create_app(settings=Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="failed to close"), TestClient(app):
        pass
    engine.dispose.assert_awaited_once()
    assert not hasattr(app.state, "checkpointer")


def test_client_error_with_no_body_preserves_protocol_headers(services):
    app = main.create_app(settings=Settings(_env_file=None))

    @app.get("/empty")
    async def empty():
        raise HTTPException(status_code=304, headers={"ETag": '"v1"'})

    with TestClient(app) as client:
        response = client.get("/empty")
        assert response.status_code == 304
        assert response.content == b""
        assert response.headers["etag"] == '"v1"'
