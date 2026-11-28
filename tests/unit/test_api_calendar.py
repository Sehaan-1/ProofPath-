"""Calendar invite API (Sub-part 2.1.3b): local task listing and `.ics` download."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from proofpath.api import calendar_routes, routes
from proofpath.calendar_ics import calendar_uid
from proofpath.config import Settings
from proofpath.main import create_app

pytestmark = pytest.mark.unit

START = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)


def _task(
    *, draft_id: uuid.UUID | None = None, title: str = "Interview prep block"
) -> SimpleNamespace:
    draft_id = draft_id or uuid.uuid4()
    return SimpleNamespace(
        id=uuid.uuid4(),
        action_draft_id=draft_id,
        opportunity_id=uuid.uuid4(),
        title=title,
        description="Work through the gap list.",
        start_at=START,
        end_at=START + timedelta(minutes=90),
        timezone="Asia/Kolkata",
        ics_uid=calendar_uid(draft_id=draft_id),
        created_at=datetime(2026, 11, 28, 11, 0, tzinfo=UTC),
    )


@pytest.fixture
def calendar_app(monkeypatch):
    app = create_app(settings=Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None)
    tasks: list[SimpleNamespace] = []

    class Repo:
        def __init__(self, _session: object) -> None:
            pass

        async def list_tasks(self, *, limit: int = 100, offset: int = 0):
            return tasks[offset : offset + limit]

        async def get_by_id(self, task_id: uuid.UUID):
            return next((task for task in tasks if task.id == task_id), None)

    monkeypatch.setattr(calendar_routes, "CalendarTaskRepo", Repo)

    @asynccontextmanager
    async def provide_session():
        yield MagicMock()

    app.dependency_overrides[routes.session_dependency] = lambda: provide_session()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, tasks
    finally:
        client.close()


def test_list_calendar_tasks_reports_invite_urls(calendar_app) -> None:
    client, tasks = calendar_app
    task = _task()
    tasks.append(task)

    response = client.get("/api/v1/calendar-tasks")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == str(task.id)
    assert body[0]["action_draft_id"] == str(task.action_draft_id)
    assert body[0]["ics_uid"] == task.ics_uid
    assert body[0]["timezone"] == "Asia/Kolkata"
    assert body[0]["invite_url"] == f"/api/v1/calendar-tasks/{task.id}/invite.ics"


def test_list_calendar_tasks_rejects_bad_paging(calendar_app) -> None:
    client, _ = calendar_app

    assert client.get("/api/v1/calendar-tasks", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/calendar-tasks", params={"offset": -1}).status_code == 422


def test_invite_download_is_an_ics_attachment(calendar_app) -> None:
    client, tasks = calendar_app
    task = _task(title="Prep; role, at Acme")
    tasks.append(task)

    response = client.get(f"/api/v1/calendar-tasks/{task.id}/invite.ics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/calendar")
    assert response.headers["content-disposition"] == (
        f'attachment; filename="proofpath-{task.id}.ics"'
    )
    body = response.text
    assert body.startswith("BEGIN:VCALENDAR\r\n")
    assert body.endswith("END:VCALENDAR\r\n")
    assert f"UID:{task.ics_uid}" in body
    assert r"SUMMARY:Prep\; role\, at Acme" in body
    assert "DTSTART:20261201T090000Z" in body


def test_downloading_the_same_invite_twice_is_byte_identical(calendar_app) -> None:
    client, tasks = calendar_app
    task = _task()
    tasks.append(task)

    first = client.get(f"/api/v1/calendar-tasks/{task.id}/invite.ics")
    second = client.get(f"/api/v1/calendar-tasks/{task.id}/invite.ics")

    assert first.content == second.content


def test_unknown_calendar_task_is_not_found(calendar_app) -> None:
    client, _ = calendar_app

    response = client.get(f"/api/v1/calendar-tasks/{uuid.uuid4()}/invite.ics")

    assert response.status_code == 404


def test_malformed_task_id_is_a_validation_error(calendar_app) -> None:
    client, _ = calendar_app

    assert client.get("/api/v1/calendar-tasks/not-a-uuid/invite.ics").status_code == 422
