"""Downloadable calendar invites for approved `calendar_block` drafts (Sub-part 2.1.3b).

The `.ics` file is generated on demand from the persisted `calendar_tasks` row, which
only the approved-only executor writes. Downloads are deterministic: the same task
always yields the same `UID` and byte content, so importing it twice updates the same
calendar event instead of creating a duplicate.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response

from proofpath.api.routes import Session
from proofpath.db.models import CalendarTask
from proofpath.db.repositories.calendar_task import CalendarTaskRepo
from proofpath.execution import calendar_invite_for_task
from proofpath.schemas.actions import CalendarTaskRecord

router = APIRouter(prefix="/api/v1", tags=["calendar"])

_ICS_MEDIA_TYPE = "text/calendar; charset=utf-8"


def calendar_task_record(task: CalendarTask) -> CalendarTaskRecord:
    """Map a persisted calendar task row to its API contract."""
    return CalendarTaskRecord(
        id=task.id,
        action_draft_id=task.action_draft_id,
        opportunity_id=task.opportunity_id,
        title=task.title,
        description=task.description,
        start_at=task.start_at,
        end_at=task.end_at,
        timezone=task.timezone,
        ics_uid=task.ics_uid,
        created_at=task.created_at,
        invite_url=f"/api/v1/calendar-tasks/{task.id}/invite.ics",
    )


@router.get("/calendar-tasks", response_model=list[CalendarTaskRecord])
async def list_calendar_tasks(
    session: Session,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[CalendarTaskRecord]:
    """List local calendar tasks materialized from approved drafts, chronologically."""
    rows = await CalendarTaskRepo(session).list_tasks(limit=limit, offset=offset)
    return [calendar_task_record(row) for row in rows]


@router.get(
    "/calendar-tasks/{task_id}/invite.ics",
    response_class=Response,
    responses={200: {"content": {_ICS_MEDIA_TYPE: {}}}},
)
async def download_calendar_invite(task_id: uuid.UUID, session: Session) -> Response:
    """Download the `.ics` invite for one approved calendar block as an attachment."""
    task = await CalendarTaskRepo(session).get_by_id(task_id)
    if task is None:
        raise HTTPException(404, "Calendar task not found.")
    return Response(
        content=calendar_invite_for_task(task),
        media_type=_ICS_MEDIA_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="proofpath-{task.id}.ics"',
            "Cache-Control": "no-store",
        },
    )
