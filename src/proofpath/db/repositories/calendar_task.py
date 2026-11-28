"""Typed async repository for local calendar tasks (approved `calendar_block` drafts)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import CalendarTask


class CalendarTaskRepo:
    """Async CRUD repository for `calendar_tasks`, written only by the executor."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_for_draft(
        self,
        *,
        action_draft_id: uuid.UUID,
        opportunity_id: uuid.UUID,
        title: str,
        start_at: datetime,
        end_at: datetime,
        ics_uid: str,
        description: str | None = None,
        timezone: str = "UTC",
    ) -> tuple[CalendarTask, bool]:
        """Insert the task for an approved draft, or return the existing one.

        Idempotent on the `action_draft_id` unique key: executing the same approved
        draft again (retry, crash replay, concurrent approval) returns the originally
        written row with `created=False` instead of duplicating the task.
        """
        insert_stmt = pg_insert(CalendarTask).values(
            id=uuid.uuid4(),
            action_draft_id=action_draft_id,
            opportunity_id=opportunity_id,
            title=title,
            description=description,
            start_at=start_at,
            end_at=end_at,
            timezone=timezone,
            ics_uid=ics_uid,
        )
        insert_result = await self._session.execute(
            insert_stmt.on_conflict_do_nothing(
                index_elements=[CalendarTask.action_draft_id]
            ).returning(CalendarTask.id)
        )
        created = insert_result.scalar_one_or_none() is not None
        await self._session.flush()
        task = await self.get_by_action_draft_id(action_draft_id)
        if task is None:  # pragma: no cover - defensive, unique key guarantees a row
            raise RuntimeError("calendar task insert did not persist a row")
        return task, created

    async def get_by_action_draft_id(self, action_draft_id: uuid.UUID) -> CalendarTask | None:
        """Fetch the calendar task materialized from `action_draft_id`, if any."""
        stmt = select(CalendarTask).where(CalendarTask.action_draft_id == action_draft_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, task_id: uuid.UUID) -> CalendarTask | None:
        """Fetch a `CalendarTask` row by UUID."""
        stmt = select(CalendarTask).where(CalendarTask.id == task_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_tasks(self, *, limit: int = 100, offset: int = 0) -> list[CalendarTask]:
        """List local calendar tasks in chronological order."""
        stmt = (
            select(CalendarTask)
            .order_by(CalendarTask.start_at, CalendarTask.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_by_action_draft_id(self, action_draft_id: uuid.UUID) -> bool:
        """Remove the calendar task materialized from `action_draft_id`, if any."""
        result = await self._session.execute(
            delete(CalendarTask)
            .where(CalendarTask.action_draft_id == action_draft_id)
            .returning(CalendarTask.id)
        )
        return result.scalar_one_or_none() is not None
