"""Typed async repository for the personal application tracker."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import ApplicationStage, ApplicationTracker


class TrackerRepo:
    """Async CRUD repository for `application_tracker`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        *,
        opportunity_id: uuid.UUID,
        stage: ApplicationStage = ApplicationStage.SAVED,
        notes: str | None = None,
    ) -> ApplicationTracker:
        """Atomically insert or update the tracker entry for `opportunity_id`."""
        now = datetime.now(UTC)
        insert_stmt = pg_insert(ApplicationTracker).values(
            id=uuid.uuid4(),
            opportunity_id=opportunity_id,
            stage=stage,
            notes=notes,
            updated_at=now,
        )
        upsert_stmt = (
            insert_stmt.on_conflict_do_update(
                index_elements=[ApplicationTracker.opportunity_id],
                set_={
                    "stage": insert_stmt.excluded.stage,
                    "notes": insert_stmt.excluded.notes,
                    "updated_at": now,
                },
            )
            .returning(ApplicationTracker)
            .execution_options(populate_existing=True)
        )

        result = await self._session.execute(upsert_stmt)
        await self._session.flush()
        return result.scalar_one()

    async def get_by_opportunity_id(
        self,
        opportunity_id: uuid.UUID,
    ) -> ApplicationTracker | None:
        """Return the tracker record for an opportunity, or `None` if untracked."""
        stmt = select(ApplicationTracker).where(ApplicationTracker.opportunity_id == opportunity_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, tracker_id: uuid.UUID) -> ApplicationTracker | None:
        """Fetch an `ApplicationTracker` row by UUID."""
        stmt = select(ApplicationTracker).where(ApplicationTracker.id == tracker_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_entries(
        self,
        *,
        stage: ApplicationStage | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ApplicationTracker]:
        """List tracked opportunities ordered by most recently updated."""
        stmt = select(ApplicationTracker).order_by(ApplicationTracker.updated_at.desc())
        if stage is not None:
            stmt = stmt.where(ApplicationTracker.stage == stage)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_by_opportunity_id(self, opportunity_id: uuid.UUID) -> bool:
        """Remove a tracker entry for `opportunity_id`."""
        result = await self._session.execute(
            delete(ApplicationTracker)
            .where(ApplicationTracker.opportunity_id == opportunity_id)
            .returning(ApplicationTracker.id)
        )
        return result.scalar_one_or_none() is not None
