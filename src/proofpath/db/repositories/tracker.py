"""Typed async repository for the personal application tracker."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import case, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import ApplicationStage, ApplicationTracker

# Forward-only lifecycle: an approved draft may advance an entry, never regress it.
_STAGE_ORDER: dict[str, int] = {
    ApplicationStage.SAVED.value: 0,
    ApplicationStage.PREPARING.value: 1,
    ApplicationStage.READY_TO_APPLY.value: 2,
    ApplicationStage.APPLIED.value: 3,
    ApplicationStage.INTERVIEWING.value: 4,
    ApplicationStage.ARCHIVED.value: 5,
}


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

    async def advance(
        self,
        *,
        opportunity_id: uuid.UUID,
        stage: ApplicationStage,
        notes: str | None = None,
    ) -> ApplicationTracker:
        """Atomically upsert a tracker entry without regressing stage or dropping notes.

        Used only by the approved-only executor (Sub-part 2.1.3a). One statement keeps
        the read-modify-write race-free: on conflict the stored stage only moves
        forward along `SAVED -> PREPARING -> READY_TO_APPLY -> APPLIED -> INTERVIEWING
        -> ARCHIVED`, and existing notes survive unless new ones are supplied.
        """
        now = datetime.now(UTC)
        insert_stmt = pg_insert(ApplicationTracker).values(
            id=uuid.uuid4(),
            opportunity_id=opportunity_id,
            stage=stage,
            notes=notes,
            updated_at=now,
        )
        stored_rank = case(_STAGE_ORDER, value=ApplicationTracker.stage, else_=0)
        proposed_rank = case(_STAGE_ORDER, value=insert_stmt.excluded.stage, else_=0)
        upsert_stmt = (
            insert_stmt.on_conflict_do_update(
                index_elements=[ApplicationTracker.opportunity_id],
                set_={
                    "stage": case(
                        (stored_rank > proposed_rank, ApplicationTracker.stage),
                        else_=insert_stmt.excluded.stage,
                    ),
                    "notes": func.coalesce(insert_stmt.excluded.notes, ApplicationTracker.notes),
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
