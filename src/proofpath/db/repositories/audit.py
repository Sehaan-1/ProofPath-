"""Append-only async repository for immutable audit logs."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import AuditActor, AuditLog


class AuditRepo:
    """Append-only repository for `audit_logs` (no update/delete operations)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        actor: AuditActor,
        event_type: str,
        resource_id: uuid.UUID | None = None,
        details_json: dict[str, Any] | None = None,
        log_id: uuid.UUID | None = None,
    ) -> AuditLog:
        """Append an immutable audit log entry and flush."""
        entry = AuditLog(
            id=log_id or uuid.uuid4(),
            actor=actor,
            event_type=event_type,
            resource_id=resource_id,
            details_json=dict(details_json) if details_json is not None else {},
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def list_events(
        self,
        *,
        actor: AuditActor | None = None,
        event_type: str | None = None,
        resource_id: uuid.UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Query immutable audit log entries ordered newest-first."""
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc())
        if actor is not None:
            stmt = stmt.where(AuditLog.actor == actor)
        if event_type is not None:
            stmt = stmt.where(AuditLog.event_type == event_type)
        if resource_id is not None:
            stmt = stmt.where(AuditLog.resource_id == resource_id)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
