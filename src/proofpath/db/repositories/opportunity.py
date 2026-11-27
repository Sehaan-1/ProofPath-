"""Typed async repository for target opportunities and parsed role requirements."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from proofpath.db.models import (
    Opportunity,
    RequirementCategory,
    RequirementPriority,
    RoleRequirement,
)


@dataclass(frozen=True, slots=True)
class RoleRequirementInput:
    """Typed input payload for inserting a `RoleRequirement`."""

    requirement_text: str
    category: RequirementCategory
    priority: RequirementPriority
    requirement_id: uuid.UUID | None = None


class OpportunityRepo:
    """Async CRUD repository for `opportunities` and `role_requirements`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_opportunity(
        self,
        *,
        company: str,
        role_title: str,
        raw_text: str,
        sanitized_text: str,
        source_url: str | None = None,
        opportunity_id: uuid.UUID | None = None,
    ) -> Opportunity:
        """Persist a new target `Opportunity` and flush."""
        opportunity = Opportunity(
            id=opportunity_id or uuid.uuid4(),
            company=company,
            role_title=role_title,
            source_url=source_url,
            raw_text=raw_text,
            sanitized_text=sanitized_text,
        )
        self._session.add(opportunity)
        await self._session.flush()
        return opportunity

    async def get_by_id(
        self,
        opportunity_id: uuid.UUID,
        *,
        include_requirements: bool = False,
    ) -> Opportunity | None:
        """Fetch an `Opportunity` by UUID, optionally eager-loading requirements."""
        stmt = select(Opportunity).where(Opportunity.id == opportunity_id)
        if include_requirements:
            stmt = stmt.options(selectinload(Opportunity.requirements))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_opportunities(
        self,
        *,
        company: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Opportunity]:
        """List opportunities ordered newest-first."""
        stmt = select(Opportunity).order_by(Opportunity.created_at.desc())
        if company is not None:
            stmt = stmt.where(Opportunity.company == company)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_opportunity(self, opportunity_id: uuid.UUID) -> bool:
        """Delete an `Opportunity` (cascades to requirements, runs, verdicts, tracker)."""
        result = await self._session.execute(
            delete(Opportunity).where(Opportunity.id == opportunity_id).returning(Opportunity.id)
        )
        return result.scalar_one_or_none() is not None

    async def add_requirements(
        self,
        opportunity_id: uuid.UUID,
        requirements: Sequence[RoleRequirementInput],
    ) -> list[RoleRequirement]:
        """Insert a batch of structured `RoleRequirement` rows."""
        models = [
            RoleRequirement(
                id=item.requirement_id or uuid.uuid4(),
                opportunity_id=opportunity_id,
                requirement_text=item.requirement_text,
                category=item.category,
                priority=item.priority,
            )
            for item in requirements
        ]
        self._session.add_all(models)
        await self._session.flush()
        return models

    async def replace_requirements(
        self,
        opportunity_id: uuid.UUID,
        requirements: Sequence[RoleRequirementInput],
    ) -> list[RoleRequirement]:
        """Atomically replace all parsed requirements for `opportunity_id`."""
        await self._session.execute(
            delete(RoleRequirement).where(RoleRequirement.opportunity_id == opportunity_id)
        )
        await self._session.flush()
        return await self.add_requirements(opportunity_id, requirements)

    async def list_requirements(
        self,
        opportunity_id: uuid.UUID,
        *,
        priority: RequirementPriority | None = None,
    ) -> list[RoleRequirement]:
        """Return structured requirements for an opportunity, optionally filtered by priority."""
        stmt = select(RoleRequirement).where(RoleRequirement.opportunity_id == opportunity_id)
        if priority is not None:
            stmt = stmt.where(RoleRequirement.priority == priority)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
