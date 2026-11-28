"""Typed async repository for approval-gated action drafts."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.db.models import ActionDraft, ActionDraftStatus, ActionType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from proofpath.schemas.actions import ActionDraftRecord

DraftDecision: TypeAlias = Literal["approved", "rejected"]


def draft_record(draft: ActionDraft) -> ActionDraftRecord:
    """Map a persisted draft row to its API/state/graph contract.

    Imported lazily: `proofpath.schemas.actions` reads `proofpath.db.models`, which
    initializes the `proofpath.db` package and therefore this module.
    """
    from proofpath.schemas.actions import (
        ActionDraftRecord,
        parse_action_draft_payload,
    )

    return ActionDraftRecord(
        id=draft.id,
        analysis_run_id=draft.analysis_run_id,
        action_type=draft.action_type,
        status=draft.status,
        payload=parse_action_draft_payload(draft.payload_json),
        idempotency_key=draft.idempotency_key,
        approved_at=draft.approved_at,
        executed_at=draft.executed_at,
    )


def decided_status(decision: DraftDecision) -> ActionDraftStatus:
    """Map a human decision to the draft status it produces."""
    return ActionDraftStatus.APPROVED if decision == "approved" else ActionDraftStatus.REJECTED


def decision_satisfied(status: ActionDraftStatus, decision: DraftDecision) -> bool:
    """Return True when `status` already reflects `decision` (idempotent retry)."""
    if decision == "approved":
        # `failed` means approved by a human whose executor then failed; the
        # decision itself is still approved.
        return status in {
            ActionDraftStatus.APPROVED,
            ActionDraftStatus.EXECUTED,
            ActionDraftStatus.FAILED,
        }
    return status is ActionDraftStatus.REJECTED


class DraftRepo:
    """Async CRUD and state-transition repository for `action_drafts`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_draft(
        self,
        *,
        analysis_run_id: uuid.UUID,
        action_type: ActionType,
        payload_json: dict[str, Any],
        idempotency_key: str,
        draft_id: uuid.UUID | None = None,
    ) -> ActionDraft:
        """Persist a new `pending_approval` action draft (never executes side effects)."""
        draft = ActionDraft(
            id=draft_id or uuid.uuid4(),
            analysis_run_id=analysis_run_id,
            action_type=action_type,
            payload_json=dict(payload_json),
            status=ActionDraftStatus.PENDING_APPROVAL,
            idempotency_key=idempotency_key,
            approved_at=None,
            executed_at=None,
        )
        self._session.add(draft)
        await self._session.flush()
        return draft

    async def create_or_get_draft(
        self,
        *,
        analysis_run_id: uuid.UUID,
        action_type: ActionType,
        payload_json: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[ActionDraft, bool]:
        """Idempotently create or fetch a draft by `idempotency_key`."""
        existing = await self.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            return existing, False
        created = await self.create_draft(
            analysis_run_id=analysis_run_id,
            action_type=action_type,
            payload_json=payload_json,
            idempotency_key=idempotency_key,
        )
        return created, True

    async def get_by_id(self, draft_id: uuid.UUID) -> ActionDraft | None:
        """Fetch an `ActionDraft` by UUID."""
        stmt = select(ActionDraft).where(ActionDraft.id == draft_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_for_update(self, draft_id: uuid.UUID) -> ActionDraft | None:
        """Fetch an `ActionDraft` by UUID holding a row lock (`SELECT ... FOR UPDATE`).

        Executors take this lock so two concurrent executions of the same approved
        draft serialize: the loser sees the committed `executed` status and performs
        no second side effect.
        """
        stmt = select(ActionDraft).where(ActionDraft.id == draft_id).with_for_update()
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, idempotency_key: str) -> ActionDraft | None:
        """Fetch an `ActionDraft` by its unique `idempotency_key`."""
        stmt = select(ActionDraft).where(ActionDraft.idempotency_key == idempotency_key)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_drafts(
        self,
        *,
        status: ActionDraftStatus | None = None,
        analysis_run_id: uuid.UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ActionDraft]:
        """List action drafts, filtered by status and/or run, in a stable order.

        Ordering is deterministic so paging never repeats or skips a draft.
        """
        stmt = select(ActionDraft)
        if status is not None:
            stmt = stmt.where(ActionDraft.status == status)
        if analysis_run_id is not None:
            stmt = stmt.where(ActionDraft.analysis_run_id == analysis_run_id)
        stmt = (
            stmt.order_by(ActionDraft.analysis_run_id, ActionDraft.action_type, ActionDraft.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def claim_decision(
        self,
        draft_id: uuid.UUID,
        *,
        decision: DraftDecision,
        decided_at: datetime | None = None,
    ) -> bool:
        """Atomically move `pending_approval` to the decided state; single winner.

        Compare-and-set on `status`: exactly one caller (across processes) can
        consume a pending draft, so a repeated or concurrent approval can never
        trigger the same side effect twice. Returns False when the draft was not
        pending (already decided, executed, failed, or missing).
        """
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'.")
        values: dict[str, Any] = {"status": decided_status(decision)}
        if decision == "approved":
            values["approved_at"] = decided_at or datetime.now(UTC)
        result = await self._session.execute(
            update(ActionDraft)
            .where(
                ActionDraft.id == draft_id,
                ActionDraft.status == ActionDraftStatus.PENDING_APPROVAL,
            )
            .values(**values)
            .returning(ActionDraft.id)
        )
        return result.scalar_one_or_none() is not None

    async def approve(
        self,
        draft_id: uuid.UUID,
        *,
        approved_at: datetime | None = None,
    ) -> ActionDraft | None:
        """Transition a `pending_approval` draft to `approved` with an approval timestamp."""
        draft = await self.get_by_id(draft_id)
        if draft is None:
            return None
        if draft.status is not ActionDraftStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Cannot approve draft in status '{draft.status.value}'; "
                "expected 'pending_approval'."
            )
        if not await self.claim_decision(draft_id, decision="approved", decided_at=approved_at):
            raise ValueError(
                "Cannot approve draft; it is no longer 'pending_approval' (concurrent decision)."
            )
        await self._session.flush()
        return draft

    async def reject(self, draft_id: uuid.UUID) -> ActionDraft | None:
        """Transition a `pending_approval` draft to `rejected`."""
        draft = await self.get_by_id(draft_id)
        if draft is None:
            return None
        if draft.status is not ActionDraftStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Cannot reject draft in status '{draft.status.value}'; "
                "expected 'pending_approval'."
            )
        if not await self.claim_decision(draft_id, decision="rejected"):
            raise ValueError(
                "Cannot reject draft; it is no longer 'pending_approval' (concurrent decision)."
            )
        await self._session.flush()
        return draft

    async def mark_executed(
        self,
        draft_id: uuid.UUID,
        *,
        executed_at: datetime | None = None,
    ) -> ActionDraft | None:
        """Mark an explicitly `approved` draft as `executed`."""
        draft = await self.get_by_id(draft_id)
        if draft is None:
            return None
        if draft.status is not ActionDraftStatus.APPROVED or draft.approved_at is None:
            raise ValueError(
                f"Cannot execute draft in status '{draft.status.value}'; "
                "explicit human approval is required first."
            )
        draft.status = ActionDraftStatus.EXECUTED
        draft.executed_at = executed_at or datetime.now(UTC)
        await self._session.flush()
        return draft

    async def mark_failed(self, draft_id: uuid.UUID) -> ActionDraft | None:
        """Mark an `approved` draft whose side-effect executor failed as `failed`."""
        draft = await self.get_by_id(draft_id)
        if draft is None:
            return None
        if draft.status is not ActionDraftStatus.APPROVED:
            raise ValueError(
                f"Cannot mark draft in status '{draft.status.value}' as failed; "
                "expected 'approved'."
            )
        draft.status = ActionDraftStatus.FAILED
        await self._session.flush()
        return draft

    async def delete_draft(self, draft_id: uuid.UUID) -> bool:
        """Delete an `ActionDraft` by UUID."""
        result = await self._session.execute(
            delete(ActionDraft).where(ActionDraft.id == draft_id).returning(ActionDraft.id)
        )
        return result.scalar_one_or_none() is not None
