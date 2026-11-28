"""Redis Streams-backed job queue for ProofPath background workers.

A Redis Stream is the *transport* and a consumer group gives reliable,
at-least-once delivery with a Pending Entries List (PEL) tracking unacked
messages. The authoritative job *state* lives in the ``background_jobs``
PostgreSQL table (correlated by job id); stream messages only carry the job id,
so Redis and PostgreSQL stay decoupled.

Primitives:

* :meth:`JobQueue.ensure_group` — idempotently create the consumer group.
* :meth:`JobQueue.enqueue` — append a job id to the stream (producer side).
* :meth:`JobQueue.dequeue` — ``XREADGROUP`` new messages for this consumer.
* :meth:`JobQueue.ack` — acknowledge a processed message (removes it from the PEL).
* :meth:`JobQueue.reclaim` — ``XAUTOCLAIM`` idle messages left behind by a
  crashed or stalled consumer, so work is never stranded in the PEL.
* :meth:`JobQueue.schedule_retry` — place a transiently-failed job into a sorted
  set keyed by its "ready-at" timestamp (Sub-part 2.2.3a exponential backoff).
* :meth:`JobQueue.drain_ready_retries` — move jobs whose backoff has elapsed back
  onto the stream for redelivery.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

#: Default stream, consumer-group, and delayed-retry keys. A dedicated DB index
#: (``REDIS_URL`` path) keeps ProofPath queue traffic isolated from other usage.
DEFAULT_STREAM = "proofpath:jobs"
DEFAULT_GROUP = "proofpath-workers"
DEFAULT_DELAY_KEY = "proofpath:jobs:delayed"

_JOB_ID_FIELD = "job_id"
_JOB_TYPE_FIELD = "job_type"
_PAYLOAD_FIELD = "payload"


@dataclass(frozen=True, slots=True)
class JobMessage:
    """One delivered stream message.

    ``job_id`` is ``None`` for a malformed message (missing/corrupt fields); the
    worker acks those immediately and logs, since there is no job row to update.
    """

    message_id: str
    job_id: uuid.UUID | None
    job_type: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetryItem:
    """A job scheduled for delayed re-enqueue after a transient failure.

    The ``ready_at`` timestamp (Unix seconds) is the ZSET score; ``drain_ready_retries``
    re-enqueues any item whose score is in the past.
    """

    job_id: uuid.UUID
    job_type: str
    payload: dict[str, Any]
    ready_at: float


def _parse_message(message_id: str, fields: dict[str, Any]) -> JobMessage:
    """Decode a raw stream entry into a `JobMessage`, tolerating corruption."""
    raw_job_id = fields.get(_JOB_ID_FIELD)
    raw_job_type = fields.get(_JOB_TYPE_FIELD)
    raw_payload = fields.get(_PAYLOAD_FIELD)

    try:
        if not isinstance(raw_job_id, str):
            raise ValueError("missing job_id field")
        job_id = uuid.UUID(raw_job_id)
    except (ValueError, TypeError):
        logger.warning("Discarding malformed stream message %r (bad job_id).", message_id)
        return JobMessage(message_id=message_id, job_id=None, job_type="", payload={})

    job_type = raw_job_type if isinstance(raw_job_type, str) else ""
    payload: dict[str, Any] = {}
    if isinstance(raw_payload, str) and raw_payload:
        try:
            decoded = json.loads(raw_payload)
            if isinstance(decoded, dict):
                payload = decoded
        except json.JSONDecodeError:
            logger.warning("Stream message %r had non-JSON payload; defaulting to {}.", message_id)

    return JobMessage(message_id=message_id, job_id=job_id, job_type=job_type, payload=payload)


def _parse_entries(stream_entries: list[Any]) -> list[JobMessage]:
    """Flatten ``XREADGROUP``/``XAUTOCLAIM`` entry lists into `JobMessage`s."""
    messages: list[JobMessage] = []
    for _msg_id, fields in stream_entries:
        if not isinstance(fields, dict):
            continue
        decoded = {k.decode() if isinstance(k, bytes) else k: v for k, v in fields.items()}
        messages.append(_parse_message(_msg_id, decoded))
    return messages


class JobQueue:
    """Async Redis Streams job queue with a consumer group."""

    def __init__(
        self,
        redis: Redis,
        *,
        stream: str = DEFAULT_STREAM,
        group: str = DEFAULT_GROUP,
        consumer: str = "worker-1",
        delay_key: str = DEFAULT_DELAY_KEY,
    ) -> None:
        self._redis = redis
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._delay_key = delay_key

    @property
    def stream(self) -> str:
        return self._stream

    @property
    def group(self) -> str:
        return self._group

    @property
    def consumer(self) -> str:
        return self._consumer

    @property
    def delay_key(self) -> str:
        return self._delay_key

    @classmethod
    def from_url(
        cls,
        redis_url: str,
        *,
        stream: str = DEFAULT_STREAM,
        group: str = DEFAULT_GROUP,
        consumer: str = "worker-1",
        delay_key: str = DEFAULT_DELAY_KEY,
    ) -> JobQueue:
        """Build a `JobQueue` backed by a new Redis client resolved from a URL."""
        return cls(
            Redis.from_url(redis_url, decode_responses=True),
            stream=stream,
            group=group,
            consumer=consumer,
            delay_key=delay_key,
        )

    async def ensure_group(self) -> None:
        """Idempotently create the consumer group (and the stream if absent).

        Starts at id ``0`` so messages enqueued before the group existed are
        still delivered — no job is silently lost on a cold worker start.
        """
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            # Group already exists — exactly what we want.

    async def enqueue(
        self,
        job_id: uuid.UUID,
        *,
        job_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Append a job to the stream; returns the assigned Redis message id."""
        message_id = await self._redis.xadd(
            self._stream,
            {
                _JOB_ID_FIELD: str(job_id),
                _JOB_TYPE_FIELD: job_type,
                _PAYLOAD_FIELD: json.dumps(payload, separators=(",", ":")),
            },
        )
        return str(message_id)

    async def dequeue(self, *, count: int = 1, block_ms: int = 0) -> list[JobMessage]:
        """Read new (never-delivered) messages for this consumer via ``XREADGROUP``.

        ``block_ms=0`` blocks indefinitely until a message arrives; a positive
        value lets the worker poll for shutdown between reads.
        """
        response: Any = await self._redis.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=count,
            block=block_ms,
        )
        if not response:
            return []
        # response == [(stream_name, [(msg_id, fields), ...])]
        _stream_name, entries = response[0]
        return _parse_entries(entries)

    async def ack(self, *message_ids: str) -> int:
        """Acknowledge messages (removes them from the group's PEL)."""
        if not message_ids:
            return 0
        return int(await self._redis.xack(self._stream, self._group, *message_ids))

    async def reclaim(
        self,
        *,
        min_idle_ms: int = 60_000,
        count: int = 10,
    ) -> list[JobMessage]:
        """Steal idle pending messages left by a stalled consumer (``XAUTOCLAIM``).

        Returns messages whose original consumer has not acked them within
        ``min_idle_ms`` so this worker can retry them — preventing stranded work
        after a crash or OOM-kill.
        """
        claimed: Any = await self._redis.xautoclaim(
            self._stream,
            self._group,
            self._consumer,
            min_idle_time=min_idle_ms,
            count=count,
        )
        _next_id, entries, _deleted = claimed
        return _parse_entries(entries)

    async def pending_count(self) -> int:
        """Number of messages delivered to the group but not yet acked (PEL size)."""
        summary: Any = await self._redis.xpending(self._stream, self._group)
        if isinstance(summary, int):
            return summary
        if isinstance(summary, dict):
            pending = summary.get("pending")
            return int(pending) if pending is not None else 0
        return 0

    async def stream_length(self) -> int:
        """Total entries in the stream (acked and pending)."""
        return int(await self._redis.xlen(self._stream))

    # ------------------------------------------------------------------
    # Delayed retry (Sub-part 2.2.3a — exponential backoff)
    # ------------------------------------------------------------------

    async def schedule_retry(
        self,
        job_id: uuid.UUID,
        *,
        job_type: str,
        payload: dict[str, Any],
        delay_seconds: float,
    ) -> float:
        """Schedule a transiently-failed job for delayed re-enqueue.

        Stores the job in a Redis sorted set keyed by its "ready-at" Unix timestamp
        (``now + delay_seconds``). The worker's idle loop drains ready items back
        onto the stream via :meth:`drain_ready_retries`. This keeps the worker
        productive during the backoff window (it does not block on a sleep) and the
        retry survives a worker restart. Returns the computed ``ready_at`` score.
        """
        if delay_seconds < 0:
            delay_seconds = 0
        ready_at = time.time() + delay_seconds
        member = json.dumps(
            {
                _JOB_ID_FIELD: str(job_id),
                _JOB_TYPE_FIELD: job_type,
                _PAYLOAD_FIELD: payload,
            },
            separators=(",", ":"),
        )
        await self._redis.zadd(self._delay_key, {member: ready_at})
        return ready_at

    async def drain_ready_retries(self, *, now: float | None = None) -> list[RetryItem]:
        """Re-enqueue all delayed jobs whose backoff has elapsed; return them.

        Atomically claims ready members with ``ZRANGEBYSCORE`` + ``ZREM`` so two
        concurrent workers never re-enqueue the same retry. Each claimed item is
        ``XADD``-ed back onto the stream as a fresh message for the consumer group.
        """
        timestamp = now if now is not None else time.time()
        ready: Any = await self._redis.zrangebyscore(self._delay_key, 0, timestamp)
        if not ready:
            return []
        # Claim atomically: remove the members we intend to re-enqueue.
        removed = await self._redis.zrem(self._delay_key, *ready)
        items: list[RetryItem] = []
        for raw in ready[:removed]:
            member = raw.decode() if isinstance(raw, bytes) else str(raw)
            try:
                decoded = json.loads(member)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Discarding corrupt delayed-retry member %r.", member)
                continue
            job_id_raw = decoded.get(_JOB_ID_FIELD)
            try:
                job_id = uuid.UUID(str(job_id_raw)) if job_id_raw else None
            except (ValueError, TypeError):
                logger.warning("Discarding delayed-retry member with bad job_id %r.", member)
                continue
            if job_id is None:
                continue
            payload = decoded.get(_PAYLOAD_FIELD, {})
            if not isinstance(payload, dict):
                payload = {}
            items.append(
                RetryItem(
                    job_id=job_id,
                    job_type=str(decoded.get(_JOB_TYPE_FIELD, "")),
                    payload=payload,
                    ready_at=timestamp,
                )
            )
        for item in items:
            await self.enqueue(item.job_id, job_type=item.job_type, payload=item.payload)
        return items

    async def delayed_count(self) -> int:
        """Number of jobs currently waiting out their backoff window."""
        return int(await self._redis.zcard(self._delay_key))

    async def close(self) -> None:
        """Close the underlying Redis client."""
        await self._redis.aclose()
