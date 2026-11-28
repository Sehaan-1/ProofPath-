"""RFC 5545 (iCalendar) invite generation for approved `calendar_block` drafts.

Sub-part 2.1.3b: ProofPath materializes an approved calendar block as a local
`calendar_tasks` row and serves a downloadable `.ics` file generated here. The
generator is deliberately dependency-free and deterministic: the same draft always
produces byte-identical output, so re-downloading never mints a new invite.

Live Google Calendar writes are **not** performed here; they stay behind the
explicit `PROOFPATH_ENABLE_GOOGLE_CALENDAR` opt-in flag handled by the executor.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from zoneinfo import available_timezones

__all__ = [
    "ICS_PRODID",
    "build_calendar_invite",
    "calendar_uid",
    "validate_timezone_name",
]

ICS_PRODID = "-//ProofPath//Approved Action Draft//EN"

_CRLF = "\r\n"
_MAX_OCTETS = 75
_FORBIDDEN_CONTROLS = frozenset(chr(code) for code in range(0x20) if code not in (0x09, 0x0A, 0x0D))


def calendar_uid(*, draft_id: uuid.UUID) -> str:
    """Deterministic, globally unique `UID` for the invite of one action draft."""
    return f"proofpath-{draft_id}@proofpath.local"


def validate_timezone_name(name: str, *, available: frozenset[str] | None = None) -> str:
    """Return `name` when it looks like a known IANA zone, else raise `ValueError`."""
    cleaned = name.strip()
    if not cleaned or len(cleaned) > 64 or any(ch in cleaned for ch in _FORBIDDEN_CONTROLS):
        raise ValueError(f"invalid timezone name: {name!r}")
    zones = available if available is not None else frozenset(available_timezones())
    if zones and cleaned not in zones and cleaned != "UTC":
        raise ValueError(f"unknown timezone name: {name!r}")
    return cleaned


def _escape_text(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch not in _FORBIDDEN_CONTROLS)
    return (
        cleaned.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """Fold one content line to <= 75 octets without splitting a UTF-8 codepoint."""
    chunks: list[str] = []
    chunk = ""
    chunk_bytes = 0
    for char in line:
        size = len(char.encode("utf-8"))
        budget = _MAX_OCTETS if not chunks else _MAX_OCTETS - 1
        if chunk_bytes + size > budget:
            chunks.append(chunk)
            chunk = char
            chunk_bytes = size
        else:
            chunk += char
            chunk_bytes += size
    chunks.append(chunk)
    return f"{_CRLF} ".join(chunks)


def _require_aware(value: datetime, *, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.tzinfo.utcoffset(value) is None
    ):
        raise ValueError(f"{field} must be timezone-aware.")
    return value


def _utc_stamp(value: datetime, *, field: str) -> str:
    return _require_aware(value, field=field).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def build_calendar_invite(
    *,
    draft_id: uuid.UUID,
    title: str,
    start_at: datetime,
    end_at: datetime,
    description: str | None = None,
    timezone_name: str = "UTC",
    stamp: datetime | None = None,
) -> str:
    """Render a single-event `.ics` document (CRLF terminated, RFC 5545 folded)."""
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title must not be blank.")
    _require_aware(start_at, field="start_at")
    _require_aware(end_at, field="end_at")
    if end_at <= start_at:
        raise ValueError("end_at must be strictly after start_at.")
    tz_name = validate_timezone_name(timezone_name)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{ICS_PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{calendar_uid(draft_id=draft_id)}",
        f"DTSTAMP:{_utc_stamp(stamp or datetime.now(UTC), field='stamp')}",
        f"DTSTART:{_utc_stamp(start_at, field='start_at')}",
        f"DTEND:{_utc_stamp(end_at, field='end_at')}",
        f"SUMMARY:{_escape_text(clean_title)}",
    ]
    if description:
        lines.append(f"DESCRIPTION:{_escape_text(description)}")
    lines.extend(
        [
            f"X-PROOFPATH-DRAFT-ID:{draft_id}",
            f"X-PROOFPATH-TIMEZONE:{tz_name}",
            "CLASS:PRIVATE",
            "STATUS:CONFIRMED",
            "TRANSP:OPAQUE",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    )
    return _CRLF.join(_fold(line) for line in lines) + _CRLF
