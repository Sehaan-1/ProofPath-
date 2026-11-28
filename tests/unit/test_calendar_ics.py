"""RFC 5545 `.ics` generation for approved calendar blocks (Sub-part 2.1.3b)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest

from proofpath.calendar_ics import (
    ICS_PRODID,
    build_calendar_invite,
    calendar_uid,
    validate_timezone_name,
)

pytestmark = pytest.mark.unit

DRAFT_ID = uuid.UUID("0b0f8f9c-6a5e-4f1d-9c37-1c5f7d5f2f11")
START = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
END = START + timedelta(minutes=90)
STAMP = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)


def _invite(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "draft_id": DRAFT_ID,
        "title": "Prep: Senior Backend role",
        "start_at": START,
        "end_at": END,
        "description": "Review the posting and the gap list.",
        "timezone_name": "Asia/Kolkata",
        "stamp": STAMP,
    }
    kwargs.update(overrides)
    return build_calendar_invite(**kwargs)  # type: ignore[arg-type]


def _unfold(ics: str) -> list[str]:
    lines: list[str] = []
    for raw in ics.split("\r\n"):
        if raw.startswith(" ") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _properties(ics: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for line in _unfold(ics):
        if not line or ":" not in line:
            continue
        name, _, value = line.partition(":")
        props.setdefault(name, value)
    return props


def test_invite_has_the_required_calendar_and_event_properties() -> None:
    ics = _invite()
    props = _properties(ics)

    assert ics.startswith("BEGIN:VCALENDAR\r\n")
    assert ics.endswith("END:VCALENDAR\r\n")
    assert props["VERSION"] == "2.0"
    assert props["PRODID"] == ICS_PRODID
    assert props["METHOD"] == "PUBLISH"
    assert props["UID"] == f"proofpath-{DRAFT_ID}@proofpath.local"
    assert props["DTSTAMP"] == "20261128T120000Z"
    assert props["DTSTART"] == "20261201T090000Z"
    assert props["DTEND"] == "20261201T103000Z"
    assert props["SUMMARY"] == "Prep: Senior Backend role"
    assert props["X-PROOFPATH-DRAFT-ID"] == str(DRAFT_ID)
    assert props["X-PROOFPATH-TIMEZONE"] == "Asia/Kolkata"
    assert props["CLASS"] == "PRIVATE"


def test_every_line_is_crlf_terminated_and_within_75_octets() -> None:
    ics = _invite(description="ü" * 400)

    assert ics.endswith("\r\n")
    assert "\n" not in ics.replace("\r\n", "")
    assert "\r" not in ics.replace("\r\n", "")
    assert all(len(line.encode("utf-8")) <= 75 for line in ics.split("\r\n"))


def test_long_utf8_text_folds_without_splitting_codepoints() -> None:
    description = "Detailed prep plan — " + "é" * 120 + " done."
    ics = _invite(description=description)

    assert _properties(ics)["DESCRIPTION"] == description
    assert all(len(line.encode("utf-8")) <= 75 for line in ics.split("\r\n"))


def test_text_values_are_escaped_per_rfc5545() -> None:
    ics = _invite(
        title="Prep; role, at Acme",
        description="Line one\nLine two\\slash",
    )
    props = _properties(ics)

    assert props["SUMMARY"] == r"Prep\; role\, at Acme"
    assert props["DESCRIPTION"] == r"Line one\nLine two\\slash"


def test_control_characters_are_stripped() -> None:
    props = _properties(_invite(title="Prep\x00\x07 block"))

    assert props["SUMMARY"] == "Prep block"


def test_times_are_normalized_to_utc() -> None:
    kolkata = timezone(timedelta(hours=5, minutes=30))
    ics = _invite(
        start_at=datetime(2026, 12, 1, 14, 30, tzinfo=kolkata),
        end_at=datetime(2026, 12, 1, 16, 0, tzinfo=kolkata),
    )
    props = _properties(ics)

    assert props["DTSTART"] == "20261201T090000Z"
    assert props["DTEND"] == "20261201T103000Z"


def test_same_draft_and_stamp_render_byte_identical_output() -> None:
    first = _invite()
    second = _invite()

    assert first == second
    assert first != _invite(title="Prep: Staff Backend role")


def test_calendar_uid_is_a_function_of_the_draft_id() -> None:
    assert calendar_uid(draft_id=DRAFT_ID) == f"proofpath-{DRAFT_ID}@proofpath.local"
    assert calendar_uid(draft_id=DRAFT_ID) != calendar_uid(draft_id=uuid.uuid4())


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": "   "},
        {"end_at": START},
        {"end_at": START - timedelta(minutes=5)},
        # Naive datetimes must be rejected, not silently assumed to be UTC.
        {"start_at": datetime(2026, 12, 1, 9, 0)},  # noqa: DTZ001
        {"stamp": datetime(2026, 11, 28, 12, 0)},  # noqa: DTZ001
    ],
)
def test_invalid_arguments_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _invite(**overrides)


def test_unknown_timezone_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        _invite(timezone_name="Mars/Olympus_Mons")
    with pytest.raises(ValueError, match="invalid timezone"):
        _invite(timezone_name="   ")


def test_timezone_validation_accepts_utc_and_known_zones() -> None:
    assert validate_timezone_name("UTC") == "UTC"
    assert validate_timezone_name(" Asia/Tokyo ") == "Asia/Tokyo"
    # A host without a tz database still validates shape, never the calendar data.
    assert validate_timezone_name("Mars/Olympus_Mons", available=frozenset()) == (
        "Mars/Olympus_Mons"
    )
