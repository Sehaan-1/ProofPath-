"""Deterministic input screening (2.3.1), not an authorization or evidence check.

Mask individual spans, never discard a JD/line. Only the detection bit belongs
in run diagnostics; do not log matched instructions or personal data.
"""

from __future__ import annotations

import re

# Keep the targets narrow: e.g. "call clients" and "follow previous experience
# requirements" are job text. A targeted verdict can name a skill (SAP S/4HANA)
# as well as "all skills". Never match an arbitrary rest-of-line after a verb.
_JD_INSTRUCTION = re.compile(
    r"""\b(?:
        system\s+override[ \t]*:
      | (?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your)\s+){0,2}
        (?:previous|prior|earlier|above|system)\s+(?:instructions?|prompts?|rules?|directives?)\b
      | (?:mark|classify|label)\s+(?:(?:all|every|each)\s+(?:skills?|requirements?|claims?)\s+
        (?:as\s+)?|(?:[\w/+#-]+\s+){1,4}as\s+)supported\b
      | (?:call|invoke|execute|run)\s+(?:the\s+)?(?:(?:tool|function)\s+)?
        [`"']?(?:approve_action|create_action_draft)\b[`"']?
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def contains_jd_injection(text: str) -> bool:
    """Check the source JD, not the trusted instructions surrounding it."""
    return _JD_INSTRUCTION.search(text) is not None


def neutralize_jd_instructions(text: str) -> str:
    """Leave an explicit marker in place of each recognized instruction."""
    return _JD_INSTRUCTION.sub("[MASKED INSTRUCTION]", text)


# Common national formats, with optional country code, or compact E.164.
# Horizontal separators only: never join numbers across lines in a resume.
_PHONE = re.compile(
    r"""(?<![\w.+$£€₹-])(?:
        (?:\+\d{1,3}[ .-]?|1[ .-]?)?
        (?: \(\d{2,4}\)[ .-]?\d{3,4}[ .-]?\d{4}
          | \d{3}[ .-]?\d{3}[ .-]?\d{4}
          | \d{2,4}[ .-]\d{3,4}[ .-]\d{4}
          | \d{4,5}[ -]\d{6}
          | \d{5}[ -]\d{5} )
      | \+[1-9]\d{7,14}
      | \d{3}-\d{4}
    )(?:[ \t]*(?:ext\.?|x)[ \t]*\d{1,6})?(?!\w|[.-]\d)""",
    re.IGNORECASE | re.VERBOSE,
)

# Numbered street addresses / PO boxes, optionally with a unit and a postal
# locality. Bound street/city words and require address structure rather than
# treating everything after the word "address" as private.
_ADDRESS = re.compile(
    r"""(?<![\w-])(?:
        P\.?[ \t]*O\.?[ \t]+Box[ \t]+\d+\b
      | \d{1,6}[a-z]?[ \t]+(?!years?\b|months?\b|days?\b|hours?\b)
        (?:[\w'-]+[ \t]+){1,5}?
        (?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|
           court|ct|circle|cir|terrace|ter|place|pl|way|parkway|pkwy)\b
    )
    (?:[ \t]+(?:NW|NE|SW|SE|N|E|W|S)\b)?
    (?:\.?(?:[ \t]*,[ \t]*|[ \t]+)
       (?:(?:apartment|apt|unit|suite|ste)\b\.?[ \t]*(?:\#[ \t]*)?|\#[ \t]*)
       (?:[a-z]?\d+[a-z]?|[a-z])\b)?
    (?:\.?[ \t]*,[ \t]*(?:[a-z'-]+[ \t]+){0,3}[a-z'-]+
       (?:[ \t]*,[ \t]*(?-i:[A-Z]{2})\b(?:[ \t]+\d{5,6}(?:-\d{4})?\b)?
         |[ \t]+[a-z]{2}[ \t]+\d{5,6}(?:-\d{4})?\b))?""",
    re.IGNORECASE | re.VERBOSE,
)

# Fallback for explicitly labelled, free-form personal addresses. Stop at the
# field/sentence boundary. If a street was already masked, leave subsequent
# prose alone rather than extending the redaction to the rest of the line.
_PERSONAL_ADDRESS_FIELD = re.compile(
    r"(?P<label>\b(?:home|personal|residential|mailing|postal)[ \t]+address[ \t]*:[ \t]*)"
    r"(?=\S)(?!\[REDACTED ADDRESS\])(?:[^.;|\r\n]|\.(?!\s|$))+(?<![ \t])",
    re.IGNORECASE,
)


def mask_pii(text: str) -> str:
    """Best-effort masking for *outbound* text; leave local source records intact."""
    # Mask whole phones first; a trailing digit group before a street address
    # must not be mistaken for a house number, leaving the rest of the phone.
    text = _PHONE.sub("[REDACTED PHONE]", text)
    text = _ADDRESS.sub("[REDACTED ADDRESS]", text)
    return _PERSONAL_ADDRESS_FIELD.sub(r"\g<label>[REDACTED ADDRESS]", text)
