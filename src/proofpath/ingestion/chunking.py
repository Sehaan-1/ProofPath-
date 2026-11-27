"""Section-aware splitting and chunking for local evidence documents."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

MAX_CHUNK_CHARS: Final[int] = 2400
MAX_SECTION_LABEL: Final[int] = 255

# Canonical resume/CV headings required by Sub-part 1.3.1a.
RESUME_SECTION_ALIASES: Final[dict[str, str]] = {
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "employment": "Experience",
    "work history": "Experience",
    "relevant experience": "Experience",
    "projects": "Projects",
    "selected projects": "Projects",
    "personal projects": "Projects",
    "technical projects": "Projects",
    "key projects": "Projects",
    "skills": "Skills",
    "technical skills": "Skills",
    "core skills": "Skills",
    "skills and tools": "Skills",
    "technologies": "Skills",
    "tech stack": "Skills",
    "education": "Education",
    "academic background": "Education",
    "academics": "Education",
    "academic history": "Education",
}

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_UNDERLINE = re.compile(r"^(?:=+|-+)\s*$")
_BOLD_HEADING = re.compile(r"^\*\*(.+?)\*\*\s*$")
_COLON_HEADING = re.compile(r"^([A-Za-z][A-Za-z0-9 &/+\-]{1,80}):\s*$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    """One retrieval-sized slice of a local document with provenance anchors."""

    content: str
    section_label: str | None
    start_line: int
    end_line: int
    token_count: int
    start_page: int | None = None
    end_page: int | None = None

    @property
    def anchor(self) -> str:
        """Fragment identifying section and 1-based line span (and page when known)."""
        label = self.section_label or "body"
        fragment = f"{label}:L{self.start_line}-L{self.end_line}"
        if self.start_page is not None:
            end_page = self.end_page if self.end_page is not None else self.start_page
            fragment = f"{fragment}&p{self.start_page}-{end_page}"
        return fragment

    def as_anchor_record(self, chunk_index: int) -> dict[str, object]:
        """JSON-serializable provenance row stored on `evidence_sources.metadata_json`."""
        return {
            "chunk_index": chunk_index,
            "section_label": self.section_label,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "anchor": self.anchor,
        }


def approx_token_count(text: str) -> int:
    """Cheap character-based token estimate (about 4 chars/token)."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def normalize_heading(raw: str) -> str:
    """Collapse whitespace and strip markdown decoration for heading comparison."""
    cleaned = raw.strip().strip("#").strip("*").strip("_").strip(":").strip()
    return _WHITESPACE.sub(" ", cleaned).casefold()


def canonicalize_resume_section(heading: str) -> str | None:
    """Map a heading to Experience/Projects/Skills/Education when it is a known alias."""
    return RESUME_SECTION_ALIASES.get(normalize_heading(heading))


def chunk_document(
    text: str,
    *,
    canonicalize_resume: bool = False,
    page_for_line: Sequence[int] | None = None,
) -> list[ParsedChunk]:
    """Split `text` on headings and size-bound the resulting sections."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []
    lines = normalized.split("\n")
    sections = _split_sections(lines, canonicalize_resume=canonicalize_resume)
    chunks: list[ParsedChunk] = []
    for heading, start_line, body_lines in sections:
        body = "\n".join(body_lines).strip("\n")
        if not body.strip():
            continue
        label = _section_label(heading, canonicalize_resume=canonicalize_resume)
        chunks.extend(
            _chunk_section(
                body,
                section_label=label,
                start_line=start_line,
                page_for_line=page_for_line,
            )
        )
    return chunks


def _section_label(heading: str | None, *, canonicalize_resume: bool) -> str | None:
    if heading is None:
        return "Summary" if canonicalize_resume else None
    if canonicalize_resume:
        canonical = canonicalize_resume_section(heading)
        if canonical is not None:
            return canonical
    label = _WHITESPACE.sub(" ", heading.strip())
    if not label:
        return None
    return label[:MAX_SECTION_LABEL]


def _split_sections(
    lines: Sequence[str],
    *,
    canonicalize_resume: bool,
) -> list[tuple[str | None, int, list[str]]]:
    """Return `(heading, content_start_line, body_lines)` in document order."""
    sections: list[tuple[str | None, int, list[str]]] = []
    current_heading: str | None = None
    current_start = 1
    current_body: list[str] = []
    index = 0
    while index < len(lines):
        heading, consumed = _heading_at(lines, index, canonicalize_resume=canonicalize_resume)
        if heading is not None:
            sections.append((current_heading, current_start, current_body))
            current_heading = heading
            current_body = []
            index += consumed
            current_start = index + 1
            continue
        current_body.append(lines[index])
        index += 1

    sections.append((current_heading, current_start, current_body))
    return [
        (heading, start_line, body)
        for heading, start_line, body in sections
        if heading is not None or any(part.strip() for part in body)
    ]


def _heading_at(
    lines: Sequence[str],
    index: int,
    *,
    canonicalize_resume: bool,
) -> tuple[str | None, int]:
    line = lines[index]
    stripped = line.strip()
    if not stripped:
        return None, 1

    atx = _ATX_HEADING.match(stripped)
    if atx is not None:
        return atx.group(2).strip(), 1

    if (
        index + 1 < len(lines)
        and _SETEXT_UNDERLINE.match(lines[index + 1].strip())
        and 1 <= len(stripped) <= 80
    ):
        return stripped, 2

    bold = _BOLD_HEADING.match(stripped)
    if bold is not None and len(bold.group(1).strip()) <= 80:
        return bold.group(1).strip(), 1

    colon = _COLON_HEADING.match(stripped)
    if colon is not None:
        candidate = colon.group(1).strip()
        if _looks_like_heading(candidate, canonicalize_resume=canonicalize_resume):
            return candidate, 1

    if _looks_like_heading(stripped, canonicalize_resume=canonicalize_resume):
        return stripped.strip(":"), 1
    return None, 1


def _looks_like_heading(text: str, *, canonicalize_resume: bool) -> bool:
    candidate = text.strip().strip(":")
    if not candidate or len(candidate) > 60 or "." in candidate:
        return False
    if canonicalize_resume_section(candidate) is not None:
        return True
    if not canonicalize_resume:
        return False
    words = candidate.split()
    if not words or len(words) > 6:
        return False
    letters = [char for char in candidate if char.isalpha()]
    if not letters:
        return False
    return all(char.isupper() for char in letters)


def _chunk_section(
    body: str,
    *,
    section_label: str | None,
    start_line: int,
    page_for_line: Sequence[int] | None,
) -> list[ParsedChunk]:
    if len(body) <= MAX_CHUNK_CHARS:
        end_line = start_line + body.count("\n")
        return [
            _make_chunk(
                body,
                section_label=section_label,
                start_line=start_line,
                end_line=end_line,
                page_for_line=page_for_line,
            )
        ]

    packed: list[ParsedChunk] = []
    buffer_start = 0
    buffer_end = 0
    for para_start, para_end in _paragraph_spans(body):
        if buffer_end > buffer_start and (para_end - buffer_start) > MAX_CHUNK_CHARS:
            packed.append(
                _chunk_from_span(
                    body,
                    buffer_start,
                    buffer_end,
                    section_label=section_label,
                    start_line=start_line,
                    page_for_line=page_for_line,
                )
            )
            buffer_start = para_start
        elif buffer_end <= buffer_start:
            buffer_start = para_start
        buffer_end = para_end

    if buffer_end > buffer_start:
        packed.append(
            _chunk_from_span(
                body,
                buffer_start,
                buffer_end,
                section_label=section_label,
                start_line=start_line,
                page_for_line=page_for_line,
            )
        )

    if any(len(item.content) > MAX_CHUNK_CHARS for item in packed):
        return _hard_split_chunks(packed, page_for_line=page_for_line)
    return [item for item in packed if item.content.strip()]


def _paragraph_spans(body: str) -> list[tuple[int, int]]:
    """Return inclusive-exclusive character spans of paragraphs split on blank lines."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for part in re.split(r"\n{2,}", body):
        found = body.find(part, offset)
        if found < 0:
            found = offset
        end = found + len(part)
        if part.strip():
            spans.append((found, end))
        offset = end
    if not spans and body.strip():
        spans.append((0, len(body)))
    return spans


def _chunk_from_span(
    body: str,
    start_offset: int,
    end_offset: int,
    *,
    section_label: str | None,
    start_line: int,
    page_for_line: Sequence[int] | None,
) -> ParsedChunk:
    snippet = body[start_offset:end_offset].strip("\n")
    rel_start = body[:start_offset].count("\n")
    rel_end = body[:end_offset].count("\n") if end_offset else rel_start
    abs_start = start_line + rel_start
    abs_end = start_line + rel_end
    return _make_chunk(
        snippet,
        section_label=section_label,
        start_line=abs_start,
        end_line=max(abs_start, abs_end),
        page_for_line=page_for_line,
    )


def _hard_split_chunks(
    chunks: Sequence[ParsedChunk],
    *,
    page_for_line: Sequence[int] | None,
) -> list[ParsedChunk]:
    result: list[ParsedChunk] = []
    for chunk in chunks:
        if len(chunk.content) <= MAX_CHUNK_CHARS:
            if chunk.content.strip():
                result.append(chunk)
            continue
        words = chunk.content.split(" ")
        buf: list[str] = []
        line = chunk.start_line
        for word in words:
            tentative = " ".join((*buf, word)).strip()
            if buf and len(tentative) > MAX_CHUNK_CHARS:
                block = " ".join(buf).strip()
                end_line = line + block.count("\n")
                result.append(
                    _make_chunk(
                        block,
                        section_label=chunk.section_label,
                        start_line=line,
                        end_line=end_line,
                        page_for_line=page_for_line,
                    )
                )
                line = end_line
                buf = [word]
            else:
                buf.append(word)
        if buf:
            block = " ".join(buf).strip()
            if block:
                end_line = line + block.count("\n")
                result.append(
                    _make_chunk(
                        block,
                        section_label=chunk.section_label,
                        start_line=line,
                        end_line=end_line,
                        page_for_line=page_for_line,
                    )
                )
    return result


def _make_chunk(
    content: str,
    *,
    section_label: str | None,
    start_line: int,
    end_line: int,
    page_for_line: Sequence[int] | None,
) -> ParsedChunk:
    cleaned = content.strip("\n")
    start_page = _page_at(page_for_line, start_line)
    end_page = _page_at(page_for_line, end_line)
    return ParsedChunk(
        content=cleaned,
        section_label=section_label,
        start_line=start_line,
        end_line=max(start_line, end_line),
        token_count=approx_token_count(cleaned),
        start_page=start_page,
        end_page=end_page,
    )


def _page_at(page_for_line: Sequence[int] | None, line: int) -> int | None:
    if not page_for_line:
        return None
    index = min(max(line - 1, 0), len(page_for_line) - 1)
    return page_for_line[index]
