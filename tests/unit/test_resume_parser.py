"""Unit tests for Sub-part 1.3.1a (resume/CV parser + section-aware chunking)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader

from proofpath.db.models import EvidenceSourceType
from proofpath.ingestion import (
    EmptyDocumentError,
    UnsupportedDocumentError,
    chunk_document,
    infer_source_type,
    parse_resume_file,
    parse_resume_text,
)
from proofpath.ingestion.chunking import MAX_CHUNK_CHARS, canonicalize_resume_section

_MARKDOWN_CV = """\
Jane Doe
AI Agent Engineer

## Experience

- Built FastAPI services with pgvector hybrid search.
- Shipped MCP tools over a cited evidence ledger.

## Projects

ProofPath — personal evidence and opportunity agent.

## Skills

Python, FastAPI, LangGraph, MCP, PostgreSQL

## Education

B.S. Computer Science
"""

_PLAIN_CV = """\
Jane Doe

EXPERIENCE
Built FastAPI services with pgvector.

PROJECTS
ProofPath evidence ledger.

SKILLS
Python, FastAPI, MCP

EDUCATION
B.S. Computer Science
"""


def _build_text_pdf(lines: list[str]) -> bytes:
    """Assemble a minimal one-page PDF containing Helvetica text lines."""
    escaped: list[str] = []
    for line in lines:
        escaped.append(line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)"))
    ops = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(escaped):
        if index:
            ops.append("0 -16 Td")
        ops.append(f"({line}) Tj")
    ops.append("ET")
    stream = ("\n".join(ops) + "\n").encode("latin-1", "replace")

    def _obj(number: int, payload: bytes) -> bytes:
        return f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"

    objects = [
        _obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        _obj(2, b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>"),
        _obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        _obj(4, f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream"),
        _obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    header = b"%PDF-1.4\n"
    body = header
    offsets = [0]
    for item in objects:
        offsets.append(len(body))
        body += item
    xref_start = len(body)
    xref = [b"xref\n", f"0 {len(offsets)}\n".encode(), b"0000000000 65535 f \n"]
    xref.extend(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    trailer = (
        b"trailer\n"
        + f"<< /Size {len(offsets)} /Root 1 0 R >>\n".encode()
        + b"startxref\n"
        + f"{xref_start}\n".encode()
        + b"%%EOF\n"
    )
    return body + b"".join(xref) + trailer


@pytest.mark.unit
def test_markdown_resume_chunks_canonical_sections_with_line_anchors() -> None:
    parsed = parse_resume_text(
        _MARKDOWN_CV,
        title="ignored",
        canonical_uri="file:///tmp/cv.md",
        source_type=EvidenceSourceType.RESUME,
    )

    labels = [chunk.section_label for chunk in parsed.chunks]
    assert labels[0] == "Summary"
    assert "Experience" in labels
    assert "Projects" in labels
    assert "Skills" in labels
    assert "Education" in labels
    assert parsed.title == "ignored"

    experience = next(chunk for chunk in parsed.chunks if chunk.section_label == "Experience")
    assert "FastAPI" in experience.content
    assert "pgvector" in experience.content
    assert experience.start_line >= 1
    assert experience.end_line >= experience.start_line
    assert experience.anchor.startswith("Experience:L")
    assert parsed.metadata["chunk_anchors"][0]["start_line"] == parsed.chunks[0].start_line


@pytest.mark.unit
def test_plain_text_all_caps_headings_map_to_canonical_sections() -> None:
    chunks = chunk_document(_PLAIN_CV, canonicalize_resume=True)
    labels = {chunk.section_label for chunk in chunks}
    assert {"Experience", "Projects", "Skills", "Education"}.issubset(labels)
    assert canonicalize_resume_section("Work Experience") == "Experience"
    assert canonicalize_resume_section("Technical Skills") == "Skills"


@pytest.mark.unit
def test_parse_resume_file_markdown_and_text_round_trip(tmp_path: Path) -> None:
    md_path = tmp_path / "profile.md"
    md_path.write_text(_MARKDOWN_CV, encoding="utf-8")
    parsed = parse_resume_file(md_path, source_type=EvidenceSourceType.RESUME)

    assert parsed.title == "Jane Doe"
    assert parsed.path == md_path.resolve()
    assert parsed.canonical_uri == md_path.resolve().as_uri()
    assert len(parsed.content_sha256) == 64
    assert parsed.source_type is EvidenceSourceType.RESUME
    assert any(chunk.section_label == "Experience" for chunk in parsed.chunks)

    txt_path = tmp_path / "profile.txt"
    txt_path.write_text(_PLAIN_CV, encoding="utf-8")
    plain = parse_resume_file(txt_path, source_type=EvidenceSourceType.RESUME)
    assert {chunk.section_label for chunk in plain.chunks} >= {
        "Experience",
        "Projects",
        "Skills",
        "Education",
    }


@pytest.mark.unit
def test_pdf_resume_extracts_sections_and_page_anchors(tmp_path: Path) -> None:
    pdf_bytes = _build_text_pdf(
        [
            "Experience",
            "Built FastAPI services with pgvector.",
            "Education",
            "B.S. Computer Science",
        ]
    )
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf_bytes)).pages)
    assert "Experience" in extracted
    assert "FastAPI" in extracted

    pdf_path = tmp_path / "profile.pdf"
    pdf_path.write_bytes(pdf_bytes)
    parsed = parse_resume_file(pdf_path, source_type=EvidenceSourceType.RESUME)
    labels = {chunk.section_label for chunk in parsed.chunks}
    assert "Experience" in labels
    assert "Education" in labels
    experience = next(chunk for chunk in parsed.chunks if chunk.section_label == "Experience")
    assert experience.start_page == 1
    assert "p1-" in experience.anchor


@pytest.mark.unit
def test_oversized_section_is_split_and_keeps_section_label() -> None:
    body = "Projects\n\n" + ("ProofPath ships cited FastAPI agents. " * 200)
    chunks = chunk_document(body, canonicalize_resume=True)
    project_chunks = [chunk for chunk in chunks if chunk.section_label == "Projects"]
    assert len(project_chunks) >= 2
    assert all(len(chunk.content) <= MAX_CHUNK_CHARS + 50 for chunk in project_chunks)


@pytest.mark.unit
def test_empty_and_unsupported_documents_fail(tmp_path: Path) -> None:
    empty = tmp_path / "blank.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        parse_resume_file(empty)

    binary = tmp_path / "notes.docx"
    binary.write_bytes(b"not-a-resume")
    with pytest.raises(UnsupportedDocumentError):
        parse_resume_file(binary)

    with pytest.raises(FileNotFoundError):
        parse_resume_file(tmp_path / "missing.md")


@pytest.mark.unit
def test_infer_source_type_from_filename() -> None:
    assert infer_source_type(Path("My_Resume.pdf")) is EvidenceSourceType.RESUME
    assert infer_source_type(Path("README.md")) is EvidenceSourceType.README
    assert infer_source_type(Path("project-notes.md")) is EvidenceSourceType.EXPERIENCE_NOTE
