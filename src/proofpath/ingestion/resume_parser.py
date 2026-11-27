"""Resume/CV parser for Markdown, plain text, and PDF (Sub-part 1.3.1a)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pypdf import PdfReader

from proofpath.db.models import EvidenceSourceType
from proofpath.ingestion.chunking import ParsedChunk, chunk_document

SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".markdown", ".txt", ".pdf"})
_TEXT_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".markdown", ".txt"})


class UnsupportedDocumentError(ValueError):
    """Raised when a local document type is not Markdown, plain text, or PDF."""


class EmptyDocumentError(ValueError):
    """Raised when a document contains no extractable text."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parsed local document ready to persist on the Evidence Ledger."""

    title: str
    path: Path | None
    canonical_uri: str
    content_sha256: str
    text: str
    chunks: tuple[ParsedChunk, ...]
    source_type: EvidenceSourceType
    metadata: dict[str, Any]


def hash_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def parse_resume_file(
    path: Path | str,
    *,
    source_type: EvidenceSourceType | None = None,
) -> ParsedDocument:
    """Load a local resume/CV/note file and return section-aware chunks with anchors."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Document not found: {resolved}")
    suffix = resolved.suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocumentError(
            f"Unsupported document type {suffix!r}; expected Markdown, .txt, or PDF."
        )

    raw = resolved.read_bytes()
    digest = hash_bytes(raw)
    inferred = source_type if source_type is not None else infer_source_type(resolved)
    canonicalize = inferred is EvidenceSourceType.RESUME

    if suffix == ".pdf":
        text, page_for_line = _extract_pdf_text(resolved)
    else:
        text = _decode_text(raw)
        page_for_line = None

    return parse_resume_text(
        text,
        title=_infer_title(text, fallback=resolved.stem),
        canonical_uri=resolved.as_uri(),
        content_sha256=digest,
        source_type=inferred,
        path=resolved,
        canonicalize_resume=canonicalize,
        page_for_line=page_for_line,
        extra_metadata={"path": str(resolved), "suffix": suffix},
    )


def parse_resume_text(
    text: str,
    *,
    title: str,
    canonical_uri: str,
    content_sha256: str | None = None,
    source_type: EvidenceSourceType = EvidenceSourceType.RESUME,
    path: Path | None = None,
    canonicalize_resume: bool | None = None,
    page_for_line: list[int] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """Parse already-loaded document text into section-aware chunks."""
    if not text.strip():
        raise EmptyDocumentError("Document contains no extractable text.")

    use_canonical = (
        canonicalize_resume
        if canonicalize_resume is not None
        else source_type is EvidenceSourceType.RESUME
    )
    chunks = chunk_document(
        text,
        canonicalize_resume=use_canonical,
        page_for_line=page_for_line,
    )
    if not chunks:
        raise EmptyDocumentError("Document contained no chunkable text after section splitting.")

    digest = content_sha256 or hash_bytes(text.encode("utf-8"))
    anchors = [chunk.as_anchor_record(index) for index, chunk in enumerate(chunks)]
    metadata: dict[str, Any] = {
        "parser": "resume" if use_canonical else "markdown",
        "chunk_anchors": anchors,
        "chunk_count": len(chunks),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return ParsedDocument(
        title=title.strip()[:512] or "Untitled evidence",
        path=path,
        canonical_uri=canonical_uri,
        content_sha256=digest,
        text=text,
        chunks=tuple(chunks),
        source_type=source_type,
        metadata=metadata,
    )


def infer_source_type(path: Path) -> EvidenceSourceType:
    """Infer ledger source type from a local filename."""
    name = path.name.casefold()
    stem = path.stem.casefold()
    if "resume" in stem or stem == "cv" or stem.startswith("cv-") or stem.endswith("-cv"):
        return EvidenceSourceType.RESUME
    if name.startswith("readme"):
        return EvidenceSourceType.README
    return EvidenceSourceType.EXPERIENCE_NOTE


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _extract_pdf_text(path: Path) -> tuple[str, list[int]]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise UnsupportedDocumentError(f"Encrypted PDF cannot be ingested: {path}")

    all_lines: list[str] = []
    page_for_line: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        extracted = extracted.replace("\r\n", "\n").replace("\r", "\n")
        page_lines = extracted.split("\n")
        if all_lines:
            all_lines.append("")
            page_for_line.append(page_number)
        all_lines.extend(page_lines)
        page_for_line.extend([page_number] * len(page_lines))

    text = "\n".join(all_lines)
    if not text.strip():
        raise EmptyDocumentError(f"PDF contained no extractable text: {path}")
    return text, page_for_line


def _infer_title(text: str, *, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if line:
            return line[:512]
    return fallback[:512]
