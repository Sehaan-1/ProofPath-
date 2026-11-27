"""Evidence ingestion, search and persisted read endpoints."""

from __future__ import annotations

import asyncio
import io
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pypdf import PdfReader
from sqlalchemy.ext.asyncio import AsyncSession

from proofpath.config import LLMProvider
from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.db.repositories.opportunity import OpportunityRepo
from proofpath.db.repositories.tracker import TrackerRepo
from proofpath.ingestion.analyse_role import AnalyseRoleExtractor, create_role_parser
from proofpath.ingestion.github_ingester import (
    GitHubEvidenceClient,
    GitHubIngestError,
    GitHubRepoRef,
    snapshot_to_document,
)
from proofpath.ingestion.job_fetcher import EmptyJobDescriptionError, JobPageFetchError
from proofpath.ingestion.markdown_ingester import IngestStatus, persist_parsed_document
from proofpath.ingestion.resume_parser import ParsedDocument, hash_bytes, parse_resume_text
from proofpath.ingestion.secure_job_fetcher import SecureJobPageFetcher
from proofpath.retrieval.embeddings import create_embedding_client
from proofpath.retrieval.hybrid_search import HybridSearcher
from proofpath.schemas.actions import TrackerRecord
from proofpath.schemas.evidence import (
    EvidenceSearchResponse,
    IngestGitHubRequest,
    IngestResponse,
    IngestTextRequest,
)
from proofpath.schemas.opportunity import (
    OpportunityCreateRequest,
    OpportunityResponse,
    RequirementRecord,
)

router = APIRouter(prefix="/api/v1")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 500_000
MAX_PDF_PAGES = 100


async def session_dependency(request: Request) -> AsyncIterator[AsyncSession]:
    """One atomic transaction per request; failed ingestion rolls back all writes."""
    async with request.app.state.session_factory.begin() as session:
        yield session


Session = Annotated[AsyncSession, Depends(session_dependency)]


async def _ingest(
    document: ParsedDocument, request: Request, session: AsyncSession
) -> IngestResponse:
    repo = EvidenceRepo(session)
    # Never replace chunks that may already be cited by persisted verdicts.
    existing = await repo.get_source_by_sha256(document.content_sha256)
    if existing is None and await repo.get_source_by_uri(document.canonical_uri) is not None:
        raise HTTPException(409, "Evidence URI already exists with different content.")
    try:
        result = await persist_parsed_document(repo, document, replace_existing=False)
    except ValueError as exc:
        if str(exc) == "Evidence URI already exists with different content.":
            raise HTTPException(409, "Evidence URI already exists with different content.") from exc
        raise
    chunks = await repo.list_chunks_for_source(result.source_id)
    unembedded = [chunk for chunk in chunks if chunk.embedding is None]
    if unembedded:
        embedder = create_embedding_client(request.app.state.settings)
        vectors = await embedder.embed([chunk.content for chunk in unembedded])
        for chunk, vector in zip(unembedded, vectors, strict=True):
            chunk.embedding = vector
        await session.flush()
    # Deduplication may return a different original URI and source type.
    source = await repo.get_source_by_id(result.source_id)
    if source is None:
        raise RuntimeError("Ingested evidence source disappeared")
    return IngestResponse(
        source_id=result.source_id,
        created=result.status is IngestStatus.CREATED,
        chunk_count=result.chunk_count,
        content_sha256=result.content_sha256,
        source_type=source.source_type,
        canonical_uri=source.canonical_uri,
    )


@router.post("/evidence/ingest/text", response_model=IngestResponse, status_code=201)
async def ingest_text(
    payload: IngestTextRequest, request: Request, session: Session
) -> IngestResponse:
    try:
        document = parse_resume_text(
            payload.content,
            title=payload.title,
            canonical_uri=payload.canonical_uri,
            source_type=payload.source_type,
        )
    except ValueError as exc:
        raise HTTPException(422, "Document contains no usable text.") from exc
    return await _ingest(document, request, session)


def _uploaded_document(
    raw: bytes, filename: str, canonical_uri: str, source_type: EvidenceSourceType
) -> ParsedDocument:
    suffix = PurePath(filename).suffix.casefold()
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        raise ValueError("Unsupported file type")
    page_for_line = None
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted or len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError("Encrypted or oversized PDF")
        lines: list[str] = []
        page_for_line = []
        for page_no, page in enumerate(reader.pages, 1):
            page_lines = (
                (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
            )
            lines.extend([*page_lines, ""])
            page_for_line.extend([page_no] * (len(page_lines) + 1))
            if sum(map(len, lines)) > MAX_TEXT_CHARS:
                raise ValueError("Extracted document is too large")
        text = "\n".join(lines)
    else:
        text = raw.decode("utf-8-sig")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Extracted document is too large")
    return parse_resume_text(
        text,
        title=PurePath(filename).name,
        canonical_uri=canonical_uri,
        content_sha256=hash_bytes(raw),
        source_type=source_type,
        page_for_line=page_for_line,
    )


@router.post("/evidence/ingest/file", response_model=IngestResponse, status_code=201)
async def ingest_file(
    request: Request,
    session: Session,
    file: Annotated[UploadFile, File()],
    canonical_uri: Annotated[str, Form(min_length=1, max_length=2048)],
    source_type: Annotated[EvidenceSourceType, Form()] = EvidenceSourceType.RESUME,
) -> IngestResponse:
    try:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File exceeds upload size limit.")
        if not file.filename:
            raise HTTPException(422, "Filename is required.")
        try:
            document = await asyncio.to_thread(
                _uploaded_document, raw, file.filename, canonical_uri, source_type
            )
        except Exception as exc:
            raise HTTPException(422, "Unsupported or invalid document.") from exc
        return await _ingest(document, request, session)
    finally:
        await file.close()


@router.post("/evidence/ingest/github", response_model=IngestResponse, status_code=201)
async def ingest_github(
    payload: IngestGitHubRequest, request: Request, session: Session
) -> IngestResponse:
    try:
        candidate = GitHubRepoRef.from_request(payload)
        if not all(
            re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", part) and part not in {".", ".."}
            for part in (candidate.owner, candidate.name)
        ):
            raise ValueError("Invalid GitHub owner or repository")
        ref = candidate
    except ValueError as exc:
        raise HTTPException(422, "Invalid GitHub repository reference.") from exc
    settings = request.app.state.settings
    token = settings.github_token.get_secret_value() if settings.github_token else None
    try:
        async with GitHubEvidenceClient(token=token) as client:
            snapshot = await client.fetch_snapshot(ref)
    except GitHubIngestError as exc:
        raise HTTPException(502, "GitHub repository could not be fetched.") from exc
    return await _ingest(snapshot_to_document(snapshot), request, session)


@router.get("/evidence/search", response_model=EvidenceSearchResponse)
async def search_evidence(
    request: Request,
    session: Session,
    query: Annotated[str, Query(min_length=1, max_length=2000)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 8,
) -> EvidenceSearchResponse:
    cleaned = query.strip()
    if not cleaned:
        raise HTTPException(422, "Search query must not be blank.")
    searcher = HybridSearcher(session, create_embedding_client(request.app.state.settings))
    return EvidenceSearchResponse(
        query=cleaned, results=await searcher.search(cleaned, top_k=top_k)
    )


@router.get("/opportunities/{opportunity_id}", response_model=OpportunityResponse)
async def get_opportunity(opportunity_id: uuid.UUID, session: Session) -> OpportunityResponse:
    opportunity = await OpportunityRepo(session).get_by_id(
        opportunity_id, include_requirements=True
    )
    if opportunity is None:
        raise HTTPException(404, "Opportunity not found.")
    if any(not item.entity for item in opportunity.requirements):
        raise HTTPException(
            409, "Requirement entities must be recovered before reading this opportunity."
        )
    return OpportunityResponse(
        id=opportunity.id,
        company=opportunity.company,
        role_title=opportunity.role_title,
        source_url=opportunity.source_url,
        created_at=opportunity.created_at,
        requirements=[
            RequirementRecord(
                id=row.id,
                opportunity_id=row.opportunity_id,
                requirement_text=row.requirement_text,
                entity=row.entity,
                category=row.category,
                priority=row.priority,
            )
            for row in opportunity.requirements
            if row.entity
        ],
    )


@router.get("/tracker", response_model=list[TrackerRecord])
async def list_tracker(
    session: Session,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[TrackerRecord]:
    return [
        TrackerRecord.model_validate(row)
        for row in await TrackerRepo(session).list_entries(limit=limit, offset=offset)
    ]


@router.post("/opportunities", response_model=OpportunityResponse, status_code=201)
async def create_opportunity(
    payload: OpportunityCreateRequest, request: Request, session: Session
) -> OpportunityResponse:
    """Extract and persist requirements from pasted text or a public posting."""
    settings = request.app.state.settings
    if settings.llm_provider is not LLMProvider.OPENAI:
        raise HTTPException(501, "Role parsing is supported only with OpenAI.")
    extractor = AnalyseRoleExtractor(OpportunityRepo(session), create_role_parser(settings))
    try:
        result = await extractor.analyse(payload, fetcher=SecureJobPageFetcher())
    except EmptyJobDescriptionError as exc:
        raise HTTPException(422, "Job description is empty.") from exc
    except JobPageFetchError as exc:
        raise HTTPException(422, "Job posting could not be fetched safely.") from exc
    return result.to_response()
