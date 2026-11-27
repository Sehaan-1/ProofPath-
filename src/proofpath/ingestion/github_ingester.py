"""Live GitHub REST API evidence ingester (Sub-slice 1.3.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Self
from urllib.parse import urlparse
from uuid import UUID

import httpx

from proofpath.db.models import EvidenceSourceType
from proofpath.db.repositories.evidence import EvidenceRepo
from proofpath.ingestion.chunking import ParsedChunk
from proofpath.ingestion.markdown_ingester import IngestStatus, persist_parsed_document
from proofpath.ingestion.resume_parser import ParsedDocument, hash_bytes, parse_resume_text
from proofpath.schemas.evidence import IngestGitHubRequest

GITHUB_API_URL: Final[str] = "https://api.github.com"
GITHUB_API_VERSION: Final[str] = "2022-11-28"
GITHUB_USER_AGENT: Final[str] = "ProofPath/0.1"
_MAX_BODY_CHARS: Final[int] = 4000
_MAX_README_CHARS: Final[int] = 200_000
_DEFAULT_ITEM_LIMIT: Final[int] = 5

_RELEASE_HEADING = re.compile(r"^Release (.+)$")
_PR_HEADING = re.compile(r"^Pull Request (\d+)$")
_SLUG_STRIP = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_SLUG_SPACE = re.compile(r"[-\s]+")


class GitHubIngestError(RuntimeError):
    """Raised when the GitHub REST API cannot be used to ingest a repository."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class GitHubRepoRef:
    """Owner/name pair for a GitHub repository."""

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def html_url(self) -> str:
        """Public provenance URL: https://github.com/<owner>/<repo>."""
        return f"https://github.com/{self.owner}/{self.name}"

    @classmethod
    def parse(cls, value: str) -> GitHubRepoRef:
        """Parse a URL, SSH remote, or ``owner/name`` slug into a repo ref."""
        raw = value.strip().removesuffix(".git").rstrip("/")
        if raw.startswith("git@github.com:"):
            raw = raw.removeprefix("git@github.com:")
        elif "github.com" in raw.casefold():
            parsed = urlparse(raw if "://" in raw else f"https://{raw}")
            if parsed.netloc.casefold() not in {"github.com", "www.github.com"}:
                raise ValueError(f"Invalid GitHub repository reference: {value!r}")
            raw = parsed.path.lstrip("/")
        parts = [segment for segment in raw.split("/") if segment]
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository reference: {value!r}")
        owner, name = parts[0], parts[1]
        if not _is_github_name(owner) or not _is_github_name(name):
            raise ValueError(f"Invalid GitHub repository reference: {value!r}")
        return cls(owner=owner, name=name)

    @classmethod
    def from_request(cls, request: IngestGitHubRequest) -> GitHubRepoRef:
        """Build a ref from the API ingest request schema."""
        if request.owner and request.name:
            return cls(owner=request.owner, name=request.name)
        if request.repo_url is None:
            raise ValueError("Provide repo_url or both owner and name.")
        return cls.parse(str(request.repo_url))


@dataclass(frozen=True, slots=True)
class GitHubRelease:
    tag_name: str
    name: str
    body: str
    html_url: str


@dataclass(frozen=True, slots=True)
class GitHubPullRequest:
    number: int
    title: str
    body: str
    html_url: str


@dataclass(frozen=True, slots=True)
class GitHubRepoSnapshot:
    """Fetched GitHub evidence prior to ledger persistence."""

    ref: GitHubRepoRef
    description: str
    default_branch: str
    html_url: str
    stargazers_count: int
    languages: tuple[str, ...]
    readme_markdown: str
    releases: tuple[GitHubRelease, ...]
    pull_requests: tuple[GitHubPullRequest, ...]


@dataclass(frozen=True, slots=True)
class GitHubIngestResult:
    """Outcome of ingesting one GitHub repository into the Evidence Ledger."""

    owner: str
    name: str
    canonical_uri: str
    content_sha256: str
    status: IngestStatus
    source_id: UUID
    chunk_count: int
    languages: tuple[str, ...]
    html_url: str


class GitHubEvidenceClient:
    """Async GitHub REST client for repository evidence (Sub-part 1.3.2a)."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = GITHUB_API_URL,
        timeout: float = 20.0,
    ) -> None:
        self._token = token.strip() if token else None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._external_client = client
        self._client: httpx.AsyncClient | None = client

    async def __aenter__(self) -> Self:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers(),
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._external_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self, *, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": GITHUB_USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise GitHubIngestError(
                "GitHubEvidenceClient must be used as an async context manager."
            )
        return self._client

    async def fetch_snapshot(
        self,
        ref: GitHubRepoRef,
        *,
        release_limit: int = _DEFAULT_ITEM_LIMIT,
        pull_limit: int = _DEFAULT_ITEM_LIMIT,
    ) -> GitHubRepoSnapshot:
        """Fetch metadata, languages, README, recent releases, and recent PRs."""
        repo = await self._get_json(f"/repos/{ref.owner}/{ref.name}")
        languages = await self._get_json(f"/repos/{ref.owner}/{ref.name}/languages")
        readme = await self._get_readme(ref)
        releases_payload = await self._get_json(
            f"/repos/{ref.owner}/{ref.name}/releases",
            params={"per_page": release_limit},
        )
        pulls_payload = await self._get_json(
            f"/repos/{ref.owner}/{ref.name}/pulls",
            params={"state": "all", "sort": "updated", "direction": "desc", "per_page": pull_limit},
        )
        html_url = str(repo.get("html_url") or ref.html_url).rstrip("/")
        language_names = (
            tuple(str(name) for name in languages) if isinstance(languages, dict) else ()
        )
        return GitHubRepoSnapshot(
            ref=ref,
            description=str(repo.get("description") or "").strip(),
            default_branch=str(repo.get("default_branch") or "main"),
            html_url=html_url,
            stargazers_count=int(repo.get("stargazers_count") or 0),
            languages=language_names,
            readme_markdown=readme,
            releases=tuple(
                _parse_release(item) for item in _as_list(releases_payload)[:release_limit]
            ),
            pull_requests=tuple(_parse_pull(item) for item in _as_list(pulls_payload)[:pull_limit]),
        )

    async def _get_readme(self, ref: GitHubRepoRef) -> str:
        client = self._require_client()
        response = await client.get(
            f"{self._base_url}/repos/{ref.owner}/{ref.name}/readme",
            headers=self._headers(accept="application/vnd.github.raw+json"),
        )
        if response.status_code == 404:
            return ""
        self._raise_for_status(response, f"Failed to fetch README for {ref.full_name}")
        text = response.text.strip()
        return text[:_MAX_README_CHARS]

    async def _get_json(self, path: str, *, params: dict[str, int | str] | None = None) -> Any:
        client = self._require_client()
        response = await client.get(
            f"{self._base_url}{path}",
            params=params,
            headers=self._headers(),
        )
        self._raise_for_status(response, f"GitHub API request failed for {path}")
        return response.json()

    def _raise_for_status(self, response: httpx.Response, message: str) -> None:
        if response.is_success:
            return
        remaining = response.headers.get("X-RateLimit-Remaining")
        detail = f"{message} (HTTP {response.status_code})"
        if response.status_code == 403 and remaining == "0":
            detail = "GitHub API rate limit exceeded. Set GITHUB_TOKEN for a higher quota."
        elif response.status_code == 401:
            detail = "GitHub API authentication failed. Check GITHUB_TOKEN."
        elif response.status_code == 404:
            detail = f"{message}: repository or resource not found."
        raise GitHubIngestError(detail, status_code=response.status_code)


class GitHubEvidenceIngester:
    """Fetch a GitHub repo and persist cited evidence on the ledger (1.3.2a+b)."""

    def __init__(self, evidence_repo: EvidenceRepo, client: GitHubEvidenceClient) -> None:
        self._repo = evidence_repo
        self._client = client

    async def ingest(
        self,
        repo: str | GitHubRepoRef | IngestGitHubRequest,
        *,
        release_limit: int = _DEFAULT_ITEM_LIMIT,
        pull_limit: int = _DEFAULT_ITEM_LIMIT,
    ) -> GitHubIngestResult:
        """Ingest one repository, skipping unchanged SHA-256 payloads."""
        ref = _coerce_ref(repo)
        snapshot = await self._client.fetch_snapshot(
            ref,
            release_limit=release_limit,
            pull_limit=pull_limit,
        )
        document = snapshot_to_document(snapshot)
        persisted = await persist_parsed_document(self._repo, document)
        return GitHubIngestResult(
            owner=ref.owner,
            name=ref.name,
            canonical_uri=document.canonical_uri,
            content_sha256=document.content_sha256,
            status=persisted.status,
            source_id=persisted.source_id,
            chunk_count=persisted.chunk_count,
            languages=snapshot.languages,
            html_url=snapshot.html_url,
        )


def snapshot_to_document(snapshot: GitHubRepoSnapshot) -> ParsedDocument:
    """Turn a GitHub snapshot into a section-chunked ledger document."""
    markdown = render_repo_markdown(snapshot)
    document = parse_resume_text(
        markdown,
        title=snapshot.ref.full_name,
        canonical_uri=snapshot.html_url,
        content_sha256=hash_bytes(markdown.encode("utf-8")),
        source_type=EvidenceSourceType.GITHUB_REPO,
        canonicalize_resume=False,
        extra_metadata={
            "parser": "github",
            "owner": snapshot.ref.owner,
            "name": snapshot.ref.name,
            "html_url": snapshot.html_url,
            "default_branch": snapshot.default_branch,
            "description": snapshot.description,
            "languages": list(snapshot.languages),
            "stargazers_count": snapshot.stargazers_count,
        },
    )
    anchors = [
        _github_anchor_record(snapshot, chunk, index) for index, chunk in enumerate(document.chunks)
    ]
    metadata = dict(document.metadata)
    metadata["chunk_anchors"] = anchors
    metadata["parser"] = "github"
    return ParsedDocument(
        title=document.title,
        path=None,
        canonical_uri=snapshot.html_url,
        content_sha256=document.content_sha256,
        text=document.text,
        chunks=document.chunks,
        source_type=EvidenceSourceType.GITHUB_REPO,
        metadata=metadata,
    )


def render_repo_markdown(snapshot: GitHubRepoSnapshot) -> str:
    """Compose a single markdown evidence document from GitHub API payloads."""
    lines = [
        f"# {snapshot.ref.full_name}",
        "",
        snapshot.description or "GitHub repository evidence.",
        "",
        f"Repository: {snapshot.html_url}",
        f"Default branch: {snapshot.default_branch}",
        "",
        "## Languages",
        "",
    ]
    if snapshot.languages:
        lines.extend(f"- {language}" for language in snapshot.languages)
    else:
        lines.append("- (not reported)")
    lines.extend(["", "## README", ""])
    lines.append(snapshot.readme_markdown.strip() or "_No README.md found in this repository._")
    lines.extend(["", "## Recent Releases", ""])
    if snapshot.releases:
        for release in snapshot.releases:
            heading = release.tag_name or release.name or "release"
            lines.append(f"### Release {heading}")
            lines.append("")
            lines.append(release.html_url)
            lines.append("")
            if release.body:
                lines.append(_truncate(release.body))
                lines.append("")
    else:
        lines.append("_No releases published._")
        lines.append("")
    lines.extend(["## Recent Pull Requests", ""])
    if snapshot.pull_requests:
        for pull in snapshot.pull_requests:
            lines.append(f"### Pull Request {pull.number}")
            lines.append("")
            lines.append(f"{pull.title}")
            lines.append("")
            lines.append(pull.html_url)
            lines.append("")
            if pull.body:
                lines.append(_truncate(pull.body))
                lines.append("")
    else:
        lines.append("_No recent pull requests._")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def github_heading_slug(heading: str) -> str:
    """Approximate GitHub's heading-anchor slug for `#section` provenance URLs."""
    slug = _SLUG_STRIP.sub("", heading.strip().casefold())
    slug = _SLUG_SPACE.sub("-", slug).strip("-")
    return slug or "section"


def canonical_github_uri(owner: str, name: str, section: str | None = None) -> str:
    """Build https://github.com/<owner>/<repo> or https://github.com/<owner>/<repo>#<section>."""
    base = f"https://github.com/{owner}/{name}"
    if section:
        return f"{base}#{github_heading_slug(section)}"
    return base


def _github_anchor_record(
    snapshot: GitHubRepoSnapshot,
    chunk: ParsedChunk,
    chunk_index: int,
) -> dict[str, object]:
    record = chunk.as_anchor_record(chunk_index)
    record["canonical_uri"] = _chunk_canonical_uri(snapshot, chunk)
    return record


def _chunk_canonical_uri(snapshot: GitHubRepoSnapshot, chunk: ParsedChunk) -> str:
    label = (chunk.section_label or "").strip()
    release_match = _RELEASE_HEADING.match(label)
    if release_match:
        tag = release_match.group(1).strip()
        return f"{snapshot.html_url}/releases/tag/{tag}"
    pr_match = _PR_HEADING.match(label)
    if pr_match:
        return f"{snapshot.html_url}/pull/{pr_match.group(1)}"
    if label:
        return f"{snapshot.html_url}#{github_heading_slug(label)}"
    return snapshot.html_url


def _coerce_ref(repo: str | GitHubRepoRef | IngestGitHubRequest) -> GitHubRepoRef:
    if isinstance(repo, GitHubRepoRef):
        return repo
    if isinstance(repo, IngestGitHubRequest):
        return GitHubRepoRef.from_request(repo)
    return GitHubRepoRef.parse(repo)


def _is_github_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))


def _as_list(payload: Any) -> list[Any]:
    return list(payload) if isinstance(payload, list) else []


def _parse_release(item: Any) -> GitHubRelease:
    data = item if isinstance(item, dict) else {}
    tag = str(data.get("tag_name") or "").strip() or "untagged"
    name = str(data.get("name") or tag).strip()
    return GitHubRelease(
        tag_name=tag,
        name=name,
        body=str(data.get("body") or "").strip(),
        html_url=str(data.get("html_url") or "").strip(),
    )


def _parse_pull(item: Any) -> GitHubPullRequest:
    data = item if isinstance(item, dict) else {}
    number = int(data.get("number") or 0)
    return GitHubPullRequest(
        number=number,
        title=str(data.get("title") or f"PR #{number}").strip(),
        body=str(data.get("body") or "").strip(),
        html_url=str(data.get("html_url") or "").strip(),
    )


def _truncate(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _MAX_BODY_CHARS:
        return stripped
    return stripped[: _MAX_BODY_CHARS - 1].rstrip() + "…"
