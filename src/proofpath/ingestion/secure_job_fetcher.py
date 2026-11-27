"""Job posting fetcher with DNS pinning, per-hop validation and bounded downloads."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.helpers import is_ip_address

from proofpath.ingestion.job_fetcher import (
    MAX_JOB_HTML_BYTES,
    PROOFPATH_USER_AGENT,
    EmptyJobDescriptionError,
    FetchedJobPage,
    JobPageFetcher,
    JobPageFetchError,
    assert_public_http_url,
    extract_job_posting_text,
)


def _public_address(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.sixtofour or ip.teredo:
            return False
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
    return ip.is_global


class PublicResolver(AbstractResolver):
    """Resolve once, reject mixed/private answers, and connect to the validated IPs."""

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[ResolveResult]:
        loop = asyncio.get_running_loop()
        try:
            results = await loop.getaddrinfo(host, port, family=family, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise JobPageFetchError("Job host could not be resolved.") from exc
        if not results or any(not _public_address(str(item[4][0])) for item in results):
            raise JobPageFetchError("Job host resolves to a non-public address.")
        return [
            ResolveResult(
                hostname=host,
                host=str(item[4][0]),
                port=port,
                family=item[0],
                proto=item[2],
                flags=socket.AI_NUMERICHOST,
            )
            for item in results
        ]

    async def close(self) -> None:
        pass


def _validate_url(url: str) -> str:
    checked = assert_public_http_url(url)
    parsed = urlsplit(checked)
    try:
        port = parsed.port
    except ValueError as exc:
        raise JobPageFetchError("Invalid URL port.") from exc
    if parsed.username is not None or parsed.password is not None:
        raise JobPageFetchError("Credentials in job URLs are not permitted.")
    if port is not None and port not in (80, 443):
        raise JobPageFetchError("Job URL port is not permitted.")
    host = parsed.hostname or ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # aiohttp may treat non-canonical numeric forms as IPs and skip its resolver.
        if is_ip_address(host):
            raise JobPageFetchError("Non-canonical IP literals are not permitted.") from None
    else:
        if not _public_address(str(address)):
            raise JobPageFetchError("Job URL points to a non-public address.")
    return checked


class SecureJobPageFetcher(JobPageFetcher):
    """Fetch without proxies or automatic redirects, validating every connection."""

    async def fetch(self, url: str) -> FetchedJobPage:
        current = _validate_url(url)
        timeout = aiohttp.ClientTimeout(total=20)
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False)
        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=False,
                headers={"User-Agent": PROOFPATH_USER_AGENT},
            ) as client:
                for _ in range(6):
                    async with client.get(current, allow_redirects=False) as response:
                        if response.status in (301, 302, 303, 307, 308):
                            location = response.headers.get("Location")
                            if not location:
                                raise JobPageFetchError("Job redirect has no destination.")
                            current = _validate_url(urljoin(current, location))
                            continue
                        if not 200 <= response.status < 300:
                            raise JobPageFetchError(
                                "Job posting request failed.", status_code=response.status
                            )
                        body = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            body.extend(chunk)
                            if len(body) > MAX_JOB_HTML_BYTES:
                                raise JobPageFetchError("Job posting exceeds download limit.")
                        content_type = response.headers.get("content-type", "")
                        try:
                            text = bytes(body).decode(response.charset or "utf-8", errors="replace")
                        except LookupError as exc:
                            raise JobPageFetchError("Invalid response character encoding.") from exc
                        extracted, title = extract_job_posting_text(text, content_type=content_type)
                        if not extracted:
                            raise EmptyJobDescriptionError(
                                "Fetched job posting contained no visible text."
                            )
                        return FetchedJobPage(
                            text=extracted,
                            source_url=current,
                            title=title,
                            content_type=content_type,
                        )
                raise JobPageFetchError("Too many job posting redirects.")
        except (TimeoutError, aiohttp.ClientError) as exc:
            raise JobPageFetchError("Job posting request failed.") from exc
