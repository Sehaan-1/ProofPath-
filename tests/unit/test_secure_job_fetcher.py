"""SSRF defenses for URL ingestion."""

import asyncio
import socket
from unittest.mock import AsyncMock

import pytest

from proofpath.ingestion.job_fetcher import JobPageFetchError
from proofpath.ingestion.secure_job_fetcher import (
    PublicResolver,
    SecureJobPageFetcher,
    _validate_url,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.1/",
        "http://2130706433/",
        "http://0177.0.0.1/",
        "http://169.254.169.254/",
        "http://[::ffff:127.0.0.1]/",
        "http://[2002:7f00:1::]/",
        "http://user:pass@example.com/",
        "http://example.com:8080/",
        "file:///etc/passwd",
    ],
)
def test_rejects_unsafe_literal_urls(url):
    with pytest.raises(JobPageFetchError):
        _validate_url(url)


async def test_dns_rejects_mixed_answers_before_connection(monkeypatch):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop,
        "getaddrinfo",
        AsyncMock(
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
            ]
        ),
    )
    with pytest.raises(JobPageFetchError, match="non-public"):
        await PublicResolver().resolve("example.com", 443)
    loop.getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
    resolved = await PublicResolver().resolve("example.com", 443)
    assert resolved[0]["host"] == "8.8.8.8"
    assert resolved[0]["hostname"] == "example.com"
    assert resolved[0]["flags"] == socket.AI_NUMERICHOST


async def test_redirect_to_private_address_is_rejected(monkeypatch):
    calls = []

    class Response:
        status = 302

        def __init__(self):
            self.headers = {"Location": "http://127.0.0.1/admin"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def get(self, url, **kwargs):
            calls.append(url)
            assert kwargs["allow_redirects"] is False
            return Response()

    monkeypatch.setattr("proofpath.ingestion.secure_job_fetcher.aiohttp.ClientSession", Client)
    monkeypatch.setattr(
        "proofpath.ingestion.secure_job_fetcher.aiohttp.TCPConnector", lambda **_kwargs: object()
    )
    with pytest.raises(JobPageFetchError):
        await SecureJobPageFetcher().fetch("https://example.com/start")
    assert calls == ["https://example.com/start"]


async def test_public_redirect_fetches_bounded_body(monkeypatch):
    calls = []

    class Response:
        def __init__(self, status, headers, body=b""):
            self.status = status
            self.headers = headers
            self.charset = "utf-8"
            self.content = self
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def iter_chunked(self, _size):
            yield self.body

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def get(self, url, **_kwargs):
            calls.append(url)
            if len(calls) == 1:
                return Response(302, {"Location": "/job"})
            return Response(200, {"content-type": "text/html"}, b"<main>Python developer</main>")

    monkeypatch.setattr("proofpath.ingestion.secure_job_fetcher.aiohttp.ClientSession", Client)
    monkeypatch.setattr(
        "proofpath.ingestion.secure_job_fetcher.aiohttp.TCPConnector", lambda **_kwargs: object()
    )
    result = await SecureJobPageFetcher().fetch("https://example.com/start")
    assert result.source_url == "https://example.com/job"
    assert "Python developer" in result.text
    assert calls == ["https://example.com/start", "https://example.com/job"]
