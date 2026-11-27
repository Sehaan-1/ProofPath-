"""Unit tests validating the ProofPath project manifest and tooling setup."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import httpx
import pytest
import respx

import proofpath

PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _package_name(requirement: str) -> str:
    """Extract normalized PEP 503 package name (without extras/specifiers/markers)."""
    base = requirement.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9._-]+)", base)
    assert match is not None, f"Could not parse package name from requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.unit
def test_package_exposes_version() -> None:
    assert isinstance(proofpath.__version__, str)
    assert proofpath.__version__


@pytest.mark.unit
def test_core_runtime_dependencies_are_pinned_and_bounded() -> None:
    manifest = _load_pyproject()
    project = manifest["project"]
    assert isinstance(project, dict)

    raw_deps = project["dependencies"]
    assert isinstance(raw_deps, list)

    dep_names = {_package_name(dep) for dep in raw_deps}
    expected_core = {
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "sqlalchemy",
        "asyncpg",
        "alembic",
        "pgvector",
        "langgraph",
        "langgraph-checkpoint-postgres",
        "psycopg",
        "mcp",
        "httpx",
        "redis",
        "openai",
        "langchain-openai",
    }
    assert expected_core.issubset(dep_names)

    # Every core dependency must declare both lower and upper bounds.
    for dep in raw_deps:
        assert ">=" in dep, f"Expected lower bound for {dep!r}"
        assert "<" in dep, f"Expected upper bound for {dep!r}"


@pytest.mark.unit
def test_optional_extras_are_isolated_from_core_dependencies() -> None:
    manifest = _load_pyproject()
    project = manifest["project"]
    assert isinstance(project, dict)

    core_names = {_package_name(dep) for dep in project["dependencies"]}
    isolated_packages = {"giskard", "guardrails-ai", "playwright", "pytest-playwright"}
    assert core_names.isdisjoint(isolated_packages)

    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)

    assert "giskard" in {_package_name(dep) for dep in extras["evals"]}
    assert "guardrails-ai" in {_package_name(dep) for dep in extras["guardrails"]}
    assert "playwright" in {_package_name(dep) for dep in extras["e2e"]}

    dev_names = {_package_name(dep) for dep in extras["dev"]}
    assert {"ruff", "mypy", "pytest", "pytest-asyncio", "respx"}.issubset(dev_names)


@pytest.mark.unit
def test_dev_tooling_is_configured_in_pyproject() -> None:
    manifest = _load_pyproject()
    tool = manifest["tool"]
    assert isinstance(tool, dict)

    assert "ruff" in tool
    assert "mypy" in tool
    assert "pytest" in tool

    mypy_cfg = tool["mypy"]
    assert isinstance(mypy_cfg, dict)
    assert mypy_cfg["strict"] is True

    pytest_cfg = tool["pytest"]
    assert isinstance(pytest_cfg, dict)
    ini_options = pytest_cfg["ini_options"]
    assert isinstance(ini_options, dict)
    assert ini_options["asyncio_mode"] == "auto"


@pytest.mark.unit
async def test_async_respx_unit_mocking_works() -> None:
    """Verify pytest-asyncio auto mode and respx work cleanly for unit HTTP edge cases."""
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.example.test/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.test/health")

    assert route.called
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
