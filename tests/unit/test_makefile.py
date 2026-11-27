"""Unit tests verifying ProofPath Makefile developer ergonomics targets."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"

REQUIRED_TARGETS = (
    "up",
    "down",
    "migrate",
    "test-unit",
    "test-int",
    "mcp-dev",
    "eval",
)


@pytest.mark.unit
def test_makefile_declares_all_required_targets() -> None:
    assert MAKEFILE_PATH.exists()
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    declared = set(re.findall(r"^([a-zA-Z0-9_-]+):", content, flags=re.MULTILINE))
    assert set(REQUIRED_TARGETS).issubset(declared)

    phony_matches = re.findall(r"^\.PHONY:\s*(.+)$", content, flags=re.MULTILINE)
    phony_targets = {target for line in phony_matches for target in line.split()}
    assert set(REQUIRED_TARGETS).issubset(phony_targets)


@pytest.mark.unit
@pytest.mark.parametrize("target", REQUIRED_TARGETS)
def test_makefile_targets_parse_cleanly_in_dry_run(target: str) -> None:
    make_bin = shutil.which("make")
    if make_bin is None:
        pytest.skip("GNU make is not installed in this environment")

    result = subprocess.run(
        [make_bin, "-n", target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"`make -n {target}` failed: {result.stderr}"
    assert result.stdout.strip(), f"`make -n {target}` produced no recipe output"
