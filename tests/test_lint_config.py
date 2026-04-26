"""Regression tests for lint rules that protect test-suite maintainability."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_e2e_tests_have_no_mutable_class_attributes() -> None:
    """The e2e suite should stay clean under Ruff's RUF012 rule."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--select",
            "RUF012",
            "tests/test_e2e.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
