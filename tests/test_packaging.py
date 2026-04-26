"""Packaging regression tests."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


def test_wheel_includes_frontend_static_assets(tmp_path: Path) -> None:
    """A wheel install must ship the frontend served by the entrypoints."""
    repo_root = Path(__file__).parents[1]
    wheel_dir = tmp_path / "wheelhouse"

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    expected_assets = {
        "frontend/index.html",
        "frontend/app.js",
        "frontend/style.css",
    }
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_paths = set(wheel.namelist())

    assert expected_assets <= wheel_paths
