"""Packaging regression tests."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_install_resolves_frontend_static_assets(tmp_path: Path) -> None:
    """A wheel install must resolve the frontend served by the entrypoints."""
    repo_root = Path(__file__).parents[1]
    wheel_dir = tmp_path / "wheelhouse"
    install_dir = tmp_path / "installed"

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--target", str(install_dir), wheels[0]],
        check=True,
        text=True,
        capture_output=True,
    )

    probe = """
from pathlib import Path

from fastapi.testclient import TestClient
from scripts.backend import BrowserServer, SharedState, create_app
from scripts.static_assets import resolve_static_dir

static_dir = resolve_static_dir()
for asset in ("index.html", "app.js", "style.css"):
    path = static_dir / asset
    assert path.is_file(), path
print(static_dir)

state = SharedState()
app = create_app(state, BrowserServer(state), static_dir)
response = TestClient(app).get("/")
assert response.status_code == 200, response.text
assert "Neural Data Viewer" in response.text
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env={**dict(os.environ), "PYTHONPATH": str(install_dir)},
        check=True,
        text=True,
        capture_output=True,
    )
    resolved_static_dir = Path(result.stdout.strip())
    assert resolved_static_dir.is_relative_to(install_dir)

    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_paths = set(wheel.namelist())

    assert {
        "scripts/frontend/index.html",
        "scripts/frontend/app.js",
        "scripts/frontend/style.css",
    } <= wheel_paths
