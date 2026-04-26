from pathlib import Path

import pytest

from scripts.static_assets import resolve_static_dir


def test_resolve_static_dir_falls_back_to_packaged_frontend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    static_dir = resolve_static_dir("frontend/")

    assert static_dir.name == "frontend"
    assert (static_dir / "index.html").is_file()
    assert (static_dir / "app.js").is_file()
    assert (static_dir / "style.css").is_file()
