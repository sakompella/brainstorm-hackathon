"""Tests for scripts/launcher.py service launcher."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

from scripts import launcher


def test_default_launcher_uses_hard_dataset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data" / "hard"
    data_dir.mkdir(parents=True)
    (data_dir / "track2_data.parquet").touch()

    processes = [
        MagicMock(pid=1111, poll=MagicMock(return_value=None)),
        MagicMock(pid=2222, poll=MagicMock(return_value=None)),
    ]

    def start_process(
        name: str,
        args: tuple[str, ...],
        log_file: Path | None,
        cwd: Path,
        state: launcher.State,
    ) -> MagicMock:
        proc = processes.pop(0)
        state.processes.append((name, proc))
        return proc

    start_process_mock = MagicMock(side_effect=start_process)

    monkeypatch.setattr(sys, "argv", ["brainstorm-all"])
    monkeypatch.setattr(launcher, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "download_data", MagicMock())
    monkeypatch.setattr(launcher, "kill_ports", MagicMock())
    monkeypatch.setattr(launcher.time, "sleep", MagicMock())
    monkeypatch.setattr(launcher.atexit, "register", MagicMock())
    monkeypatch.setattr(launcher.signal, "signal", MagicMock())
    monkeypatch.setattr(launcher, "in_container", lambda: True)
    monkeypatch.setattr(launcher, "start_process", start_process_mock)
    monkeypatch.setattr(launcher, "wait_for_processes", MagicMock())

    launcher.main()

    stream_args = start_process_mock.call_args_list[0].args[1]
    assert stream_args == (
        sys.executable,
        "-m",
        "scripts.stream_data",
        "--from-file",
        f"{tmp_path / 'data' / 'hard'}/",
    )
