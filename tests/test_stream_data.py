<<<<<<< HEAD
"""Tests for scripts/stream_data.py streamer lifecycle behavior."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.stream_data import run_server_tasks


class _FakeUvicornServer:
    def __init__(self) -> None:
        self.should_exit = False
        self.serve_exited = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)
        self.serve_exited = True


@pytest.mark.asyncio
async def test_run_server_tasks_stops_uvicorn_when_stream_finishes() -> None:
    uvicorn_server = _FakeUvicornServer()

    stream_started = asyncio.Event()

    async def stream_once() -> None:
        stream_started.set()
        await asyncio.sleep(0)

    await run_server_tasks(uvicorn_server=uvicorn_server, stream_coro=stream_once())

    assert stream_started.is_set()
    assert uvicorn_server.should_exit is True
    assert uvicorn_server.serve_exited is True


def test_no_loop_streamer_exits_after_finite_file(tmp_path: Path) -> None:
    """Finite no-loop playback should shut down the server process at EOF."""
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    pd.DataFrame(np.zeros((3, 4), dtype=np.float32)).to_parquet(
        data_dir / "track2_data.parquet"
    )
    (data_dir / "metadata.json").write_text(
        json.dumps({"sampling_rate_hz": 500.0}),
        encoding="utf-8",
    )

    port = _free_tcp_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.stream_data",
            "--from-file",
            str(data_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--batch-size",
            "2",
            "--no-loop",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        _terminate(process)
        stdout, stderr = process.communicate(timeout=2.0)
        raise AssertionError(
            "no-loop streamer did not exit after reaching the end of the file\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    assert process.returncode == 0, (
        f"streamer exited with {process.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
=======
"""Tests for scripts/stream_data.py streamer lifecycle behavior."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_no_loop_streamer_exits_after_finite_file(tmp_path: Path) -> None:
    """Finite no-loop playback should shut down the server process at EOF."""
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    pd.DataFrame(np.zeros((3, 4), dtype=np.float32)).to_parquet(
        data_dir / "track2_data.parquet"
    )
    (data_dir / "metadata.json").write_text(
        json.dumps({"sampling_rate_hz": 500.0}),
        encoding="utf-8",
    )

    port = _free_tcp_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.stream_data",
            "--from-file",
            str(data_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--batch-size",
            "2",
            "--no-loop",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        _terminate(process)
        stdout, stderr = process.communicate(timeout=2.0)
        raise AssertionError(
            "no-loop streamer did not exit after reaching the end of the file\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    assert process.returncode == 0, (
        f"streamer exited with {process.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
>>>>>>> bddde82 (test: cover no-loop streamer exit)
