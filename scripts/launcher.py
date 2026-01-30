#!/usr/bin/env python3
"""Launcher for BrainStorm services with robust process management."""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

Process = subprocess.Popen[bytes]
ProcessEntry = tuple[str, Process]


@dataclass(slots=True)
class State:
    """Mutable state container for process management."""

    processes: list[ProcessEntry] = field(default_factory=list)
    shutdown: bool = False


def find_repo_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return Path.cwd()


def kill_port(port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        for pid in filter(None, result.stdout.strip().split("\n")):
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(int(pid), signal.SIGKILL)
    except FileNotFoundError:
        pass


def kill_ports(ports: tuple[int, ...]) -> None:
    for port in ports:
        kill_port(port)


def download_data(repo_root: Path, dataset_name: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "scripts.download", dataset_name],
        cwd=repo_root,
        check=True,
    )


def start_process(
    name: str,
    args: tuple[str, ...],
    log_file: Path | None,
    cwd: Path,
    state: State,
) -> Process:
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "w")  # noqa: SIM115
        stdout = log_handle
    else:
        log_handle = None
        stdout = None

    proc = subprocess.Popen(
        args,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout else None,
        cwd=cwd,
        start_new_session=True,
    )
    if log_handle:
        log_handle.close()
    state.processes.append((name, proc))
    return proc


def cleanup(state: State, ports: tuple[int, ...]) -> None:
    if state.shutdown:
        return
    state.shutdown = True

    for name, proc in reversed(state.processes):
        if proc.poll() is None:
            print(f"Stopping {name} (PID: {proc.pid})...")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)

    time.sleep(0.5)

    for name, proc in state.processes:
        if proc.poll() is None:
            print(f"Force killing {name} (PID: {proc.pid})...")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

    kill_ports(ports)


def wait_for_processes(state: State, ports: tuple[int, ...]) -> None:
    try:
        while not state.shutdown:
            for name, proc in state.processes:
                if proc.poll() is not None:
                    print(f"{name} exited with code {proc.returncode}")
                    cleanup(state, ports)
                    return
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
        cleanup(state, ports)


def in_container() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("CONTAINER") is not None


def main() -> None:
    ports = (8765, 8000)
    repo_root = find_repo_root()
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/medium"
    data_path = repo_root / data_dir
    dataset_name = data_path.name
    use_logs = not in_container()

    if not (data_path / "track2_data.parquet").exists():
        print(f"Downloading {dataset_name} dataset...")
        download_data(repo_root, dataset_name)
    else:
        print(f"Data already present at {data_dir}")

    print("Cleaning up...")
    kill_ports(ports)
    time.sleep(1)

    state = State()
    atexit.register(lambda: cleanup(state, ports))

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        print(f"\nReceived signal {signum}, shutting down...")
        cleanup(state, ports)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    stream_log = Path("/tmp/stream.log") if use_logs else None
    backend_log = Path("/tmp/backend.log") if use_logs else None

    print(f"Starting streamer on :8765 (data: {data_dir})...")
    start_process(
        "streamer",
        (sys.executable, "-m", "scripts.stream_data", "--from-file", f"{data_path}/"),
        stream_log,
        repo_root,
        state,
    )
    time.sleep(2)

    print("Starting backend on :8000...")
    backend_args: list[str] = [
        sys.executable,
        "-m",
        "scripts.backend",
        "--upstream-url",
        "ws://localhost:8765",
    ]
    static_dir = os.environ.get("BRAINSTORM_STATIC_DIR")
    if static_dir:
        backend_args.extend(["--static-dir", static_dir])
    start_process(
        "backend",
        tuple(backend_args),
        backend_log,
        repo_root,
        state,
    )
    time.sleep(2)

    streamer_pid = state.processes[0][1].pid
    backend_pid = state.processes[1][1].pid
    log_info = (
        "\n    Logs:\n      tail -f /tmp/stream.log\n      tail -f /tmp/backend.log"
        if use_logs
        else ""
    )
    print(
        textwrap.dedent(f"""
    Ready!

      Streamer:  ws://localhost:8765  (PID: {streamer_pid})
      Backend:   http://localhost:8000 (PID: {backend_pid})

    Open http://localhost:8000 in your browser
{log_info}
    To stop: Press Ctrl+C""")
    )

    wait_for_processes(state, ports)


if __name__ == "__main__":
    main()
