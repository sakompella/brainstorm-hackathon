"""Tests for scripts/stream_data.py."""

from __future__ import annotations

import asyncio

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
