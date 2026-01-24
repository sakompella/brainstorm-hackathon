"""Tests for scripts/backend.py unified backend server."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.backend import (
    BrowserServer,
    SharedState,
    broadcast_loop,
    consume_upstream,
    create_app,
)


class TestSharedState:
    """Tests for SharedState dataclass."""

    def test_default_values(self) -> None:
        state = SharedState()
        assert state.connected_to_upstream is False
        assert state.connection_attempts == 0
        assert state.channels_coords is None
        assert state.grid_size == 32
        assert state.fs == 500.0
        assert state.batch_size == 10
        assert state.init_message is None
        assert state.message_queue is None
        assert state.browser_connections == set()

    def test_with_message_queue(self) -> None:
        state = SharedState()
        state.message_queue = asyncio.Queue(maxsize=100)
        assert state.message_queue is not None
        assert state.message_queue.maxsize == 100


class TestBrowserServer:
    """Tests for BrowserServer class."""

    def test_init(self) -> None:
        state = SharedState()
        server = BrowserServer(state)
        assert server.state is state
        assert server.connections == set()

    @pytest.mark.asyncio
    async def test_register(self) -> None:
        state = SharedState()
        server = BrowserServer(state)
        mock_ws = AsyncMock()

        await server.register(mock_ws)

        assert mock_ws in server.connections
        assert mock_ws in state.browser_connections
        assert len(server.connections) == 1

    @pytest.mark.asyncio
    async def test_unregister(self) -> None:
        state = SharedState()
        server = BrowserServer(state)
        mock_ws = AsyncMock()

        await server.register(mock_ws)
        await server.unregister(mock_ws)

        assert mock_ws not in server.connections
        assert mock_ws not in state.browser_connections
        assert len(server.connections) == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self) -> None:
        state = SharedState()
        server = BrowserServer(state)
        mock_ws = AsyncMock()

        await server.unregister(mock_ws)
        assert len(server.connections) == 0

    @pytest.mark.asyncio
    async def test_broadcast_no_connections(self) -> None:
        state = SharedState()
        server = BrowserServer(state)

        await server.broadcast('{"type": "test"}')

    @pytest.mark.asyncio
    async def test_broadcast_to_connections(self) -> None:
        state = SharedState()
        server = BrowserServer(state)
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await server.register(mock_ws1)
        await server.register(mock_ws2)

        message = '{"type": "sample_batch", "data": [1, 2, 3]}'
        await server.broadcast(message)

        mock_ws1.send_text.assert_called_once_with(message)
        mock_ws2.send_text.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected(self) -> None:
        state = SharedState()
        server = BrowserServer(state)
        mock_ws_good = AsyncMock()
        mock_ws_bad = AsyncMock()
        mock_ws_bad.send_text.side_effect = Exception("Connection closed")

        await server.register(mock_ws_good)
        await server.register(mock_ws_bad)
        assert len(server.connections) == 2

        await server.broadcast('{"type": "test"}')

        assert mock_ws_good in server.connections
        assert mock_ws_bad not in server.connections
        assert len(server.connections) == 1


class TestCreateApp:
    """Tests for create_app FastAPI factory."""

    @pytest.fixture
    def app_components(self, tmp_path: Path) -> tuple[SharedState, BrowserServer, Path]:
        static_dir = tmp_path / "frontend"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html><body>Test</body></html>")
        (static_dir / "app.js").write_text("console.log('test');")

        state = SharedState()
        state.message_queue = asyncio.Queue()
        server = BrowserServer(state)
        return state, server, static_dir

    def test_health_endpoint_disconnected(
        self, app_components: tuple[SharedState, BrowserServer, Path]
    ) -> None:
        state, server, static_dir = app_components
        app = create_app(state, server, static_dir)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["upstream_connected"] is False
        assert data["browser_clients"] == 0

    def test_health_endpoint_connected(
        self, app_components: tuple[SharedState, BrowserServer, Path]
    ) -> None:
        state, server, static_dir = app_components
        state.connected_to_upstream = True
        server.connections.add(MagicMock())
        server.connections.add(MagicMock())

        app = create_app(state, server, static_dir)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["upstream_connected"] is True
        assert data["browser_clients"] == 2

    def test_index_endpoint(
        self, app_components: tuple[SharedState, BrowserServer, Path]
    ) -> None:
        state, server, static_dir = app_components
        app = create_app(state, server, static_dir)
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        assert b"<html>" in response.content

    def test_static_file_serving(
        self, app_components: tuple[SharedState, BrowserServer, Path]
    ) -> None:
        state, server, static_dir = app_components
        app = create_app(state, server, static_dir)
        client = TestClient(app)

        response = client.get("/app.js")
        assert response.status_code == 200
        assert b"console.log" in response.content

    def test_missing_static_dir_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        state = SharedState()
        server = BrowserServer(state)
        nonexistent = Path("/nonexistent/frontend")

        create_app(state, server, nonexistent)

        assert any("Static directory not found" in r.message for r in caplog.records)


class TestWebSocketEndpoint:
    """Tests for WebSocket /ws endpoint."""

    @pytest.fixture
    def app_with_init(self, tmp_path: Path) -> tuple[TestClient, SharedState]:
        static_dir = tmp_path / "frontend"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html></html>")

        state = SharedState()
        state.message_queue = asyncio.Queue()
        state.init_message = {
            "type": "init",
            "channels_coords": [[0, 0], [0, 1]],
            "grid_size": 32,
            "fs": 500.0,
            "batch_size": 10,
        }
        server = BrowserServer(state)
        app = create_app(state, server, static_dir)
        return TestClient(app), state

    def test_websocket_receives_init_message(
        self, app_with_init: tuple[TestClient, SharedState]
    ) -> None:
        client, _state = app_with_init

        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "init"
            assert data["grid_size"] == 32
            assert data["fs"] == 500.0

    def test_websocket_no_init_when_none(self, tmp_path: Path) -> None:
        static_dir = tmp_path / "frontend"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html></html>")

        state = SharedState()
        state.message_queue = asyncio.Queue()
        server = BrowserServer(state)
        app = create_app(state, server, static_dir)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")


class TestConsumeUpstream:
    """Tests for consume_upstream WebSocket client."""

    @pytest.mark.asyncio
    async def test_parses_init_message(self) -> None:
        state = SharedState()
        state.message_queue = asyncio.Queue()

        init_msg = json.dumps(
            {
                "type": "init",
                "channels_coords": [[0, 0], [1, 1], [2, 2]],
                "grid_size": 32,
                "fs": 500.0,
                "batch_size": 10,
            }
        )

        async def mock_message_iter():
            yield init_msg

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: mock_message_iter()

        async def mock_connect(*args, **kwargs):
            return mock_ws

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_ws
        mock_context.__aexit__.return_value = None

        with patch("scripts.backend.websockets.connect", return_value=mock_context):
            await consume_upstream("ws://test:8765", state, max_retries=1)

        assert state.init_message is not None
        assert state.init_message["type"] == "init"
        assert state.grid_size == 32
        assert state.fs == 500.0
        assert state.channels_coords == ((0, 0), (1, 1), (2, 2))

    @pytest.mark.asyncio
    async def test_forwards_messages_to_queue_passthrough(self) -> None:
        """Test raw passthrough mode (process=False)."""
        state = SharedState()
        state.message_queue = asyncio.Queue()

        batch_msg = json.dumps(
            {
                "type": "sample_batch",
                "neural_data": [[0.1] * 1024],
                "start_time_s": 1.0,
            }
        )

        async def mock_message_iter():
            yield batch_msg

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: mock_message_iter()

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_ws
        mock_context.__aexit__.return_value = None

        with patch("scripts.backend.websockets.connect", return_value=mock_context):
            await consume_upstream(
                "ws://test:8765", state, process=False, max_retries=1
            )

        assert not state.message_queue.empty()
        queued = await state.message_queue.get()
        assert json.loads(queued)["type"] == "sample_batch"

    @pytest.mark.asyncio
    async def test_processes_sample_batch_with_ema(self) -> None:
        """Test signal processing mode (process=True, default)."""
        state = SharedState()
        state.message_queue = asyncio.Queue()

        batch_msg = json.dumps(
            {
                "type": "sample_batch",
                "neural_data": [[0.5] * 1024],
                "start_time_s": 1.0,
                "sample_count": 1,
            }
        )

        async def mock_message_iter():
            yield batch_msg

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: mock_message_iter()

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_ws
        mock_context.__aexit__.return_value = None

        with patch("scripts.backend.websockets.connect", return_value=mock_context):
            # Use super_easy to skip bad channel detection (which requires 500 samples)
            await consume_upstream(
                "ws://test:8765",
                state,
                process=True,
                difficulty="super_easy",
                max_retries=1,
            )

        # NeuralProcessor replaces legacy EMA API
        assert state.processor is not None
        assert state.last_result is not None
        assert "heatmap" in state.last_result
        assert state.last_result["heatmap"].shape == (32, 32)
        assert state.total_samples == 1
        assert state.message_queue.empty()

    @pytest.mark.asyncio
    async def test_retries_on_connection_failure(self) -> None:
        state = SharedState()
        state.message_queue = asyncio.Queue()

        call_count = 0

        def failing_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_context = AsyncMock()
            mock_context.__aenter__.side_effect = ConnectionRefusedError(
                "Connection refused"
            )
            return mock_context

        with patch("scripts.backend.websockets.connect", side_effect=failing_connect):
            await consume_upstream(
                "ws://test:8765", state, max_retries=3, retry_delay=0.01
            )

        assert call_count == 3


class TestBroadcastLoop:
    """Tests for broadcast_loop function."""

    @pytest.mark.asyncio
    async def test_broadcasts_queued_messages(self) -> None:
        state = SharedState()
        state.message_queue = asyncio.Queue()
        server = BrowserServer(state)

        mock_ws = AsyncMock()
        await server.register(mock_ws)

        await state.message_queue.put('{"type": "test", "value": 1}')
        await state.message_queue.put('{"type": "test", "value": 2}')

        task = asyncio.create_task(broadcast_loop(server, state))

        async def wait_for_broadcasts():
            while mock_ws.send_text.call_count < 2:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_for_broadcasts(), timeout=1.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert mock_ws.send_text.call_count >= 2

    @pytest.mark.asyncio
    async def test_handles_missing_queue(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        state = SharedState()
        server = BrowserServer(state)

        await broadcast_loop(server, state)

        assert any("queue not initialized" in r.message for r in caplog.records)
