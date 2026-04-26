<<<<<<< HEAD
"""Tests for scripts/serve.py legacy web server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scripts.serve import create_app


def test_legacy_serve_websocket_relays_upstream_stream() -> None:
    upstream_url = "ws://streamer.example/ws"
    upstream_messages = [
        json.dumps(
            {
                "type": "init",
                "channels_coords": [[0, 0], [0, 1]],
                "grid_size": 32,
                "fs": 500.0,
                "batch_size": 10,
            }
        )
    ]

    async def mock_message_iter():
        for message in upstream_messages:
            yield message

    mock_upstream = AsyncMock()
    mock_upstream.__aiter__ = lambda self: mock_message_iter()

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_upstream
    mock_context.__aexit__.return_value = None

    app = create_app(upstream_url=upstream_url)
    client = TestClient(app)

    with patch("scripts.serve.websockets.connect", return_value=mock_context) as connect:
        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()

    connect.assert_called_once_with(upstream_url)
    assert message["type"] == "init"
    assert message["grid_size"] == 32
=======
"""Tests for scripts/serve.py legacy viewer server."""

from __future__ import annotations

import json
import inspect
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

from scripts import serve


class TestLegacyServeApp:
    """Regression coverage for the legacy serve entrypoint."""

    def test_websocket_endpoint_is_registered_before_static_mount(self) -> None:
        app = serve.create_app()

        route_index = next(
            index for index, route in enumerate(app.routes) if route.path == "/ws"
        )

        assert isinstance(app.routes[route_index], WebSocketRoute)
        assert all(route.path != "/" for route in app.routes[:route_index])

    def test_cli_default_port_matches_documented_two_terminal_flow(self) -> None:
        signature = inspect.signature(serve.main)

        assert signature.parameters["port"].default.default == 8000

    def test_websocket_relay_forwards_init_message(self) -> None:
        upstream_url = "ws://streamer.example/ws"
        upstream_messages = [
            json.dumps(
                {
                    "type": "init",
                    "channels_coords": [[0, 0], [0, 1]],
                    "grid_size": 32,
                    "fs": 500.0,
                    "batch_size": 10,
                }
            )
        ]

        async def mock_message_iter():
            for message in upstream_messages:
                yield message

        mock_upstream = AsyncMock()
        mock_upstream.__aiter__ = lambda self: mock_message_iter()

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_upstream
        mock_context.__aexit__.return_value = None

        app = serve.create_app(upstream_url=upstream_url)
        client = TestClient(app)

        with patch(
            "scripts.serve.websockets.connect", return_value=mock_context
        ) as connect:
            with client.websocket_connect("/ws") as ws:
                message = ws.receive_json()

        connect.assert_called_once_with(upstream_url)
        assert message["type"] == "init"
        assert message["grid_size"] == 32
>>>>>>> 8a9a97b (test: cover legacy serve websocket path)
