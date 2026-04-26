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
