"""Tests for scripts/serve.py legacy viewer server."""

from __future__ import annotations

import asyncio
import inspect

from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

from scripts import serve
from scripts.backend import BrowserServer, SharedState


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

    def test_websocket_endpoint_forwards_init_message(self) -> None:
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
        app = serve.create_app(state=state, server=server)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()

        assert message["type"] == "init"
        assert message["grid_size"] == 32
