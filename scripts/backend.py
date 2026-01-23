"""
Unified Backend Server for Neural Data Viewer.

Acts as:
1. WebSocket client to stream_data.py (consuming raw data from upstream)
2. WebSocket server for browser connections (serving data at /ws)
3. HTTP server for static files

Architecture:
    [Parquet] --> stream_data.py (WS localhost:8765) --> backend.py (WS client)
                                                    |
                                         backend.py (WS server + HTTP localhost:8000)
                                                    |
                                                  Browser

Usage:
    # Start data streamer first
    brainstorm-stream --from-file data/hard/

    # Start unified backend
    brainstorm-backend --upstream-url ws://localhost:8765
"""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("brainstorm.backend")

cli = typer.Typer(help="Unified Backend Server for Neural Data Viewer")


@dataclass(slots=True)
class SharedState:
    """Shared state between upstream consumer and browser server."""

    connected_to_upstream: bool = False
    connection_attempts: int = 0

    channels_coords: tuple[tuple[int, int], ...] | None = None
    grid_size: int = 32
    fs: float = 500.0
    batch_size: int = 10
    init_message: dict[str, Any] | None = None

    message_queue: asyncio.Queue[str] | None = None
    browser_connections: set[WebSocket] = field(default_factory=set)


class BrowserServer:
    """Manages browser WebSocket connections and message broadcasting."""

    def __init__(self, state: SharedState):
        self.state = state
        self.connections: set[WebSocket] = set()

    async def register(self, ws: WebSocket) -> None:
        """Register a new browser connection."""
        self.connections.add(ws)
        self.state.browser_connections = self.connections
        logger.info(f"Browser connected. Total clients: {len(self.connections)}")

    async def unregister(self, ws: WebSocket) -> None:
        """Unregister a browser connection."""
        self.connections.discard(ws)
        self.state.browser_connections = self.connections
        logger.info(f"Browser disconnected. Total clients: {len(self.connections)}")

    async def broadcast(self, message: str) -> None:
        """Send message to all connected browsers."""
        if not self.connections:
            return

        disconnected: list[WebSocket] = []
        for ws in list(self.connections):
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.connections.discard(ws)
            logger.warning("Removed disconnected client during broadcast")


async def consume_upstream(
    upstream_url: str,
    state: SharedState,
    max_retries: int = 5,
    retry_delay: float = 0.5,
) -> None:
    """
    Connect as WebSocket client to stream_data.py.

    - Parse init message and cache it
    - Forward sample_batch messages to queue
    - Reconnect on failure (max retries with fixed delay)
    """
    retries = 0

    while retries < max_retries:
        try:
            logger.info(f"Connecting to upstream {upstream_url}...")
            async with websockets.connect(upstream_url, max_queue=64) as ws:
                state.connected_to_upstream = True
                state.connection_attempts = retries + 1
                retries = 0
                logger.info("Connected to upstream ✅")

                async for msg in ws:
                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "init":
                        state.init_message = data
                        coords = data.get("channels_coords")
                        state.channels_coords = (
                            tuple(tuple(c) for c in coords) if coords else None
                        )
                        state.grid_size = data.get("grid_size", 32)
                        state.fs = data.get("fs", 500.0)
                        state.batch_size = data.get("batch_size", 10)
                        logger.info(
                            f"Received init: grid_size={state.grid_size}, "
                            f"fs={state.fs}, batch_size={state.batch_size}"
                        )

                    if state.message_queue is not None and isinstance(msg, str):
                        await state.message_queue.put(msg)

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Upstream connection closed: {e}")
        except Exception as e:
            logger.error(f"Upstream connection error: {e}")
        finally:
            state.connected_to_upstream = False

        retries += 1
        if retries < max_retries:
            logger.info(
                f"Reconnecting to upstream in {retry_delay}s "
                f"(attempt {retries + 1}/{max_retries})..."
            )
            await asyncio.sleep(retry_delay)

    logger.error(f"Failed to connect to upstream after {max_retries} attempts")


async def broadcast_loop(
    server: BrowserServer,
    state: SharedState,
) -> None:
    """
    Consume from state.message_queue and broadcast to all connected browsers.
    """
    if state.message_queue is None:
        logger.error("Message queue not initialized")
        return

    while True:
        try:
            message = await state.message_queue.get()
            await server.broadcast(message)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Broadcast error: {e}")


def create_app(
    state: SharedState,
    server: BrowserServer,
    static_dir: Path,
) -> FastAPI:
    """Create the FastAPI application with routes and middleware."""
    app = FastAPI(title="Neural Data Backend Server")

    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            {
                "status": "ok",
                "upstream_connected": state.connected_to_upstream,
                "browser_clients": len(server.connections),
            }
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for browser connections."""
        await websocket.accept()
        await server.register(websocket)

        try:
            if state.init_message is not None:
                await websocket.send_text(json.dumps(state.init_message))
                logger.debug("Sent cached init message to new client")

            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await server.unregister(websocket)

    @app.get("/")
    async def index() -> FileResponse:
        """Serve index.html."""
        return FileResponse(static_dir / "index.html")

    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir), name="static")
    else:
        logger.warning(f"Static directory not found: {static_dir}")

    return app


@cli.command()
def main(
    upstream_url: str = typer.Option(
        "ws://localhost:8765",
        "--upstream-url",
        "-u",
        help="Upstream data stream WebSocket URL",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Server host (0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Server port",
    ),
    static_dir: str = typer.Option(
        "frontend/",
        "--static-dir",
        "-s",
        help="Directory to serve static files from",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """
    Start the unified backend server.

    This server connects to the data stream (stream_data.py) as a WebSocket client
    and serves both static files and a WebSocket endpoint for browsers.

    Example:
        brainstorm-backend --upstream-url ws://localhost:8765
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    static_path = Path(static_dir)
    if not static_path.is_absolute():
        static_path = Path.cwd() / static_path

    logger.info(f"Static files directory: {static_path}")
    logger.info(f"Upstream URL: {upstream_url}")
    logger.info(f"Server will listen on http://{host}:{port}")
    logger.info(f"WebSocket endpoint: ws://{host}:{port}/ws")

    state = SharedState()
    state.message_queue = asyncio.Queue(maxsize=1000)
    server = BrowserServer(state)

    app = create_app(state, server, static_path)

    async def run_server() -> None:
        """Run uvicorn server with upstream consumer and broadcast loop."""
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
        uvicorn_server = uvicorn.Server(config)

        upstream_task = asyncio.create_task(
            consume_upstream(upstream_url, state),
            name="upstream_consumer",
        )
        broadcast_task = asyncio.create_task(
            broadcast_loop(server, state),
            name="broadcast_loop",
        )

        try:
            await asyncio.gather(
                uvicorn_server.serve(),
                upstream_task,
                broadcast_task,
            )
        except asyncio.CancelledError:
            logger.info("Shutting down...")
        finally:
            upstream_task.cancel()
            broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await upstream_task
            with contextlib.suppress(asyncio.CancelledError):
                await broadcast_task

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server())

    logger.info("Server stopped")


if __name__ == "__main__":
    cli()
