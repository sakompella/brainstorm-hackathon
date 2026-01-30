"""
Unified Backend Server for Neural Data Viewer.

Acts as:
1. WebSocket client to stream_data.py (consuming raw data from upstream)
2. Signal processing middleware (ActivityEMA for per-channel activity)
3. WebSocket server for browser connections (serving data at /ws)
4. HTTP server for static files

Architecture:
    [Parquet] --> stream_data.py (WS localhost:8765) --> backend.py (WS client)
                                                    |
                                         backend.py (signal processing + WS server + HTTP localhost:8000)
                                                    |
                                                  Browser (receives "features" messages)

Usage:
    # Start data streamer first
    brainstorm-stream --from-file data/hard/

    # Start unified backend (with middleware processing enabled by default)
    brainstorm-backend --upstream-url ws://localhost:8765

    # Pass through raw data without processing
    brainstorm-backend --upstream-url ws://localhost:8765 --no-process
"""

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import typer
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scripts.signal_processing import (
    DIFFICULTY_SETTINGS,
    NeuralProcessor,
)

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

    processor: NeuralProcessor | None = None
    last_result: dict | None = None
    last_t: float = 0.0
    total_samples: int = 0

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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

    async def broadcast(self, message: str | bytes) -> None:
        """Send message to all connected browsers using parallel gather."""
        if not self.connections:
            return

        async def send_safe(ws: WebSocket) -> WebSocket | None:
            try:
                if isinstance(message, bytes):
                    await ws.send_bytes(message)
                else:
                    await ws.send_text(message)
                return None
            except Exception:
                return ws

        results = await asyncio.gather(
            *(send_safe(ws) for ws in self.connections),
            return_exceptions=True,
        )

        # send_safe returns None on success, WebSocket on failure (filter out exceptions)
        disconnected = [
            r for r in results if r is not None and not isinstance(r, BaseException)
        ]
        if disconnected:
            for ws in disconnected:
                self.connections.discard(ws)
            self.state.browser_connections = self.connections
            logger.warning(f"Removed {len(disconnected)} disconnected client(s)")


async def consume_upstream(
    upstream_url: str,
    state: SharedState,
    process: bool = True,
    difficulty: str = "hard",
    stateless: bool = False,
    max_retries: int = 5,
    retry_delay: float = 0.5,
) -> None:
    """
    Connect as WebSocket client to stream_data.py.

    - Parse init message and cache it
    - If process=True: use NeuralProcessor with full pipeline
    - If process=False: forward raw messages to queue
    - Reconnect on failure (max retries with fixed delay)
    """
    retries = 0
    settings = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS["hard"])

    while retries < max_retries:
        try:
            logger.info(f"Connecting to upstream {upstream_url}...")
            async with websockets.connect(
                upstream_url,
                max_queue=64,
                ping_interval=None,  # Disable keepalive pings (matches middleware)
                ping_timeout=None,   # Disable ping timeout
                close_timeout=60.0,   # Allow graceful shutdown
            ) as ws:
                state.connected_to_upstream = True
                state.connection_attempts = retries + 1
                retries = 0
                logger.info("Connected to upstream")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON message: {e}")
                        continue

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

                        if process:
                            state.processor = NeuralProcessor(
                                fs=state.fs,
                                settings=settings,
                                grid_size=state.grid_size,
                                stateless=stateless,
                            )
                            logger.info(
                                f"Initialized NeuralProcessor: difficulty={difficulty}, "
                                f"stateless={stateless}, notch={settings['notch_60hz']}, bandpass={settings['bandpass']}"
                            )

                    elif msg_type == "sample_batch" and process:
                        if state.processor is None:
                            state.processor = NeuralProcessor(
                                fs=state.fs,
                                settings=settings,
                                grid_size=state.grid_size,
                                stateless=stateless,
                            )

                        batch_samples = data["neural_data"]
                        start_time_s = float(data.get("start_time_s", 0.0))
                        sample_count = int(data.get("sample_count", len(batch_samples)))
                        fs = float(data.get("fs", state.fs))

                        batch = np.asarray(batch_samples, dtype=np.float32)
                        result = state.processor.process_batch(batch)

                        last_t = start_time_s + (sample_count - 1) / fs

                        async with state.lock:
                            state.last_result = result
                            state.last_t = last_t
                            state.total_samples += sample_count

                    if (
                        not process
                        and state.message_queue is not None
                        and isinstance(msg, str)
                    ):
                        try:
                            state.message_queue.put_nowait(msg)
                        except asyncio.QueueFull:
                            logger.warning("Message queue full, dropping message")

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
    Used when process=False (raw passthrough mode).
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


async def publish_features(
    server: BrowserServer,
    state: SharedState,
    out_hz: float = 20.0,
) -> None:
    """
    Broadcast computed activity features at a fixed rate (e.g., 20 Hz).
    Used when process=True (middleware processing mode).
    """
    period = 1.0 / out_hz
    last_sent_t = -1.0

    while True:
        try:
            t0 = time.perf_counter()

            async with state.lock:
                result = state.last_result
                t = state.last_t
                connected = state.connected_to_upstream
                total_samples = state.total_samples

            if result is not None and t != last_sent_t:
                confidence = 1.0 if connected else 0.0

                payload = {
                    "type": "features",
                    "t": float(t),
                    "fs": float(state.fs),
                    "n_ch": state.grid_size**2,
                    "heatmap": result["heatmap"].tolist(),
                    "centroid": result["centroid"].tolist(),
                    "center_distance": float(result["center_distance"]),
                    "confidence": confidence,
                    "bad_channels": result["bad_channels"],
                    "total_samples": int(total_samples),
                }
                # orjson is 5-10x faster than stdlib json (decode to str for JS compatibility)
                await server.broadcast(orjson.dumps(payload).decode())
                last_sent_t = t

            dt = time.perf_counter() - t0
            sleep_s = period - dt
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            else:
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Feature publish error: {e}")


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
    process: bool = typer.Option(
        True,
        "--process/--no-process",
        help="Enable signal processing. Disable for raw passthrough.",
    ),
    difficulty: str = typer.Option(
        "hard",
        "--difficulty",
        "-d",
        help="Difficulty preset (super_easy, easy, medium, hard)",
    ),
    out_hz: float = typer.Option(
        20.0,
        "--out-hz",
        help="Feature broadcast rate in Hz (when processing enabled)",
    ),
    stateless: bool = typer.Option(
        True,
        "--stateless/--stateful",
        help="Stateless mode: per-frame power (default). Stateful: EMA + accumulation.",
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

    This server connects to the data stream (stream_data.py) as a WebSocket client,
    optionally processes the signal (ActivityEMA), and serves both static files
    and a WebSocket endpoint for browsers.

    Examples:
        # With signal processing (default)
        brainstorm-backend --upstream-url ws://localhost:8765

        # Raw passthrough mode
        brainstorm-backend --upstream-url ws://localhost:8765 --no-process
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
    logger.info(
        f"Signal processing: {'enabled' if process else 'disabled (passthrough)'}"
    )
    if process:
        settings = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS["hard"])
        logger.info(f"Difficulty: {difficulty}, stateless={stateless}")
        logger.info(
            f"Settings: notch={settings['notch_60hz']}, bandpass={settings['bandpass']}, "
            f"bad_ch={settings['bad_channel_detection']}, sigma={settings['spatial_sigma']}"
        )

    state = SharedState()
    state.message_queue = asyncio.Queue(maxsize=1000)
    server = BrowserServer(state)

    app = create_app(state, server, static_path)

    async def run_server() -> None:
        """Run uvicorn server with upstream consumer and broadcast/publish loop."""
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
        uvicorn_server = uvicorn.Server(config)

        upstream_task = asyncio.create_task(
            consume_upstream(
                upstream_url,
                state,
                process=process,
                difficulty=difficulty,
                stateless=stateless,
            ),
            name="upstream_consumer",
        )

        if process:
            output_task = asyncio.create_task(
                publish_features(server, state, out_hz=out_hz),
                name="feature_publisher",
            )
        else:
            output_task = asyncio.create_task(
                broadcast_loop(server, state),
                name="broadcast_loop",
            )

        try:
            await asyncio.gather(
                uvicorn_server.serve(),
                upstream_task,
                output_task,
            )
        except asyncio.CancelledError:
            logger.info("Shutting down...")
        finally:
            upstream_task.cancel()
            output_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await upstream_task
            with contextlib.suppress(asyncio.CancelledError):
                await output_task

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server())

    logger.info("Server stopped")


if __name__ == "__main__":
    cli()
