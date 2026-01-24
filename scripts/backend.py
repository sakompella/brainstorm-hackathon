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
import typer
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scripts.signal_processing import (
    EMAState,
    compute_presence,
    create_ema_state,
    ema_activity,
    update_ema,
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

    ema_state: EMAState | None = None
    last_activity: np.ndarray | None = None
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

        if disconnected:
            self.state.browser_connections = self.connections


async def consume_upstream(
    upstream_url: str,
    state: SharedState,
    process: bool = True,
    tau_fast_s: float = 0.2,
    tau_base_s: float = 8.0,
    max_retries: int = 5,
    retry_delay: float = 0.5,
) -> None:
    """
    Connect as WebSocket client to stream_data.py.

    - Parse init message and cache it
    - If process=True: compute ActivityEMA and store features in state
    - If process=False: forward raw messages to queue
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
                            n_ch = state.grid_size**2
                            state.ema_state = create_ema_state(
                                n_ch=n_ch,
                                fs=state.fs,
                                tau_fast_s=tau_fast_s,
                                tau_base_s=tau_base_s,
                            )
                            logger.info(
                                f"Initialized EMA state: n_ch={n_ch}, "
                                f"tau_fast={tau_fast_s}s, tau_base={tau_base_s}s"
                            )

                    elif msg_type == "sample_batch" and process:
                        if state.ema_state is None:
                            n_ch = state.grid_size**2
                            state.ema_state = create_ema_state(
                                n_ch=n_ch,
                                fs=state.fs,
                                tau_fast_s=tau_fast_s,
                                tau_base_s=tau_base_s,
                            )

                        batch_samples = data["neural_data"]
                        start_time_s = float(data.get("start_time_s", 0.0))
                        sample_count = int(data.get("sample_count", len(batch_samples)))
                        fs = float(data.get("fs", state.fs))

                        batch = np.asarray(batch_samples, dtype=np.float32)
                        state.ema_state = update_ema(state.ema_state, batch)

                        last_t = start_time_s + (sample_count - 1) / fs
                        activity_snapshot = ema_activity(state.ema_state)

                        async with state.lock:
                            state.last_activity = activity_snapshot
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
                activity = state.last_activity
                t = state.last_t
                connected = state.connected_to_upstream
                total_samples = state.total_samples

            if activity is not None and t != last_sent_t:
                presence = compute_presence(activity)
                confidence = 1.0 if connected else 0.0

                payload = {
                    "type": "features",
                    "t": float(t),
                    "fs": float(state.fs),
                    "n_ch": state.grid_size**2,
                    "activity": activity.tolist(),
                    "presence": presence,
                    "confidence": confidence,
                    "total_samples": int(total_samples),
                }
                await server.broadcast(json.dumps(payload))
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
        help="Enable signal processing (ActivityEMA). Disable for raw passthrough.",
    ),
    tau_fast: float = typer.Option(
        0.2,
        "--tau-fast",
        help="Fast EMA time constant in seconds",
    ),
    tau_base: float = typer.Option(
        8.0,
        "--tau-base",
        help="Base EMA time constant in seconds",
    ),
    out_hz: float = typer.Option(
        20.0,
        "--out-hz",
        help="Feature broadcast rate in Hz (when processing enabled)",
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
        logger.info(
            f"EMA params: tau_fast={tau_fast}s, tau_base={tau_base}s, out_hz={out_hz}Hz"
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
                tau_fast_s=tau_fast,
                tau_base_s=tau_base,
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
