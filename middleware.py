#!/usr/bin/env python3
"""
Phase 1 Middleware: Raw voltage -> per-channel "activity strength" (EMA power)

IN:
  ws://localhost:8765  (brainstorm-stream)
  messages:
    - type="init"
    - type="sample_batch" with neural_data: [ [1024 floats], ... ]

OUT:
  ws://localhost:8787  (your processed feature stream)
  messages:
    - type="features"
      activity: [1024 floats]  # RMS-ish activity per channel
      t: float                # latest time (seconds)
      presence: float         # simple global activity indicator (debug)
      confidence: float       # placeholder in Phase 1
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol


# ---------------------------
# Phase 1: activity estimator
# ---------------------------

class ActivityEMA:
    """
    Maintains a fast and slow EMA of signal power (x^2) per channel.
    Phase 1 output = sqrt(fast_power)  (an RMS-ish activity level).
    """

    def __init__(
        self,
        n_ch: int,
        fs: float,
        tau_fast_s: float = 0.2,
        tau_base_s: float = 8.0,
    ):
        dt = 1.0 / fs
        # EMA coefficients; stable if <1 (they are, for reasonable tau)
        self.a_fast = float(dt / tau_fast_s)
        self.a_base = float(dt / tau_base_s)

        self.p_fast = np.zeros((n_ch,), dtype=np.float32)
        self.p_base = np.zeros((n_ch,), dtype=np.float32)

    def update_batch(self, batch: np.ndarray) -> None:
        """
        batch: (B, n_ch) float32
        """
        # batch size is typically small (e.g., 10). Loop over samples, vectorize over channels.
        for x in batch:
            x2 = x * x
            self.p_fast = (1.0 - self.a_fast) * self.p_fast + self.a_fast * x2
            self.p_base = (1.0 - self.a_base) * self.p_base + self.a_base * x2

    def activity(self) -> np.ndarray:
        return np.sqrt(self.p_fast + 1e-12)

    def normalized_activity(self) -> np.ndarray:
        # Optional: not used in Phase 1 output, but handy for Phase 3 later
        return np.log(self.p_fast + 1e-12) - np.log(self.p_base + 1e-12)


# ---------------------------
# Shared middleware state
# ---------------------------

@dataclass
class SharedState:
    fs: float = 500.0
    n_ch: int = 1024
    last_t: float = 0.0
    total_samples: int = 0
    connected_to_input: bool = False

    # Latest computed features
    last_activity: Optional[np.ndarray] = None


# ---------------------------
# WebSocket server (output)
# ---------------------------

class FeatureServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()

    async def register(self, ws: WebSocketServerProtocol) -> None:
        self.clients.add(ws)

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        self.clients.discard(ws)

    async def handler(self, ws: WebSocketServerProtocol) -> None:
        await self.register(ws)
        try:
            async for _ in ws:
                # We don't require any inbound messages; keep alive
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(ws)

    async def broadcast(self, message: str) -> None:
        if not self.clients:
            return
        await asyncio.gather(
            *[c.send(message) for c in list(self.clients)],
            return_exceptions=True,
        )

    async def run(self) -> None:
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"[feature server] listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever


# ---------------------------
# Input stream consumer
# ---------------------------

async def consume_raw_stream(
    input_url: str,
    state: SharedState,
    lock: asyncio.Lock,
    tau_fast_s: float,
    tau_base_s: float,
) -> None:
    """
    Connect to the raw brainstorm-stream and keep updating the EMA activity.
    Automatically reconnects on drop.
    """
    ema: Optional[ActivityEMA] = None

    while True:
        try:
            print(f"[input] connecting to {input_url} ...")
            async with websockets.connect(input_url, max_queue=2) as ws:
                async with lock:
                    state.connected_to_input = True
                print("[input] connected ✅")

                async for msg in ws:
                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "init":
                        fs = float(data.get("fs", state.fs))
                        n_ch = int(data.get("grid_size", 32)) ** 2  # usually 1024
                        batch_size = int(data.get("batch_size", 10))

                        async with lock:
                            state.fs = fs
                            state.n_ch = n_ch

                        ema = ActivityEMA(n_ch=n_ch, fs=fs, tau_fast_s=tau_fast_s, tau_base_s=tau_base_s)
                        print(f"[input] init: fs={fs}, n_ch={n_ch}, batch_size={batch_size}")

                    elif msg_type == "sample_batch":
                        if ema is None:
                            # If init didn't arrive yet, assume defaults
                            ema = ActivityEMA(n_ch=state.n_ch, fs=state.fs, tau_fast_s=tau_fast_s, tau_base_s=tau_base_s)

                        batch_samples = data["neural_data"]  # list[list[float]]
                        start_time_s = float(data.get("start_time_s", 0.0))
                        sample_count = int(data.get("sample_count", len(batch_samples)))
                        fs = float(data.get("fs", state.fs))

                        batch = np.asarray(batch_samples, dtype=np.float32)  # (B, 1024)
                        ema.update_batch(batch)

                        # time of last sample in the batch:
                        # start_time_s is time for first sample; samples spaced 1/fs
                        last_t = start_time_s + (sample_count - 1) / fs

                        async with lock:
                            state.last_activity = ema.activity().copy()
                            state.last_t = last_t
                            state.total_samples += sample_count

        except Exception as e:
            print(f"[input] connection error: {e}")
        finally:
            async with lock:
                state.connected_to_input = False
            print("[input] disconnected. retrying in 0.5s...")
            await asyncio.sleep(0.5)


# ---------------------------
# Feature publisher (output loop)
# ---------------------------

async def publish_features(
    server: FeatureServer,
    state: SharedState,
    lock: asyncio.Lock,
    out_hz: float = 20.0,
) -> None:
    """
    Broadcast last computed activity at a fixed rate (e.g., 20 Hz).
    """
    period = 1.0 / out_hz
    last_sent_t = -1.0

    while True:
        t0 = time.perf_counter()

        async with lock:
            activity = None if state.last_activity is None else state.last_activity.copy()
            t = state.last_t
            connected = state.connected_to_input
            total_samples = state.total_samples

        if activity is not None and t != last_sent_t:
            # A very simple global "presence" for debugging/UI: mean of top-k channels
            k = 50
            topk = np.partition(activity, -k)[-k:]
            presence = float(np.mean(topk))

            # Placeholder confidence in Phase 1 (real confidence comes after hotspot tracking)
            confidence = 1.0 if connected else 0.0

            payload = {
                "type": "features",
                "t": float(t),
                "fs": float(state.fs),
                "n_ch": int(state.n_ch),
                "activity": activity.tolist(),  # length 1024
                "presence": presence,
                "confidence": confidence,
                "total_samples": int(total_samples),
            }
            await server.broadcast(json.dumps(payload))
            last_sent_t = t

        # sleep to maintain output rate
        dt = time.perf_counter() - t0
        sleep_s = period - dt
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)


# ---------------------------
# Main
# ---------------------------

async def main() -> None:
    input_url = "ws://localhost:8765"
    out_host = "localhost"
    out_port = 8787

    # Phase 1 params
    tau_fast_s = 0.2   # responsive smoothing
    tau_base_s = 8.0   # slow baseline (not used yet, but computed)

    state = SharedState()
    lock = asyncio.Lock()

    server = FeatureServer(out_host, out_port)

    await asyncio.gather(
        server.run(),  # output server
        consume_raw_stream(input_url, state, lock, tau_fast_s, tau_base_s),  # input consumer
        publish_features(server, state, lock, out_hz=20.0),  # feature publisher
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] stopped.")
