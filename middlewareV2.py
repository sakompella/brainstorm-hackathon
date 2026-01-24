#!/usr/bin/env python3
"""
Combined Middleware (Hackathon-ready):
Raw voltage stream -> bandpower heatmap (baseline-normalized) -> WS for frontend

IN:
  ws://localhost:8765  (brainstorm-stream)
  messages:
    - type="init"
    - type="sample_batch" with neural_data: [ [1024 floats], ... ]

OUT:
  ws://localhost:8787
  messages:
    - type="features"
      heatmap: [[32x32 floats]]   # baseline-normalized, spatially smoothed
      t: float                    # latest stream time (seconds)
      presence: float             # simple "hotspot present" scalar
      confidence: float           # placeholder (Phase 3 will improve)
      band: [low_hz, high_hz]     # which band we're using
"""

import asyncio
import json
import time
from dataclasses import dataclass

import numpy as np
import websockets
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt, sosfilt_zi


# ---------------------------
# Bandpower + EMA estimator
# ---------------------------

class BandpowerEMA:
    """
    Streaming-safe bandpower:
      raw -> causal bandpass (sosfilt with state) -> power (x^2) -> EMA fast/base
    """

    def __init__(
        self,
        n_ch: int,
        fs: float,
        band_hz: tuple[float, float] = (70.0, 150.0),  # high-gamma-ish
        filter_order: int = 4,
        tau_fast_s: float = 0.25,
        tau_base_s: float = 8.0,
    ):
        self.n_ch = n_ch
        self.fs = fs
        self.band_hz = band_hz

        # Design bandpass filter (streaming / causal)
        low, high = band_hz
        self.sos = butter(
            filter_order,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        # Initialize filter state for ALL channels.
        # sosfilt expects zi shape: (n_sections, ..., 2) where ... matches signal dims excluding axis.
        zi0 = sosfilt_zi(self.sos)  # (n_sections, 2)
        self.zi = np.repeat(zi0[:, :, None], n_ch, axis=2).astype(np.float32)  # (n_sections, 2, n_ch)

        # EMA parameters
        dt = 1.0 / fs
        self.a_fast = float(dt / tau_fast_s)
        self.a_base = float(dt / tau_base_s)

        self.p_fast = np.zeros((n_ch,), dtype=np.float32)
        self.p_base = np.zeros((n_ch,), dtype=np.float32)

    def update_batch(self, batch_raw: np.ndarray) -> None:
        """
        batch_raw: (B, n_ch) float32
        """
        # 1) Causal bandpass filter across time axis (axis=0)
        y, self.zi = sosfilt(self.sos, batch_raw, axis=0, zi=self.zi)  # y: (B, n_ch)

        # 2) Power in-band
        y2 = y * y  # (B, n_ch)

        # 3) EMA update (loop over samples, vectorized over channels)
        for x2 in y2:
            self.p_fast = (1.0 - self.a_fast) * self.p_fast + self.a_fast * x2
            self.p_base = (1.0 - self.a_base) * self.p_base + self.a_base * x2

    def normalized_map(self) -> np.ndarray:
        """
        Baseline-normalized log power ratio per channel (1024,)
        """
        return (np.log(self.p_fast + 1e-12) - np.log(self.p_base + 1e-12)).astype(np.float32)


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

    # Latest computed feature map (32x32)
    last_heatmap: np.ndarray | None = None


# ---------------------------
# WebSocket server (output)
# ---------------------------

class FeatureServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: set = set()

    async def register(self, ws) -> None:
        self.clients.add(ws)

    async def unregister(self, ws) -> None:
        self.clients.discard(ws)

    async def handler(self, ws) -> None:
        await self.register(ws)
        try:
            async for _ in ws:
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
            await asyncio.Future()


# ---------------------------
# Input stream consumer
# ---------------------------

async def consume_raw_stream(
    input_url: str,
    state: SharedState,
    lock: asyncio.Lock,
    band_hz: tuple[float, float],
    filter_order: int,
    tau_fast_s: float,
    tau_base_s: float,
    spatial_sigma: float,
) -> None:
    """
    Connect to raw brainstorm-stream and update bandpower heatmap.
    """
    estimator: BandpowerEMA | None = None

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
                        n_ch = int(data.get("grid_size", 32)) ** 2
                        batch_size = int(data.get("batch_size", 10))

                        async with lock:
                            state.fs = fs
                            state.n_ch = n_ch

                        estimator = BandpowerEMA(
                            n_ch=n_ch,
                            fs=fs,
                            band_hz=band_hz,
                            filter_order=filter_order,
                            tau_fast_s=tau_fast_s,
                            tau_base_s=tau_base_s,
                        )
                        print(f"[input] init: fs={fs}, n_ch={n_ch}, batch_size={batch_size}, band={band_hz}")

                    elif msg_type == "sample_batch":
                        if estimator is None:
                            estimator = BandpowerEMA(
                                n_ch=state.n_ch,
                                fs=state.fs,
                                band_hz=band_hz,
                                filter_order=filter_order,
                                tau_fast_s=tau_fast_s,
                                tau_base_s=tau_base_s,
                            )

                        batch_samples = data["neural_data"]
                        start_time_s = float(data.get("start_time_s", 0.0))
                        sample_count = int(data.get("sample_count", len(batch_samples)))
                        fs = float(data.get("fs", state.fs))

                        batch = np.asarray(batch_samples, dtype=np.float32)  # (B, 1024)
                        estimator.update_batch(batch)

                        # dataset time of the last sample in batch
                        last_t = start_time_s + (sample_count - 1) / fs

                        # Build 32x32 map (baseline-normalized)
                        vec = estimator.normalized_map()  # (1024,)
                        grid = int(np.sqrt(vec.shape[0]))  # 32
                        heatmap = vec.reshape(grid, grid)

                        # Spatial smoothing to make blobs clear
                        if spatial_sigma > 0:
                            heatmap = gaussian_filter(heatmap, sigma=spatial_sigma)

                        async with lock:
                            state.last_heatmap = heatmap.astype(np.float32)
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
    band_hz: tuple[float, float],
    out_hz: float = 20.0,
) -> None:
    """
    Broadcast heatmap at a fixed rate (e.g., 20 Hz).
    """
    period = 1.0 / out_hz
    last_sent_t = -1.0

    while True:
        t0 = time.perf_counter()

        async with lock:
            heatmap = None if state.last_heatmap is None else state.last_heatmap.copy()
            t = state.last_t
            connected = state.connected_to_input
            total_samples = state.total_samples

        if heatmap is not None and t != last_sent_t:
            # Presence: peak-to-median is a decent simple detector
            peak = float(np.max(heatmap))
            med = float(np.median(heatmap))
            presence = peak - med

            confidence = 1.0 if connected else 0.0  # Phase 3 will improve this

            payload = {
                "type": "features",
                "t": float(t),
                "fs": float(state.fs),
                "heatmap": heatmap.tolist(),  # 32x32
                "presence": float(presence),
                "confidence": float(confidence),
                "band": [float(band_hz[0]), float(band_hz[1])],
                "total_samples": int(total_samples),
            }
            await server.broadcast(json.dumps(payload))
            last_sent_t = t

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

    # ---- Tunable parameters (good defaults for demo) ----
    band_hz = (70.0, 150.0)     # high-gamma-ish
    filter_order = 4
    tau_fast_s = 0.25           # fast responsiveness (~250ms)
    tau_base_s = 8.0            # baseline drift (~8s)
    spatial_sigma = 1.0         # 0 = no blur, 1.0 is usually nice
    out_hz = 20.0               # UI update rate
    # -----------------------------------------------------

    state = SharedState()
    lock = asyncio.Lock()
    server = FeatureServer(out_host, out_port)

    await asyncio.gather(
        server.run(),
        consume_raw_stream(
            input_url, state, lock,
            band_hz, filter_order,
            tau_fast_s, tau_base_s,
            spatial_sigma,
        ),
        publish_features(server, state, lock, band_hz, out_hz=out_hz),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] stopped.")
