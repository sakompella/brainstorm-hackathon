import asyncio
import json
import time

import numpy as np
import websockets
from scipy.ndimage import gaussian_filter

from .bandpower import BandpowerEMA
from .config import Config
from .featureserver import FeatureServer
from .state import SharedState


async def consume_raw_stream(cfg: Config, state: SharedState, lock: asyncio.Lock) -> None:
    estimator: BandpowerEMA | None = None

    while True:
        try:
            print(f"[input] connecting to {cfg.input_url} ...")
            async with websockets.connect(
                cfg.input_url,
                max_queue=cfg.max_queue,
                ping_interval=cfg.ping_interval,
                ping_timeout=cfg.ping_timeout,
                close_timeout=cfg.close_timeout,
            ) as ws:
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
                            band_hz=cfg.band_hz,
                            filter_order=cfg.filter_order,
                            tau_fast_s=cfg.tau_fast_s,
                            tau_base_s=cfg.tau_base_s,
                        )
                        print(
                            f"[input] init: fs={fs}, n_ch={n_ch}, batch_size={batch_size}, band={cfg.band_hz}"
                        )

                    elif msg_type == "sample_batch":
                        if estimator is None:
                            estimator = BandpowerEMA(
                                n_ch=state.n_ch,
                                fs=state.fs,
                                band_hz=cfg.band_hz,
                                filter_order=cfg.filter_order,
                                tau_fast_s=cfg.tau_fast_s,
                                tau_base_s=cfg.tau_base_s,
                            )

                        batch_samples = data["neural_data"]
                        start_time_s = float(data.get("start_time_s", 0.0))
                        sample_count = int(data.get("sample_count", len(batch_samples)))
                        fs = float(data.get("fs", state.fs))

                        batch = np.asarray(batch_samples, dtype=np.float32)
                        estimator.update_batch(batch)

                        last_t = start_time_s + (sample_count - 1) / fs

                        vec = estimator.normalized_vec()
                        grid = int(np.sqrt(vec.shape[0]))
                        heatmap = vec.reshape(grid, grid)

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


async def publish_features(server: FeatureServer, cfg: Config, state: SharedState, lock: asyncio.Lock) -> None:
    period = 1.0 / cfg.out_hz
    last_sent_t = -1.0

    while True:
        t0 = time.perf_counter()

        async with lock:
            heatmap = None if state.last_heatmap is None else state.last_heatmap.copy()
            t = state.last_t
            connected = state.connected_to_input
            total_samples = state.total_samples
            fs = state.fs

        if heatmap is not None and t != last_sent_t:
            if cfg.spatial_sigma > 0:
                    heatmap = gaussian_filter(heatmap, sigma=cfg.spatial_sigma)
            peak = float(np.max(heatmap))
            med = float(np.median(heatmap))
            presence = peak - med
            confidence = 1.0 if connected else 0.0

            payload = {
                "type": "features",
                "t": float(t),
                "fs": float(fs),
                "heatmap": heatmap.tolist(),
                "presence": float(presence),
                "confidence": float(confidence),
                "band": [float(cfg.band_hz[0]), float(cfg.band_hz[1])],
                "total_samples": int(total_samples),
            }
            await server.broadcast(json.dumps(payload))
            last_sent_t = t

        dt = time.perf_counter() - t0
        sleep_s = period - dt
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)


async def run_pipeline(cfg: Config) -> None:
    state = SharedState()
    lock = asyncio.Lock()
    server = FeatureServer(cfg.out_host, cfg.out_port)

    await asyncio.gather(
        server.run(),
        consume_raw_stream(cfg, state, lock),
        publish_features(server, cfg, state, lock),
    )
