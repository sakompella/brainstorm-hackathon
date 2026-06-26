# Backend & WebSocket Protocol Analysis

## Scope
Read: `scripts/backend.py`, `scripts/signal_processing.py`, `scripts/stream_data.py`, `scripts/serve.py`, `scripts/static_assets.py`, `frontend/index.html`, `frontend/src/main.ts`, and all `docs/*.md`.

## 1) WebSocket message protocol

### A. Upstream stream: `stream_data.py` -> backend
Source: `scripts/stream_data.py:49-59, 177-350`, docs `docs/data_stream.md:54-117`.

#### 1. `init`
Sent once per upstream WS connection, immediately after the browser/client connects.

```ts
type UpstreamInit = {
  type: "init";
  channels_coords: [number, number][]; // length 1024, 1-indexed row-major grid coords
  grid_size: number;                  // 32
  fs: number;                         // sampling rate, usually 500.0
  batch_size: number;                 // default 10
}
```

Shape details:
- `channels_coords` is generated row-major with `[row, col]` pairs from `[1,1]` to `[32,32]` (`scripts/stream_data.py:49-59`).
- This is text JSON over WebSocket, not binary.

#### 2. `sample_batch`
Sent continuously at the sampling rate, batched.

```ts
type UpstreamSampleBatch = {
  type: "sample_batch";
  neural_data: number[][];   // batch_size samples, each length 1024
  start_time_s: number;      // first sample timestamp in seconds
  sample_count: number;      // actual number of samples in this batch
  fs: number;                // sampling rate
}
```

Frequency:
- Sampling rate: 500 Hz
- Default batch size: 10
- Default message rate: 50 messages/second (`docs/data_stream.md:109-117`)
- Final batch may be shorter if `--no-loop` is used (`scripts/stream_data.py:86-107, 301-327`).

Client->server messages:
- No app-level client JSON schema; both `stream_data.py` and `backend.py` just `receive_text()` in a loop and ignore content (`scripts/stream_data.py:392-404`, `scripts/backend.py:454-470`).

---

### B. Browser stream: backend -> browser
Source: `scripts/backend.py:311-389, 426-480`.

#### 1. `init`
Forwarded cached upstream init. Sent once per browser connection if the backend has already seen upstream init.

```ts
type BrowserInit = UpstreamInit
```

Notes:
- The backend caches the upstream init in `SharedState.init_message` (`scripts/backend.py:203-215`).
- On each browser WS connect, it sends that cached init before waiting for anything else (`scripts/backend.py:454-470`).

#### 2. `status`
Event-driven lifecycle/status message.

```ts
type BrowserStatus = {
  type: "status";
  upstream_connected: boolean;
  upstream_state: "connecting" | "connected" | "disconnected" | "ended";
  total_samples: number;
}
```

Frequency:
- Emitted when `upstream_state` changes in `publish_features()`.
- Emitted again once on upstream completion via `publish_final_status()`.
- In raw passthrough mode (`--no-process`), no periodic status stream is produced; only the final status on upstream completion is guaranteed (`scripts/backend.py:600-615`).

#### 3. `features`
Periodic processed feature payload, only when `process=True`.

```ts
type BrowserFeatures = {
  type: "features";
  t: number;                  // timestamp of last sample in processed batch
  fs: number;                 // sampling rate
  n_ch: number;               // grid_size^2 (1024 in current config)
  heatmap: number[][];        // grid_size x grid_size, typically 32x32
  centroid: [number, number]; // [row, col] centroid in 0-indexed grid coords
  center_distance: number;    // normalized 0..1, 1=center, 0=corner
  confidence: number;         // 1.0 if upstream connected, else 0.0
  bad_channels: number;       // count of bad channels
  total_samples: number;
}
```

Frequency:
- Poll loop is `out_hz` (default 20 Hz).
- It sends only when `t` changes, so the browser gets the latest processed result at most 20 Hz (`scripts/backend.py:311-368`).
- `t` is not batch start; it is computed as `start_time_s + (sample_count - 1) / fs` (`scripts/backend.py:238-247`).

Mode differences:
- `process=True` (default): browser gets `status` + `features`.
- `process=False` (`--no-process`): browser gets raw upstream `init` + `sample_batch` messages forwarded unchanged, via an internal queue (`scripts/backend.py:254-263, 289-308, 600-603`).

Important mismatch:
- Older docs mention a `presence` concept, but the current backend does **not** emit a `presence` field. The browser payload uses `center_distance` and `confidence` instead (`scripts/backend.py:347-357`, `scripts/signal_processing.py:424-430`).

---

## 2) Backend processing pipeline
Source: `scripts/backend.py:153-263, 311-389, 485-630` and `scripts/signal_processing.py:21-431`.

### Overall flow
1. Backend connects as WS client to upstream streamer.
2. It caches the upstream `init` message and initializes `NeuralProcessor`.
3. Each `sample_batch` is converted to `np.float32` and sent through `NeuralProcessor.process_batch()`.
4. `publish_features()` polls shared state and broadcasts either:
   - processed `features` + lifecycle `status` messages, or
   - raw upstream messages in passthrough mode.
5. Browser WS clients connect to `/ws` and receive the cached init immediately.

### Shared state / lifecycle
`SharedState` carries upstream connection status, init metadata, processor state, latest processed result, latest timestamp, and total sample count (`scripts/backend.py:69-91`).

### Detailed processing chain
From `NeuralProcessor.process_batch()`:

1. **Initialization / bad-channel warmup**
   - If bad-channel detection is enabled, the processor buffers about 1 second of data before producing output (`init_samples_needed = int(1.0 * fs)`; `scripts/signal_processing.py:330-370`).
   - During this period it returns `None`.

2. **Filtering**
   - Optional 60 Hz notch filter.
   - Optional 70–150 Hz bandpass.
   - Both are stateful across batches using `sosfilt` with persistent `zi` per channel (`scripts/signal_processing.py:104-175`).

3. **Power extraction**
   - `PowerExtractor` computes batch mean squared power.
   - It maintains a fast EMA (`p_fast`) and a much slower baseline EMA (`p_base = ema_alpha / 20`).
   - Stateful mode uses log-normalized power: `log(p_fast) - log(p_base)` (`scripts/signal_processing.py:178-210`).

4. **Grid reshape / bad-channel masking / spatial smoothing**
   - `to_grid()` reshapes the 1024-vector to `32x32` row-major.
   - Bad channels are zeroed.
   - Gaussian smoothing is applied with `sigma` from difficulty settings (`scripts/signal_processing.py:213-240`).

5. **Centroid and centering score**
   - `weighted_centroid()` thresholds below the median (50th percentile) and computes intensity-weighted row/col center of mass.
   - Fallback is grid center if total mass is near zero.
   - `compute_center_distance()` maps centroid distance to a normalized `[0,1]` score where 1 = perfect center, 0 = corner (`scripts/signal_processing.py:255-298`).

### Stateless vs stateful mode
`backend.py` defaults to `stateless=True` in the CLI wrapper (`scripts/backend.py:527-531, 592-603`).

#### Stateless mode
- Uses instantaneous mean squared power per batch.
- Applies a light EMA to `smoothed_power` with `smooth_alpha = 0.15`.
- Heatmap = smoothed power grid.
- Centroid computed directly from that grid (`scripts/signal_processing.py:377-397`).

#### Stateful mode
- Uses `PowerExtractor` output (fast-vs-baseline log-normalized power).
- Builds an accumulator map with decay (`0.7`) and weight (`0.1`) over the positive part of the heatmap.
- Centroid is computed from the accumulator, not the instantaneous frame (`scripts/signal_processing.py:398-423`).

### Difficulty presets
`DIFFICULTY_SETTINGS` controls filtering and smoothing (`scripts/signal_processing.py:21-58`):
- `super_easy`: bandpass only, no notch, no bad-channel detection, sigma 0.0, ema 0.3
- `easy`: bandpass only, no notch, no bad-channel detection, sigma 0.5, ema 0.2
- `medium`: notch + bandpass + bad detection, sigma 1.0, ema 0.15
- `hard`: notch + bandpass + bad detection, sigma 1.5, ema 0.1

### Exact `features` values
The browser payload fields are produced as follows (`scripts/backend.py:344-360`):
- `heatmap` = `result["heatmap"].tolist()`
- `centroid` = `result["centroid"].tolist()`
- `center_distance` = float from signal processing
- `bad_channels` = integer count
- `confidence` = `1.0 if connected else 0.0`
- `total_samples` = count of upstream samples seen so far

---

## 3) Static file serving
Source: `scripts/backend.py:426-482`, `scripts/serve.py:19-103`, `scripts/static_assets.py:5-23`, `frontend/index.html:1-12`.

### Backend serving model
`create_app()` adds:
- `GET /health`
- `WS /ws`
- `GET /` returning `static_dir / "index.html"`
- `StaticFiles(directory=static_dir)` mounted at `/` if the directory exists (`scripts/backend.py:442-480`).

Because the exact routes are added before the root static mount, `/ws` and `/health` remain reachable.

### Static directory resolution
`resolve_static_dir()` behavior (`scripts/static_assets.py:5-23`):
- Default requested dir is `frontend`.
- Relative paths resolve against `Path.cwd()`.
- If the resolved path exists, it is used.
- If the default `frontend` does not exist, it falls back to:
  1. `SOURCE_STATIC_DIR = repo_root/frontend`
  2. `PACKAGED_STATIC_DIR = scripts/frontend`
- Otherwise it returns the unresolved path.

### URL paths the frontend must work at
The current frontend HTML uses root-relative URLs:
- `/favicon.svg`
- `/src/main.ts`
(`frontend/index.html:1-12`)

So the app must be served from the origin root `/`, not a subpath. Asset requests are also root-relative.

### Important caveat
The checked-in `frontend/` tree is a Vite/Svelte source tree, not a prebuilt bundle. A plain static file server can only serve it if browser-ready assets already exist in that directory or if you use a build output directory via `--static-dir`.

---

## 4) Dev vs production serving
Source: `docs/getting_started.md:9-39, 252-260`, `docs/data_stream.md:7-52, 119-152`, `docs/submissions.md:23-63`, `scripts/backend.py:485-630`, `scripts/serve.py:25-103`.

### Local dev (docs path)
Docs describe two approaches:
1. Direct browser connection to `stream_data.py` (`ws://localhost:8765`), with a static web server at `http://localhost:8000` (`docs/data_stream.md:9-29`).
2. Custom backend that proxies/processes before browser delivery (`docs/data_stream.md:31-52`).

### What the code actually does
- `brainstorm-backend` is the configurable unified backend.
- `brainstorm-serve` is a legacy wrapper around `backend.main` with fixed defaults (`scripts/serve.py:38-103`).

`brainstorm-serve` hard-codes:
- upstream: `ws://localhost:8765`
- host: `localhost`
- port: `8000`
- process: `True`
- stateless: `True`
- difficulty: `hard`
- out_hz: `20.0`

So it is **not** a pure static server in current code; it launches the same unified backend stack with same-origin `/ws`.

### `brainstorm-backend`
This is the configurable entrypoint:
- `--upstream-url`
- `--host`
- `--port`
- `--static-dir`
- `--process/--no-process`
- `--difficulty`
- `--out-hz`
- `--stateless/--stateful`
(`scripts/backend.py:485-630`)

### Direct vs proxy modes
- **Direct mode / raw passthrough**: browser receives upstream `init` + `sample_batch` messages unchanged; no processed features.
- **Proxy/process mode**: browser receives `init`, lifecycle `status`, and periodic `features`.

### Live evaluation notes
Docs say the live server uses `ws://<server-ip>:8765/stream` and the control client uses `/control` (`docs/data_stream.md:56-61, 154-180`; `docs/submissions.md:32-63`).
- Those endpoints are documented for the evaluation environment.
- The local streamer in this repo exposes `WS /` (`scripts/stream_data.py:389-406`).
- The local backend browser endpoint is `WS /ws` (`scripts/backend.py:454-470`).

This is a useful mismatch to keep in mind when wiring the frontend.

---

## 5) HTTP endpoints the frontend uses beyond WebSocket

### In the checked-in frontend code
I found no `fetch()`/XHR/axios usage in `frontend/` or `frontend-old/`.
The current browser-side code in `frontend/src/main.ts` only mounts the Svelte app (`frontend/src/main.ts:1-6`).

### What the backend exposes
The backend does expose HTTP routes:
- `GET /health` -> JSON status (`scripts/backend.py:442-452`)

```ts
type HealthResponse = {
  status: "ok";
  upstream_connected: boolean;
  upstream_state: "connecting" | "connected" | "disconnected" | "ended";
  browser_clients: number;
}
```

- `GET /` -> `index.html`
- static files under `/` (for example `/favicon.svg`, `/src/main.ts`, or `/app.js` in tests)

But these are serving/monitoring endpoints, not app-data endpoints the frontend actively calls.

### Non-browser HTTP endpoint in docs
Docs also mention `/control` for the `brainstorm-control` tool during live eval (`docs/data_stream.md:154-180`). That is not a browser endpoint.

**Bottom line:** the frontend itself does not use any HTTP API beyond loading static assets; all runtime data is WS-based.

---

## 6) `signal_processing.py` module
Source: `scripts/signal_processing.py:21-431`.

### Types and settings
`ProcessingSettings` keys:
- `notch_60hz: bool`
- `bandpass: bool`
- `bad_channel_detection: bool`
- `spatial_sigma: float`
- `ema_alpha: float`

`DIFFICULTY_SETTINGS` selects those flags per preset (`scripts/signal_processing.py:21-58`).

### Bad-channel detection
`BadChannelDetector.detect()` returns:
- `bad_mask: np.ndarray[bool]` of length 1024
- `BadChannelInfo` with:
  - `dead: np.ndarray[int]`
  - `artifact: np.ndarray[int]`
  - `saturated: np.ndarray[int]`
  - `variance: np.ndarray[float]`

Rules:
- `dead`: variance < `1e-8`
- `artifact`: variance > median variance * 50
- `saturated`: peak-to-peak range < `1e-8`

### Filtering
`SignalFilter.process()`:
1. Optional 60 Hz notch (`iirnotch` -> SOS)
2. Optional 70–150 Hz bandpass (`butter(..., output="sos")`)
3. Stateful `sosfilt` across batches, one zi state per channel

### Power extraction
`PowerExtractor.process()`:
- squares filtered samples
- batch-averages power across samples
- maintains fast and slow EMA states
- returns:
  - `p_fast.copy()`
  - log-normalized `normalized = log(p_fast + 1e-12) - log(p_base + 1e-12)`

### Grid conversion / localization
`to_grid()`:
- reshapes 1024 vector to `(32,32)`
- zeros bad channels if a mask exists
- optional Gaussian blur with `sigma`

`weighted_centroid()`:
- thresholds below median (50th percentile)
- computes intensity-weighted centroid `[row, col]`
- returns grid center if empty

`compute_center_distance()`:
- normalized distance score in `[0,1]`
- `1.0` at grid center, `0.0` at a corner

### `NeuralProcessor.process_batch()` return shape
Returns either `None` (during bad-channel warmup) or:

```ts
type ProcessorResult = {
  heatmap: number[][];
  centroid: [number, number];
  center_distance: number;
  bad_channels: number;
}
```

Important:
- There is no `presence` field in the current code.
- `heatmap` is the exact matrix that becomes the browser `features.heatmap`.
- `bad_channels` is a count, not the mask itself.

---

## Start here
1. `scripts/backend.py:153-360` — this is the best single entry for upstream ingestion, processing mode selection, and browser payload construction.
2. `scripts/signal_processing.py:301-431` — this defines exactly what becomes `heatmap`, `centroid`, `center_distance`, and `bad_channels`.
3. `scripts/stream_data.py:259-327, 389-406` — this defines the upstream `init` and `sample_batch` JSON the backend consumes.
