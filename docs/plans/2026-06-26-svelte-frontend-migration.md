# Svelte 5 + TypeScript Frontend Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vibecoded vanilla JS frontend with a typed, decomposed Svelte 5 + TypeScript app that is functionally identical to the original.

**Architecture:** Pure functions (colormap, direction classification, alignment, dB conversion) extracted into standalone TS modules. A single reactive WebSocket client store drives the entire UI. Two canvas components receive data as props and draw imperatively in `$effect` blocks. Five small Svelte components own the seven reactive text nodes. Vite builds to `frontend/dist/`, backend serves it unchanged.

**Tech Stack:** Svelte 5 (runes), TypeScript, Vite 8, Canvas 2D API

**Reference files:**
- Old frontend: `frontend-old/app.js`, `frontend-old/index.html`, `frontend-old/style.css`
- Backend protocol: `scripts/backend.py:311-389` (features payload), `scripts/signal_processing.py:377-430` (processor output)
- Scaffold: `frontend/` (Vite + Svelte 5 scaffold, boilerplate to be stripped)

---

## File Structure

```
frontend/
  index.html                          # Entry HTML (title, fonts, mount point)
  vite.config.ts                      # Svelte plugin + /ws dev proxy + build output
  src/
    main.ts                           # Mount App into #app
    App.svelte                        # Root layout: header + canvas + sidebar
    app.css                           # Global styles (CSS custom properties, layout, responsive)
    lib/
      types.ts                        # WS protocol message types + internal domain types
      colormap.ts                     # Magma LUT generation + value-to-color mapping (pure)
      analysis.ts                     # Direction classification, alignment check, dB math (pure)
      timeseries-buffer.ts            # Bounded FIFO buffer for time series (pure data structure)
      ws.svelte.ts                    # WebSocket client: connect/reconnect, exposes $state
      renderers/
        heatmap.ts                    # Imperative heatmap draw function (pure canvas ops)
        timeseries.ts                 # Imperative time-series draw function (pure canvas ops)
    components/
      StatusBar.svelte                # Connection dot + status text + time + FPS + channel count
      HeatmapCanvas.svelte            # DPR-aware canvas wrapper, calls heatmap renderer
      CoverageCard.svelte             # Coverage percentage display with alignment highlight
      MoveCard.svelte                 # Direction instruction display
      TimeSeriesCanvas.svelte         # DPR-aware canvas wrapper, calls time-series renderer
```

### Responsibility boundaries

| Module | Owns | Does NOT touch |
|---|---|---|
| `types.ts` | All type definitions | Nothing else |
| `colormap.ts` | Magma LUT, `valueToColorIndex()` | Canvas, DOM, state |
| `analysis.ts` | `classifyDirection()`, `checkAlignment()`, `toDb()`, `computeEmaRange()` | Canvas, DOM, state |
| `timeseries-buffer.ts` | Bounded FIFO push/read | Canvas, DOM |
| `ws.svelte.ts` | WebSocket lifecycle, reconnect backoff, `$state` stores | Canvas, DOM layout |
| `heatmap.ts` | All `ctx.*` calls for the heatmap | DOM, stores, WebSocket |
| `timeseries.ts` | All `ctx.*` calls for the EKG plot | DOM, stores, WebSocket |
| `HeatmapCanvas.svelte` | Canvas element, DPR resize, wiring renderer to data | WebSocket, analysis |
| `TimeSeriesCanvas.svelte` | Canvas element, DPR resize, wiring renderer to data | WebSocket, analysis |
| `StatusBar.svelte` | 5 text nodes (dot, status, time, FPS, channels) | Canvas |
| `CoverageCard.svelte` | Coverage text + highlight style | Canvas |
| `MoveCard.svelte` | Direction text | Canvas |
| `App.svelte` | Layout, gluing components to stores | Rendering details |

---

## Task 1: Strip scaffold boilerplate + configure Vite

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/vite.config.ts`
- Modify: `frontend/src/main.ts`
- Delete: `frontend/src/App.svelte` (will be recreated in Task 8)
- Delete: `frontend/src/app.css` (will be recreated in Task 9)
- Delete: `frontend/src/lib/Counter.svelte`
- Delete: `frontend/src/assets/hero.png`
- Delete: `frontend/src/assets/svelte.svg`
- Delete: `frontend/src/assets/vite.svg`
- Delete: `frontend/public/icons.svg`

- [ ] **Step 1: Update `index.html`**

Replace `frontend/index.html` with:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Neural Data Viewer</title>
    <link rel="icon" href="data:," />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap"
      rel="stylesheet"
    />
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 2: Configure Vite with dev proxy**

Replace `frontend/vite.config.ts` with:

```ts
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
      "/health": {
        target: "http://localhost:8000",
      },
    },
  },
});
```

- [ ] **Step 3: Simplify `main.ts`**

Replace `frontend/src/main.ts` with:

```ts
import { mount } from "svelte";
import App from "./App.svelte";
import "./app.css";

mount(App, { target: document.getElementById("app")! });
```

- [ ] **Step 4: Delete scaffold files**

```bash
rm frontend/src/lib/Counter.svelte
rm frontend/src/assets/hero.png
rm frontend/src/assets/svelte.svg
rm frontend/src/assets/vite.svg
rm frontend/public/icons.svg
```

- [ ] **Step 5: Create placeholder `App.svelte` and `app.css`**

Create `frontend/src/App.svelte`:
```svelte
<h1>Neural Data Viewer</h1>
<p>Migration in progress…</p>
```

Create `frontend/src/app.css`:
```css
/* Placeholder — will be replaced in Task 9 */
body {
  font-family: "JetBrains Mono", monospace;
  background: #0a0f1a;
  color: #e8e8ed;
  margin: 0;
}
```

- [ ] **Step 6: Verify dev server starts**

```bash
cd frontend && bun install && bun run dev
```

Expected: Vite dev server starts on http://localhost:5173, shows "Neural Data Viewer" heading.

- [ ] **Step 7: Verify build succeeds**

```bash
cd frontend && bun run build
```

Expected: Build output in `frontend/dist/` with `index.html` and JS assets.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: strip scaffold boilerplate, configure vite proxy"
```

---

## Task 2: TypeScript types for the WebSocket protocol

**Files:**
- Create: `frontend/src/lib/types.ts`

These types are derived directly from `scripts/backend.py:344-360` and `scripts/signal_processing.py:377-430`.

- [ ] **Step 1: Create `types.ts`**

Create `frontend/src/lib/types.ts`:

```ts
// ── WebSocket messages (server → browser) ───────────────────────────

/** Sent once per connection with grid metadata. */
export interface InitMessage {
  type: "init";
  channels_coords: [number, number][];
  grid_size: number;
  fs: number;
  batch_size: number;
}

/** Lifecycle event when upstream connection state changes. */
export interface StatusMessage {
  type: "status";
  upstream_connected: boolean;
  upstream_state: UpstreamState;
  total_samples: number;
}

/** Processed neural features, emitted at ~20 Hz. */
export interface FeaturesMessage {
  type: "features";
  t: number;
  fs: number;
  n_ch: number;
  heatmap: number[][];
  centroid: [number, number]; // [row, col], 0-indexed
  center_distance: number; // 0 = corner, 1 = center
  confidence: number; // 1.0 if upstream connected
  bad_channels: number;
  total_samples: number;
}

/** Raw sample batch (only in --no-process / passthrough mode). */
export interface SampleBatchMessage {
  type: "sample_batch";
  neural_data: number[][];
  start_time_s: number;
  sample_count: number;
  fs: number;
}

export type ServerMessage =
  | InitMessage
  | StatusMessage
  | FeaturesMessage
  | SampleBatchMessage;

export type UpstreamState =
  | "connecting"
  | "connected"
  | "disconnected"
  | "ended";

// ── Derived domain types (internal, not from wire) ──────────────────

/** Connection state for the UI status bar. */
export type ConnectionStatus = "connected" | "connecting" | "disconnected";

/** Result of analyzing a features frame for display. */
export interface FrameAnalysis {
  direction: Direction;
  isAligned: boolean;
}

export type Direction =
  | "Center"
  | "Right"
  | "Left"
  | "Anterior"
  | "Posterior"
  | "Anterior-Right"
  | "Anterior-Left"
  | "Posterior-Right"
  | "Posterior-Left";

/** Auto-scaled colormap range, EMA-smoothed across frames. */
export interface ValueRange {
  vMin: number;
  vMax: number;
}
```

- [ ] **Step 2: Verify types compile**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add TypeScript types for WS protocol and domain"
```

---

## Task 3: Pure functions — colormap

**Files:**
- Create: `frontend/src/lib/colormap.ts`

This extracts `generateMagmaColormap()` and `valueToColorIndex()` from `frontend-old/app.js:87-179`. The key change: `valueToColorIndex` now takes `vMin`/`vMax` as parameters instead of reading globals.

- [ ] **Step 1: Create `colormap.ts`**

Create `frontend/src/lib/colormap.ts`:

```ts
import type { ValueRange } from "./types";

/** [r, g, b] each 0–255. */
export type RgbTuple = [number, number, number];

/** Magma control points (subset, interpolated to 256). */
const MAGMA_CONTROL_POINTS: [number, number, number][] = [
  [0.001462, 0.000466, 0.013866],
  [0.013708, 0.011771, 0.068667],
  [0.039608, 0.03109, 0.133515],
  [0.074257, 0.052017, 0.19351],
  [0.113094, 0.065492, 0.243537],
  [0.154901, 0.071327, 0.284065],
  [0.198177, 0.072245, 0.316356],
  [0.241397, 0.072699, 0.340836],
  [0.284124, 0.073417, 0.358296],
  [0.326438, 0.074167, 0.369846],
  [0.368567, 0.074621, 0.3764],
  [0.410791, 0.074866, 0.378497],
  [0.453187, 0.074686, 0.376427],
  [0.495784, 0.074295, 0.370369],
  [0.538516, 0.073859, 0.360437],
  [0.581246, 0.07348, 0.346753],
  [0.623796, 0.073307, 0.329512],
  [0.666022, 0.07359, 0.308947],
  [0.707797, 0.074578, 0.28538],
  [0.74898, 0.076556, 0.259246],
  [0.789417, 0.079868, 0.230962],
  [0.828991, 0.084937, 0.200963],
  [0.867534, 0.092252, 0.169642],
  [0.904837, 0.102306, 0.137338],
  [0.940621, 0.115594, 0.104286],
  [0.974449, 0.133635, 0.070619],
  [0.99556, 0.16538, 0.039886],
  [0.998085, 0.211843, 0.021563],
  [0.987053, 0.266188, 0.024335],
  [0.968443, 0.321898, 0.042144],
  [0.948683, 0.375586, 0.064264],
  [0.932067, 0.42671, 0.088087],
  [0.921248, 0.475767, 0.111534],
  [0.917482, 0.523424, 0.133798],
  [0.920858, 0.570213, 0.154815],
  [0.931674, 0.616411, 0.175091],
  [0.949545, 0.662198, 0.195563],
  [0.973381, 0.707719, 0.217587],
  [0.993248, 0.753418, 0.243755],
  [0.998364, 0.800551, 0.282327],
  [0.987622, 0.849251, 0.337977],
  [0.96968, 0.89756, 0.41032],
  [0.963855, 0.941167, 0.49],
  [0.9806, 0.9735, 0.5601],
  [0.987053, 0.991438, 0.749504],
];

function interpolate(
  points: [number, number, number][],
  size: number,
): RgbTuple[] {
  const result: RgbTuple[] = [];
  for (let i = 0; i < size; i++) {
    const t = (i / (size - 1)) * (points.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;

    if (idx >= points.length - 1) {
      const c = points[points.length - 1];
      result.push([
        Math.round(c[0] * 255),
        Math.round(c[1] * 255),
        Math.round(c[2] * 255),
      ]);
    } else {
      const c1 = points[idx];
      const c2 = points[idx + 1];
      result.push([
        Math.round((c1[0] + frac * (c2[0] - c1[0])) * 255),
        Math.round((c1[1] + frac * (c2[1] - c1[1])) * 255),
        Math.round((c1[2] + frac * (c2[2] - c1[2])) * 255),
      ]);
    }
  }
  return result;
}

/** 256-entry magma colormap, computed once at import time. */
export const MAGMA: readonly RgbTuple[] = interpolate(
  MAGMA_CONTROL_POINTS,
  256,
);

/** Map a value to a 0–255 colormap index given the current range. */
export function valueToColorIndex(value: number, range: ValueRange): number {
  const span = range.vMax - range.vMin;
  if (span === 0) return 0;
  const normalized = (value - range.vMin) / span;
  const clamped = Math.max(0, Math.min(1, normalized));
  return Math.round(clamped * 255);
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: extract magma colormap as pure TS module"
```

---

## Task 4: Pure functions — analysis (direction, alignment, dB, EMA range)

**Files:**
- Create: `frontend/src/lib/analysis.ts`

This extracts the direction classification from `frontend-old/app.js:335-380`, alignment check from `app.js:303-314`, dB conversion from `app.js:680-686`, and EMA range update from `app.js:231-242`. All four are currently inlined in impure functions. The extracted versions take explicit parameters and return values — no globals, no DOM.

- [ ] **Step 1: Create `analysis.ts`**

Create `frontend/src/lib/analysis.ts`:

```ts
import type { Direction, FrameAnalysis, ValueRange } from "./types";

const DB_EPS = 1e-12;

/**
 * Classify the direction from array center to centroid.
 * Ported from renderHeatmap() in frontend-old/app.js:335-380.
 *
 * @param centroid - [row, col] from backend (0-indexed)
 * @param gridRows - number of grid rows (typically 32)
 * @param gridCols - number of grid cols (typically 32)
 */
export function classifyDirection(
  centroid: [number, number],
  gridRows: number,
  gridCols: number,
): { direction: Direction; magnitude: number } {
  const [cy, cx] = centroid;
  const centerRow = gridRows / 2;
  const centerCol = gridCols / 2;
  const vecX = cx - centerCol;
  const vecY = cy - centerRow;
  const mag = Math.sqrt(vecX ** 2 + vecY ** 2);

  const CENTER_THRESHOLD = 1.5;
  if (mag <= CENTER_THRESHOLD) {
    return { direction: "Center", magnitude: mag };
  }

  const unitX = vecX / mag;
  const unitY = vecY / mag;

  let direction: Direction;
  if (Math.abs(unitY) < 0.3 && unitX > 0.3) direction = "Right";
  else if (Math.abs(unitY) < 0.3 && unitX < -0.3) direction = "Left";
  else if (Math.abs(unitX) < 0.3 && unitY > 0.3) direction = "Posterior";
  else if (Math.abs(unitX) < 0.3 && unitY < -0.3) direction = "Anterior";
  else if (unitX > 0 && unitY < 0) direction = "Anterior-Right";
  else if (unitX < 0 && unitY < 0) direction = "Anterior-Left";
  else if (unitX > 0 && unitY > 0) direction = "Posterior-Right";
  else direction = "Posterior-Left";

  return { direction, magnitude: mag };
}

/**
 * Check if the array is well-aligned with the surgical target.
 * Ported from renderHeatmap() in frontend-old/app.js:303-314.
 *
 * @param centroidMagnitude - distance from array center to centroid in grid units
 * @param coverage - center_distance value from backend (0–1)
 */
export function checkAlignment(
  centroidMagnitude: number,
  coverage: number,
): boolean {
  return centroidMagnitude < 3 && coverage * 100 > 70;
}

/**
 * Analyze a features frame: produces direction + alignment for display.
 * Combines classifyDirection and checkAlignment.
 */
export function analyzeFrame(
  centroid: [number, number],
  gridRows: number,
  gridCols: number,
  coverage: number,
): FrameAnalysis {
  const { direction, magnitude } = classifyDirection(
    centroid,
    gridRows,
    gridCols,
  );
  const isAligned = checkAlignment(magnitude, coverage);
  return { direction, isAligned };
}

/**
 * Convert mean power to dB relative to a baseline.
 * Ported from ws.onmessage in frontend-old/app.js:680-686.
 */
export function toDb(meanPower: number, baselinePower: number): number {
  return (
    10 * Math.log10(Math.max(meanPower, DB_EPS) / Math.max(baselinePower, DB_EPS))
  );
}

/**
 * Compute mean power over a 2D heatmap.
 * Ported from ws.onmessage in frontend-old/app.js:672-679.
 */
export function meanHeatmapPower(heatmap: number[][]): number {
  let sum = 0;
  let count = 0;
  for (const row of heatmap) {
    for (const v of row) {
      sum += v;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Update EMA-smoothed value range from a new heatmap frame.
 * Ported from renderHeatmap() in frontend-old/app.js:231-242.
 * Returns a new ValueRange (does not mutate input).
 *
 * @param prev - previous range
 * @param heatmap - current frame's 2D data
 * @param alpha - smoothing factor (0.1 in original)
 */
export function updateValueRange(
  prev: ValueRange,
  heatmap: number[][],
  alpha: number = 0.1,
): ValueRange {
  let maxVal = 0;
  let minVal = Infinity;
  for (const row of heatmap) {
    for (const v of row) {
      if (v > maxVal) maxVal = v;
      if (v < minVal) minVal = v;
    }
  }
  return {
    vMin: (1 - alpha) * prev.vMin + alpha * minVal,
    vMax: (1 - alpha) * prev.vMax + alpha * Math.max(maxVal, 0.001),
  };
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: extract analysis pure functions (direction, alignment, dB, EMA)"
```

---

## Task 5: Time series buffer

**Files:**
- Create: `frontend/src/lib/timeseries-buffer.ts`

The original uses a plain array with `push`/`shift` capped at 500 entries (`frontend-old/app.js:25-26, 497-504`). This extracts it as a typed class so the buffer logic is testable independently.

- [ ] **Step 1: Create `timeseries-buffer.ts`**

Create `frontend/src/lib/timeseries-buffer.ts`:

```ts
export interface TimeSeriesPoint {
  t: number;
  value: number;
}

/**
 * Bounded FIFO buffer for time series data.
 * Replaces the module-level timeSeriesData + maxTimeSeriesPoints from app.js.
 */
export class TimeSeriesBuffer {
  private data: TimeSeriesPoint[] = [];
  readonly capacity: number;

  constructor(capacity: number = 500) {
    this.capacity = capacity;
  }

  push(point: TimeSeriesPoint): void {
    this.data.push(point);
    if (this.data.length > this.capacity) {
      this.data.shift();
    }
  }

  get points(): readonly TimeSeriesPoint[] {
    return this.data;
  }

  get length(): number {
    return this.data.length;
  }

  clear(): void {
    this.data = [];
  }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add TimeSeriesBuffer data structure"
```

---

## Task 6: WebSocket client store

**Files:**
- Create: `frontend/src/lib/ws.svelte.ts`

This replaces the `connect()` function, all reconnect state (`reconnectDelay`, `reconnectTimer`, `userRequestedDisconnect`, `streamEnded`), and the `onmessage` dispatch logic from `frontend-old/app.js:603-709`. It owns connection lifecycle and exposes reactive `$state` for components to read.

The `.svelte.ts` extension enables Svelte 5 runes (`$state`) in a non-component module.

- [ ] **Step 1: Create `ws.svelte.ts`**

Create `frontend/src/lib/ws.svelte.ts`:

```ts
import type {
  ConnectionStatus,
  FeaturesMessage,
  InitMessage,
  ServerMessage,
  UpstreamState,
} from "./types";

export interface WsStore {
  /** Current connection status for the UI. */
  readonly status: ConnectionStatus;
  /** Status text shown in the status bar. */
  readonly statusText: string;
  /** Latest features frame, or null if none received yet. */
  readonly features: FeaturesMessage | null;
  /** Init metadata from the server. */
  readonly init: InitMessage | null;
  /** Connect (or reconnect) to the WebSocket server. */
  connect(): void;
  /** Disconnect and stop reconnecting. */
  disconnect(): void;
  /** Clean up (call on unmount). */
  destroy(): void;
}

/**
 * Create a reactive WebSocket client.
 *
 * Replaces: connect(), stopReconnectTimer(), updateStatus(),
 * updateUpstreamStatus(), and all reconnect globals from app.js.
 */
export function createWsStore(): WsStore {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectDelay = 1000;
  let userRequestedDisconnect = false;
  let streamEnded = false;

  // Reactive state (Svelte 5 runes)
  let status = $state<ConnectionStatus>("disconnected");
  let statusText = $state("Disconnected");
  let features = $state<FeaturesMessage | null>(null);
  let init = $state<InitMessage | null>(null);

  function stopReconnectTimer(): void {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function setStatus(s: ConnectionStatus, text?: string): void {
    status = s;
    switch (s) {
      case "connected":
        statusText = "Connected";
        break;
      case "connecting":
        statusText = text ?? "Connecting…";
        break;
      case "disconnected":
        statusText = text ?? "Disconnected";
        break;
    }
  }

  function handleUpstreamState(upstream: UpstreamState): void {
    switch (upstream) {
      case "connected":
        streamEnded = false;
        setStatus("connected");
        break;
      case "connecting":
        streamEnded = false;
        setStatus("connecting");
        break;
      case "ended":
        streamEnded = true;
        setStatus("disconnected", "Stream ended");
        stopReconnectTimer();
        break;
      case "disconnected":
        streamEnded = false;
        setStatus("connecting", "Waiting for stream…");
        break;
    }
  }

  function handleMessage(raw: string): void {
    try {
      const data = JSON.parse(raw) as ServerMessage;

      switch (data.type) {
        case "init":
          init = data;
          break;
        case "status":
          handleUpstreamState(data.upstream_state);
          break;
        case "features":
          features = data;
          break;
        case "sample_batch":
          // passthrough mode — not handled by this UI
          break;
      }
    } catch (err) {
      console.error("Error parsing WS message:", err);
    }
  }

  function connect(): void {
    if (status === "connected" && ws) {
      userRequestedDisconnect = true;
      ws.close();
      return;
    }

    userRequestedDisconnect = false;
    streamEnded = false;
    stopReconnectTimer();
    setStatus("connecting");

    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${window.location.host}/ws`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        reconnectDelay = 1000;
        setStatus("connecting", "Waiting for stream…");
      };

      ws.onmessage = (event: MessageEvent) => {
        handleMessage(event.data as string);
      };

      ws.onerror = () => {
        setStatus("disconnected", "Connection error");
      };

      ws.onclose = () => {
        ws = null;

        if (userRequestedDisconnect) {
          setStatus("disconnected");
          return;
        }
        if (streamEnded) {
          setStatus("disconnected", "Stream ended");
          return;
        }

        setStatus("connecting", "Reconnecting…");
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connect();
        }, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 10_000);
      };
    } catch {
      setStatus("disconnected", "Failed to connect");
    }
  }

  function disconnect(): void {
    userRequestedDisconnect = true;
    stopReconnectTimer();
    ws?.close();
  }

  function destroy(): void {
    disconnect();
  }

  return {
    get status() {
      return status;
    },
    get statusText() {
      return statusText;
    },
    get features() {
      return features;
    },
    get init() {
      return init;
    },
    connect,
    disconnect,
    destroy,
  };
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add reactive WebSocket client store"
```

---

## Task 7: Canvas renderers (imperative draw functions)

**Files:**
- Create: `frontend/src/lib/renderers/heatmap.ts`
- Create: `frontend/src/lib/renderers/timeseries.ts`

These are pure imperative functions: they take a canvas context + data and draw. No DOM queries, no state mutation, no Svelte dependency. This is the bulk of the ported rendering logic from `frontend-old/app.js:221-408` and `app.js:495-598`.

- [ ] **Step 1: Create the renderers directory**

```bash
mkdir -p frontend/src/lib/renderers
```

- [ ] **Step 2: Create `heatmap.ts`**

Create `frontend/src/lib/renderers/heatmap.ts`:

```ts
import type { FrameAnalysis, ValueRange } from "../types";
import { MAGMA, valueToColorIndex } from "../colormap";

export interface HeatmapDrawInput {
  ctx: CanvasRenderingContext2D;
  width: number; // CSS pixels
  height: number; // CSS pixels
  heatmap: number[][];
  centroid: [number, number]; // [row, col]
  range: ValueRange;
  analysis: FrameAnalysis;
  coverage: number;
}

/**
 * Draw the full heatmap frame: cells, grid overlay, border, centroid,
 * and guidance arrow.
 *
 * Ported from renderHeatmap() in frontend-old/app.js:221-408.
 * All direction/alignment logic has been moved to analysis.ts —
 * this function only draws based on pre-computed FrameAnalysis.
 */
export function drawHeatmap(input: HeatmapDrawInput): void {
  const { ctx, width, height, heatmap, centroid, range, analysis } = input;
  const { isAligned } = analysis;

  const rows = heatmap.length;
  const cols = heatmap[0].length;

  // Layout math
  const size = Math.min(width, height);
  const padding = Math.max(4, Math.floor(size * 0.04));
  const plotSize = Math.max(1, (size - 2 * padding) * 0.85);
  const cellSize = plotSize / cols;

  // Clear
  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, width, height);

  // Centroid offset: center the hotspot at canvas center
  const [cy, cx] = centroid;
  const centerX = width / 2;
  const centerY = height / 2;
  const offsetX = centerX - (padding + (cx + 0.5) * cellSize);
  const offsetY = centerY - (padding + (cy + 0.5) * cellSize);

  // Draw cells
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const colorIdx = valueToColorIndex(heatmap[row][col], range);
      const [r, g, b] = MAGMA[colorIdx];
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(
        padding + col * cellSize + offsetX,
        padding + row * cellSize + offsetY,
        cellSize + 0.5,
        cellSize + 0.5,
      );
    }
  }

  // Grid overlay
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (let i = 0; i <= cols; i++) {
    const x = padding + i * cellSize + offsetX;
    ctx.moveTo(x, padding + offsetY);
    ctx.lineTo(x, padding + plotSize + offsetY);
  }
  for (let i = 0; i <= rows; i++) {
    const y = padding + i * cellSize + offsetY;
    ctx.moveTo(padding + offsetX, y);
    ctx.lineTo(padding + plotSize + offsetX, y);
  }
  ctx.stroke();

  // Border
  ctx.strokeStyle = isAligned
    ? "rgba(76, 222, 128, 0.8)"
    : "rgba(89, 224, 255, 0.5)";
  ctx.lineWidth = isAligned ? 12 : 4;
  ctx.strokeRect(padding + offsetX, padding + offsetY, plotSize, plotSize);

  // Centroid circle (always at canvas center)
  ctx.strokeStyle = isAligned ? "rgba(76, 222, 128, 0.9)" : "cyan";
  ctx.lineWidth = isAligned ? 12 : 4;
  ctx.beginPath();
  ctx.arc(centerX, centerY, Math.max(15, cellSize * 1.5), 0, 2 * Math.PI);
  ctx.stroke();

  // Guidance arrow (from array center to centroid)
  const arrayCenterRow = rows / 2;
  const arrayCenterCol = cols / 2;
  const vecX = cx - arrayCenterCol;
  const vecY = cy - arrayCenterRow;
  const mag = Math.sqrt(vecX ** 2 + vecY ** 2);

  if (mag > 1.5) {
    const arrayCenterX =
      padding + (arrayCenterCol + 0.5) * cellSize + offsetX;
    const arrayCenterY =
      padding + (arrayCenterRow + 0.5) * cellSize + offsetY;

    // Line from array center to canvas center
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(arrayCenterX, arrayCenterY);
    ctx.lineTo(centerX, centerY);
    ctx.stroke();

    // Arrowhead at midpoint
    const midX = (arrayCenterX + centerX) / 2;
    const midY = (arrayCenterY + centerY) / 2;
    const arrowSize = 20;
    const angle = Math.atan2(centerY - arrayCenterY, centerX - arrayCenterX);

    ctx.fillStyle = "lime";
    ctx.beginPath();
    ctx.moveTo(midX, midY);
    ctx.lineTo(
      midX - arrowSize * Math.cos(angle - Math.PI / 6),
      midY - arrowSize * Math.sin(angle - Math.PI / 6),
    );
    ctx.lineTo(
      midX - arrowSize * Math.cos(angle + Math.PI / 6),
      midY - arrowSize * Math.sin(angle + Math.PI / 6),
    );
    ctx.closePath();
    ctx.fill();
  }
}
```

- [ ] **Step 3: Create `timeseries.ts`**

Create `frontend/src/lib/renderers/timeseries.ts`:

```ts
import type { TimeSeriesPoint } from "../timeseries-buffer";

export interface TimeSeriesDrawInput {
  ctx: CanvasRenderingContext2D;
  width: number; // CSS pixels
  height: number; // CSS pixels
  points: readonly TimeSeriesPoint[];
}

const PADDING = { top: 20, right: 20, bottom: 30, left: 50 } as const;

/**
 * Draw the EKG-style high-gamma time series plot.
 *
 * Ported from plotTimeSeries() in frontend-old/app.js:495-598.
 * Newest data is positioned at 70% of the plot width.
 */
export function drawTimeSeries(input: TimeSeriesDrawInput): void {
  const { ctx, width, height, points } = input;

  // Clear
  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, width, height);

  if (points.length < 2) return;

  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;

  // Value range
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (const p of points) {
    if (p.value < minValue) minValue = p.value;
    if (p.value > maxValue) maxValue = p.value;
  }
  const valueRange = maxValue - minValue || 1;

  // Time range: newest at 70% of plot width
  const latestTime = points[points.length - 1].t;
  const oldestTime = points[0].t;
  const actualTimeRange = latestTime - oldestTime || 1;
  const displayTimeRange = actualTimeRange / 0.7;

  // Axes
  ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PADDING.left, PADDING.top);
  ctx.lineTo(PADDING.left, height - PADDING.bottom);
  ctx.lineTo(width - PADDING.right, height - PADDING.bottom);
  ctx.stroke();

  // Horizontal grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PADDING.top + (plotHeight * i) / 4;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, y);
    ctx.lineTo(width - PADDING.right, y);
    ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = maxValue - (valueRange * i) / 4;
    const y = PADDING.top + (plotHeight * i) / 4;
    ctx.fillText(value.toFixed(2), PADDING.left - 5, y);
  }

  // Data trace
  ctx.strokeStyle = "#ff6b6b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const timeFromLatest = latestTime - p.t;
    const normalizedPos = 0.7 - timeFromLatest / displayTimeRange;
    const x = PADDING.left + normalizedPos * plotWidth;
    const y =
      height - PADDING.bottom - ((p.value - minValue) / valueRange) * plotHeight;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.font = "12px JetBrains Mono, monospace";
  ctx.textAlign = "center";
  ctx.fillText("Time (s)", width / 2, height - 5);

  ctx.save();
  ctx.translate(10, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Mean Power (dB)", 0, 0);
  ctx.restore();
}
```

- [ ] **Step 4: Verify both compile**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add imperative canvas renderers for heatmap and time series"
```

---

## Task 8: Svelte components

**Files:**
- Create: `frontend/src/components/StatusBar.svelte`
- Create: `frontend/src/components/CoverageCard.svelte`
- Create: `frontend/src/components/MoveCard.svelte`
- Create: `frontend/src/components/HeatmapCanvas.svelte`
- Create: `frontend/src/components/TimeSeriesCanvas.svelte`

Each component owns exactly its piece of the DOM. Canvas components handle DPR-aware resize via `ResizeObserver` in `onMount`.

- [ ] **Step 1: Create `StatusBar.svelte`**

Create `frontend/src/components/StatusBar.svelte`:

```svelte
<script lang="ts">
  import type { ConnectionStatus } from "../lib/types";

  interface Props {
    status: ConnectionStatus;
    statusText: string;
    time: number;
    fps: number;
    channelCount: number;
    gridSize: number;
  }

  let { status, statusText, time, fps, channelCount, gridSize }: Props =
    $props();
</script>

<div class="status-bar">
  <span class="status-indicator {status}"></span>
  <span>{statusText}</span>
  <span class="separator">|</span>
  <span class="accent">t = {time.toFixed(2)}s</span>
  <span class="separator">|</span>
  <span class="accent">{fps} FPS</span>
  <span class="separator">|</span>
  <span class="accent"
    >{channelCount} channels ({gridSize}x{gridSize} data)</span
  >
</div>
```

- [ ] **Step 2: Create `CoverageCard.svelte`**

Create `frontend/src/components/CoverageCard.svelte`:

```svelte
<script lang="ts">
  interface Props {
    coverage: number;
  }

  let { coverage }: Props = $props();

  let percent = $derived(coverage * 100);
  let isGood = $derived(percent > 70);
</script>

<div class="info-card card card-coverage card-centered" class:aligned={isGood}>
  <h2>Surgical Target Coverage</h2>
  <h1>
    <span class="card-body" style:color={isGood ? "#4ade80" : undefined}>
      Coverage: {percent.toFixed(1)}%
    </span>
  </h1>
</div>

<style>
  .aligned {
    border-color: rgba(76, 222, 128, 0.6);
  }
</style>
```

- [ ] **Step 3: Create `MoveCard.svelte`**

Create `frontend/src/components/MoveCard.svelte`:

```svelte
<script lang="ts">
  import type { Direction } from "../lib/types";

  interface Props {
    direction: Direction;
  }

  let { direction }: Props = $props();
</script>

<div class="info-card card card-move card-centered">
  <h2>Move Instruction</h2>
  <h1><span class="card-body" id="direction-display">{direction}</span></h1>
</div>
```

- [ ] **Step 4: Create `HeatmapCanvas.svelte`**

Create `frontend/src/components/HeatmapCanvas.svelte`:

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import type { FeaturesMessage, FrameAnalysis, ValueRange } from "../lib/types";
  import { drawHeatmap } from "../lib/renderers/heatmap";
  import { analyzeFrame, updateValueRange } from "../lib/analysis";

  interface Props {
    features: FeaturesMessage | null;
    onAnalysis?: (analysis: FrameAnalysis) => void;
  }

  let { features, onAnalysis }: Props = $props();

  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let cssWidth = 0;
  let cssHeight = 0;
  let valueRange: ValueRange = { vMin: 0, vMax: 0.01 };

  // FPS tracking
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let fps = $state(0);

  onMount(() => {
    ctx = canvasEl.getContext("2d");

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      const rect = entry.contentRect;
      cssWidth = Math.floor(rect.width);
      cssHeight = Math.floor(rect.height);
      const dpr = window.devicePixelRatio || 1;

      canvasEl.width = Math.floor(cssWidth * dpr);
      canvasEl.height = Math.floor(cssHeight * dpr);
      canvasEl.style.width = `${cssWidth}px`;
      canvasEl.style.height = `${cssHeight}px`;

      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = "#0a0a0f";
        ctx.fillRect(0, 0, cssWidth, cssHeight);
      }
    });

    observer.observe(canvasEl.parentElement!);
    return () => observer.disconnect();
  });

  // Draw on each new features frame
  $effect(() => {
    if (!features || !ctx || cssWidth === 0) return;

    const { heatmap, centroid, center_distance: coverage } = features;
    const rows = heatmap.length;
    const cols = heatmap[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    // Update EMA range
    valueRange = updateValueRange(valueRange, heatmap);

    // Analyze frame
    const analysis = analyzeFrame(centroid, rows, cols, coverage);
    onAnalysis?.(analysis);

    // Draw
    drawHeatmap({
      ctx,
      width: cssWidth,
      height: cssHeight,
      heatmap,
      centroid,
      range: valueRange,
      analysis,
      coverage,
    });

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFpsTime = now;
    }
  });

  export { fps };
</script>

<canvas bind:this={canvasEl}></canvas>

<style>
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    border-radius: 8px;
  }
</style>
```

- [ ] **Step 5: Create `TimeSeriesCanvas.svelte`**

Create `frontend/src/components/TimeSeriesCanvas.svelte`:

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import type { FeaturesMessage } from "../lib/types";
  import { meanHeatmapPower, toDb } from "../lib/analysis";
  import { TimeSeriesBuffer } from "../lib/timeseries-buffer";
  import { drawTimeSeries } from "../lib/renderers/timeseries";

  interface Props {
    features: FeaturesMessage | null;
  }

  let { features }: Props = $props();

  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let cssWidth = 0;
  let cssHeight = 0;

  const buffer = new TimeSeriesBuffer(500);
  let baselinePower: number | null = null;

  onMount(() => {
    ctx = canvasEl.getContext("2d");

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      const rect = entry.contentRect;
      cssWidth = Math.floor(rect.width);
      cssHeight = Math.floor(rect.height);
      const dpr = window.devicePixelRatio || 1;

      canvasEl.width = Math.floor(cssWidth * dpr);
      canvasEl.height = Math.floor(cssHeight * dpr);
      canvasEl.style.width = `${cssWidth}px`;
      canvasEl.style.height = `${cssHeight}px`;

      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    });

    observer.observe(canvasEl.parentElement!);
    return () => observer.disconnect();
  });

  $effect(() => {
    if (!features || !ctx || cssWidth === 0) return;

    const meanPower = meanHeatmapPower(features.heatmap);

    // Latch baseline on first frame
    if (baselinePower === null) {
      baselinePower = Math.max(meanPower, 1e-12);
    }

    const dbPower = toDb(meanPower, baselinePower);
    buffer.push({ t: features.t, value: dbPower });

    drawTimeSeries({
      ctx,
      width: cssWidth,
      height: cssHeight,
      points: buffer.points,
    });
  });
</script>

<canvas bind:this={canvasEl}></canvas>

<style>
  canvas {
    flex: 0 0 auto;
    height: clamp(180px, 24vh, 260px);
    min-height: 180px;
    width: 100%;
    display: block;
  }
</style>
```

- [ ] **Step 6: Verify all components compile**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors. (There will be no visual test yet — `App.svelte` doesn't use them.)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: add all Svelte components (StatusBar, cards, canvases)"
```

---

## Task 9: Global styles

**Files:**
- Rewrite: `frontend/src/app.css`

Ported directly from `frontend-old/style.css`. The only changes: remove unused `.canvas-container` rule, and keep all CSS custom properties, layout, card styles, responsive breakpoints, and status indicator animations intact.

- [ ] **Step 1: Replace `app.css`**

Replace `frontend/src/app.css` with the full contents of `frontend-old/style.css` (all 313 lines). This is a direct copy — the styles are already correct and the component class names match.

The file is too long to inline here. The exact command:

```bash
cp frontend-old/style.css frontend/src/app.css
```

Then remove the unused `.canvas-container` block (lines ~186-193 of `style.css`) since no component uses that class.

- [ ] **Step 2: Verify build works**

```bash
cd frontend && bun run build
```

Expected: Build succeeds, `dist/` contains compiled CSS.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: port global styles from old frontend"
```

---

## Task 10: Wire up `App.svelte` (root layout)

**Files:**
- Rewrite: `frontend/src/App.svelte`

This is the root component that mirrors the layout from `frontend-old/index.html`. It creates the WS store, connects on mount, and passes data down to child components.

- [ ] **Step 1: Replace `App.svelte`**

Create `frontend/src/App.svelte`:

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import { createWsStore } from "./lib/ws.svelte";
  import type { FrameAnalysis } from "./lib/types";
  import StatusBar from "./components/StatusBar.svelte";
  import HeatmapCanvas from "./components/HeatmapCanvas.svelte";
  import CoverageCard from "./components/CoverageCard.svelte";
  import MoveCard from "./components/MoveCard.svelte";
  import TimeSeriesCanvas from "./components/TimeSeriesCanvas.svelte";

  const store = createWsStore();

  let analysis: FrameAnalysis | null = $state(null);
  let heatmapFps = $state(0);

  // Derived values from latest features
  let time = $derived(store.features?.t ?? 0);
  let channelCount = $derived(store.features?.n_ch ?? 0);
  let gridSize = $derived(store.features?.heatmap?.length ?? 0);
  let coverage = $derived(store.features?.center_distance ?? 0);
  let direction = $derived(analysis?.direction ?? "Center");

  onMount(() => {
    store.connect();
    return () => store.destroy();
  });

  function handleAnalysis(a: FrameAnalysis) {
    analysis = a;
  }
</script>

<div class="container">
  <header>
    <h1>Neural Activity</h1>
    <StatusBar
      status={store.status}
      statusText={store.statusText}
      {time}
      fps={heatmapFps}
      {channelCount}
      {gridSize}
    />
  </header>

  <main>
    <div class="layout">
      <section class="panel-left panel">
        <HeatmapCanvas
          features={store.features}
          onAnalysis={handleAnalysis}
          bind:fps={heatmapFps}
        />
      </section>
      <aside class="sidebar">
        <CoverageCard {coverage} />
        <MoveCard {direction} />
        <div class="info-card panel signal-monitor">
          <h2>High Gamma</h2>
          <TimeSeriesCanvas features={store.features} />
        </div>
      </aside>
    </div>
  </main>
</div>
```

- [ ] **Step 2: Verify dev server renders correctly**

Start the full stack:
```bash
# Terminal 1: data streamer
uv run brainstorm-stream --from-file data/hard/

# Terminal 2: backend
uv run brainstorm-backend --upstream-url ws://localhost:8765

# Terminal 3: frontend dev
cd frontend && bun run dev
```

Open http://localhost:5173. Expected:
- Status bar shows connection state, then "Connected"
- Heatmap renders with magma colormap, centroid circle, guidance arrow
- Coverage card shows percentage, highlights green when >70%
- Move instruction shows direction text
- High-gamma chart draws the EKG-style trace
- FPS counter updates (~20 FPS, matching backend `out_hz`)

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: wire up App.svelte root layout with all components"
```

---

## Task 11: Backend integration (static_dir for built output)

**Files:**
- Modify: `scripts/static_assets.py` (update default resolution to prefer `frontend/dist`)
- Modify: `start_all.sh` (add frontend build step if script exists)
- Modify: `Makefile` (add frontend targets)

- [ ] **Step 1: Update `static_assets.py`**

In `scripts/static_assets.py`, update `resolve_static_dir()` so that when the requested directory is `frontend` and `frontend/dist` exists, it prefers `frontend/dist`. Read the current file first to find the exact code to modify.

The key change: after resolving the path, check if `resolved / "dist"` exists and prefer it:

```python
def resolve_static_dir(requested: str = "frontend") -> Path:
    resolved = Path(requested)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved

    # Prefer built output if it exists
    dist = resolved / "dist"
    if dist.exists():
        return dist

    if resolved.exists():
        return resolved

    # Fallback chain (unchanged)
    ...
```

- [ ] **Step 2: Add frontend targets to `Makefile`**

Append to `Makefile`:

```makefile
# Frontend
.PHONY: frontend-install frontend-dev frontend-build frontend-check

frontend-install:
	cd frontend && bun install

frontend-dev:
	cd frontend && bun run dev

frontend-build:
	cd frontend && bun run build

frontend-check:
	cd frontend && bun run check
```

Update the existing `install` target to include `frontend-install` as a dependency.
Update `check-all` to include `frontend-check`.

- [ ] **Step 3: Update `start_all.sh`** (if it exists)

Add a frontend build step before starting the backend:

```bash
echo "Building frontend..."
(cd frontend && bun install && bun run build)
```

- [ ] **Step 4: Verify production-like serving**

```bash
cd frontend && bun run build
cd .. && uv run brainstorm-backend --upstream-url ws://localhost:8765
```

Open http://localhost:8000. Expected: same behavior as the dev server, served from built assets.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: integrate frontend build with backend static serving"
```

---

## Task 12: Update documentation

**Files:**
- Modify: `AGENTS.md` (update architecture section, add frontend build step to daily workflow)
- Modify: `docs/getting_started.md` (document `bun install`, `bun run dev`, `bun run build`)

- [ ] **Step 1: Update `AGENTS.md`**

Key changes:
- Daily workflow: add `cd frontend && bun install` to install step, document `bun run dev` for frontend development
- Architecture diagram: update `frontend/` to note it's a Svelte 5 + TS app built with Vite
- Remove or update the note "keep `frontend/` build-less; if introducing bundlers, document steps"
- Add `bun run build` to the `start_all.sh` flow

- [ ] **Step 2: Update `docs/getting_started.md`**

Add a "Frontend Development" section documenting:
- Prerequisites: bun (or npm/pnpm)
- `cd frontend && bun install`
- `bun run dev` — starts Vite dev server on :5173 with WS proxy to :8000
- `bun run build` — builds to `frontend/dist/`, served by backend
- `bun run check` — runs `svelte-check` for type errors

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs: update AGENTS.md and getting_started for Svelte frontend"
```

---

## Task 13: Cleanup

**Files:**
- Delete: `frontend-old/` (after confirming everything works)

- [ ] **Step 1: Final verification**

Run the full stack and verify all features work:
```bash
# Build frontend
cd frontend && bun run build && cd ..

# Start services
uv run brainstorm-stream --from-file data/hard/ &
uv run brainstorm-backend --upstream-url ws://localhost:8765 &

# Open http://localhost:8000 and verify:
# - Heatmap renders with correct colors and centroid tracking
# - Coverage card updates and highlights green when aligned
# - Move instruction shows correct direction
# - Time series plot draws EKG trace
# - Status bar shows all telemetry
# - Reconnect works if backend restarts
```

- [ ] **Step 2: Run type check**

```bash
cd frontend && bun run check
```

Expected: No errors.

- [ ] **Step 3: Remove old frontend**

```bash
rm -rf frontend-old
```

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "chore: remove old vanilla JS frontend"
```
