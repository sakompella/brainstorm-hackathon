# Code Context

## Files Retrieved
1. `frontend-old/app.js` (lines 1-733) - all app state, render logic, WebSocket lifecycle, and DOM writes.
2. `frontend-old/index.html` (lines 1-54) - the entire DOM surface JS can touch.
3. `frontend-old/style.css` (lines 1-313) - layout, responsive breakpoints, and JS-coupled visual states.

## Key Code

### 1) Module-level state (`app.js:7-47`)
| Name | Type | Purpose | Lifecycle |
|---|---|---|---|
| `MAGMA_COLORMAP` | `Array<[number,number,number]>` | 256-color lookup table for heatmap coloring | Latched once at module load via `generateMagmaColormap()` |
| `ws` | `WebSocket \| null` | Current live socket to `/ws` | Assigned in `connect()`, nulled in `ws.onclose` |
| `gridSize` | `number` | Declared 32, but unused in this file | Latched constant-like dead state |
| `isConnected` | `boolean` | UI/connection toggle state | Set in `updateStatus()` |
| `frameCount` | `number` | FPS accumulator | Incremented every `renderHeatmap()` call, reset once/sec |
| `lastFpsUpdate` | `number` | Timestamp for FPS window | Updated once/sec in `renderHeatmap()` |
| `currentFps` | `number` | Latest computed FPS | Updated once/sec in `renderHeatmap()` |
| `currentTime` | `number` | Latest stream timestamp | Updated on each `features` message |
| `baselineMeanPower` | `number \| null` | Reference power for dB conversion | Latched on first valid heatmap, then never reset |
| `DB_EPS` | `number` | Numerical floor to avoid log(0) | Constant |
| `timeSeriesData` | `Array<{t:number,value:number}>` | Bounded history for plot | Append per feature message; `shift()` when over limit (FIFO buffer, not circular ring) |
| `maxTimeSeriesPoints` | `number` | Upper bound for history length | Constant-ish cap |
| `timeSeriesCanvas` / `timeSeriesCtx` | `HTMLCanvasElement \| null` / `CanvasRenderingContext2D \| null` | High-gamma chart target | Set once in `initCanvas()`, resized later |
| `canvas` / `ctx` | `HTMLCanvasElement \| null` / `CanvasRenderingContext2D \| null` | Main heatmap target | Set once in `initCanvas()`, resized later |
| `canvasWidth` / `canvasHeight` | `number` | CSS-pixel size of main canvas | Updated on each resize |
| `vMin` / `vMax` | `number` | Auto-scaled colormap range | Exponential smoothing each heatmap render (`0.9*old + 0.1*new`) |
| `userRequestedDisconnect` | `boolean` | Distinguish deliberate close vs reconnect | Set in `connect()` before closing socket |
| `streamEnded` | `boolean` | Stop reconnecting when upstream ends | Set by `updateUpstreamStatus()` |
| `reconnectDelay` | `number` | Exponential backoff delay (ms) | Doubles on reconnect attempts, capped at 10000 |
| `reconnectTimer` | `TimeoutID \| null` | Pending reconnect timer | Set in `ws.onclose`, cleared by `stopReconnectTimer()` |

### 2) Named functions (`app.js:49-733`)

#### `stopReconnectTimer()` (`49-54`)
- **Inputs:** none
- **Outputs:** none
- **Side effects:** `clearTimeout(reconnectTimer)`; sets `reconnectTimer = null`
- **Called by:** `connect()`, `updateUpstreamStatus()`, `ws.onclose`
- **Pure?** No (timer mutation)

#### `resizeCanvasToContainer()` (`59-82`)
- **Inputs:** none
- **Outputs:** none
- **Side effects:** reads `canvas.parentElement.getBoundingClientRect()`, mutates `canvas.width/height/style`, updates `canvasWidth/Height`, calls `ctx.setTransform()` and clears canvas
- **Called by:** `initCanvas()`, resize handler in `init()`
- **Pure?** No (DOM/canvas mutation)

#### `generateMagmaColormap()` (`87-160`)
- **Inputs:** none
- **Outputs:** `Array<[r,g,b]>`
- **Side effects:** none
- **Called by:** top-level `MAGMA_COLORMAP` initialization
- **Pure?** Yes

#### `valueToColorIndex(value)` (`166-170`)
- **Inputs:** numeric value
- **Outputs:** integer 0-255
- **Side effects:** none
- **Dependencies:** mutable globals `vMin`, `vMax`
- **Called by:** `valueToColor()`, `renderHeatmap()`
- **Pure?** Not as written (implicit mutable globals)

#### `valueToColor(value)` (`175-179`)
- **Inputs:** numeric value
- **Outputs:** CSS `rgb(r,g,b)` string
- **Side effects:** none
- **Dependencies:** `valueToColorIndex()`, `MAGMA_COLORMAP`
- **Called by:** none in current file
- **Pure?** Not as written (depends on mutable `vMin/vMax`)

#### `initCanvas()` (`184-195`)
- **Inputs:** none
- **Outputs:** none
- **Side effects:** `document.getElementById()` for both canvases, assigns `ctx/timeSeriesCtx`, calls resize helpers
- **Called by:** `init()`
- **Pure?** No

#### `resizeTimeSeriesCanvas()` (`200-216`)
- **Inputs:** none
- **Outputs:** none
- **Side effects:** mutates `timeSeriesCanvas.width/height/style`, sets transform on `timeSeriesCtx`
- **Called by:** `initCanvas()`, resize handler in `init()`
- **Pure?** No

#### `renderHeatmap(heatmap, centroid, coverage)` (`221-408`)
- **Inputs:** 2D numeric array, optional `[y,x]` centroid, optional coverage scalar
- **Outputs:** none
- **Side effects:** extensive canvas drawing; DOM write to `#direction-display`; FPS state mutation; DOM write to `#fps-counter` once/sec; mutates `vMin/vMax`
- **Called by:** `ws.onmessage` for `features`
- **Pure?** No

#### `updateStatus(status, text)` (`414-437`)
- **Inputs:** status string + optional text
- **Outputs:** none
- **Side effects:** writes `#status-indicator.className`, `classList`, `#status-text.textContent`; mutates `isConnected`
- **Called by:** `updateUpstreamStatus()`, `connect()`, `ws.onerror`, `ws.onclose`
- **Pure?** No

#### `updateUpstreamStatus(data)` (`439-460`)
- **Inputs:** status payload from upstream middleware
- **Outputs:** none
- **Side effects:** mutates `streamEnded`; calls `updateStatus()`; may stop reconnect timer
- **Called by:** `ws.onmessage` when `data.type === 'status'`
- **Pure?** No

#### `updateInfoCards(data)` (`465-490`)
- **Inputs:** features payload
- **Outputs:** `center` (`data.center_distance || 0`)
- **Side effects:** writes `.card-coverage .card-body.textContent`, `style.color`, `.card-coverage.style.borderColor`
- **Called by:** `ws.onmessage` for `features`
- **Pure?** No

#### `plotTimeSeries(meanPowerDb, timestamp)` (`495-598`)
- **Inputs:** dB value and timestamp
- **Outputs:** none
- **Side effects:** mutates `timeSeriesData` (push/shift), draws full chart, clears canvas, labels axes
- **Called by:** `ws.onmessage` for `features`
- **Pure?** No

#### `connect()` (`603-709`)
- **Inputs:** none
- **Outputs:** none
- **Side effects:** constructs WebSocket URL from `window.location`, opens socket, attaches all socket handlers, mutates reconnect state and status state
- **Called by:** `init()`, reconnect timer in `ws.onclose()`
- **Pure?** No

#### `init()` (`714-730`)
- **Inputs:** none
- **Outputs:** none
- **Side effects:** initializes canvases, installs resize listener, starts connection
- **Called by:** `DOMContentLoaded` listener
- **Pure?** No

### 3) Inline callback functions (`app.js:718-733`, `622-703`)
- `window.addEventListener('resize', () => {...})`: debounce via local `resizeScheduled`, schedules canvas resizes in `requestAnimationFrame`.
- `requestAnimationFrame(() => {...})`: performs resize work and resets `resizeScheduled`.
- `ws.onopen = () => {...}`: logs, resets `reconnectDelay`, shows waiting state.
- `ws.onmessage = (event) => {...}`: parses JSON, routes `status` vs `features`, updates DOM, renders, computes dB power, plots chart.
- `ws.onerror = (error) => {...}`: logs and sets disconnected status.
- `ws.onclose = (event) => {...}`: clears `ws`, decides whether to reconnect, schedules backoff.
- `document.addEventListener('DOMContentLoaded', init)`: boot hook.

## Architecture

### Data flow
1. `DOMContentLoaded` fires → `init()`.
2. `init()` → `initCanvas()` → grabs `#neural-canvas` + `#timeseries-canvas`, gets 2D contexts, sizes both to their containers with DPR scaling.
3. `init()` installs resize debounce and calls `connect()`.
4. `connect()` opens `ws://<host>/ws` (or `wss:` on HTTPS) and wires socket handlers.
5. On `status` messages:
   - `ws.onmessage` → `updateUpstreamStatus(data)` → `updateStatus(...)` → writes status pill/text and flips `isConnected`.
6. On `features` messages:
   - `ws.onmessage` updates `currentTime` and `#time-display`.
   - It writes `#channel-count` using a hard-coded `32*32` channel count plus incoming heatmap size.
   - `updateInfoCards(data)` writes coverage text and optional highlight styling.
   - `renderHeatmap(heatmap, centroid, coverage)` draws the main canvas, updates `#direction-display`, and may refresh `#fps-counter`.
   - Mean power is computed from the whole heatmap, first sample latches `baselineMeanPower`, then dB is computed.
   - `plotTimeSeries(dbPower, currentTime)` appends to `timeSeriesData` and redraws the high-gamma chart.
7. On disconnect/stream end:
   - `ws.onclose` decides whether to reconnect, stop, or stay ended.

### Render / state coupling
- Main canvas state is driven by `canvasWidth/Height`, `vMin/vMax`, `frameCount`, `lastFpsUpdate`, and centroid/coverage from the message.
- Time-series state is driven by `timeSeriesData`, `baselineMeanPower`, and `currentTime`.
- Connection state is driven by `isConnected`, `userRequestedDisconnect`, `streamEnded`, `reconnectDelay`, and `reconnectTimer`.

## DOM Surface

| Element | Selector | Read by | Written by |
|---|---|---|---|
| Main canvas | `#neural-canvas` | `initCanvas()`, `resizeCanvasToContainer()`, `renderHeatmap()` | `initCanvas()`, `resizeCanvasToContainer()`, `renderHeatmap()` via `ctx` |
| Time-series canvas | `#timeseries-canvas` | `initCanvas()`, `resizeTimeSeriesCanvas()`, `plotTimeSeries()` | `initCanvas()`, `resizeTimeSeriesCanvas()`, `plotTimeSeries()` via `timeSeriesCtx` |
| Status dot | `#status-indicator` | `updateStatus()` | `updateStatus()` (`className`, `classList`) |
| Status text | `#status-text` | `updateStatus()` | `updateStatus()` (`textContent`) |
| Time readout | `#time-display` | none | `ws.onmessage` (`textContent`) |
| FPS readout | `#fps-counter` | none | `renderHeatmap()` (`textContent`) |
| Channel readout | `#channel-count` | none | `ws.onmessage` (`textContent`) |
| Coverage body | `.card-coverage .card-body` / `#coverage-display` | `updateInfoCards()` | `updateInfoCards()` (`textContent`, `style.color`) |
| Coverage card | `.card-coverage` | `updateInfoCards()` | `updateInfoCards()` (`style.borderColor`) |
| Direction display | `#direction-display` | none | `renderHeatmap()` (`textContent`) |
| Window | `window` | `connect()`, resize handler, `renderHeatmap()`, `plotTimeSeries()` | resize listener installed in `init()` |
| Document | `document` | `initCanvas()`, status/coverage/direction/FPS writes, boot listener | `DOMContentLoaded` listener in module footer |

### Elements present in HTML but not touched by JS
- `#coverage-title`, `#move-title`, `#high-gamma-title` are style-only.
- `.separator`, `.status-bar`, `.layout`, `.sidebar`, etc. are layout-only.

## Pure vs Impure

### Pure (or effectively pure)
- `generateMagmaColormap()` — deterministic, no external state.

### Not pure / side-effectful
- `stopReconnectTimer()` — timer mutation.
- `resizeCanvasToContainer()` — DOM read/write + canvas state.
- `valueToColorIndex()` — no side effects, but reads mutable globals (`vMin`, `vMax`); not zero-change pure.
- `valueToColor()` — same dependency issue as above.
- `initCanvas()` — DOM lookup + context creation.
- `resizeTimeSeriesCanvas()` — DOM write + canvas transform.
- `renderHeatmap()` — canvas drawing + DOM writes + state mutation.
- `updateStatus()` — DOM writes + state mutation.
- `updateUpstreamStatus()` — state mutation + calls side-effectful helper.
- `updateInfoCards()` — DOM writes.
- `plotTimeSeries()` — state mutation + canvas drawing.
- `connect()` — network I/O + event handler wiring + state mutation.
- `init()` — bootstrap side effects.
- All inline callbacks — side-effectful by design.

## Canvas Rendering Details

### Main heatmap (`renderHeatmap()`)
- **Coordinate system:** CSS pixels after DPR transform; `canvasWidth`/`canvasHeight` are CSS-pixel dimensions.
- **DPR handling:** `resizeCanvasToContainer()` sets `canvas.width/height = CSS size * dpr`, sets style width/height to CSS size, then `ctx.setTransform(dpr,0,0,dpr,0,0)`.
- **Autoscaling:** scans 2D `heatmap` for `minVal`/`maxVal`, then smooths `vMin/vMax` with EMA-like update (`0.9 old + 0.1 new`).
- **Layout math:**
  - `size = min(canvasWidth, canvasHeight)`
  - `padding = max(4, floor(size*0.04))`
  - `plotSize = max(1, (size - 2*padding)*0.85)`
  - `cellSize = plotSize / cols`
- **Drawing ops, in order:**
  1. `fillRect(0,0,canvasWidth,canvasHeight)` background.
  2. Compute centroid offset so the incoming centroid lands at canvas center.
  3. For each cell: `fillRect(x,y,cellSize+0.5,cellSize+0.5)` with magma color.
  4. Grid overlay: `beginPath()`, repeated `moveTo/lineTo`, `stroke()`.
  5. Border: `strokeRect(...)` with green/blue style depending on alignment.
  6. Centroid highlight: `arc(centerX,centerY,...)`, `stroke()`.
  7. Guidance arrow: `moveTo(arrayCenterX,arrayCenterY) -> lineTo(centerX,centerY)`, then a triangular arrowhead (`moveTo/lineTo/closePath/fill`) at the line midpoint.
  8. Direction string derived from unit vector; writes `#direction-display`.
  9. FPS counter increments each render; once per second writes `#fps-counter`.
- **Coordinate conventions:** centroid is treated as `[y, x]` from backend; array center is `[rows/2, cols/2]`.

### High-gamma chart (`plotTimeSeries()`)
- **Coordinate system:** CSS pixels after DPR transform; width/height derived as `canvas.width/dpr` and `canvas.height/dpr`.
- **Buffering model:** `timeSeriesData` is a bounded FIFO. New sample appended with `push()`, oldest removed with `shift()` when over `maxTimeSeriesPoints`.
- **Layout math:** fixed padding `{top:20,right:20,bottom:30,left:50}`.
- **Axis/scale model:**
  - `minValue` / `maxValue` taken from current buffer
  - `valueRange = maxValue - minValue || 1`
  - `latestTime` and `oldestTime` determine `actualTimeRange`
  - `displayTimeRange = actualTimeRange / 0.70`, so newest sample lands at 70% of plot width, not the far right edge.
- **Drawing ops, in order:**
  1. `fillRect(0,0,width,height)` background.
  2. Left/bottom axes via `beginPath()`, `moveTo()`, `lineTo()`, `stroke()`.
  3. 5 horizontal grid lines.
  4. 5 y-axis labels (`fillText(value.toFixed(2), ...)`).
  5. Polyline trace with `moveTo/lineTo` using x from time and y from normalized value.
  6. `stroke()` the trace.
  7. Bottom axis label `Time (s)`.
  8. Rotated y-axis label `Mean Power (dB)` via `save()/translate()/rotate()/fillText()/restore()`.

## CSS Architecture

### Design tokens (`style.css:3-21`)
- Custom properties define the theme palette and status colors:
  - blues: `--blue-cyan`, `--blue-primary`, `--blue-navy`
  - backgrounds: `--bg-app`, `--bg-move`, `--bg-coverage`, `--bg-primary/secondary/tertiary`
  - text: `--text-primary`, `--text-secondary`
  - accents: `--accent-primary`, `--accent-glow`, `--heading-accent`
  - states: `--success`, `--warning`, `--error`
  - border: `--border-color`

### Layout model
- Global reset on `*`.
- `body` uses flex column layout and decorative radial gradients.
- `.container` is a viewport-constrained shell: `min-height/max-height: 95vh`, `max-width: 1920px`, `overflow:hidden`.
- `header` is a horizontal flex row with status bar.
- `.layout` is the main flex row, split into main canvas (`.panel-left`) and `.sidebar`.
- `.panel-left` is a fixed-aspect square sized from viewport math (`min(calc(100vw - 360px), calc(95vh - 140px))`).
- `.sidebar` is a vertical flex stack with `overflow-y:auto`, `min-width:280px`, `max-width:480px`.
- `.panel` and `.card` share the same visual shell (rounded, border, inset highlight, shadow).
- `.signal-monitor canvas` is a full-width chart with height clamped between 180 and 260px.

### Responsive breakpoints
- `@media (max-width: 980px)`:
  - `.layout` switches to column layout.
  - `.panel-left` shrinks to `min(calc(100vw - 48px), 60vh)`.
  - `.sidebar` removes max-width.
- `@media (max-width: 768px)`:
  - `.container` padding shrinks.
  - `header` stacks vertically.

### JS-coupled styles
- `.status-indicator.connected` / `.status-indicator.connecting` are toggled by `updateStatus()`.
- `.card-coverage` and `.card-move` backgrounds map directly to JS-driven state cards.
- `.card-coverage .card-body` is the exact element JS rewrites for coverage text and color.
- `#direction-display` size is critical to the directional cue that `renderHeatmap()` writes.
- `#coverage-display` size is defined here, but JS targets it via `.card-coverage .card-body`.
- `#neural-canvas` and `.signal-monitor canvas` backgrounds match the JS clear color (`#0a0a0f`), so resize clears blend cleanly.
- `.status-bar #time-display`, `#fps-counter`, `#channel-count` are visually emphasized for the continuously updated telemetry.
- `.canvas-container` exists but is unused in `index.html` / `app.js` (likely legacy CSS).

## Start Here
Open `frontend-old/app.js` first. It contains the full runtime state machine, the websocket message flow, and every render/DOM mutation that drives the migration.
