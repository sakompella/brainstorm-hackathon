# WebSocket Connectivity & Data Flow Test Report
**Date:** 2026-06-26  
**Tool:** browser-use CLI (`--session ws-test`)  
**Target:** http://localhost:8000 (Neural Data Viewer)

---

## Test Setup Notes

The `browser-use` CLI initially failed to locate a Chromium browser because it looks for `Chromium.app/Contents/MacOS/Chromium` under `~/Library/Caches/ms-playwright/`, but `uvx playwright install chromium` installs it as `"Google Chrome for Testing.app"`. Two symlinks were created to resolve this:
```
~/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/Chromium.app
  -> "Google Chrome for Testing.app"
~/Library/Caches/.../Chromium.app/Contents/MacOS/Chromium
  -> "Google Chrome for Testing"
```
After this fix, all `browser-use` commands succeeded.

---

## Results

### 1. WebSocket Connection Status
| Check | Value |
|-------|-------|
| `.status-indicator` class | `status-indicator connecting` |
| Status bar text | `Waiting for stream…` |

**Finding:** The status indicator shows **`connecting`**, not `connected`. The status text reads "Waiting for stream…", which suggests the WebSocket handshake has not fully completed or the frontend is still negotiating. Data **is flowing** (see below), which implies the indicator may be set to `connecting` as a persistent/intermediate state rather than a true disconnected state — or the CSS class for "connected" is named differently than expected.

> **Possible issue:** The `.status-indicator` class contains `connecting` rather than `connected`. This warrants investigation — either the WebSocket is genuinely in a connecting state (but data flows anyway), or the class name used for the "connected" state in the UI is different from `connected`.

---

### 2. Time Incrementing (Features Flowing)

| Read | Time Value |
|------|-----------|
| First read | `t = 75.74s` |
| Second read (2s later) | `t = 77.82s` |
| Final read | `t = 90.78s` |

**Finding:** ✅ Time is clearly incrementing (~2s between reads, consistent with a ~2s wall-clock delay). **Features are flowing** — the data pipeline from streamer → backend → WebSocket → frontend is working correctly.

---

### 3. Canvas Dimensions

| Canvas | Width | Height |
|--------|-------|--------|
| Main heatmap canvas | 1006 px | 1006 px |
| Colorbar / secondary canvas | 446 px | 17886 px |

**Finding:** ✅ Both canvases have **non-zero dimensions**. The main 32×32 heatmap renders at 1006×1006 px (proper square). The tall secondary canvas (17886 px height) is likely a scrollable colorbar or channel strip — its height seems unusually large but may be intentional.

---

### 4. Coverage & Direction

| Metric | Value |
|--------|-------|
| Coverage | **75.3%** |
| Direction | **Left** |

**Finding:** ✅ The coverage card is displaying a meaningful percentage (75.3%), and the direction display shows a clear directional label ("Left"). Both actionable guidance metrics are rendering and updating.

---

### 5. Error Elements
- No `.error` DOM elements found.
- No unhandled JS exceptions detected via `window.wsState`.
- Screenshot saved to `/tmp/neural-viewer-ws-test.png` (190 KB).

---

## Summary

| Check | Result |
|-------|--------|
| WebSocket `connected` class | ⚠️ Shows `connecting` — not `connected` |
| Status bar text | ⚠️ "Waiting for stream…" (but data flows) |
| Time incrementing | ✅ Yes — t increases by ~2s per 2s wall-clock |
| Features flowing | ✅ Yes — pipeline is active |
| Main canvas sized properly | ✅ 1006×1006 px |
| Colorbar canvas sized properly | ✅ 446×17886 px (may be oversized) |
| Coverage % shown | ✅ 75.3% |
| Direction shown | ✅ "Left" |
| JS errors | ✅ None detected |

---

## Issues to Investigate

1. **`status-indicator` class is `connecting` not `connected`** — The CSS class logic may have a bug where the state never transitions to `connected` even when the WebSocket is functioning and data is streaming. The status bar text still reads "Waiting for stream…" despite data actively flowing. This is a UI/state-machine inconsistency.

2. **Colorbar canvas height of 17886 px** — This is very tall and may be a rendering bug where the height is calculated incorrectly (e.g., multiplying by the wrong factor). Worth verifying whether this is intentional.

3. **`browser-use` Chromium path bug** — The `browser-use` CLI's `_find_installed_browser_path()` only looks for `Chromium.app/Contents/MacOS/Chromium` but `uvx playwright install chromium` installs `"Google Chrome for Testing.app"`. A permanent fix would be to add the "Google Chrome for Testing" path pattern to `browser-use`, or configure `PLAYWRIGHT_BROWSERS_PATH` appropriately. The two symlinks created above are a workaround.
