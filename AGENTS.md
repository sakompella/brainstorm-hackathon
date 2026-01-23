# CLAUDE.md (Updated Jan 23, 2026)

Guidance for Claude Code (claude.ai/code) when working inside this repository. Keep this file in sync with the current toolchain and architecture described below.

## Track Overview

BrainStorm 2026 Track 2: build a real-time visualization tool to guide neurosurgeons in positioning a 32×32 (1024 channel) micro-ECoG array. The processing pipeline now supports a FastAPI-based data streamer/server plus an optional Python middleware phase for feature extraction.

## Daily Workflow

```bash
# Install deps + git hooks
make install

# Download datasets (start with super_easy, iterate on hard)
uv run python -m scripts.download super_easy
uv run python -m scripts.download hard

# Run data streamer (FastAPI WebSocket @ ws://localhost:8765)
uv run brainstorm-stream --from-file data/hard/

# Serve frontend (FastAPI static server @ http://localhost:8000)
uv run brainstorm-serve

# Optional middleware (feature WebSocket relay @ ws://localhost:8787)
uv run python middleware.py

# Validation helpers
make format       # ruff format
make lint         # ruff check --fix
make type-check   # mypy scripts/
make test         # pytest
make check-all    # run format + lint + type + tests
```

## Architecture Snapshot (Jan 2026)

```
[Parquet files] --uv--> scripts/stream_data.py (FastAPI WebSocket @8765)
        |                               |
        | optional                      v
        +--> middleware.py (feature WebSocket @8787)
                                        |
                                        v
                               Web App (example_app/ or custom)
                               served via scripts/serve.py (@8000)
```

- `scripts/stream_data.py` — FastAPI + uvicorn; streams data at 500 Hz (JSON batches).
- `middleware.py` — Optional bridge: consumes raw stream, emits activity features (`type="features"`). Keep protocol stable if extending.
- `scripts/serve.py` — FastAPI static server wrapping `example_app/` (or your replacement app).
- `scripts/download.py` — HuggingFace helper for datasets (`track2_data.parquet`, `metadata.json`, `ground_truth.parquet`).
- `scripts/control_client.py` — Sends keyboard commands during live evaluation.
- `example_app/` — Minimal reference UI (magma heatmap). Replace or extend for your solution.
- `docs/` — Authoritative specs (overview, data_stream protocol, submission rules, persona, etc.). Always check docs before changing behavior.

## WebSocket Protocols

Raw stream (`ws://localhost:8765`):
```json
{"type":"init","channels_coords":[[1,1],...],"grid_size":32,"fs":500.0,"batch_size":10}
{"type":"sample_batch","neural_data":[[...1024 floats...]...],"start_time_s":1.234,"sample_count":10,"fs":500.0}
```

Middleware features (`ws://localhost:8787`):
```json
{"type":"features","activity":[1024 floats],"t":12.34,"presence":0.42,"confidence":1.0,"total_samples":12345}
```
Maintain backward compatibility—evaluation servers expect the raw protocol.

## Data + Ground Truth

- Channels: 1024, grid ordered row-major.
- Sampling: 500 Hz batches of `batch_size` samples (default 10 → 50 msgs/sec).
- Difficulty tiers: `super_easy`, `easy`, `medium`, `hard` (develop/test on `hard`).
- `ground_truth.parquet` and `metadata.json` only for local iteration; unavailable during live eval.

## Signal Processing Reference

Typical progression (see `docs/data.md` + `docs/getting_started.md`):
1. Bandpass 70–150 Hz (high-gamma) or equivalent feature extraction.
2. Instantaneous power → log/EMA smoothing (see `middleware.py:ActivityEMA`).
3. Temporal aggregation (EMA / sliding window) for stability.
4. Reshape vector → 32×32 grid; optionally apply spatial smoothing or clustering.
5. Identify directional tuning regions (Vx+/Vx−/Vy+/Vy−) and surface guidance cues.

## Design / UX Constraints

- Operating room usage: high contrast, legible from ~6 ft.
- Focus on coherent **areas** rather than single-channel spikes.
- Provide actionable guidance: “move array ↘︎” or clear “locked-on” indicator.
- Distinguish confidence/presence metrics visually; avoid ambiguous colors.

## Code Writing Expectations

- State explicit assumptions before substantive changes.
- Never assume the happy path; handle file/network errors and reconnect logic (see `middleware.py`).
- Prefer self-documenting, typed code (Python typing, TypeScript if used). Avoid unnecessary comments.
- When FastAPI / uvicorn configs change, verify both CLI entrypoints (`brainstorm-stream`, `brainstorm-serve`).
- Do not modify streaming protocols without strong justification; coordinate updates across streamer, middleware, and frontend.
- If anything fails (commands, tests, servers), stop, explain the failure, and request guidance before continuing.

## Validation + Tooling Notes

- Ruff handles formatting + lint (`make format`, `make lint`).
- Type checking limited to `scripts/` (run `make type-check`).
- Tests via `pytest` (extend as needed for new backend/frontend logic).
- Use `uv run <command>` to ensure virtualenv consistency.
- For frontend work, keep `example_app/` build-less; if introducing bundlers, document steps in `docs/getting_started.md` and update this file.

Keep AGENTS.md updated whenever workflows, commands, or architecture change.
Thanks! 🚀
