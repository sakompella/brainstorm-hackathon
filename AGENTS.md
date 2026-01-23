# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrainStorm 2026 Track 2: Build a real-time visualization tool to guide neurosurgeons in placing a brain-computer interface array. The app processes live neural data from a 1024-channel micro-ECoG array (32×32 grid, 500 Hz) and identifies velocity-tuned regions.

## Commands

```bash
# Setup
make install                              # Install deps + git hooks

# Download data (start with super_easy, develop with hard)
uv run python -m scripts.download super_easy
uv run python -m scripts.download hard

# Run (two terminals)
uv run brainstorm-stream --from-file data/hard/   # Terminal 1: stream data
uv run brainstorm-serve                            # Terminal 2: serve web app at :8000

# Development
make format          # ruff format
make lint            # ruff check --fix
make type-check      # mypy scripts/
make test            # pytest
make check-all       # all of the above
```

## Architecture

```
stream_data.py (ws://localhost:8765)  -->  Web App (browser)
         |                                      ^
         |                                      | serves static files
         v                                      |
    [Parquet files]                       serve.py (:8000)
```

Two approaches supported:
1. **Direct connection** - Browser connects to WebSocket, processes in JS (like example_app)
2. **Custom backend** - Python/Node middleware processes data before sending to frontend

### Key Components

- `scripts/stream_data.py` - WebSocket server streaming neural data at 500 Hz (DO NOT MODIFY protocol)
- `scripts/serve.py` - Static file server for web app
- `scripts/download.py` - Downloads datasets from HuggingFace
- `scripts/control_client.py` - Sends keyboard controls during live evaluation
- `example_app/` - Basic heatmap visualization (starting point, intentionally minimal)

### WebSocket Protocol

Init message:
```json
{"type": "init", "channels_coords": [[1,1]...], "grid_size": 32, "fs": 500.0, "batch_size": 10}
```

Sample batch (50 messages/sec):
```json
{"type": "sample_batch", "neural_data": [[...1024 values...]...], "start_time_s": 1.234, "sample_count": 10, "fs": 500.0}
```

## Data

- 1024 channels in 32×32 grid, 500 Hz sampling
- Four velocity-tuned regions: Vx+, Vx-, Vy+, Vy- (respond to cursor movement direction)
- Difficulty levels: super_easy → easy → medium → hard (use hard for final testing)
- Ground truth available in development only (`ground_truth.parquet`)

## Signal Processing Hints

High-gamma band (70-150 Hz) correlates well with motor intent. Typical pipeline:
1. Bandpass filter (e.g., 70-150 Hz)
2. Power/envelope extraction
3. Temporal smoothing (EMA or sliding window)
4. Reshape to 32×32 grid
5. Spatial smoothing (optional Gaussian blur)

## Design Constraints

- Must work in operating room environment
- Readable from 6 feet away (high contrast, large indicators)
- Identify coherent **areas** of tuned activity, not individual spikes
- Provide directional guidance for array movement
- Clear "found it" signal when positioned correctly

## Code Writing Guidelines

Do not write code before stating assumptions.
Do not claim correctness you haven't verified.
Do not handle only the happy path.
Prefer self-documenting code over comments. Comments are for when something is not obvious.
Be concise.
Ask questions when in doubt. Don't guess.
When possible, prefer strongly-typed code.
Lean on the compiler. When the compiler says something is wrong, fix it, don't hack around it.
**When something fails, STOP. Output your reasoning in full. Do not touch anything more until you understand the actual cause, have articulated it, and stated your expectations, and ask for further instructions.**
