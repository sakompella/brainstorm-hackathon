#!/bin/bash
# Start BrainStorm hackathon services
# Architecture: stream_data.py (:8765) --> backend.py (:8000) --> Browser

set -e

DATA_DIR="${1:-data/easy}"

# Kill existing processes
echo "Cleaning up..."
lsof -ti:8765,8000 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Start data streamer
echo "Starting streamer on :8765 (data: $DATA_DIR)..."
uv run brainstorm-stream --from-file "$DATA_DIR/" > /tmp/stream.log 2>&1 &
STREAM_PID=$!
sleep 2

# Start backend (signal processing + static server)
echo "Starting backend on :8000..."
uv run brainstorm-backend --upstream-url ws://localhost:8765 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

echo ""
echo "Ready!"
echo ""
echo "  Streamer:  ws://localhost:8765  (PID: $STREAM_PID)"
echo "  Backend:   http://localhost:8000 (PID: $BACKEND_PID)"
echo ""
echo "Open http://localhost:8000 in your browser"
echo ""
echo "Logs:"
echo "  tail -f /tmp/stream.log"
echo "  tail -f /tmp/backend.log"
echo ""
echo "Stop: lsof -ti:8765,8000 | xargs kill"
