#!/bin/bash
# Start all services for BrainStorm hackathon

# Kill existing processes
echo "Cleaning up existing processes..."
lsof -ti:8765 2>/dev/null | xargs kill -9 2>/dev/null
lsof -ti:8787 2>/dev/null | xargs kill -9 2>/dev/null
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# Start data streamer
echo "Starting data streamer on :8765..."
uv run python scripts/stream_data.py --from-file data/easy/ > /tmp/stream.log 2>&1 &
STREAM_PID=$!
sleep 3

# Start middleware
echo "Starting middleware on :8787..."
uv run python run_middleware.py > /tmp/middleware.log 2>&1 &
MIDDLEWARE_PID=$!
sleep 2

# Start unified backend (connects to stream + serves frontend)
echo "Starting unified backend on :8000..."
uv run brainstorm-backend --upstream-url ws://localhost:8765 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

echo ""
echo "✅ All services started!"
echo ""
echo "📊 Data Stream:   ws://localhost:8765 (PID: $STREAM_PID)"
echo "⚙️  Middleware:    ws://localhost:8787 (PID: $MIDDLEWARE_PID)"
echo "🌐 Backend+UI:    http://localhost:8000 (PID: $BACKEND_PID)"
echo ""
echo "Logs:"
echo "  tail -f /tmp/stream.log"
echo "  tail -f /tmp/middleware.log"
echo "  tail -f /tmp/backend.log"
echo ""
echo "To stop all: lsof -ti:8765,8787,8000 | xargs kill"