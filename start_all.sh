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
uv run python middleware.py > /tmp/middleware.log 2>&1 &
MIDDLEWARE_PID=$!
sleep 2

# Start frontend server
echo "Starting frontend server on :8000..."
uv run python scripts/serve.py > /tmp/serve.log 2>&1 &
SERVE_PID=$!
sleep 2

echo ""
echo "✅ All services started!"
echo ""
echo "📊 Data Stream:   http://localhost:8765 (PID: $STREAM_PID)"
echo "⚙️  Middleware:    http://localhost:8787 (PID: $MIDDLEWARE_PID)"
echo "🌐 Frontend:      http://localhost:8000 (PID: $SERVE_PID)"
echo ""
echo "Logs:"
echo "  tail -f /tmp/stream.log"
echo "  tail -f /tmp/middleware.log"
echo "  tail -f /tmp/serve.log"
echo ""
echo "To stop all: lsof -ti:8765,8787,8000 | xargs kill"
