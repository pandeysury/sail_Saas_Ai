#!/bin/bash
# start-dashboard.sh - Robust server startup

cd /home/ubuntu/Desktop/Rag_System_Project/sail_Saas_Ai

echo "🔍 Checking for existing processes..."
EXISTING_PIDS=$(pgrep -f "uvicorn.*app.main:app" || true)

if [ ! -z "$EXISTING_PIDS" ]; then
    echo "🛑 Killing existing processes: $EXISTING_PIDS"
    kill -9 $EXISTING_PIDS 2>/dev/null || true
    sleep 3
fi

# Check if port is still in use
PORT_CHECK=$(netstat -tuln 2>/dev/null | grep ":8000 " || true)
if [ ! -z "$PORT_CHECK" ]; then
    echo "⚠️  Port 8000 still in use, trying to free it..."
    sudo fuser -k 8000/tcp 2>/dev/null || true
    sleep 2
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing missing dependencies..."
pip install python-multipart requests > /dev/null 2>&1

echo "🧪 Testing app import..."
python3 -c "from app.main import app; print('✅ App imports OK')" || {
    echo "❌ App import failed"
    exit 1
}

echo "🚀 Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info