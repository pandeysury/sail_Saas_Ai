#!/bin/bash
# start-server.sh - Automatic server startup script

cd /home/ubuntu/Desktop/Rag_System_Project/sail_Saas_Ai

# Kill any existing uvicorn processes
echo "🔄 Stopping existing servers..."
pkill -f "uvicorn.*app.main:app" 2>/dev/null || true
sleep 2

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Start the server
echo "🚀 Starting server on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload