#!/bin/bash

# Local Development Startup Script (Without Docker)
echo "🚀 Starting SMS RAG Application locally (without Docker)..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Please copy .env.example to .env and add your OPENAI_API_KEY"
    echo "Running: cp .env.example .env"
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "📝 Please edit .env and add your OpenAI API key before running again"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "⚠️  Warning: OPENAI_API_KEY not properly set in .env file"
    echo "Please edit .env and add your OpenAI API key"
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
echo "📁 Creating required directories..."
mkdir -p app/data/rsms/{index_store,documents}

# Set environment variables for local development
export ENVIRONMENT=local
export BASE_DIR="$(pwd)/app/data"
export STATIC_DIR="$(pwd)/static"

# Start the application
echo "🚀 Starting FastAPI application..."
echo ""
echo "🌐 Application will be available at:"
echo "   - Main App: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/healthz"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload