#!/bin/bash

# RAG PDF Chat - Development Start Script

echo "🧠 Starting RAG PDF Chat Development Environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first: https://docs.astral.sh/uv/"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "🚀 Starting Ollama service..."
    brew services start ollama
    sleep 5
fi

# Install dependencies if not already installed
if [ ! -d ".venv" ]; then
    echo "📦 Installing dependencies..."
    uv sync
fi

# Check if Mistral model is available
echo "🤖 Checking for Mistral model..."
if ! ollama list | grep -q mistral; then
    echo "📥 Pulling Mistral model..."
    ollama pull mistral
fi

echo "✅ Development environment ready!"
echo ""
echo "🔧 Available commands:"
echo "  uv run uvicorn backend.main:app --reload     # Start backend server"
echo "  uv run streamlit run frontend/app.py         # Start frontend"
echo "  docker-compose up                            # Start full stack with Docker"
echo ""
echo "🌐 URLs:"
echo "  Backend API: http://localhost:8000"
echo "  Frontend: http://localhost:8501"
echo "  Ollama: http://localhost:11434" 