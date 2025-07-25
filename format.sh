#!/bin/bash
set -e

echo "🔧 Formatting code with Black and Ruff..."

# Format code with Black
echo "📝 Running Black formatter..."
uv run black backend/ frontend/ --diff --color

# Sort imports with Ruff
echo "📦 Sorting imports with Ruff..."
uv run ruff check backend/ frontend/ --select I --fix

# Format code with Black (actual formatting)
echo "✨ Applying Black formatting..."
uv run black backend/ frontend/

echo "✅ Code formatting complete!" 