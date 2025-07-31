#!/bin/bash
set -e

echo "🚀 Running comprehensive code quality checks..."

# Install dev dependencies if needed
echo "📦 Ensuring dev dependencies are installed..."
uv sync --dev

# Run formatting
echo ""
echo "🔧 Step 1: Formatting code..."
./format.sh

# Run linting
echo ""
echo "🔍 Step 2: Running linting checks..."
uv run ruff check backend/ frontend/ --fix

# Run tests if they exist
echo ""
echo "🧪 Step 3: Running tests..."
if [ -d "tests" ] || find . -name "*test*.py" -not -path "./.venv/*" | head -1 | grep -q .; then
    uv run pytest -v
else
    echo "ℹ️  No tests found, skipping test execution"
fi

echo ""
echo "✅ All checks completed successfully!"
echo "🎉 Code is ready for commit!" 