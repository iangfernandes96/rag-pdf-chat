#!/bin/bash
set -e

echo "🔍 Running code quality checks..."

# Check formatting with Black
echo "📝 Checking Black formatting..."
if uv run black backend/ frontend/ --check --diff --color; then
    echo "✅ Black formatting check passed"
else
    echo "❌ Black formatting check failed - run ./format.sh to fix"
    exit 1
fi

# Lint with Ruff
echo "🔧 Running Ruff linter..."
if uv run ruff check backend/ frontend/; then
    echo "✅ Ruff linting check passed"
else
    echo "❌ Ruff linting check failed - run ./format.sh to fix some issues"
    exit 1
fi

echo "✅ All code quality checks passed!" 