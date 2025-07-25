# Use Python 3.13 slim image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-cache --no-dev

# Copy application code
COPY . .

# Install the local package in the uv environment
RUN uv pip install --no-deps -e .

# Create activation script to avoid uv run overhead
RUN echo '#!/bin/bash\nexport PATH="/app/.venv/bin:$PATH"\nexec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# No local data directories needed - using external databases

# Expose ports
EXPOSE 8000 8501

# Set entrypoint and default command (can be overridden in docker-compose)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"] 