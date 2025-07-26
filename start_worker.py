#!/usr/bin/env python3
"""
Script to start Celery worker for RAG PDF Chat background processing.
"""

import sys
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.celery_app import celery_app

if __name__ == "__main__":
    # Start Celery worker
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--concurrency=2",
        "--queues=document_processing,maintenance",
        "--hostname=worker@%h"
    ]) 