#!/usr/bin/env python3
"""
Script to start Arq worker for RAG PDF Chat background processing.
"""

import sys
import signal
import logging
from pathlib import Path

# Configure logging for Docker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.arq_worker import WorkerSettings

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        from arq import run_worker
        
        logging.info("Starting Arq worker...")
        # Run the Arq worker
        run_worker(WorkerSettings)
    except KeyboardInterrupt:
        logging.info("Worker stopped by user")
    except Exception as e:
        logging.error(f"Worker failed to start: {e}")
        sys.exit(1) 