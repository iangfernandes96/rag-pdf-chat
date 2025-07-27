"""
Timing utilities for performance monitoring and optimization.
"""

import functools
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def time_function(func: Callable) -> Callable:
    """
    Decorator to time synchronous functions and log performance metrics.

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_datetime = datetime.now(UTC)

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            # Log performance metrics
            logger.info(
                f"⏱️ {func.__name__} completed in {execution_time:.3f}s "
                f"({execution_time*1000:.1f}ms)"
            )

            # Add timing to result if it's a dict
            if isinstance(result, dict):
                result["processing_time"] = execution_time
                result["started_at"] = start_datetime.isoformat()
                result["completed_at"] = datetime.now(UTC).isoformat()

            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(
                f"❌ {func.__name__} failed after {execution_time:.3f}s: {str(e)}"
            )
            raise

    return wrapper


def time_async_function(func: Callable) -> Callable:
    """
    Decorator to time asynchronous functions and log performance metrics.

    Args:
        func: Async function to time

    Returns:
        Wrapped async function with timing
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_datetime = datetime.now(UTC)

        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            # Log performance metrics
            logger.info(
                f"⏱️ {func.__name__} completed in {execution_time:.3f}s "
                f"({execution_time*1000:.1f}ms)"
            )

            # Add timing to result if it's a dict
            if isinstance(result, dict):
                result["processing_time"] = execution_time
                result["started_at"] = start_datetime.isoformat()
                result["completed_at"] = datetime.now(UTC).isoformat()

            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(
                f"❌ {func.__name__} failed after {execution_time:.3f}s: {str(e)}"
            )
            raise

    return wrapper


class PerformanceMonitor:
    """Performance monitoring utility for tracking operation timing."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_datetime = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_datetime = datetime.now(UTC)
        logger.debug(f"🚀 Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            end_time = time.time()
            execution_time = end_time - self.start_time

            if exc_type is None:
                logger.info(
                    f"✅ {self.operation_name} completed in {execution_time:.3f}s "
                    f"({execution_time*1000:.1f}ms)"
                )
            else:
                logger.error(
                    f"❌ {self.operation_name} failed after {execution_time:.3f}s: "
                    f"{str(exc_val)}"
                )

    def get_elapsed_time(self) -> float:
        """Get elapsed time so far."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
