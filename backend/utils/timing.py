"""
Timing utility for performance monitoring.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function with timing
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        logger.info(f"⏱️ {func.__name__} took {execution_time:.2f}s")
        return result

    return wrapper


def time_async_function(func: Callable) -> Callable:
    """
    Decorator to time async function execution.

    Args:
        func: Async function to be timed

    Returns:
        Wrapped async function with timing
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time

        logger.info(f"⏱️ {func.__name__} took {execution_time:.2f}s")
        return result

    return wrapper
