"""
Redis-based caching service for the RAG PDF Chat application.

Provides a unified caching interface for all services with Redis persistence.
"""

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as redis

from .config import settings

logger = logging.getLogger(__name__)


class RedisCacheError(Exception):
    """Custom Redis cache error."""

    pass


class RedisCacheService:
    """Redis-based caching service with unified interface."""

    def __init__(self, redis_url: str | None = None):
        """
        Initialize Redis cache service.

        Args:
            redis_url: Redis connection URL (defaults to Arq Redis URL)
        """
        self.redis_url = redis_url or settings.arq.redis_url
        self.client: redis.Redis | None = None
        self._prefix = "cache:"
        self._default_ttl = 3600  # 1 hour default

    async def initialize(self) -> bool:
        """
        Initialize Redis connection.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            await self.client.ping()
            logger.info("✅ Redis cache service connected")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis cache: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Clean up Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("✅ Redis cache service disconnected")

    def _generate_key(self, prefix: str, *args: Any) -> str:
        """
        Generate cache key from prefix and arguments.

        Args:
            prefix: Key prefix for organization
            *args: Arguments to hash into the key

        Returns:
            Generated cache key
        """
        # Create a hash of all arguments
        content = ":".join(str(arg).strip() for arg in args)
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return f"{self._prefix}{prefix}:{hash_value}"

    async def get(self, prefix: str, *args: Any) -> Any | None:
        """
        Get cached value.

        Args:
            prefix: Key prefix
            *args: Arguments to generate key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.client:
            return None

        try:
            key = self._generate_key(prefix, *args)
            value = await self.client.get(key)

            if value:
                logger.debug(f"Cache hit for key: {key}")
                return json.loads(value)

            logger.debug(f"Cache miss for key: {key}")
            return None

        except Exception as e:
            logger.warning(f"Cache get failed: {str(e)}")
            return None

    async def set(
        self, prefix: str, value: Any, ttl_seconds: int | None = None, *args: Any
    ) -> bool:
        """
        Set cached value.

        Args:
            prefix: Key prefix
            value: Value to cache
            ttl_seconds: Time to live in seconds (defaults to 1 hour)
            *args: Arguments to generate key

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            key = self._generate_key(prefix, *args)
            ttl = ttl_seconds or self._default_ttl

            # Serialize value to JSON
            serialized_value = json.dumps(value, default=str)

            await self.client.setex(key, ttl, serialized_value)
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Cache set failed: {str(e)}")
            return False

    async def delete(self, prefix: str, *args: Any) -> bool:
        """
        Delete cached value.

        Args:
            prefix: Key prefix
            *args: Arguments to generate key

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            key = self._generate_key(prefix, *args)
            result = await self.client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return result > 0

        except Exception as e:
            logger.warning(f"Cache delete failed: {str(e)}")
            return False

    async def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with a specific prefix.

        Args:
            prefix: Prefix to match

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        try:
            pattern = f"{self._prefix}{prefix}:*"
            keys = await self.client.keys(pattern)

            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys with prefix: {prefix}")
                return deleted

            return 0

        except Exception as e:
            logger.warning(f"Cache clear prefix failed: {str(e)}")
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.client:
            return {"error": "Not connected"}

        try:
            info = await self.client.info("memory")
            keys = await self.client.keys(f"{self._prefix}*")

            return {
                "connected": True,
                "total_keys": len(keys),
                "memory_usage": info.get("used_memory_human", "Unknown"),
                "cache_prefixes": list({k.split(":")[1] for k in keys if ":" in k}),
            }

        except Exception as e:
            logger.warning(f"Failed to get cache stats: {str(e)}")
            return {"error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """
        Health check for the cache service.

        Returns:
            Health status dictionary
        """
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "Not connected to Redis"}

            await self.client.ping()

            return {"status": "healthy", "redis_url": self.redis_url, "connected": True}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "redis_url": self.redis_url}


# Global cache service instance
cache_service = RedisCacheService()
