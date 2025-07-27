"""
Async vector store service using Qdrant for embedding storage and similarity search.
"""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Match,
    PointStruct,
    VectorParams,
)

from .config import settings
from .constants import DefaultValues, SystemMessages
from .models import DocumentChunk
from .utils.timing import time_async_function

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom vector store error with context."""

    pass


class VectorStoreValidator:
    """Input validation for vector store operations."""

    @staticmethod
    def validate_search_params(
        query_embedding: list[float], limit: int, score_threshold: float
    ) -> None:
        """
        Validate search parameters.

        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold

        Raises:
            VectorStoreError: If parameters are invalid
        """
        if not query_embedding:
            raise VectorStoreError("Query embedding cannot be empty")

        if not isinstance(query_embedding, list) or not all(
            isinstance(x, int | float) for x in query_embedding
        ):
            raise VectorStoreError("Query embedding must be a list of numbers")

        if len(query_embedding) != settings.vector.vector_size:
            raise VectorStoreError(
                f"Query embedding dimension {len(query_embedding)} "
                f"must match vector size {settings.vector.vector_size}"
            )

        if (
            not isinstance(limit, int)
            or not DefaultValues.MIN_SEARCH_LIMIT
            <= limit
            <= DefaultValues.VECTOR_SEARCH_LIMIT
        ):
            raise VectorStoreError(
                f"Limit must be between {DefaultValues.MIN_SEARCH_LIMIT} and {DefaultValues.VECTOR_SEARCH_LIMIT}"
            )

        if (
            not isinstance(score_threshold, int | float)
            or not 0.0 <= score_threshold <= 1.0
        ):
            raise VectorStoreError("Score threshold must be between 0.0 and 1.0")

    @staticmethod
    def validate_embeddings_data(
        chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """
        Validate embeddings data for storage.

        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors

        Raises:
            VectorStoreError: If data is invalid
        """
        if not chunks:
            raise VectorStoreError("Chunks list cannot be empty")

        if not embeddings:
            raise VectorStoreError("Embeddings list cannot be empty")

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        # Validate embedding dimensions
        expected_dim = settings.vector.vector_size
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or len(embedding) != expected_dim:
                raise VectorStoreError(
                    f"Embedding {i} has incorrect dimensions: "
                    f"expected {expected_dim}, got {len(embedding) if isinstance(embedding, list) else 'invalid'}"
                )

            if not all(isinstance(x, int | float) for x in embedding):
                raise VectorStoreError(f"Embedding {i} contains non-numeric values")

        # Validate chunk data
        for i, chunk in enumerate(chunks):
            if not chunk.id:
                raise VectorStoreError(f"Chunk {i} missing ID")
            if not chunk.document_id:
                raise VectorStoreError(f"Chunk {i} missing document_id")
            if not chunk.content.strip():
                raise VectorStoreError(f"Chunk {i} has empty content")


def retry_with_exponential_backoff(
    max_retries: int = DefaultValues.MAX_RETRIES,
    initial_delay: float = DefaultValues.INITIAL_RETRY_DELAY,
):
    """Decorator for retrying operations with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff

            logger.error(f"All {max_retries + 1} attempts failed")
            raise last_exception

        return wrapper

    return decorator


# Removed custom performance_monitor decorator - using time_async_function instead


class SearchResultCache:
    """LRU cache for search results with TTL."""

    def __init__(
        self,
        max_size: int = DefaultValues.CACHE_SIZE,
        ttl_hours: float = DefaultValues.CACHE_TTL_HOURS,
    ):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self._cache: dict[str, dict[str, Any]] = {}

    def _create_cache_key(
        self,
        query_embedding: list[float],
        limit: int,
        score_threshold: float,
        document_filter: str | None = None,
    ) -> str:
        """Create a hash key for caching search results."""
        # Create a hash of the query parameters
        query_str = f"{query_embedding}_{limit}_{score_threshold}_{document_filter}"
        return hashlib.md5(query_str.encode()).hexdigest()

    def get(
        self,
        query_embedding: list[float],
        limit: int,
        score_threshold: float,
        document_filter: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Get cached search results if available and not expired."""
        cache_key = self._create_cache_key(
            query_embedding, limit, score_threshold, document_filter
        )

        if cache_key in self._cache:
            cached_entry = self._cache[cache_key]
            if datetime.now(UTC) - cached_entry["timestamp"] < self.ttl:
                logger.debug(SystemMessages.CACHE_HIT)
                return cached_entry["results"]
            else:
                # Remove expired entry
                del self._cache[cache_key]

        return None

    def set(
        self,
        query_embedding: list[float],
        limit: int,
        score_threshold: float,
        document_filter: str | None,
        results: list[dict[str, Any]],
    ) -> None:
        """Cache search results with TTL."""
        cache_key = self._create_cache_key(
            query_embedding, limit, score_threshold, document_filter
        )

        # Implement simple LRU by removing oldest entries when at capacity
        if len(self._cache) >= self.max_size:
            self._cleanup_cache()

        self._cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now(UTC),
        }

        logger.debug("Cached search results for query")

    def _cleanup_cache(self) -> None:
        """Remove oldest entries based on cleanup ratio."""
        if len(self._cache) < self.max_size:
            return

        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1]["timestamp"])

        # Use performance threshold for cleanup ratio
        from .constants import PerformanceThresholds

        entries_to_remove = int(
            len(sorted_entries) * PerformanceThresholds.CACHE_CLEANUP_RATIO
        )
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._cache[key]

        logger.debug(f"Cleaned up {entries_to_remove} cache entries")

    def clear(self) -> int:
        """Clear all cached entries and return count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count


class VectorStore:
    """Async vector store service using Qdrant with optimizations."""

    def __init__(self, url: str | None = None, collection_name: str | None = None):
        self.url = url or settings.vector.qdrant_url
        self.collection_name = collection_name or settings.vector.collection_name
        self.client: AsyncQdrantClient | None = None
        self.validator = VectorStoreValidator()
        self.search_cache = SearchResultCache()
        # Use optimized batch size for better performance
        from .constants import DefaultValues

        self.batch_size = DefaultValues.VECTOR_BATCH_SIZE

        # Performance monitoring
        self._performance_metrics: dict[str, dict[str, Any]] = {}

    @retry_with_exponential_backoff()
    @time_async_function
    async def connect(self) -> None:
        """Connect to Qdrant database with retry logic."""
        if self.client is not None:
            logger.info("Already connected to Qdrant")
            return

        logger.info(f"Connecting to Qdrant at {self.url}")

        try:
            self.client = AsyncQdrantClient(url=self.url)

            # Test connection
            collections = await self.client.get_collections()
            logger.info(SystemMessages.CONNECTED)
            logger.info(f"Available collections: {len(collections.collections)}")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise VectorStoreError(f"Qdrant connection failed: {str(e)}") from e

    @time_async_function
    async def create_collection(self, overwrite: bool = False) -> bool:
        """
        Create a collection for storing document embeddings.

        Args:
            overwrite: Whether to overwrite existing collection

        Returns:
            True if collection was created, False if already exists
        """
        if self.client is None:
            await self.connect()

        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name for col in collections.collections
            )

            if collection_exists:
                if overwrite:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    await self.client.delete_collection(
                        collection_name=self.collection_name
                    )
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return False

            # Create collection
            logger.info(f"Creating collection: {self.collection_name}")
            logger.info(f"Vector size: {settings.vector.vector_size}")

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.vector.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(f"Successfully created collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise VectorStoreError(f"Collection creation failed: {str(e)}") from e

    def _create_batch_points(
        self,
        chunk_batch: list[tuple[DocumentChunk, list[float]]],
        document_metadata: dict[str, Any] | None = None,
    ) -> list[PointStruct]:
        """
        Create PointStruct objects for a batch with optimized memory usage.

        Args:
            chunk_batch: Batch of (chunk, embedding) tuples
            document_metadata: Optional document-level metadata

        Returns:
            List of PointStruct objects
        """
        points = []

        for chunk, embedding in chunk_batch:
            # Prepare metadata efficiently
            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }

            # Add document metadata if provided
            if document_metadata:
                payload["document_metadata"] = document_metadata

            # Create point
            point = PointStruct(id=chunk.id, vector=embedding, payload=payload)
            points.append(point)

        return points

    @time_async_function
    @retry_with_exponential_backoff(max_retries=2)
    async def _upload_batch(self, points: list[PointStruct]) -> None:
        """Upload a batch of points with retry logic."""
        if self.client is None:
            await self.connect()

        operation_info = await self.client.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )

        logger.debug(
            f"Uploaded batch of {len(points)} points, "
            f"status: {operation_info.status}"
        )

    @time_async_function
    async def store_embeddings(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        document_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store document chunks with their embeddings using optimized batch processing.

        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors
            document_metadata: Optional document-level metadata

        Returns:
            True if successful
        """
        if self.client is None:
            await self.connect()

        # Validate input data
        self.validator.validate_embeddings_data(chunks, embeddings)

        logger.info(f"Storing {len(chunks)} chunks with embeddings")

        try:
            # Process in batches for memory efficiency
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i : i + self.batch_size]
                batch_embeddings = embeddings[i : i + self.batch_size]
                batch_num = (i // self.batch_size) + 1

                logger.debug(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch_chunks)} items)"
                )

                # Create points for this batch
                chunk_embedding_pairs = list(
                    zip(batch_chunks, batch_embeddings, strict=True)
                )
                points = self._create_batch_points(
                    chunk_embedding_pairs, document_metadata
                )

                # Upload batch with retry
                await self._upload_batch(points)

            logger.info(
                f"Successfully stored {len(chunks)} embeddings in {total_batches} batches"
            )
            return True

        except VectorStoreError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise VectorStoreError(f"Embedding storage failed: {str(e)}") from e

    def _format_search_result(self, scored_point) -> dict[str, Any]:
        """
        Format a single search result with optimized memory usage.

        Args:
            scored_point: Qdrant scored point result

        Returns:
            Formatted result dictionary
        """
        # Use dict comprehension for efficient result creation
        return {
            "chunk_id": scored_point.id,
            "score": scored_point.score,
            "content": scored_point.payload.get("content", ""),
            "document_id": scored_point.payload.get("document_id", ""),
            "chunk_index": scored_point.payload.get("chunk_index", 0),
            "start_char": scored_point.payload.get("start_char", 0),
            "end_char": scored_point.payload.get("end_char", 0),
            "metadata": scored_point.payload.get("metadata", {}),
        }

    @time_async_function
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = DefaultValues.VECTOR_SEARCH_LIMIT,
        score_threshold: float = 0.0,
        document_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks using vector similarity with caching and optimizations.

        Args:
            query_embedding: Query vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            document_filter: Optional document ID to filter by

        Returns:
            List of similar chunks with scores and metadata
        """
        if self.client is None:
            await self.connect()

        # Validate input parameters
        self.validator.validate_search_params(query_embedding, limit, score_threshold)

        # Check cache first
        cached_results = self.search_cache.get(
            query_embedding, limit, score_threshold, document_filter
        )
        if cached_results is not None:
            logger.debug(f"Returning {len(cached_results)} cached search results")
            return cached_results

        logger.info(f"Searching for {limit} similar chunks")

        try:
            # Prepare filter if document_filter is provided
            search_filter = None
            if document_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=Match(value=document_filter)
                        )
                    ]
                )

            # Perform search
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Format results efficiently
            results = [
                self._format_search_result(scored_point)
                for scored_point in search_result
            ]

            # Cache results for future queries
            self.search_cache.set(
                query_embedding, limit, score_threshold, document_filter, results
            )

            logger.info(f"Found {len(results)} similar chunks")
            return results

        except VectorStoreError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            raise VectorStoreError(f"Similarity search failed: {str(e)}") from e

    @time_async_function
    async def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """
        Retrieve a specific chunk by ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk data if found, None otherwise
        """
        if self.client is None:
            await self.connect()

        if not chunk_id or not chunk_id.strip():
            raise VectorStoreError("Chunk ID cannot be empty")

        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                return None

            point = points[0]
            return {
                "chunk_id": point.id,
                "content": point.payload.get("content", ""),
                "document_id": point.payload.get("document_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "start_char": point.payload.get("start_char", 0),
                "end_char": point.payload.get("end_char", 0),
                "metadata": point.payload.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
            return None

    @time_async_function
    async def delete_document_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document.

        Args:
            document_id: ID of the document whose chunks to delete

        Returns:
            True if successful
        """
        if self.client is None:
            await self.connect()

        if not document_id or not document_id.strip():
            raise VectorStoreError("Document ID cannot be empty")

        try:
            # Delete points with matching document_id
            operation_info = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                wait=True,
            )

            logger.info(f"Deleted chunks for document {document_id}")
            logger.info(f"Operation status: {operation_info.status}")

            # Clear cache since document data changed
            cleared_entries = self.search_cache.clear()
            if cleared_entries > 0:
                logger.debug(f"Cleared {cleared_entries} cache entries after deletion")

            return True

        except Exception as e:
            logger.error(f"Failed to delete document chunks: {str(e)}")
            return False

    @time_async_function
    async def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection with performance metrics.

        Returns:
            Dictionary with collection information
        """
        if self.client is None:
            await self.connect()

        try:
            collection_info = await self.client.get_collection(
                collection_name=self.collection_name
            )

            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "cache_stats": {
                    "cached_searches": len(self.search_cache._cache),
                    "cache_max_size": self.search_cache.max_size,
                    "cache_ttl_hours": self.search_cache.ttl.total_seconds() / 3600,
                },
                "performance_metrics": self._performance_metrics,
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}

    @time_async_function
    async def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and accessible.

        Returns:
            True if healthy
        """
        try:
            if self.client is None:
                await self.connect()

            # Simple health check
            await self.client.get_collections()
            return True

        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False

    async def clear_cache(self) -> dict[str, Any]:
        """
        Clear search cache and return statistics.

        Returns:
            Dictionary with cache clearing results
        """
        cleared_entries = self.search_cache.clear()

        logger.info(f"Cleared search cache ({cleared_entries} entries)")

        return {
            "success": True,
            "entries_cleared": cleared_entries,
            "cache_size_after": len(self.search_cache._cache),
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for all operations.

        Returns:
            Dictionary with performance statistics
        """
        return {
            "operation_metrics": self._performance_metrics,
            "cache_stats": {
                "current_size": len(self.search_cache._cache),
                "max_size": self.search_cache.max_size,
                "ttl_hours": self.search_cache.ttl.total_seconds() / 3600,
            },
            "batch_size": self.batch_size,
        }

    async def close(self) -> None:
        """Close the client connection and clean up resources."""
        if self.client is not None:
            await self.client.close()
            self.client = None

        # Clear cache
        self.search_cache.clear()

        logger.info("Vector store connection closed and resources cleaned up")
