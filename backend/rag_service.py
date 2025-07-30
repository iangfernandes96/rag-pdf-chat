"""
RAG (Retrieval-Augmented Generation) service that orchestrates
document processing, embedding, and similarity search.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import DefaultValues, SnippetSettings
from .document_ingestion import DocumentIngestionService
from .embedding_service import EmbeddingService
from .redis_cache import cache_service
from .utils.timing import time_async_function
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Custom RAG error with context."""

    pass


class RAGValidator:
    """Input validation for RAG operations."""

    @staticmethod
    def validate_search_params(query: str, limit: int, score_threshold: float) -> None:
        """
        Validate search parameters.

        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold

        Raises:
            RAGError: If parameters are invalid
        """
        if not query or not query.strip():
            raise RAGError("Query cannot be empty")

        if (
            not isinstance(limit, int)
            or not 1 <= limit <= DefaultValues.RAG_SEARCH_LIMIT
        ):
            raise RAGError(
                f"Limit must be between 1 and {DefaultValues.RAG_SEARCH_LIMIT}"
            )

        if (
            not isinstance(score_threshold, int | float)
            or not 0.0 <= score_threshold <= 1.0
        ):
            raise RAGError("Score threshold must be between 0.0 and 1.0")

    @staticmethod
    def validate_document_params(file_path: Path, original_filename: str) -> None:
        """
        Validate document processing parameters.

        Args:
            file_path: Path to the document file
            original_filename: Original filename

        Raises:
            RAGError: If parameters are invalid
        """
        if not file_path or not file_path.exists():
            raise RAGError("File path is invalid or file does not exist")

        if not original_filename or not original_filename.strip():
            raise RAGError("Original filename cannot be empty")


class SnippetOptimizer:
    """Optimized snippet generation with better context extraction."""

    # Pre-compile regex patterns for better performance
    SENTENCE_BOUNDARY = re.compile(r"[.!?]+\s+")
    WORD_BOUNDARY = re.compile(r"\b")

    @classmethod
    def create_snippet(
        cls, content: str, query: str, max_length: int = SnippetSettings.MAX_LENGTH
    ) -> str:
        """
        Create an optimized snippet with intelligent context extraction.

        Args:
            content: Full content text
            query: Search query
            max_length: Maximum snippet length

        Returns:
            Snippet with query context
        """
        if len(content) <= max_length:
            return content

        # Normalize query and content for better matching
        query_words = [word.lower().strip() for word in query.split() if word.strip()]
        content_lower = content.lower()

        # Find best position with sentence boundary awareness
        best_pos = cls._find_best_snippet_position(
            content, content_lower, query_words, max_length
        )

        # Extract snippet with proper boundaries
        snippet = cls._extract_snippet_with_boundaries(content, best_pos, max_length)

        return snippet

    @classmethod
    def _find_best_snippet_position(
        cls, content: str, content_lower: str, query_words: list[str], max_length: int
    ) -> int:
        """Find the best position to start the snippet."""
        best_pos = 0
        max_score = 0

        # Check positions at word boundaries for better context
        for i in range(0, max(1, len(content) - max_length), 50):
            snippet_text = content_lower[i : i + max_length]

            # Calculate relevance score
            score = sum(snippet_text.count(word) * len(word) for word in query_words)

            # Bonus for sentence boundaries
            if i == 0 or content[i - 1] in ".!?\n":
                score += 10

            if score > max_score:
                max_score = score
                best_pos = i

        return best_pos

    @classmethod
    def _extract_snippet_with_boundaries(
        cls, content: str, start_pos: int, max_length: int
    ) -> str:
        """Extract snippet with proper word boundaries."""
        snippet = content[start_pos : start_pos + max_length]

        # Clean up snippet boundaries
        if start_pos > 0:
            snippet = "..." + snippet
        if start_pos + max_length < len(content):
            snippet = snippet + "..."

        return snippet


class RAGService:
    """Main service that orchestrates the RAG pipeline with optimizations."""

    def __init__(self):
        """Initialize RAG service with optimized components."""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.document_ingestion = DocumentIngestionService()
        self.validator = RAGValidator()
        self.snippet_optimizer = SnippetOptimizer()

        # Initialize services
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize all services (load models, connect to databases).

        Returns:
            True if initialization successful
        """
        logger.info("Initializing RAG service...")

        try:
            # Load embedding model
            logger.info("Loading embedding model...")
            self.embedding_service.load_model()

            # Connect to vector store
            logger.info("Connecting to vector store...")
            await self.vector_store.connect()

            # Create collection if it doesn't exist
            logger.info("Setting up vector collection...")
            await self.vector_store.create_collection()

            logger.info("RAG service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            return False

    @time_async_function
    async def process_document(
        self, file_path: Path, original_filename: str
    ) -> dict[str, Any]:
        """
        Process a document through the complete RAG pipeline with optimizations.

        Args:
            file_path: Path to the document file
            original_filename: Original filename

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing document: {original_filename}")

        try:
            # Validate input parameters
            self.validator.validate_document_params(file_path, original_filename)

            # Step 1: Ingest document (PDF parsing and chunking)
            ingestion_result = self.document_ingestion.ingest_document(
                file_path, original_filename
            )

            if not ingestion_result.success:
                return {
                    "success": False,
                    "error": (
                        f"Document ingestion failed: "
                        f"{ingestion_result.error_message}"
                    ),
                    "stage": "ingestion",
                }

            document = ingestion_result.document
            chunks = ingestion_result.chunks

            if not document:
                return {
                    "success": False,
                    "error": "Document ingestion returned None",
                    "stage": "ingestion",
                }

            # Step 2: Generate embeddings for chunks (async optimization)
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embedding_results = await self.embedding_service.embed_chunks_async(chunks)

            if len(embedding_results) != len(chunks):
                return {
                    "success": False,
                    "error": ("Embedding generation failed: mismatch in chunk count"),
                    "stage": "embedding",
                }

            # Step 3: Store embeddings in vector database
            embeddings = [result.embedding for result in embedding_results]
            document_metadata = {
                "filename": document.filename,
                "original_filename": document.original_filename,
                "file_size": document.file_size,
                "page_count": document.page_count,
                "uploaded_at": document.uploaded_at.isoformat(),
            }

            logger.info("Storing embeddings in vector database")
            storage_success = await self.vector_store.store_embeddings(
                chunks, embeddings, document_metadata
            )

            if not storage_success:
                return {
                    "success": False,
                    "error": "Failed to store embeddings in vector database",
                    "stage": "storage",
                }

            # Step 4: Update chunks with embeddings
            for chunk, embedding_result in zip(chunks, embedding_results, strict=True):
                chunk.embedding = embedding_result.embedding

            logger.info(f"Successfully processed document: {original_filename}")

            return {
                "success": True,
                "document": document,
                "chunks": chunks,
                "embeddings_count": len(embeddings),
                "stats": self.document_ingestion.get_document_stats(document, chunks),
            }

        except RAGError:
            raise
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "stage": "unknown",
            }

    async def _get_query_embedding_cached(
        self, query: str
    ) -> tuple[list[float], float]:
        """
        Get query embedding with Redis caching.

        Args:
            query: Query text

        Returns:
            Tuple of (embedding, generation_time)
        """
        start_time = datetime.now(UTC)

        # Check Redis cache first
        cached_embedding = await cache_service.get("query_embedding", query)
        if cached_embedding:
            embed_time = (datetime.now(UTC) - start_time).total_seconds()
            logger.debug(f"Cache hit for query embedding: {query[:50]}...")
            return cached_embedding, embed_time

        # Generate embedding (synchronous call)
        embedding, embed_time = self.embedding_service.generate_embedding(query)

        # Cache in Redis
        await cache_service.set("query_embedding", embedding, 1800, query)  # 30 min TTL

        total_time = (datetime.now(UTC) - start_time).total_seconds()
        logger.debug(f"Generated and cached embedding for: {query[:50]}...")

        return embedding, total_time

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.1,
        document_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for relevant document chunks using semantic similarity with optimizations.

        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            document_filter: Optional document ID to filter by

        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching documents for query: '{query[:50]}...'")

        try:
            # Validate input parameters
            self.validator.validate_search_params(query, limit, score_threshold)

            # Generate embedding for query with caching
            query_embedding, embed_time = await self._get_query_embedding_cached(
                query.strip()
            )

            # Search similar chunks
            similar_chunks = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                document_filter=document_filter,
            )

            logger.info(f"Found {len(similar_chunks)} relevant chunks")

            # Enhance results with optimized snippet generation
            enhanced_results = await self._enhance_search_results(similar_chunks, query)

            return {
                "success": True,
                "query": query,
                "results": enhanced_results,
                "total_results": len(enhanced_results),
                "embed_time": embed_time,
                "search_params": {
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "document_filter": document_filter,
                },
                "cache_info": {
                    "cache_hit": embed_time < 0.001,
                },
            }

        except RAGError:
            raise
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "query": query,
            }

    async def _enhance_search_results(
        self, similar_chunks: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """
        Enhance search results with optimized snippet generation.

        Args:
            similar_chunks: List of similar chunk data
            query: Original search query

        Returns:
            List of enhanced results
        """
        enhanced_results = []

        for chunk_data in similar_chunks:
            # Use optimized snippet generation
            snippet = self.snippet_optimizer.create_snippet(
                chunk_data["content"], query
            )

            enhanced_result = {
                **chunk_data,
                "relevance_score": chunk_data["score"],
                "snippet": snippet,
                "query_match_quality": self._calculate_match_quality(
                    chunk_data["content"], query
                ),
            }
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _calculate_match_quality(self, content: str, query: str) -> float:
        """
        Calculate match quality score for better result ranking.

        Args:
            content: Chunk content
            query: Search query

        Returns:
            Match quality score (0.0 to 1.0)
        """
        query_words = {word.lower().strip() for word in query.split()}
        content_words = {word.lower() for word in content.split()}

        if not query_words:
            return 0.0

        # Calculate word overlap
        common_words = query_words.intersection(content_words)
        word_overlap = len(common_words) / len(query_words)

        # Bonus for exact phrase matches
        content_lower = content.lower()
        phrase_bonus = 0.2 if query.lower() in content_lower else 0.0

        return min(1.0, word_overlap + phrase_bonus)

    async def get_document_context(
        self, chunk_ids: list[str], context_size: int = 1
    ) -> list[str]:
        """
        Get expanded context for specific chunks.

        Args:
            chunk_ids: List of chunk IDs to get context for
            context_size: Number of surrounding chunks to include

        Returns:
            List of context strings
        """
        if not chunk_ids:
            return []

        # Process multiple chunks efficiently
        contexts = await asyncio.gather(
            *[self._get_single_chunk_context(chunk_id) for chunk_id in chunk_ids],
            return_exceptions=True,
        )

        # Handle any exceptions and return valid contexts
        result_contexts = []
        for i, context in enumerate(contexts):
            if isinstance(context, Exception):
                logger.warning(
                    f"Failed to get context for chunk {chunk_ids[i]}: {context}"
                )
                result_contexts.append("")
            else:
                result_contexts.append(context)

        return result_contexts

    async def _get_single_chunk_context(self, chunk_id: str) -> str:
        """Get context for a single chunk."""
        chunk_data = await self.vector_store.get_chunk_by_id(chunk_id)
        return chunk_data["content"] if chunk_data else ""

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the system.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if successful
        """
        if not document_id or not document_id.strip():
            raise RAGError("Document ID cannot be empty")

        logger.info(f"Deleting document: {document_id}")

        try:
            # Delete from vector store
            success = await self.vector_store.delete_document_chunks(document_id)

            if success:
                logger.info(f"Successfully deleted document: {document_id}")
            else:
                logger.warning(f"Failed to delete document: {document_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    async def get_system_status(self) -> dict[str, Any]:
        """
        Get status of all system components with performance metrics.

        Returns:
            Dictionary with unified health check format
        """
        try:
            # Get embedding model info
            embedding_info = self.embedding_service.get_model_info()

            # Get vector store info
            vector_info = await self.vector_store.get_collection_info()

            # Check health
            vector_healthy = await self.vector_store.health_check()

            # Determine overall health
            embedding_loaded = embedding_info.get("loaded", False)
            overall_healthy = embedding_loaded and vector_healthy

            return {
                "healthy": overall_healthy,
                "status": "healthy" if overall_healthy else "degraded",
                "error": (
                    None if overall_healthy else "One or more components unhealthy"
                ),
                "details": {
                    "embedding_service": embedding_info,
                    "vector_store": {
                        **vector_info,
                        "healthy": vector_healthy,
                        "url": self.vector_store.url,
                    },
                    "cache_status": {
                        "query_cache_size": 0,
                        "cache_ttl_hours": 0,
                        "max_cache_size": 0,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {"healthy": False, "status": "error", "error": str(e), "details": {}}

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clean up embedding service
            self.embedding_service.cleanup()

            logger.info("RAG service cleaned up")
        except Exception as e:
            logger.error(f"Error during RAG service cleanup: {str(e)}")
