"""
Embedding service for generating vector embeddings from text chunks.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import DocumentChunk, EmbeddingResult
from .utils.timing import time_async_function, time_function

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom embedding error with context."""

    pass


class ConfigValidator:
    """Validate embedding configuration."""

    @staticmethod
    def validate_model_config(
        model_name: str, expected_dim: int
    ) -> tuple[bool, str | None]:
        """
        Validate model configuration before loading.

        Args:
            model_name: Name of the model to validate
            expected_dim: Expected embedding dimension

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic model name validation
            if not model_name or not isinstance(model_name, str):
                return False, "Invalid model name"

            if expected_dim <= 0:
                return False, "Invalid embedding dimension"

            return True, None

        except Exception as e:
            return False, f"Validation failed: {e}"


class EmbeddingService:
    """Service for generating embeddings from text using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding.model_name
        self.batch_size = settings.embedding.batch_size
        self.model: SentenceTransformer | None = None
        self._executor = ThreadPoolExecutor(max_workers=8)
        self.validator = ConfigValidator()

    def _handle_encoding_error(
        self, operation: str, error: Exception, context: dict | None = None
    ) -> None:
        """Centralized error handling for encoding operations."""
        context_str = f" (context: {context})" if context else ""
        error_msg = f"Failed to {operation}: {str(error)}{context_str}"
        logger.error(error_msg)
        raise EmbeddingError(f"{operation.title()} failed: {str(error)}") from error

    def load_model(self) -> None:
        """Load the sentence transformer model with validation."""
        if self.model is not None:
            logger.info(f"Model {self.model_name} already loaded")
            return

        # Validate configuration first
        is_valid, error_msg = self.validator.validate_model_config(
            self.model_name, settings.vector.vector_size
        )
        if not is_valid:
            raise EmbeddingError(f"Configuration invalid: {error_msg}")

        logger.info(f"Loading embedding model: {self.model_name}")
        start_time = time.time()

        try:
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time

            # Get model info
            embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"Successfully loaded {self.model_name}")
            logger.info(f"Embedding dimension: {embedding_dim}")
            logger.info(f"Load time: {load_time:.2f} seconds")

            # Verify expected dimension matches config
            if embedding_dim != settings.vector.vector_size:
                logger.warning(
                    f"Model dimension ({embedding_dim}) != "
                    f"configured dimension ({settings.vector.vector_size})"
                )
                logger.warning("Consider updating config.vector.vector_size")

        except Exception as e:
            self._handle_encoding_error(
                "load embedding model",
                e,
                {"model_name": self.model_name},
            )

    def _preprocess_texts_batch(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """
        Preprocess texts in single pass for optimal performance.

        Args:
            texts: List of input texts

        Returns:
            Tuple of (valid_texts, original_indices)
        """
        # Single-pass filtering and cleaning
        valid_data = [
            (i, text.strip())
            for i, text in enumerate(texts)
            if text.strip() and len(text.strip()) > 3
        ]

        if not valid_data:
            return [], []

        indices, valid_texts = zip(*valid_data, strict=False)
        return list(valid_texts), list(indices)

    def _convert_embeddings_batch(self, embeddings: np.ndarray) -> list[list[float]]:
        """
        Convert batch embeddings efficiently with single vectorized operation.

        Args:
            embeddings: Numpy array of embeddings

        Returns:
            List of embedding vectors as lists
        """
        # Single vectorized conversion instead of loop
        return embeddings.astype(np.float32).tolist()

    def generate_embedding(self, text: str) -> tuple[list[float], float]:
        """
        Generate embedding for a single text with optimized processing.

        Args:
            text: Input text to embed

        Returns:
            Tuple of (embedding_vector, generation_time)
        """
        if self.model is None:
            self.load_model()

        cleaned_text = text.strip()
        if not cleaned_text:
            logger.warning("Empty text provided for embedding")
            return [0.0] * settings.vector.vector_size, 0.0

        start_time = time.time()

        try:
            # Generate embedding with optimized parameters
            embedding = self.model.encode(
                cleaned_text, convert_to_numpy=True, normalize_embeddings=True
            )
            generation_time = time.time() - start_time

            # Direct conversion without unnecessary steps
            embedding_list = embedding.astype(np.float32).tolist()

            logger.debug(
                f"Generated embedding for text ({len(cleaned_text)} chars) "
                f"in {generation_time:.3f}s"
            )

            return embedding_list, generation_time

        except Exception as e:
            self._handle_encoding_error(
                "generate embedding",
                e,
                {"text_length": len(cleaned_text)},
            )

    def generate_embeddings_batch(
        self, texts: list[str]
    ) -> tuple[list[list[float]], float]:
        """
        Generate embeddings for multiple texts with optimized batch processing.

        Args:
            texts: List of input texts to embed

        Returns:
            Tuple of (list_of_embeddings, total_generation_time)
        """
        if self.model is None:
            self.load_model()

        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return [], 0.0

        # Optimized single-pass preprocessing
        valid_texts, valid_indices = self._preprocess_texts_batch(texts)

        if not valid_texts:
            logger.warning("No valid texts found in batch")
            empty_embedding = [0.0] * settings.vector.vector_size
            return [empty_embedding] * len(texts), 0.0

        logger.info(f"Generating embeddings for {len(valid_texts)} texts")
        start_time = time.time()

        try:
            # Process in batches with memory efficiency
            all_embeddings = []

            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i : i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=len(valid_texts) > 50,
                )

                # Optimized batch conversion - single vectorized operation
                batch_embeddings_list = self._convert_embeddings_batch(batch_embeddings)
                all_embeddings.extend(batch_embeddings_list)

            generation_time = time.time() - start_time

            avg_time = generation_time / len(all_embeddings)
            logger.info(
                f"Generated {len(all_embeddings)} embeddings "
                f"in {generation_time:.2f}s"
            )
            logger.info(f"Average time per embedding: {avg_time:.3f}s")

            return all_embeddings, generation_time

        except Exception as e:
            self._handle_encoding_error(
                "generate batch embeddings",
                e,
                {"batch_size": len(valid_texts)},
            )

    @time_function
    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple chunks with optimized batch processing.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of embedding results

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            return []

        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Use optimized batch size for better performance
            from .constants import DefaultValues

            batch_size = DefaultValues.EMBEDDING_BATCH_SIZE

            results = []
            total_chunks = len(chunks)

            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]

                # Generate embeddings for this batch
                embeddings, batch_time = self.generate_embeddings_batch(batch_texts)

                # Create results for this batch
                for chunk, embedding in zip(batch_chunks, embeddings, strict=True):
                    result = EmbeddingResult(
                        chunk_id=chunk.id,
                        embedding=embedding,
                        model_name=self.model_name,
                        generation_time=batch_time / len(batch_chunks),
                    )
                    results.append(result)

                logger.debug(
                    f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} "
                    f"({len(batch_chunks)} chunks)"
                )

            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return results

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}") from e

    @time_async_function
    async def embed_chunks_async(
        self, chunks: list[DocumentChunk]
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple chunks asynchronously.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of embedding results

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self.embed_chunks, chunks)
        except Exception as e:
            logger.error(f"Failed to generate embeddings async: {str(e)}")
            raise EmbeddingError(f"Async embedding generation failed: {str(e)}") from e

    def compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings with optimized operations.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays with optimized dtype
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)

            # Optimized cosine similarity computation
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

            if norms == 0:
                return 0.0

            similarity = float(dot_product / norms)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0

    def compute_similarity_batch(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities for multiple embedding pairs efficiently.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Array of similarity scores
        """
        try:
            # Vectorized batch similarity computation
            return np.dot(embeddings1, embeddings2.T)
        except Exception as e:
            logger.error(f"Failed to compute batch similarity: {str(e)}")
            return np.zeros((len(embeddings1), len(embeddings2)))

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False, "model_name": self.model_name}

        try:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            max_length = getattr(self.model.tokenizer, "model_max_length", "Unknown")

            return {
                "loaded": True,
                "model_name": self.model_name,
                "embedding_dimension": embedding_dim,
                "max_sequence_length": max_length,
                "batch_size": self.batch_size,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"loaded": False, "error": str(e)}

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("Embedding service cleaned up")
