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

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding.model_name
        self.batch_size = settings.embedding.batch_size
        self.model: SentenceTransformer | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is not None:
            logger.info(f"Model {self.model_name} already loaded")
            return

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
                    f"Model dimension ({embedding_dim}) != configured dimension ({settings.vector.vector_size})"
                )
                logger.warning("Consider updating config.vector.vector_size")

        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Embedding model loading failed: {str(e)}") from e

    def generate_embedding(self, text: str) -> tuple[list[float], float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Tuple of (embedding_vector, generation_time)
        """
        if self.model is None:
            self.load_model()

        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * settings.vector.vector_size, 0.0

        start_time = time.time()

        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            generation_time = time.time() - start_time

            # Convert to list and ensure correct type
            embedding_list = embedding.astype(np.float32).tolist()

            logger.debug(
                f"Generated embedding for text ({len(text)} chars) in {generation_time:.3f}s"
            )

            return embedding_list, generation_time

        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e

    def generate_embeddings_batch(
        self, texts: list[str]
    ) -> tuple[list[list[float]], float]:
        """
        Generate embeddings for multiple texts in batch.

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

        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text.strip()]
        if not valid_texts:
            logger.warning("No valid texts found in batch")
            return [[0.0] * settings.vector.vector_size] * len(texts), 0.0

        logger.info(f"Generating embeddings for {len(valid_texts)} texts")
        start_time = time.time()

        try:
            # Process in batches if needed
            all_embeddings = []

            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i : i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=len(valid_texts) > 10,
                )

                # Convert to list format
                batch_embeddings_list = [
                    emb.astype(np.float32).tolist() for emb in batch_embeddings
                ]
                all_embeddings.extend(batch_embeddings_list)

            generation_time = time.time() - start_time

            logger.info(
                f"Generated {len(all_embeddings)} embeddings in {generation_time:.2f}s"
            )
            logger.info(
                f"Average time per embedding: {generation_time/len(all_embeddings):.3f}s"
            )

            return all_embeddings, generation_time

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise RuntimeError(f"Batch embedding generation failed: {str(e)}") from e

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingResult]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        logger.info(f"Embedding {len(chunks)} document chunks")

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batch
        embeddings, total_time = self.generate_embeddings_batch(texts)

        # Create EmbeddingResult objects
        results = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            result = EmbeddingResult(
                chunk_id=chunk.id,
                embedding=embedding,
                model_name=self.model_name,
                generation_time=total_time / len(chunks),  # Average time per chunk
            )
            results.append(result)

        logger.info(f"Successfully embedded {len(results)} chunks")
        return results

    async def embed_chunks_async(
        self, chunks: list[DocumentChunk]
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for document chunks asynchronously.

        Args:
            chunks: List of document chunks to embed

        Returns:
            List of EmbeddingResult objects
        """
        loop = asyncio.get_event_loop()

        # Run embedding in thread pool to avoid blocking
        return await loop.run_in_executor(self._executor, self.embed_chunks, chunks)

    def compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False, "model_name": self.model_name}

        return {
            "loaded": True,
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": getattr(
                self.model.tokenizer, "model_max_length", "Unknown"
            ),
            "batch_size": self.batch_size,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("Embedding service cleaned up")
