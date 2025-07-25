"""
Text chunking functionality for document processing.
"""

import logging
import re

from .config import settings
from .models import ChunkingStrategy, DocumentChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking with configurable strategies."""

    def __init__(self, strategy: ChunkingStrategy | None = None):
        self.strategy = strategy or ChunkingStrategy(
            chunk_size=settings.document.chunk_size,
            overlap=settings.document.chunk_overlap,
            preserve_sentences=True,
            min_chunk_size=50,
        )

    def chunk_text(
        self, document_id: str, text: str, metadata: dict | None = None
    ) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            document_id: ID of the source document
            text: Text content to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            logger.warning(f"Empty text provided for document {document_id}")
            return []

        logger.info(
            f"Chunking text for document {document_id}, length: {len(text)} characters"
        )

        if self.strategy.preserve_sentences:
            chunks = self._chunk_by_sentences(text)
        else:
            chunks = self._chunk_by_characters(text)

        # Create DocumentChunk objects
        document_chunks = []
        for i, (content, start_char, end_char) in enumerate(chunks):
            if len(content.strip()) >= self.strategy.min_chunk_size:
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=content.strip(),
                    start_char=start_char,
                    end_char=end_char,
                    token_count=self._estimate_token_count(content),
                    metadata=metadata or {},
                )
                document_chunks.append(chunk)

        logger.info(f"Created {len(document_chunks)} chunks for document {document_id}")
        return document_chunks

    def _chunk_by_sentences(self, text: str) -> list[tuple]:
        """
        Chunk text while preserving sentence boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of (chunk_content, start_char, end_char) tuples
        """
        # Split into sentences using regex
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = ""
        current_start = 0
        current_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find the actual position of this sentence in the original text
            sentence_start = text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)

            # Check if adding this sentence would exceed chunk size
            potential_chunk = (
                current_chunk + " " + sentence if current_chunk else sentence
            )

            if len(potential_chunk) > self.strategy.chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append((current_chunk, current_start, sentence_start - 1))

                # Start new chunk with overlap
                overlap_start = max(
                    current_start, sentence_start - self.strategy.overlap
                )
                current_chunk = sentence
                current_start = overlap_start
            else:
                # Add sentence to current chunk
                if not current_chunk:
                    current_start = sentence_start
                current_chunk = potential_chunk

            current_pos = sentence_end

        # Add the last chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, len(text)))

        return chunks

    def _chunk_by_characters(self, text: str) -> list[tuple]:
        """
        Chunk text by character count with overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of (chunk_content, start_char, end_char) tuples
        """
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.strategy.chunk_size, len(text))

            # Try to break at word boundary if possible
            if end < len(text):
                # Look backwards for a space to break on
                word_break = text.rfind(" ", start, end)
                if word_break > start + self.strategy.chunk_size // 2:
                    end = word_break

            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append((chunk_content, start, end))

            # Move start position with overlap
            start = max(start + 1, end - self.strategy.overlap)

        return chunks

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)

    def get_chunk_context(
        self,
        chunks: list[DocumentChunk],
        target_chunk_index: int,
        context_size: int = 1,
    ) -> str:
        """
        Get surrounding context for a specific chunk.

        Args:
            chunks: List of document chunks
            target_chunk_index: Index of the target chunk
            context_size: Number of chunks before/after to include

        Returns:
            Combined text with context
        """
        if not chunks or target_chunk_index >= len(chunks):
            return ""

        start_idx = max(0, target_chunk_index - context_size)
        end_idx = min(len(chunks), target_chunk_index + context_size + 1)

        context_chunks = chunks[start_idx:end_idx]
        return " ".join(chunk.content for chunk in context_chunks)

    def merge_small_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Merge chunks that are smaller than minimum size.

        Args:
            chunks: List of document chunks

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged_chunks = []
        current_chunk = None

        for chunk in chunks:
            if len(chunk.content) < self.strategy.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with previous chunk
                    merged_content = current_chunk.content + " " + chunk.content
                    current_chunk.content = merged_content
                    current_chunk.end_char = chunk.end_char
            else:
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                merged_chunks.append(chunk)

        # Add any remaining chunk
        if current_chunk is not None:
            merged_chunks.append(current_chunk)

        # Update chunk indices
        for i, chunk in enumerate(merged_chunks):
            chunk.chunk_index = i

        return merged_chunks
