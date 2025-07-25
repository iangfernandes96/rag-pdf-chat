"""
Document ingestion service that orchestrates PDF processing and text chunking.
"""

import logging
from pathlib import Path

from .document_parser import PDFParser
from .models import Document, DocumentChunk, ProcessingResult
from .text_chunker import TextChunker
from .utils.timing import time_function

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Dedicated document validation with comprehensive checks."""

    @staticmethod
    def validate_file(
        file_path: Path, max_size_mb: int = 50
    ) -> tuple[bool, str | None]:
        """
        Fast file validation with early exits.

        Args:
            file_path: Path to uploaded file
            max_size_mb: Maximum allowed file size in MB

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not file_path.exists():
                return False, "File does not exist"

            # Check file extension first (fastest check)
            if not file_path.suffix.lower() == ".pdf":
                return False, "Only PDF files are supported"

            # Check file size
            try:
                file_size = file_path.stat().st_size
            except OSError as e:
                return False, f"Cannot access file: {e}"

            max_size_bytes = max_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                file_size_mb = file_size / (1024 * 1024)
                return (
                    False,
                    f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB",
                )

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"


class DocumentIngestionService:
    """Service for ingesting and processing PDF documents."""

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker()
        self.validator = DocumentValidator()

    @time_function
    def ingest_document(
        self, file_path: Path, original_filename: str
    ) -> ProcessingResult:
        """
        Complete document ingestion pipeline.

        Args:
            file_path: Path to the uploaded file (temporary)
            original_filename: Original name of the file

        Returns:
            ProcessingResult with document and chunks
        """
        logger.info(f"Starting document ingestion for: {original_filename}")

        try:
            # Process PDF directly from temporary file (no permanent storage)
            logger.info(f"Processing PDF directly: {file_path}")
            pdf_result = self.pdf_parser.process_pdf(file_path, original_filename)

            if not pdf_result.success or not pdf_result.document:
                return pdf_result

            document = pdf_result.document

            # Chunk the text
            chunks = self.text_chunker.chunk_text(
                document_id=document.id,
                text=pdf_result.content,
                metadata={"source": "pdf_extraction"},
            )

            # Merge small chunks if needed
            chunks = self.text_chunker.merge_small_chunks(chunks)

            # Update document with chunk count
            document.total_chunks = len(chunks)

            logger.info(
                f"Successfully ingested document {original_filename}: "
                f"{len(chunks)} chunks created"
            )

            return ProcessingResult(
                success=True,
                document=pdf_result.document,
                chunks=chunks,
                processing_time=pdf_result.processing_time,
            )

        except Exception as e:
            error_msg = f"Document ingestion failed for {original_filename}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False, error_message=f"Ingestion failed: {str(e)}"
            )

    def validate_upload(
        self, file_path: Path, max_size_mb: int = 50
    ) -> tuple[bool, str | None]:
        """
        Validate uploaded file before processing.

        Args:
            file_path: Path to uploaded file
            max_size_mb: Maximum allowed file size in MB

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.validator.validate_file(file_path, max_size_mb)

    def get_document_stats(
        self, document: Document, chunks: list[DocumentChunk]
    ) -> dict:
        """
        Calculate document processing statistics in single pass - O(n) optimized.

        Args:
            document: Processed document
            chunks: Generated chunks

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
            }

        # Single-pass calculation for optimal performance
        total_size = total_tokens = 0
        min_size = float("inf")
        max_size = 0

        for chunk in chunks:  # O(n) - single iteration
            size = len(chunk.content)
            tokens = chunk.token_count or 0

            total_size += size
            total_tokens += tokens
            min_size = min(min_size, size)
            max_size = max(max_size, size)

        count = len(chunks)
        return {
            "total_chunks": count,
            "avg_chunk_size": total_size // count,
            "min_chunk_size": (int(min_size) if min_size != float("inf") else 0),
            "max_chunk_size": max_size,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": (total_tokens // count if count > 0 else 0),
            "document_size_bytes": document.file_size,
            "pages": document.page_count or 0,
        }

    def get_chunk_by_id(
        self, chunks: list[DocumentChunk], chunk_id: str
    ) -> DocumentChunk | None:
        """
        Retrieve a specific chunk by ID.

        Args:
            chunks: List of chunks to search
            chunk_id: ID of the chunk to find

        Returns:
            DocumentChunk if found, None otherwise
        """
        for chunk in chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_chunks_with_context(
        self,
        chunks: list[DocumentChunk],
        target_chunk_id: str,
        context_size: int = 1,
    ) -> list[DocumentChunk]:
        """
        Get a chunk along with surrounding context chunks.

        Args:
            chunks: List of all chunks
            target_chunk_id: ID of the target chunk
            context_size: Number of chunks before/after to include

        Returns:
            List of chunks including context
        """
        # Find target chunk index
        target_index = None
        for i, chunk in enumerate(chunks):
            if chunk.id == target_chunk_id:
                target_index = i
                break

        if target_index is None:
            return []

        # Calculate context range
        start_idx = max(0, target_index - context_size)
        end_idx = min(len(chunks), target_index + context_size + 1)

        return chunks[start_idx:end_idx]
