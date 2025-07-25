"""
Document ingestion service that orchestrates PDF processing and text chunking.
"""
import logging
from pathlib import Path
from typing import List, Optional

from .models import Document, DocumentChunk, ProcessingResult
from .document_parser import PDFParser
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Service for ingesting and processing PDF documents."""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker()
    
    def ingest_document(self, 
                        file_path: Path, 
                        original_filename: str) -> ProcessingResult:
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
            pdf_result = self.pdf_parser.process_pdf(
                file_path, original_filename
            )
            
            if not pdf_result.success or not pdf_result.document:
                return pdf_result
            
            document = pdf_result.document
            
            # Chunk the text
            chunks = self.text_chunker.chunk_text(
                document_id=document.id,
                text=document.content,
                metadata={"source": "pdf_extraction"}
            )
            
            # Merge small chunks if needed
            chunks = self.text_chunker.merge_small_chunks(chunks)
            
            # Update document with chunk count
            document.chunk_count = len(chunks)
            
            logger.info(
                f"Successfully ingested document {original_filename}: "
                f"{len(chunks)} chunks created"
            )
            
            return ProcessingResult(
                success=True,
                document=pdf_result.document,
                chunks=chunks,
                processing_time=pdf_result.processing_time
            )
            
        except Exception as e:
            logger.error(
                f"Document ingestion failed for {original_filename}: {str(e)}"
            )
            return ProcessingResult(
                success=False,
                error_message=f"Ingestion failed: {str(e)}"
            )
    
    def validate_upload(self, 
                       file_path: Path, 
                       max_size_mb: int = 50) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded file before processing.
        
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
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
            
            # Check file extension
            if not file_path.suffix.lower() == '.pdf':
                return False, "Only PDF files are supported"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_document_stats(self, 
                          document: Document, 
                          chunks: List[DocumentChunk]) -> dict:
        """
        Calculate document processing statistics.
        
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
                "avg_tokens_per_chunk": 0
            }
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        token_counts = [chunk.token_count or 0 for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": (
                sum(token_counts) // len(chunks) if chunks else 0
            ),
            "document_size_bytes": document.file_size,
            "pages": document.page_count or 0
        }
    
    def get_chunk_by_id(self, 
                       chunks: List[DocumentChunk], 
                       chunk_id: str) -> Optional[DocumentChunk]:
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
    
    def get_chunks_with_context(self, 
                               chunks: List[DocumentChunk], 
                               target_chunk_id: str, 
                               context_size: int = 1) -> List[DocumentChunk]:
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