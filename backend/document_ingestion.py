"""
Document ingestion service that orchestrates PDF processing and text chunking.
"""
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from .models import Document, DocumentChunk, ProcessingResult
from .document_parser import PDFParser
from .text_chunker import TextChunker
from .config import settings

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Service for ingesting and processing PDF documents."""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker()
        self.upload_dir = Path(settings.document.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def ingest_document(self, 
                       file_path: Path, 
                       original_filename: str) -> ProcessingResult:
        """
        Complete document ingestion pipeline.
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original name of the file
            
        Returns:
            ProcessingResult with document and chunks
        """
        logger.info(f"Starting document ingestion for: {original_filename}")
        
        try:
            # Generate unique filename and move to upload directory
            file_extension = file_path.suffix
            unique_filename = f"{uuid4()}{file_extension}"
            target_path = self.upload_dir / unique_filename
            
            # Copy file to permanent location
            shutil.copy2(file_path, target_path)
            logger.info(f"File copied to: {target_path}")
            
            # Process PDF
            pdf_result = self.pdf_parser.process_pdf(target_path, original_filename)
            
            if not pdf_result.success:
                return pdf_result
            
            # Chunk the text
            chunks = self.text_chunker.chunk_text(
                document_id=pdf_result.document.id,
                text=pdf_result.document.content,
                metadata={"source": "pdf_extraction"}
            )
            
            # Merge small chunks if needed
            chunks = self.text_chunker.merge_small_chunks(chunks)
            
            # Update document with chunk count
            pdf_result.document.chunk_count = len(chunks)
            
            logger.info(f"Successfully ingested document {original_filename}: "
                       f"{len(chunks)} chunks created")
            
            return ProcessingResult(
                success=True,
                document=pdf_result.document,
                chunks=chunks,
                processing_time=pdf_result.processing_time
            )
            
        except Exception as e:
            logger.error(f"Document ingestion failed for {original_filename}: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=f"Ingestion failed: {str(e)}"
            )
    
    def validate_upload(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded file before processing.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file extension
        allowed_extensions = [f".{ext}" for ext in settings.document.allowed_extensions]
        if file_path.suffix.lower() not in allowed_extensions:
            return False, f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        
        # Check file size
        max_size = settings.document.max_file_size_mb * 1024 * 1024
        if file_path.stat().st_size > max_size:
            return False, f"File size exceeds {settings.document.max_file_size_mb}MB limit"
        
        return True, None
    
    def get_document_stats(self, document: Document, chunks: List[DocumentChunk]) -> dict:
        """
        Generate statistics for a processed document.
        
        Args:
            document: Processed document
            chunks: Document chunks
            
        Returns:
            Dictionary with document statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "total_tokens": 0,
                "processing_time": 0
            }
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        token_counts = [chunk.token_count or 0 for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "file_size_mb": document.file_size / (1024 * 1024),
            "page_count": document.page_count,
            "chars_per_page": len(document.content) / (document.page_count or 1)
        }
    
    def cleanup_failed_upload(self, file_path: Path) -> None:
        """
        Clean up files from failed uploads.
        
        Args:
            file_path: Path to the file to clean up
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up failed upload: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {str(e)}")
    
    def get_chunk_by_id(self, chunks: List[DocumentChunk], chunk_id: str) -> Optional[DocumentChunk]:
        """
        Find a chunk by its ID.
        
        Args:
            chunks: List of chunks to search
            chunk_id: ID of the chunk to find
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        return next((chunk for chunk in chunks if chunk.id == chunk_id), None)
    
    def get_chunks_with_context(self, 
                               chunks: List[DocumentChunk], 
                               chunk_indices: List[int], 
                               context_size: int = 1) -> List[str]:
        """
        Get multiple chunks with their surrounding context.
        
        Args:
            chunks: List of all document chunks
            chunk_indices: Indices of chunks to retrieve
            context_size: Number of surrounding chunks to include
            
        Returns:
            List of context strings for each requested chunk
        """
        contexts = []
        for chunk_index in chunk_indices:
            context = self.text_chunker.get_chunk_context(
                chunks, chunk_index, context_size
            )
            contexts.append(context)
        
        return contexts 