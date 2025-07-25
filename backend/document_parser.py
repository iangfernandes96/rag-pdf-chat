"""
PDF document parsing functionality.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import pdfplumber
from .models import Document, ProcessingResult
from .config import settings

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Custom exception for PDF parsing errors."""
    pass


class PDFParser:
    """Handles PDF document parsing and text extraction."""
    
    def __init__(self):
        self.max_file_size = settings.document.max_file_size_mb * 1024 * 1024
    
    def validate_pdf(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate PDF file before processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if file_path.stat().st_size > self.max_file_size:
            return False, f"File size exceeds maximum allowed size of {settings.document.max_file_size_mb}MB"
        
        if file_path.suffix.lower() != '.pdf':
            return False, "File is not a PDF"
        
        try:
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) == 0:
                    return False, "PDF contains no pages"
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"
        
        return True, None
    
    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, int, dict]:
        """
        Extract text content from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, page_count, metadata)
            
        Raises:
            PDFParseError: If PDF parsing fails
        """
        try:
            logger.info(f"Starting PDF text extraction for: {file_path}")
            start_time = time.time()
            
            extracted_text = ""
            page_count = 0
            metadata = {
                "extraction_method": "pdfplumber",
                "pages_processed": [],
                "extraction_errors": []
            }
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"PDF contains {page_count} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean up the text
                            page_text = self._clean_text(page_text)
                            extracted_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                            metadata["pages_processed"].append(page_num)
                        else:
                            logger.warning(f"No text found on page {page_num}")
                            metadata["extraction_errors"].append(f"No text on page {page_num}")
                    
                    except Exception as e:
                        error_msg = f"Error extracting text from page {page_num}: {str(e)}"
                        logger.error(error_msg)
                        metadata["extraction_errors"].append(error_msg)
                        continue
            
            extraction_time = time.time() - start_time
            metadata["extraction_time_seconds"] = extraction_time
            metadata["total_characters"] = len(extracted_text)
            
            if not extracted_text.strip():
                raise PDFParseError("No text could be extracted from PDF")
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {page_count} pages in {extraction_time:.2f}s")
            
            return extracted_text.strip(), page_count, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
            raise PDFParseError(f"PDF parsing failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and formatting issues.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple whitespaces with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_pdf(self, file_path: Path, original_filename: str) -> ProcessingResult:
        """
        Process a PDF file and create Document object.
        
        Args:
            file_path: Path to the PDF file
            original_filename: Original name of the uploaded file
            
        Returns:
            ProcessingResult with document and processing information
        """
        start_time = time.time()
        
        try:
            # Validate PDF
            is_valid, error_msg = self.validate_pdf(file_path)
            if not is_valid:
                return ProcessingResult(
                    success=False,
                    error_message=error_msg,
                    processing_time=time.time() - start_time
                )
            
            # Extract text
            content, page_count, extraction_metadata = self.extract_text_from_pdf(file_path)
            
            # Create document object
            document = Document(
                filename=file_path.name,
                original_filename=original_filename,
                file_size=file_path.stat().st_size,
                content=content,
                page_count=page_count,
                processing_status="completed",
                metadata=extraction_metadata
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed PDF: {original_filename} in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                document=document,
                processing_time=processing_time
            )
            
        except PDFParseError as e:
            logger.error(f"PDF parsing error for {original_filename}: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Unexpected error processing PDF {original_filename}: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                processing_time=time.time() - start_time
            ) 