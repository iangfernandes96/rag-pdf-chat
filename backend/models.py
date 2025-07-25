"""
Data models for RAG PDF Chat application.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a text chunk from a document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    token_count: int | None = None
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


class Document(BaseModel):
    """Represents a processed document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    original_filename: str
    file_size: int
    content: str
    page_count: int | None = None
    chunk_count: int = 0
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: str | None = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of document processing operation."""

    success: bool
    document: Document | None = None
    chunks: list[DocumentChunk] = Field(default_factory=list)
    error_message: str | None = None
    processing_time: float = 0.0


class ChunkingStrategy(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = 500
    overlap: int = 100
    preserve_sentences: bool = True
    min_chunk_size: int = 50


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""

    chunk_id: str
    embedding: list[float]
    model_name: str
    generation_time: float
