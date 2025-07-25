"""
Data models for the RAG PDF Chat application.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .constants import DefaultValues


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Document(BaseModel):
    """A processed document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    original_filename: str
    file_size: int
    page_count: int
    total_chunks: int
    uploaded_at: datetime
    processing_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of document processing."""

    success: bool
    document: Document | None = None
    chunks: list[DocumentChunk] = Field(default_factory=list)
    processing_time: float = 0.0
    error_message: str | None = None


class ChunkingStrategy(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = DefaultValues.CHUNK_SIZE
    overlap: int = DefaultValues.CHUNK_OVERLAP
    preserve_sentences: bool = True
    min_chunk_size: int = DefaultValues.MIN_CHUNK_SIZE


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""

    chunk_id: str
    embedding: list[float]
    model_name: str
    generation_time: float
