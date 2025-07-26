"""
Data models for the RAG PDF Chat application.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .constants import DefaultValues, SnippetSettings


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
    content: str = ""  # Extracted text content
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


# Request/Response Models
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    success: bool
    document_id: str
    filename: str
    message: str
    chunks_created: int
    processing_time: float
    file_size: int
    page_count: int


class QueryRequest(BaseModel):
    """Request model for document querying."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=SnippetSettings.MAX_LENGTH,
        description="The question to ask about uploaded documents",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of relevant chunks to retrieve",
    )
    document_id: str | None = Field(
        default=None, description="Optional: limit search to specific document"
    )
    include_context: bool = Field(
        default=True, description="Whether to include source context in response"
    )


class QueryResponse(BaseModel):
    """Response model for document querying."""

    success: bool
    query: str
    answer: str
    sources: list[dict[str, Any]]
    response_time: float
    model_used: str
    chunks_used: int


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""

    documents: list[dict[str, Any]]
    total_count: int
    total_chunks: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: str
    services: dict[str, dict[str, Any]]
    version: str
