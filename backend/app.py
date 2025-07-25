"""
FastAPI backend for RAG PDF Chat system.
"""

import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .constants import HttpStatus, SnippetSettings
from .database import DatabaseService
from .llm_service import LLMService
from .rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Chat API",
    description="Retrieval-Augmented Generation API for PDF document querying",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (initialized on startup)
rag_service: RAGService | None = None
llm_service: LLMService | None = None
db_service: DatabaseService | None = None


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


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global rag_service, llm_service, db_service

    logger.info("🚀 Starting RAG PDF Chat API...")

    try:
        # Initialize database service
        logger.info("📊 Initializing database service...")
        db_service = DatabaseService()
        await db_service.initialize()

        # Initialize RAG service
        logger.info("🧠 Initializing RAG service...")
        rag_service = RAGService()
        await rag_service.initialize()

        # Initialize LLM service
        logger.info("🤖 Initializing LLM service...")
        llm_service = LLMService()
        await llm_service.initialize()

        logger.info("✅ All services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise RuntimeError(f"Service initialization failed: {str(e)}") from e


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on application shutdown."""
    global rag_service, llm_service, db_service

    logger.info("🛑 Shutting down RAG PDF Chat API...")

    if rag_service:
        await rag_service.cleanup()

    if llm_service:
        await llm_service.cleanup()

    if db_service:
        await db_service.cleanup()

    logger.info("✅ Shutdown completed")


# API Endpoints


@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG PDF Chat API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation API for PDF document querying",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring system status."""
    if not rag_service or not llm_service or not db_service:
        raise HTTPException(
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
            detail="Services not initialized",
        )

    try:
        # Get system status from all services
        rag_status = await rag_service.get_system_status()
        llm_status = await llm_service.get_status()
        db_status = await db_service.get_status()

        services = {
            "rag_service": rag_status,
            "llm_service": llm_status,
            "database": db_status,
        }

        # Determine overall health
        overall_healthy = all(
            [
                rag_status.get("system_healthy", False),
                llm_status.get("healthy", False),
                db_status.get("healthy", False),
            ]
        )

        return HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            services=services,
            version="1.0.0",
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        ) from e


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document for RAG querying.

    Args:
        file: PDF file to upload and process

    Returns:
        Document processing results with metadata
    """
    if not rag_service or not db_service:
        raise HTTPException(
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
            detail="Services not initialized",
        )

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=HttpStatus.BAD_REQUEST, detail="No file provided"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=HttpStatus.BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)

    max_size = settings.document.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=HttpStatus.REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.document.max_file_size_mb}MB",
        )

    # Create temporary file
    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        # Write uploaded content to temporary file
        with open(temp_file, "wb") as f:
            f.write(content)

        logger.info(f"Processing uploaded file: {file.filename} ({file_size} bytes)")

        # Process document through RAG service
        start_time = datetime.utcnow()
        result = await rag_service.process_document(temp_file, file.filename)
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        if not result["success"]:
            raise HTTPException(
                status_code=HttpStatus.UNPROCESSABLE_ENTITY,
                detail=f"Document processing failed: {result.get('error', 'Unknown error')}",
            )

        document = result["document"]
        chunks = result["chunks"]

        # Store document metadata in database
        await db_service.store_document_metadata(document, chunks)

        logger.info(
            f"Successfully processed {file.filename}: {len(chunks)} chunks created"
        )

        return DocumentUploadResponse(
            success=True,
            document_id=document.id,
            filename=document.original_filename,
            message=f"Document processed successfully. Created {len(chunks)} chunks.",
            chunks_created=len(chunks),
            processing_time=processing_time,
            file_size=file_size,
            page_count=document.page_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        ) from e
    finally:
        # Cleanup temporary file
        if temp_file.exists():
            temp_file.unlink()


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query uploaded documents using RAG (Retrieval-Augmented Generation).

    Args:
        request: Query request with question and search parameters

    Returns:
        AI-generated answer with source context
    """
    if not rag_service or not llm_service:
        raise HTTPException(
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
            detail="Services not initialized",
        )

    try:
        start_time = datetime.utcnow()

        logger.info(f"Processing query: {request.query}")

        # Search for relevant document chunks
        search_result = await rag_service.search_documents(
            query=request.query,
            limit=request.limit,
            document_filter=request.document_id,
        )

        if not search_result["success"]:
            raise HTTPException(
                status_code=HttpStatus.INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {search_result.get('error', 'Unknown error')}",
            )

        chunks = search_result["results"]

        if not chunks:
            return QueryResponse(
                success=True,
                query=request.query,
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                sources=[],
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                model_used="N/A",
                chunks_used=0,
            )

        # Generate response using LLM
        llm_response = await llm_service.generate_rag_response(
            query=request.query,
            context_chunks=chunks,
            include_sources=request.include_context,
        )

        if not llm_response["success"]:
            raise HTTPException(
                status_code=HttpStatus.INTERNAL_SERVER_ERROR,
                detail=f"LLM generation failed: {llm_response.get('error', 'Unknown error')}",
            )

        response_time = (datetime.utcnow() - start_time).total_seconds()

        # Format source information
        sources = []
        if request.include_context:
            for chunk in chunks:
                sources.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "content": (
                            chunk["content"][:200] + "..."
                            if len(chunk["content"]) > 200
                            else chunk["content"]
                        ),
                        "score": chunk["score"],
                        "chunk_index": chunk["chunk_index"],
                    }
                )

        logger.info(
            f"Query completed in {response_time:.2f}s using {len(chunks)} chunks"
        )

        return QueryResponse(
            success=True,
            query=request.query,
            answer=llm_response["answer"],
            sources=sources,
            response_time=response_time,
            model_used=llm_response["model_used"],
            chunks_used=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        ) from e


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents with metadata.

    Returns:
        List of documents with processing statistics
    """
    # Return empty list if database service not available
    if not db_service or not hasattr(db_service, "pool") or not db_service.pool:
        logger.warning("Database service not available, returning empty document list")
        return DocumentListResponse(documents=[], total_count=0, total_chunks=0)

    try:
        documents = await db_service.get_all_documents()

        total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)

        return DocumentListResponse(
            documents=documents, total_count=len(documents), total_chunks=total_chunks
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        # Return empty list instead of error for better UX
        return DocumentListResponse(documents=[], total_count=0, total_chunks=0)


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks.

    Args:
        document_id: ID of the document to delete

    Returns:
        Deletion confirmation
    """
    if not rag_service or not db_service:
        raise HTTPException(
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
            detail="Services not initialized",
        )

    try:
        # Delete from vector store
        vector_deleted = await rag_service.delete_document(document_id)

        # Delete from database
        db_deleted = await db_service.delete_document(document_id)

        if not vector_deleted or not db_deleted:
            raise HTTPException(
                status_code=HttpStatus.NOT_FOUND,
                detail="Document not found or deletion failed",
            )

        logger.info(f"Successfully deleted document: {document_id}")

        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(e)}",
        ) from e


# Error handlers
@app.exception_handler(HttpStatus.NOT_FOUND)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=HttpStatus.NOT_FOUND,
        content={"error": "Endpoint not found", "detail": str(exc)},
    )


@app.exception_handler(HttpStatus.INTERNAL_SERVER_ERROR)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=HttpStatus.INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


# Development server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info",
    )
