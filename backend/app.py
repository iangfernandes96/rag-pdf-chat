"""
FastAPI backend for RAG PDF Chat system.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .constants import HttpStatus
from .database import DatabaseService
from .job_tracker import JobTracker
from .llm_service import LLMService
from .models import (
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
)
from .rag_service import RAGService
from .tasks.document_tasks import process_document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for application startup/shutdown
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Manage application lifecycle and service initialization."""
    logger.info("🚀 Starting RAG PDF Chat API...")

    try:
        # Initialize only essential services for API server
        logger.info("📊 Initializing database service...")
        db_service = DatabaseService()
        await db_service.initialize()
        app.state.db_service = db_service

        # Initialize RAG service
        logger.info("🧠 Initializing RAG service...")
        rag_service = RAGService()
        await rag_service.initialize()
        app.state.rag_service = rag_service

        # Initialize LLM service
        logger.info("🤖 Initializing LLM service...")
        llm_service = LLMService()
        await llm_service.initialize()
        app.state.llm_service = llm_service

        # Initialize Celery app for job queuing
        logger.info("🔄 Initializing Celery app...")
        from .celery_app import celery_app

        app.state.celery_app = celery_app

        # Initialize job tracker
        logger.info("📋 Initializing job tracker...")
        job_tracker = JobTracker(redis_url=settings.celery.broker_url)
        await job_tracker.initialize()
        app.state.job_tracker = job_tracker

        logger.info("✅ API server initialized successfully")

        yield  # Application runs here

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise RuntimeError(f"Service initialization failed: {str(e)}") from e

    finally:
        # Cleanup services on shutdown
        logger.info("🛑 Shutting down RAG PDF Chat API...")

        if hasattr(app.state, "rag_service") and app.state.rag_service:
            await app.state.rag_service.cleanup()

        if hasattr(app.state, "llm_service") and app.state.llm_service:
            await app.state.llm_service.cleanup()

        if hasattr(app.state, "db_service") and app.state.db_service:
            await app.state.db_service.cleanup()

        if hasattr(app.state, "job_tracker") and app.state.job_tracker:
            await app.state.job_tracker.cleanup()

        logger.info("✅ Shutdown completed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG PDF Chat API",
    description="Retrieval-Augmented Generation API for PDF document querying",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    try:
        # Get database status (always available)
        db_status = await app.state.db_service.get_status()

        # Get system status from all services
        rag_status = await app.state.rag_service.get_system_status()
        llm_status = await app.state.llm_service.get_status()

        services = {
            "rag_service": rag_status,
            "llm_service": llm_status,
            "database": db_status,
        }

        # Determine overall health using unified format
        overall_healthy = all(
            [
                rag_status.get("healthy", False),
                llm_status.get("healthy", False),
                db_status.get("healthy", False),
            ]
        )

        return HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            timestamp=datetime.now(UTC).isoformat(),
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
    Upload a PDF document for background processing.

    Args:
        file: PDF file to upload and process

    Returns:
        Job information for tracking processing status
    """
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
    content = await file.read()
    file_size = len(content)

    max_size = settings.document.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=HttpStatus.REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.document.max_file_size_mb}MB",
        )

    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Create job record in Redis
        await app.state.job_tracker.create_job(
            job_id=job_id,
            filename=file.filename,
            file_size=file_size,
        )

        logger.info(
            f"Queuing document for processing: {file.filename} ({file_size} bytes)"
        )

        # Queue document processing task
        process_document.delay(content, file.filename, job_id)

        logger.info(f"Document processing queued with job ID: {job_id}")

        return DocumentUploadResponse(
            success=True,
            job_id=job_id,
            message="Document uploaded successfully. Processing started in background.",
            status=JobStatus.PENDING,
        )

    except Exception as e:
        logger.error(f"Failed to queue document processing: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue document processing: {str(e)}",
        ) from e


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query uploaded documents using RAG (Retrieval-Augmented Generation).

    Args:
        request: Query request with question and search parameters

    Returns:
        AI-generated answer with source context
    """
    try:
        start_time = datetime.now(UTC)

        logger.info(f"Processing query: {request.query}")

        # Search for relevant document chunks
        search_result = await app.state.rag_service.search_documents(
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
                response_time=(datetime.now(UTC) - start_time).total_seconds(),
                model_used="N/A",
                chunks_used=0,
            )

        # Generate response using LLM
        llm_response = await app.state.llm_service.generate_rag_response(
            query=request.query,
            context_chunks=chunks,
            include_sources=request.include_context,
        )

        if not llm_response["success"]:
            raise HTTPException(
                status_code=HttpStatus.INTERNAL_SERVER_ERROR,
                detail=f"LLM generation failed: {llm_response.get('error', 'Unknown error')}",
            )

        response_time = (datetime.now(UTC) - start_time).total_seconds()

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
    try:
        documents = await app.state.db_service.get_all_documents()

        total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)

        return DocumentListResponse(
            documents=documents, total_count=len(documents), total_chunks=total_chunks
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        # Return empty list instead of error for better UX
        return DocumentListResponse(documents=[], total_count=0, total_chunks=0)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a background processing job.

    Args:
        job_id: Job identifier

    Returns:
        Current job status and progress information
    """
    try:
        # Get job status from Redis
        job_info = await app.state.job_tracker.get_job(job_id)

        if not job_info:
            raise HTTPException(
                status_code=HttpStatus.NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        return JobStatusResponse(success=True, job_info=job_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}",
        ) from e


@app.get("/jobs", response_model=dict[str, Any])
async def list_jobs(status: JobStatus | None = None, limit: int = 50):
    """
    List all background processing jobs.

    Args:
        status: Filter by job status
        limit: Maximum number of jobs to return

    Returns:
        List of job information
    """
    try:
        jobs = await app.state.job_tracker.list_jobs(status=status, limit=limit)
        return {
            "success": True,
            "jobs": [job.model_dump() for job in jobs],
            "count": len(jobs),
        }

    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}",
        ) from e


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks.

    Args:
        document_id: ID of the document to delete

    Returns:
        Deletion confirmation
    """

    try:
        # Delete from vector store
        vector_deleted = await app.state.rag_service.delete_document(document_id)

        # Delete from database
        db_deleted = await app.state.db_service.delete_document(document_id)

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
        workers=4,
        log_level="info",
    )
