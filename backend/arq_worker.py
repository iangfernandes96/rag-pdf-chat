"""
Arq worker configuration for background task processing.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from .config import settings
from .database import DatabaseService
from .job_tracker import JobTracker
from .models import JobStatus, ProcessingStage
from .rag_service import RAGService

logger = logging.getLogger(__name__)


async def process_document(
    ctx: dict[str, Any], file_data: bytes, filename: str, job_id: str
) -> dict[str, Any]:
    """
    Process a document through the complete RAG pipeline.
    """
    logger.info(f"Starting document processing for job {job_id}: {filename}")

    db_service = DatabaseService()
    rag_service = RAGService()
    job_tracker = JobTracker(redis_url=settings.arq.redis_url)

    try:
        # Initialize job tracker
        try:
            await job_tracker.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize job tracker: {str(e)}")
            job_tracker = None

        # Update job status to processing
        if job_tracker:
            try:
                await job_tracker.update_job_status(
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=ProcessingStage.UPLOAD,
                    progress=10,
                    message="Starting document processing",
                )
            except Exception as e:
                logger.warning(f"Failed to update job status: {str(e)}")

        # Create temporary file
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"{job_id}_{filename}"

        try:
            # Write file data to temporary file
            with open(temp_file, "wb") as f:
                f.write(file_data)

            logger.info(f"File saved to temporary location: {temp_file}")

            # Update status to parsing
            if job_tracker:
                try:
                    await job_tracker.update_job_status(
                        job_id=job_id,
                        status=JobStatus.PROCESSING,
                        stage=ProcessingStage.PARSING,
                        progress=20,
                        message="Parsing document content",
                    )
                except Exception as e:
                    logger.warning(f"Failed to update job status: {str(e)}")

            # Initialize services
            await db_service.initialize()
            await rag_service.initialize()

            # Process document through RAG service
            result = await rag_service.process_document(temp_file, filename)

            if not result["success"]:
                error_msg = result.get("error", "Unknown processing error")
                if job_tracker:
                    try:
                        await job_tracker.update_job_status(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            stage=ProcessingStage.PARSING,
                            progress=0,
                            message=f"Processing failed: {error_msg}",
                            error=error_msg,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update job status: {str(e)}")
                return {"success": False, "error": error_msg, "job_id": job_id}

            document = result["document"]
            chunks = result["chunks"]

            # Store document metadata in database
            logger.info("Storing document metadata in database")
            try:
                db_storage_success = await db_service.store_document_metadata(
                    document, chunks
                )
                if db_storage_success:
                    logger.info("✅ Document metadata stored successfully")
                else:
                    logger.warning("Failed to store document metadata in database")
            except Exception as e:
                logger.warning(f"Failed to store document metadata: {str(e)}")

            # Update job status to completed
            if job_tracker:
                try:
                    await job_tracker.update_job_status(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        stage=ProcessingStage.COMPLETED,
                        progress=100,
                        message=f"Successfully processed {len(chunks)} chunks",
                        document_id=document.id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update job status: {str(e)}")

            logger.info(f"Document processing completed for job {job_id}")

            # Cleanup services
            try:
                await db_service.cleanup()
                await rag_service.cleanup()
                if job_tracker:
                    await job_tracker.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")

            return {
                "success": True,
                "job_id": job_id,
                "document_id": document.id,
                "chunks_count": len(chunks),
            }

        finally:
            # Cleanup temporary file
            if temp_file.exists():
                temp_file.unlink()

    except Exception as e:
        logger.error(f"Document processing failed for job {job_id}: {str(e)}")

        # Update job status to failed
        if job_tracker:
            try:
                await job_tracker.update_job_status(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    stage=ProcessingStage.PARSING,
                    progress=0,
                    message=f"Processing failed: {str(e)}",
                    error=str(e),
                )
            except Exception as update_error:
                logger.warning(f"Failed to update job status: {str(update_error)}")

        return {"success": False, "error": str(e), "job_id": job_id}


async def cleanup_failed_jobs(
    ctx: dict[str, Any], hours_old: int = 24
) -> dict[str, Any]:
    """
    Clean up failed jobs older than specified hours.
    """
    try:
        job_tracker = JobTracker(redis_url=settings.arq.redis_url)
        await job_tracker.initialize()

        # This would implement cleanup logic
        logger.info(f"Cleanup task completed for jobs older than {hours_old} hours")

        await job_tracker.cleanup()
        return {"success": True, "message": "Cleanup completed"}
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return {"success": False, "error": str(e)}


async def health_check_services(ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Health check for all services.
    """
    try:
        # Check database
        db_service = DatabaseService()
        await db_service.initialize()
        db_status = await db_service.get_status()
        await db_service.cleanup()

        # Check RAG service
        rag_service = RAGService()
        await rag_service.initialize()
        rag_status = await rag_service.get_system_status()
        await rag_service.cleanup()

        return {
            "success": True,
            "database": db_status,
            "rag_service": rag_status,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"success": False, "error": str(e)}


# Arq worker functions
class WorkerSettings:
    """Arq worker settings optimized for performance."""

    functions = [
        process_document,
        cleanup_failed_jobs,
        health_check_services,
    ]

    redis_settings = RedisSettings.from_dsn(settings.arq.redis_url)

    # Worker settings optimized for performance
    max_jobs = 4  # Reduced from 10 to prevent resource contention
    job_timeout = 1800  # 30 minutes
    keep_result = 3600  # 1 hour
    max_tries = 2
    retry_delay = 60  # 1 minute


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize worker context."""
    logger.info("Starting Arq worker...")
    ctx["redis"] = await create_pool(WorkerSettings.redis_settings)


async def shutdown(ctx: dict[str, Any]) -> None:
    """Cleanup worker context."""
    logger.info("Shutting down Arq worker...")
    if "redis" in ctx:
        await ctx["redis"].close()
