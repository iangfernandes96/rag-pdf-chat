"""
Background tasks for document processing.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from ..celery_app import celery_app
from ..database import DatabaseService
from ..job_tracker import JobTracker
from ..models import JobStatus, ProcessingStage
from ..rag_service import RAGService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def process_document(
    self, file_data: bytes, filename: str, job_id: str
) -> dict[str, Any]:
    """
    Process a document through the complete RAG pipeline.

    Args:
        file_data: Raw file data
        filename: Original filename
        job_id: Unique job identifier

    Returns:
        Processing result dictionary
    """
    logger.info(f"Starting document processing for job {job_id}: {filename}")

    # Initialize services
    db_service = DatabaseService()
    rag_service = RAGService()
    job_tracker = JobTracker()

    try:
        # Initialize job tracker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(job_tracker.initialize())
        except Exception as e:
            logger.error(f"Failed to initialize job tracker: {str(e)}")
            # Continue without job tracking if Redis is unavailable
            job_tracker = None
        finally:
            loop.close()

        # Update job status to processing
        if job_tracker:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    job_tracker.update_job_status(
                        job_id=job_id,
                        status=JobStatus.PROCESSING,
                        stage=ProcessingStage.UPLOAD,
                        progress=10,
                        message="Starting document processing",
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to update job status: {str(e)}")
            finally:
                loop.close()

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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        job_tracker.update_job_status(
                            job_id=job_id,
                            status=JobStatus.PROCESSING,
                            stage=ProcessingStage.PARSING,
                            progress=20,
                            message="Parsing document content",
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to update job status: {str(e)}")
                finally:
                    loop.close()

            # Initialize services
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(db_service.initialize())
                loop.run_until_complete(rag_service.initialize())

                # Process document through RAG service
                result = loop.run_until_complete(
                    rag_service.process_document(temp_file, filename)
                )
            except Exception as e:
                # Cleanup services on error
                try:
                    loop.run_until_complete(db_service.cleanup())
                    loop.run_until_complete(rag_service.cleanup())
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup error: {cleanup_error}")
                raise e
            finally:
                loop.close()

            if not result["success"]:
                error_msg = result.get("error", "Unknown processing error")
                if job_tracker:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            job_tracker.update_job_status(
                                job_id=job_id,
                                status=JobStatus.FAILED,
                                stage=ProcessingStage.PARSING,
                                progress=0,
                                message=f"Processing failed: {error_msg}",
                                error=error_msg,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update job status: {str(e)}")
                    finally:
                        loop.close()
                return {"success": False, "error": error_msg, "job_id": job_id}

            # Update status to completed
            document = result["document"]
            chunks = result["chunks"]

            if job_tracker:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        job_tracker.update_job_status(
                            job_id=job_id,
                            status=JobStatus.COMPLETED,
                            stage=ProcessingStage.COMPLETED,
                            progress=100,
                            message=f"Successfully processed {len(chunks)} chunks",
                            document_id=document.id,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to update job status: {str(e)}")
                finally:
                    loop.close()

            logger.info(f"Document processing completed for job {job_id}")

            # Cleanup services after successful processing
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(db_service.cleanup())
                loop.run_until_complete(rag_service.cleanup())
                if job_tracker:
                    loop.run_until_complete(job_tracker.cleanup())
                loop.close()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")

            return {
                "success": True,
                "job_id": job_id,
                "document_id": document.id,
                "chunks_count": len(chunks),
                "processing_time": result.get("processing_time", 0.0)
            }

        finally:
            # Cleanup temporary file
            if temp_file.exists():
                temp_file.unlink()

    except Exception as e:
        logger.error(f"Document processing failed for job {job_id}: {str(e)}")

        # Update job status to failed
        if job_tracker:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    job_tracker.update_job_status(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        stage=ProcessingStage.PARSING,
                        progress=0,
                        message=f"Processing failed: {str(e)}",
                        error=str(e),
                    )
                )
            except Exception as update_error:
                logger.warning(f"Failed to update job status: {str(update_error)}")
            finally:
                loop.close()

        return {
            "success": False,
            "error": str(e),
            "job_id": job_id
        }
