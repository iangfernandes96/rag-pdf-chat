"""
Redis-based job tracking service for background processing jobs.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Optional

import redis.asyncio as redis

from .models import JobInfo, JobStatus, ProcessingStage

logger = logging.getLogger(__name__)


class JobTrackerError(Exception):
    """Custom job tracker error with context."""

    pass


class JobTracker:
    """Redis-based job tracking service."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.job_key_prefix = "job:"
        self.job_ttl = 3600 * 24 * 7  # 7 days

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Job tracker connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {str(e)}")
            raise JobTrackerError(f"Redis connection failed: {str(e)}") from e

    async def create_job(
        self,
        job_id: str,
        filename: str,
        file_size: int,
    ) -> JobInfo:
        """
        Create a new job record.
        
        Args:
            job_id: Unique job identifier
            filename: Original filename
            file_size: File size in bytes
            
        Returns:
            Created job info
        """
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")

        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            stage=ProcessingStage.UPLOAD,
            progress=0,
            message="Job created, waiting to be processed",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            filename=filename,
        )

        await self._store_job(job_info)
        logger.info(f"Created job {job_id} for file {filename}")
        return job_info

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        stage: ProcessingStage,
        progress: int,
        message: str,
        error: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """
        Update job status and progress.
        
        Args:
            job_id: Job identifier
            status: New job status
            stage: Current processing stage
            progress: Progress percentage (0-100)
            message: Status message
            error: Error message if failed
            document_id: Document ID if completed
            
        Returns:
            Success status
        """
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")

        try:
            # Get existing job info
            job_info = await self.get_job(job_id)
            if not job_info:
                logger.warning(f"Job {job_id} not found for status update")
                return False

            # Update job info
            job_info.status = status
            job_info.stage = stage
            job_info.progress = progress
            job_info.message = message
            job_info.error = error
            job_info.updated_at = datetime.now(UTC)
            
            if document_id:
                job_info.document_id = document_id

            # Store updated job info
            await self._store_job(job_info)
            
            logger.info(
                f"Updated job {job_id}: {status.value} - {stage.value} "
                f"({progress}%) - {message}"
            )
            
            return True

        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {str(e)}")
            return False

    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """
        Get job information by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job information or None if not found
        """
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")

        try:
            key = f"{self.job_key_prefix}{job_id}"
            data = await self.redis_client.get(key)
            
            if not data:
                return None

            job_dict = json.loads(data)
            return JobInfo(**job_dict)

        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {str(e)}")
            return None

    async def list_jobs(
        self, 
        status: Optional[JobStatus] = None, 
        limit: int = 100
    ) -> list[JobInfo]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Filter by job status
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")

        try:
            pattern = f"{self.job_key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            jobs = []
            for key in keys[:limit]:
                data = await self.redis_client.get(key)
                if data:
                    job_dict = json.loads(data)
                    job_info = JobInfo(**job_dict)
                    
                    if status is None or job_info.status == status:
                        jobs.append(job_info)

            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            return []

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job record.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Success status
        """
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")

        try:
            key = f"{self.job_key_prefix}{job_id}"
            result = await self.redis_client.delete(key)
            logger.info(f"Deleted job {job_id}")
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False

    async def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """
        Clean up old completed/failed jobs.
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Number of jobs cleaned up
        """
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")

        try:
            cutoff_time = datetime.now(UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            cutoff_time = cutoff_time.replace(day=cutoff_time.day - days_old)

            pattern = f"{self.job_key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    job_dict = json.loads(data)
                    job_info = JobInfo(**job_dict)
                    
                    # Delete old completed or failed jobs
                    old_statuses = [JobStatus.COMPLETED, JobStatus.FAILED]
                    if (job_info.status in old_statuses 
                        and job_info.created_at < cutoff_time):
                        await self.redis_client.delete(key)
                        cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {str(e)}")
            return 0

    async def _store_job(self, job_info: JobInfo) -> None:
        """Store job information in Redis."""
        if not self.redis_client:
            raise JobTrackerError("Job tracker not initialized")
            
        key = f"{self.job_key_prefix}{job_info.job_id}"
        data = json.dumps(job_info.model_dump(), default=str)
        await self.redis_client.setex(key, self.job_ttl, data)

    async def cleanup(self) -> None:
        """Cleanup Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Job tracker Redis connection closed") 