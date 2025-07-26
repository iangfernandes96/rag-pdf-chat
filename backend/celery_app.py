"""
Celery configuration for background task processing.
"""

from celery import Celery

from .config import settings

# Create Celery app
celery_app = Celery(
    "rag_pdf_chat",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=["backend.tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=200000,  # 200MB
    result_expires=3600,  # 1 hour
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
)

# Task routing
celery_app.conf.task_routes = {
    "backend.tasks.document_tasks.*": {"queue": "document_processing"},
    "backend.tasks.maintenance_tasks.*": {"queue": "maintenance"},
}

# Task annotations for specific configurations
celery_app.conf.task_annotations = {
    "backend.tasks.document_tasks.process_document": {
        "rate_limit": "10/m",  # Max 10 documents per minute
        "time_limit": 1800,  # 30 minutes
        "soft_time_limit": 1500,  # 25 minutes
    },
    "backend.tasks.document_tasks.generate_embeddings": {
        "rate_limit": "100/m",  # Max 100 embedding batches per minute
        "time_limit": 600,  # 10 minutes
    },
}

if __name__ == "__main__":
    celery_app.start()
