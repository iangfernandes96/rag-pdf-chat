"""
Constants for the RAG PDF Chat backend application.

This module contains all hardcoded values, magic numbers, and configuration
constants used throughout the application to ensure DRY principle compliance.
"""

from typing import Final


class DefaultValues:
    """Default configuration values."""

    # Service URLs
    QDRANT_URL: Final[str] = "http://localhost:6333"
    OLLAMA_URL: Final[str] = "http://localhost:11434"

    # Celery
    CELERY_BROKER_URL: Final[str] = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: Final[str] = "redis://localhost:6379/0"

    # Document processing
    CHUNK_SIZE: Final[int] = 500
    CHUNK_OVERLAP: Final[int] = 100
    MIN_CHUNK_SIZE: Final[int] = 50

    # Cache settings
    CACHE_SIZE: Final[int] = 1000
    CACHE_TTL_HOURS: Final[float] = 1.0

    # Search limits
    RAG_SEARCH_LIMIT: Final[int] = 100
    VECTOR_SEARCH_LIMIT: Final[int] = 1000
    MIN_SEARCH_LIMIT: Final[int] = 1

    # File processing
    MAX_FILE_SIZE_MB: Final[int] = 10
    MAX_TEXT_SIZE_BYTES: Final[int] = 10_000_000  # 10MB

    # Timeouts and retries
    DEFAULT_TIMEOUT: Final[int] = 30
    HTTP_TIMEOUT: Final[int] = 120
    MAX_RETRIES: Final[int] = 3
    INITIAL_RETRY_DELAY: Final[float] = 1.0

    # Embedding settings
    VECTOR_SIZE: Final[int] = 384
    TOKEN_MULTIPLIER: Final[float] = 1.3  # tokens per word ratio
    CHARS_PER_TOKEN: Final[int] = 4

    # Batch processing
    BATCH_SIZE: Final[int] = 100

    # Database settings
    MAX_QUERY_LENGTH: Final[int] = 1000
    MAX_RESPONSE_LENGTH: Final[int] = 5000
    MODEL_NAME_LENGTH: Final[int] = 100
    STATS_DAYS: Final[int] = 30


class HttpStatus:
    """HTTP status codes."""

    OK: Final[int] = 200
    BAD_REQUEST: Final[int] = 400
    NOT_FOUND: Final[int] = 404
    REQUEST_ENTITY_TOO_LARGE: Final[int] = 413
    UNPROCESSABLE_ENTITY: Final[int] = 422
    INTERNAL_SERVER_ERROR: Final[int] = 500
    SERVICE_UNAVAILABLE: Final[int] = 503


class LoggingPrefixes:
    """Prefixes for logging messages."""

    SUCCESS: Final[str] = "✅"
    ERROR: Final[str] = "❌"
    WARNING: Final[str] = "⚠️"
    INFO: Final[str] = "ℹ️"


class FileExtensions:
    """Supported file extensions."""

    PDF: Final[str] = ".pdf"
    ALLOWED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf",)


class MimeTypes:
    """MIME types for file validation."""

    PDF: Final[str] = "application/pdf"


class ErrorMessages:
    """Standard error messages."""

    # Connection errors
    CONNECTION_FAILED: Final[str] = "Connection failed"
    CLIENT_NOT_CONNECTED: Final[str] = "Client not connected"

    # Validation errors
    EMPTY_INPUT: Final[str] = "Input cannot be empty"
    INVALID_FORMAT: Final[str] = "Invalid format"
    INVALID_DIMENSIONS: Final[str] = "Invalid dimensions"

    # File errors
    FILE_NOT_FOUND: Final[str] = "File not found"
    FILE_TOO_LARGE: Final[str] = "File too large"
    UNSUPPORTED_FORMAT: Final[str] = "Unsupported file format"

    # Processing errors
    PROCESSING_FAILED: Final[str] = "Processing failed"
    EXTRACTION_FAILED: Final[str] = "Text extraction failed"
    EMBEDDING_FAILED: Final[str] = "Embedding generation failed"

    # Database errors
    STORAGE_FAILED: Final[str] = "Storage operation failed"
    RETRIEVAL_FAILED: Final[str] = "Retrieval operation failed"
    DELETION_FAILED: Final[str] = "Deletion operation failed"


class Patterns:
    """Regular expression patterns."""

    # Text processing
    SENTENCE_BOUNDARY: Final[str] = r"(?<=[.!?])\s+"
    WORD_BOUNDARY: Final[str] = r"\b\w+\b"
    WHITESPACE: Final[str] = r"\s+"

    # Validation
    URL_PATTERN: Final[str] = r"^https?://[^\s/$.?#].[^\s]*$"


class QueryParams:
    """Default query parameters."""

    DEFAULT_LIMIT: Final[int] = 5
    DEFAULT_SCORE_THRESHOLD: Final[float] = 0.0
    MIN_SCORE_THRESHOLD: Final[float] = 0.0
    MAX_SCORE_THRESHOLD: Final[float] = 1.0


class SnippetSettings:
    """Settings for snippet generation."""

    MAX_LENGTH: Final[int] = 200
    PREVIEW_LENGTH: Final[int] = 100
    CONTEXT_BONUS: Final[int] = 10
    PHRASE_MATCH_BONUS: Final[float] = 0.2


class SystemMessages:
    """System and template messages."""

    # Status messages
    SYSTEM_HEALTHY: Final[str] = "System is healthy"
    SYSTEM_UNHEALTHY: Final[str] = "System health check failed"

    # Processing messages
    PROCESSING_DOCUMENT: Final[str] = "Processing document"
    DOCUMENT_PROCESSED: Final[str] = "Document processed successfully"

    # Cache messages
    CACHE_HIT: Final[str] = "Cache hit for query"
    CACHE_MISS: Final[str] = "Cache miss for query"
    CACHE_CLEARED: Final[str] = "Cache cleared"

    # Connection messages
    CONNECTING: Final[str] = "Connecting to service"
    CONNECTED: Final[str] = "Successfully connected"
    DISCONNECTED: Final[str] = "Connection closed"

    # Template messages
    NO_CONTEXT_FOUND: Final[str] = "No relevant context found."
    CONTEXT_PREFIX: Final[str] = "[Context {}]"
    SNIPPET_PREFIX: Final[str] = "..."
    SNIPPET_SUFFIX: Final[str] = "..."


class DatabaseTables:
    """Database table and column names."""

    # Table names
    DOCUMENTS: Final[str] = "documents"
    QUERY_SESSIONS: Final[str] = "query_sessions"

    # Common columns
    ID: Final[str] = "id"
    CREATED_AT: Final[str] = "created_at"
    UPDATED_AT: Final[str] = "updated_at"


class CacheKeys:
    """Cache key prefixes and patterns."""

    QUERY_PREFIX: Final[str] = "query:"
    MODEL_PREFIX: Final[str] = "model:"
    DOCUMENT_PREFIX: Final[str] = "doc:"


class PerformanceThresholds:
    """Performance monitoring thresholds."""

    SLOW_QUERY_MS: Final[int] = 1000
    VERY_SLOW_QUERY_MS: Final[int] = 5000
    CACHE_CLEANUP_RATIO: Final[float] = 0.2  # Remove 20% of cache entries

    # Operation timing thresholds (seconds)
    FAST_OPERATION: Final[float] = 0.1
    NORMAL_OPERATION: Final[float] = 1.0
    SLOW_OPERATION: Final[float] = 5.0


class ModelDefaults:
    """Default model and embedding settings."""

    EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
    LLM_MODEL: Final[str] = "mistral"
    MAX_TOKENS: Final[int] = 2048
    TEMPERATURE: Final[float] = 0.7


class ValidationLimits:
    """Validation limits for various inputs."""

    # Text limits
    MAX_QUERY_LENGTH: Final[int] = 1000
    MAX_CONTENT_LENGTH: Final[int] = 1_000_000
    MIN_CONTENT_LENGTH: Final[int] = 1

    # Numeric limits
    MAX_CHUNK_SIZE: Final[int] = 2000
    MIN_CHUNK_SIZE: Final[int] = 10
    MAX_OVERLAP: Final[int] = 500
    MIN_OVERLAP: Final[int] = 0

    # Collection limits
    MAX_CHUNKS_PER_DOCUMENT: Final[int] = 10000
    MAX_DOCUMENTS: Final[int] = 100000


class EnvironmentKeys:
    """Environment variable keys."""

    # Database
    DATABASE_URL: Final[str] = "DATABASE_URL"

    # Vector store
    QDRANT_URL: Final[str] = "QDRANT_URL"

    # LLM
    OLLAMA_URL: Final[str] = "OLLAMA_URL"
    OLLAMA_MODEL: Final[str] = "OLLAMA_MODEL"

    # Document processing
    CHUNK_SIZE: Final[str] = "CHUNK_SIZE"
    CHUNK_OVERLAP: Final[str] = "CHUNK_OVERLAP"
    MAX_FILE_SIZE_MB: Final[str] = "MAX_FILE_SIZE_MB"

    # System
    DEBUG: Final[str] = "DEBUG"
    RELOAD: Final[str] = "RELOAD"

    # Celery
    CELERY_BROKER_URL: Final[str] = "CELERY_BROKER_URL"
    CELERY_RESULT_BACKEND: Final[str] = "CELERY_RESULT_BACKEND"
