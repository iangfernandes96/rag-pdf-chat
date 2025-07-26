"""
Configuration settings for the RAG PDF Chat application.
"""

import os

from pydantic import BaseModel

from .constants import DefaultValues, EnvironmentKeys


class DatabaseSettings(BaseModel):
    """Database configuration."""

    url: str = os.getenv(
        EnvironmentKeys.DATABASE_URL, "postgresql://localhost/rag_pdf_chat"
    )


class VectorSettings(BaseModel):
    """Vector store configuration."""

    qdrant_url: str = DefaultValues.QDRANT_URL
    collection_name: str = "documents"
    vector_size: int = DefaultValues.VECTOR_SIZE
    batch_size: int = DefaultValues.BATCH_SIZE


class LLMSettings(BaseModel):
    """LLM configuration."""

    ollama_url: str = DefaultValues.OLLAMA_URL
    ollama_model: str = "mistral"
    timeout: int = DefaultValues.HTTP_TIMEOUT
    max_tokens: int = 2048
    temperature: float = 0.7


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str = "./cache"
    batch_size: int = DefaultValues.BATCH_SIZE


class DocumentSettings(BaseModel):
    """Document processing configuration."""

    chunk_size: int = DefaultValues.CHUNK_SIZE
    chunk_overlap: int = DefaultValues.CHUNK_OVERLAP
    max_file_size_mb: int = DefaultValues.MAX_FILE_SIZE_MB
    allowed_extensions: tuple[str, ...] = (".pdf",)


class ArqSettings(BaseModel):
    """Arq configuration."""

    redis_url: str = os.getenv(
        EnvironmentKeys.ARQ_REDIS_URL, DefaultValues.ARQ_REDIS_URL
    )


class Settings:
    """Main application settings."""

    def __init__(self):
        self.debug: bool = os.getenv(EnvironmentKeys.DEBUG, "False").lower() == "true"
        self.reload: bool = os.getenv(EnvironmentKeys.RELOAD, "False").lower() == "true"

        # Service configurations
        self.database = DatabaseSettings()
        self.vector = VectorSettings(
            qdrant_url=os.getenv(EnvironmentKeys.QDRANT_URL, DefaultValues.QDRANT_URL),
        )
        self.llm = LLMSettings(
            ollama_url=os.getenv(EnvironmentKeys.OLLAMA_URL, DefaultValues.OLLAMA_URL),
            ollama_model=os.getenv(EnvironmentKeys.OLLAMA_MODEL, "mistral"),
        )
        self.embedding = EmbeddingSettings()
        self.document = DocumentSettings(
            chunk_size=int(
                os.getenv(EnvironmentKeys.CHUNK_SIZE, str(DefaultValues.CHUNK_SIZE))
            ),
            chunk_overlap=int(
                os.getenv(
                    EnvironmentKeys.CHUNK_OVERLAP, str(DefaultValues.CHUNK_OVERLAP)
                )
            ),
            max_file_size_mb=int(
                os.getenv(
                    EnvironmentKeys.MAX_FILE_SIZE_MB,
                    str(DefaultValues.MAX_FILE_SIZE_MB),
                )
            ),
        )
        self.arq = ArqSettings()


# Global settings instance
settings = Settings()
