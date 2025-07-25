"""
Configuration settings for RAG PDF Chat application.
"""

import os

from pydantic import BaseModel


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_pdf_chat"
    echo: bool = False


class VectorSettings(BaseModel):
    """Vector database configuration settings."""

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "documents"
    vector_size: int = 384  # all-MiniLM-L6-v2 embedding size


class LLMSettings(BaseModel):
    """LLM configuration settings."""

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 120


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


class DocumentSettings(BaseModel):
    """Document processing configuration."""

    chunk_size: int = 500
    chunk_overlap: int = 100
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = ["pdf"]


class Settings:
    """Main application settings."""

    def __init__(self):
        # Application
        self.app_name: str = "RAG PDF Chat"
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # Services
        self.database = DatabaseSettings(
            url=os.getenv(
                "DATABASE_URL",
                "postgresql://rag_user:rag_password@localhost:5432/rag_pdf_chat",
            ),
            echo=self.debug,
        )

        self.vector = VectorSettings(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name="documents",
            vector_size=384,
        )

        self.llm = LLMSettings(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
            temperature=0.1,
            max_tokens=2048,
            timeout=120,
        )

        self.embedding = EmbeddingSettings(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), batch_size=32
        )

        self.document = DocumentSettings(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            allowed_extensions=["pdf"],
        )


# Global settings instance
settings = Settings()
