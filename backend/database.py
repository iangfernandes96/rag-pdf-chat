"""
Database service for PostgreSQL operations.
Handles document metadata storage, user information, and session management.
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio

import asyncpg
from asyncpg import Pool, Connection

from .config import settings
from .models import Document, DocumentChunk

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for PostgreSQL database operations."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.connection_url = settings.database.url
        
    async def initialize(self) -> bool:
        """
        Initialize database connection pool and create tables.
        
        Returns:
            True if initialization successful
        """
        logger.info("Initializing database service...")
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"✅ Connected to PostgreSQL: {result}")
            
            # Create tables
            await self._create_tables()
            
            logger.info("✅ Database service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {str(e)}")
            return False
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        async with self.pool.acquire() as conn:
            # Documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    file_size BIGINT NOT NULL,
                    content TEXT,
                    page_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    processing_status VARCHAR(50) DEFAULT 'pending',
                    error_message TEXT,
                    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    processed_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)
            
            # Document chunks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    embedding_stored BOOLEAN DEFAULT FALSE,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(document_id, chunk_index)
                );
            """)
            
            # Query sessions table (for tracking user interactions)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255),
                    query_text TEXT NOT NULL,
                    response_text TEXT,
                    model_used VARCHAR(100),
                    chunks_used INTEGER DEFAULT 0,
                    response_time FLOAT DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    document_filters UUID[]
                );
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_status 
                ON documents(processing_status);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                ON document_chunks(document_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_stored 
                ON document_chunks(embedding_stored);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_sessions_created_at 
                ON query_sessions(created_at);
            """)
            
            logger.info("✅ Database tables created/verified")
    
    async def store_document_metadata(self, 
                                    document: Document, 
                                    chunks: List[DocumentChunk]) -> bool:
        """
        Store document and chunk metadata in database.
        
        Args:
            document: Document object with metadata
            chunks: List of document chunks
            
        Returns:
            True if storage successful
        """
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Insert document
                    await conn.execute("""
                        INSERT INTO documents (
                            id, filename, original_filename, file_size,
                            content, page_count, chunk_count, processing_status,
                            uploaded_at, processed_at, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (id) DO UPDATE SET
                            chunk_count = EXCLUDED.chunk_count,
                            processing_status = EXCLUDED.processing_status,
                            processed_at = EXCLUDED.processed_at
                    """, 
                        document.id, document.filename, document.original_filename,
                        document.file_size, document.content, document.page_count,
                        len(chunks), document.processing_status,
                        document.uploaded_at, document.processed_at,
                        json.dumps(document.metadata or {})
                    )
                    
                    # Insert chunks
                    for chunk in chunks:
                        await conn.execute("""
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content,
                                start_char, end_char, token_count, metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                                content = EXCLUDED.content,
                                start_char = EXCLUDED.start_char,
                                end_char = EXCLUDED.end_char,
                                token_count = EXCLUDED.token_count,
                                metadata = EXCLUDED.metadata
                        """,
                            chunk.id, chunk.document_id, chunk.chunk_index,
                            chunk.content, chunk.start_char, chunk.end_char,
                            chunk.token_count, json.dumps(chunk.metadata or {})
                        )
            
            logger.info(f"✅ Stored document {document.id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document metadata: {str(e)}")
            return False
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID.
        
        Args:
            document_id: Document UUID
            
        Returns:
            Document data or None if not found
        """
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM documents WHERE id = $1
                """, document_id)
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents with metadata.
        
        Returns:
            List of document dictionaries
        """
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        id, filename, original_filename, file_size,
                        page_count, chunk_count, processing_status,
                        uploaded_at, processed_at, metadata
                    FROM documents 
                    ORDER BY uploaded_at DESC
                """)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get all documents: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document and all associated chunks.
        
        Args:
            document_id: Document UUID
            
        Returns:
            True if deletion successful
        """
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Delete chunks (will cascade from document deletion)
                    chunk_result = await conn.execute("""
                        DELETE FROM document_chunks WHERE document_id = $1
                    """, document_id)
                    
                    # Delete document
                    doc_result = await conn.execute("""
                        DELETE FROM documents WHERE id = $1
                    """, document_id)
                    
                    deleted = doc_result.split()[-1] == "1"
                    
                    if deleted:
                        logger.info(f"✅ Deleted document {document_id}")
                    else:
                        logger.warning(f"Document {document_id} not found for deletion")
                    
                    return deleted
                    
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def log_query_session(self, 
                              query: str, 
                              response: str,
                              model_used: str,
                              chunks_used: int = 0,
                              response_time: float = 0.0,
                              session_id: Optional[str] = None,
                              document_filters: Optional[List[str]] = None) -> bool:
        """
        Log a query session for analytics.
        
        Args:
            query: User query text
            response: Generated response
            model_used: Name of the model used
            chunks_used: Number of chunks used
            response_time: Response generation time
            session_id: Optional session identifier
            document_filters: Optional list of document IDs used
            
        Returns:
            True if logging successful
        """
        if not self.pool:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO query_sessions (
                        session_id, query_text, response_text, model_used,
                        chunks_used, response_time, document_filters
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    session_id, query[:1000], response[:5000], model_used,
                    chunks_used, response_time, document_filters or []
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log query session: {str(e)}")
            return False
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Statistics dictionary
        """
        if not self.pool:
            return {}
        
        try:
            async with self.pool.acquire() as conn:
                # Document statistics
                doc_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(file_size) as total_size,
                        SUM(page_count) as total_pages,
                        SUM(chunk_count) as total_chunks,
                        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as processed_docs,
                        COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_docs
                    FROM documents
                """)
                
                # Query statistics
                query_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(response_time) as avg_response_time,
                        AVG(chunks_used) as avg_chunks_used
                    FROM query_sessions
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                
                return {
                    "documents": dict(doc_stats) if doc_stats else {},
                    "queries": dict(query_stats) if query_stats else {},
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            return {}
    
    async def mark_chunks_embedded(self, chunk_ids: List[str]) -> bool:
        """
        Mark chunks as having embeddings stored.
        
        Args:
            chunk_ids: List of chunk UUIDs
            
        Returns:
            True if update successful
        """
        if not self.pool or not chunk_ids:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE document_chunks 
                    SET embedding_stored = TRUE 
                    WHERE id = ANY($1)
                """, chunk_ids)
            
            logger.info(f"✅ Marked {len(chunk_ids)} chunks as embedded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark chunks as embedded: {str(e)}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get database service status.
        
        Returns:
            Status information dictionary
        """
        try:
            if not self.pool:
                return {
                    "healthy": False,
                    "error": "Database pool not initialized"
                }
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                
                # Get connection info
                server_version = await conn.fetchval("SELECT version()")
                
            return {
                "healthy": True,
                "connection_url": self.connection_url.split('@')[-1],  # Hide credentials
                "server_version": server_version,
                "pool_size": len(self.pool._holders),
                "pool_max_size": self.pool._maxsize
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "connection_url": self.connection_url.split('@')[-1]
            }
    
    async def cleanup(self) -> None:
        """Clean up database connections."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("✅ Database service cleanup completed")


async def get_db_session() -> AsyncGenerator[Connection, None]:
    """
    Dependency for getting database connection in FastAPI endpoints.
    
    Yields:
        Database connection
    """
    # This would be used as a FastAPI dependency
    # For now, we'll use the global database service
    pass 