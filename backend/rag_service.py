"""
RAG (Retrieval-Augmented Generation) service that orchestrates 
document processing, embedding, and similarity search.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .document_ingestion import DocumentIngestionService
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .models import Document, DocumentChunk, ProcessingResult
from .config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Main service that orchestrates the RAG pipeline."""
    
    def __init__(self):
        self.ingestion_service = DocumentIngestionService()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        
    async def initialize(self) -> bool:
        """
        Initialize all services (load models, connect to databases).
        
        Returns:
            True if initialization successful
        """
        logger.info("Initializing RAG service...")
        
        try:
            # Load embedding model
            logger.info("Loading embedding model...")
            self.embedding_service.load_model()
            
            # Connect to vector store
            logger.info("Connecting to vector store...")
            await self.vector_store.connect()
            
            # Create collection if it doesn't exist
            logger.info("Setting up vector collection...")
            await self.vector_store.create_collection()
            
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            return False
    
    async def process_document(self, 
                              file_path: Path, 
                              original_filename: str) -> Dict[str, Any]:
        """
        Process a document through the complete RAG pipeline.
        
        Args:
            file_path: Path to the document file
            original_filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing document: {original_filename}")
        
        try:
            # Step 1: Ingest document (PDF parsing and chunking)
            ingestion_result = self.ingestion_service.ingest_document(
                file_path, original_filename
            )
            
            if not ingestion_result.success:
                return {
                    "success": False,
                    "error": f"Document ingestion failed: {ingestion_result.error_message}",
                    "stage": "ingestion"
                }
            
            document = ingestion_result.document
            chunks = ingestion_result.chunks
            
            # Step 2: Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embedding_results = self.embedding_service.embed_chunks(chunks)
            
            if len(embedding_results) != len(chunks):
                return {
                    "success": False,
                    "error": "Embedding generation failed: mismatch in chunk count",
                    "stage": "embedding"
                }
            
            # Step 3: Store embeddings in vector database
            embeddings = [result.embedding for result in embedding_results]
            document_metadata = {
                "filename": document.filename,
                "original_filename": document.original_filename,
                "file_size": document.file_size,
                "page_count": document.page_count,
                "uploaded_at": document.uploaded_at.isoformat(),
            }
            
            logger.info("Storing embeddings in vector database")
            storage_success = await self.vector_store.store_embeddings(
                chunks, embeddings, document_metadata
            )
            
            if not storage_success:
                return {
                    "success": False,
                    "error": "Failed to store embeddings in vector database",
                    "stage": "storage"
                }
            
            # Step 4: Update chunks with embeddings
            for chunk, embedding_result in zip(chunks, embedding_results):
                chunk.embedding = embedding_result.embedding
            
            logger.info(f"Successfully processed document: {original_filename}")
            
            return {
                "success": True,
                "document": document,
                "chunks": chunks,
                "embeddings_count": len(embeddings),
                "processing_time": ingestion_result.processing_time,
                "stats": self.ingestion_service.get_document_stats(document, chunks)
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "stage": "unknown"
            }
    
    async def search_documents(self, 
                              query: str, 
                              limit: int = 5,
                              score_threshold: float = 0.1,
                              document_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for relevant document chunks using semantic similarity.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            document_filter: Optional document ID to filter by
            
        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching documents for query: '{query[:50]}...'")
        
        try:
            # Generate embedding for query
            query_embedding, embed_time = self.embedding_service.generate_embedding(query)
            
            # Search similar chunks
            similar_chunks = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                document_filter=document_filter
            )
            
            logger.info(f"Found {len(similar_chunks)} relevant chunks")
            
            # Enhance results with additional context if available
            enhanced_results = []
            for chunk_data in similar_chunks:
                enhanced_result = {
                    **chunk_data,
                    "relevance_score": chunk_data["score"],
                    "snippet": self._create_snippet(chunk_data["content"], query)
                }
                enhanced_results.append(enhanced_result)
            
            return {
                "success": True,
                "query": query,
                "results": enhanced_results,
                "total_results": len(enhanced_results),
                "embed_time": embed_time,
                "search_params": {
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "document_filter": document_filter
                }
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "query": query
            }
    
    async def get_document_context(self, 
                                  chunk_ids: List[str], 
                                  context_size: int = 1) -> List[str]:
        """
        Get expanded context for specific chunks.
        
        Args:
            chunk_ids: List of chunk IDs to get context for
            context_size: Number of surrounding chunks to include
            
        Returns:
            List of context strings
        """
        contexts = []
        
        for chunk_id in chunk_ids:
            chunk_data = await self.vector_store.get_chunk_by_id(chunk_id)
            if chunk_data:
                # For now, return the chunk content
                # In a full implementation, we'd get surrounding chunks
                contexts.append(chunk_data["content"])
            else:
                contexts.append("")
        
        return contexts
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the system.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful
        """
        logger.info(f"Deleting document: {document_id}")
        
        try:
            # Delete from vector store
            success = await self.vector_store.delete_document_chunks(document_id)
            
            if success:
                logger.info(f"Successfully deleted document: {document_id}")
            else:
                logger.warning(f"Failed to delete document: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all system components.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Get embedding model info
            embedding_info = self.embedding_service.get_model_info()
            
            # Get vector store info
            vector_info = await self.vector_store.get_collection_info()
            
            # Check health
            vector_healthy = await self.vector_store.health_check()
            
            return {
                "embedding_service": embedding_info,
                "vector_store": {
                    **vector_info,
                    "healthy": vector_healthy,
                    "url": self.vector_store.url
                },
                "system_healthy": embedding_info.get("loaded", False) and vector_healthy
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {
                "error": str(e),
                "system_healthy": False
            }
    
    def _create_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """
        Create a highlighted snippet from content based on query.
        
        Args:
            content: Full content text
            query: Search query
            max_length: Maximum snippet length
            
        Returns:
            Snippet with query context
        """
        if len(content) <= max_length:
            return content
        
        # Simple snippet creation - find query terms in content
        query_words = query.lower().split()
        content_lower = content.lower()
        
        # Find best position to start snippet
        best_pos = 0
        max_matches = 0
        
        for i in range(0, len(content) - max_length, 20):
            snippet = content_lower[i:i + max_length]
            matches = sum(1 for word in query_words if word in snippet)
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        snippet = content[best_pos:best_pos + max_length]
        
        # Clean up snippet boundaries
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + max_length < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.embedding_service.cleanup()
            logger.info("RAG service cleaned up")
        except Exception as e:
            logger.error(f"Error during RAG service cleanup: {str(e)}") 