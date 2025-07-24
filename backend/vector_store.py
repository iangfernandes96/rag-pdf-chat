"""
Async vector store service using Qdrant for embedding storage and similarity search.
"""
import logging
from typing import List, Optional, Dict, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Match, FilterSelector
)

from .models import DocumentChunk
from .config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Async vector store service using Qdrant."""
    
    def __init__(self, url: Optional[str] = None, 
                 collection_name: Optional[str] = None):
        self.url = url or settings.vector.qdrant_url
        self.collection_name = collection_name or settings.vector.collection_name
        self.vector_size = settings.vector.vector_size
        self.client: Optional[AsyncQdrantClient] = None
        
    async def connect(self) -> None:
        """Connect to Qdrant database."""
        if self.client is not None:
            logger.info("Already connected to Qdrant")
            return
            
        logger.info(f"Connecting to Qdrant at {self.url}")
        
        try:
            self.client = AsyncQdrantClient(url=self.url)
            
            # Test connection
            collections = await self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
            logger.info(f"Available collections: {len(collections.collections)}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise RuntimeError(f"Qdrant connection failed: {str(e)}")
    
    async def create_collection(self, overwrite: bool = False) -> bool:
        """
        Create a collection for storing document embeddings.
        
        Args:
            overwrite: Whether to overwrite existing collection
            
        Returns:
            True if collection was created, False if already exists
        """
        if self.client is None:
            await self.connect()
            
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if overwrite:
                    logger.info(
                        f"Deleting existing collection: {self.collection_name}"
                    )
                    await self.client.delete_collection(
                        collection_name=self.collection_name
                    )
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return False
            
            # Create collection
            logger.info(f"Creating collection: {self.collection_name}")
            logger.info(f"Vector size: {self.vector_size}")
            
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise RuntimeError(f"Collection creation failed: {str(e)}")
    
    async def store_embeddings(self, 
                              chunks: List[DocumentChunk], 
                              embeddings: List[List[float]],
                              document_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store document chunks with their embeddings.
        
        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors
            document_metadata: Optional document-level metadata
            
        Returns:
            True if successful
        """
        if self.client is None:
            await self.connect()
            
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        logger.info(f"Storing {len(chunks)} chunks with embeddings")
        
        try:
            points = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Prepare metadata
                payload = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
                
                # Add document metadata if provided
                if document_metadata:
                    payload["document_metadata"] = document_metadata
                
                # Create point
                point = PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload points to Qdrant
            operation_info = await self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            
            logger.info(f"Successfully stored {len(points)} embeddings")
            logger.info(f"Operation status: {operation_info.status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise RuntimeError(f"Embedding storage failed: {str(e)}")
    
    async def search_similar(self, 
                            query_embedding: List[float], 
                            limit: int = 5,
                            score_threshold: float = 0.0,
                            document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            document_filter: Optional document ID to filter by
            
        Returns:
            List of similar chunks with scores and metadata
        """
        if self.client is None:
            await self.connect()
            
        logger.info(f"Searching for {limit} similar chunks")
        
        try:
            # Prepare filter if document_filter is provided
            search_filter = None
            if document_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=Match(value=document_filter)
                        )
                    ]
                )
            
            # Perform search
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    "chunk_id": scored_point.id,
                    "score": scored_point.score,
                    "content": scored_point.payload.get("content", ""),
                    "document_id": scored_point.payload.get("document_id", ""),
                    "chunk_index": scored_point.payload.get("chunk_index", 0),
                    "start_char": scored_point.payload.get("start_char", 0),
                    "end_char": scored_point.payload.get("end_char", 0),
                    "metadata": scored_point.payload.get("metadata", {}),
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            raise RuntimeError(f"Similarity search failed: {str(e)}")
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data if found, None otherwise
        """
        if self.client is None:
            await self.connect()
            
        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                return None
            
            point = points[0]
            return {
                "chunk_id": point.id,
                "content": point.payload.get("content", ""),
                "document_id": point.payload.get("document_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "start_char": point.payload.get("start_char", 0),
                "end_char": point.payload.get("end_char", 0),
                "metadata": point.payload.get("metadata", {}),
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
            return None
    
    async def delete_document_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of the document whose chunks to delete
            
        Returns:
            True if successful
        """
        if self.client is None:
            await self.connect()
            
        try:
            # Delete points with matching document_id using simple approach
            operation_info = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True
            )
            
            logger.info(f"Deleted chunks for document {document_id}")
            logger.info(f"Operation status: {operation_info.status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        if self.client is None:
            await self.connect()
            
        try:
            collection_info = await self.client.get_collection(
                collection_name=self.collection_name
            )
            
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and accessible.
        
        Returns:
            True if healthy
        """
        try:
            if self.client is None:
                await self.connect()
            
            # Simple health check
            await self.client.get_collections()
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close the client connection."""
        if self.client is not None:
            await self.client.close()
            self.client = None
            logger.info("Qdrant client connection closed") 