"""
LLM service for generating responses using Ollama.
Handles prompt formatting, model interaction, and response generation.
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Templates for RAG prompts."""
    
    RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context from documents. 

Guidelines:
- Use ONLY the provided context to answer questions
- If the context doesn't contain enough information, say so clearly
- Be precise and cite specific information from the context
- Provide clear, well-structured answers
- If multiple documents are referenced, acknowledge that in your response"""

    RAG_USER_PROMPT = """Context from documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question, please state that clearly."""

    SIMPLE_RAG_PROMPT = """Based on the following context from documents, please answer the user's question.

Context:
{context}

Question: {question}

Answer:"""


class LLMService:
    """Service for interacting with local LLM via Ollama."""
    
    def __init__(self):
        self.ollama_url = settings.llm.ollama_url
        self.model_name = settings.llm.ollama_model
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature
        self.timeout = settings.llm.timeout
        self.client: Optional[httpx.AsyncClient] = None
        self.model_loaded = False
        
    async def initialize(self) -> bool:
        """
        Initialize the LLM service and check Ollama availability.
        
        Returns:
            True if initialization successful
        """
        logger.info(f"Initializing LLM service with Ollama at {self.ollama_url}")
        
        try:
            # Create HTTP client
            self.client = httpx.AsyncClient(timeout=self.timeout)
            
            # Test Ollama connection
            await self._test_ollama_connection()
            
            # Ensure model is available
            await self._ensure_model_available()
            
            logger.info(f"✅ LLM service initialized with model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM service: {str(e)}")
            return False
    
    async def _test_ollama_connection(self) -> None:
        """Test connection to Ollama API."""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            logger.info("✅ Ollama connection successful")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {str(e)}")
    
    async def _ensure_model_available(self) -> None:
        """Ensure the specified model is available in Ollama."""
        try:
            # List available models
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                logger.info(f"Attempting to pull model: {self.model_name}")
                await self._pull_model()
            else:
                logger.info(f"✅ Model {self.model_name} is available")
                self.model_loaded = True
                
        except Exception as e:
            raise RuntimeError(f"Failed to check model availability: {str(e)}")
    
    async def _pull_model(self) -> None:
        """Pull model if not available locally."""
        try:
            logger.info(f"📥 Pulling model {self.model_name} (this may take a while)...")
            
            response = await self.client.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes for model pull
            )
            response.raise_for_status()
            
            logger.info(f"✅ Successfully pulled model: {self.model_name}")
            self.model_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to pull model {self.model_name}: {str(e)}")
    
    async def generate_rag_response(self, 
                                  query: str, 
                                  context_chunks: List[Dict[str, Any]],
                                  include_sources: bool = True,
                                  use_system_prompt: bool = True) -> Dict[str, Any]:
        """
        Generate a RAG response using context chunks.
        
        Args:
            query: User's question
            context_chunks: List of relevant document chunks
            include_sources: Whether to format response with source citations
            use_system_prompt: Whether to use system prompt format
            
        Returns:
            Dictionary with response data
        """
        if not self.client or not self.model_loaded:
            raise RuntimeError("LLM service not initialized")
        
        start_time = time.time()
        
        try:
            # Format context from chunks
            context = self._format_context(context_chunks)
            
            # Create prompt
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": PromptTemplate.RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": PromptTemplate.RAG_USER_PROMPT.format(
                        context=context, question=query
                    )}
                ]
            else:
                messages = [
                    {"role": "user", "content": PromptTemplate.SIMPLE_RAG_PROMPT.format(
                        context=context, question=query
                    )}
                ]
            
            logger.info(f"Generating response for query: {query[:100]}...")
            
            # Generate response
            response_data = await self._generate_completion(messages)
            
            generation_time = time.time() - start_time
            
            logger.info(f"✅ Response generated in {generation_time:.2f}s")
            
            return {
                "success": True,
                "answer": response_data["response"],
                "model_used": self.model_name,
                "generation_time": generation_time,
                "total_tokens": response_data.get("eval_count", 0),
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "context_chunks": len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate RAG response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name,
                "generation_time": time.time() - start_time
            }
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks for prompt inclusion.
        
        Args:
            chunks: List of document chunks with content and metadata
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "").strip()
            score = chunk.get("score", 0)
            doc_id = chunk.get("document_id", "unknown")
            
            # Format each chunk with metadata
            chunk_text = f"[Document {doc_id[:8]}, Relevance: {score:.3f}]\n{content}"
            context_parts.append(f"Context {i}:\n{chunk_text}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate completion using Ollama API.
        
        Args:
            messages: List of messages for the conversation
            
        Returns:
            Raw response data from Ollama
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            response = await self.client.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract the assistant's response
            if "message" in data and "content" in data["message"]:
                data["response"] = data["message"]["content"]
            else:
                raise ValueError("Invalid response format from Ollama")
            
            return data
            
        except httpx.TimeoutException:
            raise RuntimeError(f"Request timeout after {self.timeout} seconds")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Request failed: {str(e)}")
    
    async def generate_simple_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a simple response without RAG context.
        
        Args:
            prompt: Simple prompt text
            
        Returns:
            Dictionary with response data
        """
        if not self.client or not self.model_loaded:
            raise RuntimeError("LLM service not initialized")
        
        start_time = time.time()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response_data = await self._generate_completion(messages)
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "answer": response_data["response"],
                "model_used": self.model_name,
                "generation_time": generation_time,
                "total_tokens": response_data.get("eval_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate simple response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name,
                "generation_time": time.time() - start_time
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get LLM service status and model information.
        
        Returns:
            Status information dictionary
        """
        try:
            if not self.client:
                return {
                    "healthy": False,
                    "error": "Client not initialized",
                    "model": self.model_name,
                    "url": self.ollama_url
                }
            
            # Test connection
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            return {
                "healthy": True,
                "model": self.model_name,
                "model_loaded": self.model_loaded,
                "available_models": available_models,
                "url": self.ollama_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "model": self.model_name,
                "url": self.ollama_url
            }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("✅ LLM service cleanup completed")
    
    async def test_model_response(self, test_prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Test model response with a simple prompt.
        
        Args:
            test_prompt: Simple test prompt
            
        Returns:
            Test response data
        """
        logger.info(f"Testing model with prompt: {test_prompt}")
        
        try:
            result = await self.generate_simple_response(test_prompt)
            
            if result["success"]:
                logger.info(f"✅ Model test successful: {result['answer'][:100]}...")
            else:
                logger.error(f"❌ Model test failed: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Model test exception: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name
            } 