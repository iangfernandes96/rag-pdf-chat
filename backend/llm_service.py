"""
LLM service for generating responses using Ollama.
Handles prompt formatting, model interaction, and response generation.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from functools import wraps
from typing import Any

import httpx

from .config import settings
from .redis_cache import cache_service
from .utils.timing import time_async_function

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom LLM error with context."""

    pass


class ConfigValidator:
    """Validate LLM configuration."""

    @staticmethod
    def validate_llm_config(config) -> tuple[bool, str | None]:
        """
        Validate LLM configuration.

        Args:
            config: LLM configuration object

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not config.ollama_url:
            return False, "Ollama URL not configured"

        if not config.ollama_model:
            return False, "Ollama model not configured"

        if config.timeout <= 0:
            return False, "Invalid timeout value"

        if config.max_tokens <= 0:
            return False, "Invalid max_tokens value"

        try:
            from urllib.parse import urlparse

            parsed = urlparse(config.ollama_url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid Ollama URL format"
        except Exception:
            return False, "Invalid Ollama URL"

        return True, None


class OptimizedPromptTemplate:
    """Pre-compiled prompt templates for better performance."""

    RAG_SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions based on "
        "provided context from documents.\n\n"
        "Guidelines:\n"
        "- Use ONLY the provided context to answer questions\n"
        "- If the context doesn't contain enough information, say so clearly\n"
        "- Be precise and cite specific information from the context\n"
        "- Provide clear, well-structured answers\n"
        "- If multiple documents are referenced, acknowledge that in your response"
    )

    RAG_USER_TEMPLATE = (
        "Context from documents:\n{context}\n\n"
        "Question: {question}\n\n"
        "Please provide a comprehensive answer based on the context above. "
        "If the context doesn't contain sufficient information to answer "
        "the question, please state that clearly."
    )

    SIMPLE_RAG_TEMPLATE = (
        "Based on the following context from documents, "
        "please answer the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # Pre-compiled format strings for chunk formatting
    CHUNK_TEMPLATE = "[Document {doc_id}, Relevance: {score:.3f}]\n{content}"
    CONTEXT_TEMPLATE = "Context {index}:\n{chunk_text}"

    @staticmethod
    def format_rag_prompt(context: str, question: str) -> str:
        """Fast prompt formatting with pre-compiled template."""
        return OptimizedPromptTemplate.RAG_USER_TEMPLATE.format(
            context=context, question=question
        )

    @staticmethod
    def format_simple_rag_prompt(context: str, question: str) -> str:
        """Fast simple RAG prompt formatting."""
        return OptimizedPromptTemplate.SIMPLE_RAG_TEMPLATE.format(
            context=context, question=question
        )

    @staticmethod
    def format_context(chunks: list[dict[str, Any]]) -> str:
        """
        Format context chunks into a readable string using optimized operations.

        Args:
            chunks: List of context chunks with content

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        # Pre-allocate list with known size for better performance
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "").strip()
            if content:
                # Use f-string formatting for better performance
                context_parts.append(f"[Context {i}]\n{content}")

        return (
            "\n\n".join(context_parts)
            if context_parts
            else "No relevant context found."
        )


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(delay * (2**attempt))
                        continue
                    break
                except Exception:
                    # Don't retry on non-transient errors
                    raise

            raise LLMError(
                f"Operation failed after {max_retries} attempts"
            ) from last_exception

        return wrapper

    return decorator


class LLMService:
    """Optimized LLM service with caching and performance tuning."""

    def __init__(self):
        # Validate configuration first
        self.config = settings.llm
        is_valid, error_msg = ConfigValidator.validate_llm_config(self.config)
        if not is_valid:
            raise LLMError(f"Invalid LLM configuration: {error_msg}")

        # Initialize service properties
        self.ollama_url = self.config.ollama_url.rstrip("/")
        self.model_name = self.config.ollama_model
        self.timeout = self.config.timeout
        self.client: httpx.AsyncClient | None = None
        self.model_loaded = False
        self._available_models_cache: list[str] = []
        self._cache_timestamp = 0.0

        # Initialize response cache
        self.response_cache = cache_service

        # Performance-optimized base payload
        self._base_payload = {
            "model": self.model_name,
            "stream": False,
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": getattr(self.config, "top_p", 0.9),
                "top_k": getattr(self.config, "top_k", 40),
                "repeat_penalty": getattr(self.config, "repeat_penalty", 1.1),
                "num_ctx": getattr(self.config, "num_ctx", 4096),
                "num_thread": getattr(self.config, "num_thread", 4),
            },
        }

    async def initialize(self) -> bool:
        """
        Initialize the LLM service with optimized HTTP client.

        Returns:
            True if initialization successful
        """
        logger.info(f"Initializing LLM service with Ollama at {self.ollama_url}")

        try:
            # Create optimized HTTP client with connection pooling
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0, read=self.timeout, write=5.0, pool=2.0
                ),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                http2=True,  # Enable HTTP/2 for better performance
                headers={"User-Agent": "RAG-PDF-Chat/1.0"},
            )

            # Test Ollama connection
            await self._test_ollama_connection()

            # Ensure model is available with caching
            await self._ensure_model_available_cached()

            logger.info(f"✅ LLM service initialized with model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM service: {str(e)}")
            return False

    @retry_on_failure(max_retries=2, delay=1.0)
    async def _make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with automatic retry on transient failures."""
        if not self.client:
            raise LLMError("HTTP client not initialized")

        response = await getattr(self.client, method)(url, **kwargs)
        response.raise_for_status()
        return response

    async def _test_ollama_connection(self) -> None:
        """Test connection to Ollama API with retry logic."""
        try:
            await self._make_request("get", f"{self.ollama_url}/api/tags")
            logger.info("✅ Ollama connection successful")
        except Exception as e:
            raise LLMError(f"Failed to connect to Ollama: {str(e)}") from e

    async def _get_available_models_cached(self) -> list[str]:
        """
        Get available models with caching to reduce API calls.

        Returns:
            List of available model names
        """
        now = datetime.now(UTC)

        # Check cache first
        if (
            self._available_models_cache
            and now.timestamp() - self._cache_timestamp < 300  # 5 minutes
        ):
            return self._available_models_cache

        # Fetch from API
        response = await self._make_request("get", f"{self.ollama_url}/api/tags")
        models_data = response.json()
        models = [model["name"] for model in models_data.get("models", [])]

        # Cache result
        self._available_models_cache = models
        self._cache_timestamp = now.timestamp()
        return models

    async def _ensure_model_available_cached(self) -> None:
        """Ensure the specified model is available with caching."""
        try:
            available_models = await self._get_available_models_cached()

            if self.model_name not in available_models:
                logger.warning(
                    f"Model {self.model_name} not found. "
                    f"Available models: {available_models}"
                )
                logger.info(f"Attempting to pull model: {self.model_name}")

                # Clear cache and try once more after pulling
                self._available_models_cache = []
                self._cache_timestamp = 0.0
                await self._pull_model()
            else:
                logger.info(f"✅ Model {self.model_name} is available")
                self.model_loaded = True

        except Exception as e:
            raise LLMError(f"Failed to check model availability: {str(e)}") from e

    @retry_on_failure(max_retries=1, delay=2.0)
    async def _pull_model(self) -> None:
        """Pull model if not available locally."""
        try:
            logger.info(
                f"📥 Pulling model {self.model_name} (this may take a while)..."
            )

            await self._make_request(
                "post",
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 minutes for model pull
            )

            logger.info(f"✅ Successfully pulled model: {self.model_name}")
            self.model_loaded = True

        except Exception as e:
            raise LLMError(f"Failed to pull model {self.model_name}: {str(e)}") from e

    @time_async_function
    async def generate_rag_response(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        include_sources: bool = True,
        use_system_prompt: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a RAG response using context chunks with optimized processing.

        Args:
            query: User's question
            context_chunks: List of relevant document chunks
            include_sources: Whether to format response with source citations
            use_system_prompt: Whether to use system prompt format

        Returns:
            Dictionary with response data
        """
        if not self.client or not self.model_loaded:
            raise LLMError("LLM service not initialized")

        try:
            # Check cache first - use chunk IDs for consistent caching
            chunk_ids = [chunk.get("chunk_id", "") for chunk in context_chunks]
            cached_response = await self.response_cache.get(
                "llm_response", query, sorted(chunk_ids)
            )

            if cached_response:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return {
                    **cached_response,
                    "cached": True,
                    "generation_time": 0.0,
                }

            # Format context with optimized string operations
            context = OptimizedPromptTemplate.format_context(context_chunks)

            # Create messages with optimized templates
            if use_system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": OptimizedPromptTemplate.RAG_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": OptimizedPromptTemplate.format_rag_prompt(
                            context, query
                        ),
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": OptimizedPromptTemplate.format_simple_rag_prompt(
                            context, query
                        ),
                    }
                ]

            logger.info(f"Generating response for query: {query[:100]}...")

            # Generate response with optimized payload
            response_data = await self._generate_completion(messages)

            logger.info("✅ Response generated")

            result = {
                "success": True,
                "answer": response_data["response"],
                "model_used": self.model_name,
                "generation_time": 0.0,  # Will be provided by decorator
                "total_tokens": response_data.get("eval_count", 0),
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "context_chunks": len(context_chunks),
                "cached": False,
            }

            # Cache the response in Redis
            await self.response_cache.set(
                "llm_response", result, 3600, query, sorted(chunk_ids)
            )

            return result

        except Exception as e:
            logger.error(f"❌ Failed to generate RAG response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name,
                "generation_time": 0.0,  # Will be provided by decorator
                "cached": False,
            }

    async def _generate_completion(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Generate completion using optimized payload and error handling.

        Args:
            messages: List of messages for the conversation

        Returns:
            Raw response data from Ollama
        """
        try:
            # Use pre-built base payload for efficiency
            payload = self._base_payload.copy()
            payload["messages"] = messages

            response = await self._make_request(
                "post", f"{self.ollama_url}/api/chat", json=payload
            )

            data = response.json()

            # Extract the assistant's response
            if "message" in data and "content" in data["message"]:
                data["response"] = data["message"]["content"]
            else:
                raise ValueError("Invalid response format from Ollama")

            return data

        except Exception as e:
            raise LLMError(f"Request failed: {str(e)}") from e

    @time_async_function
    async def generate_simple_response(self, prompt: str) -> dict[str, Any]:
        """
        Generate a simple response without RAG context.

        Args:
            prompt: Simple prompt text

        Returns:
            Dictionary with response data
        """
        if not self.client or not self.model_loaded:
            raise LLMError("LLM service not initialized")

        try:
            messages = [{"role": "user", "content": prompt}]

            response_data = await self._generate_completion(messages)

            return {
                "success": True,
                "answer": response_data["response"],
                "model_used": self.model_name,
                "generation_time": 0.0,  # Will be provided by decorator
                "total_tokens": response_data.get("eval_count", 0),
            }

        except Exception as e:
            logger.error(f"Failed to generate simple response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name,
                "generation_time": 0.0,  # Will be provided by decorator
            }

    async def generate_rag_response_streaming(
        self, query: str, context_chunks: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Generate streaming RAG response for real-time UI updates.

        Args:
            query: User's question
            context_chunks: List of relevant document chunks

        Yields:
            Dictionary with streaming response chunks
        """
        if not self.client or not self.model_loaded:
            raise LLMError("LLM service not initialized")

        try:
            # Format context and create messages
            context = OptimizedPromptTemplate.format_context(context_chunks)
            messages = [
                {
                    "role": "system",
                    "content": OptimizedPromptTemplate.RAG_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": OptimizedPromptTemplate.format_rag_prompt(
                        context, query
                    ),
                },
            ]

            # Create streaming payload
            payload = self._base_payload.copy()
            payload["messages"] = messages
            payload["stream"] = True

            async with self.client.stream(
                "POST", f"{self.ollama_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk_data = json.loads(line)
                            if "message" in chunk_data:
                                yield {
                                    "chunk": chunk_data["message"]["content"],
                                    "done": chunk_data.get("done", False),
                                }
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Failed to generate streaming response: {str(e)}")
            yield {"error": str(e), "done": True}

    async def get_status(self) -> dict[str, Any]:
        """
        Get LLM service status and model information.

        Returns:
            Status information dictionary with unified health check format
        """
        try:
            if not self.client:
                return {
                    "healthy": False,
                    "status": "not_initialized",
                    "error": "Client not initialized",
                    "details": {
                        "model": self.model_name,
                        "url": self.ollama_url,
                    },
                }

            # Use cached model list for status
            available_models = await self._get_available_models_cached()

            return {
                "healthy": True,
                "status": "healthy",
                "error": None,
                "details": {
                    "model": self.model_name,
                    "model_loaded": self.model_loaded,
                    "available_models": available_models,
                    "url": self.ollama_url,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "timeout": self.config.timeout,
                    "cache_status": {
                        "cached_entries": 0,  # Redis cache doesn't have direct count
                        "cache_ttl_minutes": 60,  # 1 hour default
                    },
                },
            }

        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "details": {
                    "model": self.model_name,
                    "url": self.ollama_url,
                },
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            self.client = None

        # Clear cache
        self._available_models_cache = []
        self._cache_timestamp = 0.0

        logger.info("✅ LLM service cleanup completed")

    async def test_model_response(
        self, test_prompt: str = "Hello, how are you?"
    ) -> dict[str, Any]:
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
                answer_preview = result["answer"][:100]
                logger.info(f"✅ Model test successful: {answer_preview}...")
            else:
                logger.error(f"❌ Model test failed: {result['error']}")

            return result

        except Exception as e:
            logger.error(f"❌ Model test exception: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name,
            }
