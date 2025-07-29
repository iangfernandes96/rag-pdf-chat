"""
LLM service for generating responses using Google Gemini.
Handles prompt formatting, model interaction, and response generation.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from functools import wraps
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

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
        if not config.gemini_api_key:
            return False, "Gemini API key not configured"

        if not config.gemini_model:
            return False, "Gemini model not configured"

        if config.timeout <= 0:
            return False, "Invalid timeout value"

        if config.max_tokens <= 0:
            return False, "Invalid max_tokens value"

        return True, None


class OptimizedPromptTemplate:
    """Pre-compiled prompt templates optimized for Gemini."""

    RAG_SYSTEM_PROMPT = (
        "You are an expert document analysis assistant. Your role is to provide "
        "accurate, detailed answers based on the provided document context.\n\n"
        "IMPORTANT GUIDELINES:\n"
        "1. Base your answer ONLY on the provided context\n"
        "2. If the context is insufficient, clearly state what information is missing\n"
        "3. Provide specific citations and references from the context\n"
        "4. Structure your response logically with clear sections\n"
        "5. Include relevant quotes or paraphrases from the source material\n"
        "6. If multiple documents are referenced, distinguish between them\n"
        "7. Be thorough but concise - aim for comprehensive coverage\n"
        "8. Use bullet points or numbered lists when appropriate for clarity"
    )

    RAG_USER_TEMPLATE = (
        "DOCUMENT CONTEXT:\n{context}\n\n"
        "USER QUESTION: {question}\n\n"
        "INSTRUCTIONS:\n"
        "- Analyze the provided context thoroughly\n"
        "- Provide a comprehensive answer with specific details\n"
        "- Include relevant quotes or references from the documents\n"
        "- If information is missing, clearly state what cannot be answered\n"
        "- Structure your response for maximum clarity\n\n"
        "ANSWER:"
    )

    SIMPLE_RAG_TEMPLATE = (
        "Based on the following document context, provide a detailed answer to the user's question.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "Provide a comprehensive response with specific details from the context:"
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
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(delay * (2**attempt))
                        continue
                    break

            raise LLMError(
                f"Operation failed after {max_retries} attempts"
            ) from last_exception

        return wrapper

    return decorator


class LLMService:
    """Optimized LLM service with caching and performance tuning using Gemini."""

    def __init__(self):
        # Validate configuration first
        self.config = settings.llm
        is_valid, error_msg = ConfigValidator.validate_llm_config(self.config)
        if not is_valid:
            raise LLMError(f"Invalid LLM configuration: {error_msg}")

        # Initialize service properties
        self.model_name = self.config.gemini_model
        self.timeout = self.config.timeout
        self.model_loaded = False
        self._available_models_cache: list[str] = []
        self._cache_timestamp = 0.0

        # Initialize response cache
        self.response_cache = cache_service

        # Optimized generation config for better quality
        self._generation_config = genai.types.GenerationConfig(
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.8,  # Slightly lower for more deterministic output
            top_k=40,  # Keep reasonable diversity
            max_output_tokens=8192,  # Allow longer, more detailed responses
            candidate_count=1,  # Single response for consistency
        )

    async def initialize(self) -> bool:
        """
        Initialize the LLM service with Gemini API.

        Returns:
            True if initialization successful
        """
        logger.info(f"Initializing LLM service with Gemini model: {self.model_name}")

        try:
            # Configure Gemini API
            genai.configure(api_key=self.config.gemini_api_key)

            # Test Gemini connection
            await self._test_gemini_connection()

            # Ensure model is available
            await self._ensure_model_available()

            logger.info(f"✅ LLM service initialized with model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM service: {str(e)}")
            return False

    @retry_on_failure(max_retries=2, delay=1.0)
    async def _test_gemini_connection(self) -> None:
        """Test connection to Gemini API with retry logic."""
        try:
            # Run in thread pool to make synchronous call non-blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: genai.list_models())
            logger.info("✅ Gemini connection successful")
        except Exception as e:
            raise LLMError(f"Failed to connect to Gemini: {str(e)}") from e

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
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, genai.list_models)
        model_names = [model.name for model in models]

        # Cache result
        self._available_models_cache = model_names
        self._cache_timestamp = now.timestamp()
        return model_names

    async def _ensure_model_available(self) -> None:
        """Ensure the specified model is available."""
        try:
            available_models = await self._get_available_models_cached()

            # Check if our model is in the available models
            model_found = any(self.model_name in model for model in available_models)

            if not model_found:
                logger.warning(
                    f"Model {self.model_name} not found. "
                    f"Available models: {available_models[:5]}..."  # Show first 5
                )
                # For Gemini, we'll try to use the model anyway as it might be available
                # even if not in the list
                logger.info(f"Attempting to use model: {self.model_name}")
            else:
                logger.info(f"✅ Model {self.model_name} is available")

            self.model_loaded = True

        except Exception as e:
            raise LLMError(f"Failed to check model availability: {str(e)}") from e

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
        if not self.model_loaded:
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

            # Create prompt with optimized templates
            if use_system_prompt:
                prompt = (
                    f"{OptimizedPromptTemplate.RAG_SYSTEM_PROMPT}\n\n"
                    f"{OptimizedPromptTemplate.format_rag_prompt(context, query)}"
                )
            else:
                prompt = OptimizedPromptTemplate.format_simple_rag_prompt(
                    context, query
                )

            logger.info(f"Generating response for query: {query[:100]}...")

            # Generate response with optimized payload
            response_data = await self._generate_completion(prompt)

            logger.info("✅ Response generated")

            result = {
                "success": True,
                "answer": response_data["response"],
                "model_used": self.model_name,
                "generation_time": 0.0,
                "total_tokens": response_data.get("usage_metadata", {}).get(
                    "total_token_count", 0
                ),
                "prompt_tokens": response_data.get("usage_metadata", {}).get(
                    "prompt_token_count", 0
                ),
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
                "generation_time": 0.0,
                "cached": False,
            }

    async def _generate_completion(self, prompt: str) -> dict[str, Any]:
        """
        Generate completion using Gemini API in a non-blocking manner.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Raw response data from Gemini
        """
        try:
            # Get the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self._generation_config,
            )

            # Run in thread pool to make synchronous call non-blocking
            loop = asyncio.get_event_loop()
            response: GenerateContentResponse = await loop.run_in_executor(
                None, lambda: model.generate_content(prompt)
            )

            # Extract the response text
            if response.text:
                # Handle usage metadata safely
                usage_metadata = {}
                try:
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage_metadata = {
                            "total_token_count": getattr(
                                response.usage_metadata, "total_token_count", 0
                            ),
                            "prompt_token_count": getattr(
                                response.usage_metadata, "prompt_token_count", 0
                            ),
                        }
                except Exception:
                    # If usage metadata is not available, use defaults
                    usage_metadata = {
                        "total_token_count": 0,
                        "prompt_token_count": 0,
                    }

                return {"response": response.text, "usage_metadata": usage_metadata}
            else:
                raise ValueError("No response text from Gemini")

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
        if not self.model_loaded:
            raise LLMError("LLM service not initialized")

        try:
            response_data = await self._generate_completion(prompt)

            return {
                "success": True,
                "answer": response_data["response"],
                "model_used": self.model_name,
                "generation_time": 0.0,
                "total_tokens": response_data.get("usage_metadata", {}).get(
                    "total_token_count", 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to generate simple response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model_name,
                "generation_time": 0.0,
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
        if not self.model_loaded:
            raise LLMError("LLM service not initialized")

        try:
            # Format context and create prompt
            context = OptimizedPromptTemplate.format_context(context_chunks)
            prompt = (
                f"{OptimizedPromptTemplate.RAG_SYSTEM_PROMPT}\n\n"
                f"{OptimizedPromptTemplate.format_rag_prompt(context, query)}"
            )

            # Get the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self._generation_config,
            )

            def generate_stream():
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield {"chunk": chunk.text, "done": False}
                yield {"chunk": "", "done": True}

            # Execute streaming in thread pool
            async for chunk_data in self._async_stream_generator(generate_stream):
                yield chunk_data

        except Exception as e:
            logger.error(f"Failed to generate streaming response: {str(e)}")
            yield {"error": str(e), "done": True}

    async def _async_stream_generator(self, sync_generator):
        """Convert synchronous generator to asynchronous."""
        loop = asyncio.get_event_loop()

        def run_generator():
            yield from sync_generator()

        # Run the synchronous generator in thread pool
        for item in await loop.run_in_executor(None, lambda: list(run_generator())):
            yield item

    async def get_status(self) -> dict[str, Any]:
        """
        Get LLM service status and model information.

        Returns:
            Status information dictionary with unified health check format
        """
        try:
            if not self.model_loaded:
                return {
                    "healthy": False,
                    "status": "not_initialized",
                    "error": "Service not initialized",
                    "details": {
                        "model": self.model_name,
                        "provider": "gemini",
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
                    "available_models": available_models[:10],  # Show first 10
                    "provider": "gemini",
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
                    "provider": "gemini",
                },
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clear cache
        self._available_models_cache = []
        self._cache_timestamp = 0.0

        logger.info("✅ LLM service cleanup completed")
