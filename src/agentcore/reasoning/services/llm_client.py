"""
LLM Client adapter for Context Reasoning framework.

Provides async LLM client with support for:
- Stop sequences (<answer>, <continue>)
- Token counting via tiktoken
- Retry logic with exponential backoff
- Circuit breaker pattern
- Connection pooling
- Timeout handling
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

import httpx
import structlog
import tiktoken
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Too many failures, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class LLMClientConfig(BaseModel):
    """Configuration for LLM client adapter."""

    api_key: str = Field(..., description="LLM provider API key")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="LLM API base URL",
    )
    default_model: str = Field(default="gpt-4.1", description="Default LLM model")
    timeout_seconds: int = Field(default=60, ge=5, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    retry_delays: list[float] = Field(
        default=[1.0, 2.0, 4.0],
        description="Retry delays in seconds (exponential backoff)",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before circuit opens",
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        ge=10,
        description="Seconds before circuit half-opens",
    )
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="HTTP connection pool size",
    )


class GenerationResult(BaseModel):
    """Result from LLM generation."""

    content: str = Field(..., description="Generated text content")
    tokens_used: int = Field(..., ge=0, description="Total tokens consumed")
    finish_reason: str = Field(..., description="Reason for completion (stop, length, etc.)")
    model: str = Field(..., description="Model used for generation")
    stop_sequence_found: str | None = Field(
        default=None,
        description="Stop sequence that triggered completion",
    )


class LLMClient:
    """
    Async LLM client adapter for reasoning strategies.

    Features:
    - Stop sequence support for answer detection
    - Token counting via tiktoken
    - Retry logic with exponential backoff
    - Circuit breaker pattern for fault tolerance
    - Connection pooling for performance
    """

    def __init__(self, config: LLMClientConfig):
        """
        Initialize LLM client.

        Args:
            config: Client configuration
        """
        self.config = config

        # Initialize HTTP client with connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=config.connection_pool_size,
            max_connections=config.connection_pool_size,
        )
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
            limits=limits,
        )

        # Initialize token encoder (tiktoken)
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.default_model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Circuit breaker state
        self.circuit_state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.circuit_opened_at: float | None = None

        logger.info(
            "llm_client_initialized",
            base_url=config.base_url,
            model=config.default_model,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            circuit_threshold=config.circuit_breaker_threshold,
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        return len(self.tokenizer.encode(text))

    async def _check_circuit_breaker(self) -> None:
        """
        Check circuit breaker state and raise if open.

        Raises:
            RuntimeError: If circuit is open
        """
        if self.circuit_state == CircuitState.OPEN:
            # Check if timeout has passed
            current_time = asyncio.get_event_loop().time()
            if (
                self.circuit_opened_at is not None
                and current_time - self.circuit_opened_at >= self.config.circuit_breaker_timeout
            ):
                # Move to half-open state
                self.circuit_state = CircuitState.HALF_OPEN
                logger.info("circuit_breaker_half_open", attempt="testing service recovery")
            else:
                raise RuntimeError(
                    f"Circuit breaker OPEN. Service unavailable. "
                    f"Retry after {self.config.circuit_breaker_timeout}s"
                )

    def _record_success(self) -> None:
        """Record successful request, reset circuit breaker."""
        if self.circuit_state == CircuitState.HALF_OPEN:
            logger.info("circuit_breaker_closed", reason="service recovered")

        self.circuit_state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_opened_at = None

    def _record_failure(self) -> None:
        """Record failed request, update circuit breaker."""
        current_time = asyncio.get_event_loop().time()
        self.failure_count += 1
        self.last_failure_time = current_time

        if self.failure_count >= self.config.circuit_breaker_threshold:
            self.circuit_state = CircuitState.OPEN
            self.circuit_opened_at = current_time
            logger.error(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.config.circuit_breaker_threshold,
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        model: str | None = None,
    ) -> GenerationResult:
        """
        Generate completion with retry logic and circuit breaker.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stop_sequences: Stop sequences for answer detection
            model: Model to use (defaults to config)

        Returns:
            Generation result

        Raises:
            RuntimeError: If circuit breaker is open
            httpx.HTTPError: If all retries fail
        """
        # Check circuit breaker
        await self._check_circuit_breaker()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build payload
        model_to_use = model or self.config.default_model
        payload: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Retry logic with exponential backoff
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    "llm_request",
                    attempt=attempt + 1,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                )

                response = await self.client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                # Parse response
                data = response.json()
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason", "stop")

                # Detect stop sequence
                stop_sequence_found = None
                if finish_reason == "stop" and stop_sequences:
                    content = choice["message"]["content"]
                    for seq in stop_sequences:
                        if seq in content:
                            stop_sequence_found = seq
                            break

                result = GenerationResult(
                    content=choice["message"]["content"],
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason=finish_reason,
                    model=data.get("model", model_to_use),
                    stop_sequence_found=stop_sequence_found,
                )

                # Record success
                self._record_success()

                logger.info(
                    "llm_response",
                    model=result.model,
                    tokens=result.tokens_used,
                    finish_reason=result.finish_reason,
                    stop_sequence=stop_sequence_found,
                    attempt=attempt + 1,
                )

                return result

            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(
                    "llm_request_failed",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(e),
                )

                # Record failure
                self._record_failure()

                # Don't retry if circuit is now open
                if self.circuit_state == CircuitState.OPEN:
                    raise RuntimeError("Circuit breaker opened after failure") from e

                # Wait before retry (exponential backoff)
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delays[min(attempt, len(self.config.retry_delays) - 1)]
                    await asyncio.sleep(delay)

        # All retries exhausted
        self._record_failure()
        logger.error(
            "llm_request_failed_all_retries",
            max_retries=self.config.max_retries,
            error=str(last_exception),
        )
        raise last_exception or RuntimeError("All retries failed")

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self.client.aclose()
        logger.info("llm_client_closed")

    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
