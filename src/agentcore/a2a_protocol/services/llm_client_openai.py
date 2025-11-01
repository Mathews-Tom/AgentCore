"""OpenAI provider implementation of the LLM client.

This module implements the OpenAI provider extending the abstract LLMClient.
It supports both completion and streaming modes with full A2A context propagation,
retry logic, timeout handling, and error management.

This is the first working provider demonstrating feasibility and establishing
implementation patterns for Anthropic and Gemini clients.

Features:
- Async-first implementation using AsyncOpenAI
- Complete and streaming completion methods
- Automatic retry with exponential backoff (3 retries)
- Configurable timeout (default 60s)
- A2A context propagation via extra_headers (trace_id, source_agent, session_id)
- Token usage extraction and tracking
- Comprehensive error handling with custom exceptions
- Support for gpt-5, gpt-5-mini, gpt-5, gpt-5-mini models

Example:
    ```python
    from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI
    from agentcore.a2a_protocol.models.llm import LLMRequest

    client = LLMClientOpenAI(api_key="sk-...", timeout=60.0, max_retries=3)

    # Non-streaming completion
    request = LLMRequest(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        trace_id="trace-123",
    )
    response = await client.complete(request)
    print(response.content)

    # Streaming completion
    async for token in client.stream(request):
        print(token, end="", flush=True)
    ```

Error Handling:
    - ProviderError: Raised for general OpenAI API errors
    - ProviderTimeoutError: Raised when request exceeds timeout
    - Automatic retry on transient errors (rate limits, connection issues)
    - No retry on terminal errors (authentication, invalid parameters)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.metrics.llm_metrics import (
    record_rate_limit_error,
    record_rate_limit_retry_delay,
)
from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError as CustomRateLimitError,
)
from agentcore.a2a_protocol.services.llm_client_base import LLMClient


class LLMClientOpenAI(LLMClient):
    """OpenAI implementation of LLM client.

    This class provides OpenAI-specific implementation of the abstract LLMClient
    interface. It handles all OpenAI API interactions including request formatting,
    response normalization, error handling, and retry logic.

    Attributes:
        client: AsyncOpenAI client instance for API communication
        timeout: Request timeout in seconds (default 60.0)
        max_retries: Maximum number of retry attempts (default 3)
    """

    def __init__(
        self, api_key: str, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        """Initialize OpenAI client with API key, timeout, and retry configuration.

        Args:
            api_key: OpenAI API key for authentication
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum number of retry attempts on transient errors (default 3)
        """
        import logging

        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model supports reasoning_effort parameter.

        Args:
            model: Model name to check

        Returns:
            True if model supports reasoning_effort, False otherwise
        """
        return model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute non-streaming completion request with retry logic.

        This method sends a completion request to OpenAI's API with automatic
        retry on transient errors. It propagates A2A context via extra_headers
        and normalizes the response to LLMResponse format.

        Args:
            request: Unified LLM request with model, messages, and A2A context

        Returns:
            Normalized LLM response with content, usage, and latency

        Raises:
            ProviderError: For general OpenAI API errors
            ProviderTimeoutError: When request exceeds timeout

        Example:
            >>> request = LLMRequest(
            ...     model="gpt-5-mini",
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     trace_id="trace-123",
            ... )
            >>> response = await client.complete(request)
        """
        # Build extra_headers for A2A context propagation
        extra_headers: dict[str, str] = {}
        if request.trace_id:
            extra_headers["X-Trace-ID"] = request.trace_id
        if request.source_agent:
            extra_headers["X-Source-Agent"] = request.source_agent
        if request.session_id:
            extra_headers["X-Session-ID"] = request.session_id

        # Track request timing for latency metrics
        start_time = time.perf_counter()
        last_exception: Exception | None = None

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Build completion parameters (no temperature or max_tokens per CLAUDE.md)
                completion_params: dict[str, object] = {
                    "model": request.model,
                    "messages": request.messages,
                }

                # Add reasoning_effort for reasoning models (GPT-5, o1, o3)
                if request.reasoning_effort and self._is_reasoning_model(request.model):
                    completion_params["reasoning_effort"] = request.reasoning_effort

                if extra_headers:
                    completion_params["extra_headers"] = extra_headers

                response = await self.client.chat.completions.create(**completion_params)  # type: ignore[arg-type]

                # Calculate latency and normalize response
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                return self._normalize_response((response, latency_ms), request)

            except APITimeoutError as e:
                # Timeout errors should not be retried
                raise ProviderTimeoutError("openai", self.timeout) from e

            except (AuthenticationError, BadRequestError) as e:
                # Terminal errors - no retry
                raise ProviderError("openai", e) from e

            except RateLimitError as e:
                # Rate limit error - extract Retry-After header if available
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after_header = e.response.headers.get("Retry-After")
                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                        except (ValueError, TypeError):
                            pass

                # Record rate limit metrics
                record_rate_limit_error("openai", request.model)

                # Calculate backoff delay (respect Retry-After or use exponential backoff)
                if retry_after is not None:
                    backoff = min(retry_after, settings.LLM_MAX_RETRY_DELAY)
                else:
                    backoff = min(
                        settings.LLM_RETRY_EXPONENTIAL_BASE**attempt,
                        settings.LLM_MAX_RETRY_DELAY,
                    )

                # Log rate limit event
                self.logger.warning(
                    "Rate limit exceeded, retrying",
                    extra={
                        "provider": "openai",
                        "model": request.model,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "backoff_seconds": backoff,
                        "retry_after": retry_after,
                        "trace_id": request.trace_id,
                    },
                )

                # Record retry delay metric
                record_rate_limit_retry_delay("openai", request.model, backoff)

                # Check if we should retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(backoff)
                    continue

                # Max retries exceeded - raise custom rate limit error
                raise CustomRateLimitError(
                    "openai", retry_after, str(e)
                ) from e

            except (APIConnectionError, APIError) as e:
                # Transient errors - retry with exponential backoff
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff = min(
                        settings.LLM_RETRY_EXPONENTIAL_BASE**attempt,
                        settings.LLM_MAX_RETRY_DELAY,
                    )
                    await asyncio.sleep(backoff)
                    continue
                # Max retries exceeded
                raise ProviderError("openai", e) from e

        # Should never reach here, but for type safety
        if last_exception:
            raise ProviderError("openai", last_exception) from last_exception
        raise ProviderError("openai", Exception("Unknown error in retry loop"))

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:  # type: ignore[override]
        """Execute streaming completion and yield tokens as they arrive.

        This method sends a streaming completion request to OpenAI's API and
        yields content tokens immediately as they are received. It propagates
        A2A context via extra_headers.

        Args:
            request: Unified LLM request with model, messages, and A2A context

        Yields:
            Content tokens as strings in order of generation

        Raises:
            ProviderError: For general OpenAI API errors during streaming
            ProviderTimeoutError: When stream does not produce tokens within timeout

        Example:
            >>> request = LLMRequest(
            ...     model="gpt-5-mini",
            ...     messages=[{"role": "user", "content": "Count to 5"}],
            ...     stream=True,
            ... )
            >>> async for token in client.stream(request):
            ...     print(token, end="", flush=True)
        """
        # Build extra_headers for A2A context propagation
        extra_headers: dict[str, str] = {}
        if request.trace_id:
            extra_headers["X-Trace-ID"] = request.trace_id
        if request.source_agent:
            extra_headers["X-Source-Agent"] = request.source_agent
        if request.session_id:
            extra_headers["X-Session-ID"] = request.session_id

        try:
            # Build stream parameters (no temperature or max_tokens per CLAUDE.md)
            stream_params: dict[str, object] = {
                "model": request.model,
                "messages": request.messages,
                "stream": True,
            }

            # Add reasoning_effort for reasoning models (GPT-5, o1, o3)
            if request.reasoning_effort and self._is_reasoning_model(request.model):
                stream_params["reasoning_effort"] = request.reasoning_effort

            if extra_headers:
                stream_params["extra_headers"] = extra_headers

            stream_response = await self.client.chat.completions.create(**stream_params)  # type: ignore[arg-type]

            # Yield tokens as they arrive
            async for chunk in stream_response:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except APITimeoutError as e:
            raise ProviderTimeoutError("openai", self.timeout) from e

        except (AuthenticationError, BadRequestError) as e:
            raise ProviderError("openai", e) from e

        except RateLimitError as e:
            # Rate limit error during streaming - extract Retry-After header
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_header = e.response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass

            # Record rate limit metrics
            record_rate_limit_error("openai", request.model)

            # Log rate limit event
            self.logger.warning(
                "Rate limit exceeded during streaming",
                extra={
                    "provider": "openai",
                    "model": request.model,
                    "retry_after": retry_after,
                    "trace_id": request.trace_id,
                },
            )

            # Raise custom rate limit error (no retry in streaming)
            raise CustomRateLimitError("openai", retry_after, str(e)) from e

        except (APIConnectionError, APIError) as e:
            raise ProviderError("openai", e) from e

    def _normalize_response(
        self, raw_response: object, request: LLMRequest
    ) -> LLMResponse:
        """Normalize OpenAI response to unified LLMResponse format.

        This method extracts content, token usage, and metadata from the
        OpenAI-specific response object and converts it to the standardized
        LLMResponse format with A2A context propagation.

        Args:
            raw_response: Tuple of (OpenAI ChatCompletion response object, latency_ms)
            request: Original LLM request for context propagation

        Returns:
            Normalized LLM response in unified format

        Raises:
            ValueError: When raw_response is malformed or missing required fields
        """
        # Unpack response and latency from tuple
        if isinstance(raw_response, tuple):
            actual_response, latency_ms = raw_response
        else:
            raise ValueError("Invalid response format: expected tuple")

        # Type guard for OpenAI response structure
        if not hasattr(actual_response, "choices"):
            raise ValueError("Invalid OpenAI response: missing 'choices' field")
        if not hasattr(actual_response, "usage"):
            raise ValueError("Invalid OpenAI response: missing 'usage' field")

        # Extract content from first choice
        choices = actual_response.choices
        if not choices:
            raise ValueError("Invalid OpenAI response: empty 'choices' list")

        message = choices[0].message
        content = message.content

        # For GPT-5 models, check for text in annotations if content is empty
        if not content and hasattr(message, 'annotations') and message.annotations:
            for annotation in message.annotations:
                if hasattr(annotation, 'text'):
                    content = annotation.text
                    break

        if content is None:
            content = ""

        # Extract token usage
        usage = actual_response.usage
        llm_usage = LLMUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        # Build normalized response with A2A context
        return LLMResponse(
            content=content,
            usage=llm_usage,
            latency_ms=latency_ms,
            provider="openai",
            model=request.model,
            trace_id=request.trace_id,
        )
