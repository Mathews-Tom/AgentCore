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
- Support for gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5-mini models

Example:
    ```python
    from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI
    from agentcore.a2a_protocol.models.llm import LLMRequest

    client = LLMClientOpenAI(api_key="sk-...", timeout=60.0, max_retries=3)

    # Non-streaming completion
    request = LLMRequest(
        model="gpt-4.1-mini",
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

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
    ProviderTimeoutError,
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
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.timeout = timeout
        self.max_retries = max_retries

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
            ...     model="gpt-4.1-mini",
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
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,  # type: ignore[arg-type]
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    extra_headers=extra_headers if extra_headers else None,
                )

                # Calculate latency and normalize response
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                return self._normalize_response((response, latency_ms), request)

            except APITimeoutError as e:
                # Timeout errors should not be retried
                raise ProviderTimeoutError("openai", self.timeout) from e

            except (AuthenticationError, BadRequestError) as e:
                # Terminal errors - no retry
                raise ProviderError("openai", e) from e

            except (RateLimitError, APIConnectionError, APIError) as e:
                # Transient errors - retry with exponential backoff
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff = 2**attempt
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
            ...     model="gpt-4.1-mini",
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
            stream_response = await self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,  # type: ignore[arg-type]
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                extra_headers=extra_headers if extra_headers else None,
            )

            # Yield tokens as they arrive
            async for chunk in stream_response:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except APITimeoutError as e:
            raise ProviderTimeoutError("openai", self.timeout) from e

        except (AuthenticationError, BadRequestError) as e:
            raise ProviderError("openai", e) from e

        except (RateLimitError, APIConnectionError, APIError) as e:
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

        content = choices[0].message.content
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
