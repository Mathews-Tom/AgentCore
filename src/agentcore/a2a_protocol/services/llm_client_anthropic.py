"""Anthropic provider implementation of the LLM client.

This module implements the Anthropic Claude provider extending the abstract LLMClient.
It supports both completion and streaming modes with full A2A context propagation,
retry logic, timeout handling, and error management.

Key differences from OpenAI:
- System messages are handled via separate `system` parameter
- Only "user" and "assistant" roles in messages array
- Response format uses content[0].text instead of choices[0].message.content
- Token usage fields: input_tokens → prompt_tokens, output_tokens → completion_tokens

Features:
- Async-first implementation using AsyncAnthropic
- Complete and streaming completion methods
- Message format conversion (OpenAI → Anthropic)
- Automatic retry with exponential backoff (3 retries)
- Configurable timeout (default 60s)
- A2A context propagation via extra_headers (trace_id, source_agent, session_id)
- Token usage extraction and tracking
- Comprehensive error handling with custom exceptions
- Support for claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001, claude-3-opus models
- Legacy support for claude-3-5-sonnet, claude-3-5-haiku-20241022

Example:
    ```python
    from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
    from agentcore.a2a_protocol.models.llm import LLMRequest

    client = LLMClientAnthropic(api_key="sk-ant-...", timeout=60.0, max_retries=3)

    # Non-streaming completion
    request = LLMRequest(
        model="claude-3-5-haiku-20241022",
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
    - ProviderError: Raised for general Anthropic API errors
    - ProviderTimeoutError: Raised when request exceeds timeout
    - Automatic retry on transient errors (rate limits, connection issues)
    - No retry on terminal errors (authentication, invalid parameters)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from anthropic import (
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


class LLMClientAnthropic(LLMClient):
    """Anthropic Claude implementation of LLM client.

    This class provides Anthropic-specific implementation of the abstract LLMClient
    interface. It handles all Anthropic API interactions including message format
    conversion, request formatting, response normalization, error handling, and
    retry logic.

    Attributes:
        client: AsyncAnthropic client instance for API communication
        timeout: Request timeout in seconds (default 60.0)
        max_retries: Maximum number of retry attempts (default 3)
    """

    def __init__(
        self, api_key: str, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        """Initialize Anthropic client with API key, timeout, and retry configuration.

        Args:
            api_key: Anthropic API key for authentication
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum number of retry attempts on transient errors (default 3)
        """
        import logging

        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert OpenAI message format to Anthropic format.

        Anthropic requires:
        1. System messages extracted to separate `system` parameter
        2. Only "user" and "assistant" roles in messages array

        Args:
            messages: Messages in OpenAI format (may include "system" role)

        Returns:
            Tuple of (system_content, non_system_messages):
                - system_content: Extracted system message or None
                - non_system_messages: Messages with only "user" and "assistant" roles

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful"},
            ...     {"role": "user", "content": "Hello"},
            ... ]
            >>> system, msgs = client._convert_messages(messages)
            >>> assert system == "You are helpful"
            >>> assert msgs == [{"role": "user", "content": "Hello"}]
        """
        system_content: str | None = None
        non_system_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") == "system":
                # Extract system message (take last one if multiple)
                system_content = msg.get("content", "")
            else:
                # Keep user and assistant messages
                non_system_messages.append(msg)

        return system_content, non_system_messages

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute non-streaming completion request with retry logic.

        This method sends a completion request to Anthropic's API with automatic
        retry on transient errors. It converts OpenAI message format to Anthropic
        format, propagates A2A context via extra_headers, and normalizes the
        response to LLMResponse format.

        Args:
            request: Unified LLM request with model, messages, and A2A context

        Returns:
            Normalized LLM response with content, usage, and latency

        Raises:
            ProviderError: For general Anthropic API errors
            ProviderTimeoutError: When request exceeds timeout

        Example:
            >>> request = LLMRequest(
            ...     model="claude-3-5-haiku-20241022",
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     trace_id="trace-123",
            ... )
            >>> response = await client.complete(request)
        """
        # Convert message format (OpenAI → Anthropic)
        system_content, anthropic_messages = self._convert_messages(request.messages)

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
                # Build API call parameters (no temperature or max_tokens per CLAUDE.md)
                api_params: dict[str, any] = {
                    "model": request.model,
                    "messages": anthropic_messages,
                    "max_tokens": 4096,  # Required by Anthropic API
                }

                # Only include system if there's content (Anthropic doesn't accept system=None)
                if system_content:
                    api_params["system"] = [{"type": "text", "text": system_content}]

                if extra_headers:
                    api_params["extra_headers"] = extra_headers

                # Call Anthropic API with proper parameters
                response = await self.client.messages.create(**api_params)  # type: ignore[arg-type]

                # Calculate latency and normalize response
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                return self._normalize_response((response, latency_ms), request)

            except APITimeoutError as e:
                # Timeout errors should not be retried
                raise ProviderTimeoutError("anthropic", self.timeout) from e

            except (AuthenticationError, BadRequestError) as e:
                # Terminal errors - no retry
                raise ProviderError("anthropic", e) from e

            except RateLimitError as e:
                # Rate limit error - extract retry information from response
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after_header = e.response.headers.get("retry-after")
                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                        except (ValueError, TypeError):
                            pass

                # Record rate limit metrics
                record_rate_limit_error("anthropic", request.model)

                # Calculate backoff delay (respect retry-after or use exponential backoff)
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
                        "provider": "anthropic",
                        "model": request.model,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "backoff_seconds": backoff,
                        "retry_after": retry_after,
                        "trace_id": request.trace_id,
                    },
                )

                # Record retry delay metric
                record_rate_limit_retry_delay("anthropic", request.model, backoff)

                # Check if we should retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(backoff)
                    continue

                # Max retries exceeded - raise custom rate limit error
                raise CustomRateLimitError(
                    "anthropic", retry_after, str(e)
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
                raise ProviderError("anthropic", e) from e

        # Should never reach here, but for type safety
        if last_exception:
            raise ProviderError("anthropic", last_exception) from last_exception
        raise ProviderError("anthropic", Exception("Unknown error in retry loop"))

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:  # type: ignore[override]
        """Execute streaming completion and yield tokens as they arrive.

        This method sends a streaming completion request to Anthropic's API and
        yields content tokens immediately as they are received. It converts OpenAI
        message format to Anthropic format and propagates A2A context via
        extra_headers.

        Args:
            request: Unified LLM request with model, messages, and A2A context

        Yields:
            Content tokens as strings in order of generation

        Raises:
            ProviderError: For general Anthropic API errors during streaming
            ProviderTimeoutError: When stream does not produce tokens within timeout

        Example:
            >>> request = LLMRequest(
            ...     model="claude-3-5-haiku-20241022",
            ...     messages=[{"role": "user", "content": "Count to 5"}],
            ...     stream=True,
            ... )
            >>> async for token in client.stream(request):
            ...     print(token, end="", flush=True)
        """
        # Convert message format (OpenAI → Anthropic)
        system_content, anthropic_messages = self._convert_messages(request.messages)

        # Build extra_headers for A2A context propagation
        extra_headers: dict[str, str] = {}
        if request.trace_id:
            extra_headers["X-Trace-ID"] = request.trace_id
        if request.source_agent:
            extra_headers["X-Source-Agent"] = request.source_agent
        if request.session_id:
            extra_headers["X-Session-ID"] = request.session_id

        try:
            # Build API call parameters (no temperature or max_tokens per CLAUDE.md)
            api_params: dict[str, any] = {
                "model": request.model,
                "messages": anthropic_messages,
                "max_tokens": 4096,  # Required by Anthropic API
                "stream": True,
            }

            # Only include system if there's content (Anthropic doesn't accept system=None)
            if system_content:
                api_params["system"] = [{"type": "text", "text": system_content}]

            if extra_headers:
                api_params["extra_headers"] = extra_headers

            # Call Anthropic API with streaming enabled
            stream_response = await self.client.messages.create(**api_params)  # type: ignore[arg-type]

            # Yield tokens as they arrive
            async for event in stream_response:  # type: ignore[union-attr]
                # Anthropic streaming events have different types
                # We're interested in content_block_delta events with text
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        yield event.delta.text

        except APITimeoutError as e:
            raise ProviderTimeoutError("anthropic", self.timeout) from e

        except (AuthenticationError, BadRequestError) as e:
            raise ProviderError("anthropic", e) from e

        except RateLimitError as e:
            # Rate limit error during streaming
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except (ValueError, TypeError):
                        pass

            # Record rate limit metrics
            record_rate_limit_error("anthropic", request.model)

            # Log rate limit event
            self.logger.warning(
                "Rate limit exceeded during streaming",
                extra={
                    "provider": "anthropic",
                    "model": request.model,
                    "retry_after": retry_after,
                    "trace_id": request.trace_id,
                },
            )

            # Raise custom rate limit error (no retry in streaming)
            raise CustomRateLimitError("anthropic", retry_after, str(e)) from e

        except (APIConnectionError, APIError) as e:
            raise ProviderError("anthropic", e) from e

    def _normalize_response(
        self, raw_response: object, request: LLMRequest
    ) -> LLMResponse:
        """Normalize Anthropic response to unified LLMResponse format.

        This method extracts content, token usage, and metadata from the
        Anthropic-specific response object and converts it to the standardized
        LLMResponse format with A2A context propagation.

        Anthropic response structure:
        - content[0].text: Generated text content
        - usage.input_tokens: Prompt tokens
        - usage.output_tokens: Completion tokens

        Args:
            raw_response: Tuple of (Anthropic Message response object, latency_ms)
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

        # Type guard for Anthropic response structure
        if not hasattr(actual_response, "content"):
            raise ValueError("Invalid Anthropic response: missing 'content' field")
        if not hasattr(actual_response, "usage"):
            raise ValueError("Invalid Anthropic response: missing 'usage' field")

        # Extract content from first content block
        content_blocks = actual_response.content
        if not content_blocks:
            raise ValueError("Invalid Anthropic response: empty 'content' list")

        # Anthropic response content is a list of ContentBlock objects
        # Each has a 'text' attribute
        content = ""
        if hasattr(content_blocks[0], "text"):
            content = content_blocks[0].text

        # Extract token usage (Anthropic → LLM standard mapping)
        usage = actual_response.usage
        llm_usage = LLMUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

        # Build normalized response with A2A context
        return LLMResponse(
            content=content,
            usage=llm_usage,
            latency_ms=latency_ms,
            provider="anthropic",
            model=request.model,
            trace_id=request.trace_id,
        )
