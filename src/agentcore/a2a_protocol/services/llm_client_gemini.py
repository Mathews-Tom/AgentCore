"""Google Gemini provider implementation of the LLM client.

This module implements the Google Gemini provider extending the abstract LLMClient.
It supports both completion and streaming modes with A2A context propagation,
retry logic, timeout handling, and error management.

Key differences from OpenAI/Anthropic:
- Uses google.generativeai SDK with GenerativeModel
- Message format uses role/parts structure (similar to OpenAI but with 'parts' array)
- Supports 'user' and 'model' roles (not 'assistant')
- System instructions passed via system_instruction parameter
- Response format uses candidates[0].content.parts[0].text
- Token usage in usage_metadata: prompt_token_count, candidates_token_count
- Limited custom header support for A2A context

Features:
- Async-first implementation using generate_content_async
- Complete and streaming completion methods
- Message format conversion (OpenAI → Gemini)
- Automatic retry with exponential backoff (3 retries)
- Configurable timeout (default 60s)
- A2A context propagation via request_options (limited support)
- Token usage extraction and tracking
- Comprehensive error handling with custom exceptions
- Support for gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash models

Example:
    ```python
    from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
    from agentcore.a2a_protocol.models.llm import LLMRequest

    client = LLMClientGemini(api_key="...", timeout=60.0, max_retries=3)

    # Non-streaming completion
    request = LLMRequest(
        model="gemini-1.5-flash",
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
    - ProviderError: Raised for general Gemini API errors
    - ProviderTimeoutError: Raised when request exceeds timeout
    - Automatic retry on transient errors (rate limits, connection issues)
    - No retry on terminal errors (authentication, invalid parameters)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import google.generativeai as genai  # type: ignore[import-untyped,unused-ignore]
from google.api_core import exceptions as google_exceptions

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
    ProviderTimeoutError,
)
from agentcore.a2a_protocol.services.llm_client_base import LLMClient


class LLMClientGemini(LLMClient):
    """Google Gemini implementation of LLM client.

    This class provides Gemini-specific implementation of the abstract LLMClient
    interface. It handles all Gemini API interactions including message format
    conversion, request formatting, response normalization, error handling, and
    retry logic.

    Attributes:
        api_key: Gemini API key for authentication
        timeout: Request timeout in seconds (default 60.0)
        max_retries: Maximum number of retry attempts (default 3)
    """

    def __init__(
        self, api_key: str, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        """Initialize Gemini client with API key, timeout, and retry configuration.

        Args:
            api_key: Google API key for Gemini authentication
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum number of retry attempts on transient errors (default 3)
        """
        # Configure the SDK with API key
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert OpenAI message format to Gemini format.

        Gemini requires:
        1. System messages extracted to separate system_instruction parameter
        2. Role conversion: 'assistant' → 'model', 'user' remains 'user'
        3. Content converted to parts format: [{"text": content}]

        Args:
            messages: Messages in OpenAI format (may include "system" role)

        Returns:
            Tuple of (system_instruction, gemini_messages):
                - system_instruction: Extracted system message or None
                - gemini_messages: Messages with 'user'/'model' roles and parts format

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful"},
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ... ]
            >>> system, msgs = client._convert_messages(messages)
            >>> assert system == "You are helpful"
            >>> assert msgs == [
            ...     {"role": "user", "parts": [{"text": "Hello"}]},
            ...     {"role": "model", "parts": [{"text": "Hi there!"}]},
            ... ]
        """
        system_instruction: str | None = None
        gemini_messages: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Extract system message (take last one if multiple)
                system_instruction = content
            elif role == "user":
                # User messages stay as 'user'
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})  # type: ignore[dict-item]
            elif role == "assistant":
                # Assistant → model for Gemini
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})  # type: ignore[dict-item]

        return system_instruction, gemini_messages

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute non-streaming completion request with retry logic.

        This method sends a completion request to Gemini's API with automatic
        retry on transient errors. It converts OpenAI message format to Gemini
        format and normalizes the response to LLMResponse format.

        Note: Gemini has limited support for custom headers, so A2A context
        propagation via headers may not be fully supported.

        Args:
            request: Unified LLM request with model, messages, and A2A context

        Returns:
            Normalized LLM response with content, usage, and latency

        Raises:
            ProviderError: For general Gemini API errors
            ProviderTimeoutError: When request exceeds timeout

        Example:
            >>> request = LLMRequest(
            ...     model="gemini-1.5-flash",
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     trace_id="trace-123",
            ... )
            >>> response = await client.complete(request)
        """
        # Convert message format (OpenAI → Gemini)
        system_instruction, gemini_messages = self._convert_messages(request.messages)

        # Build request options for timeout
        # Note: Gemini SDK has limited support for custom headers
        request_options = {"timeout": self.timeout}

        # Track request timing for latency metrics
        start_time = time.perf_counter()
        last_exception: Exception | None = None

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Create GenerativeModel instance
                model = genai.GenerativeModel(  # type: ignore[attr-defined]
                    model_name=request.model,
                    system_instruction=system_instruction,
                )

                # Build generation config
                generation_config = genai.GenerationConfig(  # type: ignore[attr-defined]
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                )

                # Call Gemini API with async method
                response = await model.generate_content_async(
                    contents=gemini_messages,
                    generation_config=generation_config,
                    request_options=request_options,  # type: ignore[arg-type]
                )

                # Calculate latency and normalize response
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                return self._normalize_response((response, latency_ms), request)

            except asyncio.TimeoutError as e:
                # Timeout errors should not be retried
                raise ProviderTimeoutError("gemini", self.timeout) from e

            except google_exceptions.InvalidArgument as e:
                # Invalid parameters - terminal error, no retry
                raise ProviderError("gemini", e) from e

            except google_exceptions.Unauthenticated as e:
                # Authentication error - terminal error, no retry
                raise ProviderError("gemini", e) from e

            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.DeadlineExceeded,
            ) as e:
                # Transient errors - retry with exponential backoff
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff = 2**attempt
                    await asyncio.sleep(backoff)
                    continue
                # Max retries exceeded
                raise ProviderError("gemini", e) from e

            except Exception as e:
                # Catch-all for other Gemini errors
                raise ProviderError("gemini", e) from e

        # Should never reach here, but for type safety
        if last_exception:
            raise ProviderError("gemini", last_exception) from last_exception
        raise ProviderError("gemini", Exception("Unknown error in retry loop"))

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:  # type: ignore[override]
        """Execute streaming completion and yield tokens as they arrive.

        This method sends a streaming completion request to Gemini's API and
        yields content tokens immediately as they are received. It converts OpenAI
        message format to Gemini format.

        Args:
            request: Unified LLM request with model, messages, and A2A context

        Yields:
            Content tokens as strings in order of generation

        Raises:
            ProviderError: For general Gemini API errors during streaming
            ProviderTimeoutError: When stream does not produce tokens within timeout

        Example:
            >>> request = LLMRequest(
            ...     model="gemini-1.5-flash",
            ...     messages=[{"role": "user", "content": "Count to 5"}],
            ...     stream=True,
            ... )
            >>> async for token in client.stream(request):
            ...     print(token, end="", flush=True)
        """
        # Convert message format (OpenAI → Gemini)
        system_instruction, gemini_messages = self._convert_messages(request.messages)

        # Build request options for timeout
        request_options = {"timeout": self.timeout}

        try:
            # Create GenerativeModel instance
            model = genai.GenerativeModel(  # type: ignore[attr-defined]
                model_name=request.model,
                system_instruction=system_instruction,
            )

            # Build generation config
            generation_config = genai.GenerationConfig(  # type: ignore[attr-defined]
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )

            # Call Gemini API with streaming enabled
            stream_response = await model.generate_content_async(
                contents=gemini_messages,
                generation_config=generation_config,
                stream=True,
                request_options=request_options,  # type: ignore[arg-type]
            )

            # Yield tokens as they arrive
            async for chunk in stream_response:
                # Gemini streaming chunks have candidates[0].content.parts[0].text
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    text = chunk.candidates[0].content.parts[0].text
                    if text:
                        yield text

        except asyncio.TimeoutError as e:
            raise ProviderTimeoutError("gemini", self.timeout) from e

        except google_exceptions.InvalidArgument as e:
            raise ProviderError("gemini", e) from e

        except google_exceptions.Unauthenticated as e:
            raise ProviderError("gemini", e) from e

        except (
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
        ) as e:
            raise ProviderError("gemini", e) from e

        except Exception as e:
            raise ProviderError("gemini", e) from e

    def _normalize_response(
        self, raw_response: object, request: LLMRequest
    ) -> LLMResponse:
        """Normalize Gemini response to unified LLMResponse format.

        This method extracts content, token usage, and metadata from the
        Gemini-specific response object and converts it to the standardized
        LLMResponse format with A2A context propagation.

        Gemini response structure:
        - candidates[0].content.parts[0].text: Generated text content
        - usage_metadata.prompt_token_count: Prompt tokens
        - usage_metadata.candidates_token_count: Completion tokens
        - usage_metadata.total_token_count: Total tokens

        Args:
            raw_response: Tuple of (Gemini GenerateContentResponse object, latency_ms)
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

        # Type guard for Gemini response structure
        if not hasattr(actual_response, "candidates"):
            raise ValueError("Invalid Gemini response: missing 'candidates' field")

        # Extract content from first candidate
        candidates = actual_response.candidates
        if not candidates:
            raise ValueError("Invalid Gemini response: empty 'candidates' list")

        # Gemini response structure: candidates[0].content.parts[0].text
        content = ""
        if (
            candidates[0].content
            and candidates[0].content.parts
            and len(candidates[0].content.parts) > 0
        ):
            content = candidates[0].content.parts[0].text or ""

        # Extract token usage (Gemini → LLM standard mapping)
        # Gemini may not always return usage_metadata in all responses
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(actual_response, "usage_metadata"):
            usage = actual_response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            total_tokens = getattr(usage, "total_token_count", 0)

        llm_usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Build normalized response with A2A context
        return LLMResponse(
            content=content,
            usage=llm_usage,
            latency_ms=latency_ms,
            provider="gemini",
            model=request.model,
            trace_id=request.trace_id,
        )
