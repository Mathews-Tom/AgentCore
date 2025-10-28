"""Abstract base class for LLM provider clients.

This module defines the contract that all LLM provider implementations must follow.
It establishes a unified interface for OpenAI, Anthropic, and Google Gemini clients,
enabling provider-agnostic LLM operations across AgentCore.

The abstract base class ensures:
- Consistent request/response handling across providers
- Unified error handling contracts
- Streaming support for all implementations
- Provider-specific response normalization
- Type safety with full mypy compliance

Example implementation:

    ```python
    from agentcore.a2a_protocol.services.llm_client_base import LLMClient
    from agentcore.a2a_protocol.models.llm import (
        LLMRequest,
        LLMResponse,
        LLMUsage,
        ProviderError,
        ProviderTimeoutError,
    )

    class OpenAIClient(LLMClient):
        '''OpenAI implementation of LLM client.'''

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.client = openai.AsyncOpenAI(api_key=api_key)

        async def complete(self, request: LLMRequest) -> LLMResponse:
            '''Execute completion request via OpenAI API.'''
            try:
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                return self._normalize_response(response, request)
            except openai.APITimeoutError as e:
                raise ProviderTimeoutError("openai", 60.0) from e
            except openai.APIError as e:
                raise ProviderError("openai", e) from e

        async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
            '''Execute streaming completion via OpenAI API.'''
            try:
                stream = await self.client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except openai.APITimeoutError as e:
                raise ProviderTimeoutError("openai", 60.0) from e
            except openai.APIError as e:
                raise ProviderError("openai", e) from e

        def _normalize_response(
            self, raw_response: object, request: LLMRequest
        ) -> LLMResponse:
            '''Normalize OpenAI response to LLMResponse.'''
            # Provider-specific normalization logic
            return LLMResponse(
                content=raw_response.choices[0].message.content,
                usage=LLMUsage(
                    prompt_tokens=raw_response.usage.prompt_tokens,
                    completion_tokens=raw_response.usage.completion_tokens,
                    total_tokens=raw_response.usage.total_tokens,
                ),
                latency_ms=0,  # Calculate from timing
                provider="openai",
                model=request.model,
                trace_id=request.trace_id,
            )
    ```

Error Handling Contract:
    All implementations MUST raise the following exceptions:

    - ProviderError: For general provider API errors (SDK exceptions, HTTP errors)
    - ProviderTimeoutError: When request exceeds timeout limit
    - ModelNotAllowedError: When requested model is not in ALLOWED_MODELS

    Implementations SHOULD NOT catch these exceptions internally. Let them propagate
    to the LLMService layer for unified error handling and metrics.

Thread Safety:
    Implementations SHOULD be thread-safe for use in async contexts. Use async
    primitives (asyncio.Lock) if internal state requires synchronization.

Performance:
    Implementations SHOULD minimize overhead. Target <5ms abstraction latency
    compared to native SDK usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse


class LLMClient(ABC):
    """Abstract base class for all LLM provider implementations.

    This class defines the contract that OpenAI, Anthropic, and Gemini clients
    must implement. It ensures a unified interface for LLM operations across
    all providers.

    All methods are async-first to support high-concurrency scenarios and
    non-blocking I/O operations required by the A2A protocol.

    Attributes:
        None. Subclasses should define provider-specific attributes (e.g., API keys,
        client instances) in their __init__ methods.
    """

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute non-streaming completion request and return normalized response.

        This method sends a completion request to the provider's API and returns
        the result as a normalized LLMResponse. It must handle all provider-specific
        request formatting, API communication, and response parsing.

        The implementation MUST:
        1. Validate the request model against provider capabilities
        2. Convert LLMRequest to provider-specific format
        3. Execute the API call with proper timeout handling
        4. Normalize the response using _normalize_response()
        5. Propagate A2A context (trace_id) to the response
        6. Raise appropriate exceptions on errors

        Args:
            request: Unified LLM request with model, messages, and parameters.
                Contains A2A context fields (trace_id, source_agent, session_id)
                for distributed tracing.

        Returns:
            Normalized LLM response with content, usage statistics, latency,
            and A2A context propagation.

        Raises:
            ProviderError: When the provider API returns an error (authentication,
                rate limiting, invalid parameters, etc.)
            ProviderTimeoutError: When the request exceeds the configured timeout
                (default 60s per LLM_REQUEST_TIMEOUT)
            ModelNotAllowedError: When request.model is not in ALLOWED_MODELS
                configuration (should be validated before calling this method)

        Example:
            >>> client = OpenAIClient(api_key="sk-...")
            >>> request = LLMRequest(
            ...     model="gpt-4.1-mini",
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     trace_id="trace-123",
            ... )
            >>> response = await client.complete(request)
            >>> print(response.content)
            "Hello! How can I help you today?"
        """
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Execute streaming completion and yield tokens as they are generated.

        This method sends a streaming completion request to the provider's API
        and yields content tokens as they arrive. It enables real-time response
        generation for interactive applications.

        The implementation MUST:
        1. Validate the request model against provider capabilities
        2. Convert LLMRequest to provider-specific streaming format
        3. Execute the streaming API call with proper timeout handling
        4. Yield only content tokens (not metadata or deltas)
        5. Handle stream interruptions and errors gracefully
        6. Raise appropriate exceptions on errors

        The iterator SHOULD:
        - Yield tokens immediately as they arrive (no buffering)
        - Handle provider-specific chunking formats
        - Clean up resources on completion or error

        Args:
            request: Unified LLM request with model, messages, and parameters.
                Must have stream=True for streaming operations (though this
                method implicitly streams regardless of that flag).

        Yields:
            Content tokens as strings. Each yield represents a chunk of generated
            text. Tokens are yielded in order and should be concatenated by the
            consumer to reconstruct the full response.

        Raises:
            ProviderError: When the provider API returns an error during streaming
                (authentication, rate limiting, connection issues, etc.)
            ProviderTimeoutError: When the stream does not produce tokens within
                the configured timeout period
            ModelNotAllowedError: When request.model is not in ALLOWED_MODELS
                configuration (should be validated before calling this method)

        Example:
            >>> client = OpenAIClient(api_key="sk-...")
            >>> request = LLMRequest(
            ...     model="gpt-4.1-mini",
            ...     messages=[{"role": "user", "content": "Count to 5"}],
            ...     stream=True,
            ... )
            >>> async for token in client.stream(request):
            ...     print(token, end="", flush=True)
            1 2 3 4 5

        Notes:
            - Implementations should not accumulate tokens internally
            - Latency metrics should track time to first token separately
            - Final usage statistics may not be available until stream completes
        """
        pass

    @abstractmethod
    def _normalize_response(
        self, raw_response: object, request: LLMRequest
    ) -> LLMResponse:
        """Normalize provider-specific response to unified LLMResponse format.

        This method converts the provider's native response object into the
        standardized LLMResponse model. It handles provider-specific field
        mappings, units conversion, and metadata extraction.

        The implementation MUST:
        1. Extract generated content from provider response
        2. Extract token usage (prompt_tokens, completion_tokens, total_tokens)
        3. Calculate or extract latency in milliseconds
        4. Set provider and model fields
        5. Propagate A2A context (trace_id) from request to response

        The implementation SHOULD:
        - Handle missing or optional fields gracefully (use defaults)
        - Convert provider-specific token counts to standard format
        - Include timing information if available from provider

        Args:
            raw_response: Provider-specific response object. Type varies by
                provider (e.g., openai.ChatCompletion, anthropic.Message,
                google.genai.GenerateContentResponse).
            request: Original LLM request used to generate the response.
                Contains A2A context fields that should be propagated to
                the normalized response.

        Returns:
            Normalized LLM response in unified format with all required fields.

        Raises:
            ValueError: When the raw response is malformed or missing required
                fields that cannot be reasonably defaulted.

        Example:
            >>> raw = openai_client.chat.completions.create(...)
            >>> request = LLMRequest(model="gpt-4.1-mini", messages=[...])
            >>> normalized = client._normalize_response(raw, request)
            >>> assert isinstance(normalized, LLMResponse)
            >>> assert normalized.provider == "openai"

        Notes:
            - This method can be sync or async depending on provider needs
            - Most implementations will be synchronous (simple field mapping)
            - Token usage must always be present in normalized response
            - Latency should be calculated from actual request timing
        """
        pass
