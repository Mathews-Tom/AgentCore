"""LLM Service provider registry for multi-provider LLM operations.

This module implements the ProviderRegistry that manages model-to-provider mapping
and provider instance lifecycle. It unifies OpenAI, Anthropic, and Gemini clients
under a single selection interface.

The registry implements three key patterns:
- Registry Pattern: Central mapping of models to providers
- Singleton Pattern: Single instance per provider type
- Lazy Initialization: Providers created only when first requested

Features:
- Model-to-provider mapping for all supported models
- Provider instance management (singleton per provider)
- Lazy initialization (providers created on first request)
- Configuration-driven API key loading
- Model validation against ALLOWED_MODELS
- Missing API key detection and error handling

Example:
    ```python
    from agentcore.a2a_protocol.services.llm_service import ProviderRegistry
    from agentcore.a2a_protocol.models.llm import LLMRequest

    registry = ProviderRegistry(timeout=60.0, max_retries=3)

    # Get provider for a specific model
    client = registry.get_provider_for_model("gpt-4.1-mini")
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    response = await client.complete(request)

    # List all available models
    models = registry.list_available_models()
    print(models)  # ["gpt-4.1-mini", "claude-3-5-haiku-20241022", ...]
    ```

Error Handling:
    - ValueError: Raised when model is unknown or not in ALLOWED_MODELS
    - RuntimeError: Raised when API key is not configured for a provider
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.metrics.llm_metrics import (
    record_governance_violation,
    record_llm_duration,
    record_llm_error,
    record_llm_request,
    record_llm_tokens,
    track_active_requests,
)
from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    ModelNotAllowedError,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_base import LLMClient
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI

# Model-to-provider mapping
# This mapping defines which provider handles each model
MODEL_PROVIDER_MAP: dict[str, Provider] = {
    # OpenAI models
    "gpt-4.1": Provider.OPENAI,
    "gpt-4.1-mini": Provider.OPENAI,
    "gpt-5": Provider.OPENAI,
    "gpt-5-mini": Provider.OPENAI,
    # Anthropic models
    "claude-3-5-sonnet": Provider.ANTHROPIC,
    "claude-3-5-haiku-20241022": Provider.ANTHROPIC,
    "claude-3-opus": Provider.ANTHROPIC,
    # Gemini models
    "gemini-2.0-flash-exp": Provider.GEMINI,
    "gemini-1.5-pro": Provider.GEMINI,
    "gemini-1.5-flash": Provider.GEMINI,
}


class ProviderRegistry:
    """Provider registry managing model-to-provider mapping and provider instances.

    This class implements the central registry for all LLM providers. It manages
    the lifecycle of provider instances using singleton pattern and lazy initialization.

    The registry ensures that:
    - Each provider is instantiated at most once (singleton)
    - Providers are created only when needed (lazy initialization)
    - Models are validated against ALLOWED_MODELS configuration
    - API keys are validated before provider creation
    - Provider selection is deterministic and configuration-driven

    Attributes:
        _instances: Singleton cache of provider instances (class variable)
        timeout: Request timeout in seconds (default 60.0)
        max_retries: Maximum retry attempts on transient errors (default 3)
    """

    # Class variable for singleton instances
    _instances: dict[Provider, LLMClient] = {}

    def __init__(self, timeout: float = 60.0, max_retries: int = 3) -> None:
        """Initialize provider registry with timeout and retry configuration.

        Args:
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum number of retry attempts (default 3)
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def get_provider_for_model(self, model: str) -> LLMClient:
        """Get provider client for the specified model.

        This method implements the core provider selection logic. It:
        1. Validates model is in ALLOWED_MODELS configuration
        2. Looks up provider in MODEL_PROVIDER_MAP
        3. Creates provider instance if not already cached (lazy initialization)
        4. Returns cached instance if already created (singleton)

        Args:
            model: Model identifier (e.g., "gpt-4.1-mini", "claude-3-5-haiku-20241022")

        Returns:
            LLMClient instance for the provider that handles this model

        Raises:
            ModelNotAllowedError: When model is not in ALLOWED_MODELS configuration
            ValueError: When model is unknown (not in MODEL_PROVIDER_MAP)
            RuntimeError: When API key is not configured for the provider
        """
        # Validate model is in ALLOWED_MODELS
        if model not in settings.ALLOWED_MODELS:
            raise ModelNotAllowedError(model, settings.ALLOWED_MODELS)

        # Look up provider in mapping
        provider = MODEL_PROVIDER_MAP.get(model)
        if provider is None:
            raise ValueError(
                f"Unknown model: {model}. Available models: {list(MODEL_PROVIDER_MAP.keys())}"
            )

        # Create provider instance if not already cached (lazy initialization)
        if provider not in self._instances:
            self._instances[provider] = self._create_provider(provider)

        return self._instances[provider]

    def list_available_models(self) -> list[str]:
        """List all available models based on ALLOWED_MODELS configuration.

        Returns intersection of MODEL_PROVIDER_MAP keys and ALLOWED_MODELS.
        This ensures only models that are both supported and allowed are listed.

        Returns:
            List of available model identifiers sorted alphabetically
        """
        # Get intersection of mapped models and allowed models
        available = set(MODEL_PROVIDER_MAP.keys()) & set(settings.ALLOWED_MODELS)
        return sorted(available)

    def _create_provider(self, provider: Provider) -> LLMClient:
        """Create provider client instance with appropriate API key.

        This private method handles provider instantiation with validation:
        1. Retrieves API key from settings based on provider type
        2. Validates API key is configured (not None)
        3. Creates provider instance with timeout and retry settings

        Args:
            provider: Provider enum value (OPENAI, ANTHROPIC, or GEMINI)

        Returns:
            LLMClient instance for the specified provider

        Raises:
            RuntimeError: When API key is not configured for the provider
        """
        import logging
        import time

        logger = logging.getLogger(__name__)

        # Map provider to API key and client class
        if provider == Provider.OPENAI:
            api_key = settings.OPENAI_API_KEY
            if api_key is None:
                logger.error(
                    "AUDIT: Model governance violation - missing API key",
                    extra={
                        "audit_event": "governance_violation",
                        "violation_type": "missing_api_key",
                        "timestamp": time.time(),
                        "provider": provider.value,
                        "reason": "OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
                        "severity": "critical",
                    },
                )
                raise RuntimeError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
                )
            return LLMClientOpenAI(
                api_key=api_key, timeout=self.timeout, max_retries=self.max_retries
            )

        if provider == Provider.ANTHROPIC:
            api_key = settings.ANTHROPIC_API_KEY
            if api_key is None:
                logger.error(
                    "AUDIT: Model governance violation - missing API key",
                    extra={
                        "audit_event": "governance_violation",
                        "violation_type": "missing_api_key",
                        "timestamp": time.time(),
                        "provider": provider.value,
                        "reason": "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.",
                        "severity": "critical",
                    },
                )
                raise RuntimeError(
                    "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
                )
            return LLMClientAnthropic(
                api_key=api_key, timeout=self.timeout, max_retries=self.max_retries
            )

        if provider == Provider.GEMINI:
            api_key = settings.GEMINI_API_KEY
            if api_key is None:
                logger.error(
                    "AUDIT: Model governance violation - missing API key",
                    extra={
                        "audit_event": "governance_violation",
                        "violation_type": "missing_api_key",
                        "timestamp": time.time(),
                        "provider": provider.value,
                        "reason": "Gemini API key not configured. Set GEMINI_API_KEY environment variable.",
                        "severity": "critical",
                    },
                )
                raise RuntimeError(
                    "Gemini API key not configured. Set GEMINI_API_KEY environment variable."
                )
            return LLMClientGemini(
                api_key=api_key, timeout=self.timeout, max_retries=self.max_retries
            )

        # This should never happen if Provider enum is complete
        raise ValueError(f"Unknown provider: {provider}")


class LLMService:
    """Main service interface for multi-provider LLM operations.

    This class implements the Facade pattern, providing a unified interface for
    LLM completions across OpenAI, Anthropic, and Gemini providers. It orchestrates
    provider selection, model governance, A2A context propagation, and metrics collection.

    The service is the main entry point for all LLM operations in AgentCore and
    implements the following key responsibilities:

    1. Model Governance: Validates all requests against ALLOWED_MODELS configuration
    2. Provider Selection: Routes requests to appropriate provider via ProviderRegistry
    3. A2A Context Propagation: Ensures trace_id and source_agent flow through all layers
    4. Error Handling: Provides meaningful errors with proper context
    5. Structured Logging: Logs all operations with trace_id, model, provider, latency
    6. Request/Response Normalization: Unified interface regardless of provider

    This is the CRITICAL PATH service - all LLM operations flow through this interface.
    It blocks 6 downstream tickets (LLM-CLIENT-010, 011, 012, 013, 018, 019).

    Attributes:
        registry: ProviderRegistry for provider selection
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts on transient errors
        logger: Structured logger for request tracking

    Example:
        ```python
        from agentcore.a2a_protocol.services.llm_service import llm_service
        from agentcore.a2a_protocol.models.llm import LLMRequest

        # Non-streaming completion
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello"}],
            trace_id="trace-123",
            source_agent="agent-001",
        )
        response = await llm_service.complete(request)
        print(response.content)  # "Hello! How can I help you today?"

        # Streaming completion
        async for token in llm_service.stream(request):
            print(token, end="", flush=True)
        ```
    """

    def __init__(
        self, timeout: float | None = None, max_retries: int | None = None
    ) -> None:
        """Initialize LLM service with optional timeout and retry configuration.

        Args:
            timeout: Request timeout in seconds (default from settings.LLM_REQUEST_TIMEOUT)
            max_retries: Maximum retry attempts (default from settings.LLM_MAX_RETRIES)
        """
        import logging

        self.timeout = timeout if timeout is not None else settings.LLM_REQUEST_TIMEOUT
        self.max_retries = (
            max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        )
        self.registry = ProviderRegistry(timeout=self.timeout, max_retries=self.max_retries)
        self.logger = logging.getLogger(__name__)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute non-streaming LLM completion with model governance and A2A context.

        This is the main entry point for non-streaming LLM operations. It implements
        the complete flow:

        1. Model Governance: Validates request.model is in ALLOWED_MODELS
        2. Provider Selection: Gets appropriate provider from registry
        3. Request Execution: Calls provider.complete() with error handling
        4. Structured Logging: Logs completion with trace_id, model, provider, latency

        The method enforces CLAUDE.md governance rules by rejecting non-allowed models
        BEFORE calling providers, preventing cost overruns and ensuring policy compliance.

        Args:
            request: Unified LLM request with model, messages, temperature, max_tokens,
                and A2A context (trace_id, source_agent, session_id).

        Returns:
            Normalized LLM response with content, usage statistics, latency,
            provider information, and propagated A2A context.

        Raises:
            ModelNotAllowedError: When request.model is not in ALLOWED_MODELS configuration
            ProviderError: When provider API returns an error (authentication, rate limit, etc.)
            ProviderTimeoutError: When request exceeds timeout limit
            RuntimeError: When provider API key is not configured

        Example:
            >>> request = LLMRequest(
            ...     model="gpt-4.1-mini",
            ...     messages=[{"role": "user", "content": "Explain async/await"}],
            ...     temperature=0.7,
            ...     max_tokens=200,
            ...     trace_id="trace-abc-123",
            ... )
            >>> response = await llm_service.complete(request)
            >>> print(f"Provider: {response.provider}, Tokens: {response.usage.total_tokens}")
            Provider: openai, Tokens: 57
        """
        import time

        # Step 1: Model governance check
        # This MUST happen before provider selection to enforce ALLOWED_MODELS policy
        if request.model not in settings.ALLOWED_MODELS:
            # Structured audit log entry for governance violation
            self.logger.warning(
                "AUDIT: Model governance violation - disallowed model",
                extra={
                    "audit_event": "governance_violation",
                    "violation_type": "disallowed_model",
                    "timestamp": time.time(),
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "session_id": request.session_id,
                    "attempted_model": request.model,
                    "allowed_models": settings.ALLOWED_MODELS,
                    "reason": f"Model '{request.model}' is not in ALLOWED_MODELS configuration",
                    "severity": "high",
                },
            )
            # Record governance violation for monitoring
            record_governance_violation(request.model, request.source_agent)
            raise ModelNotAllowedError(request.model, settings.ALLOWED_MODELS)

        # Step 2: Provider selection via registry
        try:
            provider = self.registry.get_provider_for_model(request.model)
        except (ValueError, RuntimeError) as e:
            self.logger.error(
                "Provider selection failed",
                extra={
                    "model": request.model,
                    "error": str(e),
                    "trace_id": request.trace_id,
                },
            )
            raise

        # Step 3: Execute request with timing and metrics tracking
        start_time = time.time()
        provider_name = MODEL_PROVIDER_MAP[request.model].value

        try:
            with track_active_requests(provider_name):
                response = await provider.complete(request)

            # Calculate duration in seconds for metrics
            duration_seconds = time.time() - start_time
            elapsed_ms = int(duration_seconds * 1000)

            # Record metrics for successful request
            record_llm_request(provider_name, request.model, "success")
            record_llm_duration(provider_name, request.model, duration_seconds)
            record_llm_tokens(
                provider_name, request.model, "prompt", response.usage.prompt_tokens
            )
            record_llm_tokens(
                provider_name,
                request.model,
                "completion",
                response.usage.completion_tokens,
            )

            # Step 4: Structured logging with all context
            self.logger.info(
                "LLM completion succeeded",
                extra={
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "session_id": request.session_id,
                    "model": request.model,
                    "provider": response.provider,
                    "latency_ms": elapsed_ms,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

            return response

        except RateLimitError as e:
            duration_seconds = time.time() - start_time
            elapsed_ms = int(duration_seconds * 1000)

            # Record error metrics
            record_llm_request(provider_name, request.model, "error")
            record_llm_error(provider_name, request.model, "RateLimitError")

            self.logger.error(
                "LLM completion rate limited",
                extra={
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "model": request.model,
                    "provider": provider.__class__.__name__,
                    "error_type": "RateLimitError",
                    "error_message": str(e),
                    "latency_ms": elapsed_ms,
                    "retry_after": e.retry_after,
                },
            )
            raise

        except (ProviderError, ProviderTimeoutError) as e:
            duration_seconds = time.time() - start_time
            elapsed_ms = int(duration_seconds * 1000)

            # Record error metrics
            error_type = type(e).__name__
            record_llm_request(provider_name, request.model, "error")
            record_llm_error(provider_name, request.model, error_type)

            self.logger.error(
                "LLM completion failed",
                extra={
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "model": request.model,
                    "provider": provider.__class__.__name__,
                    "error_type": error_type,
                    "error_message": str(e),
                    "latency_ms": elapsed_ms,
                },
            )
            raise

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Execute streaming LLM completion with model governance and A2A context.

        This method provides streaming completions where tokens are yielded as they
        are generated by the provider. It implements the same governance and error
        handling as complete(), but returns an async iterator instead of a single response.

        Streaming is useful for:
        - Real-time user interfaces (showing progress)
        - Long-form content generation
        - Reducing perceived latency (time to first token)

        The method follows the same governance flow:
        1. Model Governance: Validates request.model is in ALLOWED_MODELS
        2. Provider Selection: Gets appropriate provider from registry
        3. Stream Execution: Yields tokens from provider.stream()
        4. Structured Logging: Logs stream start and completion

        Args:
            request: Unified LLM request with model, messages, temperature, max_tokens,
                and A2A context (trace_id, source_agent, session_id).

        Yields:
            Content tokens as strings. Each yield represents a chunk of generated text.
            Tokens are yielded in order and should be concatenated to reconstruct
            the full response.

        Raises:
            ModelNotAllowedError: When request.model is not in ALLOWED_MODELS configuration
            ProviderError: When provider API returns an error during streaming
            ProviderTimeoutError: When stream does not produce tokens within timeout
            RuntimeError: When provider API key is not configured

        Example:
            >>> request = LLMRequest(
            ...     model="claude-3-5-haiku-20241022",
            ...     messages=[{"role": "user", "content": "Count to 5"}],
            ...     stream=True,
            ...     trace_id="trace-xyz-789",
            ... )
            >>> async for token in llm_service.stream(request):
            ...     print(token, end="", flush=True)
            1 2 3 4 5

        Notes:
            - Tokens are yielded immediately as received (no buffering)
            - Final usage statistics are not available during streaming
            - Latency metrics track time to first token separately
            - Client must handle stream interruptions (e.g., network errors)
        """
        import time

        # Step 1: Model governance check (same as complete)
        if request.model not in settings.ALLOWED_MODELS:
            # Structured audit log entry for governance violation
            self.logger.warning(
                "AUDIT: Model governance violation - disallowed model (streaming)",
                extra={
                    "audit_event": "governance_violation",
                    "violation_type": "disallowed_model",
                    "timestamp": time.time(),
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "session_id": request.session_id,
                    "attempted_model": request.model,
                    "allowed_models": settings.ALLOWED_MODELS,
                    "reason": f"Model '{request.model}' is not in ALLOWED_MODELS configuration",
                    "severity": "high",
                    "request_type": "streaming",
                },
            )
            # Record governance violation for monitoring
            record_governance_violation(request.model, request.source_agent)
            raise ModelNotAllowedError(request.model, settings.ALLOWED_MODELS)

        # Step 2: Provider selection via registry
        try:
            provider = self.registry.get_provider_for_model(request.model)
        except (ValueError, RuntimeError) as e:
            self.logger.error(
                "Provider selection failed (streaming)",
                extra={
                    "model": request.model,
                    "error": str(e),
                    "trace_id": request.trace_id,
                },
            )
            raise

        # Log streaming request start
        start_time = time.time()
        provider_name = MODEL_PROVIDER_MAP[request.model].value

        self.logger.info(
            "LLM streaming started",
            extra={
                "trace_id": request.trace_id,
                "source_agent": request.source_agent,
                "session_id": request.session_id,
                "model": request.model,
                "provider": provider.__class__.__name__,
            },
        )

        # Step 3: Execute streaming request and yield tokens with metrics tracking
        try:
            token_count = 0
            with track_active_requests(provider_name):
                # Note: provider.stream() is an async generator, not a coroutine
                # The abstract base class signature doesn't match the implementation
                # (this is a known pattern for async generators in Python)
                async for token in provider.stream(request):  # type: ignore[attr-defined]
                    token_count += 1
                    yield token

            # Calculate duration in seconds for metrics
            duration_seconds = time.time() - start_time
            elapsed_ms = int(duration_seconds * 1000)

            # Record metrics for successful streaming request
            # Note: Token counts are not available for streaming, so we only record request metrics
            record_llm_request(provider_name, request.model, "success")
            record_llm_duration(provider_name, request.model, duration_seconds)

            # Log streaming completion
            self.logger.info(
                "LLM streaming completed",
                extra={
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "model": request.model,
                    "provider": provider.__class__.__name__,
                    "latency_ms": elapsed_ms,
                    "token_chunks": token_count,
                },
            )

        except RateLimitError as e:
            duration_seconds = time.time() - start_time
            elapsed_ms = int(duration_seconds * 1000)

            # Record error metrics
            record_llm_request(provider_name, request.model, "error")
            record_llm_error(provider_name, request.model, "RateLimitError")

            self.logger.error(
                "LLM streaming rate limited",
                extra={
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "model": request.model,
                    "provider": provider.__class__.__name__,
                    "error_type": "RateLimitError",
                    "error_message": str(e),
                    "latency_ms": elapsed_ms,
                    "retry_after": e.retry_after,
                },
            )
            raise

        except (ProviderError, ProviderTimeoutError) as e:
            duration_seconds = time.time() - start_time
            elapsed_ms = int(duration_seconds * 1000)

            # Record error metrics
            error_type = type(e).__name__
            record_llm_request(provider_name, request.model, "error")
            record_llm_error(provider_name, request.model, error_type)

            self.logger.error(
                "LLM streaming failed",
                extra={
                    "trace_id": request.trace_id,
                    "source_agent": request.source_agent,
                    "model": request.model,
                    "provider": provider.__class__.__name__,
                    "error_type": error_type,
                    "error_message": str(e),
                    "latency_ms": elapsed_ms,
                },
            )
            raise


# Global singleton instance for easy access
# This is the main entry point for all LLM operations in AgentCore
llm_service = LLMService()
