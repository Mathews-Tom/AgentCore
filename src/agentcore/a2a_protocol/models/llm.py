"""LLM client data models and enums.

This module defines the foundational data structures for multi-provider LLM operations:
- Request and response models with A2A context propagation
- Provider and model tier enumerations
- Custom exceptions for error handling

All models use built-in generics (list[], dict[], int | None) per CLAUDE.md compliance.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Provider(str, Enum):
    """LLM provider enumeration.

    Supported providers for multi-provider LLM client.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ModelTier(str, Enum):
    """Model tier classification for runtime selection.

    Used by ModelSelector to choose appropriate models based on task requirements:
    - FAST: Low latency, cost-effective models (e.g., gpt-5-mini, gemini-2.5-flash-lite)
    - BALANCED: Balance of quality and cost (e.g., claude-haiku-4-5-20251001)
    - PREMIUM: Highest quality models (e.g., gpt-5, claude-sonnet-4-5-20250929)
    """

    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"


class ModelNotAllowedError(Exception):
    """Raised when requested model is not in ALLOWED_MODELS configuration.

    This enforces model governance per CLAUDE.md rules to prevent cost overruns
    and ensure only approved models are used.

    Attributes:
        model: The requested model that was rejected
        allowed: List of allowed models from configuration
    """

    def __init__(self, model: str, allowed: list[str]) -> None:
        self.model = model
        self.allowed = allowed
        super().__init__(f"Model '{model}' not allowed. Allowed models: {allowed}")


class ProviderError(Exception):
    """Raised when provider API returns an error.

    This wraps provider-specific errors (SDK exceptions, HTTP errors) into
    a unified exception type for consistent error handling.

    Attributes:
        provider: The provider that raised the error
        original_error: The original exception from the provider SDK
    """

    def __init__(self, provider: str, original_error: Exception) -> None:
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Provider '{provider}' error: {original_error}")


class ProviderTimeoutError(Exception):
    """Raised when provider request exceeds timeout limit.

    Indicates that the provider did not respond within the configured
    LLM_REQUEST_TIMEOUT period (default 60s).

    Attributes:
        provider: The provider that timed out
        timeout_seconds: The timeout value that was exceeded
    """

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Provider '{provider}' request timed out after {timeout_seconds}s"
        )


class RateLimitError(Exception):
    """Raised when provider rate limit is exceeded.

    Indicates that the provider has returned a rate limit error (429 or equivalent).
    This exception carries retry-after information when available from the provider.

    Attributes:
        provider: The provider that rate limited the request
        retry_after: Number of seconds to wait before retrying (None if not provided)
        message: Optional message from the provider
    """

    def __init__(
        self, provider: str, retry_after: float | None = None, message: str | None = None
    ) -> None:
        self.provider = provider
        self.retry_after = retry_after
        self.message = message

        msg = f"Provider '{provider}' rate limit exceeded"
        if retry_after is not None:
            msg += f", retry after {retry_after}s"
        if message:
            msg += f": {message}"
        super().__init__(msg)


class LLMRequest(BaseModel):
    """Unified LLM request model for all providers.

    This model abstracts provider-specific request formats into a single interface.
    It includes A2A protocol context fields for distributed tracing.

    Per CLAUDE.md governance: temperature and max_tokens are not exposed.
    Providers use their default values internally for consistent behavior.

    Attributes:
        model: Model identifier (must be in ALLOWED_MODELS)
        messages: Conversation messages (non-empty list)
        stream: Enable streaming response (default False)
        reasoning_effort: Reasoning effort level for reasoning models (low/medium/high)
        trace_id: A2A trace ID for distributed tracing
        source_agent: Source agent ID for request tracking
        session_id: Session ID for conversation context
    """

    model: str = Field(..., description="Model identifier")
    messages: list[dict[str, str]] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming response")
    reasoning_effort: str | None = Field(
        default=None,
        description="Reasoning effort level for reasoning models (low/medium/high)",
    )

    # A2A Context fields for distributed tracing
    trace_id: str | None = Field(
        default=None, description="A2A trace ID for distributed tracing"
    )
    source_agent: str | None = Field(
        default=None, description="Source agent ID for request tracking"
    )
    session_id: str | None = Field(
        default=None, description="Session ID for conversation context"
    )

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str | None) -> str | None:
        """Validate reasoning_effort is one of the allowed values.

        Args:
            v: reasoning_effort value to validate

        Returns:
            Validated reasoning_effort value

        Raises:
            ValueError: If reasoning_effort is not a valid value
        """
        if v is not None and v not in ("low", "medium", "high"):
            raise ValueError(
                f"reasoning_effort must be 'low', 'medium', or 'high', got '{v}'"
            )
        return v


class LLMUsage(BaseModel):
    """Token usage statistics from LLM response.

    Tracks token consumption for cost estimation and monitoring.
    All providers normalize their usage data to this format.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the generated completion
        total_tokens: Total tokens used (prompt + completion)
    """

    prompt_tokens: int = Field(..., description="Number of tokens in the input prompt")
    completion_tokens: int = Field(
        ..., description="Number of tokens in the generated completion"
    )
    total_tokens: int = Field(..., description="Total tokens used")


class LLMResponse(BaseModel):
    """Unified LLM response model from all providers.

    This model normalizes provider-specific response formats into a single interface.
    It includes usage tracking and A2A context propagation.

    Attributes:
        content: Generated text content
        usage: Token usage statistics
        latency_ms: Request latency in milliseconds
        provider: Provider that generated the response
        model: Model that was used
        trace_id: A2A trace ID (propagated from request)
    """

    content: str = Field(..., description="Generated text content")
    usage: LLMUsage = Field(..., description="Token usage statistics")
    latency_ms: int = Field(..., description="Request latency in milliseconds")
    provider: str = Field(..., description="Provider that generated the response")
    model: str = Field(..., description="Model that was used")

    # A2A Context propagation
    trace_id: str | None = Field(
        default=None, description="A2A trace ID (propagated from request)"
    )
