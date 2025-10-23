"""Provider management for LLM orchestration.

Manages 1600+ LLM providers with capabilities mapping, health monitoring,
and automatic failover mechanisms.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderCapability(str, Enum):
    """Capabilities supported by LLM providers."""

    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    VISION = "vision"
    AUDIO = "audio"
    IMAGE_GENERATION = "image_generation"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"


class ProviderStatus(str, Enum):
    """Operational status of a provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class DataResidency(str, Enum):
    """Supported data residency regions."""

    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    GLOBAL = "global"


class ProviderMetadata(BaseModel):
    """Metadata about a provider."""

    name: str = Field(
        description="Provider display name",
    )
    description: str | None = Field(
        default=None,
        description="Provider description",
    )
    website: str | None = Field(
        default=None,
        description="Provider website URL",
    )
    documentation_url: str | None = Field(
        default=None,
        description="Provider API documentation URL",
    )
    support_email: str | None = Field(
        default=None,
        description="Provider support contact",
    )


class ProviderPricing(BaseModel):
    """Provider pricing information."""

    input_token_price: float = Field(
        description="Price per 1K input tokens in USD",
        ge=0.0,
    )
    output_token_price: float = Field(
        description="Price per 1K output tokens in USD",
        ge=0.0,
    )
    currency: str = Field(
        default="USD",
        description="Pricing currency",
    )
    free_tier: bool = Field(
        default=False,
        description="Whether provider offers a free tier",
    )
    rate_limits: dict[str, int] = Field(
        default_factory=dict,
        description="Rate limits (requests_per_minute, tokens_per_day, etc.)",
    )


class ProviderCapabilities(BaseModel):
    """Capabilities and features supported by a provider."""

    capabilities: list[ProviderCapability] = Field(
        default_factory=list,
        description="List of supported capabilities",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens supported",
        ge=1,
    )
    context_window: int | None = Field(
        default=None,
        description="Context window size in tokens",
        ge=1,
    )
    supports_system_messages: bool = Field(
        default=True,
        description="Whether provider supports system messages",
    )
    supports_function_calling: bool = Field(
        default=False,
        description="Whether provider supports function calling",
    )
    supports_streaming: bool = Field(
        default=True,
        description="Whether provider supports streaming responses",
    )
    supports_json_mode: bool = Field(
        default=False,
        description="Whether provider supports JSON mode",
    )
    data_residency: list[DataResidency] = Field(
        default_factory=lambda: [DataResidency.GLOBAL],
        description="Supported data residency regions",
    )


class ProviderHealthMetrics(BaseModel):
    """Health metrics for a provider."""

    status: ProviderStatus = Field(
        description="Current operational status",
    )
    last_check: datetime = Field(
        description="Timestamp of last health check",
    )
    success_rate: float = Field(
        default=1.0,
        description="Success rate (0.0-1.0) over last N requests",
        ge=0.0,
        le=1.0,
    )
    average_latency_ms: int | None = Field(
        default=None,
        description="Average response latency in milliseconds",
        ge=0,
    )
    error_count: int = Field(
        default=0,
        description="Number of errors in monitoring window",
        ge=0,
    )
    total_requests: int = Field(
        default=0,
        description="Total requests in monitoring window",
        ge=0,
    )
    consecutive_failures: int = Field(
        default=0,
        description="Number of consecutive failures",
        ge=0,
    )
    last_error: str | None = Field(
        default=None,
        description="Last error message if any",
    )
    availability_percent: float = Field(
        default=100.0,
        description="Availability percentage over monitoring window",
        ge=0.0,
        le=100.0,
    )


class ProviderConfiguration(BaseModel):
    """Complete provider configuration."""

    provider_id: str = Field(
        description="Unique provider identifier (e.g., 'openai', 'anthropic')",
    )
    virtual_key: str | None = Field(
        default=None,
        description="Portkey virtual key for this provider",
    )
    enabled: bool = Field(
        default=True,
        description="Whether provider is enabled for use",
    )
    priority: int = Field(
        default=100,
        description="Provider priority (higher = preferred)",
        ge=0,
    )
    weight: int = Field(
        default=1,
        description="Load balancing weight",
        ge=1,
    )
    metadata: ProviderMetadata = Field(
        description="Provider metadata",
    )
    capabilities: ProviderCapabilities = Field(
        description="Provider capabilities and features",
    )
    pricing: ProviderPricing | None = Field(
        default=None,
        description="Provider pricing information",
    )
    health: ProviderHealthMetrics | None = Field(
        default=None,
        description="Current health metrics",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(),
        description="Circuit breaker configuration",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Custom tags for filtering and grouping",
    )
    custom_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific custom configuration",
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for provider resilience."""

    failure_threshold: int = Field(
        default=5,
        description="Number of consecutive failures before opening circuit",
        ge=1,
        le=100,
    )
    success_threshold: int = Field(
        default=2,
        description="Number of consecutive successes to close circuit",
        ge=1,
        le=10,
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout before attempting to close circuit",
        ge=1,
    )
    half_open_requests: int = Field(
        default=1,
        description="Number of test requests in half-open state",
        ge=1,
        le=10,
    )


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if provider has recovered


class ProviderCircuitBreaker(BaseModel):
    """Runtime circuit breaker state for a provider."""

    provider_id: str = Field(
        description="Provider identifier",
    )
    state: CircuitBreakerState = Field(
        default=CircuitBreakerState.CLOSED,
        description="Current circuit breaker state",
    )
    consecutive_failures: int = Field(
        default=0,
        description="Number of consecutive failures",
        ge=0,
    )
    consecutive_successes: int = Field(
        default=0,
        description="Number of consecutive successes",
        ge=0,
    )
    last_failure_time: datetime | None = Field(
        default=None,
        description="Timestamp of last failure",
    )
    opened_at: datetime | None = Field(
        default=None,
        description="Timestamp when circuit was opened",
    )
    config: CircuitBreakerConfig = Field(
        description="Circuit breaker configuration",
    )


class ProviderSelectionCriteria(BaseModel):
    """Criteria for selecting a provider from the pool."""

    required_capabilities: list[ProviderCapability] = Field(
        default_factory=list,
        description="Required provider capabilities",
    )
    max_cost_per_1k_tokens: float | None = Field(
        default=None,
        description="Maximum acceptable cost per 1K tokens in USD",
        ge=0.0,
    )
    max_latency_ms: int | None = Field(
        default=None,
        description="Maximum acceptable latency in milliseconds",
        ge=0,
    )
    data_residency: DataResidency | None = Field(
        default=None,
        description="Required data residency region",
    )
    preferred_providers: list[str] = Field(
        default_factory=list,
        description="Preferred provider IDs in order of preference",
    )
    excluded_providers: list[str] = Field(
        default_factory=list,
        description="Provider IDs to exclude from selection",
    )
    min_success_rate: float = Field(
        default=0.95,
        description="Minimum acceptable success rate (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    require_healthy: bool = Field(
        default=True,
        description="Only select providers with healthy status",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Required provider tags",
    )


class ProviderSelectionResult(BaseModel):
    """Result of provider selection process."""

    provider: ProviderConfiguration = Field(
        description="Selected provider",
    )
    fallback_providers: list[ProviderConfiguration] = Field(
        default_factory=list,
        description="Fallback providers in order of preference",
    )
    selection_reason: str = Field(
        description="Reason for selecting this provider",
    )
    estimated_cost: float | None = Field(
        default=None,
        description="Estimated cost for request in USD",
        ge=0.0,
    )
    expected_latency_ms: int | None = Field(
        default=None,
        description="Expected latency in milliseconds",
        ge=0,
    )
