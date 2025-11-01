"""LLM Gateway integration data models.

Pydantic models for LLM requests, responses, and configurations.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelRequirements(BaseModel):
    """Requirements and constraints for LLM model selection.

    Defines criteria for selecting the appropriate LLM provider and model
    based on cost, performance, and capability requirements.
    """

    capabilities: list[str] = Field(
        default_factory=list,
        description="Required capabilities (e.g., 'text_generation', 'reasoning', 'vision')",
    )
    max_cost_per_token: float | None = Field(
        default=None,
        description="Maximum acceptable cost per token in USD",
        ge=0.0,
    )
    max_latency_ms: int | None = Field(
        default=None,
        description="Maximum acceptable latency in milliseconds",
        ge=0,
    )
    data_residency: str | None = Field(
        default=None,
        description="Required data residency region (e.g., 'us-east', 'eu-west')",
    )
    preferred_providers: list[str] = Field(
        default_factory=list,
        description="Preferred LLM providers in order of preference",
    )


class LLMRequest(BaseModel):
    """LLM completion request with model requirements.

    Encapsulates the request data along with model selection criteria
    and context information for routing and tracking.
    """

    model: str = Field(
        description="Target model name or identifier",
    )
    messages: list[dict[str, Any]] = Field(
        description="Conversation messages in OpenAI format",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
        ge=1,
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature for randomness",
        ge=0.0,
        le=2.0,
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    model_requirements: ModelRequirements | None = Field(
        default=None,
        description="Optional model selection requirements",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (agent_id, workflow_id, tenant_id, etc.)",
    )


class LLMResponse(BaseModel):
    """LLM completion response with metadata.

    Contains the completion result along with provider information,
    performance metrics, and cost tracking data.
    """

    id: str = Field(
        description="Unique identifier for this completion",
    )
    model: str = Field(
        description="Model that generated the response",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider that handled the request",
    )
    choices: list[dict[str, Any]] = Field(
        description="Generated completion choices",
    )
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage statistics",
    )
    cost: float | None = Field(
        default=None,
        description="Estimated cost in USD for this request",
        ge=0.0,
    )
    latency_ms: int | None = Field(
        default=None,
        description="Request latency in milliseconds",
        ge=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class ProviderConfig(BaseModel):
    """Configuration for a specific LLM provider.

    Defines connection settings, authentication, and behavior
    for connecting to an LLM provider through LLM Gateway.
    """

    provider: str = Field(
        description="Provider name (e.g., 'openai', 'anthropic', 'bedrock')",
    )
    virtual_key: str | None = Field(
        default=None,
        description="Portkey virtual key for this provider",
    )
    override_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific override parameters",
    )
    weight: int = Field(
        default=1,
        description="Load balancing weight (higher = more traffic)",
        ge=1,
    )
    retry_config: RetryConfig | None = Field(
        default=None,
        description="Retry configuration for this provider",
    )


class RetryConfig(BaseModel):
    """Retry configuration for failed requests.

    Defines retry behavior including attempts, delays, and
    exponential backoff parameters.
    """

    attempts: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=1,
        le=10,
    )
    on_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retries",
    )
    initial_delay_ms: int = Field(
        default=1000,
        description="Initial delay before first retry in milliseconds",
        ge=100,
    )
    max_delay_ms: int = Field(
        default=60000,
        description="Maximum delay between retries in milliseconds",
        ge=1000,
    )
    exponential_base: float = Field(
        default=2.0,
        description="Base for exponential backoff calculation",
        ge=1.0,
    )


class CacheConfig(BaseModel):
    """Configuration for caching LLM responses.

    Defines caching behavior including TTL, modes, and
    semantic similarity thresholds.
    """

    mode: Literal["simple", "semantic"] = Field(
        default="simple",
        description="Caching mode: 'simple' (exact match) or 'semantic' (similarity-based)",
    )
    max_age_seconds: int = Field(
        default=3600,
        description="Maximum cache entry age in seconds",
        ge=60,
    )
    semantic_threshold: float | None = Field(
        default=0.95,
        description="Similarity threshold for semantic caching (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    force_refresh: bool = Field(
        default=False,
        description="Force cache refresh for this request",
    )


class RoutingStrategy(BaseModel):
    """Strategy for routing requests to LLM providers.

    Defines how requests should be distributed across multiple
    providers based on cost, performance, and availability.
    """

    mode: Literal["loadbalance", "fallback", "cost_optimized"] = Field(
        default="loadbalance",
        description="Routing mode",
    )
    targets: list[ProviderConfig] = Field(
        description="Target providers for routing",
    )
    on_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="Status codes that trigger fallback",
    )
