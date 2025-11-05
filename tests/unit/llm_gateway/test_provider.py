"""Unit tests for LLM Gateway provider models.

This module tests the provider-related Pydantic models and enums:
- ProviderCapability: Enum of supported capabilities
- ProviderStatus: Enum of operational statuses
- DataResidency: Enum of supported regions
- ProviderMetadata: Provider information
- ProviderPricing: Pricing information and rate limits
- ProviderCapabilities: Capability and feature flags
- ProviderHealthMetrics: Health monitoring metrics
- ProviderConfiguration: Complete provider configuration
- CircuitBreakerConfig: Circuit breaker settings
- CircuitBreakerState: Circuit breaker state enum
- ProviderCircuitBreaker: Runtime circuit breaker state
- ProviderSelectionCriteria: Criteria for provider selection
- ProviderSelectionResult: Provider selection results

Target: 90%+ code coverage
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentcore.llm_gateway.provider import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    DataResidency,
    ProviderCapabilities,
    ProviderCapability,
    ProviderCircuitBreaker,
    ProviderConfiguration,
    ProviderHealthMetrics,
    ProviderMetadata,
    ProviderPricing,
    ProviderSelectionCriteria,
    ProviderSelectionResult,
    ProviderStatus,
)


class TestProviderCapability:
    """Test suite for ProviderCapability enum."""

    def test_all_capabilities(self) -> None:
        """Test all provider capabilities are defined."""
        assert ProviderCapability.TEXT_GENERATION == "text_generation"
        assert ProviderCapability.CHAT_COMPLETION == "chat_completion"
        assert ProviderCapability.REASONING == "reasoning"
        assert ProviderCapability.CODE_GENERATION == "code_generation"
        assert ProviderCapability.VISION == "vision"
        assert ProviderCapability.AUDIO == "audio"
        assert ProviderCapability.IMAGE_GENERATION == "image_generation"
        assert ProviderCapability.EMBEDDINGS == "embeddings"
        assert ProviderCapability.FUNCTION_CALLING == "function_calling"
        assert ProviderCapability.STREAMING == "streaming"
        assert ProviderCapability.JSON_MODE == "json_mode"


class TestProviderStatus:
    """Test suite for ProviderStatus enum."""

    def test_all_statuses(self) -> None:
        """Test all provider statuses are defined."""
        assert ProviderStatus.HEALTHY == "healthy"
        assert ProviderStatus.DEGRADED == "degraded"
        assert ProviderStatus.UNHEALTHY == "unhealthy"
        assert ProviderStatus.UNAVAILABLE == "unavailable"
        assert ProviderStatus.MAINTENANCE == "maintenance"


class TestDataResidency:
    """Test suite for DataResidency enum."""

    def test_all_regions(self) -> None:
        """Test all data residency regions are defined."""
        assert DataResidency.US_EAST == "us-east"
        assert DataResidency.US_WEST == "us-west"
        assert DataResidency.EU_WEST == "eu-west"
        assert DataResidency.EU_CENTRAL == "eu-central"
        assert DataResidency.ASIA_PACIFIC == "asia-pacific"
        assert DataResidency.GLOBAL == "global"


class TestProviderMetadata:
    """Test suite for ProviderMetadata model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderMetadata with minimal required fields."""
        metadata = ProviderMetadata(name="OpenAI")
        assert metadata.name == "OpenAI"
        assert metadata.description is None
        assert metadata.website is None
        assert metadata.documentation_url is None
        assert metadata.support_email is None

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderMetadata with all fields populated."""
        metadata = ProviderMetadata(
            name="OpenAI",
            description="OpenAI LLM provider",
            website="https://openai.com",
            documentation_url="https://platform.openai.com/docs",
            support_email="support@openai.com",
        )
        assert metadata.name == "OpenAI"
        assert metadata.description == "OpenAI LLM provider"
        assert metadata.website == "https://openai.com"
        assert metadata.documentation_url == "https://platform.openai.com/docs"
        assert metadata.support_email == "support@openai.com"


class TestProviderPricing:
    """Test suite for ProviderPricing model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderPricing with required fields."""
        pricing = ProviderPricing(
            input_token_price=0.0001,
            output_token_price=0.0002,
        )
        assert pricing.input_token_price == 0.0001
        assert pricing.output_token_price == 0.0002
        assert pricing.currency == "USD"
        assert pricing.free_tier is False
        assert pricing.rate_limits == {}

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderPricing with all fields populated."""
        pricing = ProviderPricing(
            input_token_price=0.00015,
            output_token_price=0.0003,
            currency="EUR",
            free_tier=True,
            rate_limits={"requests_per_minute": 60, "tokens_per_day": 1000000},
        )
        assert pricing.input_token_price == 0.00015
        assert pricing.output_token_price == 0.0003
        assert pricing.currency == "EUR"
        assert pricing.free_tier is True
        assert pricing.rate_limits["requests_per_minute"] == 60
        assert pricing.rate_limits["tokens_per_day"] == 1000000

    def test_negative_input_price_rejected(self) -> None:
        """Test that negative input token price is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderPricing(
                input_token_price=-0.0001,
                output_token_price=0.0002,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_negative_output_price_rejected(self) -> None:
        """Test that negative output token price is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderPricing(
                input_token_price=0.0001,
                output_token_price=-0.0002,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_zero_prices_allowed(self) -> None:
        """Test that zero prices are allowed (free tier)."""
        pricing = ProviderPricing(
            input_token_price=0.0,
            output_token_price=0.0,
            free_tier=True,
        )
        assert pricing.input_token_price == 0.0
        assert pricing.output_token_price == 0.0
        assert pricing.free_tier is True


class TestProviderCapabilities:
    """Test suite for ProviderCapabilities model."""

    def test_initialization_defaults(self) -> None:
        """Test ProviderCapabilities with default values."""
        caps = ProviderCapabilities()
        assert caps.capabilities == []
        assert caps.max_tokens is None
        assert caps.context_window is None
        assert caps.supports_system_messages is True
        assert caps.supports_function_calling is False
        assert caps.supports_streaming is True
        assert caps.supports_json_mode is False
        assert caps.data_residency == [DataResidency.GLOBAL]

    def test_initialization_with_capabilities(self) -> None:
        """Test ProviderCapabilities with specific capabilities."""
        caps = ProviderCapabilities(
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT_COMPLETION,
                ProviderCapability.FUNCTION_CALLING,
            ],
            max_tokens=4096,
            context_window=128000,
            supports_function_calling=True,
            supports_json_mode=True,
            data_residency=[DataResidency.US_EAST, DataResidency.EU_WEST],
        )
        assert len(caps.capabilities) == 3
        assert ProviderCapability.TEXT_GENERATION in caps.capabilities
        assert ProviderCapability.FUNCTION_CALLING in caps.capabilities
        assert caps.max_tokens == 4096
        assert caps.context_window == 128000
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert len(caps.data_residency) == 2

    def test_max_tokens_validation(self) -> None:
        """Test that max_tokens must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderCapabilities(max_tokens=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_context_window_validation(self) -> None:
        """Test that context_window must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderCapabilities(context_window=0)
        assert "greater than or equal to 1" in str(exc_info.value)


class TestProviderHealthMetrics:
    """Test suite for ProviderHealthMetrics model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderHealthMetrics with required fields."""
        now = datetime.now(UTC)
        metrics = ProviderHealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=now,
        )
        assert metrics.status == ProviderStatus.HEALTHY
        assert metrics.last_check == now
        assert metrics.success_rate == 1.0
        assert metrics.average_latency_ms is None
        assert metrics.error_count == 0
        assert metrics.total_requests == 0
        assert metrics.consecutive_failures == 0
        assert metrics.last_error is None
        assert metrics.availability_percent == 100.0

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderHealthMetrics with all fields populated."""
        now = datetime.now(UTC)
        metrics = ProviderHealthMetrics(
            status=ProviderStatus.DEGRADED,
            last_check=now,
            success_rate=0.95,
            average_latency_ms=1500,
            error_count=5,
            total_requests=100,
            consecutive_failures=2,
            last_error="Rate limit exceeded",
            availability_percent=99.5,
        )
        assert metrics.status == ProviderStatus.DEGRADED
        assert metrics.success_rate == 0.95
        assert metrics.average_latency_ms == 1500
        assert metrics.error_count == 5
        assert metrics.total_requests == 100
        assert metrics.consecutive_failures == 2
        assert metrics.last_error == "Rate limit exceeded"
        assert metrics.availability_percent == 99.5

    def test_success_rate_validation(self) -> None:
        """Test success_rate must be between 0.0 and 1.0."""
        now = datetime.now(UTC)

        with pytest.raises(ValidationError) as exc_info:
            ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=now,
                success_rate=1.1,
            )
        assert "less than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=now,
                success_rate=-0.1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_availability_percent_validation(self) -> None:
        """Test availability_percent must be between 0.0 and 100.0."""
        now = datetime.now(UTC)

        with pytest.raises(ValidationError) as exc_info:
            ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=now,
                availability_percent=100.1,
            )
        assert "less than or equal to 100" in str(exc_info.value)


class TestCircuitBreakerConfig:
    """Test suite for CircuitBreakerConfig model."""

    def test_initialization_defaults(self) -> None:
        """Test CircuitBreakerConfig with default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30
        assert config.half_open_requests == 1

    def test_initialization_custom_values(self) -> None:
        """Test CircuitBreakerConfig with custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=60,
            half_open_requests=2,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60
        assert config.half_open_requests == 2

    def test_failure_threshold_validation(self) -> None:
        """Test failure_threshold validation."""
        with pytest.raises(ValidationError) as exc_info:
            CircuitBreakerConfig(failure_threshold=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CircuitBreakerConfig(failure_threshold=101)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_success_threshold_validation(self) -> None:
        """Test success_threshold validation."""
        with pytest.raises(ValidationError) as exc_info:
            CircuitBreakerConfig(success_threshold=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CircuitBreakerConfig(success_threshold=11)
        assert "less than or equal to 10" in str(exc_info.value)


class TestCircuitBreakerState:
    """Test suite for CircuitBreakerState enum."""

    def test_all_states(self) -> None:
        """Test all circuit breaker states are defined."""
        assert CircuitBreakerState.CLOSED == "closed"
        assert CircuitBreakerState.OPEN == "open"
        assert CircuitBreakerState.HALF_OPEN == "half_open"


class TestProviderCircuitBreaker:
    """Test suite for ProviderCircuitBreaker model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderCircuitBreaker with minimal fields."""
        config = CircuitBreakerConfig()
        cb = ProviderCircuitBreaker(
            provider_id="openai",
            config=config,
        )
        assert cb.provider_id == "openai"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.consecutive_failures == 0
        assert cb.consecutive_successes == 0
        assert cb.last_failure_time is None
        assert cb.opened_at is None
        assert cb.config == config

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderCircuitBreaker with all fields populated."""
        config = CircuitBreakerConfig()
        now = datetime.now(UTC)
        cb = ProviderCircuitBreaker(
            provider_id="anthropic",
            state=CircuitBreakerState.OPEN,
            consecutive_failures=5,
            consecutive_successes=0,
            last_failure_time=now,
            opened_at=now,
            config=config,
        )
        assert cb.provider_id == "anthropic"
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.consecutive_failures == 5
        assert cb.last_failure_time == now
        assert cb.opened_at == now


class TestProviderConfiguration:
    """Test suite for ProviderConfiguration model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderConfiguration with minimal fields."""
        metadata = ProviderMetadata(name="OpenAI")
        capabilities = ProviderCapabilities()
        config = ProviderConfiguration(
            provider_id="openai",
            metadata=metadata,
            capabilities=capabilities,
        )
        assert config.provider_id == "openai"
        assert config.virtual_key is None
        assert config.enabled is True
        assert config.priority == 100
        assert config.weight == 1
        assert config.metadata == metadata
        assert config.capabilities == capabilities
        assert config.pricing is None
        assert config.health is None
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert config.tags == []
        assert config.custom_config == {}

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderConfiguration with all fields populated."""
        metadata = ProviderMetadata(name="Anthropic")
        capabilities = ProviderCapabilities(
            capabilities=[ProviderCapability.TEXT_GENERATION]
        )
        pricing = ProviderPricing(
            input_token_price=0.0001,
            output_token_price=0.0002,
        )
        now = datetime.now(UTC)
        health = ProviderHealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=now,
        )
        cb_config = CircuitBreakerConfig(failure_threshold=10)

        config = ProviderConfiguration(
            provider_id="anthropic",
            virtual_key="vk-anthropic",
            enabled=False,
            priority=200,
            weight=5,
            metadata=metadata,
            capabilities=capabilities,
            pricing=pricing,
            health=health,
            circuit_breaker=cb_config,
            tags=["production", "high-quality"],
            custom_config={"region": "us-east"},
        )
        assert config.provider_id == "anthropic"
        assert config.virtual_key == "vk-anthropic"
        assert config.enabled is False
        assert config.priority == 200
        assert config.weight == 5
        assert config.pricing == pricing
        assert config.health == health
        assert config.circuit_breaker == cb_config
        assert config.tags == ["production", "high-quality"]
        assert config.custom_config == {"region": "us-east"}

    def test_priority_validation(self) -> None:
        """Test priority validation."""
        metadata = ProviderMetadata(name="Test")
        capabilities = ProviderCapabilities()

        with pytest.raises(ValidationError) as exc_info:
            ProviderConfiguration(
                provider_id="test",
                metadata=metadata,
                capabilities=capabilities,
                priority=-1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_weight_validation(self) -> None:
        """Test weight validation."""
        metadata = ProviderMetadata(name="Test")
        capabilities = ProviderCapabilities()

        with pytest.raises(ValidationError) as exc_info:
            ProviderConfiguration(
                provider_id="test",
                metadata=metadata,
                capabilities=capabilities,
                weight=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)


class TestProviderSelectionCriteria:
    """Test suite for ProviderSelectionCriteria model."""

    def test_initialization_defaults(self) -> None:
        """Test ProviderSelectionCriteria with default values."""
        criteria = ProviderSelectionCriteria()
        assert criteria.required_capabilities == []
        assert criteria.max_cost_per_1k_tokens is None
        assert criteria.max_latency_ms is None
        assert criteria.data_residency is None
        assert criteria.preferred_providers == []
        assert criteria.excluded_providers == []
        assert criteria.min_success_rate == 0.95
        assert criteria.require_healthy is True
        assert criteria.tags == []

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderSelectionCriteria with all fields populated."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            max_cost_per_1k_tokens=0.001,
            max_latency_ms=2000,
            data_residency=DataResidency.US_EAST,
            preferred_providers=["openai", "anthropic"],
            excluded_providers=["gemini"],
            min_success_rate=0.99,
            require_healthy=False,
            tags=["production"],
        )
        assert len(criteria.required_capabilities) == 1
        assert criteria.max_cost_per_1k_tokens == 0.001
        assert criteria.max_latency_ms == 2000
        assert criteria.data_residency == DataResidency.US_EAST
        assert criteria.preferred_providers == ["openai", "anthropic"]
        assert criteria.excluded_providers == ["gemini"]
        assert criteria.min_success_rate == 0.99
        assert criteria.require_healthy is False
        assert criteria.tags == ["production"]

    def test_min_success_rate_validation(self) -> None:
        """Test min_success_rate validation."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderSelectionCriteria(min_success_rate=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ProviderSelectionCriteria(min_success_rate=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)


class TestProviderSelectionResult:
    """Test suite for ProviderSelectionResult model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderSelectionResult with minimal fields."""
        metadata = ProviderMetadata(name="OpenAI")
        capabilities = ProviderCapabilities()
        provider = ProviderConfiguration(
            provider_id="openai",
            metadata=metadata,
            capabilities=capabilities,
        )
        result = ProviderSelectionResult(
            provider=provider,
            selection_reason="Best match",
        )
        assert result.provider == provider
        assert result.fallback_providers == []
        assert result.selection_reason == "Best match"
        assert result.estimated_cost is None
        assert result.expected_latency_ms is None

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderSelectionResult with all fields populated."""
        metadata = ProviderMetadata(name="OpenAI")
        capabilities = ProviderCapabilities()
        primary = ProviderConfiguration(
            provider_id="openai",
            metadata=metadata,
            capabilities=capabilities,
        )
        fallback = ProviderConfiguration(
            provider_id="anthropic",
            metadata=ProviderMetadata(name="Anthropic"),
            capabilities=capabilities,
        )
        result = ProviderSelectionResult(
            provider=primary,
            fallback_providers=[fallback],
            selection_reason="Cost-effective and healthy",
            estimated_cost=0.00015,
            expected_latency_ms=1200,
        )
        assert result.provider == primary
        assert len(result.fallback_providers) == 1
        assert result.fallback_providers[0] == fallback
        assert result.selection_reason == "Cost-effective and healthy"
        assert result.estimated_cost == 0.00015
        assert result.expected_latency_ms == 1200

    def test_estimated_cost_validation(self) -> None:
        """Test estimated_cost validation."""
        metadata = ProviderMetadata(name="Test")
        capabilities = ProviderCapabilities()
        provider = ProviderConfiguration(
            provider_id="test",
            metadata=metadata,
            capabilities=capabilities,
        )

        with pytest.raises(ValidationError) as exc_info:
            ProviderSelectionResult(
                provider=provider,
                selection_reason="Test",
                estimated_cost=-0.01,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_expected_latency_validation(self) -> None:
        """Test expected_latency_ms validation."""
        metadata = ProviderMetadata(name="Test")
        capabilities = ProviderCapabilities()
        provider = ProviderConfiguration(
            provider_id="test",
            metadata=metadata,
            capabilities=capabilities,
        )

        with pytest.raises(ValidationError) as exc_info:
            ProviderSelectionResult(
                provider=provider,
                selection_reason="Test",
                expected_latency_ms=-100,
            )
        assert "greater than or equal to 0" in str(exc_info.value)
