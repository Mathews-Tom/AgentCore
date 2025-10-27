"""Tests for Portkey data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcore.integration.portkey.models import (
    CacheConfig,
    LLMRequest,
    LLMResponse,
    ModelRequirements,
    ProviderConfig,
    RetryConfig,
    RoutingStrategy,
)


class TestModelRequirements:
    """Test suite for ModelRequirements."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        requirements = ModelRequirements()

        assert requirements.capabilities == []
        assert requirements.max_cost_per_token is None
        assert requirements.max_latency_ms is None
        assert requirements.data_residency is None
        assert requirements.preferred_providers == []

    def test_with_capabilities(self) -> None:
        """Test setting capabilities."""
        requirements = ModelRequirements(
            capabilities=["text_generation", "reasoning"],
            max_cost_per_token=0.001,
            max_latency_ms=2000,
        )

        assert requirements.capabilities == ["text_generation", "reasoning"]
        assert requirements.max_cost_per_token == 0.001
        assert requirements.max_latency_ms == 2000

    def test_negative_cost_validation(self) -> None:
        """Test that negative cost is rejected."""
        with pytest.raises(ValidationError):
            ModelRequirements(max_cost_per_token=-0.001)

    def test_negative_latency_validation(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError):
            ModelRequirements(max_latency_ms=-100)


class TestLLMRequest:
    """Test suite for LLMRequest."""

    def test_minimal_request(self) -> None:
        """Test creating minimal valid request."""
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.stream is False
        assert request.model_requirements is None
        assert request.context == {}

    def test_full_request(self) -> None:
        """Test creating request with all fields."""
        requirements = ModelRequirements(capabilities=["text_generation"])
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            model_requirements=requirements,
            context={"agent_id": "test-agent"},
        )

        assert request.model == "gpt-4"
        assert request.stream is True
        assert request.model_requirements == requirements
        assert request.context["agent_id"] == "test-agent"

class TestLLMResponse:
    """Test suite for LLMResponse."""

    def test_minimal_response(self) -> None:
        """Test creating minimal valid response."""
        response = LLMResponse(
            id="chatcmpl-123",
            model="gpt-4",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        )

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.provider is None
        assert response.usage is None
        assert response.cost is None
        assert response.latency_ms is None
        assert response.metadata == {}

    def test_full_response(self) -> None:
        """Test creating response with all fields."""
        response = LLMResponse(
            id="chatcmpl-123",
            model="gpt-4",
            provider="openai",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost=0.001,
            latency_ms=1500,
            metadata={"trace_id": "trace-123"},
        )

        assert response.provider == "openai"
        assert response.usage is not None
        assert response.cost == 0.001
        assert response.latency_ms == 1500
        assert response.metadata["trace_id"] == "trace-123"

    def test_cost_validation(self) -> None:
        """Test cost validation."""
        # Valid costs
        LLMResponse(id="1", model="gpt-4", choices=[], cost=0.0)
        LLMResponse(id="1", model="gpt-4", choices=[], cost=1.5)

        # Invalid cost
        with pytest.raises(ValidationError):
            LLMResponse(id="1", model="gpt-4", choices=[], cost=-0.001)

    def test_latency_validation(self) -> None:
        """Test latency validation."""
        # Valid latencies
        LLMResponse(id="1", model="gpt-4", choices=[], latency_ms=0)
        LLMResponse(id="1", model="gpt-4", choices=[], latency_ms=1000)

        # Invalid latency
        with pytest.raises(ValidationError):
            LLMResponse(id="1", model="gpt-4", choices=[], latency_ms=-100)


class TestProviderConfig:
    """Test suite for ProviderConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ProviderConfig(provider="openai")

        assert config.provider == "openai"
        assert config.virtual_key is None
        assert config.override_params == {}
        assert config.weight == 1
        assert config.retry_config is None

    def test_with_retry_config(self) -> None:
        """Test provider config with retry configuration."""
        retry_config = RetryConfig(attempts=5, initial_delay_ms=500)
        config = ProviderConfig(
            provider="anthropic",
            virtual_key="vk-123",
            weight=2,
            retry_config=retry_config,
        )

        assert config.provider == "anthropic"
        assert config.virtual_key == "vk-123"
        assert config.weight == 2
        assert config.retry_config == retry_config


class TestRetryConfig:
    """Test suite for RetryConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = RetryConfig()

        assert config.attempts == 3
        assert 429 in config.on_status_codes
        assert 500 in config.on_status_codes
        assert config.initial_delay_ms == 1000
        assert config.max_delay_ms == 60000
        assert config.exponential_base == 2.0

    def test_custom_values(self) -> None:
        """Test custom retry configuration."""
        config = RetryConfig(
            attempts=5,
            on_status_codes=[429, 503],
            initial_delay_ms=500,
            max_delay_ms=30000,
            exponential_base=1.5,
        )

        assert config.attempts == 5
        assert config.on_status_codes == [429, 503]
        assert config.initial_delay_ms == 500
        assert config.max_delay_ms == 30000
        assert config.exponential_base == 1.5

    def test_attempts_validation(self) -> None:
        """Test attempts validation."""
        # Valid attempts
        RetryConfig(attempts=1)
        RetryConfig(attempts=10)

        # Invalid attempts
        with pytest.raises(ValidationError):
            RetryConfig(attempts=0)

        with pytest.raises(ValidationError):
            RetryConfig(attempts=11)


class TestCacheConfig:
    """Test suite for CacheConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = CacheConfig()

        assert config.mode == "simple"
        assert config.max_age_seconds == 3600
        assert config.semantic_threshold == 0.95
        assert config.force_refresh is False

    def test_semantic_mode(self) -> None:
        """Test semantic caching configuration."""
        config = CacheConfig(
            mode="semantic",
            max_age_seconds=7200,
            semantic_threshold=0.9,
        )

        assert config.mode == "semantic"
        assert config.max_age_seconds == 7200
        assert config.semantic_threshold == 0.9

    def test_semantic_threshold_validation(self) -> None:
        """Test semantic threshold validation."""
        # Valid thresholds
        CacheConfig(semantic_threshold=0.0)
        CacheConfig(semantic_threshold=0.5)
        CacheConfig(semantic_threshold=1.0)

        # Invalid thresholds
        with pytest.raises(ValidationError):
            CacheConfig(semantic_threshold=-0.1)

        with pytest.raises(ValidationError):
            CacheConfig(semantic_threshold=1.1)


class TestRoutingStrategy:
    """Test suite for RoutingStrategy."""

    def test_loadbalance_mode(self) -> None:
        """Test load balance routing strategy."""
        targets = [
            ProviderConfig(provider="openai", weight=2),
            ProviderConfig(provider="anthropic", weight=1),
        ]
        strategy = RoutingStrategy(mode="loadbalance", targets=targets)

        assert strategy.mode == "loadbalance"
        assert len(strategy.targets) == 2
        assert strategy.targets[0].provider == "openai"

    def test_fallback_mode(self) -> None:
        """Test fallback routing strategy."""
        targets = [
            ProviderConfig(provider="openai"),
            ProviderConfig(provider="anthropic"),
        ]
        strategy = RoutingStrategy(
            mode="fallback",
            targets=targets,
            on_status_codes=[429, 503],
        )

        assert strategy.mode == "fallback"
        assert len(strategy.targets) == 2
        assert strategy.on_status_codes == [429, 503]

    def test_cost_optimized_mode(self) -> None:
        """Test cost-optimized routing strategy."""
        targets = [ProviderConfig(provider="openai")]
        strategy = RoutingStrategy(mode="cost_optimized", targets=targets)

        assert strategy.mode == "cost_optimized"
        assert len(strategy.targets) == 1
