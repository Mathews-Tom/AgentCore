"""Unit tests for LLM Gateway data models.

This module tests the Pydantic models for LLM requests, responses, and configurations:
- ModelRequirements: Model selection criteria validation
- LLMRequest: Request model validation and constraints
- LLMResponse: Response model validation and cost tracking
- ProviderConfig: Provider configuration validation
- RetryConfig: Retry configuration and backoff validation
- CacheConfig: Cache configuration validation
- RoutingStrategy: Routing mode and target validation

Target: 90%+ code coverage
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcore.llm_gateway.models import (
    CacheConfig,
    LLMRequest,
    LLMResponse,
    ModelRequirements,
    ProviderConfig,
    RetryConfig,
    RoutingStrategy,
)


class TestModelRequirements:
    """Test suite for ModelRequirements model."""

    def test_initialization_defaults(self) -> None:
        """Test ModelRequirements with default values."""
        req = ModelRequirements()
        assert req.capabilities == []
        assert req.max_cost_per_token is None
        assert req.max_latency_ms is None
        assert req.data_residency is None
        assert req.preferred_providers == []

    def test_initialization_with_all_fields(self) -> None:
        """Test ModelRequirements with all fields populated."""
        req = ModelRequirements(
            capabilities=["text_generation", "reasoning"],
            max_cost_per_token=0.0001,
            max_latency_ms=2000,
            data_residency="us-east",
            preferred_providers=["openai", "anthropic"],
        )
        assert req.capabilities == ["text_generation", "reasoning"]
        assert req.max_cost_per_token == 0.0001
        assert req.max_latency_ms == 2000
        assert req.data_residency == "us-east"
        assert req.preferred_providers == ["openai", "anthropic"]

    def test_negative_cost_validation(self) -> None:
        """Test that negative cost per token is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRequirements(max_cost_per_token=-0.01)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_negative_latency_validation(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRequirements(max_latency_ms=-100)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_zero_cost_allowed(self) -> None:
        """Test that zero cost is allowed."""
        req = ModelRequirements(max_cost_per_token=0.0)
        assert req.max_cost_per_token == 0.0

    def test_zero_latency_allowed(self) -> None:
        """Test that zero latency is allowed."""
        req = ModelRequirements(max_latency_ms=0)
        assert req.max_latency_ms == 0

    def test_serialization(self) -> None:
        """Test model serialization to dict."""
        req = ModelRequirements(
            capabilities=["vision"],
            max_cost_per_token=0.001,
        )
        data = req.model_dump()
        assert data["capabilities"] == ["vision"]
        assert data["max_cost_per_token"] == 0.001


class TestLLMRequest:
    """Test suite for LLMRequest model."""

    def test_initialization_minimal(self) -> None:
        """Test LLMRequest with minimal required fields."""
        req = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.model == "gpt-5-mini"
        assert req.messages == [{"role": "user", "content": "Hello"}]
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.stream is False
        assert req.model_requirements is None
        assert req.context == {}

    def test_initialization_with_all_fields(self) -> None:
        """Test LLMRequest with all fields populated."""
        requirements = ModelRequirements(capabilities=["text_generation"])
        req = LLMRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1000,
            temperature=0.7,
            stream=True,
            model_requirements=requirements,
            context={"agent_id": "agent-123", "workflow_id": "wf-456"},
        )
        assert req.model == "gpt-5"
        assert req.max_tokens == 1000
        assert req.temperature == 0.7
        assert req.stream is True
        assert req.model_requirements == requirements
        assert req.context == {"agent_id": "agent-123", "workflow_id": "wf-456"}

    def test_max_tokens_validation_negative(self) -> None:
        """Test that negative max_tokens is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-5",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=-1,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_tokens_validation_zero(self) -> None:
        """Test that zero max_tokens is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-5",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_temperature_validation_negative(self) -> None:
        """Test that negative temperature is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-5",
                messages=[{"role": "user", "content": "Test"}],
                temperature=-0.1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_temperature_validation_too_high(self) -> None:
        """Test that temperature > 2.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMRequest(
                model="gpt-5",
                messages=[{"role": "user", "content": "Test"}],
                temperature=2.1,
            )
        assert "less than or equal to 2" in str(exc_info.value)

    def test_temperature_boundary_values(self) -> None:
        """Test temperature boundary values."""
        req_min = LLMRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.0,
        )
        assert req_min.temperature == 0.0

        req_max = LLMRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "Test"}],
            temperature=2.0,
        )
        assert req_max.temperature == 2.0

    def test_empty_messages(self) -> None:
        """Test request with empty messages list."""
        req = LLMRequest(
            model="gpt-5",
            messages=[],
        )
        assert req.messages == []

    def test_complex_messages(self) -> None:
        """Test request with complex message structure."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        req = LLMRequest(model="gpt-5", messages=messages)
        assert req.messages == messages


class TestLLMResponse:
    """Test suite for LLMResponse model."""

    def test_initialization_minimal(self) -> None:
        """Test LLMResponse with minimal required fields."""
        resp = LLMResponse(
            id="resp-123",
            model="gpt-5",
            choices=[{"text": "Hello!"}],
        )
        assert resp.id == "resp-123"
        assert resp.model == "gpt-5"
        assert resp.choices == [{"text": "Hello!"}]
        assert resp.provider is None
        assert resp.usage is None
        assert resp.cost is None
        assert resp.latency_ms is None
        assert resp.metadata == {}

    def test_initialization_with_all_fields(self) -> None:
        """Test LLMResponse with all fields populated."""
        resp = LLMResponse(
            id="resp-456",
            model="claude-haiku-4-5",
            provider="anthropic",
            choices=[{"message": {"role": "assistant", "content": "Test"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            cost=0.00015,
            latency_ms=1234,
            metadata={"trace_id": "trace-123"},
        )
        assert resp.id == "resp-456"
        assert resp.model == "claude-haiku-4-5"
        assert resp.provider == "anthropic"
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert resp.cost == 0.00015
        assert resp.latency_ms == 1234
        assert resp.metadata == {"trace_id": "trace-123"}

    def test_cost_validation_negative(self) -> None:
        """Test that negative cost is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(
                id="resp-123",
                model="gpt-5",
                choices=[{"text": "Test"}],
                cost=-0.01,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_latency_validation_negative(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(
                id="resp-123",
                model="gpt-5",
                choices=[{"text": "Test"}],
                latency_ms=-100,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_cost_zero_allowed(self) -> None:
        """Test that zero cost is allowed."""
        resp = LLMResponse(
            id="resp-123",
            model="gpt-5",
            choices=[{"text": "Test"}],
            cost=0.0,
        )
        assert resp.cost == 0.0

    def test_latency_zero_allowed(self) -> None:
        """Test that zero latency is allowed."""
        resp = LLMResponse(
            id="resp-123",
            model="gpt-5",
            choices=[{"text": "Test"}],
            latency_ms=0,
        )
        assert resp.latency_ms == 0

    def test_multiple_choices(self) -> None:
        """Test response with multiple choices."""
        choices = [
            {"text": "Choice 1"},
            {"text": "Choice 2"},
            {"text": "Choice 3"},
        ]
        resp = LLMResponse(
            id="resp-123",
            model="gpt-5",
            choices=choices,
        )
        assert len(resp.choices) == 3
        assert resp.choices == choices


class TestProviderConfig:
    """Test suite for ProviderConfig model."""

    def test_initialization_minimal(self) -> None:
        """Test ProviderConfig with minimal required fields."""
        config = ProviderConfig(provider="openai")
        assert config.provider == "openai"
        assert config.virtual_key is None
        assert config.override_params == {}
        assert config.weight == 1
        assert config.retry_config is None

    def test_initialization_with_all_fields(self) -> None:
        """Test ProviderConfig with all fields populated."""
        retry = RetryConfig(attempts=5)
        config = ProviderConfig(
            provider="anthropic",
            virtual_key="vk-123",
            override_params={"temperature": 0.5},
            weight=10,
            retry_config=retry,
        )
        assert config.provider == "anthropic"
        assert config.virtual_key == "vk-123"
        assert config.override_params == {"temperature": 0.5}
        assert config.weight == 10
        assert config.retry_config == retry

    def test_weight_validation_zero(self) -> None:
        """Test that zero weight is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(provider="openai", weight=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_weight_validation_negative(self) -> None:
        """Test that negative weight is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(provider="openai", weight=-1)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_weight_default(self) -> None:
        """Test default weight value."""
        config = ProviderConfig(provider="openai")
        assert config.weight == 1


class TestRetryConfig:
    """Test suite for RetryConfig model."""

    def test_initialization_defaults(self) -> None:
        """Test RetryConfig with default values."""
        config = RetryConfig()
        assert config.attempts == 3
        assert config.on_status_codes == [429, 500, 502, 503, 504]
        assert config.initial_delay_ms == 1000
        assert config.max_delay_ms == 60000
        assert config.exponential_base == 2.0

    def test_initialization_custom_values(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            attempts=5,
            on_status_codes=[429, 500],
            initial_delay_ms=500,
            max_delay_ms=30000,
            exponential_base=3.0,
        )
        assert config.attempts == 5
        assert config.on_status_codes == [429, 500]
        assert config.initial_delay_ms == 500
        assert config.max_delay_ms == 30000
        assert config.exponential_base == 3.0

    def test_attempts_validation_minimum(self) -> None:
        """Test that attempts must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            RetryConfig(attempts=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_attempts_validation_maximum(self) -> None:
        """Test that attempts cannot exceed 10."""
        with pytest.raises(ValidationError) as exc_info:
            RetryConfig(attempts=11)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_initial_delay_validation(self) -> None:
        """Test that initial_delay_ms must be at least 100."""
        with pytest.raises(ValidationError) as exc_info:
            RetryConfig(initial_delay_ms=50)
        assert "greater than or equal to 100" in str(exc_info.value)

    def test_max_delay_validation(self) -> None:
        """Test that max_delay_ms must be at least 1000."""
        with pytest.raises(ValidationError) as exc_info:
            RetryConfig(max_delay_ms=500)
        assert "greater than or equal to 1000" in str(exc_info.value)

    def test_exponential_base_validation(self) -> None:
        """Test that exponential_base must be at least 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            RetryConfig(exponential_base=0.5)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_boundary_values(self) -> None:
        """Test boundary values are accepted."""
        config = RetryConfig(
            attempts=1,
            initial_delay_ms=100,
            max_delay_ms=1000,
            exponential_base=1.0,
        )
        assert config.attempts == 1
        assert config.initial_delay_ms == 100
        assert config.max_delay_ms == 1000
        assert config.exponential_base == 1.0


class TestCacheConfig:
    """Test suite for CacheConfig model."""

    def test_initialization_defaults(self) -> None:
        """Test CacheConfig with default values."""
        config = CacheConfig()
        assert config.mode == "simple"
        assert config.max_age_seconds == 3600
        assert config.semantic_threshold == 0.95
        assert config.force_refresh is False

    def test_initialization_custom_values(self) -> None:
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            mode="semantic",
            max_age_seconds=7200,
            semantic_threshold=0.9,
            force_refresh=True,
        )
        assert config.mode == "semantic"
        assert config.max_age_seconds == 7200
        assert config.semantic_threshold == 0.9
        assert config.force_refresh is True

    def test_mode_validation(self) -> None:
        """Test that mode must be 'simple' or 'semantic'."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(mode="invalid")  # type: ignore[arg-type]
        assert "Input should be 'simple' or 'semantic'" in str(exc_info.value)

    def test_max_age_validation(self) -> None:
        """Test that max_age_seconds must be at least 60."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(max_age_seconds=30)
        assert "greater than or equal to 60" in str(exc_info.value)

    def test_semantic_threshold_validation_low(self) -> None:
        """Test that semantic_threshold must be at least 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(semantic_threshold=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_semantic_threshold_validation_high(self) -> None:
        """Test that semantic_threshold cannot exceed 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(semantic_threshold=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_semantic_threshold_boundary_values(self) -> None:
        """Test semantic_threshold boundary values."""
        config_min = CacheConfig(semantic_threshold=0.0)
        assert config_min.semantic_threshold == 0.0

        config_max = CacheConfig(semantic_threshold=1.0)
        assert config_max.semantic_threshold == 1.0


class TestRoutingStrategy:
    """Test suite for RoutingStrategy model."""

    def test_initialization_minimal(self) -> None:
        """Test RoutingStrategy with minimal required fields."""
        provider = ProviderConfig(provider="openai")
        strategy = RoutingStrategy(targets=[provider])
        assert strategy.mode == "loadbalance"
        assert len(strategy.targets) == 1
        assert strategy.targets[0].provider == "openai"
        assert strategy.on_status_codes == [429, 500, 502, 503, 504]

    def test_initialization_custom_values(self) -> None:
        """Test RoutingStrategy with custom values."""
        providers = [
            ProviderConfig(provider="openai"),
            ProviderConfig(provider="anthropic"),
        ]
        strategy = RoutingStrategy(
            mode="fallback",
            targets=providers,
            on_status_codes=[429, 500],
        )
        assert strategy.mode == "fallback"
        assert len(strategy.targets) == 2
        assert strategy.on_status_codes == [429, 500]

    def test_mode_validation(self) -> None:
        """Test that mode must be valid routing mode."""
        provider = ProviderConfig(provider="openai")
        with pytest.raises(ValidationError) as exc_info:
            RoutingStrategy(mode="invalid", targets=[provider])  # type: ignore[arg-type]
        assert "Input should be" in str(exc_info.value)

    def test_cost_optimized_mode(self) -> None:
        """Test cost_optimized routing mode."""
        providers = [
            ProviderConfig(provider="openai", weight=1),
            ProviderConfig(provider="anthropic", weight=2),
        ]
        strategy = RoutingStrategy(mode="cost_optimized", targets=providers)
        assert strategy.mode == "cost_optimized"
        assert len(strategy.targets) == 2

    def test_empty_targets(self) -> None:
        """Test routing strategy with empty targets list."""
        strategy = RoutingStrategy(targets=[])
        assert strategy.targets == []

    def test_multiple_targets_with_weights(self) -> None:
        """Test routing strategy with multiple weighted targets."""
        providers = [
            ProviderConfig(provider="openai", weight=5),
            ProviderConfig(provider="anthropic", weight=3),
            ProviderConfig(provider="gemini", weight=2),
        ]
        strategy = RoutingStrategy(mode="loadbalance", targets=providers)
        assert len(strategy.targets) == 3
        assert strategy.targets[0].weight == 5
        assert strategy.targets[1].weight == 3
        assert strategy.targets[2].weight == 2
