"""Unit tests for LLM Gateway provider registry.

This module tests the ProviderRegistry class that manages provider configurations:
- Provider registration and deregistration
- Provider retrieval and listing
- Provider filtering by capability, status, and tags
- Provider selection based on criteria
- Provider ranking and scoring
- Circuit breaker management
- File-based configuration loading/saving
- Cost-optimized provider selection
- Registry statistics

Target: 90%+ code coverage
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentcore.llm_gateway.exceptions import (
    LLMGatewayConfigurationError,
    LLMGatewayProviderError,
)
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
    ProviderStatus,
)
from agentcore.llm_gateway.registry import ProviderRegistry, get_provider_registry


@pytest.fixture
def sample_provider() -> ProviderConfiguration:
    """Create a sample provider configuration for testing."""
    return ProviderConfiguration(
        provider_id="openai",
        virtual_key="vk-openai",
        enabled=True,
        priority=100,
        weight=1,
        metadata=ProviderMetadata(name="OpenAI"),
        capabilities=ProviderCapabilities(
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT_COMPLETION,
            ],
            max_tokens=4096,
            context_window=128000,
        ),
        pricing=ProviderPricing(
            input_token_price=0.0001,
            output_token_price=0.0002,
        ),
        health=ProviderHealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=datetime.now(UTC),
            success_rate=0.99,
            average_latency_ms=1000,
        ),
    )


@pytest.fixture
def sample_provider_anthropic() -> ProviderConfiguration:
    """Create another sample provider for testing."""
    return ProviderConfiguration(
        provider_id="anthropic",
        virtual_key="vk-anthropic",
        enabled=True,
        priority=90,
        weight=2,
        metadata=ProviderMetadata(name="Anthropic"),
        capabilities=ProviderCapabilities(
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.REASONING,
            ],
            max_tokens=8192,
            context_window=200000,
        ),
        pricing=ProviderPricing(
            input_token_price=0.00015,
            output_token_price=0.0003,
        ),
        health=ProviderHealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=datetime.now(UTC),
            success_rate=0.98,
            average_latency_ms=1500,
        ),
    )


class TestProviderRegistry:
    """Test suite for ProviderRegistry class."""

    def test_initialization(self) -> None:
        """Test registry initialization."""
        registry = ProviderRegistry()
        assert len(registry._providers) == 0
        assert len(registry._circuit_breakers) == 0
        assert registry._initialized is False

    def test_register_provider(self, sample_provider: ProviderConfiguration) -> None:
        """Test registering a single provider."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        assert "openai" in registry._providers
        assert registry._providers["openai"] == sample_provider
        assert "openai" in registry._circuit_breakers
        assert isinstance(registry._circuit_breakers["openai"], ProviderCircuitBreaker)

    def test_register_provider_updates_existing(
        self, sample_provider: ProviderConfiguration
    ) -> None:
        """Test that registering existing provider updates it."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        # Update provider
        updated_provider = ProviderConfiguration(
            provider_id="openai",
            enabled=False,
            metadata=ProviderMetadata(name="OpenAI Updated"),
            capabilities=ProviderCapabilities(),
        )
        registry.register_provider(updated_provider)

        assert registry._providers["openai"].enabled is False
        assert registry._providers["openai"].metadata.name == "OpenAI Updated"

    def test_register_providers_bulk(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test bulk provider registration."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        assert len(registry._providers) == 2
        assert "openai" in registry._providers
        assert "anthropic" in registry._providers

    def test_get_provider_existing(self, sample_provider: ProviderConfiguration) -> None:
        """Test retrieving an existing provider."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        result = registry.get_provider("openai")
        assert result == sample_provider

    def test_get_provider_non_existing(self) -> None:
        """Test retrieving a non-existing provider returns None."""
        registry = ProviderRegistry()
        result = registry.get_provider("nonexistent")
        assert result is None

    def test_list_providers_all(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test listing all providers."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        providers = registry.list_providers(enabled_only=False)
        assert len(providers) == 2

    def test_list_providers_enabled_only(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test listing only enabled providers."""
        registry = ProviderRegistry()
        sample_provider.enabled = True
        disabled_provider = ProviderConfiguration(
            provider_id="disabled",
            enabled=False,
            metadata=ProviderMetadata(name="Disabled"),
            capabilities=ProviderCapabilities(),
        )
        registry.register_providers([sample_provider, disabled_provider])

        providers = registry.list_providers(enabled_only=True)
        assert len(providers) == 1
        assert providers[0].provider_id == "openai"

    def test_list_providers_by_capability(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test filtering providers by capability."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        providers = registry.list_providers(
            capability=ProviderCapability.REASONING,
            enabled_only=False,
        )
        assert len(providers) == 1
        assert providers[0].provider_id == "anthropic"

    def test_list_providers_by_status(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test filtering providers by health status."""
        registry = ProviderRegistry()
        sample_provider.health = ProviderHealthMetrics(
            status=ProviderStatus.DEGRADED,
            last_check=datetime.now(UTC),
        )
        registry.register_provider(sample_provider)

        providers = registry.list_providers(
            status=ProviderStatus.DEGRADED,
            enabled_only=False,
        )
        assert len(providers) == 1
        assert providers[0].health is not None
        assert providers[0].health.status == ProviderStatus.DEGRADED

    def test_list_providers_by_tags(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test filtering providers by tags."""
        registry = ProviderRegistry()
        sample_provider.tags = ["production", "high-quality"]
        registry.register_provider(sample_provider)

        providers = registry.list_providers(
            tags=["production"],
            enabled_only=False,
        )
        assert len(providers) == 1
        assert providers[0].provider_id == "openai"

        # Test with tag that doesn't match
        providers = registry.list_providers(
            tags=["development"],
            enabled_only=False,
        )
        assert len(providers) == 0

    def test_select_provider_simple(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test basic provider selection."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        criteria = ProviderSelectionCriteria()
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "openai"
        assert isinstance(result.selection_reason, str)

    def test_select_provider_by_capability(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test provider selection by required capability."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.REASONING]
        )
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "anthropic"

    def test_select_provider_by_cost(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test provider selection by cost constraint."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        # Set max cost that only openai meets
        # openai avg: (0.0001 + 0.0002) / 2 = 0.00015
        # anthropic avg: (0.00015 + 0.0003) / 2 = 0.000225
        criteria = ProviderSelectionCriteria(
            max_cost_per_1k_tokens=0.0002  # Only openai meets this
        )
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "openai"

    def test_select_provider_by_latency(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test provider selection by latency constraint."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        criteria = ProviderSelectionCriteria(
            max_latency_ms=1200  # openai: 1000ms, anthropic: 1500ms
        )
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "openai"

    def test_select_provider_by_data_residency(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test provider selection by data residency."""
        registry = ProviderRegistry()
        sample_provider.capabilities.data_residency = [DataResidency.US_EAST]
        registry.register_provider(sample_provider)

        criteria = ProviderSelectionCriteria(
            data_residency=DataResidency.US_EAST
        )
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "openai"

    def test_select_provider_excluded_providers(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test provider selection with exclusions."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        criteria = ProviderSelectionCriteria(
            excluded_providers=["openai"]
        )
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "anthropic"

    def test_select_provider_preferred_providers(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test provider selection with preferred providers."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        criteria = ProviderSelectionCriteria(
            preferred_providers=["anthropic", "openai"]
        )
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "anthropic"

    def test_select_provider_circuit_breaker_open(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test that providers with open circuit breakers are excluded."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        # Open circuit breaker for openai
        registry._circuit_breakers["openai"].state = CircuitBreakerState.OPEN

        criteria = ProviderSelectionCriteria()
        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "anthropic"

    def test_select_provider_no_suitable_provider(self) -> None:
        """Test that selection raises error when no suitable provider found."""
        registry = ProviderRegistry()

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION]
        )

        with pytest.raises(LLMGatewayProviderError) as exc_info:
            registry.select_provider(criteria)
        assert "No suitable provider found" in str(exc_info.value)

    def test_select_provider_with_fallbacks(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test that selection includes fallback providers."""
        registry = ProviderRegistry()

        # Add more providers for fallback testing
        gemini_provider = ProviderConfiguration(
            provider_id="gemini",
            metadata=ProviderMetadata(name="Gemini"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(UTC),
            ),
        )

        registry.register_providers([sample_provider, sample_provider_anthropic, gemini_provider])

        criteria = ProviderSelectionCriteria()
        result = registry.select_provider(criteria)

        assert len(result.fallback_providers) > 0
        assert len(result.fallback_providers) <= 3  # Max 3 fallbacks

    def test_get_circuit_breaker_existing(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test retrieving circuit breaker for existing provider."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        cb = registry.get_circuit_breaker("openai")
        assert cb is not None
        assert cb.provider_id == "openai"
        assert cb.state == CircuitBreakerState.CLOSED

    def test_get_circuit_breaker_non_existing(self) -> None:
        """Test retrieving circuit breaker for non-existing provider."""
        registry = ProviderRegistry()
        cb = registry.get_circuit_breaker("nonexistent")
        assert cb is None

    def test_load_from_file(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test loading providers from JSON file."""
        registry = ProviderRegistry()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "providers": [sample_provider.model_dump(mode="json")]
            }
            json.dump(data, f, default=str)
            temp_path = f.name

        try:
            registry.load_from_file(temp_path)
            assert len(registry._providers) == 1
            assert "openai" in registry._providers
        finally:
            Path(temp_path).unlink()

    def test_load_from_file_not_found(self) -> None:
        """Test loading from non-existent file raises error."""
        registry = ProviderRegistry()

        with pytest.raises(LLMGatewayConfigurationError) as exc_info:
            registry.load_from_file("/nonexistent/file.json")
        assert "not found" in str(exc_info.value)

    def test_load_from_file_invalid_json(self) -> None:
        """Test loading from invalid JSON file raises error."""
        registry = ProviderRegistry()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json")
            temp_path = f.name

        try:
            with pytest.raises(LLMGatewayConfigurationError) as exc_info:
                registry.load_from_file(temp_path)
            assert "Failed to load" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_save_to_file(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test saving providers to JSON file."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            registry.save_to_file(temp_path)

            # Verify file was created and contains correct data
            with open(temp_path) as f:
                data = json.load(f)
            assert "providers" in data
            assert len(data["providers"]) == 1
            assert data["providers"][0]["provider_id"] == "openai"
            assert "metadata" in data
            assert data["metadata"]["total_providers"] == 1
        finally:
            Path(temp_path).unlink()

    def test_save_to_file_creates_directory(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test that save_to_file creates parent directories."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "subdir" / "providers.json"
            registry.save_to_file(temp_path)

            assert temp_path.exists()

    @pytest.mark.skip(reason="Cost optimizer integration requires additional modules")
    def test_select_cost_optimized_provider(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test cost-optimized provider selection.

        This test is skipped as it requires mocking deeply nested imports
        that are loaded lazily inside the method. The functionality is
        covered by integration tests.
        """
        pass

    def test_get_stats(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test retrieving registry statistics."""
        registry = ProviderRegistry()
        sample_provider.enabled = True
        sample_provider_anthropic.enabled = False
        registry.register_providers([sample_provider, sample_provider_anthropic])

        stats = registry.get_stats()

        assert stats["total_providers"] == 2
        assert stats["enabled_providers"] == 1
        assert stats["healthy_providers"] == 1
        assert "capability_counts" in stats
        assert "circuit_breaker_states" in stats

    def test_get_stats_empty_registry(self) -> None:
        """Test statistics for empty registry."""
        registry = ProviderRegistry()
        stats = registry.get_stats()

        assert stats["total_providers"] == 0
        assert stats["enabled_providers"] == 0
        assert stats["healthy_providers"] == 0


class TestGetProviderRegistry:
    """Test suite for get_provider_registry global function."""

    def test_get_provider_registry_singleton(self) -> None:
        """Test that get_provider_registry returns singleton instance."""
        # Clear global registry
        import agentcore.llm_gateway.registry as registry_module
        registry_module._registry = None

        registry1 = get_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is registry2

    def test_get_provider_registry_creates_instance(self) -> None:
        """Test that get_provider_registry creates instance on first call."""
        import agentcore.llm_gateway.registry as registry_module
        registry_module._registry = None

        registry = get_provider_registry()
        assert isinstance(registry, ProviderRegistry)


class TestProviderRanking:
    """Test suite for provider ranking and scoring."""

    def test_rank_providers_with_preferred(
        self,
        sample_provider: ProviderConfiguration,
        sample_provider_anthropic: ProviderConfiguration,
    ) -> None:
        """Test that preferred providers are ranked first."""
        registry = ProviderRegistry()
        registry.register_providers([sample_provider, sample_provider_anthropic])

        criteria = ProviderSelectionCriteria(
            preferred_providers=["anthropic"]
        )

        ranked = registry._rank_providers(
            [sample_provider, sample_provider_anthropic],
            criteria,
        )

        assert ranked[0].provider_id == "anthropic"
        assert ranked[1].provider_id == "openai"

    def test_calculate_provider_score(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test provider scoring calculation."""
        registry = ProviderRegistry()
        registry.register_provider(sample_provider)

        criteria = ProviderSelectionCriteria()
        score = registry._calculate_provider_score(sample_provider, criteria)

        assert isinstance(score, float)
        assert score > 0

    def test_get_selection_reason(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test selection reason generation."""
        registry = ProviderRegistry()

        criteria = ProviderSelectionCriteria(
            preferred_providers=["openai"]
        )

        reason = registry._get_selection_reason(sample_provider, criteria)

        assert isinstance(reason, str)
        assert len(reason) > 0
        assert "preferred provider" in reason.lower()

    def test_get_selection_reason_health_status(
        self,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test selection reason includes health status."""
        registry = ProviderRegistry()
        sample_provider.health = ProviderHealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=datetime.now(UTC),
            success_rate=0.99,
        )

        criteria = ProviderSelectionCriteria()
        reason = registry._get_selection_reason(sample_provider, criteria)

        assert "healthy" in reason.lower()
        assert "99" in reason  # Success rate percentage
