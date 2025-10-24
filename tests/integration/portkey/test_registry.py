"""Comprehensive unit tests for provider registry functionality.

Tests provider registration, selection, filtering, ranking, and cost optimization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentcore.integration.portkey.cost_models import OptimizationContext, OptimizationStrategy
from agentcore.integration.portkey.exceptions import (
    PortkeyConfigurationError,
    PortkeyProviderError,
)
from agentcore.integration.portkey.provider import (
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
from agentcore.integration.portkey.registry import ProviderRegistry, get_provider_registry


class TestProviderRegistryBasics:
    """Test basic registry operations."""

    def test_registry_initialization(self) -> None:
        """Test registry initialization."""
        registry = ProviderRegistry()
        assert registry._initialized is False
        assert len(registry._providers) == 0
        assert len(registry._circuit_breakers) == 0

    def test_register_provider(self) -> None:
        """Test registering a single provider."""
        registry = ProviderRegistry()
        provider = ProviderConfiguration(
            provider_id="test_provider",
            enabled=True,
            metadata=ProviderMetadata(name="Test Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
        )

        registry.register_provider(provider)

        assert len(registry._providers) == 1
        assert "test_provider" in registry._providers
        assert "test_provider" in registry._circuit_breakers

    def test_register_provider_update_existing(self) -> None:
        """Test updating an existing provider."""
        registry = ProviderRegistry()
        provider1 = ProviderConfiguration(
            provider_id="test",
            enabled=True,
            priority=100,
            metadata=ProviderMetadata(name="Test 1"),
            capabilities=ProviderCapabilities(),
        )

        registry.register_provider(provider1)
        assert registry._providers["test"].priority == 100

        # Update with new configuration
        provider2 = ProviderConfiguration(
            provider_id="test",
            enabled=True,
            priority=200,
            metadata=ProviderMetadata(name="Test 2"),
            capabilities=ProviderCapabilities(),
        )

        registry.register_provider(provider2)

        # Should update existing
        assert len(registry._providers) == 1
        assert registry._providers["test"].priority == 200
        assert registry._providers["test"].metadata.name == "Test 2"

    def test_register_providers_bulk(self) -> None:
        """Test bulk provider registration."""
        registry = ProviderRegistry()
        providers = [
            ProviderConfiguration(
                provider_id=f"provider_{i}",
                enabled=True,
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(),
            )
            for i in range(5)
        ]

        registry.register_providers(providers)

        assert len(registry._providers) == 5
        assert len(registry._circuit_breakers) == 5

    def test_get_provider(self) -> None:
        """Test retrieving a provider by ID."""
        registry = ProviderRegistry()
        provider = ProviderConfiguration(
            provider_id="test",
            enabled=True,
            metadata=ProviderMetadata(name="Test"),
            capabilities=ProviderCapabilities(),
        )

        registry.register_provider(provider)

        retrieved = registry.get_provider("test")
        assert retrieved is not None
        assert retrieved.provider_id == "test"

    def test_get_provider_not_found(self) -> None:
        """Test retrieving non-existent provider."""
        registry = ProviderRegistry()
        result = registry.get_provider("nonexistent")
        assert result is None


class TestProviderListing:
    """Test provider listing and filtering."""

    def test_list_all_providers(self) -> None:
        """Test listing all providers."""
        registry = ProviderRegistry()
        providers = [
            ProviderConfiguration(
                provider_id=f"provider_{i}",
                enabled=i % 2 == 0,  # Alternate enabled/disabled
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(),
            )
            for i in range(6)
        ]

        registry.register_providers(providers)

        # List all (enabled only by default)
        enabled = registry.list_providers(enabled_only=True)
        assert len(enabled) == 3  # 0, 2, 4

        # List all including disabled
        all_providers = registry.list_providers(enabled_only=False)
        assert len(all_providers) == 6

    def test_list_providers_by_capability(self) -> None:
        """Test filtering providers by capability."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="text_only",
                enabled=True,
                metadata=ProviderMetadata(name="Text Only"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
            ),
            ProviderConfiguration(
                provider_id="vision_enabled",
                enabled=True,
                metadata=ProviderMetadata(name="Vision Enabled"),
                capabilities=ProviderCapabilities(
                    capabilities=[
                        ProviderCapability.TEXT_GENERATION,
                        ProviderCapability.VISION,
                    ]
                ),
            ),
            ProviderConfiguration(
                provider_id="code_gen",
                enabled=True,
                metadata=ProviderMetadata(name="Code Gen"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.CODE_GENERATION]
                ),
            ),
        ]

        registry.register_providers(providers)

        # Filter by TEXT_GENERATION
        text_providers = registry.list_providers(
            capability=ProviderCapability.TEXT_GENERATION
        )
        assert len(text_providers) == 2
        assert all(
            ProviderCapability.TEXT_GENERATION in p.capabilities.capabilities
            for p in text_providers
        )

        # Filter by VISION
        vision_providers = registry.list_providers(capability=ProviderCapability.VISION)
        assert len(vision_providers) == 1
        assert vision_providers[0].provider_id == "vision_enabled"

    def test_list_providers_by_status(self) -> None:
        """Test filtering providers by health status."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="healthy",
                enabled=True,
                metadata=ProviderMetadata(name="Healthy"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                ),
            ),
            ProviderConfiguration(
                provider_id="degraded",
                enabled=True,
                metadata=ProviderMetadata(name="Degraded"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.DEGRADED,
                    last_check=datetime.now(),
                ),
            ),
            ProviderConfiguration(
                provider_id="unhealthy",
                enabled=True,
                metadata=ProviderMetadata(name="Unhealthy"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.UNHEALTHY,
                    last_check=datetime.now(),
                ),
            ),
        ]

        registry.register_providers(providers)

        # Filter by HEALTHY status
        healthy = registry.list_providers(status=ProviderStatus.HEALTHY)
        assert len(healthy) == 1
        assert healthy[0].provider_id == "healthy"

        # Filter by DEGRADED status
        degraded = registry.list_providers(status=ProviderStatus.DEGRADED)
        assert len(degraded) == 1
        assert degraded[0].provider_id == "degraded"

    def test_list_providers_by_tags(self) -> None:
        """Test filtering providers by tags."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="prod_cheap",
                enabled=True,
                metadata=ProviderMetadata(name="Prod Cheap"),
                capabilities=ProviderCapabilities(),
                tags=["production", "cost-optimized"],
            ),
            ProviderConfiguration(
                provider_id="prod_fast",
                enabled=True,
                metadata=ProviderMetadata(name="Prod Fast"),
                capabilities=ProviderCapabilities(),
                tags=["production", "low-latency"],
            ),
            ProviderConfiguration(
                provider_id="dev",
                enabled=True,
                metadata=ProviderMetadata(name="Dev"),
                capabilities=ProviderCapabilities(),
                tags=["development"],
            ),
        ]

        registry.register_providers(providers)

        # Filter by production tag
        prod = registry.list_providers(tags=["production"])
        assert len(prod) == 2

        # Filter by multiple tags (must have all)
        prod_cheap = registry.list_providers(tags=["production", "cost-optimized"])
        assert len(prod_cheap) == 1
        assert prod_cheap[0].provider_id == "prod_cheap"


class TestProviderSelection:
    """Test provider selection logic."""

    def test_select_provider_basic(self) -> None:
        """Test basic provider selection."""
        registry = ProviderRegistry()

        provider = ProviderConfiguration(
            provider_id="test",
            enabled=True,
            metadata=ProviderMetadata(name="Test"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.99,
            ),
        )

        registry.register_provider(provider)

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION]
        )

        result = registry.select_provider(criteria)

        assert result.provider.provider_id == "test"
        assert len(result.fallback_providers) == 0

    def test_select_provider_with_fallbacks(self) -> None:
        """Test provider selection with multiple fallback options."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id=f"provider_{i}",
                enabled=True,
                priority=100 + i * 10,
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                ),
            )
            for i in range(5)
        ]

        registry.register_providers(providers)

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION]
        )

        result = registry.select_provider(criteria)

        # Should select highest priority
        assert result.provider.priority == 140  # provider_4
        # Should have up to 3 fallbacks
        assert len(result.fallback_providers) == 3

    def test_select_provider_no_match(self) -> None:
        """Test provider selection when no match found."""
        registry = ProviderRegistry()

        provider = ProviderConfiguration(
            provider_id="text_only",
            enabled=True,
            metadata=ProviderMetadata(name="Text Only"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
        )

        registry.register_provider(provider)

        # Request VISION capability (not available)
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.VISION]
        )

        with pytest.raises(PortkeyProviderError) as exc_info:
            registry.select_provider(criteria)

        assert "No suitable provider found" in str(exc_info.value)

    def test_select_provider_by_data_residency(self) -> None:
        """Test provider selection by data residency."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="us_provider",
                enabled=True,
                metadata=ProviderMetadata(name="US Provider"),
                capabilities=ProviderCapabilities(
                    data_residency=[DataResidency.US_EAST, DataResidency.US_WEST]
                ),
            ),
            ProviderConfiguration(
                provider_id="eu_provider",
                enabled=True,
                metadata=ProviderMetadata(name="EU Provider"),
                capabilities=ProviderCapabilities(
                    data_residency=[DataResidency.EU_WEST, DataResidency.EU_CENTRAL]
                ),
            ),
        ]

        registry.register_providers(providers)

        # Select US residency
        criteria = ProviderSelectionCriteria(data_residency=DataResidency.US_EAST)

        result = registry.select_provider(criteria)
        assert result.provider.provider_id == "us_provider"

    def test_select_provider_excludes_providers(self) -> None:
        """Test provider selection with exclusions."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id=f"provider_{i}",
                enabled=True,
                priority=100 + i * 10,
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(),
            )
            for i in range(3)
        ]

        registry.register_providers(providers)

        # Exclude highest priority provider
        criteria = ProviderSelectionCriteria(excluded_providers=["provider_2"])

        result = registry.select_provider(criteria)

        # Should select next best
        assert result.provider.provider_id == "provider_1"

    def test_select_provider_excludes_open_circuits(self) -> None:
        """Test that open circuit breakers are excluded."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="primary",
                enabled=True,
                priority=200,
                metadata=ProviderMetadata(name="Primary"),
                capabilities=ProviderCapabilities(),
            ),
            ProviderConfiguration(
                provider_id="fallback",
                enabled=True,
                priority=100,
                metadata=ProviderMetadata(name="Fallback"),
                capabilities=ProviderCapabilities(),
            ),
        ]

        registry.register_providers(providers)

        # Open circuit on primary
        cb = registry.get_circuit_breaker("primary")
        if cb:
            cb.state = CircuitBreakerState.OPEN

        criteria = ProviderSelectionCriteria()

        result = registry.select_provider(criteria)

        # Should select fallback (primary has open circuit)
        assert result.provider.provider_id == "fallback"

    def test_select_provider_by_cost_constraint(self) -> None:
        """Test provider selection with cost constraints."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="expensive",
                enabled=True,
                metadata=ProviderMetadata(name="Expensive"),
                capabilities=ProviderCapabilities(),
                pricing=ProviderPricing(
                    input_token_price=0.01, output_token_price=0.03
                ),
            ),
            ProviderConfiguration(
                provider_id="cheap",
                enabled=True,
                metadata=ProviderMetadata(name="Cheap"),
                capabilities=ProviderCapabilities(),
                pricing=ProviderPricing(
                    input_token_price=0.001, output_token_price=0.002
                ),
            ),
        ]

        registry.register_providers(providers)

        # Max cost of $0.005 per 1K tokens (average)
        criteria = ProviderSelectionCriteria(max_cost_per_1k_tokens=0.005)

        result = registry.select_provider(criteria)

        # Should only match cheap provider
        assert result.provider.provider_id == "cheap"

    def test_select_provider_by_latency_constraint(self) -> None:
        """Test provider selection with latency constraints."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="fast",
                enabled=True,
                metadata=ProviderMetadata(name="Fast"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                    average_latency_ms=100,
                ),
            ),
            ProviderConfiguration(
                provider_id="slow",
                enabled=True,
                metadata=ProviderMetadata(name="Slow"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                    average_latency_ms=2000,
                ),
            ),
        ]

        registry.register_providers(providers)

        # Max latency 500ms
        criteria = ProviderSelectionCriteria(max_latency_ms=500)

        result = registry.select_provider(criteria)

        # Should only match fast provider
        assert result.provider.provider_id == "fast"


class TestProviderRanking:
    """Test provider ranking and scoring."""

    def test_preferred_providers_ranked_first(self) -> None:
        """Test that preferred providers are ranked first."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="low_priority",
                enabled=True,
                priority=50,
                metadata=ProviderMetadata(name="Low Priority"),
                capabilities=ProviderCapabilities(),
            ),
            ProviderConfiguration(
                provider_id="preferred",
                enabled=True,
                priority=100,
                metadata=ProviderMetadata(name="Preferred"),
                capabilities=ProviderCapabilities(),
            ),
        ]

        registry.register_providers(providers)

        criteria = ProviderSelectionCriteria(preferred_providers=["low_priority"])

        result = registry.select_provider(criteria)

        # Preferred provider should be selected despite lower priority
        assert result.provider.provider_id == "low_priority"

    def test_provider_scoring_health(self) -> None:
        """Test that provider scoring considers health metrics."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="healthy",
                enabled=True,
                priority=100,
                metadata=ProviderMetadata(name="Healthy"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                    success_rate=0.99,
                    availability_percent=99.9,
                ),
            ),
            ProviderConfiguration(
                provider_id="degraded",
                enabled=True,
                priority=100,
                metadata=ProviderMetadata(name="Degraded"),
                capabilities=ProviderCapabilities(),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.DEGRADED,
                    last_check=datetime.now(),
                    success_rate=0.85,
                    availability_percent=85.0,
                ),
            ),
        ]

        registry.register_providers(providers)

        criteria = ProviderSelectionCriteria()

        result = registry.select_provider(criteria)

        # Healthy provider should be selected
        assert result.provider.provider_id == "healthy"


class TestCircuitBreaker:
    """Test circuit breaker management."""

    def test_get_circuit_breaker(self) -> None:
        """Test retrieving circuit breaker state."""
        registry = ProviderRegistry()

        provider = ProviderConfiguration(
            provider_id="test",
            enabled=True,
            metadata=ProviderMetadata(name="Test"),
            capabilities=ProviderCapabilities(),
        )

        registry.register_provider(provider)

        cb = registry.get_circuit_breaker("test")

        assert cb is not None
        assert cb.provider_id == "test"
        assert cb.state == CircuitBreakerState.CLOSED

    def test_get_circuit_breaker_not_found(self) -> None:
        """Test retrieving circuit breaker for non-existent provider."""
        registry = ProviderRegistry()

        cb = registry.get_circuit_breaker("nonexistent")
        assert cb is None


class TestFileOperations:
    """Test file loading and saving."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading providers from JSON file."""
        config_file = tmp_path / "providers.json"

        config_data = {
            "providers": [
                {
                    "provider_id": "test1",
                    "enabled": True,
                    "priority": 100,
                    "metadata": {"name": "Test 1"},
                    "capabilities": {"capabilities": ["text_generation"]},
                },
                {
                    "provider_id": "test2",
                    "enabled": True,
                    "priority": 200,
                    "metadata": {"name": "Test 2"},
                    "capabilities": {"capabilities": ["vision"]},
                },
            ]
        }

        config_file.write_text(json.dumps(config_data))

        registry = ProviderRegistry()
        registry.load_from_file(config_file)

        assert len(registry._providers) == 2
        assert "test1" in registry._providers
        assert "test2" in registry._providers

    def test_load_from_file_not_found(self) -> None:
        """Test loading from non-existent file."""
        registry = ProviderRegistry()

        with pytest.raises(PortkeyConfigurationError) as exc_info:
            registry.load_from_file("/nonexistent/path.json")

        assert "not found" in str(exc_info.value)

    def test_save_to_file(self, tmp_path: Path) -> None:
        """Test saving providers to JSON file."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="test1",
                enabled=True,
                metadata=ProviderMetadata(name="Test 1"),
                capabilities=ProviderCapabilities(),
            ),
            ProviderConfiguration(
                provider_id="test2",
                enabled=False,
                metadata=ProviderMetadata(name="Test 2"),
                capabilities=ProviderCapabilities(),
            ),
        ]

        registry.register_providers(providers)

        output_file = tmp_path / "output.json"
        registry.save_to_file(output_file)

        assert output_file.exists()

        # Verify content
        data = json.loads(output_file.read_text())
        assert len(data["providers"]) == 2
        assert data["metadata"]["total_providers"] == 2
        assert data["metadata"]["enabled_providers"] == 1


class TestRegistryStats:
    """Test registry statistics."""

    def test_get_stats(self) -> None:
        """Test getting registry statistics."""
        registry = ProviderRegistry()

        providers = [
            ProviderConfiguration(
                provider_id="healthy",
                enabled=True,
                metadata=ProviderMetadata(name="Healthy"),
                capabilities=ProviderCapabilities(
                    capabilities=[
                        ProviderCapability.TEXT_GENERATION,
                        ProviderCapability.CHAT_COMPLETION,
                    ]
                ),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                ),
            ),
            ProviderConfiguration(
                provider_id="degraded",
                enabled=True,
                metadata=ProviderMetadata(name="Degraded"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.VISION]
                ),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.DEGRADED,
                    last_check=datetime.now(),
                ),
            ),
            ProviderConfiguration(
                provider_id="disabled",
                enabled=False,
                metadata=ProviderMetadata(name="Disabled"),
                capabilities=ProviderCapabilities(),
            ),
        ]

        registry.register_providers(providers)

        stats = registry.get_stats()

        assert stats["total_providers"] == 3
        assert stats["enabled_providers"] == 2
        assert stats["healthy_providers"] == 1
        assert stats["capability_counts"]["text_generation"] == 1
        assert stats["capability_counts"]["vision"] == 1
        assert stats["circuit_breaker_states"]["closed"] == 3


class TestGlobalRegistryInstance:
    """Test global registry singleton."""

    def test_get_provider_registry_singleton(self) -> None:
        """Test that get_provider_registry returns singleton."""
        registry1 = get_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is registry2
