"""Unit tests for provider management functionality.

Tests provider configuration, registry, health monitoring, and failover logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.integration.portkey.health import ProviderHealthMonitor
from agentcore.integration.portkey.provider import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    DataResidency,
    ProviderCapabilities,
    ProviderCapability,
    ProviderConfiguration,
    ProviderHealthMetrics,
    ProviderMetadata,
    ProviderPricing,
    ProviderSelectionCriteria,
    ProviderStatus,
)
from agentcore.integration.portkey.registry import ProviderRegistry


class TestProviderModels:
    """Test provider data models."""

    def test_provider_capability_enum(self) -> None:
        """Test ProviderCapability enum."""
        assert ProviderCapability.TEXT_GENERATION == "text_generation"
        assert ProviderCapability.CHAT_COMPLETION == "chat_completion"
        assert ProviderCapability.VISION == "vision"
        assert ProviderCapability.CODE_GENERATION == "code_generation"

    def test_provider_status_enum(self) -> None:
        """Test ProviderStatus enum."""
        assert ProviderStatus.HEALTHY == "healthy"
        assert ProviderStatus.DEGRADED == "degraded"
        assert ProviderStatus.UNHEALTHY == "unhealthy"
        assert ProviderStatus.UNAVAILABLE == "unavailable"

    def test_provider_metadata_creation(self) -> None:
        """Test ProviderMetadata model creation."""
        metadata = ProviderMetadata(
            name="OpenAI",
            description="OpenAI GPT models",
            website="https://openai.com",
        )
        assert metadata.name == "OpenAI"
        assert metadata.description == "OpenAI GPT models"
        assert metadata.website == "https://openai.com"

    def test_provider_pricing_creation(self) -> None:
        """Test ProviderPricing model creation."""
        pricing = ProviderPricing(
            input_token_price=0.003,
            output_token_price=0.006,
            free_tier=False,
            rate_limits={"requests_per_minute": 3500},
        )
        assert pricing.input_token_price == 0.003
        assert pricing.output_token_price == 0.006
        assert not pricing.free_tier
        assert pricing.rate_limits["requests_per_minute"] == 3500

    def test_provider_capabilities_creation(self) -> None:
        """Test ProviderCapabilities model creation."""
        capabilities = ProviderCapabilities(
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.CHAT_COMPLETION,
            ],
            context_window=8192,
            supports_streaming=True,
            data_residency=[DataResidency.US_EAST, DataResidency.EU_WEST],
        )
        assert len(capabilities.capabilities) == 2
        assert ProviderCapability.TEXT_GENERATION in capabilities.capabilities
        assert capabilities.supports_streaming is True
        assert DataResidency.US_EAST in capabilities.data_residency

    def test_circuit_breaker_config_defaults(self) -> None:
        """Test CircuitBreakerConfig default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30
        assert config.half_open_requests == 1

    def test_provider_configuration_complete(self) -> None:
        """Test complete ProviderConfiguration creation."""
        config = ProviderConfiguration(
            provider_id="openai",
            virtual_key="vk_openai_123",
            enabled=True,
            priority=200,
            weight=10,
            metadata=ProviderMetadata(
                name="OpenAI",
                description="OpenAI GPT models",
            ),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                ],
                context_window=8192,
            ),
            pricing=ProviderPricing(
                input_token_price=0.003,
                output_token_price=0.006,
            ),
            tags=["production", "high-priority"],
        )
        assert config.provider_id == "openai"
        assert config.enabled is True
        assert config.priority == 200
        assert len(config.tags) == 2


class TestProviderRegistry:
    """Test ProviderRegistry functionality."""

    @pytest.fixture
    def registry(self) -> ProviderRegistry:
        """Create a fresh registry for each test."""
        return ProviderRegistry()

    @pytest.fixture
    def sample_provider(self) -> ProviderConfiguration:
        """Create a sample provider configuration."""
        return ProviderConfiguration(
            provider_id="openai",
            enabled=True,
            priority=100,
            metadata=ProviderMetadata(name="OpenAI"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                ],
            ),
            pricing=ProviderPricing(
                input_token_price=0.003,
                output_token_price=0.006,
            ),
        )

    def test_registry_initialization(self, registry: ProviderRegistry) -> None:
        """Test registry initialization."""
        assert len(registry._providers) == 0
        assert len(registry._circuit_breakers) == 0

    def test_register_single_provider(
        self,
        registry: ProviderRegistry,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test registering a single provider."""
        registry.register_provider(sample_provider)

        assert len(registry._providers) == 1
        assert "openai" in registry._providers
        assert registry._providers["openai"] == sample_provider

        # Check circuit breaker was created
        assert "openai" in registry._circuit_breakers
        assert registry._circuit_breakers["openai"].state == CircuitBreakerState.CLOSED

    def test_register_multiple_providers(self, registry: ProviderRegistry) -> None:
        """Test registering multiple providers."""
        providers = [
            ProviderConfiguration(
                provider_id="openai",
                enabled=True,
                metadata=ProviderMetadata(name="OpenAI"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
            ),
            ProviderConfiguration(
                provider_id="anthropic",
                enabled=True,
                metadata=ProviderMetadata(name="Anthropic"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
            ),
        ]

        registry.register_providers(providers)

        assert len(registry._providers) == 2
        assert "openai" in registry._providers
        assert "anthropic" in registry._providers

    def test_get_provider(
        self,
        registry: ProviderRegistry,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test retrieving a provider by ID."""
        registry.register_provider(sample_provider)

        provider = registry.get_provider("openai")
        assert provider is not None
        assert provider.provider_id == "openai"

        # Test non-existent provider
        assert registry.get_provider("nonexistent") is None

    def test_list_providers_all(
        self,
        registry: ProviderRegistry,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test listing all providers."""
        registry.register_provider(sample_provider)

        providers = registry.list_providers(enabled_only=False)
        assert len(providers) == 1
        assert providers[0].provider_id == "openai"

    def test_list_providers_enabled_only(self, registry: ProviderRegistry) -> None:
        """Test filtering providers by enabled status."""
        enabled = ProviderConfiguration(
            provider_id="openai",
            enabled=True,
            metadata=ProviderMetadata(name="OpenAI"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
        )
        disabled = ProviderConfiguration(
            provider_id="disabled",
            enabled=False,
            metadata=ProviderMetadata(name="Disabled"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
        )

        registry.register_providers([enabled, disabled])

        providers = registry.list_providers(enabled_only=True)
        assert len(providers) == 1
        assert providers[0].provider_id == "openai"

    def test_list_providers_by_capability(self, registry: ProviderRegistry) -> None:
        """Test filtering providers by capability."""
        text_provider = ProviderConfiguration(
            provider_id="text",
            enabled=True,
            metadata=ProviderMetadata(name="Text"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
        )
        vision_provider = ProviderConfiguration(
            provider_id="vision",
            enabled=True,
            metadata=ProviderMetadata(name="Vision"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.VISION,
                ]
            ),
        )

        registry.register_providers([text_provider, vision_provider])

        # Filter by vision capability
        providers = registry.list_providers(capability=ProviderCapability.VISION)
        assert len(providers) == 1
        assert providers[0].provider_id == "vision"

    def test_list_providers_by_tags(self, registry: ProviderRegistry) -> None:
        """Test filtering providers by tags."""
        prod_provider = ProviderConfiguration(
            provider_id="prod",
            enabled=True,
            metadata=ProviderMetadata(name="Production"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
            tags=["production", "high-priority"],
        )
        dev_provider = ProviderConfiguration(
            provider_id="dev",
            enabled=True,
            metadata=ProviderMetadata(name="Development"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
            tags=["development"],
        )

        registry.register_providers([prod_provider, dev_provider])

        # Filter by production tag
        providers = registry.list_providers(tags=["production"])
        assert len(providers) == 1
        assert providers[0].provider_id == "prod"

    def test_provider_selection_basic(
        self,
        registry: ProviderRegistry,
        sample_provider: ProviderConfiguration,
    ) -> None:
        """Test basic provider selection."""
        registry.register_provider(sample_provider)

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION]
        )

        result = registry.select_provider(criteria)
        assert result.provider.provider_id == "openai"
        assert len(result.fallback_providers) == 0

    def test_provider_selection_with_fallbacks(
        self, registry: ProviderRegistry
    ) -> None:
        """Test provider selection with multiple fallbacks."""
        providers = [
            ProviderConfiguration(
                provider_id=f"provider{i}",
                enabled=True,
                priority=100 - i * 10,  # Descending priority
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
            )
            for i in range(5)
        ]

        registry.register_providers(providers)

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION]
        )

        result = registry.select_provider(criteria)
        assert result.provider.provider_id == "provider0"  # Highest priority
        assert len(result.fallback_providers) == 3  # Up to 3 fallbacks
        assert result.fallback_providers[0].provider_id == "provider1"

    def test_provider_selection_with_cost_constraint(
        self, registry: ProviderRegistry
    ) -> None:
        """Test provider selection with cost constraints."""
        expensive = ProviderConfiguration(
            provider_id="expensive",
            enabled=True,
            metadata=ProviderMetadata(name="Expensive"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
            pricing=ProviderPricing(
                input_token_price=0.01,
                output_token_price=0.02,
            ),
        )
        cheap = ProviderConfiguration(
            provider_id="cheap",
            enabled=True,
            metadata=ProviderMetadata(name="Cheap"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION]
            ),
            pricing=ProviderPricing(
                input_token_price=0.001,
                output_token_price=0.002,
            ),
        )

        registry.register_providers([expensive, cheap])

        # Set max cost to exclude expensive provider
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            max_cost_per_1k_tokens=0.005,  # Average of 0.015 is too high
        )

        result = registry.select_provider(criteria)
        # Should select cheap provider
        assert result.provider.provider_id == "cheap"

    def test_get_stats(self, registry: ProviderRegistry) -> None:
        """Test registry statistics."""
        providers = [
            ProviderConfiguration(
                provider_id="openai",
                enabled=True,
                metadata=ProviderMetadata(name="OpenAI"),
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
                provider_id="anthropic",
                enabled=True,
                metadata=ProviderMetadata(name="Anthropic"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
                health=ProviderHealthMetrics(
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                ),
            ),
            ProviderConfiguration(
                provider_id="disabled",
                enabled=False,
                metadata=ProviderMetadata(name="Disabled"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION]
                ),
            ),
        ]

        registry.register_providers(providers)

        stats = registry.get_stats()
        assert stats["total_providers"] == 3
        assert stats["enabled_providers"] == 2
        assert stats["healthy_providers"] == 2
        # Only count enabled providers
        assert stats["capability_counts"][ProviderCapability.TEXT_GENERATION] == 2
        assert (
            stats["capability_counts"][ProviderCapability.CHAT_COMPLETION] == 1
        )


class TestProviderHealthMonitor:
    """Test ProviderHealthMonitor functionality."""

    @pytest.fixture
    def registry(self) -> ProviderRegistry:
        """Create a registry with sample providers."""
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
        return registry

    @pytest.fixture
    def monitor(self, registry: ProviderRegistry) -> ProviderHealthMonitor:
        """Create a health monitor."""
        return ProviderHealthMonitor(
            registry=registry,
            monitoring_window_seconds=60,
            health_check_interval_seconds=10,
        )

    def test_monitor_initialization(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test health monitor initialization."""
        assert monitor.monitoring_window == timedelta(seconds=60)
        assert monitor.health_check_interval == 10
        assert not monitor._running

    @pytest.mark.asyncio
    async def test_monitor_start_stop(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test starting and stopping monitor."""
        await monitor.start()
        assert monitor._running
        assert monitor._monitoring_task is not None

        await monitor.stop()
        assert not monitor._running

    def test_record_successful_request(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test recording a successful request."""
        monitor.record_request_success(
            provider_id="test_provider",
            latency_ms=100,
        )

        # Check request was recorded
        assert len(monitor._request_history["test_provider"]) == 1
        record = monitor._request_history["test_provider"][0]
        assert record.success is True
        assert record.latency_ms == 100

    def test_record_failed_request(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test recording a failed request."""
        monitor.record_request_failure(
            provider_id="test_provider",
            latency_ms=200,
            error="Connection timeout",
        )

        # Check request was recorded
        assert len(monitor._request_history["test_provider"]) == 1
        record = monitor._request_history["test_provider"][0]
        assert record.success is False
        assert record.latency_ms == 200
        assert record.error == "Connection timeout"

    def test_calculate_health_metrics_no_data(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test health metrics with no request data."""
        metrics = monitor._calculate_health_metrics("test_provider")

        assert metrics.status == ProviderStatus.HEALTHY
        assert metrics.success_rate == 1.0
        assert metrics.total_requests == 0
        assert metrics.error_count == 0

    def test_calculate_health_metrics_all_success(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test health metrics with all successful requests."""
        # Record 10 successful requests
        for i in range(10):
            monitor.record_request_success("test_provider", latency_ms=100 + i)

        metrics = monitor._calculate_health_metrics("test_provider")

        assert metrics.status == ProviderStatus.HEALTHY
        assert metrics.success_rate == 1.0
        assert metrics.total_requests == 10
        assert metrics.error_count == 0
        assert metrics.consecutive_failures == 0
        assert metrics.average_latency_ms == 104  # (100+101+...+109)/10

    def test_calculate_health_metrics_mixed_results(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test health metrics with mixed success/failure."""
        # Record 7 successes and 3 failures
        for _ in range(7):
            monitor.record_request_success("test_provider", latency_ms=100)
        for _ in range(3):
            monitor.record_request_failure(
                "test_provider",
                latency_ms=150,
                error="Timeout",
            )

        metrics = monitor._calculate_health_metrics("test_provider")

        assert metrics.total_requests == 10
        assert metrics.error_count == 3
        assert metrics.success_rate == 0.7
        assert metrics.consecutive_failures == 3  # Last 3 were failures
        # Status should be DEGRADED (success rate < 0.9)
        assert metrics.status == ProviderStatus.DEGRADED

    def test_determine_status_healthy(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test status determination for healthy provider."""
        status = monitor._determine_status(
            success_rate=0.99,
            consecutive_failures=0,
            average_latency_ms=100,
        )
        assert status == ProviderStatus.HEALTHY

    def test_determine_status_degraded(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test status determination for degraded provider."""
        # Low success rate
        status = monitor._determine_status(
            success_rate=0.85,
            consecutive_failures=0,
            average_latency_ms=100,
        )
        assert status == ProviderStatus.DEGRADED

        # High latency
        status = monitor._determine_status(
            success_rate=0.99,
            consecutive_failures=0,
            average_latency_ms=6000,
        )
        assert status == ProviderStatus.DEGRADED

    def test_determine_status_unhealthy(
        self, monitor: ProviderHealthMonitor
    ) -> None:
        """Test status determination for unhealthy provider."""
        # Multiple consecutive failures
        status = monitor._determine_status(
            success_rate=0.95,
            consecutive_failures=5,
            average_latency_ms=100,
        )
        assert status == ProviderStatus.UNHEALTHY

        # Very low success rate
        status = monitor._determine_status(
            success_rate=0.3,
            consecutive_failures=0,
            average_latency_ms=100,
        )
        assert status == ProviderStatus.UNHEALTHY

    def test_is_provider_available(
        self,
        monitor: ProviderHealthMonitor,
        registry: ProviderRegistry,
    ) -> None:
        """Test provider availability check."""
        # Initially should be available (enabled, no health data)
        assert monitor.is_provider_available("test_provider") is True

        # Add healthy status
        provider = registry.get_provider("test_provider")
        if provider:
            provider.health = ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
            )
        assert monitor.is_provider_available("test_provider") is True

        # Set unhealthy status
        if provider:
            provider.health = ProviderHealthMetrics(
                status=ProviderStatus.UNHEALTHY,
                last_check=datetime.now(),
            )
        assert monitor.is_provider_available("test_provider") is False

        # Disabled provider
        if provider:
            provider.enabled = False
            provider.health = ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
            )
        assert monitor.is_provider_available("test_provider") is False
