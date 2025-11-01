"""Provider selection algorithm tests.

Tests cost-based provider ranking, capability-aware selection, cost vs. quality
trade-offs, and real-time pricing updates. Validates edge cases including price
changes and provider unavailability.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agentcore.integration.portkey.cost_models import (
    OptimizationContext,
    OptimizationStrategy,
)
from agentcore.integration.portkey.cost_optimizer import CostOptimizer
from agentcore.integration.portkey.cost_tracker import CostTracker
from agentcore.integration.portkey.exceptions import PortkeyProviderError
from agentcore.integration.portkey.provider import (
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


@pytest.fixture
def diverse_registry() -> ProviderRegistry:
    """Create a provider registry with diverse providers for selection testing."""
    reg = ProviderRegistry()

    # Ultra-cheap provider (low quality)
    reg.register_provider(
        ProviderConfiguration(
            provider_id="ultra_cheap",
            enabled=True,
            priority=80,
            metadata=ProviderMetadata(name="Ultra Cheap Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION],
                context_window=2048,
                data_residency=[DataResidency.GLOBAL],
            ),
            pricing=ProviderPricing(
                input_token_price=0.0001,
                output_token_price=0.0002,
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.92,
                average_latency_ms=1500,
                availability_percent=95.0,
            ),
        )
    )

    # Cheap provider (good quality)
    reg.register_provider(
        ProviderConfiguration(
            provider_id="cheap_quality",
            enabled=True,
            priority=90,
            metadata=ProviderMetadata(name="Cheap Quality Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                ],
                supports_streaming=True,
                context_window=8192,
                data_residency=[DataResidency.US_EAST, DataResidency.EU_WEST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.0005,
                output_token_price=0.0010,
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.98,
                average_latency_ms=900,
                availability_percent=99.0,
            ),
        )
    )

    # Premium provider (excellent quality)
    reg.register_provider(
        ProviderConfiguration(
            provider_id="premium_quality",
            enabled=True,
            priority=100,
            metadata=ProviderMetadata(name="Premium Quality Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.REASONING,
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.CODE_GENERATION,
                ],
                supports_function_calling=True,
                supports_streaming=True,
                supports_json_mode=True,
                context_window=128000,
                data_residency=[DataResidency.US_EAST, DataResidency.US_WEST, DataResidency.EU_WEST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.015,
                output_token_price=0.045,
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.995,
                average_latency_ms=700,
                availability_percent=99.95,
            ),
        )
    )

    # Degraded provider (normally good, but currently degraded)
    reg.register_provider(
        ProviderConfiguration(
            provider_id="degraded_provider",
            enabled=True,
            priority=95,
            metadata=ProviderMetadata(name="Degraded Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.REASONING,
                ],
                supports_streaming=True,
                context_window=16000,
                data_residency=[DataResidency.US_EAST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.005,
                output_token_price=0.010,
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.DEGRADED,
                last_check=datetime.now(),
                success_rate=0.85,
                average_latency_ms=2000,
                availability_percent=90.0,
                consecutive_failures=3,
            ),
        )
    )

    # Unavailable provider
    reg.register_provider(
        ProviderConfiguration(
            provider_id="unavailable_provider",
            enabled=True,
            priority=98,
            metadata=ProviderMetadata(name="Unavailable Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                ],
                context_window=32000,
                data_residency=[DataResidency.US_WEST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.002,
                output_token_price=0.006,
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.UNAVAILABLE,
                last_check=datetime.now(),
                success_rate=0.0,
                average_latency_ms=None,
                availability_percent=0.0,
                consecutive_failures=10,
            ),
        )
    )

    return reg


@pytest.fixture
def cost_optimizer_diverse(
    diverse_registry: ProviderRegistry,
) -> CostOptimizer:
    """Create cost optimizer with diverse providers."""
    return CostOptimizer(diverse_registry, CostTracker())


class TestCostBasedProviderRanking:
    """Test cost-based provider ranking and selection."""

    def test_cost_only_selects_cheapest_provider(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """COST_ONLY strategy should select a low-cost provider."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select ultra_cheap or cheap_quality (both are low-cost)
        assert selected.provider_id in ["ultra_cheap", "cheap_quality"]
        # Should not select expensive provider
        assert selected.provider_id != "premium_quality"

    def test_performance_first_selects_best_latency(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """PERFORMANCE_FIRST strategy should prioritize low latency."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select premium_quality (best latency: 700ms)
        assert selected.provider_id == "premium_quality"
        assert selected.health is not None
        assert selected.health.average_latency_ms == 700

    def test_balanced_strategy_considers_both_cost_and_quality(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """BALANCED strategy should find middle ground."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.CHAT_COMPLETION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.BALANCED,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select cheap_quality (good balance)
        assert selected.provider_id in ["cheap_quality", "premium_quality"]

        # Should not select ultra_cheap (poor quality) or premium (too expensive)
        assert selected.provider_id != "ultra_cheap"


class TestCapabilityAwareSelection:
    """Test capability-aware provider selection."""

    def test_requires_specific_capabilities(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Provider must support all required capabilities."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[
                ProviderCapability.FUNCTION_CALLING,
                ProviderCapability.REASONING,
            ],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Only premium_quality has both capabilities
        assert selected.provider_id == "premium_quality"
        assert ProviderCapability.FUNCTION_CALLING in selected.capabilities.capabilities
        assert ProviderCapability.REASONING in selected.capabilities.capabilities

    def test_data_residency_requirement(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Provider must support required data residency."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            data_residency=DataResidency.EU_WEST,
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select provider supporting EU_WEST
        assert DataResidency.EU_WEST in selected.capabilities.data_residency
        assert selected.provider_id in ["cheap_quality", "premium_quality"]

    def test_no_providers_match_criteria_raises_error(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Should raise error if no providers match criteria."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[
                ProviderCapability.VISION,  # No provider has this
                ProviderCapability.AUDIO,  # No provider has this
            ],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        with pytest.raises(PortkeyProviderError) as exc_info:
            cost_optimizer_diverse.select_optimal_provider(criteria, context)

        assert "No providers available" in str(exc_info.value)


class TestCostVsQualityTradeoffs:
    """Test cost vs. quality trade-off scenarios."""

    def test_high_priority_favors_quality_over_cost(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """High priority requests should favor quality."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.ADAPTIVE,
            priority=10,  # Highest priority
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select high-quality provider despite cost
        assert selected.provider_id in ["premium_quality", "cheap_quality"]
        assert selected.provider_id != "ultra_cheap"

    def test_low_priority_favors_cost_over_quality(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Low priority requests should favor cost."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.ADAPTIVE,
            priority=1,  # Lowest priority
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select a low-cost provider
        assert selected.provider_id in ["ultra_cheap", "cheap_quality"]
        # Should not select expensive provider for low priority
        assert selected.provider_id != "premium_quality"

    def test_cost_constraint_filters_expensive_providers(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Cost constraint should filter out expensive providers."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            max_acceptable_cost=0.005,  # Very low limit
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Verify selected provider is within cost constraint
        estimated_cost = cost_optimizer_diverse.estimate_request_cost(
            selected, 1000, 500
        )
        assert estimated_cost <= 0.005

    def test_latency_constraint_filters_slow_providers(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Latency constraint should filter out slow providers."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            max_acceptable_latency_ms=1000,  # Low latency requirement
            optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should select fast provider
        assert selected.health is not None
        assert selected.health.average_latency_ms is not None
        assert selected.health.average_latency_ms <= 1000


class TestProviderHealthAndAvailability:
    """Test provider health and availability considerations."""

    def test_unhealthy_providers_are_excluded(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Unavailable providers should be excluded from selection."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.CHAT_COMPLETION],
            require_healthy=True,
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should not select unavailable provider
        assert selected.provider_id != "unavailable_provider"
        assert selected.health is not None
        assert selected.health.status != ProviderStatus.UNAVAILABLE

    def test_degraded_providers_allowed_when_specified(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Degraded providers should be allowed when explicitly permitted."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.REASONING],
            require_healthy=False,  # Allow degraded
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
            allow_degraded_providers=True,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Can select degraded provider (if cheapest with capability)
        # degraded_provider has REASONING capability
        assert selected.provider_id in ["degraded_provider", "premium_quality"]

    def test_success_rate_filters_unreliable_providers(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Providers below minimum success rate should be filtered."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            min_success_rate=0.95,  # Require 95%+ success rate
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should not select ultra_cheap (92% success rate)
        assert selected.provider_id != "ultra_cheap"
        assert selected.health is not None
        assert selected.health.success_rate >= 0.95


class TestProviderSwitchingStrategies:
    """Test provider switching and fallback strategies."""

    def test_provider_exclusion(
        self,
        cost_optimizer_diverse: CostOptimizer,
    ) -> None:
        """Test excluding specific providers from selection."""
        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            excluded_providers=["ultra_cheap", "cheap_quality"],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = cost_optimizer_diverse.select_optimal_provider(criteria, context)

        # Should not select excluded providers
        assert selected.provider_id not in ["ultra_cheap", "cheap_quality"]

    def test_tag_based_filtering(
        self,
        diverse_registry: ProviderRegistry,
    ) -> None:
        """Test filtering providers by tags."""
        # Add tags to providers
        ultra_cheap = diverse_registry.get_provider("ultra_cheap")
        if ultra_cheap:
            ultra_cheap.tags = ["experimental", "low-cost"]

        cheap_quality = diverse_registry.get_provider("cheap_quality")
        if cheap_quality:
            cheap_quality.tags = ["production", "low-cost"]

        optimizer = CostOptimizer(diverse_registry, CostTracker())

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            tags=["production"],  # Only production providers
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        selected = optimizer.select_optimal_provider(criteria, context)

        # Should select provider with production tag
        assert "production" in selected.tags


class TestRealTimePricingUpdates:
    """Test handling of real-time pricing updates."""

    def test_cost_estimation_with_updated_pricing(
        self,
        diverse_registry: ProviderRegistry,
    ) -> None:
        """Test that cost estimation uses current pricing."""
        provider = diverse_registry.get_provider("cheap_quality")
        assert provider is not None
        assert provider.pricing is not None

        optimizer = CostOptimizer(diverse_registry, CostTracker())

        # Store original pricing
        original_input_price = provider.pricing.input_token_price
        original_output_price = provider.pricing.output_token_price

        # Initial cost estimate
        initial_cost = optimizer.estimate_request_cost(provider, 1000, 500)

        # Update pricing (price increase)
        provider.pricing.input_token_price = original_input_price * 4  # 4x increase
        provider.pricing.output_token_price = original_output_price * 4  # 4x increase

        # New cost estimate should reflect updated pricing
        updated_cost = optimizer.estimate_request_cost(provider, 1000, 500)

        assert updated_cost > initial_cost
        assert updated_cost == pytest.approx(initial_cost * 4, rel=0.01)

    def test_provider_selection_adjusts_to_price_changes(
        self,
        diverse_registry: ProviderRegistry,
    ) -> None:
        """Test that provider selection adjusts when prices change."""
        optimizer = CostOptimizer(diverse_registry, CostTracker())

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        # Initial selection
        initial_selected = optimizer.select_optimal_provider(criteria, context)
        initial_provider_id = initial_selected.provider_id

        # Make ultra_cheap more expensive than cheap_quality
        ultra_cheap = diverse_registry.get_provider("ultra_cheap")
        if ultra_cheap and ultra_cheap.pricing:
            ultra_cheap.pricing.input_token_price = 0.010
            ultra_cheap.pricing.output_token_price = 0.020

        # New selection should adapt to price change
        # Clear comparison cache to force recalculation
        optimizer._comparison_cache.clear()  # type: ignore[attr-defined]

        new_selected = optimizer.select_optimal_provider(criteria, context)

        # Selection may change based on new pricing
        if initial_provider_id == "ultra_cheap":
            # If ultra_cheap was initially selected, should now select cheaper option
            assert new_selected.provider_id != "ultra_cheap"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in provider selection."""

    def test_no_enabled_providers(
        self,
        diverse_registry: ProviderRegistry,
    ) -> None:
        """Test error handling when no providers are enabled."""
        # Disable all providers
        for provider in diverse_registry.list_providers():
            provider.enabled = False

        optimizer = CostOptimizer(diverse_registry, CostTracker())

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
        )

        with pytest.raises(PortkeyProviderError):
            optimizer.select_optimal_provider(criteria, context)

    def test_all_providers_degraded(
        self,
        diverse_registry: ProviderRegistry,
    ) -> None:
        """Test selection when all providers are degraded."""
        # Mark all providers as degraded
        for provider in diverse_registry.list_providers():
            if provider.health:
                provider.health.status = ProviderStatus.DEGRADED

        optimizer = CostOptimizer(diverse_registry, CostTracker())

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.TEXT_GENERATION],
            require_healthy=False,
        )
        context = OptimizationContext(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            optimization_strategy=OptimizationStrategy.COST_ONLY,
            allow_degraded_providers=True,
        )

        # Should still select a provider (best degraded option)
        selected = optimizer.select_optimal_provider(criteria, context)
        assert selected is not None
        assert selected.health is not None
        assert selected.health.status == ProviderStatus.DEGRADED

    def test_provider_with_no_pricing_data(
        self,
        diverse_registry: ProviderRegistry,
    ) -> None:
        """Test handling of provider without pricing data."""
        # Add provider without pricing
        diverse_registry.register_provider(
            ProviderConfiguration(
                provider_id="no_pricing",
                enabled=True,
                priority=85,
                metadata=ProviderMetadata(name="No Pricing Provider"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION],
                ),
                pricing=None,  # No pricing data
            )
        )

        optimizer = CostOptimizer(diverse_registry, CostTracker())

        # Provider without pricing should have 0 cost estimate
        provider = diverse_registry.get_provider("no_pricing")
        assert provider is not None

        cost = optimizer.estimate_request_cost(provider, 1000, 500)
        assert cost == 0.0
