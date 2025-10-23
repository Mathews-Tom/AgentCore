"""Tests for cost optimization functionality."""

from datetime import datetime

import pytest

from agentcore.integration.portkey.cost_models import (
    OptimizationContext,
    OptimizationStrategy,
)
from agentcore.integration.portkey.cost_optimizer import CostOptimizer
from agentcore.integration.portkey.cost_tracker import CostTracker
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
def registry() -> ProviderRegistry:
    """Create a provider registry with test providers."""
    reg = ProviderRegistry()

    # High-cost, high-quality provider
    reg.register_provider(
        ProviderConfiguration(
            provider_id="expensive_provider",
            enabled=True,
            priority=100,
            metadata=ProviderMetadata(name="Expensive Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.REASONING,
                    ProviderCapability.FUNCTION_CALLING,
                ],
                supports_function_calling=True,
                supports_streaming=True,
                data_residency=[DataResidency.US_EAST, DataResidency.EU_WEST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.03,  # $0.03 per 1K tokens
                output_token_price=0.06,  # $0.06 per 1K tokens
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.99,
                average_latency_ms=800,
                availability_percent=99.9,
            ),
        )
    )

    # Low-cost, good-quality provider
    reg.register_provider(
        ProviderConfiguration(
            provider_id="cheap_provider",
            enabled=True,
            priority=90,
            metadata=ProviderMetadata(name="Cheap Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                ],
                supports_streaming=True,
                data_residency=[DataResidency.US_EAST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.0005,  # $0.0005 per 1K tokens
                output_token_price=0.0015,  # $0.0015 per 1K tokens
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.97,
                average_latency_ms=1200,
                availability_percent=98.5,
            ),
        )
    )

    # Medium-cost, balanced provider
    reg.register_provider(
        ProviderConfiguration(
            provider_id="balanced_provider",
            enabled=True,
            priority=95,
            metadata=ProviderMetadata(name="Balanced Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.REASONING,
                ],
                supports_streaming=True,
                supports_function_calling=True,
                data_residency=[DataResidency.US_EAST, DataResidency.US_WEST],
            ),
            pricing=ProviderPricing(
                input_token_price=0.01,  # $0.01 per 1K tokens
                output_token_price=0.03,  # $0.03 per 1K tokens
            ),
            health=ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.98,
                average_latency_ms=1000,
                availability_percent=99.0,
            ),
        )
    )

    return reg


@pytest.fixture
def cost_tracker() -> CostTracker:
    """Create a cost tracker for testing."""
    return CostTracker()


@pytest.fixture
def cost_optimizer(registry: ProviderRegistry, cost_tracker: CostTracker) -> CostOptimizer:
    """Create a cost optimizer for testing."""
    return CostOptimizer(registry, cost_tracker)


def test_select_optimal_provider_cost_only(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting optimal provider with cost-only strategy."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.COST_ONLY,
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should select cheap_provider for cost-only strategy
    assert selected.provider_id == "cheap_provider"


def test_select_optimal_provider_performance_first(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting optimal provider with performance-first strategy."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
        max_acceptable_latency_ms=1000,
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should select expensive_provider for performance (lowest latency)
    assert selected.provider_id == "expensive_provider"


def test_select_optimal_provider_balanced(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting optimal provider with balanced strategy."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.BALANCED,
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should select balanced_provider or cheap_provider for balanced strategy
    assert selected.provider_id in ["balanced_provider", "cheap_provider"]


def test_select_optimal_provider_adaptive_high_priority(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting optimal provider with adaptive strategy (high priority)."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.ADAPTIVE,
        priority=9,  # High priority
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should favor performance for high priority
    assert selected.provider_id in ["expensive_provider", "balanced_provider"]


def test_select_optimal_provider_adaptive_low_priority(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting optimal provider with adaptive strategy (low priority)."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.ADAPTIVE,
        priority=2,  # Low priority
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should favor cost for low priority
    assert selected.provider_id == "cheap_provider"


def test_estimate_request_cost(
    cost_optimizer: CostOptimizer,
    registry: ProviderRegistry,
) -> None:
    """Test cost estimation for a request."""
    provider = registry.get_provider("expensive_provider")
    assert provider is not None

    # Calculate cost for 1000 input tokens and 500 output tokens
    # Expected: (1000/1000 * 0.03) + (500/1000 * 0.06) = 0.03 + 0.03 = 0.06
    cost = cost_optimizer.estimate_request_cost(provider, 1000, 500)

    assert cost == 0.06


def test_select_with_cost_constraint(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting provider with cost constraint."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        max_acceptable_cost=0.01,  # Very low cost constraint
        optimization_strategy=OptimizationStrategy.COST_ONLY,
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should select cheap_provider to meet cost constraint
    assert selected.provider_id == "cheap_provider"

    # Verify estimated cost is within constraint
    estimated_cost = cost_optimizer.estimate_request_cost(
        selected,
        context.estimated_input_tokens,
        context.estimated_output_tokens,
    )
    assert estimated_cost <= context.max_acceptable_cost


def test_select_with_latency_constraint(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting provider with latency constraint."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        max_acceptable_latency_ms=900,  # Low latency requirement
        optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Should select expensive_provider to meet latency constraint
    assert selected.provider_id == "expensive_provider"
    assert selected.health is not None
    assert selected.health.average_latency_ms is not None
    assert selected.health.average_latency_ms <= context.max_acceptable_latency_ms


def test_select_with_capability_requirements(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test selecting provider with specific capability requirements."""
    criteria = ProviderSelectionCriteria(
        required_capabilities=[
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.REASONING,
            ProviderCapability.FUNCTION_CALLING,
        ],
    )

    context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.BALANCED,
    )

    selected = cost_optimizer.select_optimal_provider(criteria, context)

    # Only expensive_provider has all required capabilities
    assert selected.provider_id == "expensive_provider"

    # Verify capabilities
    assert ProviderCapability.TEXT_GENERATION in selected.capabilities.capabilities
    assert ProviderCapability.REASONING in selected.capabilities.capabilities
    assert ProviderCapability.FUNCTION_CALLING in selected.capabilities.capabilities


def test_generate_cost_report(
    cost_optimizer: CostOptimizer,
    cost_tracker: CostTracker,
) -> None:
    """Test generating cost report."""
    # Record some cost history
    from agentcore.integration.portkey.cost_models import CostMetrics

    for i in range(10):
        metrics = CostMetrics(
            total_cost=0.05 * (i + 1),
            input_cost=0.02 * (i + 1),
            output_cost=0.03 * (i + 1),
            input_tokens=1000 * (i + 1),
            output_tokens=1500 * (i + 1),
            provider_id="expensive_provider" if i < 5 else "cheap_provider",
            model="gpt-4",
            timestamp=datetime.now(),
            tenant_id="tenant-1",
        )
        cost_tracker.record_cost(metrics)

    report = cost_optimizer.generate_cost_report()

    assert report.summary.total_requests == 10
    assert report.summary.total_cost > 0
    assert len(report.summary.provider_breakdown) == 2
    assert "expensive_provider" in report.summary.provider_breakdown
    assert "cheap_provider" in report.summary.provider_breakdown


def test_cost_optimization_recommendations(
    cost_optimizer: CostOptimizer,
    cost_tracker: CostTracker,
) -> None:
    """Test cost optimization recommendations."""
    from agentcore.integration.portkey.cost_models import CostMetrics

    # Record high cost from expensive provider
    for i in range(20):
        metrics = CostMetrics(
            total_cost=0.10,
            input_cost=0.04,
            output_cost=0.06,
            input_tokens=2000,
            output_tokens=1000,
            provider_id="expensive_provider",
            model="gpt-4",
            timestamp=datetime.now(),
            tenant_id="tenant-1",
        )
        cost_tracker.record_cost(metrics)

    report = cost_optimizer.generate_cost_report()

    # Should have recommendations since expensive_provider dominates
    assert len(report.recommendations) > 0

    # Check for provider switch recommendation
    provider_switch_recs = [
        r for r in report.recommendations if r.type == "provider_switch"
    ]
    assert len(provider_switch_recs) > 0


def test_cost_efficiency_score(
    cost_optimizer: CostOptimizer,
    cost_tracker: CostTracker,
) -> None:
    """Test cost efficiency score calculation."""
    from agentcore.integration.portkey.cost_models import CostMetrics

    # Record cost-efficient requests
    for i in range(10):
        metrics = CostMetrics(
            total_cost=0.001,  # Very low cost
            input_cost=0.0005,
            output_cost=0.0005,
            input_tokens=1000,
            output_tokens=333,
            provider_id="cheap_provider",
            model="gpt-4",
            timestamp=datetime.now(),
            tenant_id="tenant-1",
        )
        cost_tracker.record_cost(metrics)

    report = cost_optimizer.generate_cost_report()

    # Efficiency score should be high
    assert report.cost_efficiency_score > 50.0


def test_calculate_quality_score(
    cost_optimizer: CostOptimizer,
    registry: ProviderRegistry,
) -> None:
    """Test quality score calculation."""
    expensive = registry.get_provider("expensive_provider")
    cheap = registry.get_provider("cheap_provider")

    assert expensive is not None
    assert cheap is not None

    expensive_score = cost_optimizer._calculate_quality_score(expensive)  # type: ignore[attr-defined]
    cheap_score = cost_optimizer._calculate_quality_score(cheap)  # type: ignore[attr-defined]

    # Expensive provider should have higher quality score
    # (more capabilities, higher success rate)
    assert expensive_score > cheap_score
    assert 0.0 <= expensive_score <= 1.0
    assert 0.0 <= cheap_score <= 1.0


def test_calculate_availability_score(
    cost_optimizer: CostOptimizer,
    registry: ProviderRegistry,
) -> None:
    """Test availability score calculation."""
    expensive = registry.get_provider("expensive_provider")
    assert expensive is not None

    score = cost_optimizer._calculate_availability_score(expensive)  # type: ignore[attr-defined]

    # Should have high availability score (99.9% availability)
    assert score > 0.95
    assert score <= 1.0


def test_50_percent_cost_reduction_achievable(
    cost_optimizer: CostOptimizer,
) -> None:
    """Test that 50%+ cost reduction is achievable.

    This validates the main business requirement: 50%+ cost reduction
    through intelligent routing.
    """
    criteria = ProviderSelectionCriteria(
        required_capabilities=[ProviderCapability.TEXT_GENERATION],
    )

    # Context for expensive provider selection (no optimization)
    expensive_context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
    )

    expensive_provider = cost_optimizer.select_optimal_provider(
        criteria, expensive_context
    )
    expensive_cost = cost_optimizer.estimate_request_cost(
        expensive_provider,
        expensive_context.estimated_input_tokens,
        expensive_context.estimated_output_tokens,
    )

    # Context for cost-optimized selection
    optimized_context = OptimizationContext(
        estimated_input_tokens=1000,
        estimated_output_tokens=500,
        optimization_strategy=OptimizationStrategy.COST_ONLY,
    )

    optimized_provider = cost_optimizer.select_optimal_provider(
        criteria, optimized_context
    )
    optimized_cost = cost_optimizer.estimate_request_cost(
        optimized_provider,
        optimized_context.estimated_input_tokens,
        optimized_context.estimated_output_tokens,
    )

    # Calculate cost reduction
    cost_reduction_percent = (
        (expensive_cost - optimized_cost) / expensive_cost * 100
    )

    # Verify 50%+ cost reduction is achievable
    assert cost_reduction_percent >= 50.0

    # Log the actual reduction for visibility
    print(
        f"\nCost reduction achieved: {cost_reduction_percent:.1f}%"
    )
    print(f"Expensive cost: ${expensive_cost:.4f}")
    print(f"Optimized cost: ${optimized_cost:.4f}")
    print(f"Savings: ${expensive_cost - optimized_cost:.4f}")
