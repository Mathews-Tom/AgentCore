"""Tests for Agent Optimization Target Specification."""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.integration.target_spec import (
    AgentOptimizationProfile,
    AgentOptimizationTarget,
    OptimizationTargetBuilder,
)
from agentcore.dspy_optimization.models import (
    MetricType,
    OptimizationScope,
    OptimizationTargetType,
)


def test_create_target() -> None:
    """Test creating optimization target."""
    target = AgentOptimizationTarget.create_target("agent-123")

    assert target.type == OptimizationTargetType.AGENT
    assert target.id == "agent-123"
    assert target.scope == OptimizationScope.INDIVIDUAL


def test_create_target_with_scope() -> None:
    """Test creating optimization target with custom scope."""
    target = AgentOptimizationTarget.create_target(
        "agent-123", scope=OptimizationScope.POPULATION
    )

    assert target.scope == OptimizationScope.POPULATION


def test_create_cost_optimized_objectives() -> None:
    """Test creating cost-optimized objectives."""
    objectives = AgentOptimizationTarget.create_cost_optimized_objectives()

    assert len(objectives) == 3
    assert objectives[0].metric == MetricType.COST_EFFICIENCY
    assert objectives[0].weight == 1.0
    assert objectives[1].metric == MetricType.SUCCESS_RATE
    assert objectives[2].metric == MetricType.QUALITY_SCORE


def test_create_latency_optimized_objectives() -> None:
    """Test creating latency-optimized objectives."""
    objectives = AgentOptimizationTarget.create_latency_optimized_objectives()

    assert len(objectives) == 3
    assert objectives[0].metric == MetricType.LATENCY
    assert objectives[0].weight == 1.0


def test_create_quality_optimized_objectives() -> None:
    """Test creating quality-optimized objectives."""
    objectives = AgentOptimizationTarget.create_quality_optimized_objectives()

    assert len(objectives) == 3
    assert objectives[0].metric == MetricType.QUALITY_SCORE
    assert objectives[0].weight == 1.0


def test_create_balanced_objectives() -> None:
    """Test creating balanced objectives."""
    objectives = AgentOptimizationTarget.create_balanced_objectives()

    assert len(objectives) == 4
    assert all(obj.weight >= 0.7 for obj in objectives)


def test_create_objectives_from_profile_cost() -> None:
    """Test creating objectives from cost profile."""
    objectives = AgentOptimizationTarget.create_objectives_from_profile(
        AgentOptimizationProfile.COST_OPTIMIZED
    )

    assert len(objectives) > 0
    assert objectives[0].metric == MetricType.COST_EFFICIENCY


def test_create_objectives_from_profile_latency() -> None:
    """Test creating objectives from latency profile."""
    objectives = AgentOptimizationTarget.create_objectives_from_profile(
        AgentOptimizationProfile.LATENCY_OPTIMIZED
    )

    assert len(objectives) > 0
    assert objectives[0].metric == MetricType.LATENCY


def test_create_objectives_from_profile_quality() -> None:
    """Test creating objectives from quality profile."""
    objectives = AgentOptimizationTarget.create_objectives_from_profile(
        AgentOptimizationProfile.QUALITY_OPTIMIZED
    )

    assert len(objectives) > 0
    assert objectives[0].metric == MetricType.QUALITY_SCORE


def test_create_objectives_from_profile_balanced() -> None:
    """Test creating objectives from balanced profile."""
    objectives = AgentOptimizationTarget.create_objectives_from_profile(
        AgentOptimizationProfile.BALANCED
    )

    assert len(objectives) == 4


def test_create_objectives_from_invalid_profile() -> None:
    """Test creating objectives from invalid profile."""
    with pytest.raises(ValueError, match="Unknown optimization profile"):
        AgentOptimizationTarget.create_objectives_from_profile("invalid_profile")


def test_create_custom_objectives() -> None:
    """Test creating custom objectives from configuration."""
    config = [
        {"metric": "success_rate", "target_value": 0.98, "weight": 1.0},
        {"metric": "cost_efficiency", "target_value": 0.85, "weight": 0.7},
    ]

    objectives = AgentOptimizationTarget.create_custom_objectives(config)

    assert len(objectives) == 2
    assert objectives[0].metric == MetricType.SUCCESS_RATE
    assert objectives[0].target_value == 0.98
    assert objectives[0].weight == 1.0
    assert objectives[1].metric == MetricType.COST_EFFICIENCY


def test_create_custom_objectives_missing_metric() -> None:
    """Test creating custom objectives without metric."""
    config = [{"target_value": 0.98}]

    with pytest.raises(ValueError, match="must include 'metric'"):
        AgentOptimizationTarget.create_custom_objectives(config)


def test_create_custom_objectives_invalid_metric() -> None:
    """Test creating custom objectives with invalid metric."""
    config = [{"metric": "invalid_metric"}]

    with pytest.raises(ValueError, match="Invalid metric type"):
        AgentOptimizationTarget.create_custom_objectives(config)


def test_create_custom_objectives_with_defaults() -> None:
    """Test creating custom objectives with default values."""
    config = [{"metric": "success_rate"}]

    objectives = AgentOptimizationTarget.create_custom_objectives(config)

    assert len(objectives) == 1
    assert objectives[0].target_value == 0.9  # Default
    assert objectives[0].weight == 1.0  # Default


def test_create_optimization_request() -> None:
    """Test creating complete optimization request."""
    request = AgentOptimizationTarget.create_optimization_request(
        agent_id="agent-123",
        profile=AgentOptimizationProfile.BALANCED,
    )

    assert request.target.type == OptimizationTargetType.AGENT
    assert request.target.id == "agent-123"
    assert len(request.objectives) == 4
    assert len(request.algorithms) == 2
    assert "miprov2" in request.algorithms
    assert "gepa" in request.algorithms


def test_create_optimization_request_custom_algorithms() -> None:
    """Test creating optimization request with custom algorithms."""
    request = AgentOptimizationTarget.create_optimization_request(
        agent_id="agent-123",
        algorithms=["miprov2"],
    )

    assert len(request.algorithms) == 1
    assert request.algorithms[0] == "miprov2"


def test_create_optimization_request_custom_scope() -> None:
    """Test creating optimization request with custom scope."""
    request = AgentOptimizationTarget.create_optimization_request(
        agent_id="agent-123",
        scope=OptimizationScope.POPULATION,
    )

    assert request.target.scope == OptimizationScope.POPULATION


def test_create_multi_agent_request() -> None:
    """Test creating optimization requests for multiple agents."""
    agent_ids = ["agent-1", "agent-2", "agent-3"]

    requests = AgentOptimizationTarget.create_multi_agent_request(
        agent_ids=agent_ids,
        profile=AgentOptimizationProfile.COST_OPTIMIZED,
    )

    assert len(requests) == 3
    assert all(r.target.type == OptimizationTargetType.AGENT for r in requests)
    assert [r.target.id for r in requests] == agent_ids


def test_optimization_target_builder_basic() -> None:
    """Test optimization target builder basic usage."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = builder.build()

    assert request.target.id == "agent-123"
    assert len(request.objectives) > 0


def test_optimization_target_builder_with_profile() -> None:
    """Test optimization target builder with profile."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = builder.with_profile(AgentOptimizationProfile.COST_OPTIMIZED).build()

    assert request.objectives[0].metric == MetricType.COST_EFFICIENCY


def test_optimization_target_builder_with_algorithms() -> None:
    """Test optimization target builder with custom algorithms."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = builder.with_algorithms(["miprov2"]).build()

    assert len(request.algorithms) == 1
    assert request.algorithms[0] == "miprov2"


def test_optimization_target_builder_with_scope() -> None:
    """Test optimization target builder with custom scope."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = builder.with_scope(OptimizationScope.CROSS_DOMAIN).build()

    assert request.target.scope == OptimizationScope.CROSS_DOMAIN


def test_optimization_target_builder_add_objective() -> None:
    """Test optimization target builder with custom objective."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = (
        builder.add_objective(
            metric=MetricType.SUCCESS_RATE,
            target_value=0.99,
            weight=1.0,
        ).build()
    )

    assert len(request.objectives) == 1
    assert request.objectives[0].metric == MetricType.SUCCESS_RATE
    assert request.objectives[0].target_value == 0.99
    assert request.objectives[0].weight == 1.0


def test_optimization_target_builder_multiple_objectives() -> None:
    """Test optimization target builder with multiple custom objectives."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = (
        builder.add_objective(
            metric=MetricType.SUCCESS_RATE, target_value=0.99, weight=1.0
        )
        .add_objective(
            metric=MetricType.COST_EFFICIENCY, target_value=0.85, weight=0.8
        )
        .build()
    )

    assert len(request.objectives) == 2


def test_optimization_target_builder_chaining() -> None:
    """Test optimization target builder method chaining."""
    builder = OptimizationTargetBuilder(agent_id="agent-123")
    request = (
        builder.with_profile(AgentOptimizationProfile.LATENCY_OPTIMIZED)
        .with_algorithms(["miprov2", "gepa"])
        .with_scope(OptimizationScope.POPULATION)
        .build()
    )

    assert request.target.scope == OptimizationScope.POPULATION
    assert len(request.algorithms) == 2
    assert request.objectives[0].metric == MetricType.LATENCY


def test_optimization_target_builder_custom_objectives_override_profile() -> None:
    """Test that custom objectives override profile objectives."""
    builder = OptimizationTargetBuilder(
        agent_id="agent-123",
        profile=AgentOptimizationProfile.COST_OPTIMIZED,
    )
    request = (
        builder.add_objective(
            metric=MetricType.QUALITY_SCORE,
            target_value=0.95,
            weight=1.0,
        ).build()
    )

    # Custom objectives should be used, not profile objectives
    assert len(request.objectives) == 1
    assert request.objectives[0].metric == MetricType.QUALITY_SCORE
