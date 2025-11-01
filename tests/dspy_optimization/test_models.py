"""
Tests for DSPy optimization models
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcore.dspy_optimization.models import (
    MetricType,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    OptimizationScope,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics model"""

    def test_valid_metrics(self) -> None:
        """Test creating valid performance metrics"""
        metrics = PerformanceMetrics(
            success_rate=0.85,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.9,
        )

        assert metrics.success_rate == 0.85
        assert metrics.avg_cost_per_task == 0.12
        assert metrics.avg_latency_ms == 2500
        assert metrics.quality_score == 0.9

    def test_success_rate_bounds(self) -> None:
        """Test success rate must be between 0 and 1"""
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                success_rate=1.5,
                avg_cost_per_task=0.12,
                avg_latency_ms=2500,
            )

    def test_negative_cost(self) -> None:
        """Test cost cannot be negative"""
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                success_rate=0.85,
                avg_cost_per_task=-0.12,
                avg_latency_ms=2500,
            )

    def test_negative_latency(self) -> None:
        """Test latency cannot be negative"""
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                success_rate=0.85,
                avg_cost_per_task=0.12,
                avg_latency_ms=-100,
            )


class TestOptimizationTarget:
    """Tests for OptimizationTarget model"""

    def test_agent_target(self) -> None:
        """Test creating agent optimization target"""
        target = OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="agent-123",
            scope=OptimizationScope.INDIVIDUAL,
        )

        assert target.type == OptimizationTargetType.AGENT
        assert target.id == "agent-123"
        assert target.scope == OptimizationScope.INDIVIDUAL

    def test_workflow_target(self) -> None:
        """Test creating workflow optimization target"""
        target = OptimizationTarget(
            type=OptimizationTargetType.WORKFLOW,
            id="workflow-456",
            scope=OptimizationScope.POPULATION,
        )

        assert target.type == OptimizationTargetType.WORKFLOW
        assert target.scope == OptimizationScope.POPULATION


class TestOptimizationObjective:
    """Tests for OptimizationObjective model"""

    def test_success_rate_objective(self) -> None:
        """Test success rate optimization objective"""
        objective = OptimizationObjective(
            metric=MetricType.SUCCESS_RATE,
            target_value=0.85,
            weight=0.4,
        )

        assert objective.metric == MetricType.SUCCESS_RATE
        assert objective.target_value == 0.85
        assert objective.weight == 0.4

    def test_default_weight(self) -> None:
        """Test default weight is 1.0"""
        objective = OptimizationObjective(
            metric=MetricType.QUALITY_SCORE,
            target_value=0.9,
        )

        assert objective.weight == 1.0


class TestOptimizationConstraints:
    """Tests for OptimizationConstraints model"""

    def test_default_constraints(self) -> None:
        """Test default constraint values"""
        constraints = OptimizationConstraints()

        assert constraints.max_optimization_time == 7200
        assert constraints.min_improvement_threshold == 0.05
        assert constraints.max_resource_usage == 0.2

    def test_custom_constraints(self) -> None:
        """Test custom constraint values"""
        constraints = OptimizationConstraints(
            max_optimization_time=3600,
            min_improvement_threshold=0.1,
            max_resource_usage=0.3,
        )

        assert constraints.max_optimization_time == 3600
        assert constraints.min_improvement_threshold == 0.1
        assert constraints.max_resource_usage == 0.3


class TestOptimizationRequest:
    """Tests for OptimizationRequest model"""

    def test_valid_request(self) -> None:
        """Test creating valid optimization request"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="agent-123",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                    weight=0.6,
                ),
                OptimizationObjective(
                    metric=MetricType.COST_EFFICIENCY,
                    target_value=0.9,
                    weight=0.4,
                ),
            ],
            algorithms=["miprov2", "gepa"],
        )

        assert request.target.type == OptimizationTargetType.AGENT
        assert len(request.objectives) == 2
        assert "miprov2" in request.algorithms
        assert "gepa" in request.algorithms

    def test_default_algorithms(self) -> None:
        """Test default algorithms are miprov2 and gepa"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="agent-123",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                ),
            ],
        )

        assert request.algorithms == ["miprov2", "gepa"]


class TestOptimizationResult:
    """Tests for OptimizationResult model"""

    def test_pending_result(self) -> None:
        """Test creating pending optimization result"""
        result = OptimizationResult(status=OptimizationStatus.PENDING)

        assert result.status == OptimizationStatus.PENDING
        assert result.baseline_performance is None
        assert result.optimized_performance is None
        assert result.improvement_percentage == 0.0

    def test_completed_result(self) -> None:
        """Test creating completed optimization result"""
        baseline = PerformanceMetrics(
            success_rate=0.75,
            avg_cost_per_task=0.15,
            avg_latency_ms=3000,
        )

        optimized = PerformanceMetrics(
            success_rate=0.92,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
        )

        result = OptimizationResult(
            status=OptimizationStatus.COMPLETED,
            baseline_performance=baseline,
            optimized_performance=optimized,
            improvement_percentage=22.7,
            statistical_significance=0.001,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.baseline_performance == baseline
        assert result.optimized_performance == optimized
        assert result.improvement_percentage == 22.7

    def test_failed_result(self) -> None:
        """Test creating failed optimization result"""
        result = OptimizationResult(
            status=OptimizationStatus.FAILED,
            error_message="Optimization timeout",
        )

        assert result.status == OptimizationStatus.FAILED
        assert result.error_message == "Optimization timeout"

    def test_optimization_id_generated(self) -> None:
        """Test optimization_id is automatically generated"""
        result1 = OptimizationResult(status=OptimizationStatus.PENDING)
        result2 = OptimizationResult(status=OptimizationStatus.PENDING)

        assert result1.optimization_id != result2.optimization_id
        assert len(result1.optimization_id) > 0
