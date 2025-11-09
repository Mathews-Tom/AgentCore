"""
Unit Tests for CapabilityEvaluator - ACE-025

Tests capability fitness evaluation, gap identification, and scoring algorithms.
Target: 95%+ code coverage
"""

import pytest
from datetime import UTC, datetime
from uuid import uuid4

from agentcore.ace.capability.evaluator import CapabilityEvaluator
from agentcore.ace.models.ace_models import (
    CapabilityType,
    TaskRequirement,
)


@pytest.fixture
def evaluator():
    """Create capability evaluator instance."""
    return CapabilityEvaluator()


@pytest.fixture
def task_requirements():
    """Create sample task requirements."""
    return [
        TaskRequirement(
            requirement_id="req-1",
            capability_type=CapabilityType.API,
            capability_name="api_client",
            required=True,
            weight=1.0,
        ),
        TaskRequirement(
            requirement_id="req-2",
            capability_type=CapabilityType.DATABASE,
            capability_name="database_query",
            required=True,
            weight=0.8,
        ),
        TaskRequirement(
            requirement_id="req-3",
            capability_type=CapabilityType.ANALYSIS,
            capability_name="data_analysis",
            required=False,
            weight=0.5,
        ),
    ]


@pytest.fixture
def performance_history():
    """Create sample performance history."""
    return {
        "total_executions": 100,
        "successful_executions": 85,
        "total_errors": 15,
        "execution_times": [500, 600, 450, 550, 520] * 20,
        "resource_usage": {
            "cpu_percent": 30.0,
            "memory_percent": 25.0,
        },
    }


@pytest.mark.asyncio
class TestCapabilityEvaluator:
    """Test suite for CapabilityEvaluator."""

    async def test_evaluate_fitness_basic(self, evaluator):
        """Test basic fitness evaluation."""
        fitness = await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="api_client",
            capability_name="API Client",
        )

        assert fitness is not None
        assert fitness.capability_id == "api_client"
        assert fitness.agent_id == "agent-001"
        assert 0.0 <= fitness.fitness_score <= 1.0
        assert 0.0 <= fitness.coverage_score <= 1.0
        assert 0.0 <= fitness.performance_score <= 1.0

    async def test_evaluate_fitness_with_requirements(
        self, evaluator, task_requirements
    ):
        """Test fitness evaluation with task requirements."""
        fitness = await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="api_client",
            capability_name="api_client",
            task_requirements=task_requirements,
        )

        assert fitness.coverage_score > 0.0  # Should match requirement
        assert fitness.task_type is None

    async def test_evaluate_fitness_with_performance_history(
        self, evaluator, performance_history
    ):
        """Test fitness evaluation with performance history."""
        fitness = await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="api_client",
            capability_name="API Client",
            performance_history=performance_history,
        )

        # Should have metrics from history
        assert fitness.metrics.usage_frequency == 100
        assert fitness.metrics.success_rate == 0.85  # 85/100
        assert fitness.metrics.error_correlation > 0.0
        assert fitness.metrics.avg_execution_time_ms > 0.0

    async def test_evaluate_fitness_caching(self, evaluator):
        """Test fitness evaluation caching."""
        # First evaluation
        fitness1 = await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="api_client",
            capability_name="API Client",
        )

        # Second evaluation (should hit cache)
        fitness2 = await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="api_client",
            capability_name="API Client",
        )

        # Should return same result from cache
        assert fitness1.evaluated_at == fitness2.evaluated_at

        # Cache stats
        stats = evaluator.get_cache_stats()
        assert stats["cache_size"] > 0

    async def test_evaluate_all_capabilities(
        self, evaluator, task_requirements
    ):
        """Test evaluating multiple capabilities."""
        capabilities = [
            {"id": "api_client", "name": "API Client"},
            {"id": "db_query", "name": "Database Query"},
            {"id": "file_reader", "name": "File Reader"},
        ]

        fitness_scores = await evaluator.evaluate_all_capabilities(
            agent_id="agent-001",
            current_capabilities=capabilities,
            task_requirements=task_requirements,
        )

        assert len(fitness_scores) == 3
        assert "api_client" in fitness_scores
        assert "db_query" in fitness_scores
        assert "file_reader" in fitness_scores

        # All should have valid fitness scores
        for cap_id, fitness in fitness_scores.items():
            assert 0.0 <= fitness.fitness_score <= 1.0

    async def test_identify_capability_gaps_missing(
        self, evaluator, task_requirements
    ):
        """Test identifying gaps for missing capabilities."""
        current_capabilities = ["file_reader"]  # Missing api_client, database_query

        gaps = await evaluator.identify_capability_gaps(
            agent_id="agent-001",
            current_capabilities=current_capabilities,
            task_requirements=task_requirements,
        )

        # Should identify gaps for required capabilities
        assert len(gaps) >= 2

        # Check critical gap for required capability
        critical_gaps = [g for g in gaps if g.gap_severity == "critical"]
        assert len(critical_gaps) > 0

        # Check gap details
        for gap in gaps:
            assert gap.required_capability in ["api_client", "database_query", "data_analysis"]
            assert 0.0 <= gap.impact <= 1.0
            assert gap.mitigation_suggestion is not None

    async def test_identify_capability_gaps_underperforming(
        self, evaluator, task_requirements
    ):
        """Test identifying gaps for underperforming capabilities."""
        from agentcore.ace.models.ace_models import CapabilityFitness, FitnessMetrics

        current_capabilities = ["api_client", "database_query"]

        # Create fitness scores with low performance
        fitness_scores = {
            "api_client": CapabilityFitness(
                capability_id="api_client",
                capability_name="API Client",
                agent_id="agent-001",
                fitness_score=0.2,  # Very low
                coverage_score=0.5,
                performance_score=0.2,
                metrics=FitnessMetrics(
                    success_rate=0.3,
                    error_correlation=0.7,
                    usage_frequency=10,
                    avg_execution_time_ms=2000.0,
                ),
                sample_size=10,
            ),
        }

        gaps = await evaluator.identify_capability_gaps(
            agent_id="agent-001",
            current_capabilities=current_capabilities,
            task_requirements=task_requirements,
            fitness_scores=fitness_scores,
        )

        # Should identify gap for underperforming capability
        underperforming_gaps = [
            g for g in gaps if g.current_fitness is not None and g.current_fitness < 0.5
        ]
        assert len(underperforming_gaps) > 0

    async def test_fitness_metrics_computation(self, evaluator, performance_history):
        """Test fitness metrics computation."""
        metrics = await evaluator._compute_fitness_metrics(
            agent_id="agent-001",
            capability_id="api_client",
            performance_history=performance_history,
        )

        assert metrics.success_rate == 0.85
        assert metrics.usage_frequency == 100
        assert 0.0 <= metrics.error_correlation <= 1.0
        assert metrics.avg_execution_time_ms > 0.0
        assert 0.0 <= metrics.resource_efficiency <= 1.0

    async def test_fitness_metrics_no_history(self, evaluator):
        """Test fitness metrics with no performance history."""
        metrics = await evaluator._compute_fitness_metrics(
            agent_id="agent-001",
            capability_id="api_client",
            performance_history=None,
        )

        # Should return default metrics
        assert metrics.success_rate == 0.5
        assert metrics.usage_frequency == 0
        assert metrics.error_correlation == 0.0

    async def test_coverage_score_computation(self, evaluator, task_requirements):
        """Test coverage score computation."""
        # Exact match - covers 1.0 of 2.3 total weight (api_client requirement)
        score1 = await evaluator._compute_coverage_score(
            capability_id="api_client",
            capability_name="api_client",
            task_requirements=task_requirements,
        )
        # Total weight: 1.0 + 0.8 + 0.5 = 2.3, matched weight: 1.0
        # Score: 1.0 / 2.3 ≈ 0.4348
        assert 0.43 < score1 < 0.44  # Partial coverage of total requirements

        # No match
        score2 = await evaluator._compute_coverage_score(
            capability_id="unknown",
            capability_name="unknown_capability",
            task_requirements=task_requirements,
        )
        assert score2 == 0.0  # No match

        # No requirements
        score3 = await evaluator._compute_coverage_score(
            capability_id="any",
            capability_name="any",
            task_requirements=None,
        )
        assert score3 == 1.0  # Full coverage if no requirements

    async def test_performance_score_computation(self, evaluator):
        """Test performance score computation."""
        from agentcore.ace.models.ace_models import FitnessMetrics

        metrics_high = FitnessMetrics(
            success_rate=0.95,
            error_correlation=0.05,
            usage_frequency=100,
            avg_execution_time_ms=500.0,
            resource_efficiency=0.9,
        )

        score_high = await evaluator._compute_performance_score(metrics_high)
        assert score_high > 0.8  # High performance

        metrics_low = FitnessMetrics(
            success_rate=0.3,
            error_correlation=0.7,
            usage_frequency=10,
            avg_execution_time_ms=5000.0,
            resource_efficiency=0.2,
        )

        score_low = await evaluator._compute_performance_score(metrics_low)
        assert score_low < 0.5  # Low performance

    async def test_overall_fitness_computation(self, evaluator):
        """Test overall fitness score computation."""
        from agentcore.ace.models.ace_models import FitnessMetrics

        metrics = FitnessMetrics(
            success_rate=0.85,
            error_correlation=0.15,
            usage_frequency=50,
            avg_execution_time_ms=800.0,
            resource_efficiency=0.75,
        )

        fitness = evaluator._compute_overall_fitness(
            coverage_score=0.9,
            performance_score=0.8,
            metrics=metrics,
        )

        assert 0.0 <= fitness <= 1.0
        # Should be weighted combination of inputs
        assert 0.5 < fitness < 1.0

    async def test_gap_mitigation_suggestions(self, evaluator):
        """Test gap mitigation suggestion generation."""
        req = TaskRequirement(
            requirement_id="req-1",
            capability_type=CapabilityType.API,
            capability_name="api_client",
            required=True,
            weight=1.0,
        )

        # Missing capability
        suggestion1 = evaluator._generate_gap_mitigation(
            requirement=req,
            has_capability=False,
            current_fitness=None,
        )
        assert "Add" in suggestion1

        # Low fitness
        suggestion2 = evaluator._generate_gap_mitigation(
            requirement=req,
            has_capability=True,
            current_fitness=0.2,
        )
        assert "Replace" in suggestion2 or "upgrade" in suggestion2

    async def test_cache_operations(self, evaluator):
        """Test cache management operations."""
        # Add item to cache
        await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="test_cap",
            capability_name="Test Capability",
        )

        stats_before = evaluator.get_cache_stats()
        assert stats_before["cache_size"] > 0

        # Clear cache
        evaluator.clear_cache()

        stats_after = evaluator.get_cache_stats()
        assert stats_after["cache_size"] == 0

    async def test_fitness_score_bounds(self, evaluator):
        """Test that fitness scores are always within bounds."""
        # Test with extreme values
        extreme_history = {
            "total_executions": 1000,
            "successful_executions": 1000,
            "total_errors": 0,
            "execution_times": [100] * 1000,
            "resource_usage": {"cpu_percent": 10.0, "memory_percent": 10.0},
        }

        fitness = await evaluator.evaluate_fitness(
            agent_id="agent-001",
            capability_id="perfect_cap",
            capability_name="Perfect Capability",
            performance_history=extreme_history,
        )

        # All scores should be within 0-1
        assert 0.0 <= fitness.fitness_score <= 1.0
        assert 0.0 <= fitness.coverage_score <= 1.0
        assert 0.0 <= fitness.performance_score <= 1.0
        assert 0.0 <= fitness.metrics.success_rate <= 1.0
        assert 0.0 <= fitness.metrics.error_correlation <= 1.0

    async def test_fitness_properties(self, evaluator):
        """Test CapabilityFitness model properties."""
        from agentcore.ace.models.ace_models import CapabilityFitness, FitnessMetrics

        # High fitness
        fitness_high = CapabilityFitness(
            capability_id="test",
            capability_name="Test",
            agent_id="agent-001",
            fitness_score=0.85,
            coverage_score=0.9,
            performance_score=0.8,
            metrics=FitnessMetrics(
                success_rate=0.9,
                error_correlation=0.1,
                usage_frequency=100,
                avg_execution_time_ms=500.0,
            ),
            sample_size=100,
        )

        assert fitness_high.is_fit is True
        assert fitness_high.fitness_level == "excellent"

        # Low fitness
        fitness_low = CapabilityFitness(
            capability_id="test",
            capability_name="Test",
            agent_id="agent-001",
            fitness_score=0.3,
            coverage_score=0.4,
            performance_score=0.3,
            metrics=FitnessMetrics(
                success_rate=0.3,
                error_correlation=0.7,
                usage_frequency=10,
                avg_execution_time_ms=2000.0,
            ),
            sample_size=10,
        )

        assert fitness_low.is_fit is False
        assert fitness_low.fitness_level == "poor"
