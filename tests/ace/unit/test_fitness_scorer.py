"""
Unit Tests for FitnessScorer - ACE-026

Tests multi-factor fitness scoring algorithms.
Target: 95%+ code coverage
"""

import pytest
from datetime import UTC, datetime, timedelta

from agentcore.ace.capability.fitness_scorer import FitnessScorer
from agentcore.ace.models.ace_models import FitnessMetrics


@pytest.fixture
def scorer():
    """Create fitness scorer instance."""
    return FitnessScorer()


@pytest.fixture
def high_performance_metrics():
    """Create high-performance metrics."""
    return FitnessMetrics(
        success_rate=0.95,
        error_correlation=0.05,
        usage_frequency=100,
        avg_execution_time_ms=300.0,
        resource_efficiency=0.9,
    )


@pytest.fixture
def low_performance_metrics():
    """Create low-performance metrics."""
    return FitnessMetrics(
        success_rate=0.3,
        error_correlation=0.7,
        usage_frequency=10,
        avg_execution_time_ms=5000.0,
        resource_efficiency=0.2,
    )


class TestFitnessScorer:
    """Test suite for FitnessScorer."""

    def test_scorer_initialization(self):
        """Test scorer initialization with custom weights."""
        scorer = FitnessScorer(
            coverage_weight=0.5,
            performance_weight=0.3,
            efficiency_weight=0.2,
        )

        # Weights should be normalized
        total = (
            scorer.coverage_weight
            + scorer.performance_weight
            + scorer.efficiency_weight
        )
        assert abs(total - 1.0) < 0.001  # Should sum to 1.0

    def test_compute_coverage_score_exact_match(self, scorer):
        """Test coverage score with exact capability match."""
        score = scorer.compute_coverage_score(
            capability_name="api_client",
            required_capabilities=["api_client", "database_query"],
        )

        assert score == 1.0  # Perfect match

    def test_compute_coverage_score_no_match(self, scorer):
        """Test coverage score with no match."""
        score = scorer.compute_coverage_score(
            capability_name="file_reader",
            required_capabilities=["api_client", "database_query"],
        )

        assert score == 0.0  # No match

    def test_compute_coverage_score_partial_match(self, scorer):
        """Test coverage score with partial match."""
        score = scorer.compute_coverage_score(
            capability_name="api_rest_client",
            required_capabilities=["api_client"],
        )

        # Should have some similarity (contains "api")
        assert 0.0 < score < 1.0

    def test_compute_coverage_score_no_requirements(self, scorer):
        """Test coverage score with no requirements."""
        score = scorer.compute_coverage_score(
            capability_name="any_capability",
            required_capabilities=[],
        )

        assert score == 1.0  # Full coverage if no requirements

    def test_compute_performance_score_high(
        self, scorer, high_performance_metrics
    ):
        """Test performance score with high-performing metrics."""
        score = scorer.compute_performance_score(high_performance_metrics)

        assert score > 0.8  # Should be high
        assert 0.0 <= score <= 1.0

    def test_compute_performance_score_low(
        self, scorer, low_performance_metrics
    ):
        """Test performance score with low-performing metrics."""
        score = scorer.compute_performance_score(low_performance_metrics)

        assert score < 0.5  # Should be low
        assert 0.0 <= score <= 1.0

    def test_compute_performance_score_with_history(
        self, scorer, high_performance_metrics
    ):
        """Test performance score with task history."""
        task_history = [
            {"success": True, "duration_ms": 300},
            {"success": True, "duration_ms": 350},
            {"success": True, "duration_ms": 280},
            {"success": False, "duration_ms": 500},
            {"success": True, "duration_ms": 320},
            {"success": True, "duration_ms": 310},
        ]

        score = scorer.compute_performance_score(
            high_performance_metrics, task_history=task_history
        )

        # Should include consistency bonus
        assert 0.0 <= score <= 1.0

    def test_compute_efficiency_score_fast(self, scorer, high_performance_metrics):
        """Test efficiency score with fast execution."""
        score = scorer.compute_efficiency_score(
            high_performance_metrics,
            time_budget_ms=1000.0,
        )

        # Fast execution (300ms) within budget (1000ms)
        assert score > 0.7

    def test_compute_efficiency_score_slow(self, scorer, low_performance_metrics):
        """Test efficiency score with slow execution."""
        score = scorer.compute_efficiency_score(
            low_performance_metrics,
            time_budget_ms=1000.0,
        )

        # Slow execution (5000ms) exceeds budget (1000ms)
        assert score < 0.5

    def test_compute_efficiency_score_no_budget(
        self, scorer, high_performance_metrics
    ):
        """Test efficiency score without time budget."""
        score = scorer.compute_efficiency_score(high_performance_metrics)

        # Should use default budget (5s)
        assert 0.0 <= score <= 1.0

    def test_compute_overall_fitness(self, scorer):
        """Test overall fitness computation."""
        coverage = 0.9
        performance = 0.85
        efficiency = 0.8

        overall = scorer.compute_overall_fitness(
            coverage, performance, efficiency
        )

        # Should be weighted combination
        assert 0.0 <= overall <= 1.0
        assert overall > min(coverage, performance, efficiency)
        assert overall < max(coverage, performance, efficiency)

    def test_compute_overall_fitness_bounds(self, scorer):
        """Test overall fitness stays within bounds."""
        # All perfect scores
        score_high = scorer.compute_overall_fitness(1.0, 1.0, 1.0)
        assert score_high == 1.0

        # All zero scores
        score_low = scorer.compute_overall_fitness(0.0, 0.0, 0.0)
        assert score_low == 0.0

        # Mixed scores
        score_mid = scorer.compute_overall_fitness(0.5, 0.6, 0.4)
        assert 0.0 <= score_mid <= 1.0

    def test_compute_fitness_trend_improving(self, scorer):
        """Test fitness trend detection for improving performance."""
        base_time = datetime.now(UTC)
        historical_scores = [
            (base_time + timedelta(days=i), 0.5 + i * 0.05)
            for i in range(10)
        ]

        trend = scorer.compute_fitness_trend(historical_scores)

        assert trend["trend_direction"] == 1  # Improving
        assert trend["trend_strength"] > 0.0
        assert trend["recent_average"] > 0.5
        assert trend["change_rate"] > 0.0

    def test_compute_fitness_trend_declining(self, scorer):
        """Test fitness trend detection for declining performance."""
        base_time = datetime.now(UTC)
        historical_scores = [
            (base_time + timedelta(days=i), 0.9 - i * 0.05)
            for i in range(10)
        ]

        trend = scorer.compute_fitness_trend(historical_scores)

        assert trend["trend_direction"] == -1  # Declining
        assert trend["trend_strength"] > 0.0
        assert trend["recent_average"] < 0.9
        assert trend["change_rate"] < 0.0

    def test_compute_fitness_trend_stable(self, scorer):
        """Test fitness trend detection for stable performance."""
        base_time = datetime.now(UTC)
        historical_scores = [
            (base_time + timedelta(days=i), 0.7)
            for i in range(10)
        ]

        trend = scorer.compute_fitness_trend(historical_scores)

        assert trend["trend_direction"] == 0  # Stable
        assert trend["recent_average"] == 0.7

    def test_compute_fitness_trend_insufficient_data(self, scorer):
        """Test fitness trend with insufficient data."""
        trend = scorer.compute_fitness_trend([])

        assert trend["trend_direction"] == 0
        assert trend["trend_strength"] == 0.0
        assert trend["recent_average"] == 0.0

    def test_semantic_similarity_exact_match(self, scorer):
        """Test semantic similarity with exact match."""
        similarity = scorer._compute_semantic_similarity(
            "api_client", "api_client", None
        )

        assert similarity == 1.0

    def test_semantic_similarity_contains(self, scorer):
        """Test semantic similarity with contains match."""
        similarity = scorer._compute_semantic_similarity(
            "api_rest_client", "api_client", None
        )

        assert similarity == 0.8  # Contains match

    def test_semantic_similarity_word_overlap(self, scorer):
        """Test semantic similarity with word overlap."""
        similarity = scorer._compute_semantic_similarity(
            "api_rest_client", "rest_api_service", None
        )

        # Should have word overlap (api, rest)
        assert 0.0 < similarity < 1.0

    def test_semantic_similarity_no_match(self, scorer):
        """Test semantic similarity with no match."""
        similarity = scorer._compute_semantic_similarity(
            "database_query", "file_reader", None
        )

        assert similarity == 0.0

    def test_consistency_bonus_high_consistency(self, scorer):
        """Test consistency bonus for consistent performance."""
        task_history = [{"success": True} for _ in range(10)]

        bonus = scorer._compute_consistency_bonus(task_history)

        # High consistency should give higher bonus
        assert bonus > 0.1

    def test_consistency_bonus_low_consistency(self, scorer):
        """Test consistency bonus for inconsistent performance."""
        task_history = [
            {"success": i % 2 == 0} for i in range(10)
        ]  # Alternating

        bonus = scorer._compute_consistency_bonus(task_history)

        # Low consistency should give lower bonus
        assert bonus < 0.1

    def test_consistency_bonus_insufficient_data(self, scorer):
        """Test consistency bonus with insufficient data."""
        task_history = [{"success": True}]  # Only 1 record

        bonus = scorer._compute_consistency_bonus(task_history)

        assert bonus == 0.0

    def test_weight_normalization(self):
        """Test that weights are always normalized to sum to 1."""
        scorer1 = FitnessScorer(
            coverage_weight=2.0,
            performance_weight=3.0,
            efficiency_weight=5.0,
        )

        total = (
            scorer1.coverage_weight
            + scorer1.performance_weight
            + scorer1.efficiency_weight
        )

        assert abs(total - 1.0) < 0.001

    def test_score_boundaries(self, scorer, high_performance_metrics):
        """Test that all scores respect 0-1 boundaries."""
        # Coverage
        coverage = scorer.compute_coverage_score(
            "test", ["test1", "test2"]
        )
        assert 0.0 <= coverage <= 1.0

        # Performance
        performance = scorer.compute_performance_score(high_performance_metrics)
        assert 0.0 <= performance <= 1.0

        # Efficiency
        efficiency = scorer.compute_efficiency_score(high_performance_metrics)
        assert 0.0 <= efficiency <= 1.0

        # Overall
        overall = scorer.compute_overall_fitness(coverage, performance, efficiency)
        assert 0.0 <= overall <= 1.0
