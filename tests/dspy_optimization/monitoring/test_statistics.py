"""Tests for statistical significance testing"""

import pytest

from agentcore.dspy_optimization.models import PerformanceMetrics
from agentcore.dspy_optimization.monitoring.statistics import (
    StatisticalTester,
    SignificanceTest,
    SignificanceResult,
    EffectSize,
)


@pytest.fixture
def statistical_tester():
    """Statistical tester fixture"""
    return StatisticalTester(confidence_level=0.95, significance_threshold=0.05)


@pytest.fixture
def baseline_samples():
    """Baseline performance samples with variation"""
    return [
        {
            "success_rate": 0.70 + i * 0.01,
            "avg_cost_per_task": 0.15,
            "avg_latency_ms": 3000,
            "quality_score": 0.75,
        }
        for i in range(50)
    ]


@pytest.fixture
def optimized_samples_improved():
    """Optimized samples showing significant improvement"""
    return [
        {
            "success_rate": 0.85 + i * 0.01,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.90,
        }
        for i in range(50)
    ]


@pytest.fixture
def optimized_samples_similar():
    """Optimized samples similar to baseline (no improvement)"""
    return [
        {
            "success_rate": 0.70 + i * 0.01,
            "avg_cost_per_task": 0.15,
            "avg_latency_ms": 3000,
            "quality_score": 0.75,
        }
        for i in range(50)
    ]


@pytest.mark.asyncio
async def test_statistical_tester_initialization():
    """Test tester initialization"""
    tester = StatisticalTester(confidence_level=0.95, significance_threshold=0.05)

    assert tester.confidence_level == 0.95
    assert tester.significance_threshold == 0.05


@pytest.mark.asyncio
async def test_compare_metrics_significant_improvement(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test comparing metrics with significant improvement"""
    result = await statistical_tester.compare_metrics(
        baseline_samples,
        optimized_samples_improved,
        test_type=SignificanceTest.WELCH_T_TEST,
    )

    assert isinstance(result, SignificanceResult)
    assert result.test_type == SignificanceTest.WELCH_T_TEST
    assert result.is_significant is True
    assert result.p_value < 0.05
    assert result.effect_size is not None
    assert len(result.confidence_intervals) > 0


@pytest.mark.asyncio
async def test_compare_metrics_no_improvement(
    statistical_tester, baseline_samples, optimized_samples_similar
):
    """Test comparing metrics with no significant improvement"""
    result = await statistical_tester.compare_metrics(
        baseline_samples,
        optimized_samples_similar,
        test_type=SignificanceTest.WELCH_T_TEST,
    )

    assert result.is_significant is False
    assert result.p_value > 0.05


@pytest.mark.asyncio
async def test_compare_metrics_insufficient_samples(statistical_tester):
    """Test comparing with insufficient samples"""
    baseline = [{"success_rate": 0.7}]
    optimized = [{"success_rate": 0.9}]

    with pytest.raises(ValueError, match="Insufficient samples"):
        await statistical_tester.compare_metrics(baseline, optimized)


@pytest.mark.asyncio
async def test_compare_metrics_t_test(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test standard t-test"""
    result = await statistical_tester.compare_metrics(
        baseline_samples,
        optimized_samples_improved,
        test_type=SignificanceTest.T_TEST,
    )

    assert result.test_type == SignificanceTest.T_TEST
    assert result.is_significant is True


@pytest.mark.asyncio
async def test_compare_metrics_mann_whitney(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test Mann-Whitney U test"""
    result = await statistical_tester.compare_metrics(
        baseline_samples,
        optimized_samples_improved,
        test_type=SignificanceTest.MANN_WHITNEY,
    )

    assert result.test_type == SignificanceTest.MANN_WHITNEY
    assert result.is_significant is True


@pytest.mark.asyncio
async def test_compare_metrics_paired_t_test(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test paired t-test"""
    result = await statistical_tester.compare_metrics(
        baseline_samples,
        optimized_samples_improved,
        test_type=SignificanceTest.PAIRED_T_TEST,
    )

    assert result.test_type == SignificanceTest.PAIRED_T_TEST
    assert result.is_significant is True


@pytest.mark.asyncio
async def test_validate_improvement_valid(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test validating significant improvement"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.15,
        avg_latency_ms=3000,
        quality_score=0.75,
    )

    optimized_metrics = PerformanceMetrics(
        success_rate=0.90,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.90,
    )

    is_valid, improvement, significance = await statistical_tester.validate_improvement(
        baseline_metrics,
        optimized_metrics,
        baseline_samples,
        optimized_samples_improved,
    )

    assert is_valid is True
    assert improvement > 0
    assert significance.is_significant is True


@pytest.mark.asyncio
async def test_validate_improvement_invalid(
    statistical_tester, baseline_samples, optimized_samples_similar
):
    """Test validating no improvement"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.15,
        avg_latency_ms=3000,
        quality_score=0.75,
    )

    optimized_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.15,
        avg_latency_ms=3000,
        quality_score=0.75,
    )

    is_valid, improvement, significance = await statistical_tester.validate_improvement(
        baseline_metrics,
        optimized_metrics,
        baseline_samples,
        optimized_samples_similar,
    )

    assert is_valid is False
    assert abs(improvement) < 5  # Minimal improvement


@pytest.mark.asyncio
async def test_effect_size_calculation(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test effect size calculation"""
    result = await statistical_tester.compare_metrics(
        baseline_samples, optimized_samples_improved
    )

    assert result.effect_size is not None
    assert isinstance(result.effect_size, EffectSize)
    assert result.effect_size.cohens_d > 0  # Positive effect
    assert result.effect_size.interpretation in [
        "negligible",
        "small",
        "medium",
        "large",
    ]


@pytest.mark.asyncio
async def test_effect_size_large(statistical_tester):
    """Test large effect size detection"""
    # Create samples with large difference
    baseline = [{"success_rate": 0.50 + i * 0.001} for i in range(50)]
    optimized = [{"success_rate": 0.90 + i * 0.001} for i in range(50)]

    result = await statistical_tester.compare_metrics(baseline, optimized)

    assert result.effect_size.interpretation == "large"
    assert result.effect_size.cohens_d > 0.8


@pytest.mark.asyncio
async def test_confidence_intervals(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test confidence interval calculation"""
    result = await statistical_tester.compare_metrics(
        baseline_samples, optimized_samples_improved
    )

    assert len(result.confidence_intervals) > 0

    for interval in result.confidence_intervals:
        assert interval.lower_bound <= interval.mean <= interval.upper_bound
        assert interval.confidence_level == 0.95
        assert interval.metric_name in [
            "success_rate",
            "avg_cost_per_task",
            "avg_latency_ms",
            "quality_score",
        ]


@pytest.mark.asyncio
async def test_sample_sizes_recorded(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test that sample sizes are recorded"""
    result = await statistical_tester.compare_metrics(
        baseline_samples, optimized_samples_improved
    )

    assert "baseline" in result.sample_sizes
    assert "optimized" in result.sample_sizes
    assert result.sample_sizes["baseline"] == len(baseline_samples)
    assert result.sample_sizes["optimized"] == len(optimized_samples_improved)


@pytest.mark.asyncio
async def test_metadata_includes_statistic(
    statistical_tester, baseline_samples, optimized_samples_improved
):
    """Test that metadata includes test statistic"""
    result = await statistical_tester.compare_metrics(
        baseline_samples, optimized_samples_improved
    )

    assert "test_statistic" in result.metadata
    assert isinstance(result.metadata["test_statistic"], float)


@pytest.mark.asyncio
async def test_calculate_required_sample_size(statistical_tester):
    """Test required sample size calculation"""
    # Calculate for detecting 10% improvement with baseline std of 0.1
    sample_size = await statistical_tester.calculate_required_sample_size(
        baseline_std=0.1,
        min_detectable_effect=0.1,
        power=0.8,
    )

    assert isinstance(sample_size, int)
    assert sample_size > 0
    # For these parameters, should need reasonable sample size
    assert 10 < sample_size < 1000


@pytest.mark.asyncio
async def test_different_confidence_levels():
    """Test with different confidence levels"""
    # 99% confidence level
    tester_99 = StatisticalTester(confidence_level=0.99, significance_threshold=0.01)

    baseline = [{"success_rate": 0.70 + i * 0.01} for i in range(50)]
    optimized = [{"success_rate": 0.85 + i * 0.01} for i in range(50)]

    result = await tester_99.compare_metrics(baseline, optimized)

    assert result.confidence_level == 0.99
    # Higher confidence level means stricter test
    # For significant improvement, should still be significant
    assert result.is_significant is True


@pytest.mark.asyncio
async def test_edge_case_identical_values(statistical_tester):
    """Test with identical values (no variance)"""
    baseline = [{"success_rate": 0.75}] * 50
    optimized = [{"success_rate": 0.75}] * 50

    result = await statistical_tester.compare_metrics(baseline, optimized)

    # No difference, should not be significant
    assert result.is_significant is False
    # P-value may be NaN when there's no variance, which is acceptable
    import math
    assert math.isnan(result.p_value) or result.p_value > 0.05


@pytest.mark.asyncio
async def test_edge_case_high_variance(statistical_tester):
    """Test with high variance samples"""
    import random

    random.seed(42)

    baseline = [
        {"success_rate": random.uniform(0.3, 0.9)} for _ in range(100)
    ]
    optimized = [
        {"success_rate": random.uniform(0.4, 0.95)} for _ in range(100)
    ]

    result = await statistical_tester.compare_metrics(baseline, optimized)

    # With high variance, may or may not be significant
    assert isinstance(result.is_significant, bool)
    assert 0 <= result.p_value <= 1
