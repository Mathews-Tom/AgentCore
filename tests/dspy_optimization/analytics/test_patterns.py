"""
Tests for pattern recognition
"""

import pytest

from agentcore.dspy_optimization.analytics.patterns import (
    PatternConfidence,
    PatternRecognizer,
    PatternType,
)
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationResult,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)


@pytest.fixture
def recognizer() -> PatternRecognizer:
    """Create pattern recognizer"""
    return PatternRecognizer()


@pytest.fixture
def successful_results() -> list[OptimizationResult]:
    """Create successful optimization results"""
    results = []
    for i in range(15):
        result = OptimizationResult(
            optimization_id=f"opt-{i:03d}",
            status=OptimizationStatus.COMPLETED,
            baseline_performance=PerformanceMetrics(
                success_rate=0.70,
                avg_cost_per_task=0.50,
                avg_latency_ms=500,
                quality_score=0.75,
            ),
            optimized_performance=PerformanceMetrics(
                success_rate=0.875 if i < 12 else 0.735,  # 80% success rate
                avg_cost_per_task=0.375,
                avg_latency_ms=375,
                quality_score=0.9375,
            ),
            improvement_percentage=0.25 if i < 12 else 0.05,
            statistical_significance=0.95,
            optimization_details=OptimizationDetails(
                algorithm_used="miprov2" if i < 10 else "gepa",
                iterations=10 + i,
                key_improvements=["prompt optimization"],
                parameters={
                    "target_type": "agent",
                    "max_iterations": 20,
                },
            ),
        )
        results.append(result)
    return results


@pytest.mark.asyncio
async def test_analyze_patterns(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test pattern analysis"""
    patterns = await recognizer.analyze_patterns(successful_results)

    assert len(patterns) > 0

    # Should have algorithm patterns
    algo_patterns = [p for p in patterns if p.pattern_type == PatternType.ALGORITHM_EFFECTIVENESS]
    assert len(algo_patterns) > 0


@pytest.mark.asyncio
async def test_algorithm_pattern_recognition(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test algorithm effectiveness pattern recognition"""
    patterns = await recognizer.analyze_patterns(successful_results)

    algo_patterns = [p for p in patterns if p.pattern_type == PatternType.ALGORITHM_EFFECTIVENESS]

    # Should recognize miprov2 pattern (10 uses, high success)
    mipro_pattern = next((p for p in algo_patterns if "miprov2" in p.pattern_key), None)
    assert mipro_pattern is not None
    assert mipro_pattern.sample_count >= 10
    assert mipro_pattern.success_rate >= 0.8


@pytest.mark.asyncio
async def test_get_recommendations(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test recommendation generation"""
    await recognizer.analyze_patterns(successful_results)

    target = OptimizationTarget(type=OptimizationTargetType.AGENT, id="test")
    recommendations = await recognizer.get_recommendations(target)

    assert len(recommendations) > 0
    assert any("algorithm" in rec.lower() or "mipro" in rec.lower() for rec in recommendations)


@pytest.mark.asyncio
async def test_find_similar_patterns(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test finding similar patterns"""
    await recognizer.analyze_patterns(successful_results)

    similar = await recognizer.find_similar_patterns(successful_results[0])

    assert len(similar) > 0
    assert all(p.confidence in [PatternConfidence.HIGH, PatternConfidence.MEDIUM] for p in similar)


@pytest.mark.asyncio
async def test_parameter_pattern_recognition(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test parameter combination pattern recognition"""
    patterns = await recognizer.analyze_patterns(successful_results)

    param_patterns = [p for p in patterns if p.pattern_type == PatternType.PARAMETER_COMBINATION]

    # Should find common parameter combinations
    if param_patterns:
        assert all(p.sample_count >= 3 for p in param_patterns)


@pytest.mark.asyncio
async def test_confidence_determination(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test pattern confidence calculation"""
    patterns = await recognizer.analyze_patterns(successful_results)

    # High success rate + many samples = high confidence
    high_conf = [p for p in patterns if p.confidence == PatternConfidence.HIGH]
    if high_conf:
        assert all(p.success_rate >= 0.80 and p.sample_count >= 10 for p in high_conf)


@pytest.mark.asyncio
async def test_context_based_recommendations(
    recognizer: PatternRecognizer,
    successful_results: list[OptimizationResult],
) -> None:
    """Test context-based recommendations"""
    await recognizer.analyze_patterns(successful_results)

    target = OptimizationTarget(type=OptimizationTargetType.AGENT, id="test")

    # Time-constrained context
    time_recs = await recognizer.get_recommendations(
        target,
        context={"time_constraint": "low"},
    )
    assert len(time_recs) > 0

    # Budget-constrained context
    budget_recs = await recognizer.get_recommendations(
        target,
        context={"budget_constraint": "low"},
    )
    assert len(budget_recs) > 0
