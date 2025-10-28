"""Tests for recommendation engine"""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.analytics.patterns import (
    OptimizationPattern,
    PatternConfidence,
    PatternRecognizer,
    PatternType,
)
from agentcore.dspy_optimization.insights.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from agentcore.dspy_optimization.insights.recommendation_engine import (
    Recommendation,
    RecommendationContext,
    RecommendationEngine,
    RecommendationPriority,
    RecommendationType,
)
from agentcore.dspy_optimization.models import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationTarget,
    OptimizationTargetType,
    MetricType,
    OptimizationScope,
)


@pytest.fixture
def pattern_recognizer() -> PatternRecognizer:
    """Create pattern recognizer"""
    return PatternRecognizer()


@pytest.fixture
def knowledge_base() -> KnowledgeBase:
    """Create knowledge base"""
    return KnowledgeBase()


@pytest.fixture
def recommendation_engine(
    pattern_recognizer: PatternRecognizer, knowledge_base: KnowledgeBase
) -> RecommendationEngine:
    """Create recommendation engine"""
    return RecommendationEngine(pattern_recognizer, knowledge_base)


@pytest.fixture
def sample_context() -> RecommendationContext:
    """Create sample recommendation context"""
    return RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="agent_1",
            scope=OptimizationScope.INDIVIDUAL,
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.8, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(),
    )


@pytest.mark.asyncio
async def test_generate_recommendations_empty(
    recommendation_engine: RecommendationEngine, sample_context: RecommendationContext
) -> None:
    """Test generating recommendations with no data"""
    recommendations = await recommendation_engine.generate_recommendations(sample_context)

    # Should get at least constraint-based recommendations
    assert isinstance(recommendations, list)


@pytest.mark.asyncio
async def test_pattern_recommendations(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
    sample_context: RecommendationContext,
) -> None:
    """Test pattern-based recommendations"""
    # Add pattern to knowledge base
    pattern_entry = KnowledgeEntry(
        entry_id="pattern_1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="MIPROv2 Pattern",
        description="Highly effective pattern",
        confidence_score=0.9,
        success_rate=0.85,
        applicable_targets=[OptimizationTargetType.AGENT],
        evidence=["r1", "r2", "r3"],
        context={
            "pattern_key": "miprov2_agent",
            "avg_improvement": 0.25,
            "avg_iterations": 100,
            "common_parameters": {"temperature": 0.7},
        },
    )
    await knowledge_base.add_entry(pattern_entry)

    # Generate recommendations
    recommendations = await recommendation_engine.generate_recommendations(sample_context)

    # Should include pattern recommendation
    assert len(recommendations) > 0
    algo_recs = [r for r in recommendations if r.recommendation_type == RecommendationType.ALGORITHM]
    assert len(algo_recs) > 0


@pytest.mark.asyncio
async def test_knowledge_recommendations(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
    sample_context: RecommendationContext,
) -> None:
    """Test knowledge-based recommendations"""
    # Add best practice
    best_practice = KnowledgeEntry(
        entry_id="bp1",
        entry_type=KnowledgeEntryType.BEST_PRACTICE,
        title="Use MIPROv2",
        description="Best algorithm for agents",
        confidence_score=0.8,
        success_rate=0.9,
        applicable_targets=[OptimizationTargetType.AGENT],
        usage_count=10,
    )
    await knowledge_base.add_entry(best_practice)

    # Generate recommendations
    recommendations = await recommendation_engine.generate_recommendations(sample_context)

    # Should include best practice recommendation
    strategy_recs = [
        r for r in recommendations if r.recommendation_type == RecommendationType.STRATEGY
    ]
    assert len(strategy_recs) > 0


@pytest.mark.asyncio
async def test_constraint_recommendations_time(
    recommendation_engine: RecommendationEngine,
) -> None:
    """Test time constraint recommendations"""
    context = RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT, id="agent_1", scope=OptimizationScope.INDIVIDUAL
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.8, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(max_optimization_time=1200),  # 20 minutes
        time_constraint="low",
    )

    recommendations = await recommendation_engine.generate_recommendations(context)

    # Should include fast algorithm recommendation
    fast_recs = [r for r in recommendations if "fast" in r.tags]
    assert len(fast_recs) > 0


@pytest.mark.asyncio
async def test_constraint_recommendations_resource(
    recommendation_engine: RecommendationEngine,
) -> None:
    """Test resource constraint recommendations"""
    context = RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT, id="agent_1", scope=OptimizationScope.INDIVIDUAL
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.8, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(max_resource_usage=0.05),
        budget_constraint="low",
    )

    recommendations = await recommendation_engine.generate_recommendations(context)

    # Should include resource efficiency recommendation
    resource_recs = [
        r for r in recommendations if r.recommendation_type == RecommendationType.RESOURCE
    ]
    assert len(resource_recs) > 0


@pytest.mark.asyncio
async def test_constraint_recommendations_high_target(
    recommendation_engine: RecommendationEngine,
) -> None:
    """Test high improvement target recommendations"""
    context = RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT, id="agent_1", scope=OptimizationScope.INDIVIDUAL
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.9, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(min_improvement_threshold=0.25),
    )

    recommendations = await recommendation_engine.generate_recommendations(context)

    # Should include aggressive strategy recommendation
    aggressive_recs = [r for r in recommendations if "aggressive" in r.tags]
    assert len(aggressive_recs) > 0


@pytest.mark.asyncio
async def test_anti_pattern_warnings(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
    sample_context: RecommendationContext,
) -> None:
    """Test anti-pattern warnings"""
    # Add anti-pattern
    anti_pattern = KnowledgeEntry(
        entry_id="ap1",
        entry_type=KnowledgeEntryType.ANTI_PATTERN,
        title="Avoid Algorithm X",
        description="Algorithm X fails frequently",
        confidence_score=0.7,
        applicable_targets=[OptimizationTargetType.AGENT],
        tags=["unreliable", "algorithm_x"],
        metadata={"failure_rate": 0.6},
    )
    await knowledge_base.add_entry(anti_pattern)

    # Generate recommendations
    recommendations = await recommendation_engine.generate_recommendations(sample_context)

    # Should include warning
    warnings = [r for r in recommendations if r.recommendation_type == RecommendationType.WARNING]
    assert len(warnings) > 0
    assert any(r.priority == RecommendationPriority.CRITICAL for r in warnings)


@pytest.mark.asyncio
async def test_multiple_attempts_warning(
    recommendation_engine: RecommendationEngine,
) -> None:
    """Test warning for multiple attempts"""
    context = RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT, id="agent_1", scope=OptimizationScope.INDIVIDUAL
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.8, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(),
        previous_attempts=["a1", "a2", "a3", "a4", "a5"],
    )

    recommendations = await recommendation_engine.generate_recommendations(context)

    # Should include warning about multiple attempts
    warnings = [r for r in recommendations if "multiple-attempts" in r.tags]
    assert len(warnings) > 0


@pytest.mark.asyncio
async def test_avoid_algorithms(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
) -> None:
    """Test avoiding specific algorithms"""
    # Add pattern for algorithm
    pattern_entry = KnowledgeEntry(
        entry_id="pattern_1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Algorithm X Pattern",
        description="Pattern using Algorithm X",
        confidence_score=0.8,
        success_rate=0.9,
        applicable_targets=[OptimizationTargetType.AGENT],
        context={"pattern_key": "algorithm_x_agent"},
    )
    await knowledge_base.add_entry(pattern_entry)

    # Context with avoid list
    context = RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT, id="agent_1", scope=OptimizationScope.INDIVIDUAL
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.8, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(),
        avoid_algorithms=["algorithm_x"],
    )

    recommendations = await recommendation_engine.generate_recommendations(context)

    # Should not recommend algorithm_x
    algo_recs = [r for r in recommendations if "algorithm_x" in r.tags]
    assert len(algo_recs) == 0


@pytest.mark.asyncio
async def test_prioritization(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
    sample_context: RecommendationContext,
) -> None:
    """Test recommendation prioritization"""
    # Add entries with different priorities
    high_conf_entry = KnowledgeEntry(
        entry_id="high",
        entry_type=KnowledgeEntryType.PATTERN,
        title="High Confidence",
        description="High confidence pattern",
        confidence_score=0.95,
        success_rate=0.9,
        applicable_targets=[OptimizationTargetType.AGENT],
        context={"pattern_key": "high_algo"},
    )
    await knowledge_base.add_entry(high_conf_entry)

    low_conf_entry = KnowledgeEntry(
        entry_id="low",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Low Confidence",
        description="Low confidence pattern",
        confidence_score=0.6,
        success_rate=0.65,
        applicable_targets=[OptimizationTargetType.AGENT],
        context={"pattern_key": "low_algo"},
    )
    await knowledge_base.add_entry(low_conf_entry)

    # Generate recommendations
    recommendations = await recommendation_engine.generate_recommendations(sample_context)

    # High confidence should come first
    if len(recommendations) >= 2:
        assert recommendations[0].confidence >= recommendations[-1].confidence


@pytest.mark.asyncio
async def test_deduplication(recommendation_engine: RecommendationEngine) -> None:
    """Test recommendation deduplication"""
    recommendations = [
        Recommendation(
            title="Use MIPROv2",
            description="Desc 1",
            recommendation_type=RecommendationType.ALGORITHM,
            priority=RecommendationPriority.HIGH,
            rationale="Rationale 1",
            confidence=0.9,
        ),
        Recommendation(
            title="Use MIPROv2",
            description="Desc 2",
            recommendation_type=RecommendationType.ALGORITHM,
            priority=RecommendationPriority.MEDIUM,
            rationale="Rationale 2",
            confidence=0.8,
        ),
    ]

    deduplicated = recommendation_engine._deduplicate_recommendations(recommendations)

    assert len(deduplicated) == 1
    assert deduplicated[0].title == "Use MIPROv2"


@pytest.mark.asyncio
async def test_explain_recommendation(
    recommendation_engine: RecommendationEngine,
) -> None:
    """Test recommendation explanation"""
    recommendation = Recommendation(
        title="Use MIPROv2",
        description="Best algorithm",
        recommendation_type=RecommendationType.ALGORITHM,
        priority=RecommendationPriority.HIGH,
        rationale="High success rate",
        confidence=0.9,
        supporting_evidence=["r1", "r2", "r3"],
        expected_improvement=0.25,
        estimated_time=100,
        tags=["miprov2", "reliable"],
    )

    explanation = await recommendation_engine.explain_recommendation(recommendation)

    assert explanation["title"] == "Use MIPROv2"
    assert explanation["confidence"] == 0.9
    assert explanation["expected_improvement"] == "25.0%"
    assert explanation["estimated_iterations"] == 100
    assert len(explanation["evidence_sample"]) == 3


@pytest.mark.asyncio
async def test_max_recommendations_limit(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
    sample_context: RecommendationContext,
) -> None:
    """Test maximum recommendations limit"""
    # Add many entries
    for i in range(20):
        entry = KnowledgeEntry(
            entry_id=f"entry_{i}",
            entry_type=KnowledgeEntryType.PATTERN,
            title=f"Pattern {i}",
            description="Test",
            confidence_score=0.8,
            success_rate=0.75,
            applicable_targets=[OptimizationTargetType.AGENT],
            context={"pattern_key": f"algo_{i}"},
        )
        await knowledge_base.add_entry(entry)

    # Generate with limit
    recommendations = await recommendation_engine.generate_recommendations(
        sample_context, max_recommendations=5
    )

    assert len(recommendations) <= 5


@pytest.mark.asyncio
async def test_preferred_algorithms_boost(
    recommendation_engine: RecommendationEngine,
    knowledge_base: KnowledgeBase,
) -> None:
    """Test preferred algorithms get priority boost"""
    # Add two patterns with similar confidence
    pattern1 = KnowledgeEntry(
        entry_id="p1",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Pattern 1",
        description="Pattern 1",
        confidence_score=0.75,
        success_rate=0.8,
        applicable_targets=[OptimizationTargetType.AGENT],
        context={"pattern_key": "preferred_algo"},
        tags=["preferred_algo"],
    )
    await knowledge_base.add_entry(pattern1)

    pattern2 = KnowledgeEntry(
        entry_id="p2",
        entry_type=KnowledgeEntryType.PATTERN,
        title="Pattern 2",
        description="Pattern 2",
        confidence_score=0.75,
        success_rate=0.8,
        applicable_targets=[OptimizationTargetType.AGENT],
        context={"pattern_key": "other_algo"},
        tags=["other_algo"],
    )
    await knowledge_base.add_entry(pattern2)

    # Context with preferred algorithm
    context = RecommendationContext(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT, id="agent_1", scope=OptimizationScope.INDIVIDUAL
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.8, weight=1.0
            )
        ],
        constraints=OptimizationConstraints(),
        preferred_algorithms=["preferred_algo"],
    )

    recommendations = await recommendation_engine.generate_recommendations(context)

    # Preferred algorithm should rank higher
    if len(recommendations) >= 2:
        preferred_recs = [r for r in recommendations if "preferred_algo" in r.tags]
        if preferred_recs:
            first_preferred_idx = recommendations.index(preferred_recs[0])
            assert first_preferred_idx < len(recommendations) / 2  # In top half
