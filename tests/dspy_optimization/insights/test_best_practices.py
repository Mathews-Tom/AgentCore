"""Tests for best practices extraction"""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.analytics.patterns import (
    OptimizationPattern,
    PatternConfidence,
    PatternType,
)
from agentcore.dspy_optimization.insights.best_practices import (
    BestPractice,
    BestPracticeCategory,
    BestPracticeExtractor,
)
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)


@pytest.fixture
def extractor() -> BestPracticeExtractor:
    """Create best practice extractor"""
    return BestPracticeExtractor()


@pytest.fixture
def sample_patterns() -> list[OptimizationPattern]:
    """Create sample patterns"""
    return [
        OptimizationPattern(
            pattern_type=PatternType.ALGORITHM_EFFECTIVENESS,
            pattern_key="miprov2",
            pattern_description="MIPROv2 algorithm",
            success_rate=0.85,
            sample_count=20,
            confidence=PatternConfidence.HIGH,
            avg_improvement=0.25,
            avg_iterations=100,
            avg_duration_seconds=300.0,
            best_results=["r1", "r2", "r3"],
        ),
        OptimizationPattern(
            pattern_type=PatternType.PARAMETER_COMBINATION,
            pattern_key="temp_0.7_iterations_100",
            pattern_description="Temperature 0.7 with 100 iterations",
            success_rate=0.80,
            sample_count=15,
            confidence=PatternConfidence.HIGH,
            avg_improvement=0.22,
            avg_iterations=100,
            avg_duration_seconds=250.0,
            common_parameters={"temperature": 0.7, "iterations": 100},
            best_results=["r4", "r5", "r6"],
        ),
    ]


@pytest.fixture
def sample_results() -> list[OptimizationResult]:
    """Create sample results"""
    results = []

    # Create successful results
    for i in range(10):
        result = OptimizationResult(
            optimization_id=f"opt_{i}",
            status=OptimizationStatus.COMPLETED,
            baseline_performance=PerformanceMetrics(
                success_rate=0.5, avg_cost_per_task=1.0, avg_latency_ms=100
            ),
            optimized_performance=PerformanceMetrics(
                success_rate=0.75, avg_cost_per_task=0.8, avg_latency_ms=85
            ),
            improvement_percentage=0.25,
            optimization_details=OptimizationDetails(
                algorithm_used="miprov2", iterations=100, parameters={}
            ),
        )
        results.append(result)

    # Create fast-converging results
    for i in range(5):
        result = OptimizationResult(
            optimization_id=f"fast_{i}",
            status=OptimizationStatus.COMPLETED,
            baseline_performance=PerformanceMetrics(
                success_rate=0.6, avg_cost_per_task=1.0, avg_latency_ms=100
            ),
            optimized_performance=PerformanceMetrics(
                success_rate=0.8, avg_cost_per_task=0.85, avg_latency_ms=90
            ),
            improvement_percentage=0.20,
            optimization_details=OptimizationDetails(
                algorithm_used="gepa", iterations=40, parameters={}
            ),
        )
        results.append(result)

    return results


@pytest.mark.asyncio
async def test_extract_from_patterns(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test extracting practices from patterns"""
    practices = await extractor.extract_from_patterns(sample_patterns, sample_results)

    assert len(practices) > 0
    assert all(isinstance(p, BestPractice) for p in practices)


@pytest.mark.asyncio
async def test_extract_algorithm_practices(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test extracting algorithm selection practices"""
    practices = await extractor._extract_algorithm_practices(
        sample_patterns, sample_results
    )

    # Should have algorithm practice for miprov2
    assert len(practices) > 0

    algo_practices = [
        p for p in practices if p.category == BestPracticeCategory.ALGORITHM_SELECTION
    ]
    assert len(algo_practices) > 0

    # Check practice details
    practice = algo_practices[0]
    assert "miprov2" in practice.title.lower()
    assert practice.confidence > 0.7
    assert len(practice.do_list) > 0
    assert len(practice.supporting_evidence) > 0


@pytest.mark.asyncio
async def test_extract_parameter_practices(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test extracting parameter tuning practices"""
    practices = await extractor._extract_parameter_practices(
        sample_patterns, sample_results
    )

    param_practices = [
        p for p in practices if p.category == BestPracticeCategory.PARAMETER_TUNING
    ]

    if len(param_practices) > 0:
        practice = param_practices[0]
        assert "parameter" in practice.title.lower()
        assert len(practice.metadata.get("parameters", {})) > 0


@pytest.mark.asyncio
async def test_extract_performance_practices_fast(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test extracting fast-converging performance practices"""
    # Add fast-converging pattern
    fast_pattern = OptimizationPattern(
        pattern_type=PatternType.IMPROVEMENT_TRAJECTORY,
        pattern_key="gepa_fast",
        pattern_description="Fast GEPA",
        success_rate=0.75,
        sample_count=10,
        confidence=PatternConfidence.MEDIUM,
        avg_improvement=0.20,
        avg_iterations=40,
        avg_duration_seconds=120.0,
        best_results=["r7", "r8"],
    )

    patterns = sample_patterns + [fast_pattern]
    practices = await extractor._extract_performance_practices(patterns, sample_results)

    # Should have fast-converging practice
    perf_practices = [
        p for p in practices if p.category == BestPracticeCategory.PERFORMANCE_OPTIMIZATION
    ]

    if len(perf_practices) > 0:
        practice = perf_practices[0]
        assert "fast" in practice.title.lower() or "quick" in practice.title.lower()


@pytest.mark.asyncio
async def test_extract_performance_practices_high_impact(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test extracting high-impact performance practices"""
    # Add high-performance pattern
    high_perf_pattern = OptimizationPattern(
        pattern_type=PatternType.IMPROVEMENT_TRAJECTORY,
        pattern_key="miprov2_high",
        pattern_description="High-impact MIPROv2",
        success_rate=0.80,
        sample_count=12,
        confidence=PatternConfidence.HIGH,
        avg_improvement=0.35,
        avg_iterations=150,
        avg_duration_seconds=450.0,
        best_results=["r9", "r10"],
    )

    patterns = sample_patterns + [high_perf_pattern]
    practices = await extractor._extract_performance_practices(patterns, sample_results)

    # Should have high-impact practice
    high_impact = [p for p in practices if "aggressive" in p.title.lower() or "high" in p.title.lower()]

    if len(high_impact) > 0:
        practice = high_impact[0]
        assert practice.impact == "high"


@pytest.mark.asyncio
async def test_extract_cost_practices(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test extracting cost optimization practices"""
    # Add efficient pattern
    efficient_pattern = OptimizationPattern(
        pattern_type=PatternType.COST_EFFICIENCY,
        pattern_key="gepa_efficient",
        pattern_description="Efficient GEPA",
        success_rate=0.78,
        sample_count=8,
        confidence=PatternConfidence.MEDIUM,
        avg_improvement=0.22,
        avg_iterations=60,
        avg_duration_seconds=180.0,
        best_results=["r11", "r12"],
    )

    patterns = sample_patterns + [efficient_pattern]
    practices = await extractor._extract_cost_practices(patterns, sample_results)

    cost_practices = [
        p for p in practices if p.category == BestPracticeCategory.COST_OPTIMIZATION
    ]

    if len(cost_practices) > 0:
        practice = cost_practices[0]
        assert "cost" in practice.title.lower() or "efficient" in practice.title.lower()


def test_generate_markdown_documentation(
    extractor: BestPracticeExtractor,
) -> None:
    """Test generating markdown documentation"""
    # Add some practices manually
    practice1 = BestPractice(
        practice_id="p1",
        category=BestPracticeCategory.ALGORITHM_SELECTION,
        title="Use MIPROv2",
        description="Best algorithm",
        rationale="High success rate",
        confidence=0.9,
        impact="high",
        do_list=["Use MIPROv2", "Monitor convergence"],
        dont_list=["Don't skip validation"],
    )
    extractor._practices["p1"] = practice1

    practice2 = BestPractice(
        practice_id="p2",
        category=BestPracticeCategory.PARAMETER_TUNING,
        title="Use temperature 0.7",
        description="Optimal temperature",
        rationale="Best results",
        confidence=0.8,
        impact="medium",
    )
    extractor._practices["p2"] = practice2

    # Generate documentation
    markdown = extractor.generate_markdown_documentation()

    assert "# DSPy Optimization Best Practices" in markdown
    assert "## Algorithm Selection" in markdown
    assert "## Parameter Tuning" in markdown
    assert "Use MIPROv2" in markdown
    assert "Use temperature 0.7" in markdown
    assert "**Do:**" in markdown
    assert "**Don't:**" in markdown


def test_format_practice_markdown(extractor: BestPracticeExtractor) -> None:
    """Test formatting single practice as markdown"""
    practice = BestPractice(
        practice_id="p1",
        category=BestPracticeCategory.ALGORITHM_SELECTION,
        title="Use MIPROv2",
        description="Best algorithm for most tasks",
        rationale="Proven track record",
        confidence=0.9,
        impact="high",
        do_list=["Use MIPROv2", "Monitor convergence"],
        dont_list=["Don't skip validation"],
        examples=["example1", "example2"],
        supporting_evidence=["r1", "r2", "r3"],
    )

    lines = extractor._format_practice_markdown(practice)
    markdown = "\n".join(lines)

    assert "### Use MIPROv2" in markdown
    assert "**Impact**: HIGH" in markdown
    assert "**Confidence**: 90%" in markdown
    assert "✅ Use MIPROv2" in markdown
    assert "❌ Don't skip validation" in markdown
    assert "`example1`" in markdown
    assert "Based on 3 optimization runs" in markdown


def test_get_practices_by_category(extractor: BestPracticeExtractor) -> None:
    """Test getting practices by category"""
    practice1 = BestPractice(
        practice_id="p1",
        category=BestPracticeCategory.ALGORITHM_SELECTION,
        title="Practice 1",
        description="Test",
        rationale="Test",
        confidence=0.8,
    )
    extractor._practices["p1"] = practice1

    practice2 = BestPractice(
        practice_id="p2",
        category=BestPracticeCategory.PARAMETER_TUNING,
        title="Practice 2",
        description="Test",
        rationale="Test",
        confidence=0.7,
    )
    extractor._practices["p2"] = practice2

    # Get algorithm selection practices
    algo_practices = extractor.get_practices_by_category(
        BestPracticeCategory.ALGORITHM_SELECTION
    )

    assert len(algo_practices) == 1
    assert algo_practices[0].practice_id == "p1"


def test_get_high_impact_practices(extractor: BestPracticeExtractor) -> None:
    """Test getting high-impact practices"""
    practice1 = BestPractice(
        practice_id="p1",
        category=BestPracticeCategory.ALGORITHM_SELECTION,
        title="High Impact",
        description="Test",
        rationale="Test",
        confidence=0.9,
        impact="high",
    )
    extractor._practices["p1"] = practice1

    practice2 = BestPractice(
        practice_id="p2",
        category=BestPracticeCategory.PARAMETER_TUNING,
        title="Medium Impact",
        description="Test",
        rationale="Test",
        confidence=0.8,
        impact="medium",
    )
    extractor._practices["p2"] = practice2

    practice3 = BestPractice(
        practice_id="p3",
        category=BestPracticeCategory.COST_OPTIMIZATION,
        title="Low Confidence High",
        description="Test",
        rationale="Test",
        confidence=0.5,
        impact="high",
    )
    extractor._practices["p3"] = practice3

    # Get high-impact practices with min confidence
    high_impact = extractor.get_high_impact_practices(min_confidence=0.7)

    assert len(high_impact) == 1
    assert high_impact[0].practice_id == "p1"


def test_export_practices(extractor: BestPracticeExtractor) -> None:
    """Test exporting practices"""
    practice = BestPractice(
        practice_id="p1",
        category=BestPracticeCategory.ALGORITHM_SELECTION,
        title="Test Practice",
        description="Test",
        rationale="Test",
        confidence=0.8,
    )
    extractor._practices["p1"] = practice

    exported = extractor.export_practices()

    assert len(exported) == 1
    assert isinstance(exported[0], dict)
    assert exported[0]["practice_id"] == "p1"
    assert exported[0]["title"] == "Test Practice"


@pytest.mark.asyncio
async def test_practice_confidence_levels(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test that practices have appropriate confidence levels"""
    practices = await extractor.extract_from_patterns(sample_patterns, sample_results)

    for practice in practices:
        # Confidence should be in valid range
        assert 0.0 <= practice.confidence <= 1.0

        # High-impact practices should have higher confidence
        if practice.impact == "high":
            assert practice.confidence >= 0.7


@pytest.mark.asyncio
async def test_practice_do_dont_lists(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test that practices have do/don't lists"""
    practices = await extractor.extract_from_patterns(sample_patterns, sample_results)

    # At least some practices should have do/don't lists
    practices_with_dos = [p for p in practices if len(p.do_list) > 0]
    assert len(practices_with_dos) > 0

    practices_with_donts = [p for p in practices if len(p.dont_list) > 0]
    assert len(practices_with_donts) > 0


@pytest.mark.asyncio
async def test_practice_supporting_evidence(
    extractor: BestPracticeExtractor,
    sample_patterns: list[OptimizationPattern],
    sample_results: list[OptimizationResult],
) -> None:
    """Test that practices have supporting evidence"""
    practices = await extractor.extract_from_patterns(sample_patterns, sample_results)

    # Practices should have supporting evidence
    for practice in practices:
        assert len(practice.supporting_evidence) > 0
