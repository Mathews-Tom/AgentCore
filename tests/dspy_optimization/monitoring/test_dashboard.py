"""Tests for dashboard service"""

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationScope,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.monitoring.baseline import (
    BaselineConfig,
    BaselineService,
)
from agentcore.dspy_optimization.monitoring.collector import (
    CollectorConfig,
    MetricsCollector,
)
from agentcore.dspy_optimization.monitoring.dashboard import (
    DashboardService,
    OptimizationHistory,
    PerformanceRecommendation,
    PerformanceTrend,
)


@pytest.fixture
def baseline_service():
    """Baseline service fixture"""
    config = BaselineConfig(min_samples=10)
    return BaselineService(config=config)


@pytest.fixture
def collector():
    """Metrics collector fixture"""
    return MetricsCollector()


@pytest.fixture
def dashboard_service(collector, baseline_service):
    """Dashboard service fixture"""
    return DashboardService(collector=collector, baseline_service=baseline_service)


@pytest.fixture
def sample_target():
    """Sample optimization target"""
    return OptimizationTarget(
        type=OptimizationTargetType.AGENT,
        id="test-agent-001",
        scope=OptimizationScope.INDIVIDUAL,
    )


@pytest.mark.asyncio
async def test_dashboard_service_initialization(
    dashboard_service, collector, baseline_service
):
    """Test dashboard service initialization"""
    assert dashboard_service.collector is collector
    assert dashboard_service.baseline_service is baseline_service
    assert len(dashboard_service._optimization_history) == 0


@pytest.mark.asyncio
async def test_get_dashboard_data(
    dashboard_service, sample_target, collector, baseline_service
):
    """Test getting complete dashboard data"""
    # Set up some data
    metrics = PerformanceMetrics(
        success_rate=0.85,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.9,
    )

    await collector.collect(sample_target, metrics)

    # Create baseline
    baseline_samples = [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.8,
        }
        for _ in range(50)
    ]
    await baseline_service.measure_baseline(sample_target, baseline_samples)

    # Get dashboard data
    dashboard = await dashboard_service.get_dashboard_data(sample_target, hours=24)

    assert dashboard.target == sample_target
    assert dashboard.current_metrics is not None
    assert dashboard.baseline_metrics is not None
    assert isinstance(dashboard.trends, list)
    assert isinstance(dashboard.optimization_history, list)
    assert isinstance(dashboard.recommendations, list)


@pytest.mark.asyncio
async def test_get_dashboard_data_no_metrics(dashboard_service, sample_target):
    """Test dashboard with no metrics"""
    dashboard = await dashboard_service.get_dashboard_data(sample_target)

    assert dashboard.current_metrics.success_rate == 0.0
    assert dashboard.baseline_metrics is None
    assert len(dashboard.trends) == 0


@pytest.mark.asyncio
async def test_get_performance_trends(dashboard_service, sample_target, collector):
    """Test getting performance trends"""
    # Collect metrics over time
    now = datetime.now(UTC)

    for i in range(10):
        metrics = PerformanceMetrics(
            success_rate=0.75 + i * 0.01,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.85,
        )
        snapshot = await collector.collect(sample_target, metrics)
        # Adjust timestamp
        snapshot.timestamp = now - timedelta(hours=10 - i)

    # Get trends
    start_time = now - timedelta(hours=12)
    end_time = now

    trends = await dashboard_service.get_performance_trends(
        sample_target,
        start_time,
        end_time,
        interval_minutes=120,  # 2-hour intervals
    )

    assert len(trends) > 0
    assert all(isinstance(t, PerformanceTrend) for t in trends)


@pytest.mark.asyncio
async def test_get_performance_trends_empty(dashboard_service, sample_target):
    """Test trends with no data"""
    now = datetime.now(UTC)
    start_time = now - timedelta(hours=24)
    end_time = now

    trends = await dashboard_service.get_performance_trends(
        sample_target, start_time, end_time
    )

    assert len(trends) == 0


@pytest.mark.asyncio
async def test_record_optimization(dashboard_service, sample_target):
    """Test recording optimization history"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.8,
    )

    optimized_metrics = PerformanceMetrics(
        success_rate=0.92,
        avg_cost_per_task=0.09,
        avg_latency_ms=2100,
        quality_score=0.95,
    )

    history = await dashboard_service.record_optimization(
        target=sample_target,
        algorithm="gepa",
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        improvement_percentage=22.7,
        version_id="v1.0.0",
    )

    assert isinstance(history, OptimizationHistory)
    assert history.algorithm == "gepa"
    assert history.improvement_percentage == 22.7
    assert history.status == OptimizationStatus.COMPLETED
    assert history.version_id == "v1.0.0"


@pytest.mark.asyncio
async def test_get_optimization_history(dashboard_service, sample_target):
    """Test getting optimization history"""
    # Record multiple optimizations
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.8,
    )

    for i in range(5):
        optimized_metrics = PerformanceMetrics(
            success_rate=0.80 + i * 0.02,
            avg_cost_per_task=0.11 - i * 0.01,
            avg_latency_ms=2400 - i * 100,
            quality_score=0.85 + i * 0.02,
        )

        await dashboard_service.record_optimization(
            target=sample_target,
            algorithm=f"algo_{i}",
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            improvement_percentage=10.0 + i * 2,
            version_id=f"v1.{i}",
        )

    # Get history
    history = await dashboard_service.get_optimization_history(sample_target)

    assert len(history) == 5
    # Should be sorted newest first
    assert history[0].version_id == "v1.4"
    assert history[-1].version_id == "v1.0"


@pytest.mark.asyncio
async def test_get_optimization_history_with_limit(dashboard_service, sample_target):
    """Test history with limit"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.8,
    )

    # Record 10 optimizations
    for i in range(10):
        optimized_metrics = PerformanceMetrics(
            success_rate=0.80 + i * 0.01,
            avg_cost_per_task=0.11,
            avg_latency_ms=2400,
            quality_score=0.85,
        )

        await dashboard_service.record_optimization(
            target=sample_target,
            algorithm="test",
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            improvement_percentage=10.0,
            version_id=f"v{i}",
        )

    # Get limited history
    history = await dashboard_service.get_optimization_history(sample_target, limit=3)

    assert len(history) == 3


@pytest.mark.asyncio
async def test_generate_recommendations_no_baseline(dashboard_service, sample_target):
    """Test recommendations without baseline"""
    current_metrics = PerformanceMetrics(
        success_rate=0.85,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.9,
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, None
    )

    assert len(recommendations) > 0
    assert any(r.type == "baseline_missing" for r in recommendations)


@pytest.mark.asyncio
async def test_generate_recommendations_low_success_rate(
    dashboard_service, sample_target
):
    """Test recommendations for low success rate"""
    current_metrics = PerformanceMetrics(
        success_rate=0.65,  # Below 70%
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.9,
    )

    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.8,
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, baseline_metrics
    )

    assert any(r.type == "low_success_rate" for r in recommendations)


@pytest.mark.asyncio
async def test_generate_recommendations_high_cost(dashboard_service, sample_target):
    """Test recommendations for high cost"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.10,
        avg_latency_ms=2500,
        quality_score=0.8,
    )

    current_metrics = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.13,  # 30% above baseline
        avg_latency_ms=2000,
        quality_score=0.9,
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, baseline_metrics
    )

    assert any(r.type == "high_cost" for r in recommendations)


@pytest.mark.asyncio
async def test_generate_recommendations_high_latency(dashboard_service, sample_target):
    """Test recommendations for high latency"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.8,
    )

    current_metrics = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.09,
        avg_latency_ms=3200,  # 60% above baseline
        quality_score=0.9,
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, baseline_metrics
    )

    assert any(r.type == "high_latency" for r in recommendations)


@pytest.mark.asyncio
async def test_generate_recommendations_optimization_opportunity(
    dashboard_service, sample_target
):
    """Test recommendations for optimization opportunity"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.8,
    )

    current_metrics = PerformanceMetrics(
        success_rate=0.75,  # Same as baseline
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.8,
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, baseline_metrics
    )

    assert any(r.type == "optimization_opportunity" for r in recommendations)


@pytest.mark.asyncio
async def test_generate_recommendations_low_quality(dashboard_service, sample_target):
    """Test recommendations for low quality"""
    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.8,
    )

    current_metrics = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.09,
        avg_latency_ms=2000,
        quality_score=0.55,  # Below 60%
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, baseline_metrics
    )

    assert any(r.type == "low_quality" for r in recommendations)


@pytest.mark.asyncio
async def test_recommendation_structure(dashboard_service, sample_target):
    """Test recommendation data structure"""
    current_metrics = PerformanceMetrics(
        success_rate=0.65,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.9,
    )

    baseline_metrics = PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.8,
    )

    recommendations = await dashboard_service.generate_recommendations(
        sample_target, current_metrics, baseline_metrics
    )

    for rec in recommendations:
        assert isinstance(rec, PerformanceRecommendation)
        assert rec.type is not None
        assert rec.suggestion is not None
        assert 0.0 <= rec.confidence <= 1.0
        assert rec.priority in ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_get_realtime_stats(
    dashboard_service, sample_target, collector, baseline_service
):
    """Test getting real-time statistics"""
    # Collect metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            success_rate=0.80 + i * 0.01,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.85,
        )
        await collector.collect(sample_target, metrics)

    # Create baseline
    baseline_samples = [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.8,
        }
        for _ in range(50)
    ]
    await baseline_service.measure_baseline(sample_target, baseline_samples)

    # Get stats
    stats = await dashboard_service.get_realtime_stats(sample_target)

    assert "current" in stats
    assert "1h_average" in stats
    assert "24h_average" in stats
    assert "baseline" in stats
    assert "baseline_age_hours" in stats


@pytest.mark.asyncio
async def test_get_realtime_stats_no_data(dashboard_service, sample_target):
    """Test real-time stats with no data"""
    stats = await dashboard_service.get_realtime_stats(sample_target)

    assert stats["current"] is None
    assert stats["1h_average"] is None
    assert stats["24h_average"] is None
    assert stats["baseline"] is None
    assert stats["baseline_age_hours"] is None


@pytest.mark.asyncio
async def test_trends_with_optimization_versions(
    dashboard_service, sample_target, collector
):
    """Test trends tracking optimization versions"""
    now = datetime.now(UTC)

    # Collect baseline metrics
    for i in range(5):
        metrics = PerformanceMetrics(
            success_rate=0.75,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.8,
        )
        snapshot = await collector.collect(
            sample_target,
            metrics,
            metadata={"version": "baseline", "is_optimized": False},
        )
        snapshot.timestamp = now - timedelta(hours=10 - i)

    # Collect optimized metrics
    for i in range(5, 10):
        metrics = PerformanceMetrics(
            success_rate=0.90,
            avg_cost_per_task=0.10,
            avg_latency_ms=2000,
            quality_score=0.95,
        )
        snapshot = await collector.collect(
            sample_target,
            metrics,
            metadata={"version": "v1.0", "is_optimized": True},
        )
        snapshot.timestamp = now - timedelta(hours=10 - i)

    # Get trends
    start_time = now - timedelta(hours=12)
    end_time = now

    trends = await dashboard_service.get_performance_trends(
        sample_target, start_time, end_time, interval_minutes=120
    )

    # Should have both baseline and optimized versions
    versions = {t.optimization_version for t in trends}
    assert "baseline" in versions or "v1.0" in versions
