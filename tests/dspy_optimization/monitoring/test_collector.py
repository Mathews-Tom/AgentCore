"""Tests for metrics collection framework"""

from datetime import datetime, timedelta

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.monitoring.collector import (
    MetricsCollector,
    CollectorConfig,
    AggregationMethod,
    MetricSnapshot,
)


@pytest.fixture
def collector_config():
    """Collector configuration fixture"""
    return CollectorConfig(
        aggregation_interval_seconds=300,
        retention_days=90,
        max_snapshots_per_target=1000,
    )


@pytest.fixture
def collector(collector_config):
    """Metrics collector fixture"""
    return MetricsCollector(config=collector_config)


@pytest.fixture
def sample_target():
    """Sample optimization target"""
    return OptimizationTarget(
        type=OptimizationTargetType.AGENT,
        id="test-agent-001",
        scope=OptimizationScope.INDIVIDUAL,
    )


@pytest.fixture
def sample_metrics():
    """Sample performance metrics"""
    return PerformanceMetrics(
        success_rate=0.85,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.9,
    )


@pytest.mark.asyncio
async def test_collector_config_defaults():
    """Test collector config defaults"""
    config = CollectorConfig()
    assert config.aggregation_interval_seconds == 300
    assert config.retention_days == 90
    assert config.max_snapshots_per_target == 10000


@pytest.mark.asyncio
async def test_collector_initialization(collector):
    """Test collector initialization"""
    assert collector.config is not None
    assert len(collector._snapshots) == 0
    assert len(collector._aggregations) == 0


@pytest.mark.asyncio
async def test_collect_snapshot(collector, sample_target, sample_metrics):
    """Test collecting single snapshot"""
    snapshot = await collector.collect(sample_target, sample_metrics)

    assert isinstance(snapshot, MetricSnapshot)
    assert snapshot.target == sample_target
    assert snapshot.metrics == sample_metrics
    assert snapshot.sample_count == 1


@pytest.mark.asyncio
async def test_collect_with_metadata(collector, sample_target, sample_metrics):
    """Test collecting snapshot with metadata"""
    metadata = {"version": "v1.0", "is_optimized": True}

    snapshot = await collector.collect(
        sample_target, sample_metrics, metadata=metadata
    )

    assert snapshot.metadata == metadata


@pytest.mark.asyncio
async def test_collect_batch(collector, sample_target):
    """Test collecting batch of metrics"""
    metrics_list = [
        PerformanceMetrics(
            success_rate=0.80 + i * 0.01,
            avg_cost_per_task=0.12 - i * 0.01,
            avg_latency_ms=2500 - i * 100,
            quality_score=0.85 + i * 0.01,
        )
        for i in range(10)
    ]

    snapshots = await collector.collect_batch(sample_target, metrics_list)

    assert len(snapshots) == 10
    assert all(isinstance(s, MetricSnapshot) for s in snapshots)


@pytest.mark.asyncio
async def test_get_snapshots(collector, sample_target, sample_metrics):
    """Test getting snapshots"""
    # Collect some snapshots
    for _ in range(5):
        await collector.collect(sample_target, sample_metrics)

    # Get all snapshots
    snapshots = await collector.get_snapshots(sample_target)

    assert len(snapshots) == 5
    assert all(s.target == sample_target for s in snapshots)


@pytest.mark.asyncio
async def test_get_snapshots_with_time_filter(collector, sample_target):
    """Test getting snapshots with time filter"""
    # Collect snapshots with different timestamps
    now = datetime.utcnow()

    for i in range(5):
        metrics = PerformanceMetrics(
            success_rate=0.80 + i * 0.01,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.85,
        )
        snapshot = await collector.collect(sample_target, metrics)
        # Manually adjust timestamp
        snapshot.timestamp = now - timedelta(hours=i)

    # Get snapshots from last 2 hours
    start_time = now - timedelta(hours=2)
    snapshots = await collector.get_snapshots(
        sample_target, start_time=start_time
    )

    assert len(snapshots) <= 3  # 0, 1, 2 hours ago


@pytest.mark.asyncio
async def test_get_snapshots_with_limit(collector, sample_target, sample_metrics):
    """Test getting snapshots with limit"""
    # Collect 10 snapshots
    for _ in range(10):
        await collector.collect(sample_target, sample_metrics)

    # Get limited snapshots
    snapshots = await collector.get_snapshots(sample_target, limit=5)

    assert len(snapshots) == 5


@pytest.mark.asyncio
async def test_aggregate_average(collector, sample_target):
    """Test average aggregation"""
    # Collect varied metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            success_rate=0.70 + i * 0.01,
            avg_cost_per_task=0.15 - i * 0.01,
            avg_latency_ms=3000 - i * 100,
            quality_score=0.75 + i * 0.01,
        )
        await collector.collect(sample_target, metrics)

    # Aggregate
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    aggregation = await collector.aggregate(
        sample_target,
        start_time,
        end_time,
        AggregationMethod.AVERAGE,
    )

    assert aggregation.aggregation_method == AggregationMethod.AVERAGE
    assert aggregation.snapshot_count == 10
    # Average of 0.70 to 0.79 should be around 0.745
    assert 0.74 <= aggregation.aggregated_metrics.success_rate <= 0.75


@pytest.mark.asyncio
async def test_aggregate_median(collector, sample_target):
    """Test median aggregation"""
    # Collect metrics
    for i in range(11):  # Odd number for clear median
        metrics = PerformanceMetrics(
            success_rate=0.60 + i * 0.02,
            avg_cost_per_task=0.20,
            avg_latency_ms=2000,
            quality_score=0.80,
        )
        await collector.collect(sample_target, metrics)

    # Aggregate
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    aggregation = await collector.aggregate(
        sample_target,
        start_time,
        end_time,
        AggregationMethod.MEDIAN,
    )

    # Median of [0.60, 0.62, ..., 0.80] is 0.70
    assert aggregation.aggregated_metrics.success_rate == 0.70


@pytest.mark.asyncio
async def test_aggregate_no_data(collector, sample_target):
    """Test aggregation with no data"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    with pytest.raises(ValueError, match="No metrics found"):
        await collector.aggregate(sample_target, start_time, end_time)


@pytest.mark.asyncio
async def test_get_latest_metrics(collector, sample_target):
    """Test getting latest metrics"""
    # Initially no metrics
    latest = await collector.get_latest_metrics(sample_target)
    assert latest is None

    # Collect some metrics
    metrics1 = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.85,
    )
    await collector.collect(sample_target, metrics1)

    metrics2 = PerformanceMetrics(
        success_rate=0.90,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.95,
    )
    await collector.collect(sample_target, metrics2)

    # Get latest
    latest = await collector.get_latest_metrics(sample_target)
    assert latest == metrics2


@pytest.mark.asyncio
async def test_get_average_metrics(collector, sample_target):
    """Test getting average metrics over window"""
    # Collect metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            success_rate=0.80 + i * 0.01,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.85,
        )
        await collector.collect(sample_target, metrics)

    # Get average
    avg = await collector.get_average_metrics(sample_target, window_hours=24)

    assert avg is not None
    assert 0.84 <= avg.success_rate <= 0.85


@pytest.mark.asyncio
async def test_get_average_metrics_no_data(collector, sample_target):
    """Test average metrics with no data"""
    avg = await collector.get_average_metrics(sample_target, window_hours=24)
    assert avg is None


@pytest.mark.asyncio
async def test_retention_enforcement(collector, sample_target, sample_metrics):
    """Test retention policy enforcement"""
    # Collect many snapshots
    for _ in range(1500):
        await collector.collect(sample_target, sample_metrics)

    # Should be limited to max_snapshots_per_target
    snapshots = await collector.get_snapshots(sample_target)
    assert len(snapshots) <= collector.config.max_snapshots_per_target


@pytest.mark.asyncio
async def test_percentile_95_aggregation(collector, sample_target):
    """Test 95th percentile aggregation"""
    # Collect 100 metrics
    for i in range(100):
        metrics = PerformanceMetrics(
            success_rate=0.50 + i * 0.005,  # 0.50 to 0.995
            avg_cost_per_task=0.10,
            avg_latency_ms=1000,
            quality_score=0.80,
        )
        await collector.collect(sample_target, metrics)

    # Aggregate
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    aggregation = await collector.aggregate(
        sample_target,
        start_time,
        end_time,
        AggregationMethod.PERCENTILE_95,
    )

    # 95th percentile should be around index 95
    assert aggregation.aggregated_metrics.success_rate >= 0.95


@pytest.mark.asyncio
async def test_min_max_aggregation(collector, sample_target):
    """Test min/max aggregation"""
    # Collect varied metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            success_rate=0.60 + i * 0.04,  # 0.60 to 0.96
            avg_cost_per_task=0.10,
            avg_latency_ms=1000,
            quality_score=0.80,
        )
        await collector.collect(sample_target, metrics)

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    # Test MIN
    min_agg = await collector.aggregate(
        sample_target,
        start_time,
        end_time,
        AggregationMethod.MIN,
    )
    assert min_agg.aggregated_metrics.success_rate == 0.60

    # Test MAX
    max_agg = await collector.aggregate(
        sample_target,
        start_time,
        end_time,
        AggregationMethod.MAX,
    )
    assert max_agg.aggregated_metrics.success_rate == 0.96


@pytest.mark.asyncio
async def test_multiple_targets_isolation(collector):
    """Test that multiple targets are isolated"""
    target1 = OptimizationTarget(
        type=OptimizationTargetType.AGENT,
        id="agent-001",
        scope=OptimizationScope.INDIVIDUAL,
    )

    target2 = OptimizationTarget(
        type=OptimizationTargetType.WORKFLOW,
        id="workflow-001",
        scope=OptimizationScope.POPULATION,
    )

    metrics1 = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.12,
        avg_latency_ms=2500,
        quality_score=0.85,
    )

    metrics2 = PerformanceMetrics(
        success_rate=0.90,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.95,
    )

    # Collect for both targets
    await collector.collect(target1, metrics1)
    await collector.collect(target2, metrics2)

    # Get snapshots for each
    snapshots1 = await collector.get_snapshots(target1)
    snapshots2 = await collector.get_snapshots(target2)

    assert len(snapshots1) == 1
    assert len(snapshots2) == 1
    assert snapshots1[0].metrics == metrics1
    assert snapshots2[0].metrics == metrics2
