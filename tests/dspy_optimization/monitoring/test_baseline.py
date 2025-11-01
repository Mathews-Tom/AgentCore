"""Tests for baseline performance measurement service"""

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationScope,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.monitoring.baseline import (
    BaselineConfig,
    BaselineMeasurement,
    BaselineService,
)


@pytest.fixture
def baseline_config():
    """Baseline configuration fixture"""
    return BaselineConfig(
        measurement_window_hours=24,
        min_samples=10,
        max_samples=1000,
        update_frequency_hours=168,
    )


@pytest.fixture
def baseline_service(baseline_config):
    """Baseline service fixture"""
    return BaselineService(config=baseline_config)


@pytest.fixture
def sample_target():
    """Sample optimization target"""
    return OptimizationTarget(
        type=OptimizationTargetType.AGENT,
        id="test-agent-001",
        scope=OptimizationScope.INDIVIDUAL,
    )


@pytest.fixture
def performance_samples():
    """Sample performance data"""
    return [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.8,
        }
        for _ in range(100)
    ]


@pytest.mark.asyncio
async def test_baseline_config_defaults():
    """Test baseline config defaults"""
    config = BaselineConfig()
    assert config.measurement_window_hours == 24
    assert config.min_samples == 100
    assert config.max_samples == 10000
    assert config.update_frequency_hours == 168


@pytest.mark.asyncio
async def test_baseline_service_initialization(baseline_service):
    """Test baseline service initialization"""
    assert baseline_service.config is not None
    assert len(baseline_service._baselines) == 0


@pytest.mark.asyncio
async def test_measure_baseline(baseline_service, sample_target, performance_samples):
    """Test baseline measurement"""
    measurement = await baseline_service.measure_baseline(
        sample_target, performance_samples
    )

    assert isinstance(measurement, BaselineMeasurement)
    assert measurement.target == sample_target
    assert measurement.sample_count == len(performance_samples)
    assert measurement.is_valid is True
    assert measurement.metrics.success_rate == 0.75
    assert measurement.metrics.avg_cost_per_task == 0.12
    assert measurement.metrics.avg_latency_ms == 2500
    assert measurement.metrics.quality_score == 0.8


@pytest.mark.asyncio
async def test_measure_baseline_insufficient_samples(baseline_service, sample_target):
    """Test baseline measurement with insufficient samples"""
    samples = [{"success_rate": 0.8}] * 5  # Less than min_samples

    with pytest.raises(ValueError, match="Insufficient samples"):
        await baseline_service.measure_baseline(sample_target, samples)


@pytest.mark.asyncio
async def test_measure_baseline_max_samples(baseline_service, sample_target):
    """Test baseline measurement with max samples limit"""
    # Create more samples than max_samples
    samples = [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.8,
        }
        for _ in range(2000)
    ]

    measurement = await baseline_service.measure_baseline(sample_target, samples)

    # Should be limited to max_samples
    assert measurement.sample_count == baseline_service.config.max_samples


@pytest.mark.asyncio
async def test_get_baseline(baseline_service, sample_target, performance_samples):
    """Test getting baseline measurement"""
    # Initially no baseline
    baseline = await baseline_service.get_baseline(sample_target)
    assert baseline is None

    # Measure baseline
    await baseline_service.measure_baseline(sample_target, performance_samples)

    # Now baseline exists
    baseline = await baseline_service.get_baseline(sample_target)
    assert baseline is not None
    assert baseline.target == sample_target


@pytest.mark.asyncio
async def test_update_baseline_fresh(
    baseline_service, sample_target, performance_samples
):
    """Test update baseline when baseline is fresh"""
    # Create fresh baseline
    original = await baseline_service.measure_baseline(
        sample_target, performance_samples
    )

    # Try to update immediately
    updated = await baseline_service.update_baseline(sample_target, performance_samples)

    # Should return original (still fresh)
    assert updated.id == original.id
    assert updated.created_at == original.created_at


@pytest.mark.asyncio
async def test_update_baseline_expired(baseline_service, sample_target):
    """Test update baseline when baseline is expired"""
    # Create baseline
    samples = [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.8,
        }
        for _ in range(100)
    ]

    original = await baseline_service.measure_baseline(sample_target, samples)

    # Manually expire baseline
    original.created_at = datetime.now(UTC) - timedelta(
        hours=baseline_service.config.update_frequency_hours + 1
    )

    # Update with new samples
    new_samples = [
        {
            "success_rate": 0.85,
            "avg_cost_per_task": 0.10,
            "avg_latency_ms": 2000,
            "quality_score": 0.9,
        }
        for _ in range(100)
    ]

    updated = await baseline_service.update_baseline(sample_target, new_samples)

    # Should be new baseline
    assert updated.id != original.id
    assert updated.metrics.success_rate == 0.85


@pytest.mark.asyncio
async def test_invalidate_baseline(
    baseline_service, sample_target, performance_samples
):
    """Test invalidating baseline"""
    # Create baseline
    await baseline_service.measure_baseline(sample_target, performance_samples)

    # Invalidate
    await baseline_service.invalidate_baseline(sample_target)

    # Check invalidation
    baseline = await baseline_service.get_baseline(sample_target)
    assert baseline.is_valid is False


@pytest.mark.asyncio
async def test_is_baseline_valid(baseline_service, sample_target, performance_samples):
    """Test baseline validity check"""
    # No baseline
    is_valid = await baseline_service.is_baseline_valid(sample_target)
    assert is_valid is False

    # Create baseline
    await baseline_service.measure_baseline(sample_target, performance_samples)

    # Should be valid
    is_valid = await baseline_service.is_baseline_valid(sample_target)
    assert is_valid is True

    # Invalidate
    await baseline_service.invalidate_baseline(sample_target)

    # Should be invalid
    is_valid = await baseline_service.is_baseline_valid(sample_target)
    assert is_valid is False


@pytest.mark.asyncio
async def test_is_baseline_valid_expired(baseline_service, sample_target):
    """Test baseline validity with expiration"""
    samples = [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2500,
            "quality_score": 0.8,
        }
        for _ in range(100)
    ]

    # Create baseline
    baseline = await baseline_service.measure_baseline(sample_target, samples)

    # Manually expire
    baseline.created_at = datetime.now(UTC) - timedelta(
        hours=baseline_service.config.update_frequency_hours + 1
    )

    # Should be invalid
    is_valid = await baseline_service.is_baseline_valid(sample_target)
    assert is_valid is False


@pytest.mark.asyncio
async def test_baseline_measurement_time_window(
    baseline_service, sample_target, performance_samples
):
    """Test baseline measurement time window"""
    measurement = await baseline_service.measure_baseline(
        sample_target, performance_samples
    )

    # Check time window
    expected_window = timedelta(hours=baseline_service.config.measurement_window_hours)
    actual_window = measurement.measurement_end - measurement.measurement_start

    assert actual_window == expected_window


@pytest.mark.asyncio
async def test_multiple_targets_isolation(baseline_service, performance_samples):
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

    # Measure baselines for both
    baseline1 = await baseline_service.measure_baseline(target1, performance_samples)
    baseline2 = await baseline_service.measure_baseline(target2, performance_samples)

    # Should be different
    assert baseline1.id != baseline2.id

    # Get baselines
    retrieved1 = await baseline_service.get_baseline(target1)
    retrieved2 = await baseline_service.get_baseline(target2)

    assert retrieved1.id == baseline1.id
    assert retrieved2.id == baseline2.id
