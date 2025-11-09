"""
Unit tests for BaselineTracker (ACE-010).

Tests baseline computation, rolling updates, drift detection,
and baseline reset functionality.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from scipy import stats

from agentcore.ace.models.ace_models import PerformanceBaseline, PerformanceMetrics
from agentcore.ace.monitors.baseline_tracker import BaselineTracker


# Test fixtures


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def get_session(mock_session):
    """Mock get_session callable."""
    def _get_session():
        return mock_session
    return _get_session


@pytest.fixture
def baseline_tracker(get_session):
    """Create BaselineTracker instance."""
    return BaselineTracker(get_session)


@pytest.fixture
def sample_metrics():
    """Generate sample performance metrics."""
    def _create_metrics(count: int, agent_id: str = "test-agent", stage: str = "execution"):
        metrics = []
        for i in range(count):
            metric = MagicMock()
            metric.metric_id = uuid4()
            metric.task_id = uuid4()
            metric.agent_id = agent_id
            metric.stage = stage
            # Ensure values stay within valid ranges [0, 1]
            # Use smaller variations that won't exceed bounds
            metric.stage_success_rate = 0.85 + (i * 0.002)  # Slight variation (0.85 to 0.95)
            metric.stage_error_rate = 0.10 + (i * 0.001)    # Slight variation (0.10 to 0.15)
            metric.stage_duration_ms = 2000 + (i * 10)
            metric.stage_action_count = 10 + i
            metric.overall_progress_velocity = 5.0
            metric.error_accumulation_rate = 0.2
            metric.context_staleness_score = 0.1
            metric.intervention_effectiveness = 0.8
            metric.baseline_delta = {}
            metric.recorded_at = datetime.now(UTC)
            metrics.append(metric)
        return metrics
    return _create_metrics


# Test baseline computation


@pytest.mark.asyncio
async def test_compute_baseline_success(baseline_tracker, sample_metrics):
    """Test successful baseline computation with sufficient data."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )

    assert baseline is not None
    assert baseline.agent_id == "test-agent"
    assert baseline.stage == "execution"
    assert baseline.sample_size == 10
    assert 0.0 <= baseline.mean_success_rate <= 1.0
    assert 0.0 <= baseline.mean_error_rate <= 1.0
    assert baseline.mean_duration_ms > 0
    assert baseline.mean_action_count > 0
    assert "success_rate" in baseline.std_dev
    assert "success_rate" in baseline.confidence_interval


@pytest.mark.asyncio
async def test_compute_baseline_insufficient_data(baseline_tracker, sample_metrics):
    """Test baseline computation with insufficient data returns None."""
    metrics = sample_metrics(5)  # Less than required 10

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )

    assert baseline is None


@pytest.mark.asyncio
async def test_compute_baseline_invalid_stage(baseline_tracker):
    """Test baseline computation with invalid stage raises ValueError."""
    with pytest.raises(ValueError, match="Invalid stage"):
        await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="invalid_stage",
        )


@pytest.mark.asyncio
async def test_compute_baseline_caching(baseline_tracker, sample_metrics):
    """Test baseline is cached after first computation."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ) as mock_repo:
        # First call - should fetch from DB
        baseline1 = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )

        # Second call - should use cache
        baseline2 = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )

        # Should only call DB once
        assert mock_repo.call_count == 1
        assert baseline1 == baseline2


@pytest.mark.asyncio
async def test_compute_baseline_with_task_type(baseline_tracker, sample_metrics):
    """Test baseline computation with task_type specified."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
            task_type="data_analysis",
        )

    assert baseline is not None
    assert baseline.task_type == "data_analysis"


@pytest.mark.asyncio
async def test_compute_baseline_single_sample(baseline_tracker):
    """Test baseline computation with single sample (edge case)."""
    # Single metric with no variance
    metric = MagicMock()
    metric.agent_id = "test-agent"
    metric.stage = "execution"
    metric.stage_success_rate = 0.9
    metric.stage_error_rate = 0.1
    metric.stage_duration_ms = 2000
    metric.stage_action_count = 10
    metrics = [metric]

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        # Should return None (insufficient samples)
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )

    assert baseline is None


# Test rolling baseline updates


@pytest.mark.asyncio
async def test_update_baseline_not_needed(baseline_tracker):
    """Test baseline update skipped when execution count below threshold."""
    baseline = await baseline_tracker.update_baseline(
        agent_id="test-agent",
        stage="execution",
    )

    # Should return None (counter at 1, threshold is 50)
    assert baseline is None


@pytest.mark.asyncio
async def test_update_baseline_force(baseline_tracker, sample_metrics):
    """Test forced baseline update regardless of execution count."""
    metrics = sample_metrics(50)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.update_baseline(
            agent_id="test-agent",
            stage="execution",
            force=True,
        )

    assert baseline is not None
    assert baseline.sample_size == 50


@pytest.mark.asyncio
async def test_update_baseline_after_threshold(baseline_tracker, sample_metrics):
    """Test baseline update after reaching execution threshold."""
    metrics = sample_metrics(50)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        # Simulate 50 executions
        for _ in range(49):
            result = await baseline_tracker.update_baseline(
                agent_id="test-agent",
                stage="execution",
            )
            assert result is None  # Not yet at threshold

        # 50th execution - should trigger update
        baseline = await baseline_tracker.update_baseline(
            agent_id="test-agent",
            stage="execution",
        )

    assert baseline is not None
    assert baseline.sample_size == 50


@pytest.mark.asyncio
async def test_update_baseline_insufficient_data(baseline_tracker, sample_metrics):
    """Test baseline update with insufficient data returns None."""
    metrics = sample_metrics(5)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.update_baseline(
            agent_id="test-agent",
            stage="execution",
            force=True,
        )

    assert baseline is None


@pytest.mark.asyncio
async def test_update_baseline_invalid_stage(baseline_tracker):
    """Test baseline update with invalid stage raises ValueError."""
    with pytest.raises(ValueError, match="Invalid stage"):
        await baseline_tracker.update_baseline(
            agent_id="test-agent",
            stage="invalid_stage",
            force=True,
        )


# Test drift detection


@pytest.mark.asyncio
async def test_detect_drift_no_drift(baseline_tracker):
    """Test drift detection when metrics are within confidence intervals."""
    baseline = PerformanceBaseline(
        agent_id="test-agent",
        stage="execution",
        mean_success_rate=0.9,
        mean_error_rate=0.1,
        mean_duration_ms=2000.0,
        mean_action_count=10.0,
        std_dev={
            "success_rate": 0.05,
            "error_rate": 0.05,
            "duration_ms": 200.0,
            "action_count": 2.0,
        },
        confidence_interval={
            "success_rate": (0.85, 0.95),
            "error_rate": (0.05, 0.15),
            "duration_ms": (1800.0, 2200.0),
            "action_count": (8.0, 12.0),
        },
        sample_size=50,
    )

    current_metrics = PerformanceMetrics(
        task_id=uuid4(),
        agent_id="test-agent",
        stage="execution",
        stage_success_rate=0.91,  # Within CI
        stage_error_rate=0.09,    # Within CI
        stage_duration_ms=2050,   # Within CI
        stage_action_count=11,    # Within CI
        overall_progress_velocity=5.0,
        error_accumulation_rate=0.2,
        context_staleness_score=0.1,
    )

    drift_detected, drift_details = await baseline_tracker.detect_drift(
        current_metrics, baseline
    )

    assert drift_detected is False
    assert drift_details["drift_detected"] is False
    assert len(drift_details["significant_metrics"]) == 0


@pytest.mark.asyncio
async def test_detect_drift_success_rate_degradation(baseline_tracker):
    """Test drift detection when success rate drops below baseline."""
    baseline = PerformanceBaseline(
        agent_id="test-agent",
        stage="execution",
        mean_success_rate=0.9,
        mean_error_rate=0.1,
        mean_duration_ms=2000.0,
        mean_action_count=10.0,
        std_dev={
            "success_rate": 0.05,
            "error_rate": 0.05,
            "duration_ms": 200.0,
            "action_count": 2.0,
        },
        confidence_interval={
            "success_rate": (0.85, 0.95),
            "error_rate": (0.05, 0.15),
            "duration_ms": (1800.0, 2200.0),
            "action_count": (8.0, 12.0),
        },
        sample_size=50,
    )

    current_metrics = PerformanceMetrics(
        task_id=uuid4(),
        agent_id="test-agent",
        stage="execution",
        stage_success_rate=0.70,  # Below CI lower bound
        stage_error_rate=0.30,    # Above CI upper bound
        stage_duration_ms=2000,
        stage_action_count=10,
        overall_progress_velocity=5.0,
        error_accumulation_rate=0.5,
        context_staleness_score=0.3,
    )

    drift_detected, drift_details = await baseline_tracker.detect_drift(
        current_metrics, baseline
    )

    assert drift_detected is True
    assert drift_details["drift_detected"] is True
    assert "success_rate" in drift_details["significant_metrics"]
    assert "error_rate" in drift_details["significant_metrics"]
    assert drift_details["deviations"]["success_rate"] < 0  # Negative deviation


@pytest.mark.asyncio
async def test_detect_drift_duration_increase(baseline_tracker):
    """Test drift detection when duration significantly increases."""
    baseline = PerformanceBaseline(
        agent_id="test-agent",
        stage="execution",
        mean_success_rate=0.9,
        mean_error_rate=0.1,
        mean_duration_ms=2000.0,
        mean_action_count=10.0,
        std_dev={
            "success_rate": 0.05,
            "error_rate": 0.05,
            "duration_ms": 200.0,
            "action_count": 2.0,
        },
        confidence_interval={
            "success_rate": (0.85, 0.95),
            "error_rate": (0.05, 0.15),
            "duration_ms": (1800.0, 2200.0),
            "action_count": (8.0, 12.0),
        },
        sample_size=50,
    )

    current_metrics = PerformanceMetrics(
        task_id=uuid4(),
        agent_id="test-agent",
        stage="execution",
        stage_success_rate=0.9,
        stage_error_rate=0.1,
        stage_duration_ms=3000,  # Above CI upper bound
        stage_action_count=10,
        overall_progress_velocity=3.0,
        error_accumulation_rate=0.2,
        context_staleness_score=0.1,
    )

    drift_detected, drift_details = await baseline_tracker.detect_drift(
        current_metrics, baseline
    )

    assert drift_detected is True
    assert "duration_ms" in drift_details["significant_metrics"]
    assert drift_details["deviations"]["duration_ms"] > 0  # Positive deviation


@pytest.mark.asyncio
async def test_detect_drift_agent_mismatch(baseline_tracker):
    """Test drift detection raises ValueError for mismatched agents."""
    baseline = PerformanceBaseline(
        agent_id="test-agent",
        stage="execution",
        mean_success_rate=0.9,
        mean_error_rate=0.1,
        mean_duration_ms=2000.0,
        mean_action_count=10.0,
        std_dev={},
        confidence_interval={},
        sample_size=50,
    )

    current_metrics = PerformanceMetrics(
        task_id=uuid4(),
        agent_id="different-agent",  # Mismatch
        stage="execution",
        stage_success_rate=0.9,
        stage_error_rate=0.1,
        stage_duration_ms=2000,
        stage_action_count=10,
        overall_progress_velocity=5.0,
        error_accumulation_rate=0.2,
        context_staleness_score=0.1,
    )

    with pytest.raises(ValueError, match="Agent ID mismatch"):
        await baseline_tracker.detect_drift(current_metrics, baseline)


@pytest.mark.asyncio
async def test_detect_drift_stage_mismatch(baseline_tracker):
    """Test drift detection raises ValueError for mismatched stages."""
    baseline = PerformanceBaseline(
        agent_id="test-agent",
        stage="execution",
        mean_success_rate=0.9,
        mean_error_rate=0.1,
        mean_duration_ms=2000.0,
        mean_action_count=10.0,
        std_dev={},
        confidence_interval={},
        sample_size=50,
    )

    current_metrics = PerformanceMetrics(
        task_id=uuid4(),
        agent_id="test-agent",
        stage="planning",  # Mismatch
        stage_success_rate=0.9,
        stage_error_rate=0.1,
        stage_duration_ms=2000,
        stage_action_count=10,
        overall_progress_velocity=5.0,
        error_accumulation_rate=0.2,
        context_staleness_score=0.1,
    )

    with pytest.raises(ValueError, match="Stage mismatch"):
        await baseline_tracker.detect_drift(current_metrics, baseline)


@pytest.mark.asyncio
async def test_detect_drift_p_values_computed(baseline_tracker):
    """Test that p-values are computed for significant drift."""
    baseline = PerformanceBaseline(
        agent_id="test-agent",
        stage="execution",
        mean_success_rate=0.9,
        mean_error_rate=0.1,
        mean_duration_ms=2000.0,
        mean_action_count=10.0,
        std_dev={
            "success_rate": 0.05,
            "error_rate": 0.05,
            "duration_ms": 200.0,
            "action_count": 2.0,
        },
        confidence_interval={
            "success_rate": (0.85, 0.95),
            "error_rate": (0.05, 0.15),
            "duration_ms": (1800.0, 2200.0),
            "action_count": (8.0, 12.0),
        },
        sample_size=50,
    )

    current_metrics = PerformanceMetrics(
        task_id=uuid4(),
        agent_id="test-agent",
        stage="execution",
        stage_success_rate=0.70,  # Significant drift
        stage_error_rate=0.1,
        stage_duration_ms=2000,
        stage_action_count=10,
        overall_progress_velocity=5.0,
        error_accumulation_rate=0.5,
        context_staleness_score=0.3,
    )

    drift_detected, drift_details = await baseline_tracker.detect_drift(
        current_metrics, baseline
    )

    assert "p_values" in drift_details
    assert "success_rate" in drift_details["p_values"]
    # P-value should be small for significant drift
    assert drift_details["p_values"]["success_rate"] < 0.05


# Test baseline reset


@pytest.mark.asyncio
async def test_reset_baseline(baseline_tracker, sample_metrics):
    """Test baseline reset clears cache and counters."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        # Compute baseline (adds to cache)
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )
        assert baseline is not None

        # Reset baseline
        await baseline_tracker.reset_baseline(
            agent_id="test-agent",
            stage="execution",
        )

        # Cache should be empty
        cache_key = ("test-agent", "execution", None)
        assert cache_key not in baseline_tracker._baseline_cache
        assert cache_key not in baseline_tracker._execution_counters


@pytest.mark.asyncio
async def test_reset_baseline_invalid_stage(baseline_tracker):
    """Test baseline reset with invalid stage raises ValueError."""
    with pytest.raises(ValueError, match="Invalid stage"):
        await baseline_tracker.reset_baseline(
            agent_id="test-agent",
            stage="invalid_stage",
        )


@pytest.mark.asyncio
async def test_reset_baseline_with_task_type(baseline_tracker, sample_metrics):
    """Test baseline reset with task_type specified."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        # Compute baseline with task_type
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
            task_type="data_analysis",
        )
        assert baseline is not None

        # Reset baseline with task_type
        await baseline_tracker.reset_baseline(
            agent_id="test-agent",
            stage="execution",
            task_type="data_analysis",
        )

        # Cache should be empty for this specific key
        cache_key = ("test-agent", "execution", "data_analysis")
        assert cache_key not in baseline_tracker._baseline_cache


# Test get_baseline


@pytest.mark.asyncio
async def test_get_baseline_cached(baseline_tracker, sample_metrics):
    """Test get_baseline returns cached baseline if available."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ) as mock_repo:
        # First call - computes and caches
        baseline1 = await baseline_tracker.get_baseline(
            agent_id="test-agent",
            stage="execution",
        )

        # Second call - returns cached
        baseline2 = await baseline_tracker.get_baseline(
            agent_id="test-agent",
            stage="execution",
        )

        # Should only call DB once
        assert mock_repo.call_count == 1
        assert baseline1 == baseline2


@pytest.mark.asyncio
async def test_get_baseline_computes_if_not_cached(baseline_tracker, sample_metrics):
    """Test get_baseline computes new baseline if not cached."""
    metrics = sample_metrics(10)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.get_baseline(
            agent_id="test-agent",
            stage="execution",
        )

    assert baseline is not None
    assert baseline.agent_id == "test-agent"
    assert baseline.stage == "execution"


# Edge cases


@pytest.mark.asyncio
async def test_baseline_with_zero_variance(baseline_tracker):
    """Test baseline computation when all metrics are identical (zero variance)."""
    # Create 10 identical metrics
    metrics = []
    for i in range(10):
        metric = MagicMock()
        metric.agent_id = "test-agent"
        metric.stage = "execution"
        metric.stage_success_rate = 0.9  # All identical
        metric.stage_error_rate = 0.1
        metric.stage_duration_ms = 2000
        metric.stage_action_count = 10
        metrics.append(metric)

    with patch(
        "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
        new_callable=AsyncMock,
        return_value=metrics,
    ):
        baseline = await baseline_tracker.compute_baseline(
            agent_id="test-agent",
            stage="execution",
        )

    assert baseline is not None
    # Std dev should be 0
    assert baseline.std_dev["success_rate"] == 0.0
    # Confidence interval should be tight
    assert baseline.confidence_interval["success_rate"][0] == \
           baseline.confidence_interval["success_rate"][1]


@pytest.mark.asyncio
async def test_baseline_all_stages(baseline_tracker, sample_metrics):
    """Test baseline computation works for all valid stages."""
    for stage in ["planning", "execution", "reflection", "verification"]:
        metrics = sample_metrics(10, stage=stage)

        with patch(
            "agentcore.ace.monitors.baseline_tracker.MetricsRepository.list_by_agent_stage",
            new_callable=AsyncMock,
            return_value=metrics,
        ):
            baseline = await baseline_tracker.compute_baseline(
                agent_id="test-agent",
                stage=stage,
            )

        assert baseline is not None
        assert baseline.stage == stage
