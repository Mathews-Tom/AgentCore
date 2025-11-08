"""
Unit tests for PerformanceMonitor (COMPASS ACE-1)

Tests cover:
- Metrics recording with stage validation
- Batching logic (buffer size and timeout)
- Database interaction
- Latency requirements
- Edge cases and error handling

Target: 95%+ code coverage
"""

import asyncio
from datetime import UTC, datetime
from time import perf_counter
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agentcore.ace.models.ace_models import PerformanceMetrics
from agentcore.ace.monitors.performance_monitor import PerformanceMonitor


@pytest.fixture
def mock_session():
    """Mock AsyncSession for testing."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_get_session(mock_session):
    """Mock get_session context manager."""
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_session
    async_cm.__aexit__.return_value = None
    return MagicMock(return_value=async_cm)


@pytest.fixture
def performance_monitor(mock_get_session):
    """PerformanceMonitor instance for testing."""
    return PerformanceMonitor(
        get_session=mock_get_session,
        batch_size=100,
        batch_timeout=1.0,
    )


@pytest.fixture
def sample_metrics():
    """Sample PerformanceMetrics for testing."""
    return PerformanceMetrics(
        task_id=uuid4(),
        agent_id="test-agent",
        stage="execution",
        stage_success_rate=0.85,
        stage_error_rate=0.15,
        stage_duration_ms=2500,
        stage_action_count=12,
        overall_progress_velocity=4.8,
        error_accumulation_rate=0.3,
        context_staleness_score=0.2,
        intervention_effectiveness=0.75,
        baseline_delta={"stage_success_rate": -0.05},
    )


class TestPerformanceMonitorInit:
    """Test PerformanceMonitor initialization."""

    def test_init_valid_params(self, mock_get_session):
        """Test initialization with valid parameters."""
        monitor = PerformanceMonitor(
            get_session=mock_get_session,
            batch_size=50,
            batch_timeout=2.0,
        )

        assert monitor.batch_size == 50
        assert monitor.batch_timeout == 2.0
        assert monitor._buffer == []
        assert monitor._flush_task is None

    def test_init_invalid_batch_size(self, mock_get_session):
        """Test initialization with invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            PerformanceMonitor(
                get_session=mock_get_session,
                batch_size=0,
                batch_timeout=1.0,
            )

    def test_init_invalid_batch_timeout(self, mock_get_session):
        """Test initialization with invalid batch_timeout."""
        with pytest.raises(ValueError, match="batch_timeout must be > 0"):
            PerformanceMonitor(
                get_session=mock_get_session,
                batch_size=100,
                batch_timeout=0.0,
            )


class TestRecordMetrics:
    """Test record_metrics method."""

    @pytest.mark.asyncio
    async def test_record_metrics_valid_stage(
        self, performance_monitor, sample_metrics
    ):
        """Test recording metrics with valid stage."""
        task_id = sample_metrics.task_id
        agent_id = sample_metrics.agent_id

        await performance_monitor.record_metrics(
            task_id=task_id,
            agent_id=agent_id,
            stage="execution",
            metrics=sample_metrics,
        )

        # Check buffer has metric
        assert len(performance_monitor._buffer) == 1
        assert performance_monitor._buffer[0]["task_id"] == task_id
        assert performance_monitor._buffer[0]["agent_id"] == agent_id
        assert performance_monitor._buffer[0]["stage"] == "execution"

    @pytest.mark.asyncio
    async def test_record_metrics_invalid_stage(
        self, performance_monitor, sample_metrics
    ):
        """Test recording metrics with invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stage"):
            await performance_monitor.record_metrics(
                task_id=sample_metrics.task_id,
                agent_id=sample_metrics.agent_id,
                stage="invalid_stage",
                metrics=sample_metrics,
            )

    @pytest.mark.asyncio
    async def test_record_metrics_all_valid_stages(
        self, performance_monitor, sample_metrics
    ):
        """Test recording metrics with all valid stages."""
        valid_stages = ["planning", "execution", "reflection", "verification"]

        for stage in valid_stages:
            await performance_monitor.record_metrics(
                task_id=sample_metrics.task_id,
                agent_id=sample_metrics.agent_id,
                stage=stage,
                metrics=sample_metrics,
            )

        # Check buffer has all metrics
        assert len(performance_monitor._buffer) == len(valid_stages)

    @pytest.mark.asyncio
    async def test_record_metrics_latency(
        self, performance_monitor, sample_metrics
    ):
        """Test record_metrics latency is <50ms (p95 target)."""
        latencies = []

        for _ in range(100):
            start = perf_counter()
            await performance_monitor.record_metrics(
                task_id=sample_metrics.task_id,
                agent_id=sample_metrics.agent_id,
                stage="execution",
                metrics=sample_metrics,
            )
            end = perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Check p95 latency
        latencies.sort()
        p95_latency = latencies[94]  # 95th percentile (0-indexed)

        assert p95_latency < 50.0, f"p95 latency {p95_latency:.2f}ms exceeds 50ms target"


class TestBatching:
    """Test metrics batching logic."""

    @pytest.mark.asyncio
    async def test_batch_flush_on_size(
        self, performance_monitor, sample_metrics, mock_get_session
    ):
        """Test buffer flushes when batch_size is reached."""
        # Set batch size to 5 for testing
        performance_monitor.batch_size = 5

        with patch(
            "agentcore.ace.monitors.performance_monitor.MetricsRepository.bulk_create",
            new_callable=AsyncMock,
        ) as mock_bulk_create:
            mock_bulk_create.return_value = 5

            # Add 5 metrics (should trigger flush)
            for i in range(5):
                await performance_monitor.record_metrics(
                    task_id=sample_metrics.task_id,
                    agent_id=f"agent-{i}",
                    stage="execution",
                    metrics=sample_metrics,
                )

            # Wait for flush to complete
            await asyncio.sleep(0.1)

            # Check bulk_create was called
            mock_bulk_create.assert_called_once()
            assert len(mock_bulk_create.call_args[0][1]) == 5

            # Check buffer is empty
            assert len(performance_monitor._buffer) == 0

    @pytest.mark.asyncio
    async def test_batch_flush_on_timeout(
        self, mock_get_session, sample_metrics
    ):
        """Test buffer flushes after timeout."""
        # Use short timeout for testing
        monitor = PerformanceMonitor(
            get_session=mock_get_session,
            batch_size=100,
            batch_timeout=0.2,
        )

        with patch(
            "agentcore.ace.monitors.performance_monitor.MetricsRepository.bulk_create",
            new_callable=AsyncMock,
        ) as mock_bulk_create:
            mock_bulk_create.return_value = 1

            # Add 1 metric (won't trigger size-based flush)
            await monitor.record_metrics(
                task_id=sample_metrics.task_id,
                agent_id=sample_metrics.agent_id,
                stage="execution",
                metrics=sample_metrics,
            )

            # Wait for timeout flush
            await asyncio.sleep(0.3)

            # Check bulk_create was called
            assert mock_bulk_create.called
            assert len(monitor._buffer) == 0

    @pytest.mark.asyncio
    async def test_concurrent_metric_updates(
        self, performance_monitor, sample_metrics
    ):
        """Test concurrent metric updates are handled correctly."""
        tasks = []

        for i in range(50):
            task = asyncio.create_task(
                performance_monitor.record_metrics(
                    task_id=sample_metrics.task_id,
                    agent_id=f"agent-{i}",
                    stage="execution",
                    metrics=sample_metrics,
                )
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Check all metrics were buffered
        assert len(performance_monitor._buffer) == 50


class TestGetCurrentMetrics:
    """Test get_current_metrics method."""

    @pytest.mark.asyncio
    async def test_get_current_metrics_found(
        self, performance_monitor, sample_metrics, mock_session
    ):
        """Test get_current_metrics when metrics exist."""
        from agentcore.ace.database.ace_orm import PerformanceMetricsDB

        # Create mock DB metric
        db_metric = PerformanceMetricsDB(
            metric_id=uuid4(),
            task_id=sample_metrics.task_id,
            agent_id=sample_metrics.agent_id,
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.8,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
            intervention_effectiveness=0.75,
            baseline_delta={},
            recorded_at=datetime.now(UTC),
        )

        with patch(
            "agentcore.ace.monitors.performance_monitor.MetricsRepository.get_latest_by_task",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = db_metric

            result = await performance_monitor.get_current_metrics(
                task_id=sample_metrics.task_id,
                agent_id=sample_metrics.agent_id,
            )

            assert result is not None
            assert result.task_id == sample_metrics.task_id
            assert result.agent_id == sample_metrics.agent_id
            assert result.stage == "execution"

    @pytest.mark.asyncio
    async def test_get_current_metrics_not_found(
        self, performance_monitor, sample_metrics
    ):
        """Test get_current_metrics when no metrics exist."""
        with patch(
            "agentcore.ace.monitors.performance_monitor.MetricsRepository.get_latest_by_task",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await performance_monitor.get_current_metrics(
                task_id=sample_metrics.task_id,
                agent_id=sample_metrics.agent_id,
            )

            assert result is None


class TestBaselineMethods:
    """Test baseline-related methods (stubs for now)."""

    @pytest.mark.asyncio
    async def test_get_baseline_returns_none(self, performance_monitor):
        """Test get_baseline returns None (stub implementation)."""
        result = await performance_monitor.get_baseline(
            agent_id="test-agent",
            task_type="data_analysis",
            stage="execution",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_compute_baseline_delta_no_baseline(
        self, performance_monitor, sample_metrics
    ):
        """Test compute_baseline_delta with no baseline returns empty dict."""
        result = await performance_monitor.compute_baseline_delta(
            current_metrics=sample_metrics,
            baseline=None,
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_update_baseline_returns_none(
        self, performance_monitor, sample_metrics
    ):
        """Test update_baseline returns None (stub implementation)."""
        result = await performance_monitor.update_baseline(
            agent_id="test-agent",
            stage="execution",
            metrics_history=[sample_metrics],
        )

        assert result is None


class TestFlushAndShutdown:
    """Test flush and shutdown functionality."""

    @pytest.mark.asyncio
    async def test_flush_and_shutdown(
        self, performance_monitor, sample_metrics
    ):
        """Test flush_and_shutdown flushes remaining metrics."""
        # Add some metrics
        for i in range(10):
            await performance_monitor.record_metrics(
                task_id=sample_metrics.task_id,
                agent_id=f"agent-{i}",
                stage="execution",
                metrics=sample_metrics,
            )

        with patch(
            "agentcore.ace.monitors.performance_monitor.MetricsRepository.bulk_create",
            new_callable=AsyncMock,
        ) as mock_bulk_create:
            mock_bulk_create.return_value = 10

            await performance_monitor.flush_and_shutdown()

            # Check bulk_create was called
            mock_bulk_create.assert_called_once()
            assert len(mock_bulk_create.call_args[0][1]) == 10

            # Check buffer is empty
            assert len(performance_monitor._buffer) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_buffer_flush(self, performance_monitor):
        """Test flushing empty buffer does nothing."""
        await performance_monitor._flush_buffer()
        # Should not raise error

    @pytest.mark.asyncio
    async def test_flush_error_recovery(
        self, performance_monitor, sample_metrics
    ):
        """Test buffer retains metrics on flush error."""
        # Add metric
        await performance_monitor.record_metrics(
            task_id=sample_metrics.task_id,
            agent_id=sample_metrics.agent_id,
            stage="execution",
            metrics=sample_metrics,
        )

        with patch(
            "agentcore.ace.monitors.performance_monitor.MetricsRepository.bulk_create",
            new_callable=AsyncMock,
        ) as mock_bulk_create:
            mock_bulk_create.side_effect = Exception("DB error")

            with pytest.raises(Exception, match="DB error"):
                await performance_monitor._flush_buffer()

            # Check buffer still has metric for retry
            assert len(performance_monitor._buffer) == 1

    @pytest.mark.asyncio
    async def test_intervention_effectiveness_none(
        self, performance_monitor
    ):
        """Test metrics with intervention_effectiveness=None."""
        metrics = PerformanceMetrics(
            task_id=uuid4(),
            agent_id="test-agent",
            stage="planning",
            stage_success_rate=0.9,
            stage_error_rate=0.1,
            stage_duration_ms=1000,
            stage_action_count=5,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.2,
            context_staleness_score=0.1,
            intervention_effectiveness=None,  # Can be None
            baseline_delta={},
        )

        await performance_monitor.record_metrics(
            task_id=metrics.task_id,
            agent_id=metrics.agent_id,
            stage="planning",
            metrics=metrics,
        )

        assert len(performance_monitor._buffer) == 1
        assert performance_monitor._buffer[0]["intervention_effectiveness"] is None

    @pytest.mark.asyncio
    async def test_baseline_delta_empty(
        self, performance_monitor
    ):
        """Test metrics with empty baseline_delta."""
        metrics = PerformanceMetrics(
            task_id=uuid4(),
            agent_id="test-agent",
            stage="reflection",
            stage_success_rate=0.95,
            stage_error_rate=0.05,
            stage_duration_ms=500,
            stage_action_count=8,
            overall_progress_velocity=5.0,
            error_accumulation_rate=0.1,
            context_staleness_score=0.05,
            intervention_effectiveness=0.8,
            baseline_delta={},  # Empty is valid
        )

        await performance_monitor.record_metrics(
            task_id=metrics.task_id,
            agent_id=metrics.agent_id,
            stage="reflection",
            metrics=metrics,
        )

        assert len(performance_monitor._buffer) == 1
        assert performance_monitor._buffer[0]["baseline_delta"] == {}
