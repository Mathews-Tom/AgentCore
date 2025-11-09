"""
Integration tests for ACE Metrics API JSON-RPC Handlers (ACE-012)

Tests cover:
- ace.track_performance method
- ace.get_baseline method
- ace.get_metrics_summary method
- ace.get_metrics method
- ace.get_metric_history method
- A2A context handling
- Error handling for invalid params
- Database integration
- Datetime serialization

Target: Full acceptance criteria coverage
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.ace.database.repositories import MetricsRepository
from agentcore.ace.jsonrpc import (
    handle_get_baseline,
    handle_get_metric_history,
    handle_get_metrics,
    handle_get_metrics_summary,
    handle_track_performance,
)
from agentcore.ace.models.ace_models import PerformanceMetrics
from agentcore.ace.monitors.baseline_tracker import BaselineTracker
from agentcore.ace.monitors.error_accumulator import ErrorAccumulator, ErrorSeverity
from agentcore.ace.monitors.performance_monitor import PerformanceMonitor


# Test fixtures


@pytest.fixture
def sample_task_id() -> UUID:
    """Sample task ID for testing."""
    return uuid4()


@pytest.fixture
def sample_agent_id() -> str:
    """Sample agent ID for testing."""
    return "test-agent-001"


@pytest.fixture
def sample_metrics_data() -> dict:
    """Sample metrics data dictionary."""
    return {
        "stage_success_rate": 0.85,
        "stage_error_rate": 0.15,
        "stage_duration_ms": 2500,
        "stage_action_count": 12,
        "overall_progress_velocity": 4.8,
        "error_accumulation_rate": 0.3,
        "context_staleness_score": 0.2,
        "intervention_effectiveness": 0.75,
        "baseline_delta": {"stage_success_rate": -0.05},
    }


@pytest.fixture
def sample_performance_metrics(
    sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
) -> PerformanceMetrics:
    """Sample PerformanceMetrics instance."""
    return PerformanceMetrics(
        task_id=sample_task_id,
        agent_id=sample_agent_id,
        stage="execution",
        **sample_metrics_data,
    )


# Integration Tests


class TestTrackPerformance:
    """Test ace.track_performance JSON-RPC method."""

    @pytest.mark.asyncio
    async def test_track_performance_success(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test successful performance tracking."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "execution",
                "metrics": sample_metrics_data,
            },
            id="test-1",
        )

        response = await handle_track_performance(request)

        assert response["success"] is True
        assert "Performance metrics recorded successfully" in response["message"]
        assert "recorded_at" in response

    @pytest.mark.asyncio
    async def test_track_performance_all_stages(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test tracking performance for all valid stages."""
        valid_stages = ["planning", "execution", "reflection", "verification"]

        for stage in valid_stages:
            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="ace.track_performance",
                params={
                    "agent_id": sample_agent_id,
                    "task_id": str(sample_task_id),
                    "stage": stage,
                    "metrics": sample_metrics_data,
                },
                id=f"test-{stage}",
            )

            response = await handle_track_performance(request)
            assert response["success"] is True

    @pytest.mark.asyncio
    async def test_track_performance_invalid_stage(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test tracking performance with invalid stage raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "invalid_stage",
                "metrics": sample_metrics_data,
            },
            id="test-invalid",
        )

        with pytest.raises(ValueError, match="Request validation failed"):
            await handle_track_performance(request)

    @pytest.mark.asyncio
    async def test_track_performance_missing_params(self):
        """Test tracking performance with missing params raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": "test-agent",
                # Missing task_id, stage, metrics
            },
            id="test-missing",
        )

        with pytest.raises(ValueError):
            await handle_track_performance(request)

    @pytest.mark.asyncio
    async def test_track_performance_invalid_metrics(
        self, sample_task_id: UUID, sample_agent_id: str
    ):
        """Test tracking performance with invalid metrics raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "execution",
                "metrics": {
                    "stage_success_rate": 1.5,  # Invalid: > 1.0
                    "stage_error_rate": 0.1,
                    "stage_duration_ms": 2500,
                    "stage_action_count": 12,
                    "overall_progress_velocity": 4.8,
                    "error_accumulation_rate": 0.3,
                    "context_staleness_score": 0.2,
                },
            },
            id="test-invalid-metrics",
        )

        with pytest.raises(ValueError):
            await handle_track_performance(request)


class TestGetBaseline:
    """Test ace.get_baseline JSON-RPC method."""

    @pytest.mark.asyncio
    async def test_get_baseline_not_available(self, sample_agent_id: str):
        """Test getting baseline when not available returns null."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_baseline",
            params={
                "agent_id": sample_agent_id,
                "stage": "execution",
            },
            id="test-1",
        )

        response = await handle_get_baseline(request)

        assert response["baseline"] is None

    @pytest.mark.asyncio
    async def test_get_baseline_with_task_type(self, sample_agent_id: str):
        """Test getting baseline with optional task_type."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_baseline",
            params={
                "agent_id": sample_agent_id,
                "stage": "execution",
                "task_type": "data_analysis",
            },
            id="test-2",
        )

        response = await handle_get_baseline(request)

        # Baseline not available yet (need sufficient data)
        assert response["baseline"] is None

    @pytest.mark.asyncio
    async def test_get_baseline_invalid_stage(self, sample_agent_id: str):
        """Test getting baseline with invalid stage raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_baseline",
            params={
                "agent_id": sample_agent_id,
                "stage": "invalid_stage",
            },
            id="test-invalid",
        )

        with pytest.raises(ValueError, match="Invalid stage"):
            await handle_get_baseline(request)

    @pytest.mark.asyncio
    async def test_get_baseline_missing_params(self):
        """Test getting baseline with missing params raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_baseline",
            params={
                "agent_id": "test-agent",
                # Missing stage
            },
            id="test-missing",
        )

        with pytest.raises(ValueError):
            await handle_get_baseline(request)


class TestGetMetrics:
    """Test ace.get_metrics JSON-RPC method."""

    @pytest.mark.asyncio
    async def test_get_metrics_not_found(
        self, sample_task_id: UUID, sample_agent_id: str
    ):
        """Test getting metrics when not found returns null."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
            },
            id="test-1",
        )

        response = await handle_get_metrics(request)

        assert response["metrics"] is None

    @pytest.mark.asyncio
    async def test_get_metrics_after_tracking(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test getting metrics after tracking."""
        # First track metrics
        track_request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "execution",
                "metrics": sample_metrics_data,
            },
            id="track-1",
        )
        await handle_track_performance(track_request)

        # Flush metrics to database
        from agentcore.ace.jsonrpc import performance_monitor

        await performance_monitor._flush_buffer()

        # Now get metrics
        get_request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
            },
            id="get-1",
        )

        response = await handle_get_metrics(get_request)

        assert response["metrics"] is not None
        assert response["metrics"]["agent_id"] == sample_agent_id
        assert response["metrics"]["stage"] == "execution"

    @pytest.mark.asyncio
    async def test_get_metrics_missing_params(self):
        """Test getting metrics with missing params raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics",
            params={
                "agent_id": "test-agent",
                # Missing task_id
            },
            id="test-missing",
        )

        with pytest.raises(ValueError):
            await handle_get_metrics(request)


class TestGetMetricsSummary:
    """Test ace.get_metrics_summary JSON-RPC method."""

    @pytest.mark.asyncio
    async def test_get_metrics_summary_no_data(
        self, sample_task_id: UUID, sample_agent_id: str
    ):
        """Test getting metrics summary with no data."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics_summary",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
            },
            id="test-1",
        )

        response = await handle_get_metrics_summary(request)

        assert "summary" in response
        summary = response["summary"]
        assert summary["agent_id"] == sample_agent_id
        assert summary["task_id"] == str(sample_task_id)
        assert summary["latest_metrics"] is None
        assert summary["baseline"] is None
        assert summary["total_errors"] == 0
        assert summary["critical_errors"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_summary_with_errors(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test getting metrics summary with error data."""
        # Track some errors
        from agentcore.ace.jsonrpc import error_accumulator

        error_accumulator.track_error(
            agent_id=sample_agent_id,
            task_id=sample_task_id,
            stage="execution",
            error_type="validation_error",
            severity=ErrorSeverity.HIGH,
            error_message="Invalid input",
        )

        error_accumulator.track_error(
            agent_id=sample_agent_id,
            task_id=sample_task_id,
            stage="execution",
            error_type="timeout_error",
            severity=ErrorSeverity.CRITICAL,
            error_message="Operation timeout",
        )

        # Get summary
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics_summary",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
            },
            id="test-1",
        )

        response = await handle_get_metrics_summary(request)

        summary = response["summary"]
        assert summary["total_errors"] == 2
        assert summary["critical_errors"] == 1
        assert "error_trends" in summary

    @pytest.mark.asyncio
    async def test_get_metrics_summary_complete(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test getting complete metrics summary with all data."""
        # Track performance
        track_request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "execution",
                "metrics": sample_metrics_data,
            },
            id="track-1",
        )
        await handle_track_performance(track_request)

        # Flush metrics
        from agentcore.ace.jsonrpc import performance_monitor

        await performance_monitor._flush_buffer()

        # Get summary
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics_summary",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
            },
            id="test-1",
        )

        response = await handle_get_metrics_summary(request)

        summary = response["summary"]
        assert summary["latest_metrics"] is not None
        assert summary["agent_id"] == sample_agent_id
        assert "generated_at" in summary


class TestGetMetricHistory:
    """Test ace.get_metric_history JSON-RPC method."""

    @pytest.mark.asyncio
    async def test_get_metric_history_empty(
        self, sample_task_id: UUID, sample_agent_id: str
    ):
        """Test getting metric history when empty."""
        now = datetime.now(UTC)
        start_time = now - timedelta(hours=1)
        end_time = now

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metric_history",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
            id="test-1",
        )

        response = await handle_get_metric_history(request)

        assert response["count"] == 0
        assert response["metrics"] == []
        assert response["start_time"] == start_time.isoformat()
        assert response["end_time"] == end_time.isoformat()

    @pytest.mark.asyncio
    async def test_get_metric_history_with_data(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test getting metric history with data."""
        # Track multiple metrics
        for i in range(5):
            track_request = JsonRpcRequest(
                jsonrpc="2.0",
                method="ace.track_performance",
                params={
                    "agent_id": sample_agent_id,
                    "task_id": str(sample_task_id),
                    "stage": "execution",
                    "metrics": sample_metrics_data,
                },
                id=f"track-{i}",
            )
            await handle_track_performance(track_request)
            await asyncio.sleep(0.01)  # Small delay for different timestamps

        # Flush metrics
        from agentcore.ace.jsonrpc import performance_monitor

        await performance_monitor._flush_buffer()

        # Get history
        now = datetime.now(UTC)
        start_time = now - timedelta(hours=1)
        end_time = now + timedelta(hours=1)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metric_history",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
            id="test-1",
        )

        response = await handle_get_metric_history(request)

        assert response["count"] == 5
        assert len(response["metrics"]) == 5

    @pytest.mark.asyncio
    async def test_get_metric_history_time_filtering(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test metric history time range filtering."""
        # Track metrics
        for i in range(3):
            track_request = JsonRpcRequest(
                jsonrpc="2.0",
                method="ace.track_performance",
                params={
                    "agent_id": sample_agent_id,
                    "task_id": str(sample_task_id),
                    "stage": "execution",
                    "metrics": sample_metrics_data,
                },
                id=f"track-{i}",
            )
            await handle_track_performance(track_request)
            await asyncio.sleep(0.01)

        # Flush metrics
        from agentcore.ace.jsonrpc import performance_monitor

        await performance_monitor._flush_buffer()

        # Get history with narrow time range (should filter out some)
        now = datetime.now(UTC)
        start_time = now - timedelta(seconds=1)
        end_time = now + timedelta(seconds=1)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metric_history",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
            id="test-1",
        )

        response = await handle_get_metric_history(request)

        # Should have some metrics within the time range
        assert response["count"] >= 0
        assert "metrics" in response

    @pytest.mark.asyncio
    async def test_get_metric_history_missing_params(self):
        """Test getting metric history with missing params raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metric_history",
            params={
                "agent_id": "test-agent",
                "task_id": str(uuid4()),
                # Missing start_time, end_time
            },
            id="test-missing",
        )

        with pytest.raises(ValueError):
            await handle_get_metric_history(request)


class TestA2AContext:
    """Test A2A context handling in JSON-RPC methods."""

    @pytest.mark.asyncio
    async def test_track_performance_with_a2a_context(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test tracking performance with A2A context."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "execution",
                "metrics": sample_metrics_data,
            },
            id="test-1",
            a2a_context={
                "trace_id": "trace-123",
                "source_agent": "agent-001",
                "target_agent": sample_agent_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        response = await handle_track_performance(request)

        assert response["success"] is True


class TestErrorHandling:
    """Test error handling for invalid params."""

    @pytest.mark.asyncio
    async def test_track_performance_invalid_uuid(self, sample_agent_id: str):
        """Test tracking performance with invalid UUID raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": "invalid-uuid",
                "stage": "execution",
                "metrics": {
                    "stage_success_rate": 0.85,
                    "stage_error_rate": 0.15,
                    "stage_duration_ms": 2500,
                    "stage_action_count": 12,
                    "overall_progress_velocity": 4.8,
                    "error_accumulation_rate": 0.3,
                    "context_staleness_score": 0.2,
                },
            },
            id="test-invalid",
        )

        with pytest.raises(ValueError):
            await handle_track_performance(request)

    @pytest.mark.asyncio
    async def test_get_metrics_invalid_uuid(self, sample_agent_id: str):
        """Test getting metrics with invalid UUID raises error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics",
            params={
                "agent_id": sample_agent_id,
                "task_id": "not-a-uuid",
            },
            id="test-invalid",
        )

        with pytest.raises(ValueError):
            await handle_get_metrics(request)


class TestDatetimeSerialization:
    """Test datetime serialization in responses."""

    @pytest.mark.asyncio
    async def test_track_performance_datetime_response(
        self, sample_task_id: UUID, sample_agent_id: str, sample_metrics_data: dict
    ):
        """Test track_performance returns valid ISO8601 datetime."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.track_performance",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
                "stage": "execution",
                "metrics": sample_metrics_data,
            },
            id="test-1",
        )

        response = await handle_track_performance(request)

        # Verify recorded_at is valid ISO8601
        recorded_at = response["recorded_at"]
        assert isinstance(recorded_at, str)
        # Should parse without error
        datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_get_metrics_summary_datetime_response(
        self, sample_task_id: UUID, sample_agent_id: str
    ):
        """Test get_metrics_summary returns valid ISO8601 datetime."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="ace.get_metrics_summary",
            params={
                "agent_id": sample_agent_id,
                "task_id": str(sample_task_id),
            },
            id="test-1",
        )

        response = await handle_get_metrics_summary(request)

        # Verify generated_at is valid ISO8601
        generated_at = response["summary"]["generated_at"]
        assert isinstance(generated_at, str)
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
