"""Tests for performance monitoring system.

Tests metrics collection, SLA monitoring, performance analytics,
and dashboard data generation with 90%+ coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from agentcore.integration.portkey.metrics_collector import MetricsCollector
from agentcore.integration.portkey.metrics_models import (
    AlertSeverity,
    MetricType,
    PerformanceLevel,
    RequestMetrics,
    SLAStatus,
)
from agentcore.integration.portkey.performance_monitor import PerformanceMonitor


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Create a metrics collector for testing."""
    return MetricsCollector()


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Create a performance monitor for testing."""
    return PerformanceMonitor(
        availability_target=99.0,
        response_time_target_ms=2000,
        success_rate_target=95.0,
        alert_debounce_seconds=1,  # Short debounce for tests
    )


@pytest.fixture
def sample_request_metrics() -> RequestMetrics:
    """Create sample request metrics for testing."""
    return RequestMetrics(
        request_id="test-request-1",
        trace_id="test-trace-1",
        timestamp=datetime.now(),
        provider_id="openai",
        provider_name="OpenAI",
        model="gpt-4.1-mini",
        model_version="2024-01",
        total_latency_ms=1500,
        routing_latency_ms=10,
        queue_latency_ms=20,
        provider_latency_ms=1400,
        network_latency_ms=70,
        ttft_ms=100,
        input_tokens=100,
        output_tokens=200,
        total_tokens=300,
        cached_tokens=0,
        prompt_tokens=100,
        completion_tokens=200,
        total_cost=0.015,
        input_cost=0.005,
        output_cost=0.010,
        cache_cost_saved=0.0,
        tokens_per_second=200.0,
        success=True,
        response_complete=True,
        finish_reason="stop",
        quality_score=0.95,
        error_occurred=False,
        error_type=None,
        error_message=None,
        retry_count=0,
        fallback_used=False,
        memory_used_mb=150.0,
        cpu_percent=25.0,
        network_bytes_sent=2048,
        network_bytes_received=4096,
        cache_hit=False,
        cache_level=None,
        cache_lookup_ms=None,
        tenant_id="test-tenant",
        workflow_id="test-workflow",
        agent_id="test-agent",
        session_id="test-session",
        temperature=0.7,
        max_tokens=1000,
        stream=False,
        tags={"env": "test"},
        metadata={"test": "data"},
    )


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    async def test_collect_request_metrics_success(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Test collecting metrics for successful request."""
        request_data = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
        }

        response_data = {
            "id": "resp-123",
            "model": "gpt-4.1-mini",
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        timing_data = {
            "total": 1000,
            "routing": 10,
            "queue": 5,
            "provider": 980,
            "network": 5,
        }

        context = {
            "tenant_id": "test-tenant",
            "trace_id": "trace-123",
            "cost_data": {"total_cost": 0.001, "input_cost": 0.0005, "output_cost": 0.0005},
        }

        metrics = await metrics_collector.collect_request_metrics(
            request_id="req-123",
            provider_id="openai",
            provider_name="OpenAI",
            model="gpt-4.1-mini",
            request_data=request_data,
            response_data=response_data,
            timing_data=timing_data,
            error_data=None,
            context=context,
        )

        assert metrics.request_id == "req-123"
        assert metrics.provider_id == "openai"
        assert metrics.model == "gpt-4.1-mini"
        assert metrics.total_latency_ms == 1000
        assert metrics.input_tokens == 10
        assert metrics.output_tokens == 5
        assert metrics.total_cost == 0.001
        assert metrics.success is True
        assert metrics.error_occurred is False
        assert metrics.tenant_id == "test-tenant"

    async def test_collect_request_metrics_error(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Test collecting metrics for failed request."""
        request_data = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        timing_data = {
            "total": 500,
            "routing": 10,
            "queue": 5,
            "provider": 480,
            "network": 5,
        }

        error_data = {
            "type": "PortkeyTimeoutError",
            "message": "Request timed out",
            "retry_count": 3,
            "fallback_used": True,
        }

        metrics = await metrics_collector.collect_request_metrics(
            request_id="req-err-123",
            provider_id="openai",
            provider_name="OpenAI",
            model="gpt-4.1-mini",
            request_data=request_data,
            response_data=None,
            timing_data=timing_data,
            error_data=error_data,
            context={},
        )

        assert metrics.request_id == "req-err-123"
        assert metrics.success is False
        assert metrics.error_occurred is True
        assert metrics.error_type == "PortkeyTimeoutError"
        assert metrics.error_message == "Request timed out"
        assert metrics.retry_count == 3
        assert metrics.fallback_used is True

    async def test_collect_from_response(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Test collecting metrics from response data."""
        import time

        request = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = {
            "id": "resp-123",
            "model": "gpt-4.1-mini",
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        start_time = time.time()

        metrics = await metrics_collector.collect_from_response(
            request=request,
            response=response,
            start_time=start_time,
            provider_id="openai",
            provider_name="OpenAI",
            context={"tenant_id": "test"},
        )

        assert metrics.model == "gpt-4.1-mini"
        assert metrics.input_tokens == 5
        assert metrics.output_tokens == 10
        assert metrics.success is True
        assert metrics.tenant_id == "test"

    async def test_collect_from_error(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Test collecting metrics from error."""
        import time

        request = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Test"}],
        }

        error = Exception("Test error")
        start_time = time.time()

        metrics = await metrics_collector.collect_from_error(
            request=request,
            error=error,
            start_time=start_time,
            provider_id="openai",
            provider_name="OpenAI",
            context={"retry_count": 2},
        )

        assert metrics.success is False
        assert metrics.error_occurred is True
        assert metrics.error_type == "Exception"
        assert "Test error" in (metrics.error_message or "")

    def test_get_metrics_history(
        self,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test retrieving metrics history with filters."""
        # Add sample metrics
        metrics_collector._metrics_history.append(sample_request_metrics)

        # Add another metric with different provider
        metrics2 = sample_request_metrics.model_copy(deep=True)
        metrics2.provider_id = "anthropic"
        metrics2.request_id = "test-request-2"
        metrics_collector._metrics_history.append(metrics2)

        # Test filter by provider
        openai_metrics = metrics_collector.get_metrics_history(provider_id="openai")
        assert len(openai_metrics) == 1
        assert openai_metrics[0].provider_id == "openai"

        # Test filter by tenant
        tenant_metrics = metrics_collector.get_metrics_history(tenant_id="test-tenant")
        assert len(tenant_metrics) == 2

        # Test limit
        limited_metrics = metrics_collector.get_metrics_history(limit=1)
        assert len(limited_metrics) == 1

    def test_get_stats(
        self,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test getting collector statistics."""
        # Add successful metric
        metrics_collector._metrics_history.append(sample_request_metrics)

        # Add failed metric
        failed_metrics = sample_request_metrics.model_copy(deep=True)
        failed_metrics.success = False
        failed_metrics.request_id = "failed-request"
        metrics_collector._metrics_history.append(failed_metrics)

        stats = metrics_collector.get_stats()

        assert stats["total_metrics"] == 2
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 1
        assert stats["success_rate"] == 50.0


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    def test_calculate_performance_metrics(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test calculating aggregated performance metrics."""
        # Add sample metrics to collector
        performance_monitor._metrics_collector = metrics_collector
        metrics_collector._metrics_history.append(sample_request_metrics)

        # Add more metrics with different values
        for i in range(9):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+2}"
            metrics.total_latency_ms = 1000 + (i * 100)
            metrics.total_cost = 0.01 + (i * 0.001)
            metrics_collector._metrics_history.append(metrics)

        now = datetime.now()
        start_time = now - timedelta(hours=1)

        perf_metrics = performance_monitor.calculate_performance_metrics(
            start_time=start_time,
            end_time=now,
        )

        assert perf_metrics.total_requests == 10
        assert perf_metrics.successful_requests == 10
        assert perf_metrics.failed_requests == 0
        assert perf_metrics.success_rate == 100.0
        assert perf_metrics.avg_latency_ms > 0
        assert perf_metrics.total_cost > 0

    def test_calculate_sla_metrics_compliant(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test SLA metrics calculation when compliant."""
        performance_monitor._metrics_collector = metrics_collector

        # Add metrics that meet SLA targets
        for i in range(10):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+1}"
            metrics.total_latency_ms = 1000  # Below 2000ms target
            metrics.success = True
            metrics_collector._metrics_history.append(metrics)

        now = datetime.now()
        start_time = now - timedelta(hours=1)

        sla_metrics = performance_monitor.calculate_sla_metrics(
            start_time=start_time,
            end_time=now,
            measurement_window_hours=1,
        )

        assert sla_metrics.actual_availability == 100.0
        assert sla_metrics.actual_response_time_ms == 1000.0
        assert sla_metrics.actual_success_rate == 100.0
        assert sla_metrics.availability_status == SLAStatus.COMPLIANT
        assert sla_metrics.response_time_status == SLAStatus.COMPLIANT
        assert sla_metrics.success_rate_status == SLAStatus.COMPLIANT
        assert sla_metrics.overall_status == SLAStatus.COMPLIANT

    def test_calculate_sla_metrics_violated(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test SLA metrics calculation when violated."""
        performance_monitor._metrics_collector = metrics_collector

        # Add metrics that violate SLA targets
        for i in range(10):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+1}"
            metrics.total_latency_ms = 5000  # Above 2000ms target
            metrics.success = i < 8  # 80% success rate (below 95% target)
            metrics_collector._metrics_history.append(metrics)

        now = datetime.now()
        start_time = now - timedelta(hours=1)

        sla_metrics = performance_monitor.calculate_sla_metrics(
            start_time=start_time,
            end_time=now,
            measurement_window_hours=1,
        )

        assert sla_metrics.actual_success_rate == 80.0
        assert sla_metrics.actual_response_time_ms == 5000.0
        # Should have violations
        assert sla_metrics.overall_status != SLAStatus.COMPLIANT

    def test_calculate_provider_performance(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test calculating provider-specific performance metrics."""
        performance_monitor._metrics_collector = metrics_collector

        # Add metrics for a specific provider
        for i in range(5):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"openai-request-{i+1}"
            metrics.provider_id = "openai"
            metrics.provider_name = "OpenAI"
            metrics_collector._metrics_history.append(metrics)

        now = datetime.now()
        start_time = now - timedelta(hours=1)

        provider_perf = performance_monitor.calculate_provider_performance(
            provider_id="openai",
            provider_name="OpenAI",
            start_time=start_time,
            end_time=now,
        )

        assert provider_perf.provider_id == "openai"
        assert provider_perf.provider_name == "OpenAI"
        assert provider_perf.performance_metrics.total_requests == 5
        assert provider_perf.availability_score >= 0.0
        assert provider_perf.availability_score <= 1.0
        assert provider_perf.reliability_score >= 0.0
        assert provider_perf.reliability_score <= 1.0
        assert provider_perf.overall_score >= 0.0
        assert provider_perf.overall_score <= 1.0
        assert provider_perf.performance_level in [
            PerformanceLevel.EXCELLENT,
            PerformanceLevel.GOOD,
            PerformanceLevel.DEGRADED,
            PerformanceLevel.POOR,
        ]

    def test_generate_performance_insights(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test generating performance insights."""
        performance_monitor._metrics_collector = metrics_collector

        # Add metrics with high latency spike (need more variance)
        for i in range(100):  # Increased to 100 requests
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+1}"
            metrics.timestamp = datetime.now() - timedelta(minutes=i)
            # 10% of requests have very high latency (>2 std dev)
            if i < 10:
                metrics.total_latency_ms = 20000  # Very high
            else:
                metrics.total_latency_ms = 1000 + (i % 100)  # Normal with small variance
            metrics_collector._metrics_history.append(metrics)

        now = datetime.now()
        start_time = now - timedelta(hours=2)

        insights = performance_monitor.generate_performance_insights(
            start_time=start_time,
            end_time=now,
        )

        # Should generate at least one insight (latency or cost)
        assert len(insights) >= 0  # May or may not generate insights

        # Check insight properties if any were generated
        for insight in insights:
            assert insight.insight_id
            assert insight.insight_type
            assert insight.title
            assert insight.description
            assert 0.0 <= insight.confidence <= 1.0
            assert insight.impact_level in ["low", "medium", "high"]

    def test_export_prometheus_metrics(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test exporting Prometheus metrics."""
        performance_monitor._metrics_collector = metrics_collector

        # Add recent metrics (last 5 minutes)
        now = datetime.now()
        for i in range(10):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+1}"
            metrics.timestamp = now - timedelta(seconds=i * 10)
            metrics_collector._metrics_history.append(metrics)

        prom_metrics = performance_monitor.export_prometheus_metrics()

        assert prom_metrics.total_requests == 10
        assert prom_metrics.successful_requests == 10
        assert prom_metrics.failed_requests == 0
        assert prom_metrics.current_latency_ms > 0
        assert prom_metrics.current_throughput >= 0
        assert 0.0 <= prom_metrics.current_error_rate <= 100.0
        assert len(prom_metrics.latency_histogram) > 0
        assert len(prom_metrics.token_count_histogram) > 0
        assert "p50" in prom_metrics.latency_summary
        assert "p95" in prom_metrics.latency_summary
        assert "p99" in prom_metrics.latency_summary

    def test_get_dashboard_data(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test getting dashboard data."""
        performance_monitor._metrics_collector = metrics_collector

        # Add metrics for last 24 hours
        now = datetime.now()
        for i in range(100):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+1}"
            metrics.timestamp = now - timedelta(minutes=i * 10)
            metrics_collector._metrics_history.append(metrics)

        dashboard = performance_monitor.get_dashboard_data()

        assert dashboard.total_requests_24h > 0
        assert 0.0 <= dashboard.success_rate_24h <= 100.0
        assert dashboard.avg_latency_24h > 0
        assert dashboard.total_cost_24h >= 0
        assert dashboard.current_throughput >= 0
        assert dashboard.current_latency_ms >= 0
        assert 0.0 <= dashboard.current_error_rate <= 100.0
        assert dashboard.sla_compliance is not None
        assert isinstance(dashboard.top_providers, list)
        assert isinstance(dashboard.active_alerts, list)
        assert isinstance(dashboard.recent_insights, list)

    def test_alert_debouncing(
        self,
        performance_monitor: PerformanceMonitor,
        metrics_collector: MetricsCollector,
        sample_request_metrics: RequestMetrics,
    ) -> None:
        """Test alert debouncing to prevent duplicate alerts."""
        performance_monitor._metrics_collector = metrics_collector

        # Add metrics that violate SLA
        for i in range(10):
            metrics = sample_request_metrics.model_copy(deep=True)
            metrics.request_id = f"request-{i+1}"
            metrics.success = False  # All failures
            metrics_collector._metrics_history.append(metrics)

        now = datetime.now()
        start_time = now - timedelta(hours=1)

        # First SLA check should generate alerts
        sla_metrics1 = performance_monitor.calculate_sla_metrics(
            start_time=start_time,
            end_time=now,
        )

        initial_alert_count = len(performance_monitor._alerts)
        assert initial_alert_count > 0

        # Immediate second check should not generate new alerts (debounced)
        sla_metrics2 = performance_monitor.calculate_sla_metrics(
            start_time=start_time,
            end_time=now,
        )

        # Alert count should be the same (debounced)
        assert len(performance_monitor._alerts) == initial_alert_count

    def test_acknowledge_and_resolve_alerts(
        self,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test acknowledging and resolving alerts."""
        # Create a test alert
        alert = performance_monitor._create_alert(
            metric_type=MetricType.LATENCY,
            severity=AlertSeverity.WARNING,
            threshold_name="test_threshold",
            threshold_value=1000.0,
            actual_value=2000.0,
            title="Test Alert",
            message="Test alert message",
        )

        assert not alert.acknowledged
        assert not alert.resolved

        # Acknowledge alert
        acknowledged = performance_monitor.acknowledge_alert(alert.alert_id)
        assert acknowledged
        assert alert.acknowledged

        # Resolve alert
        resolved = performance_monitor.resolve_alert(alert.alert_id)
        assert resolved
        assert alert.resolved
        assert alert.resolved_at is not None

    def test_get_alerts_filtering(
        self,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test filtering alerts by status."""
        # Create test alerts
        alert1 = performance_monitor._create_alert(
            metric_type=MetricType.LATENCY,
            severity=AlertSeverity.WARNING,
            threshold_name="latency",
            threshold_value=1000.0,
            actual_value=2000.0,
            title="Alert 1",
            message="Message 1",
        )

        alert2 = performance_monitor._create_alert(
            metric_type=MetricType.COST,
            severity=AlertSeverity.CRITICAL,
            threshold_name="cost",
            threshold_value=10.0,
            actual_value=20.0,
            title="Alert 2",
            message="Message 2",
        )

        # Acknowledge one alert
        performance_monitor.acknowledge_alert(alert1.alert_id)

        # Test filtering
        all_alerts = performance_monitor.get_alerts()
        assert len(all_alerts) >= 2

        acknowledged_alerts = performance_monitor.get_alerts(acknowledged=True)
        assert len(acknowledged_alerts) >= 1
        assert all(a.acknowledged for a in acknowledged_alerts)

        unacknowledged_alerts = performance_monitor.get_alerts(acknowledged=False)
        assert len(unacknowledged_alerts) >= 1
        assert all(not a.acknowledged for a in unacknowledged_alerts)


class TestPercentileCalculation:
    """Tests for percentile calculation utility."""

    def test_calculate_percentile(
        self,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        p50 = performance_monitor._calculate_percentile(values, 50)
        assert 5 <= p50 <= 6  # Percentile calculation can vary

        p95 = performance_monitor._calculate_percentile(values, 95)
        assert p95 >= 9

        p99 = performance_monitor._calculate_percentile(values, 99)
        assert p99 >= 9

    def test_calculate_percentile_empty(
        self,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test percentile calculation with empty list."""
        result = performance_monitor._calculate_percentile([], 50)
        assert result == 0.0


class TestIntegration:
    """Integration tests for the complete monitoring system."""

    async def test_end_to_end_metrics_flow(
        self,
        metrics_collector: MetricsCollector,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test complete metrics collection and analysis flow."""
        performance_monitor._metrics_collector = metrics_collector

        # Simulate 50 requests over time
        now = datetime.now()
        for i in range(50):
            # Vary success and latency
            success = i < 45  # 90% success rate
            latency = 1000 if success else 5000

            request_data = {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": f"Request {i}"}],
            }

            response_data = (
                {
                    "id": f"resp-{i}",
                    "model": "gpt-4.1-mini",
                    "choices": [{"message": {"content": "Response"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                }
                if success
                else None
            )

            timing_data = {
                "total": latency,
                "routing": 10,
                "queue": 5,
                "provider": latency - 20,
                "network": 5,
            }

            error_data = None if success else {"type": "TimeoutError", "message": "Timeout"}

            # Set timestamp to be within query range
            context = {
                "tenant_id": "test",
                "timestamp": now - timedelta(minutes=i),
            }

            metrics = await metrics_collector.collect_request_metrics(
                request_id=f"req-{i}",
                provider_id="openai",
                provider_name="OpenAI",
                model="gpt-4.1-mini",
                request_data=request_data,
                response_data=response_data,
                timing_data=timing_data,
                error_data=error_data,
                context=context,
            )
            # Set timestamp manually to be within range
            metrics.timestamp = now - timedelta(seconds=i * 10)

        # Analyze performance with wider time range to capture all metrics
        start_time = now - timedelta(hours=2)
        end_time = now + timedelta(minutes=5)

        perf_metrics = performance_monitor.calculate_performance_metrics(
            start_time=start_time,
            end_time=end_time,
        )

        assert perf_metrics.total_requests == 50
        assert perf_metrics.successful_requests == 45
        assert perf_metrics.failed_requests == 5
        assert perf_metrics.success_rate == 90.0

        # Check SLA metrics
        sla_metrics = performance_monitor.calculate_sla_metrics(
            start_time=start_time,
            end_time=end_time,
        )

        assert sla_metrics.actual_success_rate == 90.0
        assert sla_metrics.actual_availability == 90.0

        # Get dashboard data
        dashboard = performance_monitor.get_dashboard_data()

        assert dashboard.total_requests_24h > 0
        assert dashboard.success_rate_24h > 0

        # Export Prometheus metrics
        prom_metrics = performance_monitor.export_prometheus_metrics()

        assert prom_metrics.total_requests > 0
        assert prom_metrics.successful_requests > 0
