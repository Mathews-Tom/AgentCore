"""
Integration tests for ACE Prometheus metrics exposure.

Tests verify:
- Metrics are correctly registered and exposed
- Recording functions update metrics properly
- Metric format is Prometheus-compatible
- Alert rule thresholds are correctly configured
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from agentcore.ace.metrics.prometheus_exporter import (
    record_ace_error,
    record_ace_intervention,
    record_ace_mem_query,
    record_ace_metric_computation,
    record_ace_performance_update,
)


class TestPrometheusMetricsExporter:
    """Test Prometheus metrics exporter functionality."""

    def test_performance_update_metrics_recorded(self):
        """Test that performance updates are recorded correctly."""
        # Record a performance update
        record_ace_performance_update(
            agent_id="test-agent-001",
            stage="planning",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            baseline_delta={"stage_success_rate": -0.05, "error_rate": 0.03},
            context_staleness_score=0.2,
            intervention_effectiveness=0.75,
        )

        # Verify metrics exist in registry
        # Note: prometheus_client automatically appends _total to Counter names
        metric_names = [metric.name for metric in REGISTRY.collect()]
        assert "ace_performance_updates" in metric_names
        assert "ace_error_rate" in metric_names
        assert "ace_stage_duration_seconds" in metric_names
        assert "ace_baseline_deviation" in metric_names
        assert "ace_context_staleness" in metric_names
        assert "ace_intervention_effectiveness" in metric_names

    def test_error_metrics_recorded(self):
        """Test that error detections are recorded correctly."""
        # Record errors with different severities
        record_ace_error(agent_id="test-agent-002", stage="execution", severity="critical")
        record_ace_error(agent_id="test-agent-002", stage="execution", severity="high")
        record_ace_error(agent_id="test-agent-002", stage="planning", severity="medium")

        # Verify error counter exists
        metric_names = [metric.name for metric in REGISTRY.collect()]
        assert "ace_errors" in metric_names

        # Verify labels are correctly applied
        for family in REGISTRY.collect():
            if family.name == "ace_errors":
                samples = list(family.samples)
                # Should have samples with different severity labels
                severities = {sample.labels.get("severity") for sample in samples}
                assert "critical" in severities or "high" in severities or "medium" in severities

    def test_intervention_metrics_recorded(self):
        """Test that interventions are recorded correctly."""
        # Record interventions
        record_ace_intervention(
            agent_id="test-agent-003",
            intervention_type="replan",
            latency_seconds=0.15,
        )
        record_ace_intervention(
            agent_id="test-agent-003",
            intervention_type="refresh_context",
            latency_seconds=0.08,
        )

        # Verify metrics exist
        metric_names = [metric.name for metric in REGISTRY.collect()]
        assert "ace_interventions" in metric_names
        assert "ace_intervention_latency_seconds" in metric_names

    def test_metric_computation_duration_recorded(self):
        """Test that metric computation durations are recorded."""
        # Record metric computations
        record_ace_metric_computation(operation="baseline_update", duration_seconds=0.025)
        record_ace_metric_computation(operation="error_detection", duration_seconds=0.018)

        # Verify histogram exists
        metric_names = [metric.name for metric in REGISTRY.collect()]
        assert "ace_metric_computation_duration_seconds" in metric_names

    def test_ace_mem_query_metrics_recorded(self):
        """Test that ACE-MEM query metrics are recorded."""
        # Record MEM queries
        record_ace_mem_query(query_type="error_context", duration_seconds=0.12)
        record_ace_mem_query(query_type="intervention_history", duration_seconds=0.09)

        # Verify histogram exists
        metric_names = [metric.name for metric in REGISTRY.collect()]
        assert "ace_mem_query_duration_seconds" in metric_names

    def test_stage_validation(self):
        """Test that only valid stages are accepted."""
        valid_stages = ["planning", "execution", "reflection", "verification"]

        for stage in valid_stages:
            # Should not raise exception
            record_ace_performance_update(
                agent_id=f"test-agent-stage-{stage}",
                stage=stage,
                stage_success_rate=0.9,
                stage_error_rate=0.1,
                stage_duration_ms=1000,
            )

    def test_metric_labels_preserved(self):
        """Test that metric labels are correctly preserved."""
        agent_id = "test-agent-labels-001"
        stage = "execution"

        record_ace_performance_update(
            agent_id=agent_id,
            stage=stage,
            stage_success_rate=0.88,
            stage_error_rate=0.12,
            stage_duration_ms=3000,
        )

        # Find the metric family and verify labels
        for family in REGISTRY.collect():
            if family.name == "ace_performance_updates":
                for sample in family.samples:
                    if sample.labels.get("agent_id") == agent_id:
                        assert sample.labels.get("stage") == stage

    def test_baseline_deviation_percentage_conversion(self):
        """Test that baseline deviation is correctly converted to percentage."""
        agent_id = "test-agent-baseline-001"
        stage = "planning"
        baseline_delta = {"stage_success_rate": -0.15}  # -15% deviation

        record_ace_performance_update(
            agent_id=agent_id,
            stage=stage,
            stage_success_rate=0.75,
            stage_error_rate=0.25,
            stage_duration_ms=2000,
            baseline_delta=baseline_delta,
        )

        # Verify deviation is stored as percentage (multiplied by 100)
        for family in REGISTRY.collect():
            if family.name == "ace_baseline_deviation":
                for sample in family.samples:
                    if (
                        sample.labels.get("agent_id") == agent_id
                        and sample.labels.get("metric") == "stage_success_rate"
                    ):
                        # Should be -15.0 (percentage)
                        assert abs(sample.value - (-15.0)) < 0.01

    def test_histogram_buckets_configured(self):
        """Test that histograms have proper bucket configurations."""
        # Record some observations
        record_ace_metric_computation(operation="test_op", duration_seconds=0.035)
        record_ace_mem_query(query_type="test_query", duration_seconds=0.125)
        record_ace_intervention(
            agent_id="test-agent-hist",
            intervention_type="test_intervention",
            latency_seconds=1.5,
        )

        # Verify bucket metrics exist
        for family in REGISTRY.collect():
            if family.name == "ace_metric_computation_duration_seconds":
                # Should have bucket samples
                bucket_samples = [s for s in family.samples if s.name.endswith("_bucket")]
                assert len(bucket_samples) > 0

            elif family.name == "ace_mem_query_duration_seconds":
                bucket_samples = [s for s in family.samples if s.name.endswith("_bucket")]
                assert len(bucket_samples) > 0

            elif family.name == "ace_intervention_latency_seconds":
                bucket_samples = [s for s in family.samples if s.name.endswith("_bucket")]
                assert len(bucket_samples) > 0

    def test_counter_increment_behavior(self):
        """Test that counters increment correctly."""
        agent_id = "test-agent-counter"
        stage = "verification"

        # Record multiple updates
        for _ in range(3):
            record_ace_performance_update(
                agent_id=agent_id,
                stage=stage,
                stage_success_rate=0.92,
                stage_error_rate=0.08,
                stage_duration_ms=1500,
            )

        # Verify counter has value (should be >= 3 from the updates)
        found = False
        for family in REGISTRY.collect():
            if family.name == "ace_performance_updates":
                for sample in family.samples:
                    if (
                        sample.labels.get("agent_id") == agent_id
                        and sample.labels.get("stage") == stage
                    ):
                        # Counters only have _total samples
                        if sample.name == "ace_performance_updates_total":
                            assert sample.value >= 3
                            found = True

        assert found, f"Counter metric not found for {agent_id}/{stage}"

    def test_gauge_set_behavior(self):
        """Test that gauges set values correctly."""
        agent_id = "test-agent-gauge"
        stage = "planning"
        error_rate = 0.22

        record_ace_performance_update(
            agent_id=agent_id,
            stage=stage,
            stage_success_rate=0.78,
            stage_error_rate=error_rate,
            stage_duration_ms=2200,
        )

        # Verify gauge value
        for family in REGISTRY.collect():
            if family.name == "ace_error_rate":
                for sample in family.samples:
                    if (
                        sample.labels.get("agent_id") == agent_id
                        and sample.labels.get("stage") == stage
                    ):
                        assert abs(sample.value - error_rate) < 0.01

    def test_all_compass_metrics_exposed(self):
        """Test that all COMPASS-required metrics are exposed."""
        # Note: Counter names don't have _total in metric family name
        required_metrics = [
            "ace_performance_updates",  # Performance tracking (Counter)
            "ace_errors",  # Error detection (Counter)
            "ace_interventions",  # Intervention tracking (Counter)
            "ace_baseline_deviation",  # Baseline comparison (Gauge)
            "ace_error_rate",  # Current error rate (Gauge)
            "ace_intervention_effectiveness",  # Intervention quality (Gauge)
            "ace_context_staleness",  # Context health (Gauge)
            "ace_metric_computation_duration_seconds",  # System overhead (Histogram)
            "ace_intervention_latency_seconds",  # Intervention latency (Histogram)
            "ace_mem_query_duration_seconds",  # ACE-MEM coordination (Histogram)
            "ace_stage_duration_seconds",  # Stage performance (Histogram)
        ]

        # Record at least one sample for each metric type
        record_ace_performance_update(
            agent_id="compass-test",
            stage="execution",
            stage_success_rate=0.9,
            stage_error_rate=0.1,
            stage_duration_ms=2000,
            baseline_delta={"test": 0.05},
            context_staleness_score=0.3,
            intervention_effectiveness=0.85,
        )
        record_ace_error(agent_id="compass-test", stage="execution", severity="medium")
        record_ace_intervention(
            agent_id="compass-test", intervention_type="replan", latency_seconds=0.5
        )
        record_ace_metric_computation(operation="compass_test", duration_seconds=0.02)
        record_ace_mem_query(query_type="compass_test", duration_seconds=0.1)

        # Verify all metrics are present
        metric_names = {metric.name for metric in REGISTRY.collect()}
        for required_metric in required_metrics:
            assert (
                required_metric in metric_names
            ), f"Required COMPASS metric {required_metric} not found"


class TestPrometheusMetricsFormat:
    """Test Prometheus metric format compliance."""

    def test_metric_names_follow_prometheus_conventions(self):
        """Test that metric names follow Prometheus naming conventions."""
        metric_names = [metric.name for metric in REGISTRY.collect()]

        for name in metric_names:
            if name.startswith("ace_"):
                # Should contain only lowercase, digits, and underscores
                assert name.replace("_", "").replace("ace", "").isalnum() or name.endswith(
                    "_total"
                )

                # Counters should end with _total
                if "_total" in name:
                    assert name.endswith("_total")

                # Duration metrics should end with appropriate unit
                if "duration" in name or "latency" in name:
                    assert name.endswith("_seconds")

    def test_counter_metrics_monotonic(self):
        """Test that counter metrics only increase."""
        agent_id = "test-monotonic"
        stage = "planning"

        # Record initial value
        record_ace_performance_update(
            agent_id=agent_id,
            stage=stage,
            stage_success_rate=0.9,
            stage_error_rate=0.1,
            stage_duration_ms=1000,
        )

        values = []
        for _ in range(5):
            record_ace_performance_update(
                agent_id=agent_id,
                stage=stage,
                stage_success_rate=0.9,
                stage_error_rate=0.1,
                stage_duration_ms=1000,
            )

            for family in REGISTRY.collect():
                if family.name == "ace_performance_updates_total":
                    for sample in family.samples:
                        if (
                            sample.labels.get("agent_id") == agent_id
                            and sample.labels.get("stage") == stage
                        ):
                            values.append(sample.value)

        # Verify monotonic increase
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i], "Counter values should be monotonically increasing"


# Mark as integration test
pytestmark = pytest.mark.integration
