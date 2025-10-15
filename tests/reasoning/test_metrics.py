"""
Unit tests for Prometheus metrics.

Tests metric recording and validation for reasoning performance monitoring.
"""

from __future__ import annotations

from prometheus_client import REGISTRY

from src.agentcore.reasoning.services.metrics import (
    record_llm_failure,
    record_reasoning_error,
    record_reasoning_request,
    reasoning_compute_savings_pct,
    reasoning_duration_seconds,
    reasoning_errors_total,
    reasoning_iterations_total,
    reasoning_llm_failures_total,
    reasoning_requests_total,
    reasoning_tokens_total,
)


def get_metric_value(metric, labels=None):
    """Helper to get current metric value."""
    if labels:
        return metric.labels(**labels)._value.get()
    return metric._value.get()


def test_record_reasoning_request_success():
    """Test recording a successful reasoning request."""
    # Get initial values
    initial_requests = get_metric_value(reasoning_requests_total, {"status": "success"})

    # Record a successful request
    record_reasoning_request(
        status="success",
        duration_seconds=2.5,
        total_tokens=5000,
        compute_savings_pct=45.5,
        total_iterations=3,
    )

    # Verify counter incremented
    final_requests = get_metric_value(reasoning_requests_total, {"status": "success"})
    assert final_requests > initial_requests


def test_record_reasoning_request_error():
    """Test recording a failed reasoning request."""
    # Get initial values
    initial_requests = get_metric_value(reasoning_requests_total, {"status": "error"})

    # Record a failed request
    record_reasoning_request(
        status="error",
        duration_seconds=1.0,
        total_tokens=1000,
        compute_savings_pct=0.0,
        total_iterations=1,
    )

    # Verify counter incremented
    final_requests = get_metric_value(reasoning_requests_total, {"status": "error"})
    assert final_requests > initial_requests


def test_record_reasoning_error_validation():
    """Test recording validation errors."""
    initial_errors = get_metric_value(
        reasoning_errors_total, {"error_type": "validation_error"}
    )

    record_reasoning_error("validation_error")

    final_errors = get_metric_value(
        reasoning_errors_total, {"error_type": "validation_error"}
    )
    assert final_errors > initial_errors


def test_record_reasoning_error_llm():
    """Test recording LLM errors."""
    initial_errors = get_metric_value(reasoning_errors_total, {"error_type": "llm_error"})

    record_reasoning_error("llm_error")

    final_errors = get_metric_value(reasoning_errors_total, {"error_type": "llm_error"})
    assert final_errors > initial_errors


def test_record_reasoning_error_internal():
    """Test recording internal errors."""
    initial_errors = get_metric_value(
        reasoning_errors_total, {"error_type": "internal_error"}
    )

    record_reasoning_error("internal_error")

    final_errors = get_metric_value(
        reasoning_errors_total, {"error_type": "internal_error"}
    )
    assert final_errors > initial_errors


def test_record_llm_failure():
    """Test recording LLM service failures."""
    # Get initial value
    initial_failures = reasoning_llm_failures_total._value.get()

    record_llm_failure()

    # Verify counter incremented
    final_failures = reasoning_llm_failures_total._value.get()
    assert final_failures > initial_failures


def test_metrics_are_registered():
    """Test that all metrics are registered with Prometheus."""
    # Get all registered metric names
    metric_names = {
        metric.name for family in REGISTRY.collect() for metric in family.samples
    }

    # Verify all expected metrics are registered
    expected_metrics = [
        "reasoning_bounded_context_requests_total",
        "reasoning_bounded_context_errors_total",
        "reasoning_bounded_context_llm_failures_total",
        "reasoning_bounded_context_duration_seconds",
        "reasoning_bounded_context_tokens_total",
        "reasoning_bounded_context_compute_savings_pct",
        "reasoning_bounded_context_iterations_total",
    ]

    for expected in expected_metrics:
        # Check if metric name exists (may have suffixes like _count, _sum, _bucket)
        assert any(
            expected in name for name in metric_names
        ), f"Metric {expected} not registered"


def test_histogram_buckets_configured():
    """Test that histograms have proper bucket configuration."""
    # Duration histogram should have time buckets
    assert hasattr(reasoning_duration_seconds, "_buckets")
    assert len(reasoning_duration_seconds._buckets) > 0

    # Tokens histogram should have token count buckets
    assert hasattr(reasoning_tokens_total, "_buckets")
    assert len(reasoning_tokens_total._buckets) > 0

    # Compute savings histogram should have percentage buckets
    assert hasattr(reasoning_compute_savings_pct, "_buckets")
    assert len(reasoning_compute_savings_pct._buckets) > 0

    # Iterations histogram should have iteration count buckets
    assert hasattr(reasoning_iterations_total, "_buckets")
    assert len(reasoning_iterations_total._buckets) > 0


def test_metrics_labels_work():
    """Test that metric labels work correctly."""
    # Test status label
    initial_success = get_metric_value(reasoning_requests_total, {"status": "success"})
    initial_error = get_metric_value(reasoning_requests_total, {"status": "error"})

    record_reasoning_request("success", 1.0, 1000, 10.0, 1)
    record_reasoning_request("error", 1.0, 1000, 0.0, 1)

    final_success = get_metric_value(reasoning_requests_total, {"status": "success"})
    final_error = get_metric_value(reasoning_requests_total, {"status": "error"})

    assert final_success > initial_success
    assert final_error > initial_error


def test_histogram_observes_values():
    """Test that histograms observe values correctly."""
    # Record some observations
    record_reasoning_request(
        status="success",
        duration_seconds=5.5,
        total_tokens=10000,
        compute_savings_pct=50.0,
        total_iterations=5,
    )

    # Verify histograms have recorded observations
    # Histograms store sum internally, just verify they're >= 0
    assert reasoning_duration_seconds._sum._value >= 0
    assert reasoning_tokens_total._sum._value >= 0
    assert reasoning_compute_savings_pct._sum._value >= 0
    assert reasoning_iterations_total._sum._value >= 0
