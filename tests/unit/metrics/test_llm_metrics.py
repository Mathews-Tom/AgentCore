"""Unit tests for LLM metrics instrumentation.

This module tests that all Prometheus metrics are correctly defined and
that recording functions properly update the metrics.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from agentcore.a2a_protocol.metrics.llm_metrics import (
    record_governance_violation,
    record_llm_duration,
    record_llm_error,
    record_llm_request,
    record_llm_tokens,
    track_active_requests,
)


@pytest.fixture(autouse=True)
def setup_metrics() -> None:
    """Initialize and cleanup metrics for each test.

    This ensures metrics are created fresh for each test and cleaned up after.
    The autouse=True makes this run before every test automatically.
    """
    # Import to trigger lazy initialization
    from agentcore.a2a_protocol.metrics import llm_metrics  # noqa: F401

    # Unregister metrics from REGISTRY if they exist in cache
    for metric_name, metric in list(llm_metrics._metrics_cache.items()):
        try:
            REGISTRY.unregister(metric)
        except (KeyError, ValueError):
            # Metric not in registry or already unregistered
            pass

    # Clear the metrics cache
    llm_metrics._metrics_cache.clear()

    # Reset lazy-initialized metric globals
    llm_metrics._llm_requests_total = None
    llm_metrics._llm_requests_duration_seconds = None
    llm_metrics._llm_tokens_total = None
    llm_metrics._llm_errors_total = None
    llm_metrics._llm_active_requests = None
    llm_metrics._llm_governance_violations_total = None
    llm_metrics._llm_rate_limit_errors_total = None
    llm_metrics._llm_rate_limit_retry_delay_seconds = None


def get_metric_value(metric_name: str, labels: dict[str, str] | None = None) -> float:
    """Get current value of a metric from Prometheus registry.

    Args:
        metric_name: Name of the metric (sample name, not metric name)
        labels: Label dict to filter on (optional)

    Returns:
        Current metric value
    """
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            # Match on sample name, not metric name (for counters with _total suffix)
            if sample.name == metric_name:
                if labels is None or all(
                    sample.labels.get(k) == v for k, v in labels.items()
                ):
                    return sample.value
    return 0.0


class TestLLMMetricsDefinition:
    """Test that all required metrics are properly defined."""

    def test_llm_requests_total_exists(self) -> None:
        """Test that llm_requests_total counter exists with correct labels."""
        # Trigger metric creation by recording a request
        record_llm_request("openai", "gpt-4.1-mini", "success")

        # Verify metric exists in registry
        # Note: Prometheus counters with _total suffix have metric name WITHOUT suffix
        metric_found = False
        for metric in REGISTRY.collect():
            if metric.name == "llm_requests":
                metric_found = True
                assert metric.type == "counter"
                # Verify labels are present
                for sample in metric.samples:
                    if sample.name == "llm_requests_total" and "provider" in sample.labels:
                        assert "model" in sample.labels
                        assert "status" in sample.labels
        assert metric_found, "llm_requests counter not found"

    def test_llm_requests_duration_seconds_exists(self) -> None:
        """Test that llm_requests_duration_seconds histogram exists with correct labels."""
        # Trigger metric creation
        record_llm_duration("anthropic", "claude-3-5-haiku-20241022", 1.5)

        # Verify metric exists in registry
        metric_found = False
        for metric in REGISTRY.collect():
            if metric.name == "llm_requests_duration_seconds":
                metric_found = True
                assert metric.type == "histogram"
                # Verify labels are present
                for sample in metric.samples:
                    if "provider" in sample.labels:
                        assert "model" in sample.labels
        assert metric_found, "llm_requests_duration_seconds metric not found"

    def test_llm_tokens_total_exists(self) -> None:
        """Test that llm_tokens_total counter exists with correct labels."""
        # Trigger metric creation
        record_llm_tokens("gemini", "gemini-1.5-flash", "prompt", 100)

        # Verify metric exists in registry
        metric_found = False
        for metric in REGISTRY.collect():
            if metric.name == "llm_tokens":
                metric_found = True
                assert metric.type == "counter"
                # Verify labels are present
                for sample in metric.samples:
                    if sample.name == "llm_tokens_total" and "provider" in sample.labels:
                        assert "model" in sample.labels
                        assert "token_type" in sample.labels
        assert metric_found, "llm_tokens counter not found"

    def test_llm_errors_total_exists(self) -> None:
        """Test that llm_errors_total counter exists with correct labels."""
        # Trigger metric creation
        record_llm_error("openai", "gpt-5", "ProviderError")

        # Verify metric exists in registry
        metric_found = False
        for metric in REGISTRY.collect():
            if metric.name == "llm_errors":
                metric_found = True
                assert metric.type == "counter"
                # Verify labels are present
                for sample in metric.samples:
                    if sample.name == "llm_errors_total" and "provider" in sample.labels:
                        assert "model" in sample.labels
                        assert "error_type" in sample.labels
        assert metric_found, "llm_errors counter not found"

    def test_llm_active_requests_exists(self) -> None:
        """Test that llm_active_requests gauge exists with correct labels."""
        # Trigger metric creation
        with track_active_requests("gemini"):
            pass

        # Verify metric exists in registry
        metric_found = False
        for metric in REGISTRY.collect():
            if metric.name == "llm_active_requests":
                metric_found = True
                assert metric.type == "gauge"
                # Verify labels are present
                for sample in metric.samples:
                    if "provider" in sample.labels:
                        pass  # Just verify label exists
        assert metric_found, "llm_active_requests metric not found"

    def test_llm_governance_violations_total_exists(self) -> None:
        """Test that llm_governance_violations_total counter exists with correct labels."""
        # Trigger metric creation
        record_governance_violation("gpt-10-ultra", "test-agent-999")

        # Verify metric exists in registry
        metric_found = False
        for metric in REGISTRY.collect():
            if metric.name == "llm_governance_violations":
                metric_found = True
                assert metric.type == "counter"
                # Verify labels are present
                for sample in metric.samples:
                    if sample.name == "llm_governance_violations_total" and "model" in sample.labels:
                        assert "source_agent" in sample.labels
        assert metric_found, "llm_governance_violations counter not found"


class TestLLMMetricsRecording:
    """Test that recording functions correctly update metrics."""

    def test_record_llm_request_increments_counter(self) -> None:
        """Test that record_llm_request increments the counter."""
        # Get initial value
        initial = get_metric_value(
            "llm_requests_total",
            {"provider": "openai", "model": "gpt-4.1", "status": "success"},
        )

        # Record a successful request
        record_llm_request("openai", "gpt-4.1", "success")

        # Verify it incremented
        value = get_metric_value(
            "llm_requests_total",
            {"provider": "openai", "model": "gpt-4.1", "status": "success"},
        )
        assert value == initial + 1.0

    def test_record_llm_request_different_labels(self) -> None:
        """Test that different labels create separate metrics."""
        # Get initial values
        init_success = get_metric_value(
            "llm_requests_total",
            {"provider": "openai", "model": "gpt-5-mini", "status": "success"},
        )
        init_error = get_metric_value(
            "llm_requests_total",
            {"provider": "openai", "model": "gpt-5-mini", "status": "error"},
        )
        init_anthropic = get_metric_value(
            "llm_requests_total",
            {
                "provider": "anthropic",
                "model": "claude-3-opus",
                "status": "success",
            },
        )

        # Record requests with different labels
        record_llm_request("openai", "gpt-5-mini", "success")
        record_llm_request("openai", "gpt-5-mini", "error")
        record_llm_request("anthropic", "claude-3-opus", "success")

        # Verify each combination incremented independently
        assert (
            get_metric_value(
                "llm_requests_total",
                {"provider": "openai", "model": "gpt-5-mini", "status": "success"},
            )
            == init_success + 1.0
        )
        assert (
            get_metric_value(
                "llm_requests_total",
                {"provider": "openai", "model": "gpt-5-mini", "status": "error"},
            )
            == init_error + 1.0
        )
        assert (
            get_metric_value(
                "llm_requests_total",
                {
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "status": "success",
                },
            )
            == init_anthropic + 1.0
        )

    def test_record_llm_duration_observes_histogram(self) -> None:
        """Test that record_llm_duration observes values in histogram."""
        # Get initial count
        init_count = get_metric_value(
            "llm_requests_duration_seconds_count",
            {"provider": "gemini", "model": "gemini-2.0-flash-exp"},
        )

        # Record some durations
        record_llm_duration("gemini", "gemini-2.0-flash-exp", 0.5)
        record_llm_duration("gemini", "gemini-2.0-flash-exp", 1.5)
        record_llm_duration("gemini", "gemini-2.0-flash-exp", 2.5)

        # Verify histogram count increased by 3
        count = get_metric_value(
            "llm_requests_duration_seconds_count",
            {"provider": "gemini", "model": "gemini-2.0-flash-exp"},
        )
        assert count == init_count + 3.0

    def test_record_llm_tokens_increments_by_count(self) -> None:
        """Test that record_llm_tokens increments by specified count."""
        # Get initial value
        init_prompt = get_metric_value(
            "llm_tokens_total",
            {"provider": "openai", "model": "gpt-4.1", "token_type": "prompt"},
        )
        init_completion = get_metric_value(
            "llm_tokens_total",
            {"provider": "openai", "model": "gpt-4.1", "token_type": "completion"},
        )

        # Record prompt tokens
        record_llm_tokens("openai", "gpt-4.1", "prompt", 100)
        record_llm_tokens("openai", "gpt-4.1", "prompt", 50)

        # Verify counter increased by total
        value = get_metric_value(
            "llm_tokens_total",
            {"provider": "openai", "model": "gpt-4.1", "token_type": "prompt"},
        )
        assert value == init_prompt + 150.0

        # Record completion tokens
        record_llm_tokens("openai", "gpt-4.1", "completion", 75)

        # Verify completion tokens tracked separately
        value = get_metric_value(
            "llm_tokens_total",
            {"provider": "openai", "model": "gpt-4.1", "token_type": "completion"},
        )
        assert value == init_completion + 75.0

    def test_record_llm_error_increments_counter(self) -> None:
        """Test that record_llm_error increments the error counter."""
        # Get initial values
        init_provider = get_metric_value(
            "llm_errors_total",
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "error_type": "ProviderError",
            },
        )
        init_timeout = get_metric_value(
            "llm_errors_total",
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "error_type": "ProviderTimeoutError",
            },
        )

        # Record errors
        record_llm_error("anthropic", "claude-3-5-sonnet", "ProviderError")
        record_llm_error("anthropic", "claude-3-5-sonnet", "ProviderError")
        record_llm_error("anthropic", "claude-3-5-sonnet", "ProviderTimeoutError")

        # Verify error counts incremented correctly
        provider_error_count = get_metric_value(
            "llm_errors_total",
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "error_type": "ProviderError",
            },
        )
        assert provider_error_count == init_provider + 2.0

        timeout_error_count = get_metric_value(
            "llm_errors_total",
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "error_type": "ProviderTimeoutError",
            },
        )
        assert timeout_error_count == init_timeout + 1.0

    def test_record_governance_violation_increments_counter(self) -> None:
        """Test that record_governance_violation increments the counter."""
        # Get initial values
        init1 = get_metric_value(
            "llm_governance_violations_total",
            {"model": "forbidden-model-1", "source_agent": "test-agent-111"},
        )
        init2 = get_metric_value(
            "llm_governance_violations_total",
            {"model": "forbidden-model-1", "source_agent": "test-agent-222"},
        )

        # Record violations
        record_governance_violation("forbidden-model-1", "test-agent-111")
        record_governance_violation("forbidden-model-1", "test-agent-111")
        record_governance_violation("forbidden-model-1", "test-agent-222")

        # Verify violation counts incremented correctly
        value1 = get_metric_value(
            "llm_governance_violations_total",
            {"model": "forbidden-model-1", "source_agent": "test-agent-111"},
        )
        assert value1 == init1 + 2.0

        value2 = get_metric_value(
            "llm_governance_violations_total",
            {"model": "forbidden-model-1", "source_agent": "test-agent-222"},
        )
        assert value2 == init2 + 1.0

    def test_record_governance_violation_unknown_agent(self) -> None:
        """Test that governance violations with None source_agent use 'unknown'."""
        # Get initial value
        init = get_metric_value(
            "llm_governance_violations_total",
            {"model": "forbidden-model-2", "source_agent": "unknown"},
        )

        # Record violation with None source_agent
        record_governance_violation("forbidden-model-2", None)

        # Verify it's recorded as 'unknown'
        value = get_metric_value(
            "llm_governance_violations_total",
            {"model": "forbidden-model-2", "source_agent": "unknown"},
        )
        assert value == init + 1.0

    def test_track_active_requests_context_manager(self) -> None:
        """Test that track_active_requests increments and decrements gauge."""
        # Get initial state
        initial = get_metric_value("llm_active_requests", {"provider": "anthropic"})

        # Enter context manager
        with track_active_requests("anthropic"):
            # Should be incremented
            active = get_metric_value("llm_active_requests", {"provider": "anthropic"})
            assert active == initial + 1.0

        # After exiting, should be back to initial
        final = get_metric_value("llm_active_requests", {"provider": "anthropic"})
        assert final == initial

    def test_track_active_requests_exception_handling(self) -> None:
        """Test that track_active_requests decrements gauge even on exception."""
        # Get initial state
        initial = get_metric_value("llm_active_requests", {"provider": "gemini"})

        # Raise exception inside context manager
        with pytest.raises(ValueError):
            with track_active_requests("gemini"):
                # Should be incremented
                active = get_metric_value("llm_active_requests", {"provider": "gemini"})
                assert active == initial + 1.0
                raise ValueError("Test exception")

        # After exception, should be back to initial
        final = get_metric_value("llm_active_requests", {"provider": "gemini"})
        assert final == initial

    def test_track_active_requests_multiple_providers(self) -> None:
        """Test that active requests are tracked separately per provider."""
        # Get initial values
        init_openai = get_metric_value("llm_active_requests", {"provider": "openai"})
        init_anthropic = get_metric_value(
            "llm_active_requests", {"provider": "anthropic"}
        )

        with track_active_requests("openai"):
            with track_active_requests("anthropic"):
                # Both should be incremented from initial
                openai_active = get_metric_value(
                    "llm_active_requests", {"provider": "openai"}
                )
                anthropic_active = get_metric_value(
                    "llm_active_requests", {"provider": "anthropic"}
                )
                assert openai_active == init_openai + 1.0
                assert anthropic_active == init_anthropic + 1.0

            # Anthropic should be back to initial
            anthropic_active = get_metric_value(
                "llm_active_requests", {"provider": "anthropic"}
            )
            assert anthropic_active == init_anthropic

            # OpenAI should still be incremented
            openai_active = get_metric_value(
                "llm_active_requests", {"provider": "openai"}
            )
            assert openai_active == init_openai + 1.0

        # Both should be back to initial
        openai_final = get_metric_value("llm_active_requests", {"provider": "openai"})
        anthropic_final = get_metric_value(
            "llm_active_requests", {"provider": "anthropic"}
        )
        assert openai_final == init_openai
        assert anthropic_final == init_anthropic
