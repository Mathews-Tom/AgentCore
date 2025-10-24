"""Comprehensive unit tests for provider health monitoring.

Tests health metrics tracking, circuit breaker management, and background monitoring.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.integration.portkey.health import ProviderHealthMonitor, RequestRecord
from agentcore.integration.portkey.provider import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    ProviderCapabilities,
    ProviderConfiguration,
    ProviderHealthMetrics,
    ProviderMetadata,
    ProviderStatus,
)
from agentcore.integration.portkey.registry import ProviderRegistry


@pytest.fixture
def registry() -> ProviderRegistry:
    """Create a test registry with sample providers."""
    registry = ProviderRegistry()

    providers = [
        ProviderConfiguration(
            provider_id="provider1",
            enabled=True,
            metadata=ProviderMetadata(name="Provider 1"),
            capabilities=ProviderCapabilities(),
        ),
        ProviderConfiguration(
            provider_id="provider2",
            enabled=True,
            metadata=ProviderMetadata(name="Provider 2"),
            capabilities=ProviderCapabilities(),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=10,
            ),
        ),
    ]

    registry.register_providers(providers)
    return registry


@pytest.fixture
def health_monitor(registry: ProviderRegistry) -> ProviderHealthMonitor:
    """Create a health monitor instance."""
    return ProviderHealthMonitor(
        registry=registry,
        monitoring_window_seconds=300,
        health_check_interval_seconds=30,
    )


class TestHealthMonitorInitialization:
    """Test health monitor initialization."""

    def test_initialization(self, registry: ProviderRegistry) -> None:
        """Test health monitor initialization."""
        monitor = ProviderHealthMonitor(
            registry=registry,
            monitoring_window_seconds=600,
            health_check_interval_seconds=60,
        )

        assert monitor.registry is registry
        assert monitor.monitoring_window == timedelta(seconds=600)
        assert monitor.health_check_interval == 60
        assert len(monitor._request_history) == 0
        assert monitor._running is False


class TestRequestTracking:
    """Test request recording and tracking."""

    def test_record_request_success(self, health_monitor: ProviderHealthMonitor) -> None:
        """Test recording a successful request."""
        health_monitor.record_request(
            provider_id="provider1",
            success=True,
            latency_ms=100,
        )

        history = health_monitor._request_history["provider1"]
        assert len(history) == 1
        assert history[0].success is True
        assert history[0].latency_ms == 100
        assert history[0].error is None

    def test_record_request_failure(self, health_monitor: ProviderHealthMonitor) -> None:
        """Test recording a failed request."""
        health_monitor.record_request(
            provider_id="provider1",
            success=False,
            latency_ms=500,
            error="Connection timeout",
        )

        history = health_monitor._request_history["provider1"]
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].latency_ms == 500
        assert history[0].error == "Connection timeout"

    def test_record_request_success_helper(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test record_request_success helper method."""
        health_monitor.record_request_success(
            provider_id="provider1",
            latency_ms=150,
        )

        # Verify recorded
        history = health_monitor._request_history["provider1"]
        assert len(history) == 1
        assert history[0].success is True

        # Verify circuit breaker updated
        cb = registry.get_circuit_breaker("provider1")
        assert cb is not None
        assert cb.consecutive_failures == 0
        assert cb.consecutive_successes == 1

    def test_record_request_failure_helper(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test record_request_failure helper method."""
        health_monitor.record_request_failure(
            provider_id="provider1",
            latency_ms=500,
            error="Timeout error",
        )

        history = health_monitor._request_history["provider1"]
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].error == "Timeout error"

    def test_request_history_max_size(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test that request history has max size limit."""
        # Record more than maxlen (1000)
        for i in range(1100):
            health_monitor.record_request(
                provider_id="provider1",
                success=True,
                latency_ms=100,
            )

        # Should only keep last 1000
        history = health_monitor._request_history["provider1"]
        assert len(history) == 1000


class TestHealthMetricsCalculation:
    """Test health metrics calculation."""

    def test_calculate_metrics_no_history(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test metrics calculation with no request history."""
        metrics = health_monitor._calculate_health_metrics("provider1")

        assert metrics.status == ProviderStatus.HEALTHY
        assert metrics.success_rate == 1.0
        assert metrics.average_latency_ms is None
        assert metrics.error_count == 0
        assert metrics.total_requests == 0

    def test_calculate_metrics_all_success(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test metrics with all successful requests."""
        for _ in range(10):
            health_monitor.record_request(
                provider_id="provider1",
                success=True,
                latency_ms=100,
            )

        metrics = health_monitor._calculate_health_metrics("provider1")

        assert metrics.status == ProviderStatus.HEALTHY
        assert metrics.success_rate == 1.0
        assert metrics.average_latency_ms == 100
        assert metrics.error_count == 0
        assert metrics.total_requests == 10
        assert metrics.consecutive_failures == 0

    def test_calculate_metrics_with_failures(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test metrics with mixed success and failures."""
        # 7 successes, 3 failures
        for _ in range(7):
            health_monitor.record_request(
                provider_id="provider1",
                success=True,
                latency_ms=100,
            )

        for _ in range(3):
            health_monitor.record_request(
                provider_id="provider1",
                success=False,
                latency_ms=500,
                error="Error",
            )

        metrics = health_monitor._calculate_health_metrics("provider1")

        assert metrics.total_requests == 10
        assert metrics.success_rate == 0.7
        assert metrics.error_count == 3
        # Average latency only counts successes
        assert metrics.average_latency_ms == 100
        # Last 3 were failures
        assert metrics.consecutive_failures == 3

    def test_calculate_metrics_consecutive_failures(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test calculation of consecutive failures."""
        # 5 successes
        for _ in range(5):
            health_monitor.record_request(
                provider_id="provider1",
                success=True,
                latency_ms=100,
            )

        # Then 4 failures
        for _ in range(4):
            health_monitor.record_request(
                provider_id="provider1",
                success=False,
                latency_ms=500,
                error="Error",
            )

        metrics = health_monitor._calculate_health_metrics("provider1")

        assert metrics.consecutive_failures == 4

    def test_calculate_metrics_last_error(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test that last error is captured."""
        health_monitor.record_request(
            provider_id="provider1",
            success=False,
            latency_ms=100,
            error="First error",
        )

        health_monitor.record_request(
            provider_id="provider1",
            success=False,
            latency_ms=100,
            error="Second error",
        )

        metrics = health_monitor._calculate_health_metrics("provider1")

        assert metrics.last_error == "Second error"


class TestStatusDetermination:
    """Test health status determination logic."""

    def test_determine_status_healthy(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test status determination for healthy provider."""
        status = health_monitor._determine_status(
            success_rate=0.99,
            consecutive_failures=0,
            average_latency_ms=100,
        )

        assert status == ProviderStatus.HEALTHY

    def test_determine_status_degraded_success_rate(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test degraded status due to low success rate."""
        status = health_monitor._determine_status(
            success_rate=0.85,
            consecutive_failures=0,
            average_latency_ms=100,
        )

        assert status == ProviderStatus.DEGRADED

    def test_determine_status_degraded_consecutive_failures(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test degraded status due to consecutive failures."""
        status = health_monitor._determine_status(
            success_rate=0.95,
            consecutive_failures=3,
            average_latency_ms=100,
        )

        assert status == ProviderStatus.DEGRADED

    def test_determine_status_degraded_high_latency(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test degraded status due to high latency."""
        status = health_monitor._determine_status(
            success_rate=0.99,
            consecutive_failures=0,
            average_latency_ms=6000,  # >5 seconds
        )

        assert status == ProviderStatus.DEGRADED

    def test_determine_status_unhealthy_low_success_rate(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test unhealthy status due to very low success rate."""
        status = health_monitor._determine_status(
            success_rate=0.4,
            consecutive_failures=0,
            average_latency_ms=100,
        )

        assert status == ProviderStatus.UNHEALTHY

    def test_determine_status_unhealthy_many_failures(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test unhealthy status due to many consecutive failures."""
        status = health_monitor._determine_status(
            success_rate=0.9,
            consecutive_failures=5,
            average_latency_ms=100,
        )

        assert status == ProviderStatus.UNHEALTHY


class TestProviderAvailability:
    """Test provider availability checks."""

    def test_is_provider_available_enabled_healthy(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that enabled healthy provider is available."""
        provider = registry.get_provider("provider1")
        if provider:
            provider.health = ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
            )

        assert health_monitor.is_provider_available("provider1") is True

    def test_is_provider_available_degraded(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that degraded provider is still available."""
        provider = registry.get_provider("provider1")
        if provider:
            provider.health = ProviderHealthMetrics(
                status=ProviderStatus.DEGRADED,
                last_check=datetime.now(),
            )

        assert health_monitor.is_provider_available("provider1") is True

    def test_is_provider_available_disabled(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that disabled provider is not available."""
        provider = registry.get_provider("provider1")
        if provider:
            provider.enabled = False

        assert health_monitor.is_provider_available("provider1") is False

    def test_is_provider_available_not_found(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test that non-existent provider is not available."""
        assert health_monitor.is_provider_available("nonexistent") is False

    def test_is_provider_available_circuit_open(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that provider with open circuit is not available."""
        cb = registry.get_circuit_breaker("provider1")
        if cb:
            cb.state = CircuitBreakerState.OPEN

        assert health_monitor.is_provider_available("provider1") is False


class TestCircuitBreakerManagement:
    """Test circuit breaker state management."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that circuit breaker opens after threshold failures."""
        cb = registry.get_circuit_breaker("provider2")
        assert cb is not None
        assert cb.config.failure_threshold == 3

        # Record failures up to threshold
        for _ in range(3):
            health_monitor.record_request_failure(
                provider_id="provider2",
                latency_ms=500,
                error="Error",
            )

            # Give asyncio time to process
            await asyncio.sleep(0.01)

        # Circuit should have tracked failures
        assert cb.consecutive_failures >= 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test circuit breaker transition from OPEN to HALF_OPEN."""
        cb = registry.get_circuit_breaker("provider1")
        assert cb is not None

        # Manually open circuit
        cb.state = CircuitBreakerState.OPEN
        cb.opened_at = datetime.now() - timedelta(seconds=40)  # Past timeout

        metrics = ProviderHealthMetrics(
            status=ProviderStatus.DEGRADED,
            last_check=datetime.now(),
        )

        await health_monitor._update_circuit_breaker("provider1", metrics)

        # Should transition to HALF_OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_successes(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test circuit breaker closes after threshold successes in HALF_OPEN."""
        cb = registry.get_circuit_breaker("provider1")
        assert cb is not None

        # Set to HALF_OPEN
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.consecutive_successes = 2  # At threshold

        metrics = ProviderHealthMetrics(
            status=ProviderStatus.HEALTHY,
            last_check=datetime.now(),
        )

        await health_monitor._update_circuit_breaker("provider1", metrics)

        # Should close circuit
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_handle_request_failure_opens_circuit(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that request failures can open circuit."""
        cb = registry.get_circuit_breaker("provider2")
        assert cb is not None
        assert cb.config.failure_threshold == 3

        # Simulate failures
        for _ in range(3):
            await health_monitor._handle_request_failure("provider2", cb)

        # Circuit should open
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_handle_request_failure_in_half_open_reopens(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test that failure in HALF_OPEN reopens circuit."""
        cb = registry.get_circuit_breaker("provider1")
        assert cb is not None

        # Set to HALF_OPEN
        cb.state = CircuitBreakerState.HALF_OPEN

        # Single failure should reopen
        await health_monitor._handle_request_failure("provider1", cb)

        assert cb.state == CircuitBreakerState.OPEN


class TestGetProviderMetrics:
    """Test getting provider metrics."""

    def test_get_provider_metrics_exists(
        self, health_monitor: ProviderHealthMonitor, registry: ProviderRegistry
    ) -> None:
        """Test getting metrics for provider with health data."""
        provider = registry.get_provider("provider1")
        if provider:
            provider.health = ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=datetime.now(),
                success_rate=0.99,
            )

        metrics = health_monitor.get_provider_metrics("provider1")

        assert metrics is not None
        assert metrics.success_rate == 0.99

    def test_get_provider_metrics_not_found(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test getting metrics for non-existent provider."""
        metrics = health_monitor.get_provider_metrics("nonexistent")
        assert metrics is None

    def test_get_provider_metrics_no_health(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test getting metrics for provider without health data."""
        metrics = health_monitor.get_provider_metrics("provider1")
        assert metrics is None


@pytest.mark.asyncio
class TestBackgroundMonitoring:
    """Test background monitoring loop."""

    async def test_start_monitoring(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test starting background monitoring."""
        assert health_monitor._running is False
        assert health_monitor._monitoring_task is None

        await health_monitor.start()

        assert health_monitor._running is True
        assert health_monitor._monitoring_task is not None

        await health_monitor.stop()

    async def test_stop_monitoring(self, health_monitor: ProviderHealthMonitor) -> None:
        """Test stopping background monitoring."""
        await health_monitor.start()
        assert health_monitor._running is True

        await health_monitor.stop()

        assert health_monitor._running is False

    async def test_start_when_already_running(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test starting when already running does nothing."""
        await health_monitor.start()
        task1 = health_monitor._monitoring_task

        await health_monitor.start()
        task2 = health_monitor._monitoring_task

        # Should be same task
        assert task1 is task2

        await health_monitor.stop()

    async def test_stop_when_not_running(
        self, health_monitor: ProviderHealthMonitor
    ) -> None:
        """Test stopping when not running does nothing."""
        assert health_monitor._running is False

        # Should not raise
        await health_monitor.stop()

        assert health_monitor._running is False


class TestRequestRecord:
    """Test RequestRecord class."""

    def test_request_record_creation(self) -> None:
        """Test creating a request record."""
        timestamp = datetime.now()
        record = RequestRecord(
            timestamp=timestamp,
            success=True,
            latency_ms=150,
            error=None,
        )

        assert record.timestamp == timestamp
        assert record.success is True
        assert record.latency_ms == 150
        assert record.error is None

    def test_request_record_with_error(self) -> None:
        """Test creating a request record with error."""
        timestamp = datetime.now()
        record = RequestRecord(
            timestamp=timestamp,
            success=False,
            latency_ms=500,
            error="Connection timeout",
        )

        assert record.success is False
        assert record.error == "Connection timeout"
