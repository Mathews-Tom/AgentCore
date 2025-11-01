"""Tests for health monitoring."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import HttpUrl

from agentcore.gateway.health_monitor import HealthMonitor
from agentcore.gateway.models import HealthCheckConfig, ServiceEndpoint, ServiceHealth, ServiceStatus


@pytest.fixture
def health_config() -> HealthCheckConfig:
    """Create health check configuration for testing."""
    return HealthCheckConfig(
        endpoint="/health",
        interval_seconds=1,
        timeout_seconds=2.0,
        healthy_threshold=2,
        unhealthy_threshold=2)


@pytest.fixture
def sample_service() -> ServiceEndpoint:
    """Create sample service for testing."""
    return ServiceEndpoint(
        service_id="test-svc",
        name="Test Service",
        base_url=HttpUrl("http://localhost:8001"))


@pytest.mark.asyncio
async def test_is_healthy() -> None:
    """Test is_healthy check."""
    monitor = HealthMonitor()

    # No health data yet
    assert not await monitor.is_healthy("test-svc")

    # Add health data manually for testing
    monitor._health_status["test-svc"] = ServiceHealth(
        service_id="test-svc",
        status=ServiceStatus.HEALTHY,
        last_check=datetime.now(UTC))

    assert await monitor.is_healthy("test-svc")


@pytest.mark.asyncio
async def test_get_all_health() -> None:
    """Test getting all health statuses."""
    monitor = HealthMonitor()

    # Add multiple services
    for i in range(3):
        monitor._health_status[f"svc-{i}"] = ServiceHealth(
            service_id=f"svc-{i}",
            status=ServiceStatus.HEALTHY,
            last_check=datetime.now(UTC))

    all_health = await monitor.get_all_health()
    assert len(all_health) == 3


@pytest.mark.asyncio
async def test_start_stop_monitoring(sample_service: ServiceEndpoint) -> None:
    """Test starting and stopping monitoring."""
    monitor = HealthMonitor()

    # Start monitoring
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200)

        await monitor.start_monitoring(sample_service)
        assert "test-svc" in monitor._monitor_tasks
        assert "test-svc" in monitor._health_status

        # Stop monitoring
        await monitor.stop_monitoring("test-svc")
        assert "test-svc" not in monitor._monitor_tasks


@pytest.mark.asyncio
async def test_shutdown() -> None:
    """Test monitor shutdown."""
    monitor = HealthMonitor()

    # Add services
    for i in range(2):
        service_id = f"svc-{i}"
        monitor._health_status[service_id] = ServiceHealth(
            service_id=service_id,
            status=ServiceStatus.HEALTHY,
            last_check=datetime.now(UTC))
        # Create dummy task
        monitor._monitor_tasks[service_id] = asyncio.create_task(asyncio.sleep(100))

    await monitor.shutdown()
    assert len(monitor._monitor_tasks) == 0


@pytest.mark.asyncio
async def test_update_health_success() -> None:
    """Test updating health status after successful check."""
    monitor = HealthMonitor()

    # Initialize health status
    monitor._health_status["test-svc"] = ServiceHealth(
        service_id="test-svc",
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.now(UTC))

    # Update with successful checks
    for _ in range(3):
        await monitor._update_health_success("test-svc", 50.0)

    health = monitor._health_status["test-svc"]
    assert health.status == ServiceStatus.HEALTHY
    assert health.consecutive_successes == 3
    assert health.consecutive_failures == 0
    assert health.response_time_ms == 50.0


@pytest.mark.asyncio
async def test_update_health_failure() -> None:
    """Test updating health status after failed check."""
    monitor = HealthMonitor()

    # Initialize health status
    monitor._health_status["test-svc"] = ServiceHealth(
        service_id="test-svc",
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.now(UTC))

    # Update with failed checks
    for _ in range(3):
        await monitor._update_health_failure("test-svc", "Connection failed")

    health = monitor._health_status["test-svc"]
    assert health.status == ServiceStatus.UNHEALTHY
    assert health.consecutive_failures == 3
    assert health.consecutive_successes == 0
    assert "Connection failed" in (health.error_message or "")
