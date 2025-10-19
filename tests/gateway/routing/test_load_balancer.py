"""Tests for load balancing algorithms."""

import pytest
from pydantic import HttpUrl

from agentcore.gateway.load_balancer import (
    LeastConnectionsStrategy,
    LoadBalancer,
    RandomStrategy,
    RoundRobinStrategy,
    WeightedRoundRobinStrategy,
)
from agentcore.gateway.models import LoadBalancingAlgorithm, RoutingMetrics, ServiceEndpoint


@pytest.fixture
def sample_services() -> list[ServiceEndpoint]:
    """Create sample services for testing."""
    return [
        ServiceEndpoint(
            service_id="svc-1",
            name="Service 1",
            base_url=HttpUrl("http://localhost:8001"),
            weight=1,
        ),
        ServiceEndpoint(
            service_id="svc-2",
            name="Service 2",
            base_url=HttpUrl("http://localhost:8002"),
            weight=2,
        ),
        ServiceEndpoint(
            service_id="svc-3",
            name="Service 3",
            base_url=HttpUrl("http://localhost:8003"),
            weight=3,
        ),
    ]


@pytest.mark.asyncio
async def test_round_robin_strategy(sample_services: list[ServiceEndpoint]) -> None:
    """Test round-robin load balancing."""
    strategy = RoundRobinStrategy()
    metrics: dict[str, RoutingMetrics] = {}

    # Select services in round-robin order
    selections = []
    for _ in range(6):
        selected = await strategy.select_service(sample_services, metrics)
        assert selected is not None
        selections.append(selected.service_id)

    # Should cycle through services: svc-1, svc-2, svc-3, svc-1, svc-2, svc-3
    assert selections == ["svc-1", "svc-2", "svc-3", "svc-1", "svc-2", "svc-3"]


@pytest.mark.asyncio
async def test_least_connections_strategy(sample_services: list[ServiceEndpoint]) -> None:
    """Test least connections load balancing."""
    strategy = LeastConnectionsStrategy()
    metrics = {
        "svc-1": RoutingMetrics(service_id="svc-1", active_connections=5),
        "svc-2": RoutingMetrics(service_id="svc-2", active_connections=2),
        "svc-3": RoutingMetrics(service_id="svc-3", active_connections=8),
    }

    selected = await strategy.select_service(sample_services, metrics)
    assert selected is not None
    assert selected.service_id == "svc-2"  # Least connections


@pytest.mark.asyncio
async def test_random_strategy(sample_services: list[ServiceEndpoint]) -> None:
    """Test random load balancing."""
    strategy = RandomStrategy()
    metrics: dict[str, RoutingMetrics] = {}

    # Select multiple times and verify all services get selected
    selections = set()
    for _ in range(50):
        selected = await strategy.select_service(sample_services, metrics)
        assert selected is not None
        selections.add(selected.service_id)

    # With 50 selections, we should hit all 3 services
    assert len(selections) == 3


@pytest.mark.asyncio
async def test_weighted_round_robin_strategy(sample_services: list[ServiceEndpoint]) -> None:
    """Test weighted round-robin load balancing."""
    strategy = WeightedRoundRobinStrategy()
    metrics: dict[str, RoutingMetrics] = {}

    # Select services and count selections
    selections = []
    for _ in range(18):  # Total weight is 6, so 3 cycles
        selected = await strategy.select_service(sample_services, metrics)
        assert selected is not None
        selections.append(selected.service_id)

    # Count selections per service
    counts = {
        "svc-1": selections.count("svc-1"),
        "svc-2": selections.count("svc-2"),
        "svc-3": selections.count("svc-3"),
    }

    # Verify proportions match weights (1:2:3)
    # In 18 selections, expect: svc-1=3, svc-2=6, svc-3=9
    assert counts["svc-1"] == 3
    assert counts["svc-2"] == 6
    assert counts["svc-3"] == 9


@pytest.mark.asyncio
async def test_load_balancer_round_robin() -> None:
    """Test load balancer with round-robin algorithm."""
    lb = LoadBalancer(algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)

    services = [
        ServiceEndpoint(
            service_id=f"svc-{i}",
            name=f"Service {i}",
            base_url=HttpUrl(f"http://localhost:800{i}"),
        )
        for i in range(3)
    ]

    selections = []
    for _ in range(6):
        selected = await lb.select_service(services)
        assert selected is not None
        selections.append(selected.service_id)

    assert selections == ["svc-0", "svc-1", "svc-2", "svc-0", "svc-1", "svc-2"]


@pytest.mark.asyncio
async def test_load_balancer_filters_disabled() -> None:
    """Test load balancer filters disabled services."""
    lb = LoadBalancer(algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)

    services = [
        ServiceEndpoint(
            service_id="enabled",
            name="Enabled",
            base_url=HttpUrl("http://localhost:8001"),
            enabled=True,
        ),
        ServiceEndpoint(
            service_id="disabled",
            name="Disabled",
            base_url=HttpUrl("http://localhost:8002"),
            enabled=False,
        ),
    ]

    selected = await lb.select_service(services)
    assert selected is not None
    assert selected.service_id == "enabled"


@pytest.mark.asyncio
async def test_record_request_metrics() -> None:
    """Test recording request metrics."""
    lb = LoadBalancer()

    await lb.record_request_start("svc-1")
    metrics = await lb.get_metrics("svc-1")

    assert metrics is not None
    assert metrics.total_requests == 1
    assert metrics.active_connections == 1

    await lb.record_request_end("svc-1", success=True, response_time_ms=50.0)
    metrics = await lb.get_metrics("svc-1")

    assert metrics is not None
    assert metrics.successful_requests == 1
    assert metrics.failed_requests == 0
    assert metrics.active_connections == 0
    assert metrics.avg_response_time_ms == 50.0


@pytest.mark.asyncio
async def test_record_failed_request() -> None:
    """Test recording failed request metrics."""
    lb = LoadBalancer()

    await lb.record_request_start("svc-1")
    await lb.record_request_end("svc-1", success=False, response_time_ms=100.0)

    metrics = await lb.get_metrics("svc-1")
    assert metrics is not None
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 1


@pytest.mark.asyncio
async def test_reset_metrics() -> None:
    """Test resetting metrics."""
    lb = LoadBalancer()

    await lb.record_request_start("svc-1")
    await lb.record_request_end("svc-1", success=True, response_time_ms=50.0)

    await lb.reset_metrics("svc-1")
    metrics = await lb.get_metrics("svc-1")

    assert metrics is not None
    assert metrics.total_requests == 0
    assert metrics.successful_requests == 0


@pytest.mark.asyncio
async def test_get_all_metrics() -> None:
    """Test getting all metrics."""
    lb = LoadBalancer()

    for i in range(3):
        service_id = f"svc-{i}"
        await lb.record_request_start(service_id)
        await lb.record_request_end(service_id, success=True, response_time_ms=50.0)

    all_metrics = await lb.get_all_metrics()
    assert len(all_metrics) == 3
    assert "svc-0" in all_metrics
    assert "svc-1" in all_metrics
    assert "svc-2" in all_metrics


@pytest.mark.asyncio
async def test_empty_services() -> None:
    """Test handling empty service list."""
    lb = LoadBalancer()
    selected = await lb.select_service([])
    assert selected is None
