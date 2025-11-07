"""
Tests for Service Proxy with Load Balancing and Circuit Breaker Integration

Covers request proxying, failover, circuit breaker integration, and retry logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request

from gateway.routing.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)
from gateway.routing.discovery import ServiceDiscovery, ServiceInstance, ServiceStatus
from gateway.routing.load_balancer import LoadBalancingAlgorithm
from gateway.routing.proxy import ServiceProxy


def create_mock_response(
    status_code: int = 200,
    content: bytes = b"ok",
    headers: dict[str, str] | None = None,
    url: str = "http://localhost:8000/api/test",
) -> httpx.Response:
    """Create a proper mock httpx.Response with request attached."""
    mock_request = httpx.Request("GET", url)
    return httpx.Response(
        status_code,
        content=content,
        headers=headers or {"content-type": "application/json"},
        request=mock_request,
    )


@pytest.fixture
def service_discovery() -> ServiceDiscovery:
    """Create service discovery for testing."""
    discovery = ServiceDiscovery(
        health_check_interval=60,  # Long interval for tests
        health_check_timeout=5,
        max_failures=3,
    )

    # Register test services
    for i in range(3):
        instance = ServiceInstance(
            service_name="test-service",
            instance_id=f"instance-{i}",
            host="localhost",
            port=8000 + i,
            protocol="http",
            status=ServiceStatus.HEALTHY,
        )
        discovery.register(instance)

    return discovery


@pytest.fixture
def circuit_config() -> CircuitBreakerConfig:
    """Create circuit breaker configuration for testing."""
    return CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=1,
        timeout=1.0,
    )


@pytest.fixture
def service_proxy(
    service_discovery: ServiceDiscovery, circuit_config: CircuitBreakerConfig
) -> ServiceProxy:
    """Create service proxy for testing."""
    return ServiceProxy(
        discovery=service_discovery,
        load_balancing_algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
        circuit_breaker_config=circuit_config,
        request_timeout=5.0,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_successful_proxy_request(service_proxy: ServiceProxy) -> None:
    """Test successful request proxying."""
    mock_response = create_mock_response(200, b"success")

    with patch.object(
        service_proxy._http_client, "request", new_callable=AsyncMock
    ) as mock_http_request:
        mock_http_request.return_value = mock_response

        response = await service_proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
            method="GET",
        )

        assert response.status_code == 200
        assert response.body == b"success"
        mock_http_request.assert_called_once()


@pytest.mark.asyncio
async def test_no_healthy_instances(service_discovery: ServiceDiscovery) -> None:
    """Test error when no healthy instances available."""
    # Mark all instances as unhealthy
    for instance in service_discovery._services["test-service"]:
        instance.status = ServiceStatus.UNHEALTHY

    proxy = ServiceProxy(discovery=service_discovery)

    with pytest.raises(HTTPException) as exc_info:
        await proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
        )

    assert exc_info.value.status_code == 503
    assert "unavailable" in exc_info.value.detail


@pytest.mark.asyncio
async def test_round_robin_distribution(service_proxy: ServiceProxy) -> None:
    """Test requests are distributed using round-robin."""
    selected_instances: list[str] = []

    async def mock_request(*args, **kwargs) -> httpx.Response:
        # Extract instance from URL
        url = kwargs.get("url", "")
        for i in range(3):
            if f":800{i}/" in url:
                selected_instances.append(f"instance-{i}")
                break
        return create_mock_response(200, b"ok", url=url)

    with patch.object(
        service_proxy._http_client, "request", side_effect=mock_request
    ):
        # Make 6 requests
        for _ in range(6):
            await service_proxy.proxy_request(
                service_name="test-service",
                path="/api/test",
            )

    # Should cycle through instances twice
    assert selected_instances == [
        "instance-0",
        "instance-1",
        "instance-2",
        "instance-0",
        "instance-1",
        "instance-2",
    ]


@pytest.mark.asyncio
async def test_retry_on_network_error(service_proxy: ServiceProxy) -> None:
    """Test retry logic on network errors."""
    call_count = 0

    async def mock_request_with_retry(*args, **kwargs) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        url = kwargs.get("url", "http://localhost:8000/api/test")
        if call_count < 2:
            raise httpx.ConnectError("Connection failed")
        return create_mock_response(200, b"success", url=url)

    with patch.object(
        service_proxy._http_client, "request", side_effect=mock_request_with_retry
    ):
        response = await service_proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
        )

        assert response.status_code == 200
        assert call_count == 2  # Failed once, succeeded on retry


@pytest.mark.asyncio
async def test_fail_after_max_retries(service_proxy: ServiceProxy) -> None:
    """Test failure after exhausting all retries."""

    async def always_fail(*args, **kwargs) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    with patch.object(service_proxy._http_client, "request", side_effect=always_fail):
        with pytest.raises(HTTPException) as exc_info:
            await service_proxy.proxy_request(
                service_name="test-service",
                path="/api/test",
            )

        assert exc_info.value.status_code == 503
        assert "failed" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_circuit_breaker_integration(service_proxy: ServiceProxy) -> None:
    """Test circuit breaker prevents cascading failures."""

    async def failing_request(*args, **kwargs) -> httpx.Response:
        raise httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=httpx.Response(500)
        )

    with patch.object(
        service_proxy._http_client, "request", side_effect=failing_request
    ):
        # First requests will fail normally
        for _ in range(2):
            with pytest.raises(HTTPException):
                await service_proxy.proxy_request(
                    service_name="test-service",
                    path="/api/test",
                )

        # Circuit should now be open for instance-0
        # Next request should try instance-1 (round robin)
        with pytest.raises(HTTPException):
            await service_proxy.proxy_request(
                service_name="test-service",
                path="/api/test",
            )


@pytest.mark.asyncio
async def test_failover_to_healthy_instance(
    service_proxy: ServiceProxy, service_discovery: ServiceDiscovery
) -> None:
    """Test failover to healthy instance when one fails."""
    call_count = 0

    async def selective_failure(*args, **kwargs) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        url = kwargs.get("url", "")

        # First instance fails
        if ":8000/" in url:
            raise httpx.ConnectError("Instance 0 down")

        # Others succeed
        return create_mock_response(200, b"success", url=url)

    with patch.object(
        service_proxy._http_client, "request", side_effect=selective_failure
    ):
        response = await service_proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
        )

        assert response.status_code == 200
        # Should have tried instance-0 first (failed), then instance-1 (succeeded)
        assert call_count == 2


@pytest.mark.asyncio
async def test_least_connections_strategy(service_discovery: ServiceDiscovery) -> None:
    """Test least connections load balancing."""
    proxy = ServiceProxy(
        discovery=service_discovery,
        load_balancing_algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
    )

    mock_response = create_mock_response(200, b"ok")

    with patch.object(proxy._http_client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        # Make multiple concurrent requests
        tasks = [
            proxy.proxy_request(
                service_name="test-service",
                path="/api/test",
            )
            for _ in range(5)
        ]

        import asyncio

        await asyncio.gather(*tasks)

        # All requests should succeed
        assert mock_request.call_count == 5


@pytest.mark.asyncio
async def test_custom_headers_forwarded(service_proxy: ServiceProxy) -> None:
    """Test custom headers are forwarded to backend."""
    mock_response = create_mock_response(200, b"ok")

    with patch.object(
        service_proxy._http_client, "request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_response

        custom_headers = {
            "X-Request-ID": "test-123",
            "Authorization": "Bearer token",
        }

        await service_proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
            headers=custom_headers,
        )

        # Verify headers were passed
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["X-Request-ID"] == "test-123"
        assert call_kwargs["headers"]["Authorization"] == "Bearer token"


@pytest.mark.asyncio
async def test_request_body_forwarded(service_proxy: ServiceProxy) -> None:
    """Test request body is forwarded to backend."""
    mock_response = create_mock_response(200, b"ok")
    request_body = b'{"key": "value"}'

    with patch.object(
        service_proxy._http_client, "request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_response

        await service_proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
            method="POST",
            data=request_body,
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["content"] == request_body


@pytest.mark.asyncio
async def test_client_ip_for_ip_hash(
    service_discovery: ServiceDiscovery, circuit_config: CircuitBreakerConfig
) -> None:
    """Test client IP is used for IP hash load balancing."""
    proxy = ServiceProxy(
        discovery=service_discovery,
        load_balancing_algorithm=LoadBalancingAlgorithm.IP_HASH,
        circuit_breaker_config=circuit_config,
    )

    mock_response = create_mock_response(200, b"ok")
    mock_request = MagicMock()
    mock_request.client.host = "192.168.1.100"

    selected_instances: list[str] = []

    async def track_instance(*args, **kwargs) -> httpx.Response:
        url = kwargs.get("url", "")
        for i in range(3):
            if f":800{i}/" in url:
                selected_instances.append(f"instance-{i}")
                break
        return create_mock_response(200, b"ok", url=url)

    with patch.object(proxy._http_client, "request", side_effect=track_instance):
        # Make multiple requests from same IP
        for _ in range(5):
            await proxy.proxy_request(
                service_name="test-service",
                path="/api/test",
                request=mock_request,
            )

    # All requests from same IP should go to same instance
    assert len(set(selected_instances)) == 1


@pytest.mark.asyncio
async def test_get_service_status(service_proxy: ServiceProxy) -> None:
    """Test getting service status information."""
    status = await service_proxy.get_service_status("test-service")

    assert status["service_name"] == "test-service"
    assert status["total_instances"] == 3
    assert status["healthy_instances"] == 3
    assert status["load_balancing_algorithm"] == LoadBalancingAlgorithm.ROUND_ROBIN
    assert len(status["instances"]) == 3


@pytest.mark.asyncio
async def test_connection_tracking(service_proxy: ServiceProxy) -> None:
    """Test connection count is tracked correctly."""
    mock_response = create_mock_response(200, b"ok")

    with patch.object(
        service_proxy._http_client, "request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_response

        # Track connection increments/decrements
        increment_calls: list[str] = []
        decrement_calls: list[str] = []

        original_increment = service_proxy.load_balancer.increment_connections
        original_decrement = service_proxy.load_balancer.decrement_connections

        def track_increment(instance_id: str) -> None:
            increment_calls.append(instance_id)
            original_increment(instance_id)

        def track_decrement(instance_id: str) -> None:
            decrement_calls.append(instance_id)
            original_decrement(instance_id)

        service_proxy.load_balancer.increment_connections = track_increment
        service_proxy.load_balancer.decrement_connections = track_decrement

        await service_proxy.proxy_request(
            service_name="test-service",
            path="/api/test",
        )

        # Connection should be incremented and then decremented
        assert len(increment_calls) == 1
        assert len(decrement_calls) == 1
        assert increment_calls[0] == decrement_calls[0]


@pytest.mark.asyncio
async def test_proxy_close(service_proxy: ServiceProxy) -> None:
    """Test proxy cleanup."""
    await service_proxy.close()
    # Verify HTTP client is closed
    assert service_proxy._http_client.is_closed
