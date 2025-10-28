"""Tests for backend router integration."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import HttpUrl

from agentcore.gateway.models import ServiceEndpoint
from agentcore.gateway.routing import BackendRouter


@pytest.fixture
def backend_router() -> BackendRouter:
    """Create backend router for testing."""
    return BackendRouter()


@pytest.fixture
def sample_service() -> ServiceEndpoint:
    """Create sample service."""
    return ServiceEndpoint(
        service_id="backend-1",
        name="Backend Service 1",
        base_url=HttpUrl("http://localhost:8001"))


@pytest.mark.asyncio
async def test_register_backend(
    backend_router: BackendRouter, sample_service: ServiceEndpoint
) -> None:
    """Test registering backend service."""
    await backend_router.register_backend(sample_service, start_monitoring=False)

    registered = await backend_router.registry.get_service("backend-1")
    assert registered is not None
    assert registered.service_id == "backend-1"


@pytest.mark.asyncio
async def test_unregister_backend(
    backend_router: BackendRouter, sample_service: ServiceEndpoint
) -> None:
    """Test unregistering backend service."""
    await backend_router.register_backend(sample_service, start_monitoring=False)
    await backend_router.unregister_backend("backend-1")

    registered = await backend_router.registry.get_service("backend-1")
    assert registered is None


@pytest.mark.asyncio
async def test_route_request_no_services(backend_router: BackendRouter) -> None:
    """Test routing when no services available."""
    with pytest.raises(RuntimeError, match="No healthy backend services"):
        await backend_router.route_request(
            method="GET",
            path="/test")


@pytest.mark.asyncio
async def test_get_service_health(backend_router: BackendRouter) -> None:
    """Test getting service health status."""
    health = await backend_router.get_service_health()
    assert "services" in health
    assert len(health["services"]) == 0


@pytest.mark.asyncio
async def test_shutdown(backend_router: BackendRouter) -> None:
    """Test router shutdown."""
    await backend_router.shutdown()
    # Should not raise errors


@pytest.mark.asyncio
async def test_proxy_request(
    backend_router: BackendRouter, sample_service: ServiceEndpoint
) -> None:
    """Test proxying request to backend."""
    await backend_router.register_backend(sample_service, start_monitoring=False)

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_response = httpx.Response(
            200,
            json={"result": "success"},
            headers={"content-type": "application/json"})
        mock_request.return_value = mock_response

        status, headers, body = await backend_router._proxy_request(
            service=sample_service,
            method="POST",
            path="/api/test",
            json_data={"test": "data"})

        assert status == 200
        assert b"success" in body
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_filter_headers(backend_router: BackendRouter) -> None:
    """Test filtering hop-by-hop headers."""
    headers = {
        "Authorization": "Bearer token",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=5",
    }

    filtered = backend_router._filter_headers(headers)

    assert "Authorization" in filtered
    assert "Content-Type" in filtered
    assert "Connection" not in filtered
    assert "Keep-Alive" not in filtered
