"""Tests for service discovery and registry."""

import pytest
from pydantic import HttpUrl

from agentcore.gateway.models import ServiceEndpoint
from agentcore.gateway.service_discovery import ServiceRegistry


@pytest.fixture
def service_registry() -> ServiceRegistry:
    """Create a service registry for testing."""
    return ServiceRegistry()


@pytest.fixture
def sample_service() -> ServiceEndpoint:
    """Create a sample service endpoint."""
    return ServiceEndpoint(
        service_id="test-service-1",
        name="Test Service",
        base_url=HttpUrl("http://localhost:8001"),
        priority=100,
        weight=1)


@pytest.mark.asyncio
async def test_register_service(
    service_registry: ServiceRegistry, sample_service: ServiceEndpoint
) -> None:
    """Test service registration."""
    await service_registry.register(sample_service)

    retrieved = await service_registry.get_service("test-service-1")
    assert retrieved is not None
    assert retrieved.service_id == "test-service-1"
    assert retrieved.name == "Test Service"


@pytest.mark.asyncio
async def test_unregister_service(
    service_registry: ServiceRegistry, sample_service: ServiceEndpoint
) -> None:
    """Test service unregistration."""
    await service_registry.register(sample_service)
    await service_registry.unregister("test-service-1")

    retrieved = await service_registry.get_service("test-service-1")
    assert retrieved is None


@pytest.mark.asyncio
async def test_list_services(service_registry: ServiceRegistry) -> None:
    """Test listing services."""
    # Register multiple services
    for i in range(3):
        service = ServiceEndpoint(
            service_id=f"service-{i}",
            name=f"Service {i}",
            base_url=HttpUrl(f"http://localhost:800{i}"))
        await service_registry.register(service)

    services = await service_registry.list_services()
    assert len(services) == 3


@pytest.mark.asyncio
async def test_list_enabled_only(service_registry: ServiceRegistry) -> None:
    """Test listing only enabled services."""
    # Register enabled and disabled services
    enabled = ServiceEndpoint(
        service_id="enabled",
        name="Enabled",
        base_url=HttpUrl("http://localhost:8001"),
        enabled=True)
    disabled = ServiceEndpoint(
        service_id="disabled",
        name="Disabled",
        base_url=HttpUrl("http://localhost:8002"),
        enabled=False)

    await service_registry.register(enabled)
    await service_registry.register(disabled)

    services = await service_registry.list_services(enabled_only=True)
    assert len(services) == 1
    assert services[0].service_id == "enabled"

    all_services = await service_registry.list_services(enabled_only=False)
    assert len(all_services) == 2


@pytest.mark.asyncio
async def test_update_service(
    service_registry: ServiceRegistry, sample_service: ServiceEndpoint
) -> None:
    """Test updating service configuration."""
    await service_registry.register(sample_service)

    updated = await service_registry.update_service(
        "test-service-1", {"priority": 200, "weight": 5}
    )

    assert updated is not None
    assert updated.priority == 200
    assert updated.weight == 5


@pytest.mark.asyncio
async def test_update_nonexistent_service(
    service_registry: ServiceRegistry) -> None:
    """Test updating nonexistent service."""
    result = await service_registry.update_service("nonexistent", {"priority": 100})
    assert result is None


@pytest.mark.asyncio
async def test_get_services_by_name(service_registry: ServiceRegistry) -> None:
    """Test getting services by name pattern."""
    services_data = [
        ("svc-1", "API Service"),
        ("svc-2", "API Gateway"),
        ("svc-3", "Database Service"),
    ]

    for service_id, name in services_data:
        service = ServiceEndpoint(
            service_id=service_id,
            name=name,
            base_url=HttpUrl("http://localhost:8000"))
        await service_registry.register(service)

    api_services = await service_registry.get_services_by_name("api")
    assert len(api_services) == 2

    db_services = await service_registry.get_services_by_name("database")
    assert len(db_services) == 1


@pytest.mark.asyncio
async def test_registry_length(service_registry: ServiceRegistry) -> None:
    """Test registry length tracking."""
    assert len(service_registry) == 0

    service = ServiceEndpoint(
        service_id="test",
        name="Test",
        base_url=HttpUrl("http://localhost:8000"))
    await service_registry.register(service)

    assert len(service_registry) == 1
