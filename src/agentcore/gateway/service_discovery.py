"""
Service Discovery and Registry

Manages backend service registration and discovery.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from agentcore.gateway.models import ServiceEndpoint, ServiceStatus

logger = structlog.get_logger(__name__)


class ServiceRegistry:
    """Registry for backend services."""

    def __init__(self) -> None:
        """Initialize service registry."""
        self._services: dict[str, ServiceEndpoint] = {}
        self._lock = asyncio.Lock()

    async def register(self, service: ServiceEndpoint) -> None:
        """Register a backend service.

        Args:
            service: Service endpoint configuration
        """
        async with self._lock:
            self._services[service.service_id] = service
            logger.info(
                "service_registered",
                service_id=service.service_id,
                name=service.name,
                base_url=str(service.base_url),
            )

    async def unregister(self, service_id: str) -> None:
        """Unregister a backend service.

        Args:
            service_id: Service identifier
        """
        async with self._lock:
            if service_id in self._services:
                service = self._services.pop(service_id)
                logger.info(
                    "service_unregistered", service_id=service_id, name=service.name
                )
            else:
                logger.warning("service_not_found", service_id=service_id)

    async def get_service(self, service_id: str) -> ServiceEndpoint | None:
        """Get service by ID.

        Args:
            service_id: Service identifier

        Returns:
            Service endpoint or None if not found
        """
        return self._services.get(service_id)

    async def list_services(
        self, enabled_only: bool = True
    ) -> list[ServiceEndpoint]:
        """List all registered services.

        Args:
            enabled_only: Only return enabled services

        Returns:
            List of service endpoints
        """
        services = list(self._services.values())
        if enabled_only:
            return [s for s in services if s.enabled]
        return services

    async def update_service(
        self, service_id: str, updates: dict[str, Any]
    ) -> ServiceEndpoint | None:
        """Update service configuration.

        Args:
            service_id: Service identifier
            updates: Dictionary of fields to update

        Returns:
            Updated service endpoint or None if not found
        """
        async with self._lock:
            service = self._services.get(service_id)
            if not service:
                logger.warning("service_not_found", service_id=service_id)
                return None

            updated_data = service.model_dump()
            updated_data.update(updates)
            updated_service = ServiceEndpoint(**updated_data)
            self._services[service_id] = updated_service

            logger.info("service_updated", service_id=service_id, updates=updates)
            return updated_service

    async def get_services_by_name(self, name: str) -> list[ServiceEndpoint]:
        """Get services by name pattern.

        Args:
            name: Service name to match

        Returns:
            List of matching services
        """
        return [s for s in self._services.values() if name.lower() in s.name.lower()]

    def __len__(self) -> int:
        """Get number of registered services."""
        return len(self._services)
