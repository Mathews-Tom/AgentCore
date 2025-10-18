"""
Service Discovery

Backend service registration and discovery system.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ServiceStatus(str, Enum):
    """Service instance status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceInstance:
    """Backend service instance."""

    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: datetime | None = None
    failure_count: int = 0
    weight: int = 1

    @property
    def url(self) -> str:
        """Get service instance URL."""
        return f"{self.protocol}://{self.host}:{self.port}"

    def mark_healthy(self) -> None:
        """Mark instance as healthy."""
        self.status = ServiceStatus.HEALTHY
        self.last_health_check = datetime.now()
        self.failure_count = 0

    def mark_unhealthy(self) -> None:
        """Mark instance as unhealthy."""
        self.status = ServiceStatus.UNHEALTHY
        self.last_health_check = datetime.now()
        self.failure_count += 1


class ServiceDiscovery:
    """
    Service discovery and registry.

    Manages backend service instances and health monitoring.
    """

    def __init__(
        self,
        health_check_interval: int = 10,
        health_check_timeout: int = 5,
        max_failures: int = 3,
    ):
        """
        Initialize service discovery.

        Args:
            health_check_interval: Health check interval in seconds
            health_check_timeout: Health check timeout in seconds
            max_failures: Maximum failures before marking unhealthy
        """
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_failures = max_failures

        self._services: dict[str, list[ServiceInstance]] = {}
        self._health_check_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start service discovery."""
        if self._running:
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service discovery started")

    async def stop(self) -> None:
        """Stop service discovery."""
        if not self._running:
            return

        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Service discovery stopped")

    def register(self, instance: ServiceInstance) -> None:
        """
        Register a service instance.

        Args:
            instance: Service instance to register
        """
        if instance.service_name not in self._services:
            self._services[instance.service_name] = []

        # Check if instance already registered
        existing = next(
            (i for i in self._services[instance.service_name] if i.instance_id == instance.instance_id),
            None,
        )

        if existing:
            # Update existing instance
            idx = self._services[instance.service_name].index(existing)
            self._services[instance.service_name][idx] = instance
        else:
            # Add new instance
            self._services[instance.service_name].append(instance)

        logger.info(
            "Service instance registered",
            service=instance.service_name,
            instance_id=instance.instance_id,
            url=instance.url,
        )

    def deregister(self, service_name: str, instance_id: str) -> None:
        """
        Deregister a service instance.

        Args:
            service_name: Service name
            instance_id: Instance ID
        """
        if service_name not in self._services:
            return

        self._services[service_name] = [
            i for i in self._services[service_name] if i.instance_id != instance_id
        ]

        logger.info(
            "Service instance deregistered",
            service=service_name,
            instance_id=instance_id,
        )

    def get_instances(self, service_name: str, healthy_only: bool = True) -> list[ServiceInstance]:
        """
        Get all instances of a service.

        Args:
            service_name: Service name
            healthy_only: Only return healthy instances

        Returns:
            List of service instances
        """
        instances = self._services.get(service_name, [])

        if healthy_only:
            instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]

        return instances

    def get_instance(self, service_name: str, instance_id: str) -> ServiceInstance | None:
        """
        Get a specific service instance.

        Args:
            service_name: Service name
            instance_id: Instance ID

        Returns:
            Service instance or None
        """
        instances = self._services.get(service_name, [])
        return next((i for i in instances if i.instance_id == instance_id), None)

    def get_services(self) -> list[str]:
        """Get all registered service names."""
        return list(self._services.keys())

    async def _health_check_loop(self) -> None:
        """Periodically check health of all service instances."""
        import httpx

        while self._running:
            try:
                for service_name, instances in self._services.items():
                    for instance in instances:
                        try:
                            # Perform health check
                            async with httpx.AsyncClient(timeout=self.health_check_timeout) as client:
                                response = await client.get(f"{instance.url}/health")

                                if response.status_code == 200:
                                    instance.mark_healthy()
                                else:
                                    instance.mark_unhealthy()

                        except Exception as e:
                            logger.warning(
                                "Health check failed",
                                service=service_name,
                                instance_id=instance.instance_id,
                                error=str(e),
                            )
                            instance.mark_unhealthy()

                            # Remove if too many failures
                            if instance.failure_count >= self.max_failures:
                                logger.error(
                                    "Service instance removed due to repeated failures",
                                    service=service_name,
                                    instance_id=instance.instance_id,
                                    failures=instance.failure_count,
                                )
                                self.deregister(service_name, instance.instance_id)

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(self.health_check_interval)
