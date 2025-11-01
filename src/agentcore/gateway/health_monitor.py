"""
Health Monitoring for Backend Services

Monitors backend service health with configurable checks.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import structlog

from agentcore.gateway.models import (
    HealthCheckConfig,
    ServiceEndpoint,
    ServiceHealth,
    ServiceStatus,
)

logger = structlog.get_logger(__name__)


class HealthMonitor:
    """Monitor backend service health with periodic checks."""

    def __init__(self, config: HealthCheckConfig | None = None) -> None:
        """Initialize health monitor.

        Args:
            config: Health check configuration
        """
        self.config = config or HealthCheckConfig()
        self._health_status: dict[str, ServiceHealth] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._running = False

    async def start_monitoring(self, service: ServiceEndpoint) -> None:
        """Start monitoring a service.

        Args:
            service: Service endpoint to monitor
        """
        if service.service_id in self._monitor_tasks:
            logger.warning(
                "monitor_already_running", service_id=service.service_id
            )
            return

        # Initialize health status
        self._health_status[service.service_id] = ServiceHealth(
            service_id=service.service_id,
            status=ServiceStatus.UNKNOWN,
            last_check=datetime.now(UTC),
        )

        # Start monitoring task
        task = asyncio.create_task(self._monitor_service(service))
        self._monitor_tasks[service.service_id] = task

        logger.info("monitor_started", service_id=service.service_id)

    async def stop_monitoring(self, service_id: str) -> None:
        """Stop monitoring a service.

        Args:
            service_id: Service identifier
        """
        task = self._monitor_tasks.pop(service_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("monitor_stopped", service_id=service_id)

    async def get_health(self, service_id: str) -> ServiceHealth | None:
        """Get current health status for a service.

        Args:
            service_id: Service identifier

        Returns:
            Service health status or None if not monitored
        """
        return self._health_status.get(service_id)

    async def get_all_health(self) -> dict[str, ServiceHealth]:
        """Get health status for all monitored services.

        Returns:
            Dictionary mapping service IDs to health status
        """
        return self._health_status.copy()

    async def is_healthy(self, service_id: str) -> bool:
        """Check if service is healthy.

        Args:
            service_id: Service identifier

        Returns:
            True if service is healthy
        """
        health = self._health_status.get(service_id)
        return health is not None and health.status == ServiceStatus.HEALTHY

    async def _monitor_service(self, service: ServiceEndpoint) -> None:
        """Monitor service health in a loop.

        Args:
            service: Service endpoint to monitor
        """
        while True:
            try:
                await self._check_health(service)
                await asyncio.sleep(self.config.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "monitor_error", service_id=service.service_id
                )
                await asyncio.sleep(self.config.interval_seconds)

    async def _check_health(self, service: ServiceEndpoint) -> None:
        """Perform health check for a service.

        Args:
            service: Service endpoint to check
        """
        service_id = service.service_id
        check_url = f"{service.base_url}{self.config.endpoint}"

        start_time = datetime.now(UTC)

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.get(check_url)

            response_time = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            if response.status_code == 200:
                await self._update_health_success(service_id, response_time)
            else:
                await self._update_health_failure(
                    service_id, f"HTTP {response.status_code}"
                )

        except httpx.TimeoutException:
            await self._update_health_failure(service_id, "Health check timeout")
        except httpx.ConnectError:
            await self._update_health_failure(service_id, "Connection failed")
        except Exception as e:
            await self._update_health_failure(service_id, str(e))

    async def _update_health_success(
        self, service_id: str, response_time_ms: float
    ) -> None:
        """Update health status after successful check.

        Args:
            service_id: Service identifier
            response_time_ms: Response time in milliseconds
        """
        async with self._lock:
            health = self._health_status.get(service_id)
            if not health:
                return

            health.consecutive_successes += 1
            health.consecutive_failures = 0
            health.response_time_ms = response_time_ms
            health.error_message = None
            health.last_check = datetime.now(UTC)

            # Update status based on threshold
            if health.consecutive_successes >= self.config.healthy_threshold:
                previous_status = health.status
                health.status = ServiceStatus.HEALTHY
                if previous_status != ServiceStatus.HEALTHY:
                    logger.info(
                        "service_healthy",
                        service_id=service_id,
                        response_time_ms=response_time_ms,
                    )

    async def _update_health_failure(
        self, service_id: str, error_message: str
    ) -> None:
        """Update health status after failed check.

        Args:
            service_id: Service identifier
            error_message: Error description
        """
        async with self._lock:
            health = self._health_status.get(service_id)
            if not health:
                return

            health.consecutive_failures += 1
            health.consecutive_successes = 0
            health.error_message = error_message
            health.last_check = datetime.now(UTC)

            # Update status based on threshold
            if health.consecutive_failures >= self.config.unhealthy_threshold:
                previous_status = health.status
                health.status = ServiceStatus.UNHEALTHY
                if previous_status != ServiceStatus.UNHEALTHY:
                    logger.warning(
                        "service_unhealthy",
                        service_id=service_id,
                        error=error_message,
                        consecutive_failures=health.consecutive_failures,
                    )

    async def shutdown(self) -> None:
        """Shutdown all health monitoring tasks."""
        service_ids = list(self._monitor_tasks.keys())
        for service_id in service_ids:
            await self.stop_monitoring(service_id)

        logger.info("health_monitor_shutdown")
