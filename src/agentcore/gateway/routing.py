"""
Backend Service Routing

Intelligent routing and proxying to backend services with health monitoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from agentcore.gateway.health_monitor import HealthMonitor
from agentcore.gateway.load_balancer import LoadBalancer
from agentcore.gateway.models import (
    HealthCheckConfig,
    LoadBalancingAlgorithm,
    ServiceEndpoint,
)
from agentcore.gateway.service_discovery import ServiceRegistry

logger = structlog.get_logger(__name__)


class BackendRouter:
    """Routes and proxies requests to backend services."""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        load_balancer: LoadBalancer | None = None,
        health_monitor: HealthMonitor | None = None,
    ) -> None:
        """Initialize backend router.

        Args:
            registry: Service registry
            load_balancer: Load balancer
            health_monitor: Health monitor
        """
        self.registry = registry or ServiceRegistry()
        self.load_balancer = load_balancer or LoadBalancer()
        self.health_monitor = health_monitor or HealthMonitor()

    async def register_backend(
        self, service: ServiceEndpoint, start_monitoring: bool = True
    ) -> None:
        """Register a backend service.

        Args:
            service: Service endpoint configuration
            start_monitoring: Whether to start health monitoring
        """
        await self.registry.register(service)

        if start_monitoring:
            await self.health_monitor.start_monitoring(service)

        logger.info(
            "backend_registered",
            service_id=service.service_id,
            monitoring=start_monitoring,
        )

    async def unregister_backend(self, service_id: str) -> None:
        """Unregister a backend service.

        Args:
            service_id: Service identifier
        """
        await self.health_monitor.stop_monitoring(service_id)
        await self.registry.unregister(service_id)

        logger.info("backend_unregistered", service_id=service_id)

    async def route_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        content: bytes | None = None,
        service_filter: list[str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Route request to a backend service.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            params: Query parameters
            json_data: JSON request body
            content: Raw request body
            service_filter: Optional list of service IDs to consider

        Returns:
            Tuple of (status_code, headers, body)

        Raises:
            RuntimeError: If no healthy services available
        """
        # Get available services
        services = await self.registry.list_services(enabled_only=True)

        if service_filter:
            services = [s for s in services if s.service_id in service_filter]

        # Filter to healthy services
        healthy_services = []
        for service in services:
            if await self.health_monitor.is_healthy(service.service_id):
                healthy_services.append(service)

        if not healthy_services:
            logger.error("no_healthy_services", total_services=len(services))
            raise RuntimeError("No healthy backend services available")

        # Select service using load balancer
        selected_service = await self.load_balancer.select_service(healthy_services)
        if not selected_service:
            raise RuntimeError("Load balancer failed to select service")

        # Proxy request
        return await self._proxy_request(
            selected_service,
            method,
            path,
            headers,
            params,
            json_data,
            content,
        )

    async def _proxy_request(
        self,
        service: ServiceEndpoint,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        content: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Proxy request to backend service.

        Args:
            service: Target service
            method: HTTP method
            path: Request path
            headers: Request headers
            params: Query parameters
            json_data: JSON request body
            content: Raw request body

        Returns:
            Tuple of (status_code, headers, body)
        """
        service_id = service.service_id
        url = f"{service.base_url}{path}"

        # Remove hop-by-hop headers
        if headers:
            headers = self._filter_headers(headers)

        start_time = datetime.now(timezone.utc)
        await self.load_balancer.record_request_start(service_id)

        try:
            async with httpx.AsyncClient(
                timeout=service.timeout_seconds,
                follow_redirects=False,
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    content=content,
                )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            await self.load_balancer.record_request_end(
                service_id, success=True, response_time_ms=response_time
            )

            logger.info(
                "request_proxied",
                service_id=service_id,
                method=method,
                path=path,
                status_code=response.status_code,
                response_time_ms=response_time,
            )

            return (
                response.status_code,
                dict(response.headers),
                response.content,
            )

        except httpx.TimeoutException as e:
            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            await self.load_balancer.record_request_end(
                service_id, success=False, response_time_ms=response_time
            )

            logger.error(
                "request_timeout",
                service_id=service_id,
                method=method,
                path=path,
                timeout=service.timeout_seconds,
            )
            raise RuntimeError(f"Backend service timeout: {e}") from e

        except httpx.RequestError as e:
            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            await self.load_balancer.record_request_end(
                service_id, success=False, response_time_ms=response_time
            )

            logger.error(
                "request_error",
                service_id=service_id,
                method=method,
                path=path,
                error=str(e),
            )
            raise RuntimeError(f"Backend service error: {e}") from e

    def _filter_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Filter out hop-by-hop headers.

        Args:
            headers: Original headers

        Returns:
            Filtered headers
        """
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }

        return {
            k: v for k, v in headers.items() if k.lower() not in hop_by_hop
        }

    async def get_service_health(self) -> dict[str, Any]:
        """Get health status of all backend services.

        Returns:
            Dictionary with health information
        """
        all_health = await self.health_monitor.get_all_health()
        all_metrics = await self.load_balancer.get_all_metrics()

        return {
            "services": [
                {
                    "service_id": service_id,
                    "status": health.status.value,
                    "response_time_ms": health.response_time_ms,
                    "error_message": health.error_message,
                    "last_check": health.last_check.isoformat(),
                    "metrics": (
                        {
                            "total_requests": metrics.total_requests,
                            "successful_requests": metrics.successful_requests,
                            "failed_requests": metrics.failed_requests,
                            "active_connections": metrics.active_connections,
                            "avg_response_time_ms": metrics.avg_response_time_ms,
                        }
                        if (metrics := all_metrics.get(service_id))
                        else None
                    ),
                }
                for service_id, health in all_health.items()
            ]
        }

    async def shutdown(self) -> None:
        """Shutdown router and cleanup resources."""
        await self.health_monitor.shutdown()
        logger.info("backend_router_shutdown")


# Export classes at module level
__all__ = ["BackendRouter", "HealthMonitor", "LoadBalancer"]
