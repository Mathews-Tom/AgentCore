"""
Service Proxy

Integrates load balancing, circuit breakers, and service discovery for
intelligent request routing to backend services.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Request, Response

from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
)
from .discovery import ServiceDiscovery
from .load_balancer import LoadBalancer, LoadBalancingAlgorithm

logger = structlog.get_logger()


class ServiceProxy:
    """
    Service proxy with load balancing and circuit breaker protection.

    Handles routing requests to backend services with:
    - Load balancing across service instances
    - Circuit breaker pattern for resilience
    - Automatic failover and retry
    """

    def __init__(
        self,
        discovery: ServiceDiscovery,
        load_balancing_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        request_timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """
        Initialize service proxy.

        Args:
            discovery: Service discovery instance
            load_balancing_algorithm: Algorithm for load balancing
            circuit_breaker_config: Circuit breaker configuration
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.discovery = discovery
        self.load_balancer = LoadBalancer(discovery, load_balancing_algorithm)
        self.circuit_breakers = CircuitBreakerRegistry(circuit_breaker_config)
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._http_client = httpx.AsyncClient(timeout=request_timeout)

    async def proxy_request(
        self,
        service_name: str,
        path: str,
        method: str = "GET",
        request: Request | None = None,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
    ) -> Response:
        """
        Proxy request to backend service with load balancing and circuit breaker.

        Args:
            service_name: Target service name
            path: Request path
            method: HTTP method
            request: Original FastAPI request (for extracting client IP)
            headers: Custom headers to forward
            data: Request body data

        Returns:
            FastAPI Response with backend response data

        Raises:
            HTTPException: If request fails after retries
        """
        client_ip = None
        if request:
            client_ip = request.client.host if request.client else None

        last_error = None
        attempt = 0

        while attempt <= self.max_retries:
            try:
                # Select backend instance
                instance = self.load_balancer.select_instance(service_name, client_ip)

                if not instance:
                    logger.error(
                        "No healthy instances available",
                        service=service_name,
                        attempt=attempt,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"Service {service_name} unavailable",
                    )

                # Get circuit breaker for this instance
                breaker = self.circuit_breakers.get(instance.instance_id)

                # Track connection for least connections algorithm
                self.load_balancer.increment_connections(instance.instance_id)

                try:
                    # Make request through circuit breaker
                    response = await breaker.call_async(
                        self._make_request,
                        instance.url,
                        path,
                        method,
                        headers,
                        data,
                    )

                    logger.info(
                        "Request proxied successfully",
                        service=service_name,
                        instance=instance.instance_id,
                        path=path,
                        status_code=response.status_code,
                        attempt=attempt,
                    )

                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.headers.get("content-type"),
                    )

                finally:
                    # Decrement connection count
                    self.load_balancer.decrement_connections(instance.instance_id)

            except CircuitBreakerOpenError as e:
                logger.warning(
                    "Circuit breaker is open",
                    service=service_name,
                    error=str(e),
                    attempt=attempt,
                )
                last_error = e

                # Try next instance on circuit breaker open
                attempt += 1
                continue

            except httpx.HTTPError as e:
                logger.warning(
                    "HTTP error during proxy request",
                    service=service_name,
                    error=str(e),
                    attempt=attempt,
                )
                last_error = e

                # Retry on network errors
                attempt += 1
                continue

            except Exception as e:
                logger.error(
                    "Unexpected error during proxy request",
                    service=service_name,
                    error=str(e),
                    attempt=attempt,
                )
                last_error = e

                # Don't retry on unexpected errors
                break

        # All retries exhausted
        error_msg = str(last_error) if last_error else "Unknown error"
        logger.error(
            "Request failed after all retries",
            service=service_name,
            path=path,
            max_retries=self.max_retries,
            error=error_msg,
        )

        raise HTTPException(
            status_code=503,
            detail=f"Service {service_name} request failed: {error_msg}",
        )

    async def _make_request(
        self,
        base_url: str,
        path: str,
        method: str,
        headers: dict[str, str] | None,
        data: bytes | None,
    ) -> httpx.Response:
        """
        Make HTTP request to backend service.

        Args:
            base_url: Base URL of service instance
            path: Request path
            method: HTTP method
            headers: Request headers
            data: Request body data

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: On request failure
        """
        url = f"{base_url}{path}"

        response = await self._http_client.request(
            method=method,
            url=url,
            headers=headers,
            content=data,
        )

        # Raise for 5xx errors (circuit breaker will catch)
        response.raise_for_status()

        return response

    async def get_service_status(self, service_name: str) -> dict[str, Any]:
        """
        Get status of service including instances and circuit breakers.

        Args:
            service_name: Service name

        Returns:
            Service status information
        """
        instances = self.discovery.get_instances(service_name, healthy_only=False)

        instance_stats = []
        for instance in instances:
            breaker = self.circuit_breakers.get(instance.instance_id)
            instance_stats.append(
                {
                    "instance_id": instance.instance_id,
                    "url": instance.url,
                    "status": instance.status,
                    "circuit_breaker": breaker.get_stats(),
                }
            )

        return {
            "service_name": service_name,
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == "healthy"]),
            "load_balancing_algorithm": self.load_balancer.algorithm,
            "instances": instance_stats,
        }

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self._http_client.aclose()
