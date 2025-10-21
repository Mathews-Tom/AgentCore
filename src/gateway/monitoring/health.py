"""
Health Check Endpoint with Metrics

Provides comprehensive health checks for the gateway and its dependencies,
including database, Redis, backend services, and resource utilization.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Health check metrics
HEALTH_STATUS = Gauge(
    "gateway_health_status",
    "Overall gateway health status (1=healthy, 0=unhealthy)",
)

COMPONENT_HEALTH = Gauge(
    "gateway_component_health_status",
    "Component health status (1=healthy, 0=unhealthy)",
    ["component"],
)

HEALTH_CHECK_DURATION = Gauge(
    "gateway_health_check_duration_seconds",
    "Duration of health check in seconds",
    ["component"],
)

# Readiness metrics
READINESS_STATUS = Gauge(
    "gateway_readiness_status",
    "Gateway readiness status (1=ready, 0=not ready)",
)


class HealthChecker:
    """
    Health checker for gateway and dependencies.

    Performs health checks on various components and tracks their status
    using Prometheus metrics.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        backend_services: dict[str, str] | None = None,
        check_timeout: float = 5.0,
    ) -> None:
        """
        Initialize health checker.

        Args:
            redis_url: Redis connection URL
            backend_services: Dictionary of backend service names to URLs
            check_timeout: Timeout for health checks in seconds
        """
        self.redis_url = redis_url
        self.backend_services = backend_services or {}
        self.check_timeout = check_timeout
        self._http_client = httpx.AsyncClient(timeout=check_timeout)

    async def check_redis(self) -> tuple[bool, str]:
        """
        Check Redis connectivity.

        Returns:
            Tuple of (is_healthy, status_message)
        """
        if not self.redis_url:
            return True, "Redis not configured"

        start_time = time.time()
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(self.redis_url)
            await asyncio.wait_for(redis_client.ping(), timeout=self.check_timeout)
            await redis_client.aclose()

            duration = time.time() - start_time
            COMPONENT_HEALTH.labels(component="redis").set(1)
            HEALTH_CHECK_DURATION.labels(component="redis").set(duration)

            return True, "Redis is healthy"

        except TimeoutError:
            duration = time.time() - start_time
            COMPONENT_HEALTH.labels(component="redis").set(0)
            HEALTH_CHECK_DURATION.labels(component="redis").set(duration)
            return False, "Redis health check timed out"

        except Exception as e:
            duration = time.time() - start_time
            COMPONENT_HEALTH.labels(component="redis").set(0)
            HEALTH_CHECK_DURATION.labels(component="redis").set(duration)
            logger.error(f"Redis health check failed: {e}")
            return False, f"Redis error: {str(e).lower()}"

    async def check_backend_service(
        self, service_name: str, service_url: str
    ) -> tuple[bool, str]:
        """
        Check backend service health.

        Args:
            service_name: Name of the service
            service_url: Health check URL

        Returns:
            Tuple of (is_healthy, status_message)
        """
        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                self._http_client.get(f"{service_url}/health"),
                timeout=self.check_timeout,
            )

            duration = time.time() - start_time
            is_healthy = response.status_code == 200

            COMPONENT_HEALTH.labels(component=service_name).set(1 if is_healthy else 0)
            HEALTH_CHECK_DURATION.labels(component=service_name).set(duration)

            if is_healthy:
                return True, f"{service_name} is healthy"
            return False, f"{service_name} returned status {response.status_code}"

        except TimeoutError:
            duration = time.time() - start_time
            COMPONENT_HEALTH.labels(component=service_name).set(0)
            HEALTH_CHECK_DURATION.labels(component=service_name).set(duration)
            return False, f"{service_name} health check timed out"

        except Exception as e:
            duration = time.time() - start_time
            COMPONENT_HEALTH.labels(component=service_name).set(0)
            HEALTH_CHECK_DURATION.labels(component=service_name).set(duration)
            logger.error(f"{service_name} health check failed: {e}")
            return False, f"{service_name} error: {str(e)}"

    async def check_all(self) -> dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Dictionary containing health check results
        """
        start_time = time.time()
        results: dict[str, Any] = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {},
        }

        # Check Redis
        redis_healthy, redis_message = await self.check_redis()
        results["checks"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "message": redis_message,
        }

        # Check backend services
        backend_checks = [
            self.check_backend_service(name, url)
            for name, url in self.backend_services.items()
        ]

        if backend_checks:
            backend_results = await asyncio.gather(
                *backend_checks, return_exceptions=True
            )

            for (name, _), result in zip(
                self.backend_services.items(), backend_results, strict=False
            ):
                if isinstance(result, Exception):
                    results["checks"][name] = {
                        "status": "unhealthy",
                        "message": f"Check failed: {str(result)}",
                    }
                    COMPONENT_HEALTH.labels(component=name).set(0)
                else:
                    is_healthy, message = result
                    results["checks"][name] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "message": message,
                    }

        # Determine overall health
        all_healthy = all(
            check["status"] == "healthy" for check in results["checks"].values()
        )

        if not all_healthy:
            results["status"] = "unhealthy"

        # Update metrics
        HEALTH_STATUS.set(1 if all_healthy else 0)
        results["duration_seconds"] = time.time() - start_time

        return results

    async def check_readiness(self) -> dict[str, Any]:
        """
        Check if gateway is ready to serve traffic.

        A simpler check than full health check, focusing on critical dependencies.

        Returns:
            Dictionary containing readiness status
        """
        results: dict[str, Any] = {
            "status": "ready",
            "timestamp": time.time(),
        }

        # Check Redis (critical for rate limiting)
        if self.redis_url:
            redis_healthy, _ = await self.check_redis()
            if not redis_healthy:
                results["status"] = "not_ready"
                results["reason"] = "Redis unavailable"
                READINESS_STATUS.set(0)
                return results

        READINESS_STATUS.set(1)
        return results

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self._http_client.aclose()
