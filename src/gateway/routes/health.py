"""
Health Check Endpoints

System health and readiness endpoints for monitoring and load balancing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from gateway.config import settings
from gateway.models.health import (
    HealthCheckDetail,
    HealthResponse,
    LivenessResponse,
    MetricsInfo,
    ReadinessResponse,
)

router = APIRouter()
logger = structlog.get_logger()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Comprehensive health check endpoint with metrics.

    Returns the service status, version information, and dependency health.
    Used by monitoring systems to verify service and dependencies are healthy.
    """
    # Use health checker if available
    if hasattr(request.app.state, "health_checker"):
        result = await request.app.state.health_checker.check_all()

        checks = {
            name: HealthCheckDetail(
                status=check["status"],
                details=check["message"]
            )
            for name, check in result["checks"].items()
        }

        return HealthResponse(
            status=result["status"],
            version=settings.GATEWAY_VERSION,
            timestamp=datetime.now(UTC).isoformat(),
            checks=checks,
        )

    # Fallback to basic health check
    return HealthResponse(
        status="healthy",
        version=settings.GATEWAY_VERSION,
        timestamp=datetime.now(UTC).isoformat(),
        checks={
            "application": HealthCheckDetail(
                status="healthy", details="Gateway application is running"
            )
        },
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(request: Request) -> ReadinessResponse:
    """
    Readiness check endpoint with dependency validation.

    Verifies that the service is ready to accept requests.
    Checks connectivity to critical dependencies (Redis, backend services).
    """
    # Use health checker if available
    if hasattr(request.app.state, "health_checker"):
        result = await request.app.state.health_checker.check_readiness()

        if result["status"] != "ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("reason", "Service not ready")
            )

        return ReadinessResponse(
            status=result["status"],
            ready=True,
            checks={"critical_dependencies": True}
        )

    # Fallback to basic readiness check
    checks: dict[str, bool] = {}
    all_ready = True

    # Check application readiness
    try:
        checks["application"] = True
        logger.debug("Application readiness check passed")
    except Exception as e:
        checks["application"] = False
        all_ready = False
        logger.error("Application readiness check failed", error=str(e))

    # Backend services placeholder
    checks["backend_services"] = True

    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready"
        )

    return ReadinessResponse(status="ready", ready=all_ready, checks=checks)


@router.get("/live", response_model=LivenessResponse)
async def liveness_check() -> LivenessResponse:
    """
    Liveness check endpoint.

    Simple endpoint that returns 200 OK if the service is alive.
    Used by Kubernetes and other orchestrators for liveness probes.
    """
    return LivenessResponse(status="alive")


@router.get("/metrics-info", response_model=MetricsInfo)
async def metrics_info() -> MetricsInfo:
    """
    Metrics information endpoint.

    Returns information about the metrics endpoint.
    """
    return MetricsInfo(endpoint="/metrics", format="prometheus")
