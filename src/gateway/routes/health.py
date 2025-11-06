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


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Comprehensive health check",
    description="""
Check overall system health including dependencies and components.

**Health Status:**
- `healthy`: All systems operational
- `degraded`: Some non-critical services down
- `unhealthy`: Critical services unavailable

**Components Checked:**
- Redis: Session storage and rate limiting
- Backend Services: A2A Protocol, Agent Runtime
- JWT Manager: Token signing and validation
- Session Manager: Session lifecycle management

**Use Cases:**
- Monitoring system health checks
- Load balancer health probes
- Operational dashboards
- Alerting and incident response

**Monitoring Integration:**
```bash
# Prometheus format
curl http://localhost:8080/metrics

# JSON health check
curl http://localhost:8080/health

# Parse with jq
curl -s http://localhost:8080/health | jq '.status'
```

**No authentication required** - public endpoint for monitoring.
    """,
    responses={
        200: {
            "description": "Health check successful",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "0.1.0",
                        "timestamp": "2025-10-18T10:30:00Z",
                        "checks": {
                            "redis": {"status": "healthy", "details": "Connected"},
                            "jwt": {"status": "healthy", "details": "Keys loaded"},
                        },
                    }
                }
            },
        },
    },
)
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
            name: HealthCheckDetail(status=check["status"], details=check["message"])
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


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Service readiness check",
    description="""
Check if service is ready to accept traffic.

**Difference from /health:**
- `/health`: Overall system health (may be unhealthy but recovering)
- `/ready`: Ready to serve requests right now

**Kubernetes Integration:**
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 3
```

**Use Cases:**
- Load balancer routing decisions
- Kubernetes readiness probes
- Rolling deployment health gates
- Traffic shifting during deployments

**Returns 503 if not ready** - prevents routing traffic to unhealthy instances.

**No authentication required** - public endpoint for orchestration.
    """,
    responses={
        200: {
            "description": "Service is ready",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ready",
                        "ready": True,
                        "checks": {
                            "redis": True,
                            "backend_services": True,
                        },
                    }
                }
            },
        },
        503: {
            "description": "Service not ready",
            "content": {
                "application/json": {
                    "example": {"detail": "Service not ready"}
                }
            },
        },
    },
)
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
                detail=result.get("reason", "Service not ready"),
            )

        return ReadinessResponse(
            status=result["status"], ready=True, checks={"critical_dependencies": True}
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


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Service liveness check",
    description="""
Simple liveness probe - is the process running?

**Purpose:**
- Detect deadlocks and infinite loops
- Restart crashed or frozen instances
- Basic process health verification

**Kubernetes Integration:**
```yaml
livenessProbe:
  httpGet:
    path: /live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3
```

**Always returns 200 OK** unless process is completely dead.

**No authentication required** - public endpoint for orchestration.
    """,
    responses={
        200: {
            "description": "Service is alive",
            "content": {
                "application/json": {"example": {"status": "alive"}}
            },
        },
    },
)
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
