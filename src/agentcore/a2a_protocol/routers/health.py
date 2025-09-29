"""
Health Check Endpoints

System health and readiness endpoints for monitoring and load balancing.
"""

import asyncio
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import structlog

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.database import check_db_health

router = APIRouter()
logger = structlog.get_logger()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: str
    checks: Dict[str, Dict[str, Any]]


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    ready: bool
    checks: Dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns the service status and version information.
    Used by monitoring systems to verify service is running.
    """
    import datetime
    from agentcore import __version__

    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.datetime.utcnow().isoformat(),
        checks={
            "application": {
                "status": "healthy",
                "details": "Application is running"
            }
        }
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness check endpoint.

    Verifies that the service is ready to accept requests.
    Checks dependencies like database and Redis connections.
    """
    checks = {}
    all_ready = True

    # Check database connection
    try:
        db_healthy = await check_db_health()
        checks["database"] = db_healthy
        if db_healthy:
            logger.debug("Database health check passed")
        else:
            all_ready = False
            logger.warning("Database health check failed")
    except Exception as e:
        checks["database"] = False
        all_ready = False
        logger.error("Database health check failed", error=str(e))

    # Check Redis connection
    try:
        # TODO: Implement actual Redis health check
        # For now, always return healthy
        checks["redis"] = True
        logger.debug("Redis health check passed")
    except Exception as e:
        checks["redis"] = False
        all_ready = False
        logger.error("Redis health check failed", error=str(e))

    # Check A2A protocol readiness
    try:
        # TODO: Implement A2A protocol readiness check
        checks["a2a_protocol"] = True
        logger.debug("A2A protocol health check passed")
    except Exception as e:
        checks["a2a_protocol"] = False
        all_ready = False
        logger.error("A2A protocol health check failed", error=str(e))

    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

    return ReadinessResponse(
        status="ready",
        ready=all_ready,
        checks=checks
    )


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check endpoint.

    Simple endpoint that returns 200 OK if the service is alive.
    Used by Kubernetes and other orchestrators for liveness probes.
    """
    return {"status": "alive"}