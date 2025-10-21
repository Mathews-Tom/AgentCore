"""
Health Check Models

Pydantic models for health and readiness endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthCheckDetail(BaseModel):
    """Individual health check detail."""

    status: str = Field(..., description="Health check status")
    details: str = Field(..., description="Additional details")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Gateway version")
    timestamp: str = Field(..., description="Check timestamp in ISO format")
    checks: dict[str, HealthCheckDetail] = Field(
        ..., description="Individual health checks"
    )


class ReadinessResponse(BaseModel):
    """Readiness check response model."""

    status: str = Field(..., description="Readiness status")
    ready: bool = Field(..., description="Whether service is ready")
    checks: dict[str, bool] = Field(..., description="Individual readiness checks")


class LivenessResponse(BaseModel):
    """Liveness check response model."""

    status: str = Field(..., description="Liveness status")


class MetricsInfo(BaseModel):
    """Metrics information model."""

    endpoint: str = Field(..., description="Metrics endpoint path")
    format: str = Field(..., description="Metrics format")
