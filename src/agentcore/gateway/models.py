"""
Gateway Layer Models

Data models for service routing, health monitoring, and load balancing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ServiceStatus(str, Enum):
    """Backend service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithm types."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"


class ServiceEndpoint(BaseModel):
    """Backend service endpoint configuration."""

    service_id: str = Field(..., description="Unique service identifier")
    name: str = Field(..., description="Service name")
    base_url: HttpUrl = Field(..., description="Base URL for the service")
    priority: int = Field(default=100, ge=1, le=1000, description="Routing priority")
    weight: int = Field(default=1, ge=1, le=100, description="Load balancing weight")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Request timeout")
    max_connections: int = Field(
        default=100, ge=1, description="Maximum concurrent connections"
    )
    enabled: bool = Field(default=True, description="Whether service is enabled")


class HealthCheckConfig(BaseModel):
    """Health check configuration for backend services."""

    endpoint: str = Field(default="/api/v1/health", description="Health check endpoint")
    interval_seconds: int = Field(
        default=10, ge=1, description="Health check interval"
    )
    timeout_seconds: float = Field(default=5.0, gt=0, description="Health check timeout")
    healthy_threshold: int = Field(
        default=2, ge=1, description="Consecutive successes for healthy"
    )
    unhealthy_threshold: int = Field(
        default=3, ge=1, description="Consecutive failures for unhealthy"
    )


class ServiceHealth(BaseModel):
    """Current health status of a backend service."""

    service_id: str
    status: ServiceStatus
    response_time_ms: float | None = None
    error_message: str | None = None
    last_check: datetime
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingMetrics(BaseModel):
    """Metrics for service routing and load balancing."""

    service_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    active_connections: int = 0
    avg_response_time_ms: float = 0.0
    last_request: datetime | None = None
