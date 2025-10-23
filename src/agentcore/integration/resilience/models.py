"""Resilience pattern data models.

Pydantic models for circuit breaker, bulkhead, and timeout configurations.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CircuitBreakerState(str, Enum):
    """Circuit breaker state enumeration.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Failing, requests are rejected immediately
        HALF_OPEN: Testing recovery, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration.

    Controls failure detection, timeout, and recovery behavior.
    """

    name: str = Field(
        ...,
        description="Circuit breaker name",
    )

    failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Number of failures before opening circuit",
    )

    success_threshold: int = Field(
        default=2,
        ge=1,
        description="Number of successes in HALF_OPEN to close circuit",
    )

    timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Seconds to wait in OPEN state before HALF_OPEN",
    )

    half_open_max_requests: int = Field(
        default=3,
        ge=1,
        description="Maximum requests allowed in HALF_OPEN state",
    )

    model_config = {"frozen": True}


class CircuitBreakerMetrics(BaseModel):
    """Circuit breaker runtime metrics.

    Tracks state transitions, success/failure counts, and timing.
    """

    name: str = Field(
        ...,
        description="Circuit breaker name",
    )

    state: CircuitBreakerState = Field(
        ...,
        description="Current circuit state",
    )

    failure_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive failures in current state",
    )

    success_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive successes in current state",
    )

    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total requests processed",
    )

    total_failures: int = Field(
        default=0,
        ge=0,
        description="Total failures across all states",
    )

    total_successes: int = Field(
        default=0,
        ge=0,
        description="Total successes across all states",
    )

    total_rejections: int = Field(
        default=0,
        ge=0,
        description="Total requests rejected (OPEN state)",
    )

    last_failure_time: datetime | None = Field(
        default=None,
        description="Timestamp of last failure",
    )

    last_state_change: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last state transition",
    )

    opened_at: datetime | None = Field(
        default=None,
        description="Timestamp when circuit opened",
    )


class BulkheadConfig(BaseModel):
    """Bulkhead pattern configuration.

    Controls resource isolation and concurrent request limits.
    """

    name: str = Field(
        ...,
        description="Bulkhead name",
    )

    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent requests allowed",
    )

    queue_size: int = Field(
        default=10,
        ge=0,
        description="Maximum queued requests when at capacity",
    )

    queue_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Maximum time to wait in queue",
    )

    model_config = {"frozen": True}


class TimeoutConfig(BaseModel):
    """Timeout configuration.

    Controls operation timeout behavior.
    """

    name: str = Field(
        ...,
        description="Timeout name",
    )

    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Operation timeout in seconds",
    )

    connect_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Connection timeout (if different from operation timeout)",
    )

    read_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Read timeout (if different from operation timeout)",
    )

    model_config = {"frozen": True}


class ResilienceConfig(BaseModel):
    """Combined resilience configuration.

    Bundles circuit breaker, bulkhead, and timeout configs.
    """

    circuit_breaker: CircuitBreakerConfig | None = Field(
        default=None,
        description="Circuit breaker configuration",
    )

    bulkhead: BulkheadConfig | None = Field(
        default=None,
        description="Bulkhead configuration",
    )

    timeout: TimeoutConfig | None = Field(
        default=None,
        description="Timeout configuration",
    )

    enable_fallback: bool = Field(
        default=False,
        description="Enable fallback handlers on failure",
    )

    model_config = {"frozen": True}
