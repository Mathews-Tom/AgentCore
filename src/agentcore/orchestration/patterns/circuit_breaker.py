"""
Circuit Breaker Pattern Implementation

Fault tolerance with circuit breakers, retry policies, and health monitoring.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.streams.producer import StreamProducer


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure threshold exceeded
    HALF_OPEN = "half_open"  # Testing recovery


class RetryStrategy(str, Enum):
    """Retry strategy types."""

    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    FIXED = "fixed"  # Fixed delay
    IMMEDIATE = "immediate"  # No delay


class HealthStatus(str, Enum):
    """Health status of monitored service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""

    # Failure thresholds
    failure_threshold: int = Field(default=5, description="Failures before opening")
    success_threshold: int = Field(
        default=2, description="Successes to close from half-open"
    )
    timeout_seconds: int = Field(default=60, description="Open state timeout")

    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL, description="Retry backoff strategy"
    )
    initial_retry_delay_seconds: float = Field(
        default=1.0, description="Initial retry delay"
    )
    max_retry_delay_seconds: float = Field(
        default=60.0, description="Maximum retry delay"
    )
    retry_multiplier: float = Field(default=2.0, description="Backoff multiplier")

    # Health monitoring
    health_check_interval_seconds: int = Field(
        default=30, description="Health check interval"
    )
    sliding_window_size: int = Field(
        default=100, description="Request window for metrics"
    )

    model_config = {"frozen": False}


class RetryPolicy(BaseModel):
    """
    Retry policy with exponential backoff.

    Manages retry attempts with configurable backoff strategies.
    """

    policy_id: UUID = Field(default_factory=uuid4, description="Policy identifier")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL, description="Backoff strategy"
    )
    initial_delay_seconds: float = Field(default=1.0, description="Initial delay")
    max_delay_seconds: float = Field(default=60.0, description="Maximum delay")
    multiplier: float = Field(default=2.0, description="Backoff multiplier")
    jitter: bool = Field(default=True, description="Add random jitter")

    current_attempt: int = Field(default=0, description="Current attempt number")
    last_attempt_at: datetime | None = None

    model_config = {"frozen": False}

    def calculate_delay(self) -> float:
        """Calculate next retry delay based on strategy."""
        if self.strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        if self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay_seconds
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay_seconds * (self.current_attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay_seconds * (
                self.multiplier**self.current_attempt
            )
        else:
            delay = self.initial_delay_seconds

        # Cap at max delay
        delay = min(delay, self.max_delay_seconds)

        # Add jitter
        if self.jitter:
            import random

            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def should_retry(self) -> bool:
        """Check if should retry based on attempt count."""
        return self.current_attempt < self.max_retries

    def record_attempt(self) -> None:
        """Record a retry attempt."""
        self.current_attempt += 1
        self.last_attempt_at = datetime.now(UTC)

    def reset(self) -> None:
        """Reset retry state."""
        self.current_attempt = 0
        self.last_attempt_at = None


class CircuitBreakerMetrics(BaseModel):
    """Metrics for circuit breaker monitoring."""

    total_requests: int = Field(default=0, description="Total requests")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    rejected_requests: int = Field(default=0, description="Rejected (open) requests")

    consecutive_failures: int = Field(default=0, description="Consecutive failures")
    consecutive_successes: int = Field(default=0, description="Consecutive successes")

    last_failure_at: datetime | None = None
    last_success_at: datetime | None = None
    state_changed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last state change"
    )

    # Sliding window for error rate
    recent_requests: list[bool] = Field(
        default_factory=list, description="Recent request outcomes (True=success)"
    )
    window_size: int = Field(default=100, description="Window size")

    model_config = {"frozen": False}

    def record_success(self) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_at = datetime.now(UTC)
        self._add_to_window(True)

    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_at = datetime.now(UTC)
        self._add_to_window(False)

    def record_rejection(self) -> None:
        """Record a rejected request (circuit open)."""
        self.rejected_requests += 1

    def _add_to_window(self, success: bool) -> None:
        """Add result to sliding window."""
        self.recent_requests.append(success)
        if len(self.recent_requests) > self.window_size:
            self.recent_requests.pop(0)

    def get_error_rate(self) -> float:
        """Calculate error rate from sliding window."""
        if not self.recent_requests:
            return 0.0
        failures = sum(1 for r in self.recent_requests if not r)
        return failures / len(self.recent_requests)

    def get_success_rate(self) -> float:
        """Calculate success rate from sliding window."""
        return 1.0 - self.get_error_rate()


class CircuitBreaker(BaseModel):
    """
    Circuit breaker for fault tolerance.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests rejected
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    breaker_id: UUID = Field(default_factory=uuid4, description="Breaker identifier")
    service_name: str = Field(description="Service being protected")
    config: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig, description="Breaker configuration"
    )

    state: CircuitState = Field(
        default=CircuitState.CLOSED, description="Current state"
    )
    metrics: CircuitBreakerMetrics = Field(
        default_factory=CircuitBreakerMetrics, description="Breaker metrics"
    )

    opened_at: datetime | None = None
    half_opened_at: datetime | None = None
    closed_at: datetime | None = None

    model_config = {"frozen": False}

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit allows request
        if not await self.can_execute():
            self.metrics.record_rejection()
            raise RuntimeError(f"Circuit breaker open for {self.service_name}")

        # Execute with error handling
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise

    async def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if self.opened_at and datetime.now(UTC) > self.opened_at + timedelta(
                seconds=self.config.timeout_seconds
            ):
                await self._transition_to_half_open()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open
            return True

        return False

    async def record_success(self) -> None:
        """Record successful execution."""
        self.metrics.record_success()

        if self.state == CircuitState.HALF_OPEN:
            # Check if should close
            if (
                self.metrics.consecutive_successes
                >= self.config.success_threshold
            ):
                await self._transition_to_closed()

    async def record_failure(self) -> None:
        """Record failed execution."""
        self.metrics.record_failure()

        if self.state == CircuitState.CLOSED:
            # Check if should open
            if (
                self.metrics.consecutive_failures
                >= self.config.failure_threshold
            ):
                await self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens circuit
            await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now(UTC)
        self.metrics.state_changed_at = self.opened_at

    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.half_opened_at = datetime.now(UTC)
        self.metrics.state_changed_at = self.half_opened_at
        self.metrics.consecutive_successes = 0

    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self.state = CircuitState.CLOSED
        self.closed_at = datetime.now(UTC)
        self.metrics.state_changed_at = self.closed_at
        self.metrics.consecutive_failures = 0

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes = 0
        self.closed_at = datetime.now(UTC)

    def get_status(self) -> dict[str, Any]:
        """Get current breaker status."""
        return {
            "breaker_id": str(self.breaker_id),
            "service_name": self.service_name,
            "state": self.state,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "rejected_requests": self.metrics.rejected_requests,
                "error_rate": self.metrics.get_error_rate(),
                "success_rate": self.metrics.get_success_rate(),
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
            },
            "state_info": {
                "opened_at": self.opened_at.isoformat() if self.opened_at else None,
                "half_opened_at": self.half_opened_at.isoformat()
                if self.half_opened_at
                else None,
                "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            },
        }


class HealthCheck(BaseModel):
    """Health check result for a service."""

    service_name: str = Field(description="Service being monitored")
    status: HealthStatus = Field(description="Health status")
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Check timestamp"
    )

    response_time_ms: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class HealthMonitor(BaseModel):
    """
    Health monitoring for services with circuit breakers.

    Tracks service health and enables automatic recovery.
    """

    monitor_id: UUID = Field(default_factory=uuid4, description="Monitor identifier")
    service_name: str = Field(description="Service being monitored")
    check_interval_seconds: int = Field(
        default=30, description="Health check interval"
    )

    current_status: HealthStatus = Field(
        default=HealthStatus.UNKNOWN, description="Current health status"
    )
    last_check: HealthCheck | None = None
    check_history: list[HealthCheck] = Field(
        default_factory=list, description="Recent health checks"
    )
    max_history_size: int = Field(default=100, description="Max history entries")

    # Monitoring state
    is_monitoring: bool = Field(default=False, description="Monitoring active")
    _monitor_task: asyncio.Task | None = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    async def start_monitoring(
        self,
        health_check_func: Callable[[], Any],
    ) -> None:
        """
        Start health monitoring.

        Args:
            health_check_func: Async function to check health
        """
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(health_check_func)
        )

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(
        self,
        health_check_func: Callable[[], Any],
    ) -> None:
        """Internal monitoring loop."""
        while self.is_monitoring:
            try:
                await self.check_health(health_check_func)
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring despite errors
                await asyncio.sleep(self.check_interval_seconds)

    async def check_health(
        self,
        health_check_func: Callable[[], Any],
    ) -> HealthCheck:
        """
        Perform health check.

        Args:
            health_check_func: Async function to check health

        Returns:
            HealthCheck result
        """
        start_time = datetime.now(UTC)

        try:
            # Execute health check
            result = await asyncio.wait_for(
                health_check_func(),
                timeout=10.0,  # 10 second timeout
            )

            response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Determine status based on response time
            if response_time < 100:
                status = HealthStatus.HEALTHY
            elif response_time < 500:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            check = HealthCheck(
                service_name=self.service_name,
                status=status,
                checked_at=datetime.now(UTC),
                response_time_ms=response_time,
                metadata={"result": result},
            )

        except asyncio.TimeoutError:
            check = HealthCheck(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                checked_at=datetime.now(UTC),
                error_message="Health check timeout",
            )

        except Exception as e:
            check = HealthCheck(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                checked_at=datetime.now(UTC),
                error_message=str(e),
            )

        # Update state
        self.current_status = check.status
        self.last_check = check
        self._add_to_history(check)

        return check

    def _add_to_history(self, check: HealthCheck) -> None:
        """Add check to history."""
        self.check_history.append(check)
        if len(self.check_history) > self.max_history_size:
            self.check_history.pop(0)

    def get_health_summary(self) -> dict[str, Any]:
        """Get health monitoring summary."""
        recent_checks = self.check_history[-10:]
        healthy_count = sum(1 for c in recent_checks if c.status == HealthStatus.HEALTHY)

        return {
            "monitor_id": str(self.monitor_id),
            "service_name": self.service_name,
            "current_status": self.current_status,
            "is_monitoring": self.is_monitoring,
            "last_check": {
                "status": self.last_check.status if self.last_check else None,
                "checked_at": self.last_check.checked_at.isoformat()
                if self.last_check
                else None,
                "response_time_ms": self.last_check.response_time_ms
                if self.last_check
                else None,
                "error": self.last_check.error_message if self.last_check else None,
            },
            "recent_health_rate": healthy_count / len(recent_checks)
            if recent_checks
            else 0.0,
            "total_checks": len(self.check_history),
        }


class FaultToleranceCoordinator(BaseModel):
    """
    Coordinator for fault tolerance patterns.

    Manages circuit breakers, retry policies, and health monitoring.
    """

    coordinator_id: UUID = Field(
        default_factory=uuid4, description="Coordinator identifier"
    )

    # Managed components
    _circuit_breakers: dict[str, CircuitBreaker] = {}
    _health_monitors: dict[str, HealthMonitor] = {}
    _stream_producer: StreamProducer | None = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    def register_circuit_breaker(
        self,
        service_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Register a circuit breaker for a service."""
        if service_name in self._circuit_breakers:
            return self._circuit_breakers[service_name]

        breaker = CircuitBreaker(
            service_name=service_name,
            config=config or CircuitBreakerConfig(),
        )
        self._circuit_breakers[service_name] = breaker
        return breaker

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker | None:
        """Get circuit breaker for service."""
        return self._circuit_breakers.get(service_name)

    def register_health_monitor(
        self,
        service_name: str,
        check_interval_seconds: int = 30,
    ) -> HealthMonitor:
        """Register health monitor for a service."""
        if service_name in self._health_monitors:
            return self._health_monitors[service_name]

        monitor = HealthMonitor(
            service_name=service_name,
            check_interval_seconds=check_interval_seconds,
        )
        self._health_monitors[service_name] = monitor
        return monitor

    def get_health_monitor(self, service_name: str) -> HealthMonitor | None:
        """Get health monitor for service."""
        return self._health_monitors.get(service_name)

    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        retry_policy: RetryPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with retry policy.

        Args:
            func: Async function to execute
            retry_policy: Retry configuration
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        retry_policy.reset()
        last_exception = None

        while retry_policy.should_retry():
            try:
                result = await func(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e
                retry_policy.record_attempt()

                if retry_policy.should_retry():
                    delay = retry_policy.calculate_delay()
                    await asyncio.sleep(delay)
                else:
                    break

        raise last_exception if last_exception else RuntimeError("Execution failed")

    async def execute_with_circuit_breaker(
        self,
        service_name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            service_name: Service identifier
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit open or function fails
        """
        breaker = self.get_circuit_breaker(service_name)
        if not breaker:
            breaker = self.register_circuit_breaker(service_name)

        return await breaker.call(func, *args, **kwargs)

    async def execute_with_fault_tolerance(
        self,
        service_name: str,
        func: Callable[..., Any],
        retry_policy: RetryPolicy | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with full fault tolerance (circuit breaker + retry).

        Args:
            service_name: Service identifier
            func: Async function to execute
            retry_policy: Optional retry configuration
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit open or all retries exhausted
        """
        breaker = self.get_circuit_breaker(service_name)
        if not breaker:
            breaker = self.register_circuit_breaker(service_name)

        if retry_policy:
            return await self.execute_with_retry(
                lambda: breaker.call(func, *args, **kwargs),
                retry_policy,
            )
        else:
            return await breaker.call(func, *args, **kwargs)

    async def get_coordinator_status(self) -> dict[str, Any]:
        """Get coordinator status summary."""
        circuit_breakers_status = {
            name: breaker.get_status()
            for name, breaker in self._circuit_breakers.items()
        }

        health_monitors_status = {
            name: monitor.get_health_summary()
            for name, monitor in self._health_monitors.items()
        }

        return {
            "coordinator_id": str(self.coordinator_id),
            "circuit_breakers": circuit_breakers_status,
            "health_monitors": health_monitors_status,
            "total_breakers": len(self._circuit_breakers),
            "total_monitors": len(self._health_monitors),
        }
