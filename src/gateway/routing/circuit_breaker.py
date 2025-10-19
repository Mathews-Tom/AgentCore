"""
Circuit Breaker

Circuit breaker pattern implementation for resilient backend service calls.
Prevents cascading failures by failing fast when a service is unhealthy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service is failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    """Number of failures before opening circuit"""

    success_threshold: int = 2
    """Number of successes in half-open before closing"""

    timeout: float = 60.0
    """Time in seconds before attempting recovery (half-open)"""

    expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception
    """Exceptions that count as failures"""


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_state_change: float = field(default_factory=time.time)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, service: str, retry_after: float):
        """
        Initialize circuit breaker open error.

        Args:
            service: Service name
            retry_after: Seconds until retry
        """
        self.service = service
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker is open for service '{service}'. "
            f"Retry after {retry_after:.1f} seconds."
        )


class CircuitBreaker:
    """
    Circuit breaker for resilient service calls.

    Implements the circuit breaker pattern to prevent cascading failures
    by failing fast when a backend service is experiencing issues.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests fail immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, service_name: str, config: CircuitBreakerConfig | None = None):
        """
        Initialize circuit breaker.

        Args:
            service_name: Name of the service
            config: Circuit breaker configuration
        """
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self.stats.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.stats.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.stats.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.stats.state == CircuitState.HALF_OPEN

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        self.stats.total_calls += 1

        # Check if we should transition to half-open
        if self.is_open:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                retry_after = self._time_until_retry()
                logger.debug(
                    "Circuit breaker is open, failing fast",
                    service=self.service_name,
                    retry_after=retry_after,
                )
                raise CircuitBreakerOpenError(self.service_name, retry_after)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    async def call_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        self.stats.total_calls += 1

        # Check if we should transition to half-open
        if self.is_open:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                retry_after = self._time_until_retry()
                logger.debug(
                    "Circuit breaker is open, failing fast",
                    service=self.service_name,
                    retry_after=retry_after,
                )
                raise CircuitBreakerOpenError(self.service_name, retry_after)

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        """Handle successful call."""
        self.stats.total_successes += 1

        if self.is_half_open:
            self.stats.success_count += 1

            if self.stats.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.is_closed:
            # Reset failure count on success in closed state
            self.stats.failure_count = 0

        logger.debug(
            "Circuit breaker call succeeded",
            service=self.service_name,
            state=self.state,
            success_count=self.stats.success_count,
        )

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.stats.total_failures += 1
        self.stats.failure_count += 1
        self.stats.last_failure_time = time.time()

        logger.warning(
            "Circuit breaker call failed",
            service=self.service_name,
            state=self.state,
            failure_count=self.stats.failure_count,
            threshold=self.config.failure_threshold,
        )

        if self.is_half_open:
            # Any failure in half-open immediately opens circuit
            self._transition_to_open()
        elif self.is_closed:
            # Check if we've exceeded failure threshold
            if self.stats.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.is_open:
            return False

        if self.stats.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.stats.last_failure_time
        return time_since_failure >= self.config.timeout

    def _time_until_retry(self) -> float:
        """Calculate time until retry is allowed."""
        if self.stats.last_failure_time is None:
            return 0.0

        time_since_failure = time.time() - self.stats.last_failure_time
        return max(0.0, self.config.timeout - time_since_failure)

    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        old_state = self.stats.state
        self.stats.state = CircuitState.OPEN
        self.stats.last_state_change = time.time()

        logger.error(
            "Circuit breaker opened",
            service=self.service_name,
            old_state=old_state,
            failure_count=self.stats.failure_count,
            timeout=self.config.timeout,
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        old_state = self.stats.state
        self.stats.state = CircuitState.HALF_OPEN
        self.stats.success_count = 0
        self.stats.failure_count = 0
        self.stats.last_state_change = time.time()

        logger.info(
            "Circuit breaker half-opened (testing recovery)",
            service=self.service_name,
            old_state=old_state,
        )

    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        old_state = self.stats.state
        self.stats.state = CircuitState.CLOSED
        self.stats.success_count = 0
        self.stats.failure_count = 0
        self.stats.last_state_change = time.time()

        logger.info(
            "Circuit breaker closed (service recovered)",
            service=self.service_name,
            old_state=old_state,
        )

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        logger.info("Circuit breaker manually reset", service=self.service_name)
        self._transition_to_closed()

    def get_stats(self) -> dict[str, Any]:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary of statistics
        """
        uptime = time.time() - self.stats.last_state_change

        return {
            "service": self.service_name,
            "state": self.stats.state,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_calls": self.stats.total_calls,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "failure_rate": (
                self.stats.total_failures / self.stats.total_calls
                if self.stats.total_calls > 0
                else 0.0
            ),
            "state_uptime_seconds": uptime,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        """
        Initialize circuit breaker registry.

        Args:
            default_config: Default configuration for new circuit breakers
        """
        self.default_config = default_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(
        self, service_name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker for service.

        Args:
            service_name: Service name
            config: Optional custom configuration

        Returns:
            Circuit breaker instance
        """
        if service_name not in self._breakers:
            breaker_config = config or self.default_config
            self._breakers[service_name] = CircuitBreaker(service_name, breaker_config)

        return self._breakers[service_name]

    def get_all_stats(self) -> list[dict[str, Any]]:
        """
        Get statistics for all circuit breakers.

        Returns:
            List of statistics dictionaries
        """
        return [breaker.get_stats() for breaker in self._breakers.values()]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

        logger.info("All circuit breakers reset")
