"""Circuit breaker pattern implementation.

Three-state circuit breaker (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) for
fault tolerance and preventing cascading failures.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, TypeVar

import structlog

from agentcore.integration.resilience.exceptions import CircuitBreakerOpenError
from agentcore.integration.resilience.models import (
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerState,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for fault tolerance.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

    State transitions:
    - CLOSED -> OPEN: When failure_threshold is exceeded
    - OPEN -> HALF_OPEN: After timeout_seconds elapsed
    - HALF_OPEN -> CLOSED: When success_threshold is reached
    - HALF_OPEN -> OPEN: On any failure

    Thread-safe for async operations.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self._lock = asyncio.Lock()

        # Initialize metrics
        self.metrics = CircuitBreakerMetrics(
            name=config.name,
            state=CircuitBreakerState.CLOSED,
        )

        logger.info(
            "circuit_breaker_initialized",
            name=config.name,
            failure_threshold=config.failure_threshold,
            timeout_seconds=config.timeout_seconds,
        )

    async def call(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute operation through circuit breaker.

        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN
            Exception: If operation fails
        """
        async with self._lock:
            # Check if circuit should transition states
            await self._check_state_transition()

            # Record request
            self.metrics.total_requests += 1

            # Handle OPEN state
            if self.metrics.state == CircuitBreakerState.OPEN:
                self.metrics.total_rejections += 1
                logger.warning(
                    "circuit_breaker_rejected",
                    name=self.config.name,
                    state=self.metrics.state,
                    failures=self.metrics.failure_count,
                )
                raise CircuitBreakerOpenError(
                    name=self.config.name,
                    failure_count=self.metrics.failure_count,
                    threshold=self.config.failure_threshold,
                )

            # Handle HALF_OPEN state with request limit
            if self.metrics.state == CircuitBreakerState.HALF_OPEN:
                # Check if we've exceeded half-open request limit
                in_flight = self.metrics.success_count + self.metrics.failure_count
                if in_flight >= self.config.half_open_max_requests:
                    self.metrics.total_rejections += 1
                    raise CircuitBreakerOpenError(
                        name=self.config.name,
                        failure_count=self.metrics.failure_count,
                        threshold=self.config.failure_threshold,
                    )

        # Execute operation (outside lock to allow concurrent requests in CLOSED)
        try:
            result = await operation(*args, **kwargs)

            # Record success
            async with self._lock:
                await self._on_success()

            return result

        except Exception as e:
            # Record failure
            async with self._lock:
                await self._on_failure()

            raise

    async def _check_state_transition(self) -> None:
        """Check if circuit breaker should transition states.

        Handles:
        - OPEN -> HALF_OPEN after timeout
        """
        if self.metrics.state == CircuitBreakerState.OPEN:
            if self.metrics.opened_at is None:
                return

            elapsed = (datetime.now(UTC) - self.metrics.opened_at).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                # Transition to HALF_OPEN
                self.metrics.state = CircuitBreakerState.HALF_OPEN
                self.metrics.failure_count = 0
                self.metrics.success_count = 0
                self.metrics.last_state_change = datetime.now(UTC)

                logger.info(
                    "circuit_breaker_half_open",
                    name=self.config.name,
                    elapsed_seconds=elapsed,
                )

    async def _on_success(self) -> None:
        """Handle successful request.

        Updates metrics and potentially transitions state to CLOSED.
        """
        self.metrics.total_successes += 1
        self.metrics.success_count += 1
        self.metrics.failure_count = 0

        logger.debug(
            "circuit_breaker_success",
            name=self.config.name,
            state=self.metrics.state,
            success_count=self.metrics.success_count,
        )

        # HALF_OPEN -> CLOSED when success threshold reached
        if self.metrics.state == CircuitBreakerState.HALF_OPEN:
            if self.metrics.success_count >= self.config.success_threshold:
                self.metrics.state = CircuitBreakerState.CLOSED
                self.metrics.success_count = 0
                self.metrics.failure_count = 0
                self.metrics.opened_at = None
                self.metrics.last_state_change = datetime.now(UTC)

                logger.info(
                    "circuit_breaker_closed",
                    name=self.config.name,
                    total_requests=self.metrics.total_requests,
                    total_successes=self.metrics.total_successes,
                )

    async def _on_failure(self) -> None:
        """Handle failed request.

        Updates metrics and potentially transitions state to OPEN.
        """
        self.metrics.total_failures += 1
        self.metrics.failure_count += 1
        self.metrics.success_count = 0
        self.metrics.last_failure_time = datetime.now(UTC)

        logger.warning(
            "circuit_breaker_failure",
            name=self.config.name,
            state=self.metrics.state,
            failure_count=self.metrics.failure_count,
            threshold=self.config.failure_threshold,
        )

        # Transition to OPEN on threshold
        if (
            self.metrics.state == CircuitBreakerState.CLOSED
            and self.metrics.failure_count >= self.config.failure_threshold
        ):
            self._open_circuit()

        # HALF_OPEN -> OPEN on any failure
        elif self.metrics.state == CircuitBreakerState.HALF_OPEN:
            self._open_circuit()

    def _open_circuit(self) -> None:
        """Open the circuit breaker.

        Transitions to OPEN state and records timestamp.
        """
        self.metrics.state = CircuitBreakerState.OPEN
        self.metrics.opened_at = datetime.now(UTC)
        self.metrics.last_state_change = datetime.now(UTC)
        self.metrics.success_count = 0

        logger.error(
            "circuit_breaker_opened",
            name=self.config.name,
            failure_count=self.metrics.failure_count,
            threshold=self.config.failure_threshold,
            total_failures=self.metrics.total_failures,
        )

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state.

        Clears all metrics and counters. Use with caution.
        """
        self.metrics = CircuitBreakerMetrics(
            name=self.config.name,
            state=CircuitBreakerState.CLOSED,
        )

        logger.info(
            "circuit_breaker_reset",
            name=self.config.name,
        )

    async def __aenter__(self) -> CircuitBreaker:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        pass


def circuit_breaker(
    config: CircuitBreakerConfig,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for wrapping async functions with circuit breaker.

    Args:
        config: Circuit breaker configuration

    Returns:
        Decorator function

    Example:
        @circuit_breaker(CircuitBreakerConfig(name="my_service"))
        async def call_service():
            return await http_client.get("/api/endpoint")
    """
    breaker = CircuitBreaker(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """Global registry for circuit breakers.

    Singleton registry for managing circuit breakers across the application.
    """

    _instance: CircuitBreakerRegistry | None = None
    _lock = asyncio.Lock()

    def __new__(cls) -> CircuitBreakerRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._breakers: dict[str, CircuitBreaker] = {}
        return cls._instance

    async def get_or_create(
        self,
        config: CircuitBreakerConfig,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.

        Args:
            config: Circuit breaker configuration

        Returns:
            Circuit breaker instance
        """
        async with self._lock:
            if config.name not in self._breakers:
                self._breakers[config.name] = CircuitBreaker(config)

            return self._breakers[config.name]

    async def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            Circuit breaker instance or None if not found
        """
        return self._breakers.get(name)

    async def remove(self, name: str) -> None:
        """Remove circuit breaker from registry.

        Args:
            name: Circuit breaker name
        """
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]

    async def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        async with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_metrics(self) -> dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers.

        Returns:
            Dictionary mapping names to metrics
        """
        return {name: breaker.metrics for name, breaker in self._breakers.items()}
