"""
Circuit breaker implementation for fault tolerance.

This module provides circuit breaker pattern to prevent cascading failures
and allow systems to recover from transient errors.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

import structlog

from ..models.error_types import CircuitBreakerConfig

logger = structlog.get_logger()

T = TypeVar("T")


class CircuitState(str, Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    Implements the circuit breaker pattern with closed, open, and half-open states.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker identifier
            config: Configuration for circuit behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: datetime | None = None
        self._half_open_attempts: int = 0
        self._lock = asyncio.Lock()

        logger.info(
            "circuit_breaker_created",
            name=name,
            config=self.config.model_dump(),
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            # Check if circuit should transition states
            await self._check_state_transition()

            # Block if circuit is open
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open"
                )

        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            await self._record_success()
            return result

        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise

    async def _check_state_transition(self) -> None:
        """Check and perform state transitions."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    # Transition to half-open
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        elapsed_seconds=elapsed,
                    )

    async def _record_success(self) -> None:
        """Record successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    "circuit_breaker_success",
                    name=self.name,
                    success_count=self._success_count,
                    threshold=self.config.success_threshold,
                )

                # Check if we can close the circuit
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_attempts = 0
                    logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                    )

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """
        Record failed execution.

        Args:
            error: Exception that occurred
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                error=str(error),
                failure_count=self._failure_count,
                threshold=self.config.failure_threshold,
            )

            if self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._success_count = 0
                    logger.error(
                        "circuit_breaker_opened",
                        name=self.name,
                        failure_count=self._failure_count,
                    )

            elif self._state == CircuitState.HALF_OPEN:
                # Increment half-open attempts
                self._half_open_attempts += 1

                # Return to open if max attempts reached
                if self._half_open_attempts >= self.config.half_open_max_attempts:
                    self._state = CircuitState.OPEN
                    self._success_count = 0
                    self._half_open_attempts = 0
                    logger.error(
                        "circuit_breaker_reopened",
                        name=self.name,
                        attempts=self._half_open_attempts,
                    )

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_attempts = 0
            self._last_failure_time = None
            logger.info("circuit_breaker_reset", name=self.name)

    def get_stats(self) -> dict[str, Any]:
        """
        Get circuit breaker statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "half_open_attempts": self._half_open_attempts,
            "last_failure_time": (
                self._last_failure_time.isoformat()
                if self._last_failure_time
                else None
            ),
            "config": self.config.model_dump(),
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management of circuit breakers for different services.
    """

    def __init__(self) -> None:
        """Initialize circuit breaker registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker.

        Args:
            name: Circuit breaker identifier
            config: Configuration for new breakers

        Returns:
            Circuit breaker instance
        """
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    async def remove_breaker(self, name: str) -> None:
        """
        Remove circuit breaker.

        Args:
            name: Circuit breaker identifier
        """
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]

    async def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all circuit breakers.

        Returns:
            Dictionary mapping breaker names to their stats
        """
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    def list_breakers(self) -> list[str]:
        """
        List all circuit breaker names.

        Returns:
            List of circuit breaker names
        """
        return list(self._breakers.keys())


# Global registry instance
_circuit_breaker_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """
    Get global circuit breaker registry.

    Returns:
        Global registry instance
    """
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry
