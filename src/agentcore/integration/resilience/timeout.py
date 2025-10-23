"""Timeout management for operations.

Async timeout enforcement with cascading timeout handling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

import structlog

from agentcore.integration.resilience.exceptions import ResilienceTimeoutError
from agentcore.integration.resilience.models import TimeoutConfig

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class TimeoutManager:
    """Timeout manager for async operations.

    Provides timeout enforcement using asyncio.wait_for with:
    - Operation-level timeouts
    - Connection-specific timeouts
    - Read-specific timeouts
    - Cascading timeout handling
    """

    def __init__(self, config: TimeoutConfig) -> None:
        """Initialize timeout manager.

        Args:
            config: Timeout configuration
        """
        self.config = config

        logger.info(
            "timeout_manager_initialized",
            name=config.name,
            timeout_seconds=config.timeout_seconds,
        )

    async def execute(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        timeout_override: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute operation with timeout.

        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            timeout_override: Optional timeout override in seconds
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            ResilienceTimeoutError: If operation times out
            Exception: If operation fails
        """
        timeout = timeout_override or self.config.timeout_seconds

        logger.debug(
            "timeout_execute_start",
            name=self.config.name,
            timeout_seconds=timeout,
        )

        try:
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=timeout,
            )

            logger.debug(
                "timeout_execute_success",
                name=self.config.name,
            )

            return result

        except asyncio.TimeoutError as e:
            logger.warning(
                "timeout_exceeded",
                name=self.config.name,
                timeout_seconds=timeout,
            )

            raise ResilienceTimeoutError(
                operation=self.config.name,
                timeout_seconds=timeout,
            ) from e

    async def __aenter__(self) -> TimeoutManager:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        pass


def with_timeout(
    config: TimeoutConfig,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for wrapping async functions with timeout.

    Args:
        config: Timeout configuration

    Returns:
        Decorator function

    Example:
        @with_timeout(TimeoutConfig(name="my_service", timeout_seconds=5.0))
        async def call_service():
            return await http_client.get("/api/endpoint")
    """
    manager = TimeoutManager(config)

    def decorator(
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await manager.execute(func, *args, **kwargs)

        return wrapper

    return decorator


async def with_timeout_direct(
    operation: Callable[..., Awaitable[T]],
    timeout_seconds: float,
    operation_name: str = "operation",
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute operation with direct timeout.

    Utility function for one-off timeout enforcement without config.

    Args:
        operation: Async callable to execute
        timeout_seconds: Timeout in seconds
        operation_name: Name for error messages
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        Operation result

    Raises:
        ResilienceTimeoutError: If operation times out
        Exception: If operation fails

    Example:
        result = await with_timeout_direct(
            http_client.get,
            timeout_seconds=5.0,
            operation_name="api_call",
            "/api/endpoint"
        )
    """
    try:
        return await asyncio.wait_for(
            operation(*args, **kwargs),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as e:
        raise ResilienceTimeoutError(
            operation=operation_name,
            timeout_seconds=timeout_seconds,
        ) from e
