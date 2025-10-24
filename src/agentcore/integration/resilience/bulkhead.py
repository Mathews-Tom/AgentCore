"""Bulkhead pattern implementation.

Resource isolation using semaphores and queues to prevent resource exhaustion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

import structlog

from agentcore.integration.resilience.exceptions import (
    BulkheadRejectedError,
    ResilienceTimeoutError,
)
from agentcore.integration.resilience.models import BulkheadConfig

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class BulkheadMetrics:
    """Bulkhead runtime metrics."""

    def __init__(self, name: str) -> None:
        """Initialize metrics.

        Args:
            name: Bulkhead name
        """
        self.name = name
        self.total_requests = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_timeouts = 0
        self.current_concurrent = 0
        self.current_queued = 0
        self.max_concurrent_seen = 0
        self.max_queued_seen = 0
        self.last_rejection_time: datetime | None = None


class Bulkhead:
    """Bulkhead pattern for resource isolation.

    Implements the bulkhead pattern using asyncio.Semaphore for:
    - Limiting concurrent requests
    - Queue-based overflow handling
    - Per-service/per-tenant isolation
    - Preventing resource exhaustion

    Thread-safe for async operations.
    """

    def __init__(self, config: BulkheadConfig) -> None:
        """Initialize bulkhead.

        Args:
            config: Bulkhead configuration
        """
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._queue: asyncio.Queue[asyncio.Future[Any]] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._lock = asyncio.Lock()
        self.metrics = BulkheadMetrics(config.name)

        logger.info(
            "bulkhead_initialized",
            name=config.name,
            max_concurrent=config.max_concurrent_requests,
            queue_size=config.queue_size,
        )

    async def execute(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute operation with bulkhead protection.

        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            BulkheadRejectedError: If bulkhead rejects request
            ResilienceTimeoutError: If queued request times out
            Exception: If operation fails
        """
        async with self._lock:
            self.metrics.total_requests += 1

        # Try to acquire semaphore (non-blocking)
        acquired = self._semaphore.locked() is False

        if not acquired:
            # Semaphore is locked, try to queue
            if self._queue.full():
                # Queue is full, reject request
                async with self._lock:
                    self.metrics.total_rejected += 1
                    self.metrics.last_rejection_time = datetime.now()

                logger.warning(
                    "bulkhead_rejected",
                    name=self.config.name,
                    max_concurrent=self.config.max_concurrent_requests,
                    queue_size=self.config.queue_size,
                )

                raise BulkheadRejectedError(
                    name=self.config.name,
                    max_concurrent=self.config.max_concurrent_requests,
                    queue_size=self.config.queue_size,
                )

            # Add to queue
            future: asyncio.Future[Any] = asyncio.Future()

            async with self._lock:
                self.metrics.current_queued += 1
                self.metrics.max_queued_seen = max(
                    self.metrics.max_queued_seen,
                    self.metrics.current_queued,
                )

            try:
                await asyncio.wait_for(
                    self._queue.put(future),
                    timeout=self.config.queue_timeout_seconds,
                )
            except asyncio.TimeoutError as e:
                async with self._lock:
                    self.metrics.current_queued -= 1
                    self.metrics.total_timeouts += 1

                logger.warning(
                    "bulkhead_queue_timeout",
                    name=self.config.name,
                    timeout_seconds=self.config.queue_timeout_seconds,
                )

                raise ResilienceTimeoutError(
                    operation=f"bulkhead_queue:{self.config.name}",
                    timeout_seconds=self.config.queue_timeout_seconds,
                ) from e

            # Wait for queue to be processed
            try:
                await asyncio.wait_for(
                    future,
                    timeout=self.config.queue_timeout_seconds,
                )
            except asyncio.TimeoutError as e:
                async with self._lock:
                    self.metrics.current_queued -= 1
                    self.metrics.total_timeouts += 1

                raise ResilienceTimeoutError(
                    operation=f"bulkhead_wait:{self.config.name}",
                    timeout_seconds=self.config.queue_timeout_seconds,
                ) from e
            finally:
                async with self._lock:
                    self.metrics.current_queued -= 1

        # Acquire semaphore
        async with self._semaphore:
            async with self._lock:
                self.metrics.total_accepted += 1
                self.metrics.current_concurrent += 1
                self.metrics.max_concurrent_seen = max(
                    self.metrics.max_concurrent_seen,
                    self.metrics.current_concurrent,
                )

            try:
                # Execute operation
                result = await operation(*args, **kwargs)

                logger.debug(
                    "bulkhead_success",
                    name=self.config.name,
                    concurrent=self.metrics.current_concurrent,
                )

                return result

            finally:
                async with self._lock:
                    self.metrics.current_concurrent -= 1

                # Process next queued request
                if not self._queue.empty():
                    try:
                        next_future = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=0.1,
                        )
                        next_future.set_result(None)
                    except asyncio.TimeoutError:
                        pass
                    except Exception as e:
                        logger.warning(
                            "bulkhead_queue_processing_error",
                            error=str(e),
                        )

    async def __aenter__(self) -> Bulkhead:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        pass


def bulkhead(
    config: BulkheadConfig,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for wrapping async functions with bulkhead.

    Args:
        config: Bulkhead configuration

    Returns:
        Decorator function

    Example:
        @bulkhead(BulkheadConfig(name="my_service", max_concurrent_requests=5))
        async def call_service():
            return await http_client.get("/api/endpoint")
    """
    bh = Bulkhead(config)

    def decorator(
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await bh.execute(func, *args, **kwargs)

        return wrapper

    return decorator


class BulkheadRegistry:
    """Global registry for bulkheads.

    Singleton registry for managing bulkheads across the application.
    """

    _instance: BulkheadRegistry | None = None
    _lock = asyncio.Lock()

    def __new__(cls) -> BulkheadRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._bulkheads: dict[str, Bulkhead] = {}
        return cls._instance

    async def get_or_create(
        self,
        config: BulkheadConfig,
    ) -> Bulkhead:
        """Get existing bulkhead or create new one.

        Args:
            config: Bulkhead configuration

        Returns:
            Bulkhead instance
        """
        async with self._lock:
            if config.name not in self._bulkheads:
                self._bulkheads[config.name] = Bulkhead(config)

            return self._bulkheads[config.name]

    async def get(self, name: str) -> Bulkhead | None:
        """Get bulkhead by name.

        Args:
            name: Bulkhead name

        Returns:
            Bulkhead instance or None if not found
        """
        return self._bulkheads.get(name)

    async def remove(self, name: str) -> None:
        """Remove bulkhead from registry.

        Args:
            name: Bulkhead name
        """
        async with self._lock:
            if name in self._bulkheads:
                del self._bulkheads[name]

    def get_all_metrics(self) -> dict[str, BulkheadMetrics]:
        """Get metrics for all bulkheads.

        Returns:
            Dictionary mapping names to metrics
        """
        return {
            name: bulkhead.metrics
            for name, bulkhead in self._bulkheads.items()
        }
