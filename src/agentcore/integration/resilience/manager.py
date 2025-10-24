"""Resilience manager orchestrator.

Unified orchestration combining circuit breaker, bulkhead, and timeout patterns.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog

from agentcore.integration.resilience.bulkhead import Bulkhead, BulkheadRegistry
from agentcore.integration.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)
from agentcore.integration.resilience.models import ResilienceConfig
from agentcore.integration.resilience.timeout import TimeoutManager

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ResilienceManager:
    """Unified resilience orchestrator.

    Combines circuit breaker, bulkhead, and timeout patterns for:
    - Comprehensive fault tolerance
    - Resource isolation
    - Graceful degradation
    - Fallback handling

    Execution order:
    1. Circuit breaker (fail fast if circuit is open)
    2. Bulkhead (enforce resource limits)
    3. Timeout (enforce operation timeout)
    4. Execute operation
    5. Fallback handler (if configured and operation fails)
    """

    def __init__(
        self,
        config: ResilienceConfig,
        fallback_handler: Callable[..., Awaitable[T]] | None = None,
    ) -> None:
        """Initialize resilience manager.

        Args:
            config: Resilience configuration
            fallback_handler: Optional fallback handler for graceful degradation
        """
        self.config = config
        self._fallback_handler = fallback_handler

        # Initialize patterns based on config
        self._circuit_breaker: CircuitBreaker | None = None
        self._bulkhead: Bulkhead | None = None
        self._timeout_manager: TimeoutManager | None = None

        logger.info(
            "resilience_manager_initialized",
            circuit_breaker_enabled=config.circuit_breaker is not None,
            bulkhead_enabled=config.bulkhead is not None,
            timeout_enabled=config.timeout is not None,
            fallback_enabled=config.enable_fallback and fallback_handler is not None,
        )

    async def initialize(self) -> None:
        """Initialize resilience patterns.

        Creates circuit breaker, bulkhead, and timeout manager instances
        based on configuration.
        """
        # Initialize circuit breaker
        if self.config.circuit_breaker:
            registry = CircuitBreakerRegistry()
            self._circuit_breaker = await registry.get_or_create(
                self.config.circuit_breaker
            )

        # Initialize bulkhead
        if self.config.bulkhead:
            registry = BulkheadRegistry()
            self._bulkhead = await registry.get_or_create(self.config.bulkhead)

        # Initialize timeout manager
        if self.config.timeout:
            self._timeout_manager = TimeoutManager(self.config.timeout)

        logger.info("resilience_manager_initialized_complete")

    async def execute(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute operation with resilience patterns.

        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result or fallback result

        Raises:
            Exception: If all resilience patterns fail and no fallback
        """
        # Ensure manager is initialized
        if not self._is_initialized():
            await self.initialize()

        try:
            # Wrap operation with resilience patterns
            protected_operation = self._wrap_operation(operation)

            # Execute with circuit breaker
            if self._circuit_breaker:
                return await self._circuit_breaker.call(
                    protected_operation, *args, **kwargs
                )
            else:
                return await protected_operation(*args, **kwargs)

        except Exception as e:
            # Try fallback if enabled
            if self.config.enable_fallback and self._fallback_handler:
                logger.warning(
                    "resilience_fallback_triggered",
                    error=str(e),
                    error_type=type(e).__name__,
                )

                try:
                    return await self._fallback_handler(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        "resilience_fallback_failed",
                        error=str(fallback_error),
                        original_error=str(e),
                    )
                    raise

            # No fallback, re-raise original error
            raise

    def _wrap_operation(
        self,
        operation: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """Wrap operation with bulkhead and timeout.

        Args:
            operation: Original operation

        Returns:
            Wrapped operation
        """

        async def wrapped(*args: Any, **kwargs: Any) -> T:
            # Apply bulkhead protection
            if self._bulkhead:
                return await self._bulkhead.execute(
                    self._wrap_with_timeout(operation),
                    *args,
                    **kwargs,
                )
            else:
                return await self._wrap_with_timeout(operation)(*args, **kwargs)

        return wrapped

    def _wrap_with_timeout(
        self,
        operation: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """Wrap operation with timeout.

        Args:
            operation: Original operation

        Returns:
            Wrapped operation
        """

        async def wrapped(*args: Any, **kwargs: Any) -> T:
            # Apply timeout
            if self._timeout_manager:
                return await self._timeout_manager.execute(
                    operation, *args, **kwargs
                )
            else:
                return await operation(*args, **kwargs)

        return wrapped

    def _is_initialized(self) -> bool:
        """Check if manager is initialized.

        Returns:
            True if all configured patterns are initialized
        """
        if self.config.circuit_breaker and not self._circuit_breaker:
            return False

        if self.config.bulkhead and not self._bulkhead:
            return False

        if self.config.timeout and not self._timeout_manager:
            return False

        return True

    async def __aenter__(self) -> ResilienceManager:
        """Enter async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        pass


class ResilienceRegistry:
    """Global registry for resilience managers.

    Singleton registry for managing resilience configurations across the application.
    """

    _instance: ResilienceRegistry | None = None

    def __new__(cls) -> ResilienceRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._managers: dict[str, ResilienceManager] = {}
        return cls._instance

    def register(
        self,
        name: str,
        config: ResilienceConfig,
        fallback_handler: Callable[..., Awaitable[Any]] | None = None,
    ) -> ResilienceManager:
        """Register a resilience manager.

        Args:
            name: Manager name
            config: Resilience configuration
            fallback_handler: Optional fallback handler

        Returns:
            Resilience manager instance
        """
        if name not in self._managers:
            self._managers[name] = ResilienceManager(config, fallback_handler)

        return self._managers[name]

    def get(self, name: str) -> ResilienceManager | None:
        """Get resilience manager by name.

        Args:
            name: Manager name

        Returns:
            Resilience manager instance or None if not found
        """
        return self._managers.get(name)

    def remove(self, name: str) -> None:
        """Remove resilience manager from registry.

        Args:
            name: Manager name
        """
        if name in self._managers:
            del self._managers[name]

    def get_all(self) -> dict[str, ResilienceManager]:
        """Get all resilience managers.

        Returns:
            Dictionary mapping names to managers
        """
        return self._managers.copy()
