"""
Error recovery service with automatic recovery mechanisms.

This module provides comprehensive error recovery including retry strategies,
circuit breakers, and graceful degradation for agent runtime.
"""

from __future__ import annotations

import asyncio
import random
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, TypeVar

import structlog

from ..models.error_types import (
    DEFAULT_RECOVERY_STRATEGIES,
    SEVERITY_MAX_RETRIES,
    DegradationLevel,
    ErrorCategory,
    ErrorMetadata,
    ErrorRecoveryResult,
    ErrorSeverity,
    RecoveryStrategy,
    RetryConfig,
)
from .circuit_breaker import CircuitBreakerError, get_circuit_breaker_registry

logger = structlog.get_logger()

T = TypeVar("T")


class ErrorRecoveryService:
    """
    Service for handling error recovery with multiple strategies.

    Provides automatic recovery mechanisms including retries, circuit breakers,
    failover, and graceful degradation.
    """

    def __init__(self) -> None:
        """Initialize error recovery service."""
        self._circuit_breaker_registry = get_circuit_breaker_registry()
        self._error_history: dict[str, list[ErrorMetadata]] = {}
        self._degradation_state: dict[str, DegradationLevel] = {}
        self._lock = asyncio.Lock()

    async def execute_with_recovery(
        self,
        func: Callable[..., Any],
        *args: Any,
        agent_id: str | None = None,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        retry_config: RetryConfig | None = None,
        use_circuit_breaker: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, ErrorRecoveryResult | None]:
        """
        Execute function with automatic error recovery.

        Args:
            func: Function to execute
            *args: Positional arguments
            agent_id: Associated agent identifier
            error_category: Category of potential errors
            retry_config: Retry configuration
            use_circuit_breaker: Whether to use circuit breaker
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, recovery_result or None if no error)

        Raises:
            Exception: If all recovery attempts fail
        """
        config = retry_config or RetryConfig()
        start_time = datetime.now()
        attempts = 0
        last_error: Exception | None = None
        recovery_result: ErrorRecoveryResult | None = None

        # Get applicable recovery strategies
        strategies = self._get_recovery_strategies(error_category)

        # Try primary execution
        for strategy in strategies:
            try:
                if strategy == RecoveryStrategy.RETRY_EXPONENTIAL:
                    result = await self._execute_with_exponential_retry(
                        func,
                        config,
                        agent_id,
                        use_circuit_breaker,
                        *args,
                        **kwargs,
                    )
                    return result, None

                elif strategy == RecoveryStrategy.RETRY_CONSTANT:
                    result = await self._execute_with_constant_retry(
                        func,
                        config,
                        agent_id,
                        use_circuit_breaker,
                        *args,
                        **kwargs,
                    )
                    return result, None

                elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                    if use_circuit_breaker:
                        result = await self._execute_with_circuit_breaker(
                            func,
                            agent_id or "default",
                            *args,
                            **kwargs,
                        )
                        return result, None

                elif strategy == RecoveryStrategy.DEGRADE:
                    # Attempt with degraded functionality
                    result, degradation = await self._execute_degraded(
                        func,
                        agent_id,
                        *args,
                        **kwargs,
                    )
                    recovery_result = ErrorRecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.DEGRADE,
                        attempts=1,
                        duration_seconds=(
                            datetime.now() - start_time
                        ).total_seconds(),
                        degradation_level=degradation,
                        message="Executed with degraded functionality",
                    )
                    return result, recovery_result

            except Exception as e:
                last_error = e
                attempts += 1
                logger.warning(
                    "recovery_strategy_failed",
                    strategy=strategy.value,
                    agent_id=agent_id,
                    error=str(e),
                )
                continue

        # All strategies failed
        if last_error:
            duration = (datetime.now() - start_time).total_seconds()
            error_metadata = self._create_error_metadata(
                last_error,
                error_category,
                agent_id,
            )
            await self._record_error(agent_id or "unknown", error_metadata)

            recovery_result = ErrorRecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.NONE,
                attempts=attempts,
                duration_seconds=duration,
                new_error=error_metadata,
                message=f"All recovery strategies exhausted: {str(last_error)}",
            )

            logger.error(
                "error_recovery_failed",
                agent_id=agent_id,
                attempts=attempts,
                duration=duration,
                error=str(last_error),
            )
            raise last_error

        # Should not reach here
        raise RuntimeError("Unexpected error recovery state")

    async def _execute_with_exponential_retry(
        self,
        func: Callable[..., Any],
        config: RetryConfig,
        agent_id: str | None,
        use_circuit_breaker: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute with exponential backoff retry."""
        last_error: Exception | None = None
        delay = config.initial_delay_seconds

        for attempt in range(config.max_attempts):
            try:
                if use_circuit_breaker and agent_id:
                    return await self._execute_with_circuit_breaker(
                        func,
                        agent_id,
                        *args,
                        **kwargs,
                    )
                elif asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except CircuitBreakerError:
                # Don't retry if circuit breaker is open
                raise

            except Exception as e:
                last_error = e
                logger.debug(
                    "retry_attempt_failed",
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts,
                    agent_id=agent_id,
                    error=str(e),
                )

                if attempt < config.max_attempts - 1:
                    # Calculate next delay with jitter
                    if config.jitter:
                        jitter_factor = random.uniform(0.5, 1.5)
                        actual_delay = delay * jitter_factor
                    else:
                        actual_delay = delay

                    actual_delay = min(actual_delay, config.max_delay_seconds)
                    await asyncio.sleep(actual_delay)

                    # Exponential backoff
                    delay *= config.exponential_base

        if last_error:
            raise last_error
        raise RuntimeError("Retry loop completed without result")

    async def _execute_with_constant_retry(
        self,
        func: Callable[..., Any],
        config: RetryConfig,
        agent_id: str | None,
        use_circuit_breaker: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute with constant delay retry."""
        last_error: Exception | None = None

        for attempt in range(config.max_attempts):
            try:
                if use_circuit_breaker and agent_id:
                    return await self._execute_with_circuit_breaker(
                        func,
                        agent_id,
                        *args,
                        **kwargs,
                    )
                elif asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except CircuitBreakerError:
                raise

            except Exception as e:
                last_error = e
                logger.debug(
                    "retry_attempt_failed",
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts,
                    agent_id=agent_id,
                    error=str(e),
                )

                if attempt < config.max_attempts - 1:
                    await asyncio.sleep(config.initial_delay_seconds)

        if last_error:
            raise last_error
        raise RuntimeError("Retry loop completed without result")

    async def _execute_with_circuit_breaker(
        self,
        func: Callable[..., Any],
        circuit_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute with circuit breaker protection."""
        breaker = await self._circuit_breaker_registry.get_breaker(circuit_name)
        return await breaker.call(func, *args, **kwargs)

    async def _execute_degraded(
        self,
        func: Callable[..., Any],
        agent_id: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, DegradationLevel]:
        """
        Execute with degraded functionality.

        Args:
            func: Function to execute
            agent_id: Agent identifier
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, degradation_level)
        """
        # Set degradation level
        degradation_level = await self._determine_degradation_level(agent_id)

        # Add degradation flag to kwargs
        kwargs["degraded"] = True
        kwargs["degradation_level"] = degradation_level

        # Execute with degraded mode
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)

        # Record degradation state
        if agent_id:
            async with self._lock:
                self._degradation_state[agent_id] = degradation_level

        logger.info(
            "executed_degraded",
            agent_id=agent_id,
            degradation_level=degradation_level.value,
        )

        return result, degradation_level

    async def _determine_degradation_level(
        self,
        agent_id: str | None,
    ) -> DegradationLevel:
        """
        Determine appropriate degradation level based on error history.

        Args:
            agent_id: Agent identifier

        Returns:
            Appropriate degradation level
        """
        if not agent_id:
            return DegradationLevel.MINIMAL

        async with self._lock:
            if agent_id not in self._error_history:
                return DegradationLevel.REDUCED

            errors = self._error_history[agent_id]
            recent_errors = [
                e for e in errors
                if (datetime.now() - e.timestamp).total_seconds() < 300
            ]

            if len(recent_errors) >= 10:
                return DegradationLevel.EMERGENCY
            elif len(recent_errors) >= 5:
                return DegradationLevel.MINIMAL
            else:
                return DegradationLevel.REDUCED

    def _get_recovery_strategies(
        self,
        error_category: ErrorCategory,
    ) -> list[RecoveryStrategy]:
        """
        Get applicable recovery strategies for error category.

        Args:
            error_category: Error category

        Returns:
            List of recovery strategies to try
        """
        return DEFAULT_RECOVERY_STRATEGIES.get(
            error_category,
            [RecoveryStrategy.RETRY_CONSTANT, RecoveryStrategy.MANUAL],
        )

    def _create_error_metadata(
        self,
        error: Exception,
        category: ErrorCategory,
        agent_id: str | None,
    ) -> ErrorMetadata:
        """
        Create error metadata from exception.

        Args:
            error: Exception that occurred
            category: Error category
            agent_id: Agent identifier

        Returns:
            Error metadata
        """
        # Determine severity based on exception type
        severity = self._determine_error_severity(error, category)

        # Extract stack trace
        stack_trace = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        )

        return ErrorMetadata(
            error_id=str(uuid.uuid4()),
            category=category,
            severity=severity,
            message=str(error),
            details={
                "exception_type": type(error).__name__,
                "exception_module": type(error).__module__,
            },
            timestamp=datetime.now(),
            agent_id=agent_id,
            stack_trace=stack_trace,
        )

    def _determine_error_severity(
        self,
        error: Exception,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        """
        Determine error severity from exception and category.

        Args:
            error: Exception
            category: Error category

        Returns:
            Error severity level
        """
        # Security and critical infrastructure errors are always critical
        if category in (ErrorCategory.SECURITY, ErrorCategory.INFRASTRUCTURE):
            return ErrorSeverity.CRITICAL

        # Resource exhaustion is high severity
        if category == ErrorCategory.RESOURCE_EXHAUSTION:
            return ErrorSeverity.HIGH

        # Check exception type
        error_type = type(error).__name__

        if error_type in ("SystemError", "MemoryError", "OSError"):
            return ErrorSeverity.CRITICAL
        elif error_type in ("RuntimeError", "ValueError", "KeyError"):
            return ErrorSeverity.HIGH
        elif error_type in ("TimeoutError", "ConnectionError"):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    async def _record_error(
        self,
        agent_id: str,
        error_metadata: ErrorMetadata,
    ) -> None:
        """
        Record error in history for analysis.

        Args:
            agent_id: Agent identifier
            error_metadata: Error metadata to record
        """
        async with self._lock:
            if agent_id not in self._error_history:
                self._error_history[agent_id] = []

            self._error_history[agent_id].append(error_metadata)

            # Keep only recent errors (last 100)
            if len(self._error_history[agent_id]) > 100:
                self._error_history[agent_id] = self._error_history[agent_id][-100:]

    async def get_error_history(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> list[ErrorMetadata]:
        """
        Get error history for agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of errors to return

        Returns:
            List of recent error metadata
        """
        async with self._lock:
            if agent_id not in self._error_history:
                return []
            return self._error_history[agent_id][-limit:]

    async def get_degradation_state(
        self,
        agent_id: str,
    ) -> DegradationLevel | None:
        """
        Get current degradation state for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Current degradation level or None
        """
        async with self._lock:
            return self._degradation_state.get(agent_id)

    async def reset_degradation(self, agent_id: str) -> None:
        """
        Reset degradation state for agent.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            if agent_id in self._degradation_state:
                del self._degradation_state[agent_id]
            logger.info("degradation_reset", agent_id=agent_id)

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get error recovery statistics.

        Returns:
            Statistics dictionary
        """
        async with self._lock:
            total_errors = sum(
                len(errors) for errors in self._error_history.values()
            )

            errors_by_category: dict[str, int] = {}
            errors_by_severity: dict[str, int] = {}

            for errors in self._error_history.values():
                for error in errors:
                    category = error.category.value
                    severity = error.severity.value

                    errors_by_category[category] = (
                        errors_by_category.get(category, 0) + 1
                    )
                    errors_by_severity[severity] = (
                        errors_by_severity.get(severity, 0) + 1
                    )

            return {
                "total_errors": total_errors,
                "agents_with_errors": len(self._error_history),
                "errors_by_category": errors_by_category,
                "errors_by_severity": errors_by_severity,
                "degraded_agents": len(self._degradation_state),
                "circuit_breakers": (
                    self._circuit_breaker_registry.get_all_stats()
                ),
            }


# Global service instance
_error_recovery_service: ErrorRecoveryService | None = None


def get_error_recovery_service() -> ErrorRecoveryService:
    """
    Get global error recovery service.

    Returns:
        Global service instance
    """
    global _error_recovery_service
    if _error_recovery_service is None:
        _error_recovery_service = ErrorRecoveryService()
    return _error_recovery_service
