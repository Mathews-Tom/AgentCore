"""Enhanced retry logic with exponential backoff and jitter.

Implements retry strategies for tool execution with configurable backoff algorithms.
"""

import asyncio
import random
from enum import Enum
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger()


class BackoffStrategy(str, Enum):
    """Backoff strategy for retries."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class RetryHandler:
    """Enhanced retry handler with configurable backoff strategies."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        jitter: bool = True,
    ):
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            strategy: Backoff strategy to use
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.strategy == BackoffStrategy.EXPONENTIAL:
            # Exponential backoff: base_delay * 2^attempt
            delay = self.base_delay * (2**attempt)
        elif self.strategy == BackoffStrategy.LINEAR:
            # Linear backoff: base_delay * (attempt + 1)
            delay = self.base_delay * (attempt + 1)
        else:  # FIXED
            # Fixed backoff: base_delay
            delay = self.base_delay

        # Apply max_delay cap
        delay = min(delay, self.max_delay)

        # Add jitter if enabled (±25% random variation)
        if self.jitter:
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure delay is positive

        return delay

    async def retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
        on_retry: Callable[[Exception, int, float], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            retryable_exceptions: Exceptions that should trigger retry
            on_retry: Optional callback called on each retry (exception, attempt, delay)
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        "retry_succeeded",
                        attempt=attempt,
                        max_retries=self.max_retries,
                    )
                return result

            except retryable_exceptions as e:
                last_exception = e

                # Check if we should retry
                if attempt >= self.max_retries:
                    logger.error(
                        "retry_exhausted",
                        attempt=attempt,
                        max_retries=self.max_retries,
                        error=str(e),
                    )
                    raise

                # Calculate delay
                delay = self.calculate_delay(attempt)

                logger.warning(
                    "retry_attempt",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay=delay,
                    error=str(e),
                    strategy=self.strategy.value,
                )

                # Call retry callback if provided
                if on_retry:
                    on_retry(e, attempt + 1, delay)

                # Wait before retrying
                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic failed unexpectedly")


async def retry_with_backoff(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Convenience function for retry with backoff.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        strategy: Backoff strategy to use
        jitter: Whether to add random jitter
        retryable_exceptions: Exceptions that trigger retry
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function execution

    Raises:
        Last exception if all retries exhausted
    """
    handler = RetryHandler(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        jitter=jitter,
    )
    return await handler.retry(
        func,
        *args,
        retryable_exceptions=retryable_exceptions,
        **kwargs,
    )


class CircuitBreaker:
    """Circuit breaker pattern for tool execution.

    Prevents cascading failures by temporarily disabling failing tools.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half_open

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function

        Raises:
            Exception: If circuit is open or function fails
        """
        # Check circuit state
        if self.state == "open":
            if self.last_failure_time:
                elapsed = asyncio.get_event_loop().time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info("circuit_breaker_half_open")
                    self.state = "half_open"
                else:
                    raise Exception(
                        f"Circuit breaker is open. Retry after "
                        f"{self.recovery_timeout - elapsed:.2f}s"
                    )

        try:
            result = await func(*args, **kwargs)

            # Success - reset if in half_open state
            if self.state == "half_open":
                logger.info("circuit_breaker_closed")
                self.state = "closed"
                self.failure_count = 0

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()

            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    "circuit_breaker_opened",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                )
                self.state = "open"

            raise e

    def reset(self) -> None:
        """Reset circuit breaker state."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("circuit_breaker_reset")
