"""Resilience pattern exceptions.

Custom exception hierarchy for circuit breaker, bulkhead, and timeout errors.
"""

from __future__ import annotations


class ResilienceError(Exception):
    """Base exception for all resilience pattern errors."""

    def __init__(self, message: str) -> None:
        """Initialize resilience error.

        Args:
            message: Error description
        """
        self.message = message
        super().__init__(message)


class CircuitBreakerOpenError(ResilienceError):
    """Exception raised when circuit breaker is open.

    Raised when a request is attempted while the circuit breaker is in the
    OPEN state, preventing the request from being executed to protect the
    downstream service.
    """

    def __init__(
        self,
        name: str,
        failure_count: int,
        threshold: int,
    ) -> None:
        """Initialize circuit breaker open error.

        Args:
            name: Circuit breaker name
            failure_count: Current failure count
            threshold: Failure threshold
        """
        self.name = name
        self.failure_count = failure_count
        self.threshold = threshold
        super().__init__(
            f"Circuit breaker '{name}' is OPEN: {failure_count}/{threshold} failures"
        )


class BulkheadRejectedError(ResilienceError):
    """Exception raised when bulkhead rejects request.

    Raised when the maximum number of concurrent requests is reached and
    the queue is full, preventing resource exhaustion.
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int,
        queue_size: int,
    ) -> None:
        """Initialize bulkhead rejected error.

        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent requests
            queue_size: Queue size
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        super().__init__(
            f"Bulkhead '{name}' rejected request: "
            f"{max_concurrent} concurrent, {queue_size} queued"
        )


class ResilienceTimeoutError(ResilienceError):
    """Exception raised when operation times out.

    Raised when an operation exceeds its configured timeout duration,
    preventing indefinite blocking.
    """

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
    ) -> None:
        """Initialize resilience timeout error.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout value in seconds
        """
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s"
        )
