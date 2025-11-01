"""Tests for retry handler with exponential backoff."""

import asyncio
import time

import pytest

from agentcore.agent_runtime.services.retry_handler import (
    BackoffStrategy,
    CircuitBreaker,
    RetryHandler,
    retry_with_backoff,
)


class RetryTestException(Exception):
    """Custom exception for retry handler tests."""


@pytest.mark.asyncio
async def test_retry_success_first_attempt():
    """Test successful execution on first attempt."""
    call_count = 0

    async def successful_func():
        nonlocal call_count
        call_count += 1
        return "success"

    handler = RetryHandler(max_retries=3)
    result = await handler.retry(successful_func)

    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_success_after_failures():
    """Test successful execution after retries."""
    call_count = 0

    async def eventually_successful():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryTestException("Temporary failure")
        return "success"

    handler = RetryHandler(max_retries=5, base_delay=0.1)
    result = await handler.retry(
        eventually_successful,
        retryable_exceptions=(RetryTestException,),
    )

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    """Test retry exhaustion."""
    call_count = 0

    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise RetryTestException("Permanent failure")

    handler = RetryHandler(max_retries=3, base_delay=0.1)

    with pytest.raises(RetryTestException):
        await handler.retry(
            always_fails,
            retryable_exceptions=(RetryTestException,),
        )

    assert call_count == 4  # Initial + 3 retries


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test exponential backoff delays."""
    handler = RetryHandler(
        max_retries=5,
        base_delay=1.0,
        strategy=BackoffStrategy.EXPONENTIAL,
        jitter=False,  # Disable jitter for predictable testing
    )

    # Test delay calculation
    assert handler.calculate_delay(0) == 1.0  # 1 * 2^0
    assert handler.calculate_delay(1) == 2.0  # 1 * 2^1
    assert handler.calculate_delay(2) == 4.0  # 1 * 2^2
    assert handler.calculate_delay(3) == 8.0  # 1 * 2^3


@pytest.mark.asyncio
async def test_linear_backoff():
    """Test linear backoff delays."""
    handler = RetryHandler(
        max_retries=5,
        base_delay=1.0,
        strategy=BackoffStrategy.LINEAR,
        jitter=False,
    )

    assert handler.calculate_delay(0) == 1.0  # 1 * 1
    assert handler.calculate_delay(1) == 2.0  # 1 * 2
    assert handler.calculate_delay(2) == 3.0  # 1 * 3
    assert handler.calculate_delay(3) == 4.0  # 1 * 4


@pytest.mark.asyncio
async def test_fixed_backoff():
    """Test fixed backoff delays."""
    handler = RetryHandler(
        max_retries=5,
        base_delay=2.0,
        strategy=BackoffStrategy.FIXED,
        jitter=False,
    )

    assert handler.calculate_delay(0) == 2.0
    assert handler.calculate_delay(1) == 2.0
    assert handler.calculate_delay(2) == 2.0
    assert handler.calculate_delay(3) == 2.0


@pytest.mark.asyncio
async def test_max_delay_cap():
    """Test maximum delay cap."""
    handler = RetryHandler(
        max_retries=10,
        base_delay=1.0,
        max_delay=5.0,
        strategy=BackoffStrategy.EXPONENTIAL,
        jitter=False,
    )

    # Even with exponential backoff, delay should not exceed max_delay
    assert handler.calculate_delay(10) <= 5.0


@pytest.mark.asyncio
async def test_jitter():
    """Test jitter adds randomness."""
    handler = RetryHandler(
        max_retries=5,
        base_delay=1.0,
        strategy=BackoffStrategy.EXPONENTIAL,
        jitter=True,
    )

    # With jitter, delays should vary
    delays = [handler.calculate_delay(2) for _ in range(10)]

    # All delays should be positive
    assert all(d > 0 for d in delays)

    # Should have variation (not all same)
    assert len(set(delays)) > 1


@pytest.mark.asyncio
async def test_retry_callback():
    """Test on_retry callback."""
    callback_calls = []

    def on_retry_callback(exception, attempt, delay):
        callback_calls.append((exception, attempt, delay))

    call_count = 0

    async def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RetryTestException(f"Failure {call_count}")
        return "success"

    handler = RetryHandler(max_retries=5, base_delay=0.1)
    await handler.retry(
        fail_twice,
        retryable_exceptions=(RetryTestException,),
        on_retry=on_retry_callback,
    )

    # Should have 2 retry callbacks
    assert len(callback_calls) == 2
    assert all(isinstance(e, RetryTestException) for e, _, _ in callback_calls)


@pytest.mark.asyncio
async def test_non_retryable_exception():
    """Test that non-retryable exceptions are not retried."""
    call_count = 0

    async def raises_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Non-retryable error")

    handler = RetryHandler(max_retries=3, base_delay=0.1)

    with pytest.raises(ValueError):
        await handler.retry(
            raises_value_error,
            retryable_exceptions=(RetryTestException,),  # Only retry RetryTestException
        )

    # Should only be called once (no retries)
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_convenience():
    """Test retry_with_backoff convenience function."""
    call_count = 0

    async def eventually_works():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RetryTestException("Temporary failure")
        return "success"

    result = await retry_with_backoff(
        eventually_works,
        max_retries=3,
        base_delay=0.1,
        strategy=BackoffStrategy.EXPONENTIAL,
        retryable_exceptions=(RetryTestException,),
    )

    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_closed():
    """Test circuit breaker in closed state."""
    call_count = 0

    async def successful_func():
        nonlocal call_count
        call_count += 1
        return "success"

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    # Circuit should be closed, calls should succeed
    assert breaker.state == "closed"
    result = await breaker.call(successful_func)
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    """Test circuit breaker opens after failures."""
    call_count = 0

    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise RetryTestException("Failure")

    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        expected_exception=RetryTestException,
    )

    # Make calls until circuit opens
    for i in range(3):
        with pytest.raises(RetryTestException):
            await breaker.call(always_fails)

    # Circuit should now be open
    assert breaker.state == "open"
    assert call_count == 3

    # Further calls should fail immediately without executing function
    with pytest.raises(Exception, match="Circuit breaker is open"):
        await breaker.call(always_fails)

    # Call count should not increase (circuit is open)
    assert call_count == 3


@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    """Test circuit breaker recovery after timeout."""
    call_count = 0

    async def conditionally_fails():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise RetryTestException("Failure")
        return "success"

    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=0.5,
        expected_exception=RetryTestException,
    )

    # Open the circuit
    for i in range(3):
        with pytest.raises(RetryTestException):
            await breaker.call(conditionally_fails)

    assert breaker.state == "open"

    # Wait for recovery timeout
    await asyncio.sleep(0.6)

    # Circuit should transition to half_open and allow one test call
    result = await breaker.call(conditionally_fails)
    assert result == "success"
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_reset():
    """Test manual circuit breaker reset."""
    async def always_fails():
        raise RetryTestException("Failure")

    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=10.0,
        expected_exception=RetryTestException,
    )

    # Open the circuit
    for i in range(2):
        with pytest.raises(RetryTestException):
            await breaker.call(always_fails)

    assert breaker.state == "open"

    # Manual reset
    breaker.reset()

    assert breaker.state == "closed"
    assert breaker.failure_count == 0
