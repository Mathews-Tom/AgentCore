"""
Tests for Circuit Breaker Implementation

Covers circuit breaker states, transitions, error handling, and recovery.
"""

from __future__ import annotations

import asyncio
import pytest

from gateway.routing.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
)


@pytest.fixture
def circuit_config() -> CircuitBreakerConfig:
    """Create circuit breaker configuration for testing."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=1.0,
        expected_exception=Exception,
    )


@pytest.fixture
def circuit_breaker(circuit_config: CircuitBreakerConfig) -> CircuitBreaker:
    """Create circuit breaker for testing."""
    return CircuitBreaker("test-service", circuit_config)


def test_initial_state(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit breaker starts in closed state."""
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.is_closed
    assert not circuit_breaker.is_open
    assert not circuit_breaker.is_half_open


def test_successful_call(circuit_breaker: CircuitBreaker) -> None:
    """Test successful function call through circuit breaker."""

    def successful_func() -> str:
        return "success"

    result = circuit_breaker.call(successful_func)
    assert result == "success"
    assert circuit_breaker.stats.total_calls == 1
    assert circuit_breaker.stats.total_successes == 1
    assert circuit_breaker.stats.total_failures == 0


def test_failed_call(circuit_breaker: CircuitBreaker) -> None:
    """Test failed function call through circuit breaker."""

    def failing_func() -> None:
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        circuit_breaker.call(failing_func)

    assert circuit_breaker.stats.total_calls == 1
    assert circuit_breaker.stats.total_failures == 1
    assert circuit_breaker.stats.failure_count == 1


def test_open_after_threshold(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit opens after reaching failure threshold."""

    def failing_func() -> None:
        raise ValueError("Test error")

    # Fail enough times to open circuit (threshold = 3)
    for _ in range(3):
        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

    assert circuit_breaker.is_open
    assert circuit_breaker.stats.failure_count == 3


def test_fail_fast_when_open(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit breaker fails fast when open."""

    def failing_func() -> None:
        raise ValueError("Test error")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

    # Next call should fail immediately without calling function
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        circuit_breaker.call(lambda: "success")

    assert exc_info.value.service == "test-service"
    assert exc_info.value.retry_after > 0


@pytest.mark.asyncio
async def test_async_successful_call(circuit_breaker: CircuitBreaker) -> None:
    """Test successful async function call."""

    async def async_func() -> str:
        return "async success"

    result = await circuit_breaker.call_async(async_func)
    assert result == "async success"
    assert circuit_breaker.stats.total_successes == 1


@pytest.mark.asyncio
async def test_async_failed_call(circuit_breaker: CircuitBreaker) -> None:
    """Test failed async function call."""

    async def async_failing_func() -> None:
        raise ValueError("Async error")

    with pytest.raises(ValueError):
        await circuit_breaker.call_async(async_failing_func)

    assert circuit_breaker.stats.total_failures == 1


@pytest.mark.asyncio
async def test_half_open_transition(circuit_breaker: CircuitBreaker) -> None:
    """Test transition from open to half-open after timeout."""

    def failing_func() -> None:
        raise ValueError("Test error")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

    assert circuit_breaker.is_open

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Next call should attempt recovery (half-open)
    def successful_func() -> str:
        return "recovery"

    result = circuit_breaker.call(successful_func)
    assert result == "recovery"
    assert circuit_breaker.is_half_open


@pytest.mark.asyncio
async def test_half_open_success_closes(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit closes after success threshold in half-open state."""

    def failing_func() -> None:
        raise ValueError("Test error")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Succeed enough times to close (threshold = 2)
    def successful_func() -> str:
        return "success"

    circuit_breaker.call(successful_func)
    assert circuit_breaker.is_half_open

    circuit_breaker.call(successful_func)
    assert circuit_breaker.is_closed
    assert circuit_breaker.stats.failure_count == 0


@pytest.mark.asyncio
async def test_half_open_failure_reopens(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit reopens immediately on failure in half-open state."""

    def failing_func() -> None:
        raise ValueError("Test error")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # First call moves to half-open, second call fails and reopens
    def successful_func() -> str:
        return "success"

    circuit_breaker.call(successful_func)
    assert circuit_breaker.is_half_open

    # Any failure in half-open immediately opens circuit
    with pytest.raises(ValueError):
        circuit_breaker.call(failing_func)

    assert circuit_breaker.is_open


def test_manual_reset(circuit_breaker: CircuitBreaker) -> None:
    """Test manual circuit breaker reset."""

    def failing_func() -> None:
        raise ValueError("Test error")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

    assert circuit_breaker.is_open

    # Manual reset
    circuit_breaker.reset()
    assert circuit_breaker.is_closed
    assert circuit_breaker.stats.failure_count == 0


def test_get_stats(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit breaker statistics."""

    def successful_func() -> str:
        return "success"

    def failing_func() -> None:
        raise ValueError("Test error")

    # Mix of successes and failures
    circuit_breaker.call(successful_func)
    circuit_breaker.call(successful_func)

    with pytest.raises(ValueError):
        circuit_breaker.call(failing_func)

    stats = circuit_breaker.get_stats()

    assert stats["service"] == "test-service"
    assert stats["state"] == CircuitState.CLOSED
    assert stats["total_calls"] == 3
    assert stats["total_successes"] == 2
    assert stats["total_failures"] == 1
    assert stats["failure_rate"] == 1 / 3


def test_circuit_breaker_registry() -> None:
    """Test circuit breaker registry for managing multiple services."""
    registry = CircuitBreakerRegistry()

    # Get breakers for different services
    breaker1 = registry.get("service-1")
    breaker2 = registry.get("service-2")

    assert breaker1.service_name == "service-1"
    assert breaker2.service_name == "service-2"
    assert breaker1 is not breaker2

    # Same service returns same breaker
    breaker1_again = registry.get("service-1")
    assert breaker1 is breaker1_again


def test_registry_get_all_stats() -> None:
    """Test getting stats for all circuit breakers in registry."""
    registry = CircuitBreakerRegistry()

    # Create breakers and make some calls
    for i in range(3):
        breaker = registry.get(f"service-{i}")
        breaker.call(lambda: "success")

    all_stats = registry.get_all_stats()
    assert len(all_stats) == 3
    assert all([s["total_calls"] == 1 for s in all_stats])


def test_registry_reset_all() -> None:
    """Test resetting all circuit breakers in registry."""
    # Use custom config with lower threshold
    config = CircuitBreakerConfig(failure_threshold=2)
    registry = CircuitBreakerRegistry(default_config=config)

    # Open some circuits
    def failing_func() -> None:
        raise ValueError("Test error")

    for i in range(2):
        breaker = registry.get(f"service-{i}")
        for _ in range(2):  # Match threshold
            with pytest.raises(ValueError):
                breaker.call(failing_func)

    # All should be open
    assert all(registry.get(f"service-{i}").is_open for i in range(2))

    # Reset all
    registry.reset_all()

    # All should be closed
    assert all(registry.get(f"service-{i}").is_closed for i in range(2))


def test_custom_exception_type() -> None:
    """Test circuit breaker with custom exception type."""

    class CustomError(Exception):
        pass

    config = CircuitBreakerConfig(
        failure_threshold=2,
        expected_exception=CustomError,
    )
    breaker = CircuitBreaker("test", config)

    # CustomError should be caught
    def custom_error_func() -> None:
        raise CustomError("Custom")

    with pytest.raises(CustomError):
        breaker.call(custom_error_func)

    assert breaker.stats.failure_count == 1

    # Other exceptions should not be caught by circuit breaker
    def other_error_func() -> None:
        raise ValueError("Other")

    with pytest.raises(ValueError):
        breaker.call(other_error_func)

    # Failure count should not increase
    assert breaker.stats.failure_count == 1


def test_concurrent_calls(circuit_breaker: CircuitBreaker) -> None:
    """Test circuit breaker handles concurrent calls correctly."""
    import threading

    results: list[str] = []
    errors: list[Exception] = []

    def make_call() -> None:
        try:
            result = circuit_breaker.call(lambda: "success")
            results.append(result)
        except Exception as e:
            errors.append(e)

    # Make 10 concurrent calls
    threads = [threading.Thread(target=make_call) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 10
    assert len(errors) == 0
    assert circuit_breaker.stats.total_calls == 10
    assert circuit_breaker.stats.total_successes == 10
