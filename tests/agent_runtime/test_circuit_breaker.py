"""Tests for circuit breaker implementation."""

import asyncio

import pytest

from agentcore.agent_runtime.models.error_types import CircuitBreakerConfig
from agentcore.agent_runtime.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState)


@pytest.fixture
def config() -> CircuitBreakerConfig:
    """Create test circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1.0,
        half_open_max_attempts=2)


@pytest.fixture
def breaker(config: CircuitBreakerConfig) -> CircuitBreaker:
    """Create test circuit breaker."""
    return CircuitBreaker("test_breaker", config)


@pytest.mark.asyncio
class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    async def test_initial_state(self, breaker: CircuitBreaker) -> None:
        """Test circuit breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    async def test_successful_execution(self, breaker: CircuitBreaker) -> None:
        """Test successful function execution."""

        async def success_func() -> str:
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    async def test_failure_recording(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test failure recording and circuit opening."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Execute failures up to threshold
        for i in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == config.failure_threshold

    async def test_circuit_open_blocks_execution(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test that open circuit blocks execution."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        # Further attempts should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await breaker.call(fail_func)

    async def test_half_open_transition(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test transition from open to half-open after timeout."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(config.timeout_seconds + 0.1)

        # Next attempt should transition to half-open
        # This happens internally during state check
        async def success_func() -> str:
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"

    async def test_circuit_closing_after_success(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test circuit closes after successful executions in half-open."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        async def success_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        # Wait for timeout
        await asyncio.sleep(config.timeout_seconds + 0.1)

        # Execute successful calls
        for _ in range(config.success_threshold):
            result = await breaker.call(success_func)
            assert result == "success"

        # Circuit should be closed
        assert breaker.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test that failures in half-open reopen the circuit."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        # Wait for timeout
        await asyncio.sleep(config.timeout_seconds + 0.1)

        # Fail in half-open state
        for _ in range(config.half_open_max_attempts):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        # Circuit should be open again
        assert breaker.state == CircuitState.OPEN

    async def test_reset(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test manual circuit reset."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    async def test_stats(
        self,
        breaker: CircuitBreaker,
        config: CircuitBreakerConfig) -> None:
        """Test statistics collection."""
        stats = breaker.get_stats()

        assert stats["name"] == "test_breaker"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["failure_count"] == 0
        assert "config" in stats

    async def test_sync_function_execution(
        self,
        breaker: CircuitBreaker) -> None:
        """Test execution of synchronous functions."""

        def sync_func() -> str:
            return "sync_result"

        result = await breaker.call(sync_func)
        assert result == "sync_result"


@pytest.mark.asyncio
class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""

    async def test_get_or_create_breaker(self) -> None:
        """Test getting or creating circuit breaker."""
        registry = CircuitBreakerRegistry()

        breaker1 = await registry.get_breaker("test1")
        assert breaker1.name == "test1"

        breaker2 = await registry.get_breaker("test1")
        assert breaker1 is breaker2

    async def test_remove_breaker(self) -> None:
        """Test removing circuit breaker."""
        registry = CircuitBreakerRegistry()

        await registry.get_breaker("test")
        assert "test" in registry.list_breakers()

        await registry.remove_breaker("test")
        assert "test" not in registry.list_breakers()

    async def test_reset_all(self) -> None:
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=2)

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Create and open multiple breakers
        for name in ["test1", "test2"]:
            breaker = await registry.get_breaker(name, config)
            for _ in range(2):
                with pytest.raises(ValueError):
                    await breaker.call(fail_func)
            assert breaker.state == CircuitState.OPEN

        # Reset all
        await registry.reset_all()

        # All should be closed
        for name in ["test1", "test2"]:
            breaker = await registry.get_breaker(name)
            assert breaker.state == CircuitState.CLOSED

    async def test_get_all_stats(self) -> None:
        """Test getting statistics for all breakers."""
        registry = CircuitBreakerRegistry()

        await registry.get_breaker("test1")
        await registry.get_breaker("test2")

        stats = registry.get_all_stats()
        assert "test1" in stats
        assert "test2" in stats

    async def test_list_breakers(self) -> None:
        """Test listing all breaker names."""
        registry = CircuitBreakerRegistry()

        await registry.get_breaker("test1")
        await registry.get_breaker("test2")

        breakers = registry.list_breakers()
        assert "test1" in breakers
        assert "test2" in breakers
        assert len(breakers) == 2
