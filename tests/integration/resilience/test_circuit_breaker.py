"""Circuit breaker pattern tests."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from agentcore.integration.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry)
from agentcore.integration.resilience.exceptions import CircuitBreakerOpenError
from agentcore.integration.resilience.models import (
    CircuitBreakerConfig,
    CircuitBreakerState)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def config(self) -> CircuitBreakerConfig:
        """Create test configuration."""
        return CircuitBreakerConfig(
            name="test_breaker",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,
            half_open_max_requests=2)

    @pytest.fixture
    async def breaker(
        self, config: CircuitBreakerConfig
    ) -> CircuitBreaker:
        """Create circuit breaker instance."""
        return CircuitBreaker(config)

    async def test_initial_state_closed(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit breaker starts in CLOSED state."""
        assert breaker.metrics.state == CircuitBreakerState.CLOSED
        assert breaker.metrics.failure_count == 0
        assert breaker.metrics.success_count == 0

    async def test_successful_request(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test successful request increments success count."""

        async def success_operation() -> str:
            return "success"

        result = await breaker.call(success_operation)

        assert result == "success"
        assert breaker.metrics.state == CircuitBreakerState.CLOSED
        assert breaker.metrics.total_successes == 1
        assert breaker.metrics.success_count == 1
        assert breaker.metrics.failure_count == 0

    async def test_failed_request(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test failed request increments failure count."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await breaker.call(fail_operation)

        assert breaker.metrics.state == CircuitBreakerState.CLOSED
        assert breaker.metrics.total_failures == 1
        assert breaker.metrics.failure_count == 1
        assert breaker.metrics.success_count == 0
        assert breaker.metrics.last_failure_time is not None

    async def test_circuit_opens_on_threshold(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit opens when failure threshold is exceeded."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Trigger failures to reach threshold (3 failures)
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        assert breaker.metrics.state == CircuitBreakerState.OPEN
        assert breaker.metrics.failure_count == 3
        assert breaker.metrics.opened_at is not None

    async def test_circuit_rejects_when_open(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit rejects requests when OPEN."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        assert breaker.metrics.state == CircuitBreakerState.OPEN

        # Attempt request while OPEN
        async def success_operation() -> str:
            return "success"

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call(success_operation)

        assert exc_info.value.name == "test_breaker"
        assert exc_info.value.failure_count == 3
        assert breaker.metrics.total_rejections == 1

    async def test_circuit_transitions_to_half_open(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit transitions to HALF_OPEN after timeout."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        assert breaker.metrics.state == CircuitBreakerState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Next request should transition to HALF_OPEN
        async def success_operation() -> str:
            return "success"

        result = await breaker.call(success_operation)

        assert result == "success"
        assert breaker.metrics.state == CircuitBreakerState.HALF_OPEN

    async def test_half_open_closes_on_success_threshold(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test HALF_OPEN transitions to CLOSED after success threshold."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        # Wait for timeout
        await asyncio.sleep(1.1)

        async def success_operation() -> str:
            return "success"

        # First success (HALF_OPEN)
        await breaker.call(success_operation)
        assert breaker.metrics.state == CircuitBreakerState.HALF_OPEN

        # Second success (should close circuit)
        await breaker.call(success_operation)
        assert breaker.metrics.state == CircuitBreakerState.CLOSED
        assert breaker.metrics.success_count == 0  # Reset
        assert breaker.metrics.failure_count == 0  # Reset

    async def test_half_open_reopens_on_failure(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test HALF_OPEN transitions back to OPEN on failure."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Transition to HALF_OPEN with success
        async def success_operation() -> str:
            return "success"

        await breaker.call(success_operation)
        assert breaker.metrics.state == CircuitBreakerState.HALF_OPEN

        # Fail in HALF_OPEN (should reopen)
        with pytest.raises(ValueError):
            await breaker.call(fail_operation)

        assert breaker.metrics.state == CircuitBreakerState.OPEN

    async def test_half_open_request_limit(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test HALF_OPEN transitions to CLOSED after success threshold."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        # Wait for timeout
        await asyncio.sleep(1.1)

        async def success_operation() -> str:
            return "success"

        # Execute requests sequentially in HALF_OPEN
        # First success (transitions to HALF_OPEN)
        result1 = await breaker.call(success_operation)
        assert result1 == "success"

        # Second success (should close circuit based on success_threshold=2)
        result2 = await breaker.call(success_operation)
        assert result2 == "success"

        # After 2 successes, circuit should close
        assert breaker.metrics.state == CircuitBreakerState.CLOSED

    async def test_reset_circuit_breaker(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test reset functionality."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_operation)

        assert breaker.metrics.state == CircuitBreakerState.OPEN

        # Reset
        breaker.reset()

        assert breaker.metrics.state == CircuitBreakerState.CLOSED
        assert breaker.metrics.failure_count == 0
        assert breaker.metrics.success_count == 0
        assert breaker.metrics.total_requests == 0


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""

    async def test_get_or_create(self) -> None:
        """Test get_or_create returns same instance for same name."""
        registry = CircuitBreakerRegistry()

        config1 = CircuitBreakerConfig(name="test1")
        config2 = CircuitBreakerConfig(name="test1")

        breaker1 = await registry.get_or_create(config1)
        breaker2 = await registry.get_or_create(config2)

        assert breaker1 is breaker2

    async def test_get_by_name(self) -> None:
        """Test getting circuit breaker by name."""
        registry = CircuitBreakerRegistry()

        config = CircuitBreakerConfig(name="test_get")
        await registry.get_or_create(config)

        breaker = await registry.get("test_get")
        assert breaker is not None
        assert breaker.config.name == "test_get"

    async def test_remove(self) -> None:
        """Test removing circuit breaker."""
        registry = CircuitBreakerRegistry()

        config = CircuitBreakerConfig(name="test_remove")
        await registry.get_or_create(config)

        await registry.remove("test_remove")

        breaker = await registry.get("test_remove")
        assert breaker is None

    async def test_get_all_metrics(self) -> None:
        """Test getting all metrics."""
        registry = CircuitBreakerRegistry()

        # Create multiple breakers
        for i in range(3):
            config = CircuitBreakerConfig(name=f"test_metrics_{i}")
            await registry.get_or_create(config)

        metrics = registry.get_all_metrics()
        assert len(metrics) >= 3
        assert all(
            isinstance(name, str) for name in metrics.keys()
        )
