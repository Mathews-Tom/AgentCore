"""Resilience manager tests."""

from __future__ import annotations

import asyncio

import pytest

from agentcore.integration.resilience.exceptions import CircuitBreakerOpenError
from agentcore.integration.resilience.manager import (
    ResilienceManager,
    ResilienceRegistry)
from agentcore.integration.resilience.models import (
    BulkheadConfig,
    CircuitBreakerConfig,
    ResilienceConfig,
    TimeoutConfig)


class TestResilienceManager:
    """Test resilience manager functionality."""

    @pytest.fixture
    def full_config(self) -> ResilienceConfig:
        """Create full resilience configuration."""
        return ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(
                name="test_circuit",
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=1.0),
            bulkhead=BulkheadConfig(
                name="test_bulkhead",
                max_concurrent_requests=2,
                queue_size=2,
                queue_timeout_seconds=1.0),
            timeout=TimeoutConfig(
                name="test_timeout",
                timeout_seconds=1.0),
            enable_fallback=False)

    @pytest.fixture
    async def manager(
        self, full_config: ResilienceConfig
    ) -> ResilienceManager:
        """Create resilience manager instance."""
        mgr = ResilienceManager(full_config)
        await mgr.initialize()
        return mgr

    async def test_successful_request(
        self, manager: ResilienceManager
    ) -> None:
        """Test successful request through all patterns."""

        async def operation() -> str:
            return "success"

        result = await manager.execute(operation)
        assert result == "success"

    async def test_circuit_breaker_integration(
        self, manager: ResilienceManager
    ) -> None:
        """Test circuit breaker is applied."""

        async def fail_operation() -> None:
            raise ValueError("test error")

        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await manager.execute(fail_operation)

        # Circuit should be open
        async def success_operation() -> str:
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            await manager.execute(success_operation)

    async def test_bulkhead_integration(self) -> None:
        """Test bulkhead is applied."""
        # Create config with only bulkhead (no circuit breaker)
        config = ResilienceConfig(
            bulkhead=BulkheadConfig(
                name="test_only_bulkhead",
                max_concurrent_requests=2,
                queue_size=2))

        manager = ResilienceManager(config)
        await manager.initialize()

        async def fast_operation() -> str:
            await asyncio.sleep(0.1)
            return "success"

        # Execute requests sequentially to verify bulkhead works
        results = []
        for _ in range(3):
            result = await manager.execute(fast_operation)
            results.append(result)

        # All should succeed
        assert all(r == "success" for r in results)

    async def test_timeout_integration(
        self, manager: ResilienceManager
    ) -> None:
        """Test timeout is applied."""

        async def very_slow_operation() -> str:
            await asyncio.sleep(5.0)
            return "success"

        with pytest.raises(Exception):  # Timeout or circuit breaker
            await manager.execute(very_slow_operation)

    async def test_fallback_handler(self) -> None:
        """Test fallback handler is invoked on failure."""
        fallback_called = False

        async def fallback_handler() -> str:
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(
                name="test_fallback_circuit",
                failure_threshold=2),
            enable_fallback=True)

        manager = ResilienceManager(config, fallback_handler)
        await manager.initialize()

        async def fail_operation() -> str:
            raise ValueError("test error")

        # First two failures should open circuit
        for _ in range(2):
            result = await manager.execute(fail_operation)
            # Should get fallback result
            assert result == "fallback_result"

        assert fallback_called

    async def test_circuit_breaker_only(self) -> None:
        """Test with only circuit breaker enabled."""
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(
                name="test_cb_only",
                failure_threshold=2))

        manager = ResilienceManager(config)
        await manager.initialize()

        async def operation() -> str:
            return "success"

        result = await manager.execute(operation)
        assert result == "success"

    async def test_bulkhead_only(self) -> None:
        """Test with only bulkhead enabled."""
        config = ResilienceConfig(
            bulkhead=BulkheadConfig(
                name="test_bh_only",
                max_concurrent_requests=2))

        manager = ResilienceManager(config)
        await manager.initialize()

        async def operation() -> str:
            return "success"

        result = await manager.execute(operation)
        assert result == "success"

    async def test_timeout_only(self) -> None:
        """Test with only timeout enabled."""
        config = ResilienceConfig(
            timeout=TimeoutConfig(
                name="test_timeout_only",
                timeout_seconds=1.0))

        manager = ResilienceManager(config)
        await manager.initialize()

        async def fast_operation() -> str:
            await asyncio.sleep(0.1)
            return "success"

        result = await manager.execute(fast_operation)
        assert result == "success"

    async def test_context_manager(self) -> None:
        """Test resilience manager as context manager."""
        # Use simple config to avoid circuit breaker interference
        config = ResilienceConfig(
            timeout=TimeoutConfig(
                name="test_context_timeout",
                timeout_seconds=1.0))

        async def operation() -> str:
            return "success"

        async with ResilienceManager(config) as manager:
            result = await manager.execute(operation)
            assert result == "success"


class TestResilienceRegistry:
    """Test resilience registry."""

    async def test_register(self) -> None:
        """Test registering resilience manager."""
        registry = ResilienceRegistry()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(name="test_register"))

        manager = registry.register("test_mgr", config)
        assert manager is not None

        # Should return same instance
        manager2 = registry.register("test_mgr", config)
        assert manager is manager2

    async def test_get(self) -> None:
        """Test getting manager by name."""
        registry = ResilienceRegistry()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(name="test_get"))

        registry.register("test_get_mgr", config)

        manager = registry.get("test_get_mgr")
        assert manager is not None

    async def test_remove(self) -> None:
        """Test removing manager."""
        registry = ResilienceRegistry()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(name="test_remove"))

        registry.register("test_remove_mgr", config)
        registry.remove("test_remove_mgr")

        manager = registry.get("test_remove_mgr")
        assert manager is None

    async def test_get_all(self) -> None:
        """Test getting all managers."""
        registry = ResilienceRegistry()

        for i in range(3):
            config = ResilienceConfig(
                circuit_breaker=CircuitBreakerConfig(name=f"test_{i}"))
            registry.register(f"mgr_{i}", config)

        managers = registry.get_all()
        assert len(managers) >= 3
