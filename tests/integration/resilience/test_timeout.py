"""Timeout manager tests."""

from __future__ import annotations

import asyncio

import pytest

from agentcore.integration.resilience.exceptions import ResilienceTimeoutError
from agentcore.integration.resilience.models import TimeoutConfig
from agentcore.integration.resilience.timeout import (
    TimeoutManager,
    with_timeout_direct)


class TestTimeoutManager:
    """Test timeout manager functionality."""

    @pytest.fixture
    def config(self) -> TimeoutConfig:
        """Create test configuration."""
        return TimeoutConfig(
            name="test_timeout",
            timeout_seconds=0.5)

    @pytest.fixture
    async def manager(self, config: TimeoutConfig) -> TimeoutManager:
        """Create timeout manager instance."""
        return TimeoutManager(config)

    async def test_successful_request(
        self, manager: TimeoutManager
    ) -> None:
        """Test successful request within timeout."""

        async def fast_operation() -> str:
            await asyncio.sleep(0.1)
            return "success"

        result = await manager.execute(fast_operation)
        assert result == "success"

    async def test_timeout_exceeded(
        self, manager: TimeoutManager
    ) -> None:
        """Test timeout is enforced."""

        async def slow_operation() -> str:
            await asyncio.sleep(2.0)
            return "success"

        with pytest.raises(ResilienceTimeoutError) as exc_info:
            await manager.execute(slow_operation)

        assert exc_info.value.operation == "test_timeout"
        assert exc_info.value.timeout_seconds == 0.5

    async def test_timeout_override(
        self, manager: TimeoutManager
    ) -> None:
        """Test timeout can be overridden."""

        async def medium_operation() -> str:
            await asyncio.sleep(0.6)
            return "success"

        # Should timeout with default (0.5s)
        with pytest.raises(ResilienceTimeoutError):
            await manager.execute(medium_operation)

        # Should succeed with override (1.0s)
        result = await manager.execute(
            medium_operation, timeout_override=1.0
        )
        assert result == "success"

    async def test_with_timeout_direct(self) -> None:
        """Test direct timeout utility function."""

        async def fast_operation() -> str:
            await asyncio.sleep(0.1)
            return "success"

        result = await with_timeout_direct(
            fast_operation,
            timeout_seconds=0.5,
            operation_name="test_op")
        assert result == "success"

    async def test_with_timeout_direct_exceeds(self) -> None:
        """Test direct timeout enforcement."""

        async def slow_operation() -> str:
            await asyncio.sleep(2.0)
            return "success"

        with pytest.raises(ResilienceTimeoutError) as exc_info:
            await with_timeout_direct(
                slow_operation,
                timeout_seconds=0.5,
                operation_name="test_op")

        assert exc_info.value.operation == "test_op"
        assert exc_info.value.timeout_seconds == 0.5
