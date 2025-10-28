"""Bulkhead pattern tests."""

from __future__ import annotations

import asyncio

import pytest

from agentcore.integration.resilience.bulkhead import Bulkhead, BulkheadRegistry
from agentcore.integration.resilience.exceptions import (
    BulkheadRejectedError,
    ResilienceTimeoutError)
from agentcore.integration.resilience.models import BulkheadConfig


class TestBulkhead:
    """Test bulkhead functionality."""

    @pytest.fixture
    def config(self) -> BulkheadConfig:
        """Create test configuration."""
        return BulkheadConfig(
            name="test_bulkhead",
            max_concurrent_requests=2,
            queue_size=2,
            queue_timeout_seconds=1.0)

    @pytest.fixture
    async def bulkhead(self, config: BulkheadConfig) -> Bulkhead:
        """Create bulkhead instance."""
        return Bulkhead(config)

    async def test_successful_request(
        self, bulkhead: Bulkhead
    ) -> None:
        """Test successful request execution."""

        async def operation() -> str:
            return "success"

        result = await bulkhead.execute(operation)

        assert result == "success"
        assert bulkhead.metrics.total_requests == 1
        assert bulkhead.metrics.total_accepted == 1
        assert bulkhead.metrics.total_rejected == 0

    async def test_concurrent_limit(
        self, bulkhead: Bulkhead
    ) -> None:
        """Test concurrent request limiting."""

        async def slow_operation() -> str:
            await asyncio.sleep(0.5)
            return "success"

        # Start 2 concurrent requests (at limit)
        tasks = [bulkhead.execute(slow_operation) for _ in range(2)]

        # Complete tasks
        results = await asyncio.gather(*tasks)
        assert all(r == "success" for r in results)

        # Verify metrics tracked the executions
        assert bulkhead.metrics.total_accepted == 2

    async def test_queue_overflow_rejection(
        self, bulkhead: Bulkhead
    ) -> None:
        """Test rejection when queue is full."""

        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "success"

        # Start requests to fill concurrent slots + queue
        # 2 concurrent + 2 queued = 4 total
        # 5th request should be rejected
        tasks = [bulkhead.execute(slow_operation) for _ in range(5)]

        # Wait for rejection
        await asyncio.sleep(0.1)

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least one should be rejected
        rejections = sum(
            1 for r in results if isinstance(r, BulkheadRejectedError)
        )
        assert rejections >= 1
        assert bulkhead.metrics.total_rejected >= 1

    async def test_queue_timeout(
        self, bulkhead: Bulkhead
    ) -> None:
        """Test queue timeout behavior."""
        # Create bulkhead with very short queue timeout
        config = BulkheadConfig(
            name="test_timeout",
            max_concurrent_requests=1,
            queue_size=10,
            queue_timeout_seconds=0.1)
        bh = Bulkhead(config)

        async def long_operation() -> str:
            await asyncio.sleep(2.0)
            return "success"

        # Start one long operation (occupies the slot)
        task1 = asyncio.create_task(bh.execute(long_operation))

        # Wait to ensure first task is running
        await asyncio.sleep(0.05)

        # Try to execute second operation (should queue and timeout)
        with pytest.raises(ResilienceTimeoutError):
            await bh.execute(long_operation)

        # Cancel first task
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

    async def test_metrics_tracking(
        self, bulkhead: Bulkhead
    ) -> None:
        """Test metrics are tracked correctly."""

        async def operation() -> str:
            return "success"

        # Execute several requests
        for _ in range(5):
            await bulkhead.execute(operation)

        assert bulkhead.metrics.total_requests == 5
        assert bulkhead.metrics.total_accepted == 5
        assert bulkhead.metrics.total_rejected == 0
        assert bulkhead.metrics.current_concurrent == 0

    async def test_max_concurrent_tracking(
        self, bulkhead: Bulkhead
    ) -> None:
        """Test max concurrent requests tracking."""

        async def slow_operation() -> str:
            await asyncio.sleep(0.2)
            return "success"

        # Start 2 concurrent requests
        tasks = [bulkhead.execute(slow_operation) for _ in range(2)]

        # Complete tasks
        await asyncio.gather(*tasks)

        # Check max concurrent was tracked (should be at least 1)
        assert bulkhead.metrics.max_concurrent_seen >= 1


class TestBulkheadRegistry:
    """Test bulkhead registry."""

    async def test_get_or_create(self) -> None:
        """Test get_or_create returns same instance."""
        registry = BulkheadRegistry()

        config1 = BulkheadConfig(name="test1")
        config2 = BulkheadConfig(name="test1")

        bulkhead1 = await registry.get_or_create(config1)
        bulkhead2 = await registry.get_or_create(config2)

        assert bulkhead1 is bulkhead2

    async def test_get_by_name(self) -> None:
        """Test getting bulkhead by name."""
        registry = BulkheadRegistry()

        config = BulkheadConfig(name="test_get")
        await registry.get_or_create(config)

        bulkhead = await registry.get("test_get")
        assert bulkhead is not None
        assert bulkhead.config.name == "test_get"

    async def test_remove(self) -> None:
        """Test removing bulkhead."""
        registry = BulkheadRegistry()

        config = BulkheadConfig(name="test_remove")
        await registry.get_or_create(config)

        await registry.remove("test_remove")

        bulkhead = await registry.get("test_remove")
        assert bulkhead is None

    async def test_get_all_metrics(self) -> None:
        """Test getting all metrics."""
        registry = BulkheadRegistry()

        # Create multiple bulkheads
        for i in range(3):
            config = BulkheadConfig(name=f"test_metrics_{i}")
            await registry.get_or_create(config)

        metrics = registry.get_all_metrics()
        assert len(metrics) >= 3
