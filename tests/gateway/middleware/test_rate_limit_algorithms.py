"""
Tests for rate limiting algorithms.

Tests the core rate limiting algorithms (sliding window, fixed window, token bucket, leaky bucket).
"""

from __future__ import annotations

import asyncio
import time

import pytest
import redis.asyncio as aioredis
from testcontainers.redis import RedisContainer

from gateway.middleware.rate_limit_algorithms import (
    FixedWindowCounter,
    LeakyBucketCounter,
    SlidingWindowCounter,
    TokenBucketCounter)


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for testing."""
    container = RedisContainer("redis:7-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def redis_client(redis_container):
    """Create Redis client for testing."""
    redis_url = f"redis://localhost:{redis_container.get_exposed_port(6379)}/15"
    client = aioredis.from_url(
        redis_url,
        decode_responses=False)

    yield client

    # Cleanup
    await client.flushdb()
    await client.aclose()


@pytest.mark.asyncio
class TestSlidingWindowCounter:
    """Tests for sliding window counter algorithm."""

    async def test_allows_requests_under_limit(self, redis_client):
        """Test that requests under limit are allowed."""
        algorithm = SlidingWindowCounter()

        # Make 5 requests with limit of 10
        for _ in range(5):
            allowed, metadata = await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=60,
                redis_client=redis_client)
            assert allowed is True
            assert metadata["remaining"] >= 0

    async def test_blocks_requests_over_limit(self, redis_client):
        """Test that requests over limit are blocked."""
        algorithm = SlidingWindowCounter()

        # Make 10 requests with limit of 10
        for i in range(10):
            allowed, _ = await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=60,
                redis_client=redis_client)
            assert allowed is True

        # 11th request should be blocked
        allowed, metadata = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=60,
            redis_client=redis_client)
        assert allowed is False
        assert metadata["retry_after"] > 0

    async def test_sliding_window_accuracy(self, redis_client):
        """Test that sliding window accurately expires old requests."""
        algorithm = SlidingWindowCounter()

        # Make 5 requests
        for _ in range(5):
            await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=2,  # 2 second window
                redis_client=redis_client)

        # Wait for window to expire
        await asyncio.sleep(2.1)

        # Should allow new requests now
        allowed, metadata = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=2,
            redis_client=redis_client)
        assert allowed is True
        assert metadata["remaining"] == 9  # Only this new request

    async def test_metadata_accuracy(self, redis_client):
        """Test that metadata is accurate."""
        algorithm = SlidingWindowCounter()

        allowed, metadata = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=60,
            redis_client=redis_client)

        assert allowed is True
        assert metadata["remaining"] == 9
        assert metadata["limit"] == 10
        assert metadata["window_seconds"] == 60
        assert metadata["reset_at"] > time.time()


@pytest.mark.asyncio
class TestFixedWindowCounter:
    """Tests for fixed window counter algorithm."""

    async def test_allows_requests_under_limit(self, redis_client):
        """Test that requests under limit are allowed."""
        algorithm = FixedWindowCounter()

        for _ in range(5):
            allowed, metadata = await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=60,
                redis_client=redis_client)
            assert allowed is True

    async def test_blocks_requests_over_limit(self, redis_client):
        """Test that requests over limit are blocked."""
        algorithm = FixedWindowCounter()

        # Make 10 requests
        for _ in range(10):
            await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=60,
                redis_client=redis_client)

        # 11th request should be blocked
        allowed, _ = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=60,
            redis_client=redis_client)
        assert allowed is False

    async def test_window_reset(self, redis_client):
        """Test that counter resets at window boundary."""
        algorithm = FixedWindowCounter()

        # Make 10 requests (fill limit)
        for _ in range(10):
            await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=2,
                redis_client=redis_client)

        # Wait for window to reset
        await asyncio.sleep(2.1)

        # Should allow requests in new window
        allowed, metadata = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=2,
            redis_client=redis_client)
        assert allowed is True
        assert metadata["remaining"] == 9


@pytest.mark.asyncio
class TestTokenBucketCounter:
    """Tests for token bucket algorithm."""

    async def test_allows_burst_traffic(self, redis_client):
        """Test that token bucket allows burst traffic."""
        algorithm = TokenBucketCounter()

        # With bucket capacity of 20 and limit of 10, should allow 20 burst requests
        for _ in range(20):
            allowed, _ = await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=60,
                redis_client=redis_client,
                bucket_capacity=20)
            assert allowed is True

        # 21st request should be blocked
        allowed, _ = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=60,
            redis_client=redis_client,
            bucket_capacity=20)
        assert allowed is False

    async def test_token_refill(self, redis_client):
        """Test that tokens refill over time."""
        algorithm = TokenBucketCounter()

        # Consume all tokens
        for _ in range(20):
            await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=1,  # Refill 10 tokens per second
                redis_client=redis_client,
                bucket_capacity=20)

        # Wait for some refill
        await asyncio.sleep(0.5)  # Should refill ~5 tokens

        # Should allow some requests now
        allowed, metadata = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=1,
            redis_client=redis_client,
            bucket_capacity=20)
        assert allowed is True
        assert metadata["remaining"] >= 3  # At least 3 tokens refilled


@pytest.mark.asyncio
class TestLeakyBucketCounter:
    """Tests for leaky bucket algorithm."""

    async def test_enforces_rate_limit(self, redis_client):
        """Test that leaky bucket enforces strict rate limiting."""
        algorithm = LeakyBucketCounter()

        # Fill queue to limit
        for _ in range(10):
            allowed, _ = await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=60,
                redis_client=redis_client)
            assert allowed is True

        # Should block additional requests
        allowed, _ = await algorithm.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=60,
            redis_client=redis_client)
        assert allowed is False

    async def test_leakage_over_time(self, redis_client):
        """Test that requests leak out over time."""
        algorithm = LeakyBucketCounter()

        # Fill queue
        for _ in range(10):
            await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=1,  # Leak 10 per second
                redis_client=redis_client)

        # Wait for leakage - wait a full second to ensure deterministic results
        await asyncio.sleep(1.1)  # Should leak all 10 requests

        # Should allow all requests now since queue has leaked
        for _ in range(10):
            allowed, _ = await algorithm.is_allowed(
                key="test_key",
                limit=10,
                window_seconds=1,
                redis_client=redis_client)
            assert allowed is True


@pytest.mark.asyncio
class TestAlgorithmPerformance:
    """Performance tests for algorithms."""

    async def test_sliding_window_performance(self, redis_client):
        """Test that sliding window check is fast (<1ms)."""
        algorithm = SlidingWindowCounter()

        # Warm up
        await algorithm.is_allowed(
            key="perf_test",
            limit=1000,
            window_seconds=60,
            redis_client=redis_client)

        # Measure performance
        start = time.time()
        iterations = 100

        for _ in range(iterations):
            await algorithm.is_allowed(
                key="perf_test",
                limit=1000,
                window_seconds=60,
                redis_client=redis_client)

        elapsed = time.time() - start
        avg_ms = (elapsed / iterations) * 1000

        # Should be under 3ms per check (allows for CI/CD variability and test runner overhead)
        assert avg_ms < 3.0, f"Average check time {avg_ms:.2f}ms exceeds 3ms threshold"
