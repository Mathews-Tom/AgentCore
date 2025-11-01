"""Tests for Redis-based rate limiter."""

import asyncio

import pytest
from testcontainers.redis import RedisContainer

from agentcore.agent_runtime.services.rate_limiter import (
    RateLimitExceeded,
    RateLimiter,
)


@pytest.fixture(scope="module")
def redis_container():
    """Start Redis container for tests."""
    with RedisContainer("redis:7-alpine") as redis:
        yield redis


@pytest.fixture
async def rate_limiter(redis_container):
    """Create rate limiter with test Redis."""
    # Get connection details
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    redis_url = f"redis://{host}:{port}/0"

    limiter = RateLimiter(redis_url=redis_url)
    await limiter.connect()
    yield limiter
    await limiter.disconnect()


@pytest.mark.asyncio
async def test_rate_limit_basic(rate_limiter: RateLimiter):
    """Test basic rate limiting."""
    tool_id = "test_tool"
    limit = 3
    window = 1

    # Should allow first 3 requests
    for i in range(limit):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # 4th request should be rate limited
    with pytest.raises(RateLimitExceeded) as exc_info:
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    assert exc_info.value.tool_id == tool_id
    assert exc_info.value.limit == limit
    assert exc_info.value.window_seconds == window
    assert exc_info.value.retry_after > 0


@pytest.mark.asyncio
async def test_rate_limit_window_reset(rate_limiter: RateLimiter):
    """Test rate limit resets after window."""
    tool_id = "test_tool_reset"
    limit = 2
    window = 1

    # Use up the limit
    for i in range(limit):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Should be rate limited
    with pytest.raises(RateLimitExceeded):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Wait for window to reset
    await asyncio.sleep(window + 0.1)

    # Should allow requests again
    await rate_limiter.check_rate_limit(tool_id, limit, window)


@pytest.mark.asyncio
async def test_rate_limit_per_identifier(rate_limiter: RateLimiter):
    """Test per-identifier rate limiting."""
    tool_id = "test_tool_identifier"
    limit = 2
    window = 1

    # Agent 1 uses up their limit
    for i in range(limit):
        await rate_limiter.check_rate_limit(tool_id, limit, window, identifier="agent1")

    # Agent 1 should be rate limited
    with pytest.raises(RateLimitExceeded):
        await rate_limiter.check_rate_limit(tool_id, limit, window, identifier="agent1")

    # Agent 2 should still be able to make requests
    await rate_limiter.check_rate_limit(tool_id, limit, window, identifier="agent2")


@pytest.mark.asyncio
async def test_get_remaining(rate_limiter: RateLimiter):
    """Test getting remaining rate limit."""
    tool_id = "test_tool_remaining"
    limit = 5
    window = 2

    # Initially should have full limit
    status = await rate_limiter.get_remaining(tool_id, limit, window)
    assert status["limit"] == limit
    assert status["remaining"] == limit
    assert status["window_seconds"] == window

    # Use 2 requests
    await rate_limiter.check_rate_limit(tool_id, limit, window)
    await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Should have 3 remaining
    status = await rate_limiter.get_remaining(tool_id, limit, window)
    assert status["remaining"] == limit - 2


@pytest.mark.asyncio
async def test_reset_rate_limit(rate_limiter: RateLimiter):
    """Test resetting rate limit."""
    tool_id = "test_tool_reset_manual"
    limit = 1
    window = 10

    # Use up the limit
    await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Should be rate limited
    with pytest.raises(RateLimitExceeded):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Reset the limit
    await rate_limiter.reset(tool_id)

    # Should allow requests again
    await rate_limiter.check_rate_limit(tool_id, limit, window)


@pytest.mark.asyncio
async def test_sliding_window(rate_limiter: RateLimiter):
    """Test sliding window algorithm."""
    tool_id = "test_sliding_window"
    limit = 3
    window = 2

    # Make 3 requests at t=0
    for i in range(limit):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Should be rate limited
    with pytest.raises(RateLimitExceeded):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Wait 1 second (halfway through window)
    await asyncio.sleep(1)

    # Should still be rate limited (requests still in window)
    with pytest.raises(RateLimitExceeded):
        await rate_limiter.check_rate_limit(tool_id, limit, window)

    # Wait another 1.5 seconds (total 2.5s, past window)
    await asyncio.sleep(1.5)

    # Should allow new requests
    await rate_limiter.check_rate_limit(tool_id, limit, window)


@pytest.mark.asyncio
async def test_concurrent_requests(rate_limiter: RateLimiter):
    """Test rate limiting with concurrent requests."""
    tool_id = "test_concurrent"
    limit = 5
    window = 1

    # Launch 10 concurrent requests
    tasks = [
        rate_limiter.check_rate_limit(tool_id, limit, window)
        for _ in range(10)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Should have exactly 'limit' successes and rest failures
    successes = sum(1 for r in results if not isinstance(r, Exception))
    failures = sum(1 for r in results if isinstance(r, RateLimitExceeded))

    assert successes == limit
    assert failures == 10 - limit
