"""Tests for quota manager implementation."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from testcontainers.redis import RedisContainer

from agentcore.agent_runtime.services.quota_manager import (
    QuotaExceeded,
    QuotaManager,
)


@pytest.fixture(scope="module")
def redis_container():
    """Fixture to start and stop Redis container."""
    with RedisContainer("redis:7-alpine") as redis:
        yield redis


@pytest.fixture
async def quota_manager(redis_container):
    """Fixture for quota manager with test Redis."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    redis_url = f"redis://{host}:{port}/0"
    manager = QuotaManager(redis_url=redis_url, key_prefix="test:quota:")
    await manager.connect()
    yield manager
    await manager.disconnect()


@pytest.fixture(autouse=True)
async def cleanup_redis(quota_manager):
    """Clean up Redis keys after each test."""
    yield
    # Clean up all test keys
    if quota_manager._redis:
        keys = await quota_manager._redis.keys("test:quota:*")
        if keys:
            await quota_manager._redis.delete(*keys)


@pytest.mark.asyncio
async def test_check_quota_daily_basic(quota_manager: QuotaManager):
    """Test basic daily quota checking."""
    tool_id = "test_tool"
    daily_quota = 10

    # Should allow up to quota limit
    for i in range(daily_quota):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
        )

    # Next request should exceed quota
    with pytest.raises(QuotaExceeded) as exc_info:
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
        )

    assert exc_info.value.tool_id == tool_id
    assert exc_info.value.quota_type == "daily"
    assert exc_info.value.limit == daily_quota


@pytest.mark.asyncio
async def test_check_quota_monthly_basic(quota_manager: QuotaManager):
    """Test basic monthly quota checking."""
    tool_id = "test_tool"
    monthly_quota = 5

    # Should allow up to quota limit
    for i in range(monthly_quota):
        await quota_manager.check_quota(
            tool_id=tool_id,
            monthly_quota=monthly_quota,
        )

    # Next request should exceed quota
    with pytest.raises(QuotaExceeded) as exc_info:
        await quota_manager.check_quota(
            tool_id=tool_id,
            monthly_quota=monthly_quota,
        )

    assert exc_info.value.tool_id == tool_id
    assert exc_info.value.quota_type == "monthly"
    assert exc_info.value.limit == monthly_quota


@pytest.mark.asyncio
async def test_check_quota_both_limits(quota_manager: QuotaManager):
    """Test quota checking with both daily and monthly limits."""
    tool_id = "test_tool"
    daily_quota = 3
    monthly_quota = 10

    # Should allow up to daily quota
    for i in range(daily_quota):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
            monthly_quota=monthly_quota,
        )

    # Next request should exceed daily quota (not monthly yet)
    with pytest.raises(QuotaExceeded) as exc_info:
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
            monthly_quota=monthly_quota,
        )

    assert exc_info.value.quota_type == "daily"
    assert exc_info.value.limit == daily_quota


@pytest.mark.asyncio
async def test_check_quota_per_identifier(quota_manager: QuotaManager):
    """Test quota tracking per identifier (agent/user)."""
    tool_id = "test_tool"
    daily_quota = 5

    # Agent 1 uses quota
    for i in range(daily_quota):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
            identifier="agent1",
        )

    # Agent 1 should be blocked
    with pytest.raises(QuotaExceeded):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
            identifier="agent1",
        )

    # Agent 2 should still have quota
    await quota_manager.check_quota(
        tool_id=tool_id,
        daily_quota=daily_quota,
        identifier="agent2",
    )


@pytest.mark.asyncio
async def test_get_quota_status_daily(quota_manager: QuotaManager):
    """Test getting daily quota status."""
    tool_id = "test_tool"
    daily_quota = 10

    # Use 3 out of 10
    for i in range(3):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
        )

    status = await quota_manager.get_quota_status(
        tool_id=tool_id,
        daily_quota=daily_quota,
    )

    assert status["daily_limit"] == daily_quota
    assert status["daily_used"] == 3
    assert status["daily_remaining"] == 7
    assert status["daily_reset_at"] is not None
    # Parse and verify reset time is in the future
    reset_time = datetime.fromisoformat(status["daily_reset_at"])
    assert reset_time > datetime.now(UTC)

    # Monthly should be None
    assert status["monthly_limit"] is None
    assert status["monthly_used"] == 0
    assert status["monthly_remaining"] is None


@pytest.mark.asyncio
async def test_get_quota_status_monthly(quota_manager: QuotaManager):
    """Test getting monthly quota status."""
    tool_id = "test_tool"
    monthly_quota = 100

    # Use 25 out of 100
    for i in range(25):
        await quota_manager.check_quota(
            tool_id=tool_id,
            monthly_quota=monthly_quota,
        )

    status = await quota_manager.get_quota_status(
        tool_id=tool_id,
        monthly_quota=monthly_quota,
    )

    assert status["monthly_limit"] == monthly_quota
    assert status["monthly_used"] == 25
    assert status["monthly_remaining"] == 75
    assert status["monthly_reset_at"] is not None
    # Parse and verify reset time is in the future
    reset_time = datetime.fromisoformat(status["monthly_reset_at"])
    assert reset_time > datetime.now(UTC)


@pytest.mark.asyncio
async def test_get_quota_status_both(quota_manager: QuotaManager):
    """Test getting quota status with both daily and monthly limits."""
    tool_id = "test_tool"
    daily_quota = 10
    monthly_quota = 100

    # Use 5 requests
    for i in range(5):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
            monthly_quota=monthly_quota,
        )

    status = await quota_manager.get_quota_status(
        tool_id=tool_id,
        daily_quota=daily_quota,
        monthly_quota=monthly_quota,
    )

    # Check daily
    assert status["daily_limit"] == daily_quota
    assert status["daily_used"] == 5
    assert status["daily_remaining"] == 5

    # Check monthly
    assert status["monthly_limit"] == monthly_quota
    assert status["monthly_used"] == 5
    assert status["monthly_remaining"] == 95


@pytest.mark.asyncio
async def test_reset_quota_daily(quota_manager: QuotaManager):
    """Test resetting daily quota."""
    tool_id = "test_tool"
    daily_quota = 5

    # Use entire quota
    for i in range(daily_quota):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
        )

    # Should be blocked
    with pytest.raises(QuotaExceeded):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
        )

    # Reset daily quota
    await quota_manager.reset_quota(tool_id, quota_type="daily")

    # Should now be allowed
    await quota_manager.check_quota(
        tool_id=tool_id,
        daily_quota=daily_quota,
    )


@pytest.mark.asyncio
async def test_reset_quota_monthly(quota_manager: QuotaManager):
    """Test resetting monthly quota."""
    tool_id = "test_tool"
    monthly_quota = 3

    # Use entire quota
    for i in range(monthly_quota):
        await quota_manager.check_quota(
            tool_id=tool_id,
            monthly_quota=monthly_quota,
        )

    # Should be blocked
    with pytest.raises(QuotaExceeded):
        await quota_manager.check_quota(
            tool_id=tool_id,
            monthly_quota=monthly_quota,
        )

    # Reset monthly quota
    await quota_manager.reset_quota(tool_id, quota_type="monthly")

    # Should now be allowed
    await quota_manager.check_quota(
        tool_id=tool_id,
        monthly_quota=monthly_quota,
    )


@pytest.mark.asyncio
async def test_reset_quota_all(quota_manager: QuotaManager):
    """Test resetting all quotas."""
    tool_id = "test_tool"
    daily_quota = 2
    monthly_quota = 5

    # Use quotas
    for i in range(2):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=daily_quota,
            monthly_quota=monthly_quota,
        )

    # Reset all quotas
    await quota_manager.reset_quota(tool_id)

    # Check status - should be 0 used
    status = await quota_manager.get_quota_status(
        tool_id=tool_id,
        daily_quota=daily_quota,
        monthly_quota=monthly_quota,
    )

    assert status["daily_used"] == 0
    assert status["monthly_used"] == 0


@pytest.mark.asyncio
async def test_quota_key_format_daily(quota_manager: QuotaManager):
    """Test Redis key format for daily quota."""
    tool_id = "test_tool"
    identifier = "agent123"

    key = quota_manager._get_quota_key(tool_id, "daily", identifier)

    # Should contain tool_id, daily, date, and identifier
    assert tool_id in key
    assert "daily" in key
    assert identifier in key
    # Should have today's date in YYYY-MM-DD format
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    assert today in key


@pytest.mark.asyncio
async def test_quota_key_format_monthly(quota_manager: QuotaManager):
    """Test Redis key format for monthly quota."""
    tool_id = "test_tool"
    identifier = "agent123"

    key = quota_manager._get_quota_key(tool_id, "monthly", identifier)

    # Should contain tool_id, monthly, month, and identifier
    assert tool_id in key
    assert "monthly" in key
    assert identifier in key
    # Should have this month in YYYY-MM format
    this_month = datetime.now(UTC).strftime("%Y-%m")
    assert this_month in key


@pytest.mark.asyncio
async def test_quota_reset_time_daily(quota_manager: QuotaManager):
    """Test daily quota reset time calculation."""
    reset_time = quota_manager._get_reset_time("daily")

    now = datetime.now(UTC)
    # Should be midnight tomorrow
    expected = datetime.combine(
        now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )

    assert reset_time == expected


@pytest.mark.asyncio
async def test_quota_reset_time_monthly(quota_manager: QuotaManager):
    """Test monthly quota reset time calculation."""
    reset_time = quota_manager._get_reset_time("monthly")

    now = datetime.now(UTC)
    # Should be start of next month
    if now.month == 12:
        expected = datetime(now.year + 1, 1, 1, tzinfo=UTC)
    else:
        expected = datetime(now.year, now.month + 1, 1, tzinfo=UTC)

    assert reset_time == expected


@pytest.mark.asyncio
async def test_concurrent_quota_checks(quota_manager: QuotaManager):
    """Test concurrent quota checking maintains consistency."""
    tool_id = "test_tool"
    daily_quota = 10

    # Run 15 concurrent requests (only 10 should succeed)
    tasks = [
        quota_manager.check_quota(tool_id=tool_id, daily_quota=daily_quota)
        for _ in range(15)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes and failures
    successes = sum(1 for r in results if not isinstance(r, Exception))
    failures = sum(1 for r in results if isinstance(r, QuotaExceeded))

    # Should have exactly daily_quota successes
    assert successes == daily_quota
    assert failures == 5


@pytest.mark.asyncio
async def test_quota_ttl(quota_manager: QuotaManager):
    """Test that quota keys have appropriate TTL."""
    tool_id = "test_tool"
    daily_quota = 10

    # Use quota
    await quota_manager.check_quota(
        tool_id=tool_id,
        daily_quota=daily_quota,
    )

    # Get the Redis key
    key = quota_manager._get_quota_key(tool_id, "daily", None)

    # Check TTL (should be less than or equal to seconds until midnight)
    ttl = await quota_manager._redis.ttl(key)
    max_ttl = (
        datetime.combine(
            datetime.now(UTC).date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=UTC,
        )
        - datetime.now(UTC)
    ).total_seconds()

    assert 0 < ttl <= max_ttl + 1  # +1 for timing variance


@pytest.mark.asyncio
async def test_unlimited_quota(quota_manager: QuotaManager):
    """Test that None quota means unlimited."""
    tool_id = "test_tool"

    # Should allow any number of requests with None quota
    for i in range(100):
        await quota_manager.check_quota(
            tool_id=tool_id,
            daily_quota=None,
            monthly_quota=None,
        )

    # Status should show unlimited
    status = await quota_manager.get_quota_status(
        tool_id=tool_id,
        daily_quota=None,
        monthly_quota=None,
    )

    assert status["daily_limit"] is None
    assert status["daily_remaining"] is None
    assert status["monthly_limit"] is None
    assert status["monthly_remaining"] is None
