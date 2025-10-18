"""
Rate Limiting Algorithms

Implements various rate limiting algorithms for distributed systems using Redis.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger()


class RateLimitAlgorithm(ABC):
    """Base class for rate limiting algorithms."""

    @abstractmethod
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        redis_client: aioredis.Redis[bytes],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Unique identifier for the rate limit (e.g., user_id, ip_address)
            limit: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
            redis_client: Redis client instance

        Returns:
            Tuple of (is_allowed, metadata) where metadata contains:
                - remaining: Number of requests remaining
                - reset_at: Unix timestamp when the limit resets
                - retry_after: Seconds to wait before retry (if blocked)
        """
        pass


class SlidingWindowCounter(RateLimitAlgorithm):
    """
    Sliding window counter algorithm.

    Provides accurate rate limiting by tracking individual request timestamps.
    Uses Redis Sorted Sets (ZSET) for efficient timestamp-based operations.

    Time Complexity: O(log N) for add, O(log N) for count
    Space Complexity: O(N) where N is the number of requests in the window

    This is the most accurate algorithm but has higher memory overhead.
    """

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        redis_client: aioredis.Redis[bytes],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed using sliding window counter.

        Implementation:
        1. Remove expired entries (older than window_seconds)
        2. Count remaining entries
        3. If under limit, add current timestamp
        4. Return result with metadata
        """
        now = time.time()
        window_start = now - window_seconds

        # Redis Lua script for atomic operations
        # This ensures thread-safety across distributed instances
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])

        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

        -- Count current entries
        local current = redis.call('ZCARD', key)

        local allowed = 0
        local remaining = limit - current

        if current < limit then
            -- Add current request timestamp
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, window_seconds)
            allowed = 1
            remaining = limit - current - 1
        end

        -- Get oldest entry for reset calculation
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local reset_at = now + window_seconds
        if #oldest > 0 then
            reset_at = tonumber(oldest[2]) + window_seconds
        end

        return {allowed, remaining, reset_at}
        """

        # Execute Lua script atomically
        result = await redis_client.eval(
            lua_script,
            1,
            key,
            str(now),
            str(window_start),
            str(limit),
            str(window_seconds),
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = float(result[2])
        retry_after = max(0, int(reset_at - now)) if not allowed else 0

        metadata = {
            "remaining": max(0, remaining),
            "reset_at": int(reset_at),
            "retry_after": retry_after,
            "limit": limit,
            "window_seconds": window_seconds,
        }

        return allowed, metadata


class FixedWindowCounter(RateLimitAlgorithm):
    """
    Fixed window counter algorithm.

    Simpler and more memory-efficient than sliding window, but less accurate.
    Uses Redis STRING with increment operations.

    Time Complexity: O(1)
    Space Complexity: O(1)

    Note: Suffers from boundary issues where requests at window edges
    can result in double the rate limit being allowed briefly.
    """

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        redis_client: aioredis.Redis[bytes],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed using fixed window counter.

        Implementation:
        1. Calculate current window key based on timestamp
        2. Increment counter for current window
        3. Set expiration if new window
        4. Check if under limit
        """
        now = time.time()
        window_key = f"{key}:{int(now // window_seconds)}"

        # Increment counter atomically
        current = await redis_client.incr(window_key)

        # Set expiration on first request in window
        if current == 1:
            await redis_client.expire(window_key, window_seconds)

        allowed = current <= limit
        remaining = max(0, limit - current)
        reset_at = int((int(now // window_seconds) + 1) * window_seconds)
        retry_after = max(0, reset_at - int(now)) if not allowed else 0

        metadata = {
            "remaining": remaining,
            "reset_at": reset_at,
            "retry_after": retry_after,
            "limit": limit,
            "window_seconds": window_seconds,
        }

        return allowed, metadata


class TokenBucketCounter(RateLimitAlgorithm):
    """
    Token bucket algorithm.

    Allows burst traffic up to bucket capacity while maintaining average rate.
    Uses Redis HASH to store bucket state (tokens, last_refill).

    Time Complexity: O(1)
    Space Complexity: O(1)

    Good for APIs that need to handle occasional bursts while maintaining
    a steady average rate.
    """

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        redis_client: aioredis.Redis[bytes],
        bucket_capacity: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed using token bucket algorithm.

        Args:
            key: Unique identifier for the rate limit
            limit: Refill rate (tokens per window)
            window_seconds: Refill interval in seconds
            redis_client: Redis client instance
            bucket_capacity: Maximum bucket size (defaults to limit * 2 for burst)

        Implementation:
        1. Calculate tokens to refill based on time elapsed
        2. Refill bucket up to capacity
        3. Try to consume 1 token
        4. Return result with metadata
        """
        now = time.time()
        capacity = bucket_capacity or (limit * 2)  # Allow 2x burst by default
        refill_rate = limit / window_seconds  # Tokens per second

        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])

        -- Get current state
        local tokens = tonumber(redis.call('HGET', key, 'tokens') or capacity)
        local last_refill = tonumber(redis.call('HGET', key, 'last_refill') or now)

        -- Calculate refill
        local time_passed = now - last_refill
        local tokens_to_add = time_passed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)

        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end

        -- Update state
        redis.call('HSET', key, 'tokens', tokens)
        redis.call('HSET', key, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- Keep state for 1 hour

        return {allowed, math.floor(tokens)}
        """

        result = await redis_client.eval(
            lua_script,
            1,
            key,
            str(now),
            str(capacity),
            str(refill_rate),
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = int(now + (capacity / refill_rate)) if not allowed else 0
        retry_after = max(0, int(1 / refill_rate)) if not allowed else 0

        metadata = {
            "remaining": remaining,
            "reset_at": reset_at,
            "retry_after": retry_after,
            "limit": limit,
            "window_seconds": window_seconds,
            "bucket_capacity": capacity,
        }

        return allowed, metadata


class LeakyBucketCounter(RateLimitAlgorithm):
    """
    Leaky bucket algorithm.

    Enforces strict rate limiting by processing requests at a constant rate.
    Uses Redis LIST as a queue.

    Time Complexity: O(1)
    Space Complexity: O(N) where N is queue size

    Good for strict rate control where bursts should be queued or rejected.
    """

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        redis_client: aioredis.Redis[bytes],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed using leaky bucket algorithm.

        Implementation:
        1. Remove leaked (processed) requests based on time elapsed
        2. Check queue size against limit
        3. Add request to queue if under limit
        4. Return result with metadata
        """
        now = time.time()
        leak_rate = limit / window_seconds  # Requests per second

        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local leak_rate = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])

        -- Get last leak time
        local last_leak = tonumber(redis.call('GET', key .. ':last_leak') or now)

        -- Calculate leakage
        local time_passed = now - last_leak
        local leaked = math.floor(time_passed * leak_rate)

        -- Get current queue size
        local queue_size = redis.call('LLEN', key)

        -- Apply leakage
        if leaked > 0 then
            for i = 1, math.min(leaked, queue_size) do
                redis.call('LPOP', key)
            end
            redis.call('SET', key .. ':last_leak', now)
            redis.call('EXPIRE', key .. ':last_leak', window_seconds)
        end

        -- Check queue size after leakage
        queue_size = redis.call('LLEN', key)

        local allowed = 0
        if queue_size < limit then
            redis.call('RPUSH', key, now)
            redis.call('EXPIRE', key, window_seconds)
            allowed = 1
        end

        return {allowed, limit - queue_size}
        """

        result = await redis_client.eval(
            lua_script,
            1,
            key,
            str(now),
            str(limit),
            str(leak_rate),
            str(window_seconds),
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = int(now + window_seconds)
        retry_after = max(0, int(1 / leak_rate)) if not allowed else 0

        metadata = {
            "remaining": max(0, remaining),
            "reset_at": reset_at,
            "retry_after": retry_after,
            "limit": limit,
            "window_seconds": window_seconds,
        }

        return allowed, metadata
