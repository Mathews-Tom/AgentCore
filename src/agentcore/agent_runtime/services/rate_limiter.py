"""Redis-based rate limiter for tool execution.

Implements token bucket and sliding window rate limiting algorithms.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from redis import asyncio as aioredis

logger = structlog.get_logger()


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        tool_id: str,
        limit: int,
        window_seconds: int,
        retry_after: float,
    ):
        """Initialize rate limit exceeded exception.

        Args:
            tool_id: Tool identifier
            limit: Rate limit (requests per window)
            window_seconds: Time window in seconds
            retry_after: Seconds until retry is allowed
        """
        self.tool_id = tool_id
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for tool {tool_id}: "
            f"{limit} requests per {window_seconds}s. "
            f"Retry after {retry_after:.2f}s"
        )


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""

    # Lua script for atomic rate limit check
    # Returns: 1 if allowed, 0 if rate limited
    _RATE_LIMIT_SCRIPT = """
    local key = KEYS[1]
    local window_start_ms = tonumber(ARGV[1])
    local now_ms = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local window_seconds = tonumber(ARGV[4])
    local unique_id = ARGV[5]

    -- Remove old entries outside the window
    redis.call('ZREMRANGEBYSCORE', key, 0, window_start_ms)

    -- Count current entries
    local count = redis.call('ZCARD', key)

    -- Check if limit exceeded
    if count >= limit then
        return 0  -- Rate limited
    end

    -- Add current request with unique member (timestamp:uuid)
    redis.call('ZADD', key, now_ms, unique_id)

    -- Set expiration
    redis.call('EXPIRE', key, window_seconds)

    return 1  -- Allowed
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentcore:ratelimit:",
    ):
        """Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("rate_limiter_connected", redis_url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("rate_limiter_disconnected")

    async def check_rate_limit(
        self,
        tool_id: str,
        limit: int,
        window_seconds: int,
        identifier: str | None = None,
    ) -> None:
        """Check if rate limit is exceeded.

        Args:
            tool_id: Tool identifier
            limit: Maximum requests per window
            window_seconds: Time window in seconds
            identifier: Optional identifier (e.g., agent_id) for per-user limits

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self._redis:
            await self.connect()

        # Build Redis key
        key_parts = [self.key_prefix, tool_id]
        if identifier:
            key_parts.append(identifier)
        key = ":".join(key_parts)

        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=window_seconds)
        window_start_ms = int(window_start.timestamp() * 1000)
        now_ms = int(now.timestamp() * 1000)

        # Generate unique ID for this request (timestamp:uuid)
        unique_id = f"{now_ms}:{uuid.uuid4().hex[:8]}"

        # Execute Lua script atomically
        # Returns 1 if allowed, 0 if rate limited
        # Use eval instead of evalsha to ensure script execution
        result = await self._redis.eval(
            self._RATE_LIMIT_SCRIPT,
            1,  # Number of keys
            key,  # KEYS[1]
            window_start_ms,  # ARGV[1]
            now_ms,  # ARGV[2]
            limit,  # ARGV[3]
            window_seconds,  # ARGV[4]
            unique_id,  # ARGV[5]
        )

        if result == 0:
            # Rate limited - get current count for logging
            count = await self._redis.zcard(key)

            # Calculate retry_after based on oldest entry in window
            oldest_entries = await self._redis.zrange(key, 0, 0, withscores=True)
            if oldest_entries:
                oldest_timestamp_ms = oldest_entries[0][1]
                oldest_time = datetime.fromtimestamp(oldest_timestamp_ms / 1000, tz=UTC)
                retry_after = (
                    oldest_time + timedelta(seconds=window_seconds) - now
                ).total_seconds()
                retry_after = max(0, retry_after)
            else:
                retry_after = window_seconds

            logger.warning(
                "rate_limit_exceeded",
                tool_id=tool_id,
                identifier=identifier,
                limit=limit,
                window_seconds=window_seconds,
                count=count,
                retry_after=retry_after,
            )

            raise RateLimitExceeded(
                tool_id=tool_id,
                limit=limit,
                window_seconds=window_seconds,
                retry_after=retry_after,
            )

        # Get count for logging (after adding the request)
        count = await self._redis.zcard(key)

        logger.debug(
            "rate_limit_check_passed",
            tool_id=tool_id,
            identifier=identifier,
            count=count,
            limit=limit,
        )

    async def get_remaining(
        self,
        tool_id: str,
        limit: int,
        window_seconds: int,
        identifier: str | None = None,
    ) -> dict[str, Any]:
        """Get rate limit status.

        Args:
            tool_id: Tool identifier
            limit: Maximum requests per window
            window_seconds: Time window in seconds
            identifier: Optional identifier for per-user limits

        Returns:
            Dictionary with rate limit status:
            - limit: Maximum requests per window
            - remaining: Remaining requests in current window
            - reset_at: When the window resets (ISO format)
        """
        if not self._redis:
            await self.connect()

        key_parts = [self.key_prefix, tool_id]
        if identifier:
            key_parts.append(identifier)
        key = ":".join(key_parts)

        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=window_seconds)
        window_start_ms = int(window_start.timestamp() * 1000)

        # Remove old entries and count
        async with self._redis.pipeline() as pipe:
            pipe.zremrangebyscore(key, 0, window_start_ms)
            pipe.zcard(key)
            results = await pipe.execute()

        count = results[1]
        remaining = max(0, limit - count)

        # Calculate reset time
        oldest_entries = await self._redis.zrange(key, 0, 0, withscores=True)
        if oldest_entries:
            oldest_timestamp_ms = oldest_entries[0][1]
            oldest_time = datetime.fromtimestamp(oldest_timestamp_ms / 1000, tz=UTC)
            reset_at = oldest_time + timedelta(seconds=window_seconds)
        else:
            reset_at = now + timedelta(seconds=window_seconds)

        return {
            "limit": limit,
            "remaining": remaining,
            "reset_at": reset_at.isoformat(),
            "window_seconds": window_seconds,
        }

    async def reset(
        self,
        tool_id: str,
        identifier: str | None = None,
    ) -> None:
        """Reset rate limit for a tool.

        Args:
            tool_id: Tool identifier
            identifier: Optional identifier for per-user limits
        """
        if not self._redis:
            await self.connect()

        key_parts = [self.key_prefix, tool_id]
        if identifier:
            key_parts.append(identifier)
        key = ":".join(key_parts)

        await self._redis.delete(key)
        logger.info("rate_limit_reset", tool_id=tool_id, identifier=identifier)


# Global rate limiter instance
_global_rate_limiter: RateLimiter | None = None


def get_rate_limiter(redis_url: str | None = None) -> RateLimiter:
    """Get global rate limiter instance.

    Args:
        redis_url: Optional Redis URL (uses default if not provided)

    Returns:
        Global RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        if redis_url:
            _global_rate_limiter = RateLimiter(redis_url=redis_url)
        else:
            _global_rate_limiter = RateLimiter()
    return _global_rate_limiter
