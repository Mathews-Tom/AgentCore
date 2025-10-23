"""Rate limiting with token bucket algorithm.

Implements distributed rate limiting using Redis for coordination across instances.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from redis.asyncio import Redis

from agentcore.integration.api.exceptions import APIRateLimitError
from agentcore.integration.api.models import RateLimitConfig

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter with Redis backend.

    Implements rate limiting using token bucket algorithm:
    - Tokens refill at a constant rate (requests_per_window / window_seconds)
    - Each request consumes one token
    - Requests are blocked when bucket is empty
    - Supports burst by allowing token accumulation up to burst_size
    """

    def __init__(
        self,
        config: RateLimitConfig,
        redis_client: Redis[Any] | None = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
            redis_client: Optional Redis client for distributed rate limiting
        """
        self.config = config
        self._redis = redis_client
        self._local_buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_update)

        # Calculate token refill rate
        self.tokens_per_second = config.requests_per_window / config.window_seconds
        self.max_tokens = config.burst_size or config.requests_per_window

        logger.info(
            "rate_limiter_initialized",
            requests_per_window=config.requests_per_window,
            window_seconds=config.window_seconds,
            max_tokens=self.max_tokens,
            distributed=redis_client is not None,
        )

    async def acquire(
        self,
        key: str,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """Acquire tokens from the rate limiter.

        Args:
            key: Rate limit key (e.g., endpoint, tenant_id)
            tokens: Number of tokens to acquire (default: 1)
            timeout: Maximum time to wait for tokens (seconds)

        Returns:
            True if tokens acquired, False if timeout exceeded

        Raises:
            APIRateLimitError: If rate limit exceeded and timeout is None
        """
        if not self.config.enabled:
            return True

        start_time = time.monotonic()

        while True:
            # Check if we can acquire tokens
            if self._redis:
                acquired = await self._acquire_distributed(key, tokens)
            else:
                acquired = await self._acquire_local(key, tokens)

            if acquired:
                logger.debug(
                    "rate_limit_acquired",
                    key=key,
                    tokens=tokens,
                )
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    logger.warning(
                        "rate_limit_timeout",
                        key=key,
                        tokens=tokens,
                        elapsed=elapsed,
                    )
                    return False

            # If no timeout, raise error immediately
            if timeout is None:
                retry_after = int(tokens / self.tokens_per_second)
                logger.warning(
                    "rate_limit_exceeded",
                    key=key,
                    tokens=tokens,
                    retry_after=retry_after,
                )
                raise APIRateLimitError(
                    f"Rate limit exceeded for key '{key}'",
                    retry_after=retry_after,
                    status_code=429,
                )

            # Wait before retry (adaptive backoff based on token deficit)
            wait_time = min(tokens / self.tokens_per_second, 1.0)
            await asyncio.sleep(wait_time)

    async def _acquire_local(self, key: str, tokens: int) -> bool:
        """Acquire tokens from local in-memory bucket.

        Args:
            key: Rate limit key
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        now = time.monotonic()

        # Get or create bucket
        if key not in self._local_buckets:
            self._local_buckets[key] = (float(self.max_tokens), now)

        current_tokens, last_update = self._local_buckets[key]

        # Refill tokens based on time elapsed
        time_elapsed = now - last_update
        refill_tokens = time_elapsed * self.tokens_per_second
        current_tokens = min(current_tokens + refill_tokens, self.max_tokens)

        # Try to consume tokens
        if current_tokens >= tokens:
            current_tokens -= tokens
            self._local_buckets[key] = (current_tokens, now)
            return True

        # Not enough tokens
        self._local_buckets[key] = (current_tokens, now)
        return False

    async def _acquire_distributed(self, key: str, tokens: int) -> bool:
        """Acquire tokens from distributed Redis bucket.

        Uses Redis Lua script for atomic token bucket operations.

        Args:
            key: Rate limit key
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        if not self._redis:
            return await self._acquire_local(key, tokens)

        # Redis key
        redis_key = f"rate_limit:{key}"

        # Lua script for atomic token bucket
        # Returns 1 if acquired, 0 if not enough tokens
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local tokens_per_second = tonumber(ARGV[2])
        local tokens_to_consume = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        local ttl = tonumber(ARGV[5])

        -- Get current bucket state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local current_tokens = tonumber(bucket[1]) or max_tokens
        local last_update = tonumber(bucket[2]) or now

        -- Refill tokens based on time elapsed
        local time_elapsed = now - last_update
        local refill = time_elapsed * tokens_per_second
        current_tokens = math.min(current_tokens + refill, max_tokens)

        -- Try to consume tokens
        if current_tokens >= tokens_to_consume then
            current_tokens = current_tokens - tokens_to_consume
            redis.call('HMSET', key, 'tokens', current_tokens, 'last_update', now)
            redis.call('EXPIRE', key, ttl)
            return 1
        else
            -- Update tokens but don't consume
            redis.call('HMSET', key, 'tokens', current_tokens, 'last_update', now)
            redis.call('EXPIRE', key, ttl)
            return 0
        end
        """

        now = time.time()
        ttl = self.config.window_seconds * 2  # Keep bucket for 2 windows

        try:
            result = await self._redis.eval(
                lua_script,
                1,
                redis_key,
                str(self.max_tokens),
                str(self.tokens_per_second),
                str(tokens),
                str(now),
                str(ttl),
            )
            return bool(result)
        except Exception as e:
            logger.error(
                "rate_limit_redis_error",
                key=key,
                error=str(e),
            )
            # Fall back to local rate limiting
            return await self._acquire_local(key, tokens)

    async def get_remaining(self, key: str) -> int:
        """Get remaining tokens for a key.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining tokens
        """
        if not self.config.enabled:
            return self.max_tokens

        if self._redis:
            return await self._get_remaining_distributed(key)
        return await self._get_remaining_local(key)

    async def _get_remaining_local(self, key: str) -> int:
        """Get remaining tokens from local bucket.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining tokens
        """
        now = time.monotonic()

        if key not in self._local_buckets:
            return self.max_tokens

        current_tokens, last_update = self._local_buckets[key]

        # Refill tokens based on time elapsed
        time_elapsed = now - last_update
        refill_tokens = time_elapsed * self.tokens_per_second
        current_tokens = min(current_tokens + refill_tokens, self.max_tokens)

        return int(current_tokens)

    async def _get_remaining_distributed(self, key: str) -> int:
        """Get remaining tokens from distributed Redis bucket.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining tokens
        """
        if not self._redis:
            return await self._get_remaining_local(key)

        redis_key = f"rate_limit:{key}"

        try:
            bucket = await self._redis.hmget(redis_key, "tokens", "last_update")
            if not bucket[0]:
                return self.max_tokens

            current_tokens = float(bucket[0])
            last_update = float(bucket[1]) if bucket[1] else time.time()

            # Refill tokens
            now = time.time()
            time_elapsed = now - last_update
            refill = time_elapsed * self.tokens_per_second
            current_tokens = min(current_tokens + refill, self.max_tokens)

            return int(current_tokens)
        except Exception as e:
            logger.error(
                "rate_limit_redis_error",
                key=key,
                error=str(e),
            )
            return self.max_tokens

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Rate limit key
        """
        if self._redis:
            redis_key = f"rate_limit:{key}"
            await self._redis.delete(redis_key)
        else:
            self._local_buckets.pop(key, None)

        logger.info("rate_limit_reset", key=key)


# Global rate limiter registry
_rate_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(
    config: RateLimitConfig,
    redis_client: Redis[Any] | None = None,
) -> RateLimiter:
    """Get or create a rate limiter instance.

    Args:
        config: Rate limit configuration
        redis_client: Optional Redis client

    Returns:
        RateLimiter instance
    """
    # Use config hash as key
    key = f"{config.requests_per_window}_{config.window_seconds}_{config.burst_size}"

    if key not in _rate_limiters:
        _rate_limiters[key] = RateLimiter(config, redis_client)

    return _rate_limiters[key]
