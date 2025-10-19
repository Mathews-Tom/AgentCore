"""
Rate Limiter

Redis-based distributed rate limiter with multiple algorithms and policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import redis.asyncio as aioredis
import structlog
from redis.asyncio.connection import ConnectionPool

from gateway.middleware.rate_limit_algorithms import (
    FixedWindowCounter,
    LeakyBucketCounter,
    RateLimitAlgorithm,
    SlidingWindowCounter,
    TokenBucketCounter,
)

logger = structlog.get_logger()


class RateLimitType(str, Enum):
    """Rate limit types for different scopes."""

    CLIENT_IP = "client_ip"
    ENDPOINT = "endpoint"
    USER = "user"
    GLOBAL = "global"


class RateLimitAlgorithmType(str, Enum):
    """Supported rate limiting algorithms."""

    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitPolicy:
    """
    Rate limit policy configuration.

    Defines the limits and behavior for a specific rate limit scope.
    """

    limit: int  # Maximum requests allowed
    window_seconds: int  # Time window in seconds
    algorithm: RateLimitAlgorithmType = RateLimitAlgorithmType.SLIDING_WINDOW
    burst_multiplier: float = 1.0  # Multiplier for burst capacity (token bucket)
    enabled: bool = True


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.

    Contains information about whether the request was allowed and metadata
    for client response headers.
    """

    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: int
    limit_type: RateLimitType
    key: str


class RateLimiter:
    """
    Redis-based distributed rate limiter.

    Supports multiple rate limiting algorithms and scopes (client, endpoint, user, global).
    Designed for high-performance with <1ms overhead.
    """

    def __init__(
        self,
        redis_url: str,
        default_algorithm: RateLimitAlgorithmType = RateLimitAlgorithmType.SLIDING_WINDOW,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            default_algorithm: Default algorithm to use if not specified in policy
        """
        self.redis_url = redis_url
        self.default_algorithm = default_algorithm

        self._client: aioredis.Redis[bytes] | None = None
        self._pool: ConnectionPool | None = None

        # Algorithm instances (cached for reuse)
        self._algorithms: dict[RateLimitAlgorithmType, RateLimitAlgorithm] = {
            RateLimitAlgorithmType.SLIDING_WINDOW: SlidingWindowCounter(),
            RateLimitAlgorithmType.FIXED_WINDOW: FixedWindowCounter(),
            RateLimitAlgorithmType.TOKEN_BUCKET: TokenBucketCounter(),
            RateLimitAlgorithmType.LEAKY_BUCKET: LeakyBucketCounter(),
        }

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        logger.info("Initializing rate limiter", redis_url=self.redis_url)

        # Create connection pool optimized for rate limiting
        self._pool = ConnectionPool.from_url(
            self.redis_url,
            decode_responses=False,
            max_connections=50,  # Higher pool for rate limiting
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )

        # Create Redis client
        self._client = aioredis.Redis(connection_pool=self._pool)

        # Verify connection
        await self._client.ping()

        logger.info("Rate limiter initialized successfully")

    async def close(self) -> None:
        """Close Redis connection and cleanup resources."""
        logger.info("Closing rate limiter")

        if self._client:
            await self._client.aclose()
            self._client = None

        if self._pool:
            await self._pool.aclose()
            self._pool = None

        logger.info("Rate limiter closed")

    @property
    def client(self) -> aioredis.Redis[bytes]:
        """Get Redis client instance."""
        if not self._client:
            raise RuntimeError("Rate limiter not initialized. Call initialize() first.")
        return self._client

    def _get_rate_limit_key(self, limit_type: RateLimitType, identifier: str) -> str:
        """
        Generate Redis key for rate limit.

        Args:
            limit_type: Type of rate limit
            identifier: Unique identifier (IP, user_id, endpoint, etc.)

        Returns:
            Redis key string
        """
        return f"ratelimit:{limit_type.value}:{identifier}"

    async def check_rate_limit(
        self,
        limit_type: RateLimitType,
        identifier: str,
        policy: RateLimitPolicy,
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limit.

        Args:
            limit_type: Type of rate limit to check
            identifier: Unique identifier for the limit scope
            policy: Rate limit policy to apply

        Returns:
            RateLimitResult with allow/deny decision and metadata
        """
        if not policy.enabled:
            # Rate limiting disabled for this policy
            return RateLimitResult(
                allowed=True,
                limit=policy.limit,
                remaining=policy.limit,
                reset_at=0,
                retry_after=0,
                limit_type=limit_type,
                key=identifier,
            )

        # Get rate limit key
        key = self._get_rate_limit_key(limit_type, identifier)

        # Get algorithm instance
        algorithm = self._algorithms.get(
            policy.algorithm, self._algorithms[self.default_algorithm]
        )

        # Check rate limit
        try:
            # Handle token bucket burst multiplier
            if policy.algorithm == RateLimitAlgorithmType.TOKEN_BUCKET:
                allowed, metadata = await algorithm.is_allowed(
                    key=key,
                    limit=policy.limit,
                    window_seconds=policy.window_seconds,
                    redis_client=self.client,
                    bucket_capacity=int(policy.limit * policy.burst_multiplier),
                )
            else:
                allowed, metadata = await algorithm.is_allowed(
                    key=key,
                    limit=policy.limit,
                    window_seconds=policy.window_seconds,
                    redis_client=self.client,
                )

            result = RateLimitResult(
                allowed=allowed,
                limit=policy.limit,
                remaining=metadata["remaining"],
                reset_at=metadata["reset_at"],
                retry_after=metadata["retry_after"],
                limit_type=limit_type,
                key=identifier,
            )

            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    limit_type=limit_type.value,
                    identifier=identifier,
                    limit=policy.limit,
                    window_seconds=policy.window_seconds,
                    retry_after=metadata["retry_after"],
                )

            return result

        except Exception as e:
            logger.error(
                "Rate limit check failed",
                error=str(e),
                limit_type=limit_type.value,
                identifier=identifier,
            )
            # Fail open on errors (allow request but log)
            return RateLimitResult(
                allowed=True,
                limit=policy.limit,
                remaining=0,
                reset_at=0,
                retry_after=0,
                limit_type=limit_type,
                key=identifier,
            )

    async def check_multiple_limits(
        self,
        checks: list[tuple[RateLimitType, str, RateLimitPolicy]],
    ) -> list[RateLimitResult]:
        """
        Check multiple rate limits concurrently.

        This is more efficient than checking limits sequentially when
        multiple limits need to be enforced (e.g., client + endpoint + user).

        Args:
            checks: List of (limit_type, identifier, policy) tuples

        Returns:
            List of RateLimitResult in same order as checks
        """
        import asyncio

        tasks = [
            self.check_rate_limit(limit_type, identifier, policy)
            for limit_type, identifier, policy in checks
        ]

        return await asyncio.gather(*tasks)

    async def reset_limit(self, limit_type: RateLimitType, identifier: str) -> bool:
        """
        Reset rate limit for a specific identifier.

        Useful for administrative overrides or testing.

        Args:
            limit_type: Type of rate limit to reset
            identifier: Unique identifier for the limit scope

        Returns:
            True if limit was reset, False otherwise
        """
        key = self._get_rate_limit_key(limit_type, identifier)

        try:
            # Delete all keys matching the pattern
            deleted = await self.client.delete(key)
            # Also delete auxiliary keys (e.g., token bucket state)
            await self.client.delete(f"{key}:last_leak")

            logger.info(
                "Rate limit reset",
                limit_type=limit_type.value,
                identifier=identifier,
                deleted=deleted,
            )

            return deleted > 0

        except Exception as e:
            logger.error(
                "Rate limit reset failed",
                error=str(e),
                limit_type=limit_type.value,
                identifier=identifier,
            )
            return False

    async def get_limit_info(
        self,
        limit_type: RateLimitType,
        identifier: str,
    ) -> dict[str, Any] | None:
        """
        Get current rate limit info without consuming a request.

        Args:
            limit_type: Type of rate limit
            identifier: Unique identifier for the limit scope

        Returns:
            Dictionary with current limit state, or None if not found
        """
        key = self._get_rate_limit_key(limit_type, identifier)

        try:
            # Check if key exists
            exists = await self.client.exists(key)

            if not exists:
                return None

            # Get TTL
            ttl = await self.client.ttl(key)

            # Try to get count (works for sorted set and string)
            count = 0
            key_type = await self.client.type(key)

            if key_type == b"zset":
                count = await self.client.zcard(key)
            elif key_type == b"string":
                count_bytes = await self.client.get(key)
                if count_bytes:
                    count = int(count_bytes)

            return {
                "limit_type": limit_type.value,
                "identifier": identifier,
                "current_count": count,
                "ttl_seconds": ttl,
                "key_type": key_type.decode("utf-8"),
            }

        except Exception as e:
            logger.error(
                "Failed to get limit info",
                error=str(e),
                limit_type=limit_type.value,
                identifier=identifier,
            )
            return None

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on rate limiter.

        Returns:
            Dictionary with health status information
        """
        try:
            # Ping Redis
            await self.client.ping()

            # Count rate limit keys
            keys = await self.client.keys("ratelimit:*")

            return {
                "status": "healthy",
                "active_limits": len(keys),
                "redis_url": self.redis_url,
                "algorithms": [algo.value for algo in RateLimitAlgorithmType],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
