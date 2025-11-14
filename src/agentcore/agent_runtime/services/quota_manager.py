"""Redis-based quota management for tool execution.

Implements daily and monthly quota tracking with automatic reset.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from redis import asyncio as aioredis

logger = structlog.get_logger()


class QuotaExceeded(Exception):
    """Exception raised when quota is exceeded."""

    def __init__(
        self,
        tool_id: str,
        quota_type: str,
        limit: int,
        reset_at: datetime,
    ):
        """Initialize quota exceeded exception.

        Args:
            tool_id: Tool identifier
            quota_type: Type of quota (daily or monthly)
            limit: Quota limit
            reset_at: When the quota resets
        """
        self.tool_id = tool_id
        self.quota_type = quota_type
        self.limit = limit
        self.reset_at = reset_at
        super().__init__(
            f"Quota exceeded for tool {tool_id}: "
            f"{quota_type} limit of {limit} reached. "
            f"Resets at {reset_at.isoformat()}"
        )


class QuotaManager:
    """Redis-based quota manager using atomic counters."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentcore:quota:",
    ):
        """Initialize quota manager.

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
            logger.info("quota_manager_connected", redis_url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("quota_manager_disconnected")

    def _get_quota_key(
        self,
        tool_id: str,
        quota_type: str,
        identifier: str | None = None,
    ) -> str:
        """Get Redis key for quota tracking.

        Args:
            tool_id: Tool identifier
            quota_type: Type of quota (daily or monthly)
            identifier: Optional identifier (e.g., agent_id) for per-user quotas

        Returns:
            Redis key for quota counter
        """
        now = datetime.now(UTC)

        if quota_type == "daily":
            # Key format: agentcore:quota:tool_id:daily:YYYY-MM-DD[:identifier]
            date_key = now.strftime("%Y-%m-%d")
        elif quota_type == "monthly":
            # Key format: agentcore:quota:tool_id:monthly:YYYY-MM[:identifier]
            date_key = now.strftime("%Y-%m")
        else:
            raise ValueError(f"Invalid quota type: {quota_type}")

        key_parts = [self.key_prefix, tool_id, quota_type, date_key]
        if identifier:
            key_parts.append(identifier)

        return ":".join(key_parts)

    def _get_reset_time(self, quota_type: str) -> datetime:
        """Calculate when quota resets.

        Args:
            quota_type: Type of quota (daily or monthly)

        Returns:
            Datetime when quota resets
        """
        now = datetime.now(UTC)

        if quota_type == "daily":
            # Reset at midnight UTC
            tomorrow = now.date() + timedelta(days=1)
            return datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)
        elif quota_type == "monthly":
            # Reset at start of next month
            if now.month == 12:
                next_month = datetime(now.year + 1, 1, 1, tzinfo=UTC)
            else:
                next_month = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
            return next_month
        else:
            raise ValueError(f"Invalid quota type: {quota_type}")

    def _get_ttl_seconds(self, quota_type: str) -> int:
        """Get TTL in seconds for quota key.

        Args:
            quota_type: Type of quota (daily or monthly)

        Returns:
            TTL in seconds
        """
        reset_time = self._get_reset_time(quota_type)
        now = datetime.now(UTC)
        return int((reset_time - now).total_seconds())

    async def check_quota(
        self,
        tool_id: str,
        daily_quota: int | None = None,
        monthly_quota: int | None = None,
        identifier: str | None = None,
    ) -> None:
        """Check if quota is exceeded and increment if within limits.

        Args:
            tool_id: Tool identifier
            daily_quota: Daily quota limit (None = unlimited)
            monthly_quota: Monthly quota limit (None = unlimited)
            identifier: Optional identifier for per-user quotas

        Raises:
            QuotaExceeded: If quota is exceeded
        """
        if not self._redis:
            await self.connect()

        # Check daily quota
        if daily_quota is not None:
            await self._check_and_increment(
                tool_id=tool_id,
                quota_type="daily",
                limit=daily_quota,
                identifier=identifier,
            )

        # Check monthly quota
        if monthly_quota is not None:
            await self._check_and_increment(
                tool_id=tool_id,
                quota_type="monthly",
                limit=monthly_quota,
                identifier=identifier,
            )

    async def _check_and_increment(
        self,
        tool_id: str,
        quota_type: str,
        limit: int,
        identifier: str | None = None,
    ) -> None:
        """Check quota and increment counter atomically.

        Args:
            tool_id: Tool identifier
            quota_type: Type of quota (daily or monthly)
            limit: Quota limit
            identifier: Optional identifier for per-user quotas

        Raises:
            QuotaExceeded: If quota is exceeded
        """
        key = self._get_quota_key(tool_id, quota_type, identifier)
        ttl_seconds = self._get_ttl_seconds(quota_type)

        # Atomic increment with check
        async with self._redis.pipeline() as pipe:
            while True:
                try:
                    # Watch key for changes
                    await pipe.watch(key)

                    # Get current count
                    current = await pipe.get(key)
                    count = int(current) if current else 0

                    # Check if quota exceeded
                    if count >= limit:
                        await pipe.unwatch()
                        reset_at = self._get_reset_time(quota_type)

                        logger.warning(
                            "quota_exceeded",
                            tool_id=tool_id,
                            quota_type=quota_type,
                            identifier=identifier,
                            count=count,
                            limit=limit,
                            reset_at=reset_at.isoformat(),
                        )

                        raise QuotaExceeded(
                            tool_id=tool_id,
                            quota_type=quota_type,
                            limit=limit,
                            reset_at=reset_at,
                        )

                    # Increment counter with TTL
                    pipe.multi()
                    pipe.incr(key)
                    pipe.expire(key, ttl_seconds)
                    await pipe.execute()

                    # Success - break out of retry loop
                    break

                except aioredis.WatchError:
                    # Key was modified during transaction, retry
                    logger.debug("quota_check_retry", key=key)
                    continue

        new_count = count + 1
        logger.debug(
            "quota_check_passed",
            tool_id=tool_id,
            quota_type=quota_type,
            identifier=identifier,
            count=new_count,
            limit=limit,
        )

    async def get_quota_status(
        self,
        tool_id: str,
        daily_quota: int | None = None,
        monthly_quota: int | None = None,
        identifier: str | None = None,
    ) -> dict[str, Any]:
        """Get quota status for a tool.

        Args:
            tool_id: Tool identifier
            daily_quota: Daily quota limit (None = unlimited)
            monthly_quota: Monthly quota limit (None = unlimited)
            identifier: Optional identifier for per-user quotas

        Returns:
            Dictionary with quota status:
            - daily_limit: Daily quota limit (null if unlimited)
            - daily_used: Used daily quota
            - daily_remaining: Remaining daily quota
            - daily_reset_at: When daily quota resets (ISO format)
            - monthly_limit: Monthly quota limit (null if unlimited)
            - monthly_used: Used monthly quota
            - monthly_remaining: Remaining monthly quota
            - monthly_reset_at: When monthly quota resets (ISO format)
        """
        if not self._redis:
            await self.connect()

        status: dict[str, Any] = {}

        # Get daily quota status
        if daily_quota is not None:
            daily_key = self._get_quota_key(tool_id, "daily", identifier)
            daily_current = await self._redis.get(daily_key)
            daily_used = int(daily_current) if daily_current else 0
            daily_remaining = max(0, daily_quota - daily_used)
            daily_reset = self._get_reset_time("daily")

            status["daily_limit"] = daily_quota
            status["daily_used"] = daily_used
            status["daily_remaining"] = daily_remaining
            status["daily_reset_at"] = daily_reset.isoformat()
        else:
            status["daily_limit"] = None
            status["daily_used"] = 0
            status["daily_remaining"] = None
            status["daily_reset_at"] = None

        # Get monthly quota status
        if monthly_quota is not None:
            monthly_key = self._get_quota_key(tool_id, "monthly", identifier)
            monthly_current = await self._redis.get(monthly_key)
            monthly_used = int(monthly_current) if monthly_current else 0
            monthly_remaining = max(0, monthly_quota - monthly_used)
            monthly_reset = self._get_reset_time("monthly")

            status["monthly_limit"] = monthly_quota
            status["monthly_used"] = monthly_used
            status["monthly_remaining"] = monthly_remaining
            status["monthly_reset_at"] = monthly_reset.isoformat()
        else:
            status["monthly_limit"] = None
            status["monthly_used"] = 0
            status["monthly_remaining"] = None
            status["monthly_reset_at"] = None

        return status

    async def reset_quota(
        self,
        tool_id: str,
        quota_type: str | None = None,
        identifier: str | None = None,
    ) -> None:
        """Reset quota for a tool.

        Args:
            tool_id: Tool identifier
            quota_type: Type of quota to reset (daily, monthly, or None for both)
            identifier: Optional identifier for per-user quotas
        """
        if not self._redis:
            await self.connect()

        quota_types = ["daily", "monthly"] if quota_type is None else [quota_type]

        for qtype in quota_types:
            key = self._get_quota_key(tool_id, qtype, identifier)
            await self._redis.delete(key)
            logger.info(
                "quota_reset",
                tool_id=tool_id,
                quota_type=qtype,
                identifier=identifier,
            )


# Global quota manager instance
_global_quota_manager: QuotaManager | None = None


def get_quota_manager(redis_url: str | None = None) -> QuotaManager:
    """Get global quota manager instance.

    Args:
        redis_url: Optional Redis URL (uses default if not provided)

    Returns:
        Global QuotaManager instance
    """
    global _global_quota_manager
    if _global_quota_manager is None:
        if redis_url:
            _global_quota_manager = QuotaManager(redis_url=redis_url)
        else:
            _global_quota_manager = QuotaManager()
    return _global_quota_manager
