"""
ACE Redis Cache Service

Provides caching layer for playbooks and baselines to reduce database load.
Implements TTL-based caching with automatic invalidation.

Performance targets:
- Playbook cache: 10min TTL
- Baseline cache: 1hr TTL
- Cache hit rate: >80%
- Latency reduction: >50% on cache hits
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as aioredis
import structlog
from pydantic import BaseModel

from agentcore.ace.models.ace_models import ContextPlaybook, PerformanceBaseline

logger = structlog.get_logger(__name__)


class CacheConfig(BaseModel):
    """Redis cache configuration."""

    redis_url: str = "redis://localhost:6379/0"
    playbook_ttl_seconds: int = 600  # 10 minutes
    baseline_ttl_seconds: int = 3600  # 1 hour
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0


class ACECacheService:
    """
    Redis-based caching service for ACE components.

    Provides high-performance caching for:
    - Context playbooks (10min TTL)
    - Performance baselines (1hr TTL)

    Features:
    - Automatic TTL-based expiration
    - JSON serialization
    - Connection pooling
    - Async-first design
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """
        Initialize cache service.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self._redis: aioredis.Redis | None = None
        self._connected = False
        logger.info(
            "ACE cache service initialized",
            playbook_ttl=self.config.playbook_ttl_seconds,
            baseline_ttl=self.config.baseline_ttl_seconds,
        )

    async def connect(self) -> None:
        """
        Establish Redis connection.

        Raises:
            redis.ConnectionError: If connection fails
        """
        if self._connected:
            return

        try:
            self._redis = await aioredis.from_url(  # type: ignore[no-untyped-call]
                self.config.redis_url,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("Redis connection established", url=self.config.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis and self._connected:
            await self._redis.aclose()
            self._connected = False
            logger.info("Redis connection closed")

    async def get_playbook(self, agent_id: str) -> ContextPlaybook | None:
        """
        Retrieve playbook from cache.

        Args:
            agent_id: Agent identifier

        Returns:
            Cached playbook or None if not found

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            key = f"ace:playbook:{agent_id}"
            data = await self._redis.get(key)

            if data:
                logger.debug("Cache hit", key=key)
                playbook_dict = json.loads(data)
                return ContextPlaybook(**playbook_dict)

            logger.debug("Cache miss", key=key)
            return None

        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None

    async def set_playbook(
        self, agent_id: str, playbook: ContextPlaybook
    ) -> bool:
        """
        Cache playbook with 10min TTL.

        Args:
            agent_id: Agent identifier
            playbook: Playbook to cache

        Returns:
            True if cached successfully, False otherwise

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            key = f"ace:playbook:{agent_id}"
            data = playbook.model_dump_json()
            await self._redis.setex(
                key,
                self.config.playbook_ttl_seconds,
                data,
            )
            logger.debug(
                "Playbook cached",
                key=key,
                ttl=self.config.playbook_ttl_seconds,
            )
            return True

        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def invalidate_playbook(self, agent_id: str) -> bool:
        """
        Invalidate cached playbook.

        Args:
            agent_id: Agent identifier

        Returns:
            True if invalidated, False if key didn't exist

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            key = f"ace:playbook:{agent_id}"
            result = await self._redis.delete(key)
            logger.debug("Playbook invalidated", key=key, deleted=bool(result))
            return bool(result)

        except Exception as e:
            logger.warning("Cache invalidate failed", key=key, error=str(e))
            return False

    async def get_baseline(
        self, agent_id: str, task_type: str
    ) -> PerformanceBaseline | None:
        """
        Retrieve baseline from cache.

        Args:
            agent_id: Agent identifier
            task_type: Task type identifier

        Returns:
            Cached baseline or None if not found

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            key = f"ace:baseline:{agent_id}:{task_type}"
            data = await self._redis.get(key)

            if data:
                logger.debug("Cache hit", key=key)
                baseline_dict = json.loads(data)
                return PerformanceBaseline(**baseline_dict)

            logger.debug("Cache miss", key=key)
            return None

        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None

    async def set_baseline(
        self,
        agent_id: str,
        task_type: str,
        baseline: PerformanceBaseline,
    ) -> bool:
        """
        Cache baseline with 1hr TTL.

        Args:
            agent_id: Agent identifier
            task_type: Task type identifier
            baseline: Baseline to cache

        Returns:
            True if cached successfully, False otherwise

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            key = f"ace:baseline:{agent_id}:{task_type}"
            data = baseline.model_dump_json()
            await self._redis.setex(
                key,
                self.config.baseline_ttl_seconds,
                data,
            )
            logger.debug(
                "Baseline cached",
                key=key,
                ttl=self.config.baseline_ttl_seconds,
            )
            return True

        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def invalidate_baseline(
        self, agent_id: str, task_type: str
    ) -> bool:
        """
        Invalidate cached baseline.

        Args:
            agent_id: Agent identifier
            task_type: Task type identifier

        Returns:
            True if invalidated, False if key didn't exist

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            key = f"ace:baseline:{agent_id}:{task_type}"
            result = await self._redis.delete(key)
            logger.debug("Baseline invalidated", key=key, deleted=bool(result))
            return bool(result)

        except Exception as e:
            logger.warning("Cache invalidate failed", key=key, error=str(e))
            return False

    async def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._connected or not self._redis:
            raise RuntimeError("Cache service not connected")

        try:
            info = await self._redis.info("stats")
            return {
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_keys": await self._redis.dbsize(),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                        1,
                    )
                ),
            }
        except Exception as e:
            logger.warning("Failed to get cache stats", error=str(e))
            return {}
