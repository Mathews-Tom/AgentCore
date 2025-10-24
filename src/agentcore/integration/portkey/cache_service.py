"""Multi-level caching service for LLM responses.

Implements L1 (in-memory) and L2 (Redis) caching layers with
intelligent cache management, semantic similarity matching,
and comprehensive statistics tracking.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

import redis.asyncio as aioredis
import structlog

from agentcore.a2a_protocol.config import settings
from agentcore.integration.portkey.cache_models import (
    CacheConfig,
    CacheEntry,
    CacheKey,
    CacheMode,
    CacheStats,
    EvictionPolicy,
)
from agentcore.integration.portkey.exceptions import PortkeyError
from agentcore.integration.portkey.models import LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)


class L1Cache:
    """In-memory LRU cache for hot data.

    Thread-safe in-memory cache with configurable size limit
    and LRU eviction policy for sub-millisecond lookups.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ) -> None:
        """Initialize L1 cache.

        Args:
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for entries in seconds
            eviction_policy: Eviction policy (LRU, LFU, or TTL)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy

        # Use OrderedDict for LRU implementation
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()

        logger.info(
            "l1_cache_initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            eviction_policy=eviction_policy.value,
        )

    def get(self, cache_key: str) -> CacheEntry | None:
        """Get entry from L1 cache.

        Args:
            cache_key: Cache key hash

        Returns:
            Cache entry if found and valid, None otherwise
        """
        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                return None

            # Check if expired
            if entry.is_expired():
                # Remove expired entry
                del self._cache[cache_key]
                return None

            # Update access tracking
            entry.update_access()

            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(cache_key)

            return entry

    def set(self, cache_key: str, entry: CacheEntry) -> None:
        """Set entry in L1 cache.

        Args:
            cache_key: Cache key hash
            entry: Cache entry to store
        """
        with self._lock:
            # Set TTL if not already set
            if entry.ttl_seconds is None:
                entry.ttl_seconds = self.ttl_seconds

            # Check if we need to evict
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_one()

            # Store entry
            self._cache[cache_key] = entry

            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(cache_key)

    def delete(self, cache_key: str) -> bool:
        """Delete entry from L1 cache.

        Args:
            cache_key: Cache key hash

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from L1 cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of entries in cache
        """
        with self._lock:
            return len(self._cache)

    def _evict_one(self) -> None:
        """Evict one entry based on eviction policy.

        Must be called with lock held.
        """
        if not self._cache:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove oldest (first) entry
            self._cache.popitem(last=False)

        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used entry
            min_access_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count,
            )
            del self._cache[min_access_key]

        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove oldest entry by creation time
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at,
            )
            del self._cache[oldest_key]


class L2Cache:
    """Redis-based distributed cache for persistent storage.

    Async Redis cache for sharing cached responses across
    multiple AgentCore instances.
    """

    def __init__(
        self,
        ttl_seconds: int = 86400,
        key_prefix: str = "agentcore:llm:cache:",
    ) -> None:
        """Initialize L2 cache.

        Args:
            ttl_seconds: Time-to-live for entries in seconds
            key_prefix: Redis key prefix for cache entries
        """
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

        # Initialize Redis client (will be connected lazily)
        self._redis: aioredis.Redis[bytes] | None = None
        self._connected = False

        logger.info(
            "l2_cache_initialized",
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
        )

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            # Check if cluster URLs are configured
            if settings.REDIS_CLUSTER_URLS:
                # Use cluster mode
                from redis.asyncio.cluster import RedisCluster

                self._redis = RedisCluster.from_url(
                    settings.REDIS_CLUSTER_URLS[0],
                    decode_responses=False,
                )
            else:
                # Use single instance
                self._redis = await aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                )

            # Test connection
            await self._redis.ping()

            self._connected = True
            logger.info("l2_cache_connected", redis_url=settings.REDIS_URL)

        except Exception as e:
            logger.error("l2_cache_connection_failed", error=str(e))
            # Don't raise - allow graceful degradation to L1 only
            self._connected = False

    async def get(self, cache_key: str) -> CacheEntry | None:
        """Get entry from L2 cache.

        Args:
            cache_key: Cache key hash

        Returns:
            Cache entry if found, None otherwise
        """
        if not self._connected or self._redis is None:
            return None

        try:
            redis_key = f"{self.key_prefix}{cache_key}"
            data = await self._redis.get(redis_key)

            if data is None:
                return None

            # Deserialize entry
            import json

            entry_dict = json.loads(data)
            entry = CacheEntry(**entry_dict)

            # Check if expired (Redis TTL should handle this, but double-check)
            if entry.is_expired():
                await self.delete(cache_key)
                return None

            # Update access tracking
            entry.update_access()

            # Store updated entry back
            await self.set(cache_key, entry)

            return entry

        except Exception as e:
            logger.warning("l2_cache_get_failed", cache_key=cache_key, error=str(e))
            return None

    async def set(self, cache_key: str, entry: CacheEntry) -> None:
        """Set entry in L2 cache.

        Args:
            cache_key: Cache key hash
            entry: Cache entry to store
        """
        if not self._connected or self._redis is None:
            return

        try:
            # Set TTL if not already set
            if entry.ttl_seconds is None:
                entry.ttl_seconds = self.ttl_seconds

            # Serialize entry
            import json

            data = entry.model_dump_json()
            redis_key = f"{self.key_prefix}{cache_key}"

            # Store with TTL
            await self._redis.setex(
                redis_key,
                entry.ttl_seconds,
                data,
            )

        except Exception as e:
            logger.warning("l2_cache_set_failed", cache_key=cache_key, error=str(e))

    async def delete(self, cache_key: str) -> bool:
        """Delete entry from L2 cache.

        Args:
            cache_key: Cache key hash

        Returns:
            True if entry was deleted, False if not found
        """
        if not self._connected or self._redis is None:
            return False

        try:
            redis_key = f"{self.key_prefix}{cache_key}"
            result = await self._redis.delete(redis_key)
            return result > 0

        except Exception as e:
            logger.warning("l2_cache_delete_failed", cache_key=cache_key, error=str(e))
            return False

    async def clear(self) -> None:
        """Clear all cache entries with prefix."""
        if not self._connected or self._redis is None:
            return

        try:
            # Scan for keys with prefix and delete
            pattern = f"{self.key_prefix}*"
            async for key in self._redis.scan_iter(match=pattern, count=100):
                await self._redis.delete(key)

            logger.info("l2_cache_cleared")

        except Exception as e:
            logger.warning("l2_cache_clear_failed", error=str(e))

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._connected = False
            logger.info("l2_cache_closed")


class CacheService:
    """Multi-level caching service for LLM responses.

    Coordinates L1 (in-memory) and L2 (Redis) caching layers
    with intelligent cache management and statistics tracking.
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
    ) -> None:
        """Initialize cache service.

        Args:
            config: Cache configuration (uses defaults if not provided)
        """
        self.config = config or CacheConfig()
        self.config.validate_config()

        # Initialize cache layers
        self.l1_cache: L1Cache | None = None
        if self.config.l1_enabled:
            self.l1_cache = L1Cache(
                max_size=self.config.l1_max_size,
                ttl_seconds=self.config.l1_ttl_seconds,
                eviction_policy=self.config.l1_eviction_policy,
            )

        self.l2_cache: L2Cache | None = None
        if self.config.l2_enabled:
            self.l2_cache = L2Cache(
                ttl_seconds=self.config.l2_ttl_seconds,
                key_prefix=self.config.l2_key_prefix,
            )

        # Initialize statistics
        self.stats = CacheStats() if self.config.stats_enabled else None

        logger.info(
            "cache_service_initialized",
            l1_enabled=self.config.l1_enabled,
            l2_enabled=self.config.l2_enabled,
            mode=self.config.mode.value,
        )

    async def connect(self) -> None:
        """Connect cache service (connects L2 Redis)."""
        if self.l2_cache:
            await self.l2_cache.connect()

    async def get(
        self,
        request: LLMRequest,
    ) -> tuple[LLMResponse | None, str | None]:
        """Get cached response for request.

        Args:
            request: LLM request to lookup

        Returns:
            Tuple of (response, cache_level) where cache_level is 'l1', 'l2', or None
        """
        if not self.config.enabled:
            return None, None

        start_time = time.time()

        # Generate cache key
        # Note: provider is intentionally set to None for lookup to enable
        # cache hits regardless of which provider served the request
        cache_key_obj = CacheKey.from_request(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            provider=None,  # Provider-agnostic caching
        )

        # Use appropriate hash based on cache mode
        if self.config.mode == CacheMode.EXACT:
            cache_key = cache_key_obj.to_hash()
        else:
            cache_key = cache_key_obj.to_semantic_key()

        try:
            # Try L1 cache first
            if self.l1_cache:
                entry = self.l1_cache.get(cache_key)
                if entry:
                    latency_ms = int((time.time() - start_time) * 1000)

                    # Record hit
                    if self.stats:
                        self.stats.record_hit("l1", latency_ms, entry.cost)

                    logger.debug(
                        "cache_hit_l1",
                        cache_key=cache_key[:16],
                        latency_ms=latency_ms,
                    )

                    # Convert to response
                    response = self._entry_to_response(entry)
                    return response, "l1"

            # Try L2 cache
            if self.l2_cache:
                entry = await self.l2_cache.get(cache_key)
                if entry:
                    latency_ms = int((time.time() - start_time) * 1000)

                    # Promote to L1
                    if self.l1_cache:
                        self.l1_cache.set(cache_key, entry)

                    # Record hit
                    if self.stats:
                        self.stats.record_hit("l2", latency_ms, entry.cost)

                    logger.debug(
                        "cache_hit_l2",
                        cache_key=cache_key[:16],
                        latency_ms=latency_ms,
                    )

                    # Convert to response
                    response = self._entry_to_response(entry)
                    return response, "l2"

            # Cache miss
            if self.stats:
                self.stats.record_miss()

            logger.debug("cache_miss", cache_key=cache_key[:16])

            return None, None

        except Exception as e:
            logger.warning("cache_get_failed", error=str(e))

            if self.stats:
                self.stats.record_error()

            return None, None

    async def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
    ) -> None:
        """Store response in cache.

        Args:
            request: Original LLM request
            response: LLM response to cache
        """
        if not self.config.enabled:
            return

        # Generate cache key
        # Note: provider is intentionally set to None to enable
        # cache hits regardless of which provider served the request
        cache_key_obj = CacheKey.from_request(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            provider=None,  # Provider-agnostic caching
        )

        # Use appropriate hash based on cache mode
        if self.config.mode == CacheMode.EXACT:
            cache_key = cache_key_obj.to_hash()
        else:
            cache_key = cache_key_obj.to_semantic_key()

        try:
            # Create cache entry
            entry = CacheEntry(
                cache_key=cache_key,
                response_id=response.id,
                model=response.model,
                provider=response.provider,
                choices=response.choices,
                usage=response.usage,
                cost=response.cost,
                latency_ms=response.latency_ms,
                ttl_seconds=None,  # Will be set by cache layer
            )

            # Store in L1
            if self.l1_cache:
                self.l1_cache.set(cache_key, entry)

            # Store in L2
            if self.l2_cache:
                await self.l2_cache.set(cache_key, entry)

            logger.debug(
                "cache_set",
                cache_key=cache_key[:16],
                model=response.model,
                provider=response.provider,
            )

        except Exception as e:
            logger.warning("cache_set_failed", error=str(e))

            if self.stats:
                self.stats.record_error()

    async def invalidate(
        self,
        request: LLMRequest,
    ) -> bool:
        """Invalidate cached response for request.

        Args:
            request: LLM request to invalidate

        Returns:
            True if entry was invalidated, False otherwise
        """
        if not self.config.enabled:
            return False

        # Generate cache key
        cache_key_obj = CacheKey.from_request(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        cache_key = cache_key_obj.to_hash()

        deleted = False

        # Delete from L1
        if self.l1_cache:
            deleted = self.l1_cache.delete(cache_key) or deleted

        # Delete from L2
        if self.l2_cache:
            deleted = await self.l2_cache.delete(cache_key) or deleted

        if deleted:
            logger.debug("cache_invalidated", cache_key=cache_key[:16])

        return deleted

    async def clear(self) -> None:
        """Clear all cache entries."""
        if self.l1_cache:
            self.l1_cache.clear()

        if self.l2_cache:
            await self.l2_cache.clear()

        logger.info("cache_cleared")

    def get_stats(self) -> CacheStats | None:
        """Get cache statistics.

        Returns:
            Cache statistics or None if stats disabled
        """
        return self.stats

    def get_l1_size(self) -> int:
        """Get L1 cache size.

        Returns:
            Number of entries in L1 cache
        """
        if self.l1_cache:
            return self.l1_cache.size()
        return 0

    async def close(self) -> None:
        """Close cache service and cleanup resources."""
        if self.l2_cache:
            await self.l2_cache.close()

        logger.info("cache_service_closed")

    def _entry_to_response(self, entry: CacheEntry) -> LLMResponse:
        """Convert cache entry to LLM response.

        Args:
            entry: Cache entry

        Returns:
            LLM response
        """
        return LLMResponse(
            id=entry.response_id,
            model=entry.model,
            provider=entry.provider,
            choices=entry.choices,
            usage=entry.usage,
            cost=entry.cost,
            latency_ms=1,  # Cache hit is ~1ms
            metadata={
                "cached": True,
                "cache_age_seconds": entry.get_age_seconds(),
                "cache_access_count": entry.access_count,
            },
        )
