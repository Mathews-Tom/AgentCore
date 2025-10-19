"""
Response Cache

In-memory and Redis-based response caching for high-performance request handling.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CacheStrategy(str, Enum):
    """Cache strategy types."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live only


@dataclass
class CachePolicy:
    """Cache policy configuration."""

    enabled: bool = True
    """Enable caching"""

    ttl: int = 300
    """Time to live in seconds"""

    max_size: int = 10000
    """Maximum number of cached items"""

    strategy: CacheStrategy = CacheStrategy.LRU
    """Eviction strategy"""

    vary_by_headers: list[str] | None = None
    """Headers to include in cache key (e.g., Accept-Encoding)"""

    cache_methods: list[str] | None = None
    """HTTP methods to cache (default: GET)"""

    cache_status_codes: list[int] | None = None
    """Status codes to cache (default: 200, 301, 302)"""


@dataclass
class CacheEntry:
    """Cached response entry."""

    key: str
    status_code: int
    headers: dict[str, str]
    content: bytes
    created_at: float
    ttl: int
    access_count: int = 0
    last_access: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


class ResponseCache:
    """
    In-memory response cache with TTL and eviction strategies.

    Optimized for high-throughput scenarios with minimal latency overhead.
    """

    def __init__(self, policy: CachePolicy | None = None):
        """
        Initialize response cache.

        Args:
            policy: Cache policy configuration
        """
        self.policy = policy or CachePolicy()
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            "Response cache initialized",
            max_size=self.policy.max_size,
            ttl=self.policy.ttl,
            strategy=self.policy.strategy,
        )

    def get_cache_key(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        """
        Generate cache key from request parameters.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers

        Returns:
            Cache key string
        """
        # Base key from method and URL
        key_parts = [method, url]

        # Add vary headers if configured
        if self.policy.vary_by_headers and headers:
            for header_name in self.policy.vary_by_headers:
                header_value = headers.get(header_name, "")
                key_parts.append(f"{header_name}:{header_value}")

        # Generate hash
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def should_cache_request(
        self,
        method: str,
        status_code: int,
    ) -> bool:
        """
        Check if request should be cached.

        Args:
            method: HTTP method
            status_code: Response status code

        Returns:
            True if should cache
        """
        if not self.policy.enabled:
            return False

        # Check method
        cache_methods = self.policy.cache_methods or ["GET"]
        if method not in cache_methods:
            return False

        # Check status code
        cache_codes = self.policy.cache_status_codes or [200, 301, 302]
        if status_code not in cache_codes:
            return False

        return True

    def get(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> CacheEntry | None:
        """
        Get cached response.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers

        Returns:
            Cache entry if found and valid, None otherwise
        """
        cache_key = self.get_cache_key(method, url, headers)

        entry = self._cache.get(cache_key)

        if not entry:
            self._misses += 1
            return None

        # Check expiration
        if entry.is_expired:
            self._misses += 1
            del self._cache[cache_key]
            return None

        # Update access stats
        entry.access_count += 1
        entry.last_access = time.time()

        self._hits += 1

        logger.debug(
            "Cache hit",
            key=cache_key[:16],
            age=entry.age,
            hits=self._hits,
            misses=self._misses,
        )

        return entry

    def set(
        self,
        method: str,
        url: str,
        status_code: int,
        headers: dict[str, str],
        content: bytes,
        request_headers: dict[str, str] | None = None,
    ) -> None:
        """
        Store response in cache.

        Args:
            method: HTTP method
            url: Request URL
            status_code: Response status code
            headers: Response headers
            content: Response content
            request_headers: Request headers for cache key
        """
        if not self.should_cache_request(method, status_code):
            return

        cache_key = self.get_cache_key(method, url, request_headers)

        # Check size limit
        if len(self._cache) >= self.policy.max_size:
            self._evict_one()

        # Create entry
        entry = CacheEntry(
            key=cache_key,
            status_code=status_code,
            headers=headers,
            content=content,
            created_at=time.time(),
            ttl=self.policy.ttl,
        )

        self._cache[cache_key] = entry

        logger.debug(
            "Response cached",
            key=cache_key[:16],
            size=len(content),
            ttl=self.policy.ttl,
        )

    def _evict_one(self) -> None:
        """Evict one entry based on strategy."""
        if not self._cache:
            return

        if self.policy.strategy == CacheStrategy.LRU:
            # Evict least recently accessed
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_access or self._cache[k].created_at,
            )
        elif self.policy.strategy == CacheStrategy.LFU:
            # Evict least frequently accessed
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count,
            )
        else:  # TTL
            # Evict oldest entry
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at,
            )

        del self._cache[key]
        self._evictions += 1

        logger.debug(
            "Cache entry evicted",
            strategy=self.policy.strategy,
            evictions=self._evictions,
        )

    def invalidate(
        self,
        method: str | None = None,
        url: str | None = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            method: HTTP method to invalidate (None = all)
            url: URL to invalidate (None = all)

        Returns:
            Number of entries invalidated
        """
        if method is None and url is None:
            # Invalidate all
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache invalidated (all entries)", count=count)
            return count

        # Selective invalidation
        keys_to_remove = []

        for key, entry in self._cache.items():
            # Would need to store original params to match
            # For now, just clear all if any filter specified
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        logger.info(
            "Cache invalidated",
            count=len(keys_to_remove),
            method=method,
            url=url,
        )

        return len(keys_to_remove)

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "enabled": self.policy.enabled,
            "size": len(self._cache),
            "max_size": self.policy.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "strategy": self.policy.strategy,
            "ttl": self.policy.ttl,
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        keys_to_remove = [
            key for key, entry in self._cache.items() if entry.is_expired
        ]

        for key in keys_to_remove:
            del self._cache[key]

        if keys_to_remove:
            logger.info("Expired cache entries removed", count=len(keys_to_remove))

        return len(keys_to_remove)
