"""Cache data models for LLM response caching.

Defines models for cache entries, keys, statistics, and configuration
for multi-level caching system (L1 in-memory + L2 Redis).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CacheMode(str, Enum):
    """Cache lookup mode."""

    EXACT = "exact"  # Exact match on cache key
    SEMANTIC = "semantic"  # Semantic similarity matching


class EvictionPolicy(str, Enum):
    """Cache eviction policy."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live only


class CacheKey(BaseModel):
    """Cache key for LLM requests with semantic hashing.

    Generates consistent hash keys from LLM request parameters
    for cache lookups. Supports both exact and semantic matching.
    """

    model: str = Field(description="Model name")
    messages: list[dict[str, Any]] = Field(description="Conversation messages")
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    provider: str | None = Field(
        default=None,
        description="Optional provider constraint",
    )

    def to_hash(self) -> str:
        """Generate deterministic hash for cache key.

        Returns:
            SHA256 hash string for this cache key
        """
        # Create canonical JSON representation
        key_dict = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": self.provider,
        }

        # Sort keys for deterministic hashing
        canonical = json.dumps(key_dict, sort_keys=True)

        # Generate SHA256 hash
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_semantic_key(self) -> str:
        """Generate semantic key for similarity matching.

        For semantic caching, we hash only the messages content,
        ignoring other parameters that don't affect semantic meaning.

        Returns:
            SHA256 hash of message content only
        """
        # Extract message content only
        content = json.dumps(self.messages, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    @classmethod
    def from_request(
        cls,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
    ) -> CacheKey:
        """Create cache key from request parameters.

        Args:
            model: Model name
            messages: Conversation messages
            temperature: Optional sampling temperature
            max_tokens: Optional max tokens
            provider: Optional provider constraint

        Returns:
            Cache key instance
        """
        return cls(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
        )


class CacheEntry(BaseModel):
    """Cached LLM response with metadata.

    Stores the complete response along with metadata for
    cache management (TTL, access tracking, cost savings).
    """

    cache_key: str = Field(description="Cache key hash")
    response_id: str = Field(description="Original response ID")
    model: str = Field(description="Model that generated response")
    provider: str | None = Field(
        default=None,
        description="Provider that handled request",
    )
    choices: list[dict[str, Any]] = Field(description="Response choices")
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage statistics",
    )
    cost: float | None = Field(
        default=None,
        description="Original request cost in USD",
        ge=0.0,
    )
    latency_ms: int | None = Field(
        default=None,
        description="Original request latency",
        ge=0,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Cache entry creation timestamp",
    )
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last access timestamp",
    )
    access_count: int = Field(
        default=0,
        description="Number of times accessed",
        ge=0,
    )
    ttl_seconds: int | None = Field(
        default=None,
        description="Time-to-live in seconds",
        ge=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def is_expired(self) -> bool:
        """Check if cache entry has expired.

        Returns:
            True if entry is expired, False otherwise
        """
        if self.ttl_seconds is None:
            return False

        age_seconds = (datetime.now(UTC) - self.created_at).total_seconds()
        return age_seconds > self.ttl_seconds

    def update_access(self) -> None:
        """Update access tracking for cache entry."""
        self.accessed_at = datetime.now(UTC)
        self.access_count += 1

    def get_age_seconds(self) -> int:
        """Get age of cache entry in seconds.

        Returns:
            Age in seconds since creation
        """
        return int((datetime.now(UTC) - self.created_at).total_seconds())


class CacheStats(BaseModel):
    """Cache statistics for monitoring and optimization.

    Tracks hit rates, latency, cost savings, and other metrics
    for cache performance analysis.
    """

    total_requests: int = Field(
        default=0,
        description="Total requests processed",
        ge=0,
    )
    cache_hits: int = Field(
        default=0,
        description="Number of cache hits",
        ge=0,
    )
    cache_misses: int = Field(
        default=0,
        description="Number of cache misses",
        ge=0,
    )
    l1_hits: int = Field(
        default=0,
        description="L1 (in-memory) cache hits",
        ge=0,
    )
    l2_hits: int = Field(
        default=0,
        description="L2 (Redis) cache hits",
        ge=0,
    )
    evictions: int = Field(
        default=0,
        description="Number of cache evictions",
        ge=0,
    )
    errors: int = Field(
        default=0,
        description="Number of cache errors",
        ge=0,
    )
    total_latency_saved_ms: int = Field(
        default=0,
        description="Total latency saved from cache hits",
        ge=0,
    )
    total_cost_saved: float = Field(
        default=0.0,
        description="Total cost saved from cache hits in USD",
        ge=0.0,
    )
    avg_cache_latency_ms: float = Field(
        default=0.0,
        description="Average cache lookup latency",
        ge=0.0,
    )

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0.0-1.0)
        """
        if self.total_requests == 0:
            return 0.0

        return self.cache_hits / self.total_requests

    def get_l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate.

        Returns:
            L1 hit rate as percentage (0.0-1.0)
        """
        if self.cache_hits == 0:
            return 0.0

        return self.l1_hits / self.cache_hits

    def get_l2_hit_rate(self) -> float:
        """Calculate L2 cache hit rate.

        Returns:
            L2 hit rate as percentage (0.0-1.0)
        """
        if self.cache_hits == 0:
            return 0.0

        return self.l2_hits / self.cache_hits

    def record_hit(
        self,
        level: str,
        latency_ms: int,
        cost_saved: float | None = None,
    ) -> None:
        """Record a cache hit.

        Args:
            level: Cache level ('l1' or 'l2')
            latency_ms: Cache lookup latency
            cost_saved: Cost saved from cache hit
        """
        self.total_requests += 1
        self.cache_hits += 1

        if level == "l1":
            self.l1_hits += 1
        elif level == "l2":
            self.l2_hits += 1

        if cost_saved:
            self.total_cost_saved += cost_saved

        # Update average latency
        total_hits = self.cache_hits
        self.avg_cache_latency_ms = (
            self.avg_cache_latency_ms * (total_hits - 1) + latency_ms
        ) / total_hits

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.total_requests += 1
        self.cache_misses += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1

    def record_error(self) -> None:
        """Record a cache error."""
        self.errors += 1

    @property
    def hits(self) -> int:
        """Alias for cache_hits (backward compatibility).

        Returns:
            Number of cache hits
        """
        return self.cache_hits

    @property
    def misses(self) -> int:
        """Alias for cache_misses (backward compatibility).

        Returns:
            Number of cache misses
        """
        return self.cache_misses


class CacheConfig(BaseModel):
    """Configuration for cache behavior.

    Defines cache modes, TTLs, size limits, and eviction policies
    for both L1 and L2 cache layers.
    """

    enabled: bool = Field(
        default=True,
        description="Enable caching",
    )

    # Cache modes
    mode: CacheMode = Field(
        default=CacheMode.EXACT,
        description="Cache lookup mode (exact or semantic)",
    )

    # L1 (In-memory) Cache Configuration
    l1_enabled: bool = Field(
        default=True,
        description="Enable L1 in-memory cache",
    )
    l1_max_size: int = Field(
        default=1000,
        description="Maximum L1 cache entries",
        ge=1,
    )
    l1_ttl_seconds: int = Field(
        default=3600,
        description="L1 cache TTL in seconds (1 hour default)",
        ge=1,
    )
    l1_eviction_policy: EvictionPolicy = Field(
        default=EvictionPolicy.LRU,
        description="L1 eviction policy",
    )

    # L2 (Redis) Cache Configuration
    l2_enabled: bool = Field(
        default=True,
        description="Enable L2 Redis cache",
    )
    l2_ttl_seconds: int = Field(
        default=86400,
        description="L2 cache TTL in seconds (24 hours default)",
        ge=60,
    )
    l2_key_prefix: str = Field(
        default="agentcore:llm:cache:",
        description="Redis key prefix for cache entries",
    )

    # Semantic Similarity Configuration
    semantic_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for semantic caching (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    # Performance Configuration
    cache_lookup_timeout_ms: int = Field(
        default=10,
        description="Maximum cache lookup timeout in milliseconds",
        ge=1,
        le=1000,
    )

    # Monitoring Configuration
    stats_enabled: bool = Field(
        default=True,
        description="Enable cache statistics tracking",
    )

    def validate_config(self) -> None:
        """Validate cache configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.l1_enabled and not self.l2_enabled:
            raise ValueError("At least one cache level (L1 or L2) must be enabled")

        if self.mode == CacheMode.SEMANTIC and self.semantic_threshold <= 0:
            raise ValueError("Semantic threshold must be > 0 for semantic caching")
