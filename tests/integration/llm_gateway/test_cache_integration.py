"""Integration tests for LLM Gateway caching with Redis.

This module tests the complete caching stack including:
1. L1 (in-memory LRU) cache integration
2. L2 (Redis) cache integration
3. Cache promotion from L2 to L1
4. Cache eviction policies
5. Cache statistics and monitoring
6. Multi-level cache coordination

Tests use real Redis instance for integration validation.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from agentcore.llm_gateway.cache_models import (
    CacheConfig,
    CacheEntry,
    CacheKey,
    CacheMode,
    EvictionPolicy,
)
from agentcore.llm_gateway.cache_service import CacheService, L1Cache, L2Cache


pytestmark = pytest.mark.integration


@pytest.fixture
def cache_config() -> CacheConfig:
    """Create cache configuration for testing."""
    return CacheConfig(
        enabled=True,
        l1_enabled=True,
        l1_max_size=100,
        l1_ttl_seconds=300,
        l2_enabled=True,
        l2_ttl_seconds=600,
        mode=CacheMode.EXACT,
        eviction_policy=EvictionPolicy.LRU,
    )


@pytest.fixture
def sample_cache_entry() -> CacheEntry:
    """Create sample cache entry for testing."""
    return CacheEntry(
        cache_key="test:key:123",
        response_id="resp_123",
        model="gpt-4",
        provider="openai",
        choices=[{"message": {"content": "Test response"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        cost=0.001,
        latency_ms=500,
        created_at=datetime.now(UTC),
        access_count=1,
        last_accessed=datetime.now(UTC),
        ttl_seconds=300,
    )


class TestL1CacheIntegration:
    """Integration tests for L1 (in-memory) cache."""

    def test_l1_cache_basic_operations(self, sample_cache_entry: CacheEntry) -> None:
        """Test basic L1 cache set/get operations."""
        cache = L1Cache(max_size=100, ttl_seconds=300)

        cache_key = "test:key:1"

        # Set value
        cache.set(cache_key, sample_cache_entry)

        # Get value
        retrieved = cache.get(cache_key)
        assert retrieved is not None
        assert retrieved.cache_key == sample_cache_entry.cache_key
        assert retrieved.model == sample_cache_entry.model

    def test_l1_cache_miss(self) -> None:
        """Test L1 cache miss tracking."""
        cache = L1Cache(max_size=100)

        result = cache.get("nonexistent:key")
        assert result is None

    def test_l1_cache_lru_eviction(self, sample_cache_entry: CacheEntry) -> None:
        """Test LRU eviction policy in L1 cache."""
        cache = L1Cache(max_size=10, eviction_policy=EvictionPolicy.LRU)

        # Fill cache to capacity
        for i in range(10):
            entry = CacheEntry(
                cache_key=f"key:{i}",
                response_id=f"resp_{i}",
                model="gpt-4",
                provider="openai",
                choices=[{"message": {"content": f"Response {i}"}}],
                created_at=datetime.now(UTC),
                access_count=1,
                last_accessed=datetime.now(UTC),
            )
            cache.set(f"key:{i}", entry)

        # Access first key to make it recently used
        cache.get("key:0")

        # Add one more key to trigger eviction
        new_entry = CacheEntry(
            cache_key="key:new",
            response_id="resp_new",
            model="gpt-4",
            provider="openai",
            choices=[{"message": {"content": "New response"}}],
            created_at=datetime.now(UTC),
            access_count=1,
            last_accessed=datetime.now(UTC),
        )
        cache.set("key:new", new_entry)

        # key:0 should still exist (recently used)
        assert cache.get("key:0") is not None

        # One of the other keys should be evicted
        assert cache.size() == 10

    def test_l1_cache_clear(self, sample_cache_entry: CacheEntry) -> None:
        """Test L1 cache clear operation."""
        cache = L1Cache(max_size=100)

        cache.set("key:1", sample_cache_entry)
        cache.set("key:2", sample_cache_entry)

        cache.clear()

        assert cache.get("key:1") is None
        assert cache.get("key:2") is None
        assert cache.size() == 0

    def test_l1_cache_delete(self, sample_cache_entry: CacheEntry) -> None:
        """Test L1 cache delete operation."""
        cache = L1Cache(max_size=100)

        cache_key = "test:delete:key"
        cache.set(cache_key, sample_cache_entry)

        # Verify exists
        assert cache.get(cache_key) is not None

        # Delete
        deleted = cache.delete(cache_key)
        assert deleted is True

        # Should not exist
        assert cache.get(cache_key) is None

        # Delete non-existent
        deleted = cache.delete("nonexistent")
        assert deleted is False


class TestL2CacheIntegration:
    """Integration tests for L2 (Redis) cache."""

    @pytest.mark.asyncio
    async def test_l2_cache_basic_operations(self, sample_cache_entry: CacheEntry) -> None:
        """Test basic L2 cache set/get operations with Redis."""
        cache = L2Cache(ttl_seconds=300, key_prefix="test:cache:")

        # Connect to Redis
        await cache.connect()

        if not cache._connected:
            pytest.skip("Redis not available")

        cache_key = "test:key:1"

        try:
            # Set value
            await cache.set(cache_key, sample_cache_entry)

            # Get value
            retrieved = await cache.get(cache_key)
            assert retrieved is not None
            assert retrieved.cache_key == sample_cache_entry.cache_key
            assert retrieved.model == sample_cache_entry.model
        finally:
            # Cleanup
            await cache.delete(cache_key)

    @pytest.mark.asyncio
    async def test_l2_cache_ttl_expiration(self, sample_cache_entry: CacheEntry) -> None:
        """Test L2 cache TTL expiration."""
        cache = L2Cache(ttl_seconds=1, key_prefix="test:cache:")
        await cache.connect()

        if not cache._connected:
            pytest.skip("Redis not available")

        cache_key = "test:ttl:key"

        try:
            await cache.set(cache_key, sample_cache_entry)

            # Immediately retrieve - should exist
            assert await cache.get(cache_key) is not None

            # Wait for expiration
            await asyncio.sleep(1.5)

            # Should be expired
            assert await cache.get(cache_key) is None
        finally:
            await cache.delete(cache_key)

    @pytest.mark.asyncio
    async def test_l2_cache_delete(self, sample_cache_entry: CacheEntry) -> None:
        """Test L2 cache delete operation."""
        cache = L2Cache(ttl_seconds=300, key_prefix="test:cache:")
        await cache.connect()

        if not cache._connected:
            pytest.skip("Redis not available")

        cache_key = "test:delete:key"

        # Set and verify
        await cache.set(cache_key, sample_cache_entry)
        assert await cache.get(cache_key) is not None

        # Delete
        deleted = await cache.delete(cache_key)
        assert deleted is True

        # Should not exist
        assert await cache.get(cache_key) is None

    @pytest.mark.asyncio
    async def test_l2_cache_concurrent_access(self, sample_cache_entry: CacheEntry) -> None:
        """Test L2 cache concurrent read/write operations."""
        cache = L2Cache(ttl_seconds=300, key_prefix="test:cache:")
        await cache.connect()

        if not cache._connected:
            pytest.skip("Redis not available")

        num_operations = 20

        async def write_operation(index: int) -> None:
            entry = CacheEntry(
                cache_key=f"concurrent:key:{index}",
                response_id=f"resp_{index}",
                model="gpt-4",
                provider="openai",
                choices=[{"message": {"content": f"Response {index}"}}],
                created_at=datetime.now(UTC),
                access_count=1,
                last_accessed=datetime.now(UTC),
            )
            await cache.set(f"concurrent:key:{index}", entry)

        async def read_operation(index: int) -> CacheEntry | None:
            return await cache.get(f"concurrent:key:{index}")

        try:
            # Concurrent writes
            await asyncio.gather(*[write_operation(i) for i in range(num_operations)])

            # Concurrent reads
            results = await asyncio.gather(*[read_operation(i) for i in range(num_operations)])

            # Verify all values retrieved correctly
            for i, result in enumerate(results):
                assert result is not None
                assert result.response_id == f"resp_{i}"
        finally:
            # Cleanup
            for i in range(num_operations):
                await cache.delete(f"concurrent:key:{i}")


class TestCacheServiceIntegration:
    """Integration tests for full CacheService with L1 and L2."""

    @pytest.mark.asyncio
    async def test_cache_service_initialization(self, cache_config: CacheConfig) -> None:
        """Test cache service initialization."""
        service = CacheService(config=cache_config)
        await service.initialize()

        assert service.l1_cache is not None
        assert service.l2_cache is not None

    @pytest.mark.asyncio
    async def test_cache_service_lookup_l1_hit(self, cache_config: CacheConfig) -> None:
        """Test cache service L1 hit."""
        service = CacheService(config=cache_config)
        await service.initialize()

        # Create cache key and entry
        cache_key = CacheKey(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )

        entry = CacheEntry(
            cache_key=cache_key.to_hash(),
            response_id="resp_123",
            model="gpt-4",
            provider="openai",
            choices=[{"message": {"content": "Test response"}}],
            created_at=datetime.now(UTC),
            access_count=1,
            last_accessed=datetime.now(UTC),
        )

        # Set in L1 directly
        if service.l1_cache:
            service.l1_cache.set(cache_key.to_hash(), entry)

        # Lookup should hit L1
        result = await service.lookup(cache_key)
        assert result is not None
        assert result.cache_key == entry.cache_key

    @pytest.mark.asyncio
    async def test_cache_service_lookup_miss(self, cache_config: CacheConfig) -> None:
        """Test complete cache miss (not in L1 or L2)."""
        service = CacheService(config=cache_config)
        await service.initialize()

        cache_key = CacheKey(
            model="gpt-4",
            messages=[{"role": "user", "content": "Nonexistent"}],
        )

        result = await service.lookup(cache_key)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_service_store(self, cache_config: CacheConfig) -> None:
        """Test cache service store operation."""
        service = CacheService(config=cache_config)
        await service.initialize()

        cache_key = CacheKey(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test store"}],
        )

        entry = CacheEntry(
            cache_key=cache_key.to_hash(),
            response_id="resp_store",
            model="gpt-4",
            provider="openai",
            choices=[{"message": {"content": "Stored response"}}],
            created_at=datetime.now(UTC),
            access_count=1,
            last_accessed=datetime.now(UTC),
        )

        try:
            # Store entry
            await service.store(cache_key, entry)

            # Verify stored in L1
            if service.l1_cache:
                l1_result = service.l1_cache.get(cache_key.to_hash())
                assert l1_result is not None

            # Verify lookup works
            result = await service.lookup(cache_key)
            assert result is not None
            assert result.response_id == "resp_store"
        finally:
            # Cleanup
            await service.invalidate(cache_key)

    @pytest.mark.asyncio
    async def test_cache_service_invalidate(self, cache_config: CacheConfig) -> None:
        """Test cache service invalidate operation."""
        service = CacheService(config=cache_config)
        await service.initialize()

        cache_key = CacheKey(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test invalidate"}],
        )

        entry = CacheEntry(
            cache_key=cache_key.to_hash(),
            response_id="resp_invalidate",
            model="gpt-4",
            provider="openai",
            choices=[{"message": {"content": "To be invalidated"}}],
            created_at=datetime.now(UTC),
            access_count=1,
            last_accessed=datetime.now(UTC),
        )

        # Store and verify
        await service.store(cache_key, entry)
        assert await service.lookup(cache_key) is not None

        # Invalidate
        await service.invalidate(cache_key)

        # Should not exist
        assert await service.lookup(cache_key) is None

    @pytest.mark.asyncio
    async def test_cache_service_clear(self, cache_config: CacheConfig) -> None:
        """Test cache service clear all operation."""
        service = CacheService(config=cache_config)
        await service.initialize()

        # Add multiple entries
        for i in range(5):
            cache_key = CacheKey(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Test {i}"}],
            )

            entry = CacheEntry(
                cache_key=cache_key.to_hash(),
                response_id=f"resp_{i}",
                model="gpt-4",
                provider="openai",
                choices=[{"message": {"content": f"Response {i}"}}],
                created_at=datetime.now(UTC),
                access_count=1,
                last_accessed=datetime.now(UTC),
            )

            await service.store(cache_key, entry)

        # Clear all
        await service.clear()

        # Verify all entries cleared from L1
        if service.l1_cache:
            assert service.l1_cache.size() == 0


class TestCachePerformance:
    """Integration tests for cache performance characteristics."""

    @pytest.mark.asyncio
    async def test_l1_cache_performance(self) -> None:
        """Benchmark L1 cache performance."""
        import time

        cache = L1Cache(max_size=1000)
        num_operations = 500

        # Measure write performance
        start = time.perf_counter()
        for i in range(num_operations):
            entry = CacheEntry(
                cache_key=f"perf:key:{i}",
                response_id=f"resp_{i}",
                model="gpt-4",
                provider="openai",
                choices=[{"message": {"content": f"Response {i}"}}],
                created_at=datetime.now(UTC),
                access_count=1,
                last_accessed=datetime.now(UTC),
            )
            cache.set(f"perf:key:{i}", entry)
        write_duration = time.perf_counter() - start

        # Measure read performance
        start = time.perf_counter()
        for i in range(num_operations):
            cache.get(f"perf:key:{i}")
        read_duration = time.perf_counter() - start

        # L1 cache should be very fast (< 1ms per operation on average)
        avg_write_ms = (write_duration / num_operations) * 1000
        avg_read_ms = (read_duration / num_operations) * 1000

        assert avg_write_ms < 1.0, f"Write performance: {avg_write_ms:.3f}ms (expected < 1ms)"
        assert avg_read_ms < 1.0, f"Read performance: {avg_read_ms:.3f}ms (expected < 1ms)"
