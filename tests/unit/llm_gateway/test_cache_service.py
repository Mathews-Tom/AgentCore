"""Unit tests for LLM Gateway cache service.

This module tests the multi-level caching system for LLM responses:
- L1Cache: In-memory LRU cache
- L2Cache: Redis distributed cache
- CacheService: Multi-level cache coordination
- Cache hit/miss tracking
- Cache eviction policies (LRU, LFU, TTL)
- TTL expiration
- Cache statistics

Target: 90%+ code coverage
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.llm_gateway.cache_models import (
    CacheConfig,
    CacheEntry,
    CacheMode,
    EvictionPolicy,
)
from agentcore.llm_gateway.cache_service import CacheService, L1Cache, L2Cache
from agentcore.llm_gateway.models import LLMRequest, LLMResponse


@pytest.fixture
def sample_cache_entry() -> CacheEntry:
    """Create a sample cache entry for testing."""
    return CacheEntry(
        cache_key="test-key-123",
        response_id="resp-456",
        model="gpt-5",
        provider="openai",
        choices=[{"message": {"role": "assistant", "content": "Test response"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        cost=0.00015,
        latency_ms=1000,
    )


@pytest.fixture
def sample_llm_request() -> LLMRequest:
    """Create a sample LLM request for testing."""
    return LLMRequest(
        model="gpt-5",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def sample_llm_response() -> LLMResponse:
    """Create a sample LLM response for testing."""
    return LLMResponse(
        id="resp-789",
        model="gpt-5",
        provider="openai",
        choices=[{"message": {"role": "assistant", "content": "Hi there!"}}],
        usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        cost=0.0001,
        latency_ms=800,
    )


class TestL1Cache:
    """Test suite for L1Cache (in-memory cache)."""

    def test_initialization_defaults(self) -> None:
        """Test L1Cache initialization with default values."""
        cache = L1Cache()
        assert cache.max_size == 1000
        assert cache.ttl_seconds == 3600
        assert cache.eviction_policy == EvictionPolicy.LRU
        assert len(cache._cache) == 0

    def test_initialization_custom_values(self) -> None:
        """Test L1Cache initialization with custom values."""
        cache = L1Cache(
            max_size=500,
            ttl_seconds=7200,
            eviction_policy=EvictionPolicy.LFU,
        )
        assert cache.max_size == 500
        assert cache.ttl_seconds == 7200
        assert cache.eviction_policy == EvictionPolicy.LFU

    def test_get_miss(self) -> None:
        """Test cache miss returns None."""
        cache = L1Cache()
        result = cache.get("nonexistent-key")
        assert result is None

    def test_set_and_get(self, sample_cache_entry: CacheEntry) -> None:
        """Test setting and retrieving cache entry."""
        cache = L1Cache()
        cache.set("test-key", sample_cache_entry)

        result = cache.get("test-key")
        assert result is not None
        assert result.cache_key == sample_cache_entry.cache_key
        assert result.response_id == sample_cache_entry.response_id

    def test_get_updates_access_count(self, sample_cache_entry: CacheEntry) -> None:
        """Test that get updates access tracking."""
        cache = L1Cache()
        cache.set("test-key", sample_cache_entry)

        initial_count = sample_cache_entry.access_count
        cache.get("test-key")
        result = cache.get("test-key")

        assert result is not None
        assert result.access_count > initial_count

    def test_get_moves_to_end_lru(self, sample_cache_entry: CacheEntry) -> None:
        """Test that LRU moves accessed items to end."""
        cache = L1Cache(eviction_policy=EvictionPolicy.LRU)

        entry1 = CacheEntry(
            cache_key="key1",
            response_id="resp1",
            model="gpt-5",
            choices=[],
        )
        entry2 = CacheEntry(
            cache_key="key2",
            response_id="resp2",
            model="gpt-5",
            choices=[],
        )

        cache.set("key1", entry1)
        cache.set("key2", entry2)

        # Access key1 to move it to end
        cache.get("key1")

        # Get all keys - most recently accessed should be last
        keys = list(cache._cache.keys())
        assert keys[-1] == "key1"

    def test_expired_entry_removed_on_get(self) -> None:
        """Test that expired entries are removed on get."""
        cache = L1Cache(ttl_seconds=1)

        entry = CacheEntry(
            cache_key="test-key",
            response_id="resp-id",
            model="gpt-5",
            choices=[],
            ttl_seconds=1,
            created_at=datetime.now(UTC) - timedelta(seconds=2),  # Already expired
        )

        cache.set("test-key", entry)

        result = cache.get("test-key")
        assert result is None
        assert "test-key" not in cache._cache

    def test_delete_existing(self, sample_cache_entry: CacheEntry) -> None:
        """Test deleting existing entry."""
        cache = L1Cache()
        cache.set("test-key", sample_cache_entry)

        result = cache.delete("test-key")
        assert result is True
        assert cache.get("test-key") is None

    def test_delete_non_existing(self) -> None:
        """Test deleting non-existing entry returns False."""
        cache = L1Cache()
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear(self, sample_cache_entry: CacheEntry) -> None:
        """Test clearing all cache entries."""
        cache = L1Cache()
        cache.set("key1", sample_cache_entry)
        cache.set("key2", sample_cache_entry)

        cache.clear()
        assert len(cache._cache) == 0
        assert cache.size() == 0

    def test_size(self, sample_cache_entry: CacheEntry) -> None:
        """Test getting cache size."""
        cache = L1Cache()
        assert cache.size() == 0

        cache.set("key1", sample_cache_entry)
        assert cache.size() == 1

        cache.set("key2", sample_cache_entry)
        assert cache.size() == 2

    def test_eviction_lru(self) -> None:
        """Test LRU eviction policy."""
        cache = L1Cache(max_size=2, eviction_policy=EvictionPolicy.LRU)

        entry1 = CacheEntry(cache_key="key1", response_id="r1", model="gpt-5", choices=[])
        entry2 = CacheEntry(cache_key="key2", response_id="r2", model="gpt-5", choices=[])
        entry3 = CacheEntry(cache_key="key3", response_id="r3", model="gpt-5", choices=[])

        cache.set("key1", entry1)
        cache.set("key2", entry2)
        cache.set("key3", entry3)  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_eviction_lfu(self) -> None:
        """Test LFU eviction policy."""
        cache = L1Cache(max_size=2, eviction_policy=EvictionPolicy.LFU)

        entry1 = CacheEntry(cache_key="key1", response_id="r1", model="gpt-5", choices=[], access_count=5)
        entry2 = CacheEntry(cache_key="key2", response_id="r2", model="gpt-5", choices=[], access_count=2)
        entry3 = CacheEntry(cache_key="key3", response_id="r3", model="gpt-5", choices=[], access_count=10)

        cache.set("key1", entry1)
        cache.set("key2", entry2)
        cache.set("key3", entry3)  # Should evict key2 (lowest access count)

        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None

    def test_eviction_ttl(self) -> None:
        """Test TTL eviction policy."""
        cache = L1Cache(max_size=2, eviction_policy=EvictionPolicy.TTL)

        old_time = datetime.now(UTC) - timedelta(hours=2)
        recent_time = datetime.now(UTC) - timedelta(minutes=5)

        entry1 = CacheEntry(
            cache_key="key1",
            response_id="r1",
            model="gpt-5",
            choices=[],
            created_at=old_time,
        )
        entry2 = CacheEntry(
            cache_key="key2",
            response_id="r2",
            model="gpt-5",
            choices=[],
            created_at=recent_time,
        )
        entry3 = CacheEntry(
            cache_key="key3",
            response_id="r3",
            model="gpt-5",
            choices=[],
        )

        cache.set("key1", entry1)
        cache.set("key2", entry2)
        cache.set("key3", entry3)  # Should evict key1 (oldest)

        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_set_ttl_from_cache_if_not_set(self) -> None:
        """Test that cache TTL is applied if entry doesn't have one."""
        cache = L1Cache(ttl_seconds=3600)

        entry = CacheEntry(
            cache_key="test-key",
            response_id="resp-id",
            model="gpt-5",
            choices=[],
            ttl_seconds=None,
        )

        cache.set("test-key", entry)

        result = cache.get("test-key")
        assert result is not None
        assert result.ttl_seconds == 3600


class TestL2Cache:
    """Test suite for L2Cache (Redis cache)."""

    def test_initialization(self) -> None:
        """Test L2Cache initialization."""
        cache = L2Cache(ttl_seconds=86400, key_prefix="test:cache:")
        assert cache.ttl_seconds == 86400
        assert cache.key_prefix == "test:cache:"
        assert cache._connected is False
        assert cache._redis is None

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        """Test successful Redis connection."""
        cache = L2Cache()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("agentcore.llm_gateway.cache_service.aioredis.from_url", new_callable=AsyncMock) as mock_from_url:
            mock_from_url.return_value = mock_redis

            await cache.connect()

            assert cache._connected is True
            assert cache._redis is not None
            mock_redis.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_failure_graceful_degradation(self) -> None:
        """Test that connection failure doesn't raise exception."""
        cache = L2Cache()

        with patch("agentcore.llm_gateway.cache_service.aioredis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")

            await cache.connect()

            assert cache._connected is False

    @pytest.mark.asyncio
    async def test_get_not_connected(self) -> None:
        """Test get returns None when not connected."""
        cache = L2Cache()
        result = await cache.get("test-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_hit(self, sample_cache_entry: CacheEntry) -> None:
        """Test cache hit."""
        cache = L2Cache()
        cache._connected = True

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=sample_cache_entry.model_dump_json().encode())
        cache._redis = mock_redis

        result = await cache.get("test-key")

        assert result is not None
        assert result.cache_key == sample_cache_entry.cache_key
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_miss(self) -> None:
        """Test cache miss."""
        cache = L2Cache()
        cache._connected = True

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        cache._redis = mock_redis

        result = await cache.get("test-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_not_connected(self, sample_cache_entry: CacheEntry) -> None:
        """Test set does nothing when not connected."""
        cache = L2Cache()
        await cache.set("test-key", sample_cache_entry)  # Should not raise

    @pytest.mark.asyncio
    async def test_set_success(self, sample_cache_entry: CacheEntry) -> None:
        """Test successful cache set."""
        cache = L2Cache(ttl_seconds=3600)
        cache._connected = True

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        cache._redis = mock_redis

        await cache.set("test-key", sample_cache_entry)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "agentcore:llm:cache:test-key"
        assert call_args[1] == 3600

    @pytest.mark.asyncio
    async def test_delete_not_connected(self) -> None:
        """Test delete returns False when not connected."""
        cache = L2Cache()
        result = await cache.delete("test-key")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Test successful cache delete."""
        cache = L2Cache()
        cache._connected = True

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)
        cache._redis = mock_redis

        result = await cache.delete("test-key")
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_not_connected(self) -> None:
        """Test clear does nothing when not connected."""
        cache = L2Cache()
        await cache.clear()  # Should not raise

    @pytest.mark.asyncio
    async def test_clear_success(self) -> None:
        """Test successful cache clear."""
        cache = L2Cache(key_prefix="test:cache:")
        cache._connected = True

        mock_redis = AsyncMock()
        # Create async generator for scan_iter
        async def mock_scan_iter(*args, **kwargs):
            for key in [b"test:cache:key1", b"test:cache:key2"]:
                yield key
        mock_redis.scan_iter = mock_scan_iter
        mock_redis.delete = AsyncMock()
        cache._redis = mock_redis

        await cache.clear()

        assert mock_redis.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing Redis connection."""
        cache = L2Cache()
        cache._connected = True

        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        cache._redis = mock_redis

        await cache.close()

        assert cache._connected is False
        mock_redis.aclose.assert_called_once()


class TestCacheService:
    """Test suite for CacheService (multi-level cache)."""

    def test_initialization_defaults(self) -> None:
        """Test CacheService initialization with defaults."""
        service = CacheService()

        assert service.config.enabled is True
        assert service.l1_cache is not None
        assert service.l2_cache is not None
        assert service.stats is not None

    def test_initialization_l1_only(self) -> None:
        """Test CacheService with L1 only."""
        config = CacheConfig(l1_enabled=True, l2_enabled=False)
        service = CacheService(config=config)

        assert service.l1_cache is not None
        assert service.l2_cache is None

    def test_initialization_l2_only(self) -> None:
        """Test CacheService with L2 only."""
        config = CacheConfig(l1_enabled=False, l2_enabled=True)
        service = CacheService(config=config)

        assert service.l1_cache is None
        assert service.l2_cache is not None

    def test_initialization_disabled(self) -> None:
        """Test CacheService when disabled."""
        config = CacheConfig(enabled=False)
        service = CacheService(config=config)

        assert service.config.enabled is False

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """Test cache service connection."""
        service = CacheService()

        with patch.object(service.l2_cache, "connect", new_callable=AsyncMock) as mock_connect:
            await service.connect()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_disabled_returns_none(
        self,
        sample_llm_request: LLMRequest,
    ) -> None:
        """Test get returns None when caching is disabled."""
        config = CacheConfig(enabled=False)
        service = CacheService(config=config)

        response, level = await service.get(sample_llm_request)
        assert response is None
        assert level is None

    @pytest.mark.asyncio
    async def test_get_l1_hit(
        self,
        sample_llm_request: LLMRequest,
        sample_cache_entry: CacheEntry,
    ) -> None:
        """Test cache hit from L1."""
        service = CacheService()

        # Mock L1 cache hit
        with patch.object(service.l1_cache, "get", return_value=sample_cache_entry):
            response, level = await service.get(sample_llm_request)

            assert response is not None
            assert level == "l1"
            assert response.id == sample_cache_entry.response_id

    @pytest.mark.asyncio
    async def test_get_l2_hit_promotes_to_l1(
        self,
        sample_llm_request: LLMRequest,
        sample_cache_entry: CacheEntry,
    ) -> None:
        """Test cache hit from L2 promotes to L1."""
        service = CacheService()

        # Mock L1 miss and L2 hit
        with patch.object(service.l1_cache, "get", return_value=None):
            with patch.object(service.l2_cache, "get", return_value=sample_cache_entry, new_callable=AsyncMock):
                with patch.object(service.l1_cache, "set") as mock_l1_set:
                    response, level = await service.get(sample_llm_request)

                    assert response is not None
                    assert level == "l2"
                    mock_l1_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_miss(
        self,
        sample_llm_request: LLMRequest,
    ) -> None:
        """Test cache miss from all levels."""
        service = CacheService()

        # Mock cache misses
        with patch.object(service.l1_cache, "get", return_value=None):
            with patch.object(service.l2_cache, "get", return_value=None, new_callable=AsyncMock):
                response, level = await service.get(sample_llm_request)

                assert response is None
                assert level is None

    @pytest.mark.asyncio
    async def test_set_disabled_does_nothing(
        self,
        sample_llm_request: LLMRequest,
        sample_llm_response: LLMResponse,
    ) -> None:
        """Test set does nothing when caching is disabled."""
        config = CacheConfig(enabled=False)
        service = CacheService(config=config)

        await service.set(sample_llm_request, sample_llm_response)  # Should not raise

    @pytest.mark.asyncio
    async def test_set_stores_in_both_levels(
        self,
        sample_llm_request: LLMRequest,
        sample_llm_response: LLMResponse,
    ) -> None:
        """Test set stores in both L1 and L2."""
        service = CacheService()

        with patch.object(service.l1_cache, "set") as mock_l1_set:
            with patch.object(service.l2_cache, "set", new_callable=AsyncMock) as mock_l2_set:
                await service.set(sample_llm_request, sample_llm_response)

                mock_l1_set.assert_called_once()
                mock_l2_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate(
        self,
        sample_llm_request: LLMRequest,
    ) -> None:
        """Test cache invalidation."""
        service = CacheService()

        with patch.object(service.l1_cache, "delete", return_value=True):
            with patch.object(service.l2_cache, "delete", return_value=True, new_callable=AsyncMock):
                result = await service.invalidate(sample_llm_request)
                assert result is True

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing all cache entries."""
        service = CacheService()

        with patch.object(service.l1_cache, "clear") as mock_l1_clear:
            with patch.object(service.l2_cache, "clear", new_callable=AsyncMock) as mock_l2_clear:
                await service.clear()

                mock_l1_clear.assert_called_once()
                mock_l2_clear.assert_called_once()

    def test_get_stats(self) -> None:
        """Test retrieving cache statistics."""
        service = CacheService()
        stats = service.get_stats()

        assert stats is not None
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0

    def test_get_stats_disabled(self) -> None:
        """Test get_stats returns None when stats disabled."""
        config = CacheConfig(stats_enabled=False)
        service = CacheService(config=config)

        stats = service.get_stats()
        assert stats is None

    def test_get_l1_size(self) -> None:
        """Test getting L1 cache size."""
        service = CacheService()
        size = service.get_l1_size()
        assert size == 0

    def test_get_l1_size_when_disabled(self) -> None:
        """Test get_l1_size returns 0 when L1 is disabled."""
        config = CacheConfig(l1_enabled=False)
        service = CacheService(config=config)

        size = service.get_l1_size()
        assert size == 0

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing cache service."""
        service = CacheService()

        with patch.object(service.l2_cache, "close", new_callable=AsyncMock) as mock_close:
            await service.close()
            mock_close.assert_called_once()

    def test_entry_to_response(self, sample_cache_entry: CacheEntry) -> None:
        """Test converting cache entry to response."""
        service = CacheService()
        response = service._entry_to_response(sample_cache_entry)

        assert response.id == sample_cache_entry.response_id
        assert response.model == sample_cache_entry.model
        assert response.provider == sample_cache_entry.provider
        assert response.metadata["cached"] is True
        assert "cache_age_seconds" in response.metadata

    @pytest.mark.asyncio
    async def test_cache_mode_exact(
        self,
        sample_llm_request: LLMRequest,
        sample_llm_response: LLMResponse,
    ) -> None:
        """Test exact cache mode."""
        config = CacheConfig(mode=CacheMode.EXACT)
        service = CacheService(config=config)

        # Store and retrieve
        with patch.object(service.l1_cache, "set"):
            with patch.object(service.l2_cache, "set", new_callable=AsyncMock):
                await service.set(sample_llm_request, sample_llm_response)

    @pytest.mark.asyncio
    async def test_cache_mode_semantic(
        self,
        sample_llm_request: LLMRequest,
        sample_llm_response: LLMResponse,
    ) -> None:
        """Test semantic cache mode."""
        config = CacheConfig(mode=CacheMode.SEMANTIC)
        service = CacheService(config=config)

        # Store and retrieve
        with patch.object(service.l1_cache, "set"):
            with patch.object(service.l2_cache, "set", new_callable=AsyncMock):
                await service.set(sample_llm_request, sample_llm_response)

    @pytest.mark.asyncio
    async def test_stats_tracking_hit(
        self,
        sample_llm_request: LLMRequest,
        sample_cache_entry: CacheEntry,
    ) -> None:
        """Test that stats are updated on cache hit."""
        service = CacheService()

        with patch.object(service.l1_cache, "get", return_value=sample_cache_entry):
            await service.get(sample_llm_request)

            stats = service.get_stats()
            assert stats is not None
            assert stats.total_requests == 1
            assert stats.cache_hits == 1
            assert stats.l1_hits == 1

    @pytest.mark.asyncio
    async def test_stats_tracking_miss(
        self,
        sample_llm_request: LLMRequest,
    ) -> None:
        """Test that stats are updated on cache miss."""
        service = CacheService()

        with patch.object(service.l1_cache, "get", return_value=None):
            with patch.object(service.l2_cache, "get", return_value=None, new_callable=AsyncMock):
                await service.get(sample_llm_request)

                stats = service.get_stats()
                assert stats is not None
                assert stats.total_requests == 1
                assert stats.cache_misses == 1
