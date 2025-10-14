"""Tests for performance optimization service."""

import asyncio
from datetime import UTC, datetime

import pytest

from agentcore.agent_runtime.models.agent_config import (
    AgentConfig,
    AgentPhilosophy,
    ResourceLimits,
    SecurityProfile,
)
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.agent_runtime.services.performance_optimizer import (
    CacheEntry,
    ContainerPool,
    PerformanceCache,
    PerformanceOptimizer,
    ResourcePredictor,
)


class TestCacheEntry:
    """Test cache entry with TTL."""

    def test_cache_entry_creation(self) -> None:
        """Test cache entry is created with correct attributes."""
        entry = CacheEntry("test_value", ttl_seconds=60)

        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert not entry.is_expired()

    def test_cache_entry_expiration(self) -> None:
        """Test cache entry expires after TTL."""
        entry = CacheEntry("test_value", ttl_seconds=0)

        # Immediately expired
        assert entry.is_expired()

    def test_cache_entry_access_tracking(self) -> None:
        """Test cache entry tracks access count and time."""
        entry = CacheEntry("test_value")

        first_access_time = entry.last_accessed
        value = entry.access()

        assert value == "test_value"
        assert entry.access_count == 1
        assert entry.last_accessed >= first_access_time


class TestPerformanceCache:
    """Test performance cache implementation."""

    def test_cache_set_and_get(self) -> None:
        """Test basic cache set and get operations."""
        cache = PerformanceCache(max_size=10)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_cache_miss(self) -> None:
        """Test cache returns None for missing keys."""
        cache = PerformanceCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_expiration(self) -> None:
        """Test cache entries expire after TTL."""
        cache = PerformanceCache(default_ttl=0)

        cache.set("key1", "value1")
        result = cache.get("key1")

        # Should be expired immediately
        assert result is None

    def test_cache_invalidation(self) -> None:
        """Test cache entry can be manually invalidated."""
        cache = PerformanceCache()

        cache.set("key1", "value1")
        cache.invalidate("key1")
        result = cache.get("key1")

        assert result is None

    def test_cache_clear(self) -> None:
        """Test cache can be cleared."""
        cache = PerformanceCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self) -> None:
        """Test cache statistics tracking."""
        cache = PerformanceCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_eviction_on_max_size(self) -> None:
        """Test cache evicts entries when max size is reached."""
        cache = PerformanceCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should trigger eviction

        stats = cache.get_stats()
        assert stats["size"] <= 2

    def test_cache_lru_eviction(self) -> None:
        """Test cache evicts least recently used entries."""
        cache = PerformanceCache(max_size=2, default_ttl=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 to make key2 the LRU
        cache.get("key1")

        # Add key3, should evict key2
        cache.set("key3", "value3")

        assert cache.get("key1") is not None
        assert cache.get("key3") is not None


class TestContainerPool:
    """Test container pool implementation."""

    @pytest.mark.asyncio
    async def test_pool_cold_start(self) -> None:
        """Test container acquisition when pool is empty (cold start)."""
        pool = ContainerPool(pool_size=2)

        async def create_container() -> str:
            return "container-1"

        container_id, is_warm = await pool.acquire(
            AgentPhilosophy.REACT, create_container
        )

        assert container_id == "container-1"
        assert not is_warm  # Cold start

    @pytest.mark.asyncio
    async def test_pool_warm_start(self) -> None:
        """Test container acquisition from pool (warm start)."""
        pool = ContainerPool(pool_size=2)

        # Pre-warm the pool
        await pool.release(AgentPhilosophy.REACT, "container-1")

        async def create_container() -> str:
            return "container-2"

        container_id, is_warm = await pool.acquire(
            AgentPhilosophy.REACT, create_container
        )

        assert container_id == "container-1"
        assert is_warm  # Warm start

    @pytest.mark.asyncio
    async def test_pool_release(self) -> None:
        """Test releasing container back to pool."""
        pool = ContainerPool(pool_size=2)

        await pool.release(AgentPhilosophy.REACT, "container-1")

        stats = pool.get_pool_stats()
        assert stats["pools"][AgentPhilosophy.REACT.value]["available"] == 1

    @pytest.mark.asyncio
    async def test_pool_max_size_enforcement(self) -> None:
        """Test pool enforces maximum size."""
        pool = ContainerPool(pool_size=2)

        await pool.release(AgentPhilosophy.REACT, "container-1")
        await pool.release(AgentPhilosophy.REACT, "container-2")
        await pool.release(AgentPhilosophy.REACT, "container-3")  # Should not be added

        stats = pool.get_pool_stats()
        assert stats["pools"][AgentPhilosophy.REACT.value]["available"] == 2

    @pytest.mark.asyncio
    async def test_pool_pre_warm(self) -> None:
        """Test pool pre-warming functionality."""
        pool = ContainerPool(pool_size=3, warm_start_enabled=True)

        container_count = 0

        async def create_container() -> str:
            nonlocal container_count
            container_count += 1
            return f"container-{container_count}"

        await pool.pre_warm(AgentPhilosophy.REACT, create_container, count=2)

        stats = pool.get_pool_stats()
        assert stats["pools"][AgentPhilosophy.REACT.value]["available"] == 2

    @pytest.mark.asyncio
    async def test_pool_stats(self) -> None:
        """Test pool statistics reporting."""
        pool = ContainerPool(pool_size=5)

        await pool.release(AgentPhilosophy.REACT, "container-1")
        await pool.release(AgentPhilosophy.CHAIN_OF_THOUGHT, "container-2")

        stats = pool.get_pool_stats()

        assert stats["pool_size"] == 5
        assert stats["warm_start_enabled"] is True
        assert AgentPhilosophy.REACT.value in stats["pools"]
        assert AgentPhilosophy.CHAIN_OF_THOUGHT.value in stats["pools"]


class TestResourcePredictor:
    """Test resource prediction."""

    def test_predictor_default_requirements(self) -> None:
        """Test predictor returns defaults for unknown philosophy."""
        predictor = ResourcePredictor()

        requirements = predictor.predict_requirements("unknown_philosophy")

        assert "cpu_percent" in requirements
        assert "memory_mb" in requirements
        assert "execution_time_seconds" in requirements

    def test_predictor_learning_from_usage(self) -> None:
        """Test predictor learns from historical usage."""
        predictor = ResourcePredictor()

        # Record some usage
        for _ in range(5):
            predictor.record_usage(
                "react",
                {
                    "cpu_percent": 50.0,
                    "memory_usage_mb": 200.0,
                    "execution_time": 30.0,
                },
            )

        requirements = predictor.predict_requirements("react")

        # Should predict higher than average (with buffer)
        assert requirements["cpu_percent"] > 50.0
        assert requirements["memory_mb"] > 200.0

    def test_predictor_confidence(self) -> None:
        """Test predictor confidence increases with more data."""
        predictor = ResourcePredictor(history_size=10)

        # No data yet
        confidence_0 = predictor.get_prediction_confidence("react")
        assert confidence_0 == 0.0

        # Add some data
        for _ in range(5):
            predictor.record_usage("react", {"cpu_percent": 50.0})

        confidence_5 = predictor.get_prediction_confidence("react")
        assert 0.0 < confidence_5 < 1.0

        # Fill the history
        for _ in range(5):
            predictor.record_usage("react", {"cpu_percent": 50.0})

        confidence_10 = predictor.get_prediction_confidence("react")
        assert confidence_10 == 1.0

    def test_predictor_history_limit(self) -> None:
        """Test predictor keeps limited history."""
        predictor = ResourcePredictor(history_size=10)

        # Add more than history size
        for i in range(15):
            predictor.record_usage("react", {"value": float(i)})

        # Should only keep last 10
        history = predictor._history["react"]
        assert len(history) == 10
        assert history[0]["value"] == 5.0  # First entry should be from iteration 5


class TestPerformanceOptimizer:
    """Test main performance optimizer."""

    @pytest.mark.asyncio
    async def test_optimizer_initialization(self) -> None:
        """Test optimizer initializes correctly."""
        optimizer = PerformanceOptimizer(
            enable_caching=True,
            enable_pooling=True,
            enable_gc_optimization=True,
        )

        assert optimizer._enable_caching is True
        assert optimizer._enable_pooling is True
        assert optimizer._cache is not None
        assert optimizer._pool is not None

    @pytest.mark.asyncio
    async def test_optimizer_start_stop(self) -> None:
        """Test optimizer starts and stops background tasks."""
        optimizer = PerformanceOptimizer()

        await optimizer.start()
        assert optimizer._optimization_task is not None

        await optimizer.stop()
        assert optimizer._optimization_task is None

    @pytest.mark.asyncio
    async def test_optimizer_container_creation_without_pool(self) -> None:
        """Test container creation optimization without pooling."""
        optimizer = PerformanceOptimizer(enable_pooling=False)

        config = AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            resource_limits=ResourceLimits(),
            security_profile=SecurityProfile(),
        )

        async def create_container() -> str:
            return "container-1"

        container_id, metrics = await optimizer.optimize_container_creation(
            config,
            create_container,
        )

        assert container_id == "container-1"
        assert not metrics["warm_start"]

    def test_optimizer_tool_metadata_caching(self) -> None:
        """Test tool metadata caching."""
        optimizer = PerformanceOptimizer(enable_caching=True)

        metadata = {"name": "calculator", "type": "math"}

        optimizer.cache_tool_metadata("calc", metadata)
        cached = optimizer.get_cached_tool_metadata("calc")

        assert cached == metadata

    def test_optimizer_execution_pattern_caching(self) -> None:
        """Test execution pattern caching."""
        optimizer = PerformanceOptimizer(enable_caching=True)

        pattern = {"steps": ["think", "act", "observe"]}

        optimizer.cache_execution_pattern("react:basic", pattern)
        cached = optimizer.get_cached_pattern("react:basic")

        assert cached == pattern

    def test_optimizer_resource_tracking(self) -> None:
        """Test agent resource usage tracking."""
        optimizer = PerformanceOptimizer()

        agent = AgentExecutionState(
            agent_id="test-agent",
            status="running",
            created_at=datetime.now(UTC),
            last_updated=datetime.now(UTC),
        )

        metrics = {
            "philosophy": "react",
            "cpu_percent": 50.0,
            "memory_usage_mb": 200.0,
        }

        optimizer.track_agent_resources(agent, metrics)

        # Check agent is tracked with weak reference
        assert "test-agent" in optimizer._tracked_agents

    def test_optimizer_resource_prediction(self) -> None:
        """Test resource requirement prediction."""
        optimizer = PerformanceOptimizer()

        config = AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            resource_limits=ResourceLimits(),
            security_profile=SecurityProfile(),
        )

        # Train with some data
        for _ in range(5):
            optimizer._predictor.record_usage(
                "react",
                {
                    "cpu_percent": 50.0,
                    "memory_usage_mb": 200.0,
                    "execution_time": 30.0,
                },
            )

        requirements = optimizer.predict_resource_requirements(config)

        assert "cpu_percent" in requirements
        assert "memory_mb" in requirements
        assert "execution_time_seconds" in requirements

    @pytest.mark.asyncio
    async def test_optimizer_memory_optimization(self) -> None:
        """Test memory optimization."""
        optimizer = PerformanceOptimizer(enable_gc_optimization=True)

        result = await optimizer.optimize_memory()

        assert result["enabled"] is True
        assert "objects_collected" in result
        assert "memory_released_mb" in result

    @pytest.mark.asyncio
    async def test_optimizer_memory_optimization_disabled(self) -> None:
        """Test memory optimization when disabled."""
        optimizer = PerformanceOptimizer(enable_gc_optimization=False)

        result = await optimizer.optimize_memory()

        assert result["enabled"] is False

    def test_optimizer_performance_metrics(self) -> None:
        """Test performance metrics reporting."""
        optimizer = PerformanceOptimizer(
            enable_caching=True,
            enable_pooling=True,
        )

        # Generate some activity
        optimizer._optimization_metrics["warm_starts"] = 10
        optimizer._optimization_metrics["cold_starts"] = 5

        metrics = optimizer.get_performance_metrics()

        assert "optimization_metrics" in metrics
        assert "cache_stats" in metrics
        assert "pool_stats" in metrics
        assert "warm_start_rate_percent" in metrics
        assert metrics["warm_start_rate_percent"] == round((10 / 15) * 100, 2)

    @pytest.mark.asyncio
    async def test_optimizer_background_loop(self) -> None:
        """Test background optimization loop runs."""
        optimizer = PerformanceOptimizer()

        await optimizer.start()

        # Let it run briefly
        await asyncio.sleep(0.1)

        await optimizer.stop()

        # Should have completed without errors
        assert optimizer._optimization_task is None
