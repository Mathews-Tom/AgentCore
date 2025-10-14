"""
Performance optimization service for agent runtime.

This module provides performance enhancements including container pooling,
memory optimization, caching strategies, and resource prediction for improved
agent execution throughput and reduced latency.
"""

import asyncio
import gc
import weakref
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from ..models.agent_config import AgentConfig, AgentPhilosophy
from ..models.agent_state import AgentExecutionState

logger = structlog.get_logger()


class CacheEntry:
    """Cache entry with TTL and access tracking."""

    def __init__(self, value: Any, ttl_seconds: int = 300) -> None:
        """
        Initialize cache entry.

        Args:
            value: Cached value
            ttl_seconds: Time-to-live in seconds
        """
        self.value = value
        self.created_at = datetime.now(UTC)
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now(UTC) > self.expires_at

    def access(self) -> Any:
        """Access cached value and update metrics."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)
        return self.value


class PerformanceCache:
    """High-performance cache with TTL and eviction policies."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300) -> None:
        """
        Initialize performance cache.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.access()

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Evict expired entries if cache is full
        if len(self._cache) >= self._max_size:
            self._evict_expired()

        # If still full, evict least recently used
        if len(self._cache) >= self._max_size:
            self._evict_lru()

        ttl = ttl if ttl is not None else self._default_ttl
        self._cache[key] = CacheEntry(value, ttl)

    def invalidate(self, key: str) -> None:
        """
        Invalidate cache entry.

        Args:
            key: Cache key
        """
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
        }

    def _evict_expired(self) -> None:
        """Evict all expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]


class ContainerPool:
    """Pre-warmed container pool for fast agent startup."""

    def __init__(self, pool_size: int = 10, warm_start_enabled: bool = True) -> None:
        """
        Initialize container pool.

        Args:
            pool_size: Target number of pre-warmed containers per philosophy
            warm_start_enabled: Whether to pre-warm containers
        """
        self._pool_size = pool_size
        self._warm_start_enabled = warm_start_enabled
        self._pools: dict[AgentPhilosophy, list[str]] = defaultdict(list)
        self._creation_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        philosophy: AgentPhilosophy,
        create_fn: Callable[[], str],
    ) -> tuple[str, bool]:
        """
        Acquire a container from the pool or create new one.

        Args:
            philosophy: Agent philosophy type
            create_fn: Async function to create new container

        Returns:
            Tuple of (container_id, is_warm_start)
        """
        async with self._lock:
            # Try to get from pool
            if self._pools[philosophy]:
                container_id = self._pools[philosophy].pop(0)
                creation_time = self._creation_times.get(container_id, 0)

                logger.info(
                    "container_acquired_from_pool",
                    philosophy=philosophy.value,
                    container_id=container_id,
                    startup_time_ms=creation_time,
                )

                return container_id, True

            # Create new container (cold start)
            import time

            start_time = time.time()
            container_id = await create_fn()
            creation_time = (time.time() - start_time) * 1000
            self._creation_times[container_id] = creation_time

            logger.info(
                "container_created_cold_start",
                philosophy=philosophy.value,
                container_id=container_id,
                creation_time_ms=creation_time,
            )

            return container_id, False

    async def release(
        self,
        philosophy: AgentPhilosophy,
        container_id: str,
    ) -> None:
        """
        Release a container back to the pool.

        Args:
            philosophy: Agent philosophy type
            container_id: Container identifier
        """
        async with self._lock:
            # Only keep up to pool_size containers
            if len(self._pools[philosophy]) < self._pool_size:
                self._pools[philosophy].append(container_id)

                logger.info(
                    "container_released_to_pool",
                    philosophy=philosophy.value,
                    container_id=container_id,
                    pool_size=len(self._pools[philosophy]),
                )
            else:
                # Pool is full, allow container to be destroyed
                logger.info(
                    "container_released_pool_full",
                    philosophy=philosophy.value,
                    container_id=container_id,
                )

    async def pre_warm(
        self,
        philosophy: AgentPhilosophy,
        create_fn: Callable[[], str],
        count: int | None = None,
    ) -> None:
        """
        Pre-warm containers for a philosophy.

        Args:
            philosophy: Agent philosophy type
            create_fn: Async function to create containers
            count: Number of containers to pre-warm (uses pool_size if None)
        """
        if not self._warm_start_enabled:
            return

        count = count if count is not None else self._pool_size

        async with self._lock:
            current_size = len(self._pools[philosophy])
            to_create = max(0, count - current_size)

            if to_create > 0:
                logger.info(
                    "container_pool_prewarming",
                    philosophy=philosophy.value,
                    count=to_create,
                )

                # Create containers in parallel
                tasks = [create_fn() for _ in range(to_create)]
                container_ids = await asyncio.gather(*tasks)

                self._pools[philosophy].extend(container_ids)

                logger.info(
                    "container_pool_prewarmed",
                    philosophy=philosophy.value,
                    pool_size=len(self._pools[philosophy]),
                )

    def get_pool_stats(self) -> dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool metrics
        """
        stats = {
            "pool_size": self._pool_size,
            "warm_start_enabled": self._warm_start_enabled,
            "pools": {},
        }

        for philosophy, containers in self._pools.items():
            stats["pools"][philosophy.value] = {
                "available": len(containers),
                "utilization_percent": round(
                    (len(containers) / self._pool_size) * 100, 2
                ),
            }

        return stats


class ResourcePredictor:
    """Predicts resource requirements based on historical usage."""

    def __init__(self, history_size: int = 100) -> None:
        """
        Initialize resource predictor.

        Args:
            history_size: Number of historical samples to keep
        """
        self._history: dict[str, list[dict[str, float]]] = defaultdict(list)
        self._history_size = history_size

    def record_usage(
        self,
        philosophy: str,
        metrics: dict[str, float],
    ) -> None:
        """
        Record resource usage for learning.

        Args:
            philosophy: Agent philosophy type
            metrics: Resource usage metrics
        """
        history = self._history[philosophy]
        history.append(metrics)

        # Keep only recent history
        if len(history) > self._history_size:
            history.pop(0)

    def predict_requirements(
        self,
        philosophy: str,
    ) -> dict[str, float]:
        """
        Predict resource requirements for a philosophy.

        Args:
            philosophy: Agent philosophy type

        Returns:
            Predicted resource requirements
        """
        history = self._history.get(philosophy, [])

        if not history:
            # Return conservative defaults
            return {
                "cpu_percent": 50.0,
                "memory_mb": 256.0,
                "execution_time_seconds": 60.0,
            }

        # Calculate averages with 20% buffer
        avg_cpu = sum(h.get("cpu_percent", 0) for h in history) / len(history)
        avg_memory = sum(h.get("memory_usage_mb", 0) for h in history) / len(history)
        avg_time = sum(h.get("execution_time", 0) for h in history) / len(history)

        return {
            "cpu_percent": round(avg_cpu * 1.2, 2),
            "memory_mb": round(avg_memory * 1.2, 2),
            "execution_time_seconds": round(avg_time * 1.2, 2),
        }

    def get_prediction_confidence(self, philosophy: str) -> float:
        """
        Get confidence level for predictions.

        Args:
            philosophy: Agent philosophy type

        Returns:
            Confidence level between 0.0 and 1.0
        """
        history = self._history.get(philosophy, [])
        return min(len(history) / self._history_size, 1.0)


class PerformanceOptimizer:
    """Main performance optimization service."""

    def __init__(
        self,
        enable_caching: bool = True,
        enable_pooling: bool = True,
        enable_gc_optimization: bool = True,
        cache_size: int = 1000,
        pool_size: int = 10,
    ) -> None:
        """
        Initialize performance optimizer.

        Args:
            enable_caching: Enable caching optimizations
            enable_pooling: Enable container pooling
            enable_gc_optimization: Enable garbage collection optimization
            cache_size: Maximum cache size
            pool_size: Container pool size per philosophy
        """
        self._enable_caching = enable_caching
        self._enable_pooling = enable_pooling
        self._enable_gc_optimization = enable_gc_optimization

        # Initialize components
        self._cache = PerformanceCache(max_size=cache_size) if enable_caching else None
        self._pool = (
            ContainerPool(pool_size=pool_size, warm_start_enabled=enable_pooling)
            if enable_pooling
            else None
        )
        self._predictor = ResourcePredictor()

        # Weak references to tracked agents for memory optimization
        self._tracked_agents: weakref.WeakValueDictionary[str, AgentExecutionState] = (
            weakref.WeakValueDictionary()
        )

        # Performance metrics
        self._optimization_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "warm_starts": 0,
            "cold_starts": 0,
            "gc_collections": 0,
            "memory_released_mb": 0.0,
        }

        # Background task for periodic optimization
        self._optimization_task: asyncio.Task[None] | None = None

        logger.info(
            "performance_optimizer_initialized",
            caching_enabled=enable_caching,
            pooling_enabled=enable_pooling,
            gc_optimization_enabled=enable_gc_optimization,
        )

    async def start(self) -> None:
        """Start background optimization tasks."""
        if self._optimization_task is None:
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            logger.info("performance_optimization_started")

    async def stop(self) -> None:
        """Stop background optimization tasks."""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
            self._optimization_task = None
            logger.info("performance_optimization_stopped")

    async def optimize_container_creation(
        self,
        config: AgentConfig,
        create_fn: Callable[[], str],
    ) -> tuple[str, dict[str, Any]]:
        """
        Optimize container creation with pooling.

        Args:
            config: Agent configuration
            create_fn: Async function to create container

        Returns:
            Tuple of (container_id, optimization_metrics)
        """
        metrics = {"warm_start": False, "startup_time_ms": 0.0}

        if self._pool:
            container_id, is_warm = await self._pool.acquire(
                config.philosophy,
                create_fn,
            )
            metrics["warm_start"] = is_warm

            if is_warm:
                self._optimization_metrics["warm_starts"] += 1
            else:
                self._optimization_metrics["cold_starts"] += 1
        else:
            # No pooling, direct creation
            import time

            start_time = time.time()
            container_id = await create_fn()
            metrics["startup_time_ms"] = (time.time() - start_time) * 1000
            self._optimization_metrics["cold_starts"] += 1

        return container_id, metrics

    def cache_tool_metadata(self, tool_id: str, metadata: dict[str, Any]) -> None:
        """
        Cache tool metadata for faster lookup.

        Args:
            tool_id: Tool identifier
            metadata: Tool metadata
        """
        if self._cache:
            self._cache.set(f"tool:{tool_id}", metadata, ttl=600)

    def get_cached_tool_metadata(self, tool_id: str) -> dict[str, Any] | None:
        """
        Get cached tool metadata.

        Args:
            tool_id: Tool identifier

        Returns:
            Cached metadata or None
        """
        if self._cache:
            result = self._cache.get(f"tool:{tool_id}")
            if result:
                self._optimization_metrics["cache_hits"] += 1
                return result
            self._optimization_metrics["cache_misses"] += 1

        return None

    def cache_execution_pattern(
        self,
        pattern_key: str,
        pattern_data: dict[str, Any],
    ) -> None:
        """
        Cache execution pattern for optimization.

        Args:
            pattern_key: Pattern identifier
            pattern_data: Pattern data
        """
        if self._cache:
            self._cache.set(f"pattern:{pattern_key}", pattern_data, ttl=300)

    def get_cached_pattern(self, pattern_key: str) -> dict[str, Any] | None:
        """
        Get cached execution pattern.

        Args:
            pattern_key: Pattern identifier

        Returns:
            Cached pattern or None
        """
        if self._cache:
            result = self._cache.get(f"pattern:{pattern_key}")
            if result:
                self._optimization_metrics["cache_hits"] += 1
                return result
            self._optimization_metrics["cache_misses"] += 1

        return None

    def track_agent_resources(
        self,
        agent: AgentExecutionState,
        metrics: dict[str, float],
    ) -> None:
        """
        Track agent resource usage for prediction.

        Args:
            agent: Agent execution state
            metrics: Resource usage metrics
        """
        # Use weak reference to avoid preventing garbage collection
        self._tracked_agents[agent.agent_id] = agent

        # Record for prediction
        if "philosophy" in metrics:
            self._predictor.record_usage(str(metrics["philosophy"]), metrics)

    def predict_resource_requirements(
        self,
        config: AgentConfig,
    ) -> dict[str, float]:
        """
        Predict resource requirements for agent.

        Args:
            config: Agent configuration

        Returns:
            Predicted resource requirements
        """
        return self._predictor.predict_requirements(config.philosophy.value)

    async def optimize_memory(self) -> dict[str, Any]:
        """
        Perform memory optimization.

        Returns:
            Optimization results
        """
        if not self._enable_gc_optimization:
            return {"enabled": False}

        import psutil

        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)

        # Force garbage collection
        collected = gc.collect()

        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_released = memory_before - memory_after

        self._optimization_metrics["gc_collections"] += 1
        self._optimization_metrics["memory_released_mb"] += memory_released

        logger.info(
            "memory_optimization_completed",
            objects_collected=collected,
            memory_released_mb=round(memory_released, 2),
        )

        return {
            "enabled": True,
            "objects_collected": collected,
            "memory_before_mb": round(memory_before, 2),
            "memory_after_mb": round(memory_after, 2),
            "memory_released_mb": round(memory_released, 2),
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Performance metrics dictionary
        """
        metrics = {
            "optimization_metrics": self._optimization_metrics.copy(),
        }

        if self._cache:
            metrics["cache_stats"] = self._cache.get_stats()

        if self._pool:
            metrics["pool_stats"] = self._pool.get_pool_stats()

        # Calculate derived metrics
        total_starts = (
            self._optimization_metrics["warm_starts"]
            + self._optimization_metrics["cold_starts"]
        )
        if total_starts > 0:
            metrics["warm_start_rate_percent"] = round(
                (self._optimization_metrics["warm_starts"] / total_starts) * 100,
                2,
            )

        return metrics

    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        try:
            while True:
                # Wait 5 minutes between optimization runs
                await asyncio.sleep(300)

                # Perform memory optimization
                await self.optimize_memory()

                # Clean up cache
                if self._cache:
                    self._cache._evict_expired()

                logger.debug("optimization_cycle_completed")

        except asyncio.CancelledError:
            logger.info("optimization_loop_cancelled")
            raise


# Global optimizer instance
_global_optimizer: PerformanceOptimizer | None = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer
