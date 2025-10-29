"""
Tests for resource pooling
"""

import asyncio

import pytest

from agentcore.dspy_optimization.scalability.resource_pool import (
    ResourcePool,
    ResourceType,
    PoolConfig,
    OptimizationResourceManager,
)


class MockResource:
    """Mock resource for testing"""

    def __init__(self, resource_id: int) -> None:
        self.resource_id = resource_id
        self.closed = False

    async def close(self) -> None:
        """Close resource"""
        self.closed = True


async def resource_factory() -> MockResource:
    """Factory for creating mock resources"""
    return MockResource(id(object()))


class TestResourcePool:
    """Test resource pool functionality"""

    @pytest.mark.asyncio
    async def test_initialize_pool(self):
        """Test pool initialization"""
        config = PoolConfig(min_size=5, max_size=20)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)

        await pool.initialize()

        stats = pool.get_stats()
        assert stats["available"] == 5
        assert stats["total_created"] == 5

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test acquiring and releasing resources"""
        config = PoolConfig(min_size=3, max_size=10)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        # Acquire resource
        resource = await pool.acquire()
        assert resource is not None

        stats = pool.get_stats()
        assert stats["available"] == 2
        assert stats["active"] == 1

        # Release resource
        await pool.release(resource)

        stats = pool.get_stats()
        assert stats["available"] == 3
        assert stats["active"] == 0

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquire timeout"""
        config = PoolConfig(min_size=1, max_size=1)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        # Acquire only resource
        resource1 = await pool.acquire()

        # Try to acquire another (should timeout)
        with pytest.raises(asyncio.TimeoutError):
            await pool.acquire(timeout=0.1)

        # Release and try again
        await pool.release(resource1)
        resource2 = await pool.acquire(timeout=0.1)
        assert resource2 is not None

    @pytest.mark.asyncio
    async def test_create_on_demand(self):
        """Test creating resources on demand"""
        config = PoolConfig(min_size=2, max_size=5)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        # Acquire all pre-created resources plus one more
        resources = []
        for _ in range(3):
            resource = await pool.acquire()
            resources.append(resource)

        stats = pool.get_stats()
        assert stats["total_created"] == 3  # 2 initial + 1 on-demand
        assert stats["active"] == 3

        # Release all
        for resource in resources:
            await pool.release(resource)

    @pytest.mark.asyncio
    async def test_max_size_enforcement(self):
        """Test max pool size enforcement"""
        config = PoolConfig(min_size=1, max_size=2)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        # Acquire max resources
        resource1 = await pool.acquire()
        resource2 = await pool.acquire()

        # Try to acquire beyond max
        with pytest.raises(asyncio.TimeoutError):
            await pool.acquire(timeout=0.1)

        # Release one
        await pool.release(resource1)

        # Should be able to acquire now
        resource3 = await pool.acquire(timeout=0.1)
        assert resource3 is not None

    @pytest.mark.asyncio
    async def test_pool_statistics(self):
        """Test pool statistics"""
        config = PoolConfig(min_size=5, max_size=10)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        stats = pool.get_stats()

        assert stats["resource_type"] == "worker"
        assert stats["available"] == 5
        assert stats["active"] == 0
        assert stats["total_created"] == 5
        assert stats["total_acquired"] == 0
        assert stats["total_released"] == 0
        assert stats["utilization"] == 0.0

        # Acquire resource
        resource = await pool.acquire()

        stats = pool.get_stats()
        assert stats["available"] == 4
        assert stats["active"] == 1
        assert stats["total_acquired"] == 1
        assert stats["utilization"] == 0.1  # 1/10

    @pytest.mark.asyncio
    async def test_drain_pool(self):
        """Test draining pool"""
        config = PoolConfig(min_size=5, max_size=10)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        await pool.drain()

        stats = pool.get_stats()
        assert stats["available"] == 0

    @pytest.mark.asyncio
    async def test_release_non_active_resource(self):
        """Test releasing resource not from pool"""
        config = PoolConfig(min_size=2, max_size=5)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        # Create resource outside pool
        external_resource = MockResource(999)

        # Release should log warning but not crash
        await pool.release(external_resource)

    @pytest.mark.asyncio
    async def test_resource_reuse(self):
        """Test that resources are reused"""
        config = PoolConfig(min_size=1, max_size=3)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        # Acquire and release
        resource1 = await pool.acquire()
        resource1_id = id(resource1)
        await pool.release(resource1)

        # Acquire again - should get same resource
        resource2 = await pool.acquire()
        resource2_id = id(resource2)

        assert resource1_id == resource2_id

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent resource access"""
        config = PoolConfig(min_size=5, max_size=20)
        pool = ResourcePool(ResourceType.WORKER, resource_factory, config)
        await pool.initialize()

        async def worker(worker_id: int) -> int:
            resource = await pool.acquire()
            await asyncio.sleep(0.01)  # Simulate work
            await pool.release(resource)
            return worker_id

        # Run many workers concurrently
        tasks = [worker(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        stats = pool.get_stats()
        assert stats["active"] == 0  # All released
        assert stats["total_acquired"] == 50
        assert stats["total_released"] == 50


class TestOptimizationResourceManager:
    """Test optimization resource manager"""

    @pytest.mark.asyncio
    async def test_initialize_worker_pool(self):
        """Test initializing worker pool"""
        manager = OptimizationResourceManager()

        pool = await manager.initialize_worker_pool()

        assert pool is not None
        assert pool.resource_type == ResourceType.WORKER

    @pytest.mark.asyncio
    async def test_initialize_llm_pool(self):
        """Test initializing LLM client pool"""
        manager = OptimizationResourceManager()

        pool = await manager.initialize_llm_pool()

        assert pool is not None
        assert pool.resource_type == ResourceType.LLM_CLIENT

    @pytest.mark.asyncio
    async def test_get_pool(self):
        """Test getting pool by type"""
        manager = OptimizationResourceManager()

        await manager.initialize_worker_pool()
        pool = await manager.get_pool(ResourceType.WORKER)

        assert pool is not None
        assert pool.resource_type == ResourceType.WORKER

    @pytest.mark.asyncio
    async def test_get_nonexistent_pool(self):
        """Test getting non-existent pool"""
        manager = OptimizationResourceManager()

        pool = await manager.get_pool(ResourceType.WORKER)

        assert pool is None

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting all pool statistics"""
        manager = OptimizationResourceManager()

        await manager.initialize_worker_pool()
        await manager.initialize_llm_pool()

        stats = manager.get_all_stats()

        assert "worker" in stats
        assert "llm_client" in stats

    @pytest.mark.asyncio
    async def test_drain_all_pools(self):
        """Test draining all pools"""
        manager = OptimizationResourceManager()

        await manager.initialize_worker_pool()
        await manager.initialize_llm_pool()

        await manager.drain_all_pools()

        stats = manager.get_all_stats()
        assert all(pool["available"] == 0 for pool in stats.values())

    @pytest.mark.asyncio
    async def test_custom_pool_config(self):
        """Test initializing pool with custom config"""
        manager = OptimizationResourceManager()
        config = PoolConfig(min_size=10, max_size=50)

        pool = await manager.initialize_worker_pool(config)

        stats = pool.get_stats()
        assert stats["available"] == 10

    @pytest.mark.asyncio
    async def test_multiple_pool_types(self):
        """Test managing multiple pool types"""
        manager = OptimizationResourceManager()

        worker_pool = await manager.initialize_worker_pool()
        llm_pool = await manager.initialize_llm_pool()

        assert worker_pool.resource_type == ResourceType.WORKER
        assert llm_pool.resource_type == ResourceType.LLM_CLIENT

        stats = manager.get_all_stats()
        assert len(stats) == 2
