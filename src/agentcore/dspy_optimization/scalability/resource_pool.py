"""
Resource pooling for optimization workloads

Manages pooled resources (workers, connections, GPU memory) for efficient
concurrent optimization execution.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType
from agentcore.dspy_optimization.gpu.memory import MemoryManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceType(str, Enum):
    """Types of pooled resources"""

    WORKER = "worker"
    CONNECTION = "connection"
    GPU_MEMORY = "gpu_memory"
    LLM_CLIENT = "llm_client"


@dataclass
class PoolConfig:
    """Configuration for resource pool"""

    min_size: int = 5
    max_size: int = 100
    idle_timeout: int = 300  # seconds
    enable_monitoring: bool = True


class ResourcePool(Generic[T]):
    """
    Generic resource pool with lifecycle management

    Key features:
    - Dynamic pool sizing
    - Resource reuse
    - Idle timeout
    - Health checking
    - Usage statistics
    """

    def __init__(
        self,
        resource_type: ResourceType,
        factory: Any,
        config: PoolConfig | None = None,
    ) -> None:
        """
        Initialize resource pool

        Args:
            resource_type: Type of resource being pooled
            factory: Factory function to create new resources
            config: Pool configuration
        """
        self.resource_type = resource_type
        self.factory = factory
        self.config = config or PoolConfig()
        self._pool: asyncio.Queue[T] = asyncio.Queue()
        self._active: set[int] = set()
        self._total_created = 0
        self._total_acquired = 0
        self._total_released = 0

    async def initialize(self) -> None:
        """Initialize pool with minimum resources"""
        for _ in range(self.config.min_size):
            resource = await self._create_resource()
            await self._pool.put(resource)

        logger.info(
            f"Initialized {self.resource_type.value} pool "
            f"(size: {self.config.min_size}/{self.config.max_size})"
        )

    async def acquire(self, timeout: float | None = None) -> T:
        """
        Acquire resource from pool

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Pooled resource

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        # Try to get from pool
        try:
            resource = await asyncio.wait_for(
                self._pool.get(), timeout=timeout or 30.0
            )
            self._active.add(id(resource))
            self._total_acquired += 1
            logger.debug(f"Acquired {self.resource_type.value} resource")
            return resource

        except asyncio.TimeoutError:
            # Pool empty and at max size
            if self._total_created >= self.config.max_size:
                raise asyncio.TimeoutError(
                    f"No {self.resource_type.value} resources available"
                )

            # Create new resource
            resource = await self._create_resource()
            self._active.add(id(resource))
            self._total_acquired += 1
            logger.debug(f"Created new {self.resource_type.value} resource")
            return resource

    async def release(self, resource: T) -> None:
        """
        Release resource back to pool

        Args:
            resource: Resource to release
        """
        resource_id = id(resource)

        if resource_id not in self._active:
            logger.warning("Releasing non-active resource")
            return

        self._active.discard(resource_id)
        self._total_released += 1

        # Return to pool if not at max size
        if self._pool.qsize() < self.config.max_size:
            await self._pool.put(resource)
            logger.debug(f"Released {self.resource_type.value} resource to pool")
        else:
            # Pool full, discard
            await self._destroy_resource(resource)

    async def drain(self) -> None:
        """Drain all resources from pool"""
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                await self._destroy_resource(resource)
            except asyncio.QueueEmpty:
                break

        logger.info(f"Drained {self.resource_type.value} pool")

    def get_stats(self) -> dict[str, Any]:
        """
        Get pool statistics

        Returns:
            Dictionary with pool stats
        """
        return {
            "resource_type": self.resource_type.value,
            "available": self._pool.qsize(),
            "active": len(self._active),
            "total_created": self._total_created,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "utilization": len(self._active) / self.config.max_size
            if self.config.max_size > 0
            else 0.0,
        }

    async def _create_resource(self) -> T:
        """
        Create new resource using factory

        Returns:
            Created resource
        """
        resource = await self.factory() if asyncio.iscoroutinefunction(self.factory) else self.factory()
        self._total_created += 1
        return resource

    async def _destroy_resource(self, resource: T) -> None:
        """
        Destroy resource

        Args:
            resource: Resource to destroy
        """
        # Check if resource has cleanup method
        if hasattr(resource, "close"):
            if asyncio.iscoroutinefunction(resource.close):
                await resource.close()
            else:
                resource.close()

        del resource


class OptimizationResourceManager:
    """
    Manages all resource pools for optimization workloads

    Coordinates:
    - Worker pools
    - LLM client pools
    - GPU memory pools
    - Database connection pools
    """

    def __init__(
        self,
        device_manager: DeviceManager | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        """
        Initialize resource manager

        Args:
            device_manager: GPU device manager (optional)
            memory_manager: GPU memory manager (optional)
        """
        self.device_manager = device_manager
        self.memory_manager = memory_manager
        self._pools: dict[ResourceType, ResourcePool[Any]] = {}

    async def initialize_worker_pool(
        self, config: PoolConfig | None = None
    ) -> ResourcePool[asyncio.Queue[Any]]:
        """
        Initialize worker pool

        Args:
            config: Pool configuration

        Returns:
            Worker resource pool
        """
        config = config or PoolConfig(min_size=10, max_size=100)

        async def worker_factory() -> asyncio.Queue[Any]:
            """Create worker queue"""
            return asyncio.Queue()

        pool = ResourcePool(ResourceType.WORKER, worker_factory, config)
        await pool.initialize()
        self._pools[ResourceType.WORKER] = pool

        logger.info("Initialized worker pool")
        return pool

    async def initialize_llm_pool(
        self, config: PoolConfig | None = None
    ) -> ResourcePool[Any]:
        """
        Initialize LLM client pool

        Args:
            config: Pool configuration

        Returns:
            LLM client pool
        """
        config = config or PoolConfig(min_size=5, max_size=50)

        def llm_factory() -> Any:
            """Create LLM client"""
            import dspy

            return dspy.LM("openai/gpt-4.1-mini")

        pool = ResourcePool(ResourceType.LLM_CLIENT, llm_factory, config)
        await pool.initialize()
        self._pools[ResourceType.LLM_CLIENT] = pool

        logger.info("Initialized LLM client pool")
        return pool

    async def initialize_gpu_memory_pool(
        self, config: PoolConfig | None = None
    ) -> ResourcePool[Any] | None:
        """
        Initialize GPU memory pool

        Args:
            config: Pool configuration

        Returns:
            GPU memory pool or None if GPU unavailable
        """
        if not self.device_manager or not self.memory_manager:
            logger.warning("GPU managers not available - skipping GPU memory pool")
            return None

        if self.device_manager.current_device.device_type == DeviceType.CPU:
            logger.info("CPU device - skipping GPU memory pool")
            return None

        config = config or PoolConfig(min_size=10, max_size=100)

        def memory_factory() -> Any:
            """Create memory block"""
            # Allocate 1MB tensor
            shape = (256, 1024)  # 1MB float32
            return self.memory_manager.memory_pool.allocate(shape)

        pool = ResourcePool(ResourceType.GPU_MEMORY, memory_factory, config)
        await pool.initialize()
        self._pools[ResourceType.GPU_MEMORY] = pool

        logger.info("Initialized GPU memory pool")
        return pool

    async def get_pool(self, resource_type: ResourceType) -> ResourcePool[Any] | None:
        """
        Get resource pool by type

        Args:
            resource_type: Resource type

        Returns:
            Resource pool or None if not initialized
        """
        return self._pools.get(resource_type)

    async def drain_all_pools(self) -> None:
        """Drain all resource pools"""
        for resource_type, pool in self._pools.items():
            await pool.drain()
            logger.info(f"Drained {resource_type.value} pool")

    def get_all_stats(self) -> dict[str, Any]:
        """
        Get statistics for all pools

        Returns:
            Dictionary with all pool stats
        """
        return {
            resource_type.value: pool.get_stats()
            for resource_type, pool in self._pools.items()
        }

    def log_resource_summary(self) -> None:
        """Log resource usage summary"""
        stats = self.get_all_stats()

        for resource_type, pool_stats in stats.items():
            logger.info(
                f"{resource_type.upper()} Pool - "
                f"Available: {pool_stats['available']}, "
                f"Active: {pool_stats['active']}, "
                f"Utilization: {pool_stats['utilization'] * 100:.1f}%"
            )
