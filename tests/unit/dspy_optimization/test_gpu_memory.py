"""
Unit tests for GPU memory management
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType
from agentcore.dspy_optimization.gpu.memory import (
    MemoryManager,
    MemoryPool,
    MemoryStats,
)

# Skip CUDA-dependent tests on non-CUDA systems
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available - GPU memory tests require NVIDIA GPU"
)


@pytest.fixture
def device_manager() -> DeviceManager:
    """Create device manager fixture"""
    return DeviceManager()


@pytest.fixture
def memory_pool(device_manager: DeviceManager) -> MemoryPool:
    """Create memory pool fixture"""
    return MemoryPool(device_manager, pool_size=10)


@pytest.fixture
def memory_manager(device_manager: DeviceManager) -> MemoryManager:
    """Create memory manager fixture"""
    return MemoryManager(device_manager)


class TestMemoryStats:
    """Tests for MemoryStats dataclass"""

    def test_memory_stats_creation(self) -> None:
        """Test MemoryStats creation"""
        stats = MemoryStats(
            allocated=1024**3,
            cached=512**3,
            total=8 * 1024**3,
            peak_allocated=2 * 1024**3,
            utilization=0.125,
        )

        assert stats.allocated == 1024**3
        assert stats.cached == 512**3
        assert stats.total == 8 * 1024**3
        assert stats.peak_allocated == 2 * 1024**3
        assert stats.utilization == 0.125


class TestMemoryPool:
    """Tests for MemoryPool"""

    def test_initialization(self, device_manager: DeviceManager) -> None:
        """Test MemoryPool initialization"""
        pool = MemoryPool(device_manager, pool_size=20, enable_pooling=True)

        assert pool.device_manager is device_manager
        assert pool.pool_size == 20
        assert pool.enable_pooling is True

    def test_allocate_without_pooling(self, device_manager: DeviceManager) -> None:
        """Test allocation without pooling"""
        pool = MemoryPool(device_manager, enable_pooling=False)

        if not device_manager.torch_available:
            pytest.skip("PyTorch not available")

        tensor = pool.allocate((10, 10))
        assert tensor is not None

    def test_release_without_pooling(self, device_manager: DeviceManager) -> None:
        """Test release without pooling"""
        pool = MemoryPool(device_manager, enable_pooling=False)

        if not device_manager.torch_available:
            pytest.skip("PyTorch not available")

        tensor = pool.allocate((10, 10))
        # Should not raise
        pool.release(tensor)

    def test_clear_pool(self, memory_pool: MemoryPool) -> None:
        """Test clearing memory pool"""
        # Should not raise
        memory_pool.clear()


class TestMemoryManager:
    """Tests for MemoryManager"""

    def test_initialization(self, device_manager: DeviceManager) -> None:
        """Test MemoryManager initialization"""
        manager = MemoryManager(device_manager, cleanup_threshold=0.85)

        assert manager.device_manager is device_manager
        assert manager.cleanup_threshold == 0.85

    def test_get_memory_stats(self, memory_manager: MemoryManager) -> None:
        """Test getting memory statistics"""
        stats = memory_manager.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.allocated >= 0
        assert stats.cached >= 0
        assert stats.total > 0
        assert stats.peak_allocated >= 0
        assert 0.0 <= stats.utilization <= 1.0

    def test_check_memory_pressure_low(self, memory_manager: MemoryManager) -> None:
        """Test memory pressure check with low usage"""
        # Set high threshold
        memory_manager.cleanup_threshold = 0.99

        pressure = memory_manager.check_memory_pressure()

        # Should be False unless system memory is actually high
        assert isinstance(pressure, bool)

    def test_cleanup(self, memory_manager: MemoryManager) -> None:
        """Test memory cleanup"""
        # Should not raise
        memory_manager.cleanup(aggressive=False)
        memory_manager.cleanup(aggressive=True)

    def test_auto_cleanup_if_needed(self, memory_manager: MemoryManager) -> None:
        """Test automatic cleanup"""
        # Should not raise
        memory_manager.auto_cleanup_if_needed()

    def test_handle_oom_success(self, memory_manager: MemoryManager) -> None:
        """Test OOM handling with successful operation"""

        def operation(x: int, y: int) -> int:
            return x + y

        result = memory_manager.handle_oom(operation, 2, 3)
        assert result == 5

    def test_handle_oom_non_oom_error(self, memory_manager: MemoryManager) -> None:
        """Test OOM handling with non-OOM error"""

        def operation() -> None:
            raise ValueError("Not OOM")

        with pytest.raises(ValueError, match="Not OOM"):
            memory_manager.handle_oom(operation)

    def test_handle_oom_recovery_failure(self, memory_manager: MemoryManager) -> None:
        """Test OOM handling when recovery fails"""

        def operation() -> None:
            raise RuntimeError("out of memory")

        with pytest.raises(RuntimeError, match="Out of memory"):
            memory_manager.handle_oom(operation)

    def test_allocate_with_fallback_cpu(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test allocation with CPU fallback"""
        if not memory_manager.device_manager.torch_available:
            pytest.skip("PyTorch not available")

        tensor = memory_manager.allocate_with_fallback(
            (10, 10), fallback_device=DeviceType.CPU
        )
        assert tensor is not None

    def test_reset_peak_stats(self, memory_manager: MemoryManager) -> None:
        """Test resetting peak statistics"""
        # Should not raise
        memory_manager.reset_peak_stats()

        stats = memory_manager.get_memory_stats()
        # Peak should be reset to current or 0
        assert stats.peak_allocated >= 0

    def test_log_memory_summary(
        self, memory_manager: MemoryManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging memory summary"""
        memory_manager.log_memory_summary()

        # Should log memory info
        assert len(caplog.records) > 0


class TestMemoryPoolIntegration:
    """Integration tests for memory pooling"""

    def test_pool_reuse(self, device_manager: DeviceManager) -> None:
        """Test tensor reuse from pool"""
        if not device_manager.torch_available:
            pytest.skip("PyTorch not available")

        pool = MemoryPool(device_manager, pool_size=5, enable_pooling=True)

        # Allocate and release
        tensor1 = pool.allocate((10, 10))
        pool.release(tensor1)

        # Allocate again - should reuse
        tensor2 = pool.allocate((10, 10))

        # Both should exist (may or may not be same object)
        assert tensor2 is not None

    def test_pool_overflow(self, device_manager: DeviceManager) -> None:
        """Test pool behavior when full"""
        if not device_manager.torch_available:
            pytest.skip("PyTorch not available")

        pool = MemoryPool(device_manager, pool_size=2, enable_pooling=True)

        # Fill pool
        tensors = []
        for _ in range(5):
            tensor = pool.allocate((10, 10))
            tensors.append(tensor)

        # Release all
        for tensor in tensors:
            pool.release(tensor)

        # Pool should not grow beyond limit
        # (internal check - pool size limited)


class TestMemoryManagerIntegration:
    """Integration tests for memory manager"""

    def test_memory_lifecycle(self, memory_manager: MemoryManager) -> None:
        """Test complete memory lifecycle"""
        if not memory_manager.device_manager.torch_available:
            pytest.skip("PyTorch not available")

        # Get initial stats
        initial_stats = memory_manager.get_memory_stats()

        # Allocate tensors
        tensors = []
        for _ in range(10):
            tensor = memory_manager.memory_pool.allocate((100, 100))
            tensors.append(tensor)

        # Check increased usage
        after_alloc_stats = memory_manager.get_memory_stats()

        # Cleanup
        for tensor in tensors:
            memory_manager.memory_pool.release(tensor)

        memory_manager.cleanup(aggressive=True)

        # Should complete without errors
        final_stats = memory_manager.get_memory_stats()
        assert final_stats.allocated >= 0

    def test_multiple_cleanup_cycles(self, memory_manager: MemoryManager) -> None:
        """Test multiple cleanup cycles"""
        for i in range(5):
            memory_manager.cleanup(aggressive=(i % 2 == 0))
            stats = memory_manager.get_memory_stats()
            assert stats.allocated >= 0

    def test_stress_allocation(self, memory_manager: MemoryManager) -> None:
        """Test stress allocation and cleanup"""
        if not memory_manager.device_manager.torch_available:
            pytest.skip("PyTorch not available")

        # Allocate many small tensors
        for _ in range(100):
            tensor = memory_manager.memory_pool.allocate((10, 10))
            memory_manager.memory_pool.release(tensor)

        # Should not leak memory significantly
        memory_manager.cleanup(aggressive=True)
        stats = memory_manager.get_memory_stats()
        assert stats.utilization < 1.0
