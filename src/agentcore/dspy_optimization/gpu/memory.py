"""
Memory management for GPU acceleration

Provides memory pooling, automatic cleanup, and OOM handling for efficient
GPU memory usage during optimization.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    allocated: int  # Bytes currently allocated
    cached: int  # Bytes in cache
    total: int  # Total device memory
    peak_allocated: int  # Peak memory allocated
    utilization: float  # Memory utilization (0.0-1.0)


class MemoryPool:
    """
    Memory pool for efficient tensor allocation

    Provides pre-allocated memory pools to reduce allocation overhead
    and fragmentation during GPU operations.

    Key features:
    - Pre-allocated memory blocks
    - Automatic reuse
    - Size-based pooling
    - Cleanup on low memory
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        pool_size: int = 100,
        enable_pooling: bool = True,
    ) -> None:
        """
        Initialize memory pool

        Args:
            device_manager: Device manager
            pool_size: Maximum number of cached tensors
            enable_pooling: Enable memory pooling
        """
        self.device_manager = device_manager
        self.pool_size = pool_size
        self.enable_pooling = enable_pooling
        self._pool: dict[tuple[tuple[int, ...], Any], list[Any]] = {}
        self._torch = None

        if device_manager.torch_available:
            import torch

            self._torch = torch

    def allocate(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """
        Allocate tensor from pool or create new

        Args:
            shape: Tensor shape
            dtype: Data type (default: float32)

        Returns:
            Allocated tensor
        """
        if not self.enable_pooling or not self._torch:
            # Direct allocation without pooling
            return self._create_tensor(shape, dtype)

        # Try to get from pool
        key = (shape, dtype)
        if key in self._pool and self._pool[key]:
            tensor = self._pool[key].pop()
            logger.debug(f"Reused tensor from pool: {shape}")
            return tensor

        # Create new tensor
        return self._create_tensor(shape, dtype)

    def release(self, tensor: Any) -> None:
        """
        Release tensor back to pool

        Args:
            tensor: Tensor to release
        """
        if not self.enable_pooling or not self._torch:
            # No pooling, just delete
            del tensor
            return

        # Add to pool if not full
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        key = (shape, dtype)

        if key not in self._pool:
            self._pool[key] = []

        if len(self._pool[key]) < self.pool_size:
            # Clear tensor data
            tensor.zero_()
            self._pool[key].append(tensor)
            logger.debug(f"Returned tensor to pool: {shape}")
        else:
            # Pool full, delete
            del tensor

    def clear(self) -> None:
        """Clear all pooled tensors"""
        for tensors in self._pool.values():
            for tensor in tensors:
                del tensor
        self._pool.clear()
        logger.info("Cleared memory pool")

    def _create_tensor(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """
        Create new tensor on device

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            New tensor
        """
        if not self._torch:
            raise RuntimeError("PyTorch not available")

        if dtype is None:
            dtype = self._torch.float32

        device = self.device_manager.get_torch_device()
        tensor = self._torch.zeros(shape, dtype=dtype, device=device)
        logger.debug(f"Created new tensor: {shape}")
        return tensor


class MemoryManager:
    """
    Manager for GPU memory optimization

    Handles automatic memory cleanup, OOM recovery, and memory monitoring
    for efficient GPU usage.

    Key features:
    - Automatic garbage collection
    - OOM detection and recovery
    - Memory statistics tracking
    - Periodic cleanup
    - Memory pool management
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        cleanup_threshold: float = 0.9,
        enable_pooling: bool = True,
    ) -> None:
        """
        Initialize memory manager

        Args:
            device_manager: Device manager
            cleanup_threshold: Memory usage threshold for automatic cleanup
            enable_pooling: Enable memory pooling
        """
        self.device_manager = device_manager
        self.cleanup_threshold = cleanup_threshold
        self.memory_pool = MemoryPool(device_manager, enable_pooling=enable_pooling)
        self._torch = None
        self._peak_memory: int = 0

        if device_manager.torch_available:
            import torch

            self._torch = torch

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics

        Returns:
            MemoryStats object
        """
        stats = self.device_manager.get_memory_stats()
        allocated = stats["allocated"]
        cached = stats["cached"]
        total = stats["total"]

        # Update peak memory
        if allocated > self._peak_memory:
            self._peak_memory = allocated

        # Calculate utilization
        utilization = allocated / total if total > 0 else 0.0

        return MemoryStats(
            allocated=allocated,
            cached=cached,
            total=total,
            peak_allocated=self._peak_memory,
            utilization=utilization,
        )

    def check_memory_pressure(self) -> bool:
        """
        Check if memory usage is high

        Returns:
            True if memory pressure is high
        """
        stats = self.get_memory_stats()
        return stats.utilization > self.cleanup_threshold

    def cleanup(self, aggressive: bool = False) -> None:
        """
        Perform memory cleanup

        Args:
            aggressive: Perform aggressive cleanup (clear cache and pool)
        """
        # Clear memory pool
        if aggressive:
            self.memory_pool.clear()

        # Force garbage collection
        gc.collect()

        # Clear device cache
        self.device_manager.clear_cache()

        # PyTorch-specific cleanup
        if self._torch and self.device_manager.current_device.device_type == DeviceType.CUDA:
            self._torch.cuda.empty_cache()
            if aggressive:
                self._torch.cuda.reset_peak_memory_stats()

        logger.info(f"Memory cleanup completed (aggressive={aggressive})")

    def auto_cleanup_if_needed(self) -> None:
        """Automatically cleanup if memory pressure is high"""
        if self.check_memory_pressure():
            logger.warning("High memory pressure detected, performing cleanup")
            self.cleanup(aggressive=False)

    def handle_oom(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Handle out-of-memory errors with recovery

        Args:
            operation: Operation that may cause OOM
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result

        Raises:
            RuntimeError: If OOM cannot be recovered
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Try operation
                result = operation(*args, **kwargs)
                return result

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM detected (attempt {retry_count + 1}/{max_retries})")

                    # Cleanup and retry
                    self.cleanup(aggressive=retry_count > 0)
                    retry_count += 1

                    if retry_count >= max_retries:
                        logger.error("Failed to recover from OOM after multiple attempts")
                        raise RuntimeError("Out of memory - cannot recover") from e
                else:
                    # Not an OOM error, re-raise
                    raise

        raise RuntimeError("OOM handling failed")

    def allocate_with_fallback(
        self,
        shape: tuple[int, ...],
        dtype: Any = None,
        fallback_device: DeviceType = DeviceType.CPU,
    ) -> Any:
        """
        Allocate tensor with automatic fallback on OOM

        Args:
            shape: Tensor shape
            dtype: Data type
            fallback_device: Device to fallback to on OOM

        Returns:
            Allocated tensor
        """
        try:
            # Try allocation on current device
            tensor = self.memory_pool.allocate(shape, dtype)
            return tensor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"OOM during allocation, falling back to {fallback_device.value}"
                )

                # Cleanup
                self.cleanup(aggressive=True)

                # Switch to fallback device
                original_device = self.device_manager.current_device
                try:
                    self.device_manager.set_device(fallback_device)
                    tensor = self.memory_pool.allocate(shape, dtype)
                    return tensor
                finally:
                    # Restore original device
                    self.device_manager.set_device(
                        original_device.device_type,
                        original_device.device_id,
                    )
            else:
                raise

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics"""
        self._peak_memory = 0

        if self._torch and self.device_manager.current_device.device_type == DeviceType.CUDA:
            self._torch.cuda.reset_peak_memory_stats()

        logger.info("Reset peak memory statistics")

    def log_memory_summary(self) -> None:
        """Log memory usage summary"""
        stats = self.get_memory_stats()

        logger.info(
            f"Memory Summary - "
            f"Allocated: {stats.allocated / 1024**2:.2f} MB, "
            f"Cached: {stats.cached / 1024**2:.2f} MB, "
            f"Total: {stats.total / 1024**2:.2f} MB, "
            f"Peak: {stats.peak_allocated / 1024**2:.2f} MB, "
            f"Utilization: {stats.utilization * 100:.1f}%"
        )
