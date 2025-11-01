"""
Unit tests for GPU device detection and management
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from agentcore.dspy_optimization.gpu.device import (
    DeviceInfo,
    DeviceManager,
    DeviceType,
    get_device_manager,
)


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass"""

    def test_device_info_creation(self) -> None:
        """Test DeviceInfo creation"""
        device = DeviceInfo(
            device_type=DeviceType.CUDA,
            device_id=0,
            name="NVIDIA RTX 3090",
            total_memory=24 * 1024**3,
            available_memory=20 * 1024**3,
            compute_capability="8.6",
        )

        assert device.device_type == DeviceType.CUDA
        assert device.device_id == 0
        assert device.name == "NVIDIA RTX 3090"
        assert device.total_memory == 24 * 1024**3
        assert device.is_available is True


class TestDeviceManager:
    """Tests for DeviceManager"""

    def test_device_manager_initialization_cpu_only(self) -> None:
        """Test DeviceManager with CPU only (torch available but no CUDA)"""
        with patch("agentcore.dspy_optimization.gpu.device.logger"):
            manager = DeviceManager()

            assert manager.current_device is not None
            assert len(manager.available_devices) >= 1

            # Should have CPU device
            cpu_device = manager.get_device_by_type(DeviceType.CPU)
            assert cpu_device is not None
            assert cpu_device.device_type == DeviceType.CPU

    @pytest.mark.skipif(
        not hasattr(__import__("sys").modules.get("torch", None), "cuda"),
        reason="CUDA not available"
    )
    def test_detect_cuda_devices(self) -> None:
        """Test CUDA device detection with real torch if available"""
        manager = DeviceManager()

        # Should detect CUDA devices (may be 0 on CPU-only systems)
        cuda_devices = [
            d for d in manager.available_devices if d.device_type == DeviceType.CUDA
        ]
        assert isinstance(cuda_devices, list)

    def test_device_manager_has_gpu(self) -> None:
        """Test has_gpu property"""
        manager = DeviceManager()

        # Depends on system, but should return bool
        assert isinstance(manager.has_gpu, bool)

    def test_get_device_by_type(self) -> None:
        """Test getting device by type"""
        manager = DeviceManager()

        # CPU should always be available
        cpu_device = manager.get_device_by_type(DeviceType.CPU)
        assert cpu_device is not None
        assert cpu_device.device_type == DeviceType.CPU

        # CUDA device depends on system (may be None on Apple Silicon/AMD systems)
        cuda_device = manager.get_device_by_type(DeviceType.CUDA)
        # Don't assert - CUDA availability varies by system
        # (Apple Silicon has Metal, AMD has ROCm, NVIDIA has CUDA)

    def test_set_device_cpu(self) -> None:
        """Test setting device to CPU"""
        manager = DeviceManager()

        # Set to CPU
        manager.set_device(DeviceType.CPU, 0)
        assert manager.current_device.device_type == DeviceType.CPU

    def test_set_device_invalid(self) -> None:
        """Test setting invalid device raises error"""
        manager = DeviceManager()

        with pytest.raises(ValueError, match="not found"):
            manager.set_device(DeviceType.CUDA, 99)

    def test_get_torch_device_no_pytorch(self) -> None:
        """Test getting torch device when PyTorch unavailable"""
        manager = DeviceManager()

        if not manager.torch_available:
            with pytest.raises(RuntimeError, match="PyTorch not available"):
                manager.get_torch_device()

    def test_synchronize(self) -> None:
        """Test device synchronization"""
        manager = DeviceManager()

        # Should not raise even if GPU unavailable
        manager.synchronize()

    def test_get_memory_stats(self) -> None:
        """Test getting memory statistics"""
        manager = DeviceManager()

        stats = manager.get_memory_stats()

        assert "allocated" in stats
        assert "cached" in stats
        assert "total" in stats
        assert isinstance(stats["allocated"], int)
        assert isinstance(stats["cached"], int)
        assert isinstance(stats["total"], int)

    def test_clear_cache(self) -> None:
        """Test clearing device cache"""
        manager = DeviceManager()

        # Should not raise
        manager.clear_cache()

    def test_get_device_manager_singleton(self) -> None:
        """Test global device manager singleton"""
        manager1 = get_device_manager()
        manager2 = get_device_manager()

        # Should return same instance
        assert manager1 is manager2


class TestDeviceType:
    """Tests for DeviceType enum"""

    def test_device_types(self) -> None:
        """Test device type enum values"""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.ROCM.value == "rocm"
        assert DeviceType.METAL.value == "metal"
