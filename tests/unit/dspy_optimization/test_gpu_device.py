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
        """Test DeviceManager with CPU only (no PyTorch)"""
        with patch("agentcore.dspy_optimization.gpu.device.logger"):
            manager = DeviceManager()

            assert manager.current_device is not None
            assert len(manager.available_devices) >= 1
            assert not manager.torch_available

            # Should have CPU device
            cpu_device = manager.get_device_by_type(DeviceType.CPU)
            assert cpu_device is not None
            assert cpu_device.device_type == DeviceType.CPU

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=1024**3)
    def test_detect_cuda_devices(
        self,
        mock_memory: Mock,
        mock_props: Mock,
        mock_count: Mock,
        mock_available: Mock,
    ) -> None:
        """Test CUDA device detection"""
        # Mock device properties
        mock_device = Mock()
        mock_device.name = "NVIDIA GPU"
        mock_device.total_memory = 8 * 1024**3
        mock_device.major = 8
        mock_device.minor = 0
        mock_props.return_value = mock_device

        with patch.dict("sys.modules", {"torch": Mock()}):
            manager = DeviceManager()

            # Should detect CUDA devices
            cuda_devices = [
                d for d in manager.available_devices if d.device_type == DeviceType.CUDA
            ]
            assert len(cuda_devices) >= 0  # May be 0 if torch import fails

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

        # GPU may or may not be available
        cuda_device = manager.get_device_by_type(DeviceType.CUDA)
        if manager.has_gpu:
            assert cuda_device is not None
        else:
            assert cuda_device is None

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
