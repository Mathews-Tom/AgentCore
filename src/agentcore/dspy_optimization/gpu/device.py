"""
Device detection and management for GPU acceleration

Provides automatic detection of CUDA, ROCm, and Metal devices with graceful
fallback to CPU when GPU is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported device types for acceleration"""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"


@dataclass
class DeviceInfo:
    """Information about available compute device"""

    device_type: DeviceType
    device_id: int
    name: str
    total_memory: int  # In bytes
    available_memory: int  # In bytes
    compute_capability: str | None = None
    is_available: bool = True


class DeviceManager:
    """
    Manager for compute device detection and selection

    Automatically detects available GPU devices (CUDA, ROCm, Metal) and
    provides graceful fallback to CPU when GPU is unavailable.

    Key features:
    - Automatic device detection
    - Multi-GPU support
    - Memory tracking
    - Device switching
    - Graceful CPU fallback
    """

    def __init__(self) -> None:
        """Initialize device manager and detect available devices"""
        self._current_device: DeviceInfo | None = None
        self._available_devices: list[DeviceInfo] = []
        self._torch_available = False
        self._torch: Any = None

        self._detect_devices()

    def _detect_devices(self) -> None:
        """Detect all available compute devices"""
        # Try to import PyTorch for GPU support
        try:
            import torch

            self._torch = torch
            self._torch_available = True
        except ImportError:
            logger.warning("PyTorch not available, GPU acceleration disabled")
            self._add_cpu_device()
            return

        # Detect CUDA devices
        if self._torch.cuda.is_available():
            self._detect_cuda_devices()

        # Detect ROCm devices (AMD GPUs using PyTorch ROCm build)
        if hasattr(self._torch, "hip") and self._torch.hip.is_available():
            self._detect_rocm_devices()

        # Detect Metal devices (Apple Silicon)
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            self._detect_metal_devices()

        # Always add CPU as fallback
        self._add_cpu_device()

        # Set default device (prefer GPU if available)
        if self._available_devices:
            self._current_device = self._available_devices[0]
            logger.info(
                f"Default device set to: {self._current_device.device_type.value} "
                f"({self._current_device.name})"
            )

    def _detect_cuda_devices(self) -> None:
        """Detect NVIDIA CUDA devices"""
        if not self._torch:
            return

        device_count = self._torch.cuda.device_count()
        logger.info(f"Detected {device_count} CUDA device(s)")

        for device_id in range(device_count):
            properties = self._torch.cuda.get_device_properties(device_id)
            total_memory = properties.total_memory
            available_memory = total_memory - self._torch.cuda.memory_allocated(device_id)

            device_info = DeviceInfo(
                device_type=DeviceType.CUDA,
                device_id=device_id,
                name=properties.name,
                total_memory=total_memory,
                available_memory=available_memory,
                compute_capability=f"{properties.major}.{properties.minor}",
            )
            self._available_devices.append(device_info)
            logger.info(
                f"CUDA Device {device_id}: {device_info.name} "
                f"({device_info.total_memory / 1024**3:.2f} GB)"
            )

    def _detect_rocm_devices(self) -> None:
        """Detect AMD ROCm devices"""
        if not self._torch or not hasattr(self._torch, "hip"):
            return

        device_count = self._torch.hip.device_count()
        logger.info(f"Detected {device_count} ROCm device(s)")

        for device_id in range(device_count):
            # ROCm devices through PyTorch HIP interface
            device_info = DeviceInfo(
                device_type=DeviceType.ROCM,
                device_id=device_id,
                name=f"AMD GPU {device_id}",
                total_memory=0,  # Would need ROCm-specific API
                available_memory=0,
            )
            self._available_devices.append(device_info)
            logger.info(f"ROCm Device {device_id}: {device_info.name}")

    def _detect_metal_devices(self) -> None:
        """Detect Apple Metal devices (Apple Silicon)"""
        if not self._torch:
            return

        # Metal Performance Shaders backend for Apple Silicon
        device_info = DeviceInfo(
            device_type=DeviceType.METAL,
            device_id=0,
            name="Apple Silicon GPU",
            total_memory=0,  # Would need Metal-specific API
            available_memory=0,
        )
        self._available_devices.append(device_info)
        logger.info("Metal Device: Apple Silicon GPU")

    def _add_cpu_device(self) -> None:
        """Add CPU as fallback device"""
        import psutil

        # Get system memory info
        memory_info = psutil.virtual_memory()

        device_info = DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name="CPU",
            total_memory=memory_info.total,
            available_memory=memory_info.available,
        )
        self._available_devices.append(device_info)
        logger.info(f"CPU Device: {memory_info.total / 1024**3:.2f} GB RAM")

    @property
    def current_device(self) -> DeviceInfo:
        """Get current active device"""
        if not self._current_device:
            raise RuntimeError("No device available")
        return self._current_device

    @property
    def available_devices(self) -> list[DeviceInfo]:
        """Get list of all available devices"""
        return self._available_devices.copy()

    @property
    def has_gpu(self) -> bool:
        """Check if GPU device is available"""
        return any(
            device.device_type != DeviceType.CPU
            for device in self._available_devices
        )

    @property
    def torch_available(self) -> bool:
        """Check if PyTorch is available"""
        return self._torch_available

    def get_device_by_type(self, device_type: DeviceType) -> DeviceInfo | None:
        """
        Get device by type

        Args:
            device_type: Type of device to get

        Returns:
            DeviceInfo if found, None otherwise
        """
        for device in self._available_devices:
            if device.device_type == device_type:
                return device
        return None

    def set_device(self, device_type: DeviceType, device_id: int = 0) -> None:
        """
        Set current active device

        Args:
            device_type: Type of device to set
            device_id: Device ID (for multi-GPU systems)

        Raises:
            ValueError: If device not found
        """
        for device in self._available_devices:
            if device.device_type == device_type and device.device_id == device_id:
                self._current_device = device

                # Set PyTorch device if available
                if self._torch_available and self._torch:
                    if device_type == DeviceType.CUDA:
                        self._torch.cuda.set_device(device_id)
                    elif device_type == DeviceType.METAL:
                        # Metal uses "mps" device
                        pass

                logger.info(f"Switched to device: {device_type.value}:{device_id}")
                return

        raise ValueError(
            f"Device {device_type.value}:{device_id} not found"
        )

    def get_torch_device(self) -> Any:
        """
        Get PyTorch device object for current device

        Returns:
            torch.device object

        Raises:
            RuntimeError: If PyTorch not available
        """
        if not self._torch_available or not self._torch:
            raise RuntimeError("PyTorch not available")

        device_type = self.current_device.device_type

        if device_type == DeviceType.CUDA:
            return self._torch.device(f"cuda:{self.current_device.device_id}")
        elif device_type == DeviceType.METAL:
            return self._torch.device("mps")
        elif device_type == DeviceType.CPU:
            return self._torch.device("cpu")
        else:
            # Fallback to CPU
            logger.warning(f"Unknown device type {device_type}, falling back to CPU")
            return self._torch.device("cpu")

    def synchronize(self) -> None:
        """Synchronize device (wait for all operations to complete)"""
        if not self._torch_available or not self._torch:
            return

        device_type = self.current_device.device_type

        if device_type == DeviceType.CUDA:
            self._torch.cuda.synchronize()
        elif device_type == DeviceType.METAL:
            # Metal synchronization
            if hasattr(self._torch.mps, "synchronize"):
                self._torch.mps.synchronize()

    def get_memory_stats(self) -> dict[str, int]:
        """
        Get memory statistics for current device

        Returns:
            Dictionary with memory stats (allocated, cached, total)
        """
        if not self._torch_available or not self._torch:
            return {"allocated": 0, "cached": 0, "total": 0}

        device_type = self.current_device.device_type

        if device_type == DeviceType.CUDA:
            return {
                "allocated": self._torch.cuda.memory_allocated(),
                "cached": self._torch.cuda.memory_reserved(),
                "total": self.current_device.total_memory,
            }
        else:
            return {
                "allocated": 0,
                "cached": 0,
                "total": self.current_device.total_memory,
            }

    def clear_cache(self) -> None:
        """Clear device memory cache"""
        if not self._torch_available or not self._torch:
            return

        device_type = self.current_device.device_type

        if device_type == DeviceType.CUDA:
            self._torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")


# Global device manager instance
_device_manager: DeviceManager | None = None


def get_device_manager() -> DeviceManager:
    """
    Get global device manager instance

    Returns:
        DeviceManager singleton
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager
