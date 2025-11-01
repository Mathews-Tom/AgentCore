"""
GPU acceleration module for DSPy optimization

Provides GPU acceleration for optimization algorithms with automatic device
detection, memory management, and graceful fallback to CPU when GPU is unavailable.
"""

from agentcore.dspy_optimization.gpu.device import (
    DeviceManager,
    DeviceType,
    DeviceInfo,
    get_device_manager,
)
from agentcore.dspy_optimization.gpu.tensor_ops import (
    TensorOperations,
    BatchProcessor,
)
from agentcore.dspy_optimization.gpu.memory import (
    MemoryManager,
    MemoryPool,
    MemoryStats,
)
from agentcore.dspy_optimization.gpu.benchmark import (
    PerformanceBenchmark,
    BenchmarkResult,
    benchmark_operation,
)

__all__ = [
    "DeviceManager",
    "DeviceType",
    "DeviceInfo",
    "get_device_manager",
    "TensorOperations",
    "BatchProcessor",
    "MemoryManager",
    "MemoryPool",
    "MemoryStats",
    "PerformanceBenchmark",
    "BenchmarkResult",
    "benchmark_operation",
]
