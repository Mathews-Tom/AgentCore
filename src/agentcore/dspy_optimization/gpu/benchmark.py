"""
Performance benchmarking for GPU acceleration

Provides tools for benchmarking CPU vs GPU performance, memory usage tracking,
and speed improvement analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType
from agentcore.dspy_optimization.gpu.memory import MemoryManager
from agentcore.dspy_optimization.gpu.tensor_ops import TensorOperations


@dataclass
class BenchmarkResult:
    """Result of performance benchmark"""

    operation_name: str
    cpu_time: float  # Seconds
    gpu_time: float | None  # Seconds (None if GPU unavailable)
    cpu_memory: int  # Bytes
    gpu_memory: int | None  # Bytes
    speedup: float | None  # GPU speedup factor
    efficiency: float  # Operations per second
    device_used: str


class PerformanceBenchmark:
    """
    Performance benchmarking for GPU operations

    Provides comprehensive benchmarking tools for comparing CPU vs GPU
    performance, analyzing memory usage, and measuring speed improvements.

    Key features:
    - CPU vs GPU comparison
    - Memory usage tracking
    - Speedup calculation
    - Multiple operation benchmarks
    - Statistical analysis
    """

    def __init__(self, device_manager: DeviceManager) -> None:
        """
        Initialize performance benchmark

        Args:
            device_manager: Device manager for GPU/CPU selection
        """
        self.device_manager = device_manager
        self.memory_manager = MemoryManager(device_manager)
        self.tensor_ops = TensorOperations(device_manager)

    def benchmark_operation(
        self,
        operation: Callable[..., Any],
        name: str,
        *args: Any,
        warmup: int = 3,
        iterations: int = 10,
        compare_cpu: bool = True,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Benchmark an operation on GPU and optionally CPU

        Args:
            operation: Operation to benchmark
            name: Operation name
            *args: Operation arguments
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
            compare_cpu: Compare with CPU performance
            **kwargs: Operation keyword arguments

        Returns:
            BenchmarkResult with timing and memory stats
        """
        # Determine device for GPU benchmark
        gpu_device_type = DeviceType.CPU  # Default fallback
        if self.device_manager.has_gpu:
            # Use whatever GPU is available (CUDA, ROCm, or Metal)
            gpu_device_type = self.device_manager.current_device.device_type
            if gpu_device_type == DeviceType.CPU:
                # If current device is CPU, try to find a GPU
                for device in self.device_manager.available_devices:
                    if device.device_type != DeviceType.CPU:
                        gpu_device_type = device.device_type
                        break

        # GPU benchmark
        gpu_time, gpu_memory = self._benchmark_device(
            operation,
            gpu_device_type,
            warmup,
            iterations,
            *args,
            **kwargs,
        )

        # CPU benchmark for comparison
        cpu_time = gpu_time
        cpu_memory = gpu_memory

        if compare_cpu and self.device_manager.has_gpu and gpu_device_type != DeviceType.CPU:
            cpu_time, cpu_memory = self._benchmark_device(
                operation,
                DeviceType.CPU,
                warmup,
                iterations,
                *args,
                **kwargs,
            )

        # Calculate speedup
        speedup = None
        if gpu_time and cpu_time and self.device_manager.has_gpu:
            speedup = cpu_time / gpu_time

        # Calculate efficiency (ops/sec)
        efficiency = iterations / gpu_time if gpu_time > 0 else 0

        device_used = self.device_manager.current_device.device_type.value

        return BenchmarkResult(
            operation_name=name,
            cpu_time=cpu_time,
            gpu_time=gpu_time if self.device_manager.has_gpu else None,
            cpu_memory=cpu_memory,
            gpu_memory=gpu_memory if self.device_manager.has_gpu else None,
            speedup=speedup,
            efficiency=efficiency,
            device_used=device_used,
        )

    def _benchmark_device(
        self,
        operation: Callable[..., Any],
        device_type: DeviceType,
        warmup: int,
        iterations: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[float, int]:
        """
        Benchmark operation on specific device

        Args:
            operation: Operation to benchmark
            device_type: Device to benchmark on
            warmup: Warmup iterations
            iterations: Benchmark iterations
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Tuple of (time_seconds, memory_bytes)
        """
        # Switch to device
        original_device = self.device_manager.current_device
        try:
            if device_type != original_device.device_type:
                self.device_manager.set_device(device_type)

            # Reset memory stats
            self.memory_manager.reset_peak_stats()

            # Warmup
            for _ in range(warmup):
                operation(*args, **kwargs)
                self.device_manager.synchronize()

            # Clear cache before timing
            self.memory_manager.cleanup()

            # Benchmark
            start_time = time.perf_counter()

            for _ in range(iterations):
                operation(*args, **kwargs)
                self.device_manager.synchronize()

            end_time = time.perf_counter()

            # Get memory stats
            stats = self.memory_manager.get_memory_stats()
            peak_memory = stats.peak_allocated

            elapsed_time = end_time - start_time

            return elapsed_time, peak_memory

        finally:
            # Restore original device
            if device_type != original_device.device_type:
                self.device_manager.set_device(
                    original_device.device_type,
                    original_device.device_id,
                )

    def benchmark_matrix_operations(
        self,
        matrix_size: int = 1000,
        iterations: int = 10,
    ) -> dict[str, BenchmarkResult]:
        """
        Benchmark matrix operations

        Args:
            matrix_size: Size of square matrices
            iterations: Number of iterations

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        # Generate test matrices
        a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

        # Matrix multiplication
        results["matmul"] = self.benchmark_operation(
            self.tensor_ops.matrix_multiply,
            "Matrix Multiplication",
            a,
            b,
            iterations=iterations,
        )

        return results

    def benchmark_distance_computation(
        self,
        n_points: int = 10000,
        n_centroids: int = 100,
        n_features: int = 128,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark distance computation (common in optimization)

        Args:
            n_points: Number of data points
            n_centroids: Number of centroids
            n_features: Feature dimensionality
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        points = np.random.randn(n_points, n_features).astype(np.float32)
        centroids = np.random.randn(n_centroids, n_features).astype(np.float32)

        return self.benchmark_operation(
            self.tensor_ops.compute_distances,
            "Distance Computation",
            points,
            centroids,
            iterations=iterations,
        )

    def benchmark_normalization(
        self,
        array_size: tuple[int, ...] = (10000, 128),
        iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark vector normalization

        Args:
            array_size: Size of array to normalize
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        data = np.random.randn(*array_size).astype(np.float32)

        return self.benchmark_operation(
            self.tensor_ops.normalize,
            "Vector Normalization",
            data,
            iterations=iterations,
        )

    def benchmark_cosine_similarity(
        self,
        n_vectors_a: int = 1000,
        n_vectors_b: int = 1000,
        n_features: int = 256,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark cosine similarity computation

        Args:
            n_vectors_a: Number of vectors in first set
            n_vectors_b: Number of vectors in second set
            n_features: Feature dimensionality
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        a = np.random.randn(n_vectors_a, n_features).astype(np.float32)
        b = np.random.randn(n_vectors_b, n_features).astype(np.float32)

        return self.benchmark_operation(
            self.tensor_ops.cosine_similarity,
            "Cosine Similarity",
            a,
            b,
            iterations=iterations,
        )

    def run_comprehensive_benchmark(self) -> dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark suite

        Returns:
            Dictionary of all benchmark results
        """
        results = {}

        # Matrix operations
        results.update(self.benchmark_matrix_operations())

        # Distance computation
        results["distances"] = self.benchmark_distance_computation()

        # Normalization
        results["normalization"] = self.benchmark_normalization()

        # Cosine similarity
        results["cosine_similarity"] = self.benchmark_cosine_similarity()

        return results

    def print_benchmark_results(self, results: dict[str, BenchmarkResult]) -> None:
        """
        Print benchmark results in formatted table

        Args:
            results: Dictionary of benchmark results
        """
        print("\n" + "=" * 100)
        print("GPU ACCELERATION BENCHMARK RESULTS")
        print("=" * 100)

        for name, result in results.items():
            print(f"\n{result.operation_name}:")
            print(f"  Device: {result.device_used}")
            print(f"  CPU Time: {result.cpu_time * 1000:.2f} ms")

            if result.gpu_time is not None:
                print(f"  GPU Time: {result.gpu_time * 1000:.2f} ms")

            if result.speedup is not None:
                print(f"  Speedup: {result.speedup:.2f}x")

            print(f"  CPU Memory: {result.cpu_memory / 1024**2:.2f} MB")

            if result.gpu_memory is not None:
                print(f"  GPU Memory: {result.gpu_memory / 1024**2:.2f} MB")

            print(f"  Efficiency: {result.efficiency:.2f} ops/sec")

        print("\n" + "=" * 100)


def benchmark_operation(
    operation: Callable[..., Any],
    name: str,
    device_manager: DeviceManager,
    *args: Any,
    **kwargs: Any,
) -> BenchmarkResult:
    """
    Convenience function to benchmark a single operation

    Args:
        operation: Operation to benchmark
        name: Operation name
        device_manager: Device manager
        *args: Operation arguments
        **kwargs: Operation keyword arguments

    Returns:
        BenchmarkResult
    """
    benchmark = PerformanceBenchmark(device_manager)
    return benchmark.benchmark_operation(operation, name, *args, **kwargs)
