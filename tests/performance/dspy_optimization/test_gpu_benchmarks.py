"""
Performance tests for GPU acceleration

Validates GPU performance with CPU vs GPU comparisons and memory usage.
Uses DSP-010 GPU infrastructure for comprehensive benchmarking.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    from agentcore.dspy_optimization.gpu.benchmark import (
        BenchmarkResult,
        PerformanceBenchmark,
    )
    from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType
    from agentcore.dspy_optimization.gpu.memory import MemoryManager
    from agentcore.dspy_optimization.gpu.tensor_ops import TensorOperations
except ImportError as e:
    pytest.skip(f"Required dependencies not available: {e}", allow_module_level=True)

# Skip CUDA-dependent tests on non-CUDA systems (e.g., Apple Silicon with Metal)
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available - tests require NVIDIA GPU"
)


class TestGPUBenchmarks:
    """Test GPU acceleration benchmarks"""

    @pytest.fixture
    def device_manager(self):
        """Initialize device manager"""
        return DeviceManager()

    @pytest.fixture
    def perf_benchmark(self, device_manager):
        """Initialize performance benchmark"""
        return PerformanceBenchmark(device_manager)

    @pytest.mark.performance
    def test_gpu_availability(self, device_manager):
        """Test GPU availability detection"""
        print(f"\nGPU Available: {device_manager.has_gpu}")
        print(f"Current Device: {device_manager.current_device.device_type.value}")
        print(f"Available Devices: {len(device_manager.available_devices)}")

        for device in device_manager.available_devices:
            print(f"  - {device.device_type.value} (ID: {device.device_id})")

        # Should have at least CPU
        assert len(device_manager.available_devices) >= 1
        assert any(d.device_type == DeviceType.CPU for d in device_manager.available_devices)

    @pytest.mark.performance
    def test_matrix_multiplication_benchmark(self, perf_benchmark):
        """Benchmark matrix multiplication (CPU vs GPU)"""
        results = perf_benchmark.benchmark_matrix_operations(
            matrix_size=1000,
            iterations=10
        )

        assert "matmul" in results
        result = results["matmul"]

        print(f"\nMatrix Multiplication (1000x1000):")
        print(f"  CPU Time: {result.cpu_time * 1000:.2f} ms")
        if result.gpu_time is not None:
            print(f"  GPU Time: {result.gpu_time * 1000:.2f} ms")
            if result.speedup is not None:
                print(f"  Speedup: {result.speedup:.2f}x")

        print(f"  CPU Memory: {result.cpu_memory / 1024**2:.2f} MB")
        if result.gpu_memory is not None:
            print(f"  GPU Memory: {result.gpu_memory / 1024**2:.2f} MB")

        print(f"  Efficiency: {result.efficiency:.2f} ops/sec")

        # Verify results
        assert result.cpu_time > 0
        assert result.cpu_memory > 0
        assert result.efficiency > 0

        # If GPU available, should see speedup
        if result.gpu_time is not None and result.speedup is not None:
            print(f"\nGPU VALIDATION: {result.speedup:.2f}x speedup achieved")

    @pytest.mark.performance
    def test_distance_computation_benchmark(self, perf_benchmark):
        """Benchmark distance computation (common in optimization)"""
        result = perf_benchmark.benchmark_distance_computation(
            n_points=10000,
            n_centroids=100,
            n_features=128,
            iterations=10
        )

        print(f"\nDistance Computation (10k points, 100 centroids, 128 features):")
        print(f"  CPU Time: {result.cpu_time * 1000:.2f} ms")
        if result.gpu_time is not None:
            print(f"  GPU Time: {result.gpu_time * 1000:.2f} ms")
            if result.speedup is not None:
                print(f"  Speedup: {result.speedup:.2f}x")

        assert result.cpu_time > 0
        assert result.efficiency > 0

    @pytest.mark.performance
    def test_normalization_benchmark(self, perf_benchmark):
        """Benchmark vector normalization"""
        result = perf_benchmark.benchmark_normalization(
            array_size=(10000, 128),
            iterations=10
        )

        print(f"\nVector Normalization (10k x 128):")
        print(f"  CPU Time: {result.cpu_time * 1000:.2f} ms")
        if result.gpu_time is not None:
            print(f"  GPU Time: {result.gpu_time * 1000:.2f} ms")
            if result.speedup is not None:
                print(f"  Speedup: {result.speedup:.2f}x")

        assert result.cpu_time > 0
        assert result.efficiency > 0

    @pytest.mark.performance
    def test_cosine_similarity_benchmark(self, perf_benchmark):
        """Benchmark cosine similarity computation"""
        result = perf_benchmark.benchmark_cosine_similarity(
            n_vectors_a=1000,
            n_vectors_b=1000,
            n_features=256,
            iterations=10
        )

        print(f"\nCosine Similarity (1k x 1k vectors, 256 features):")
        print(f"  CPU Time: {result.cpu_time * 1000:.2f} ms")
        if result.gpu_time is not None:
            print(f"  GPU Time: {result.gpu_time * 1000:.2f} ms")
            if result.speedup is not None:
                print(f"  Speedup: {result.speedup:.2f}x")

        assert result.cpu_time > 0
        assert result.efficiency > 0

    @pytest.mark.performance
    @pytest.mark.slow
    def test_comprehensive_benchmark_suite(self, perf_benchmark):
        """Run comprehensive GPU benchmark suite"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE GPU BENCHMARK SUITE")
        print("=" * 80)

        results = perf_benchmark.run_comprehensive_benchmark()

        # Print all results
        perf_benchmark.print_benchmark_results(results)

        # Verify all benchmarks ran
        assert len(results) >= 4
        assert "matmul" in results
        assert "distances" in results
        assert "normalization" in results
        assert "cosine_similarity" in results

        # Verify all completed successfully
        for name, result in results.items():
            assert result.cpu_time > 0
            assert result.efficiency > 0

        # If GPU available, calculate average speedup
        if any(r.speedup is not None for r in results.values()):
            speedups = [r.speedup for r in results.values() if r.speedup is not None]
            avg_speedup = sum(speedups) / len(speedups)
            print(f"\nAverage GPU Speedup: {avg_speedup:.2f}x")

    @pytest.mark.performance
    def test_memory_usage_tracking(self, device_manager):
        """Test memory usage tracking"""
        memory_manager = MemoryManager(device_manager)
        tensor_ops = TensorOperations(device_manager)

        # Reset stats
        memory_manager.reset_peak_stats()

        # Perform operations
        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)

        result = tensor_ops.matrix_multiply(a, b)

        # Get memory stats
        stats = memory_manager.get_memory_stats()

        print(f"\nMemory Usage:")
        print(f"  Current Allocated: {stats.current_allocated / 1024**2:.2f} MB")
        print(f"  Peak Allocated: {stats.peak_allocated / 1024**2:.2f} MB")
        print(f"  Total Reserved: {stats.total_reserved / 1024**2:.2f} MB")
        print(f"  Active Tensors: {stats.active_tensors}")

        assert stats.current_allocated >= 0
        assert stats.peak_allocated >= 0

    @pytest.mark.performance
    def test_memory_cleanup(self, device_manager):
        """Test memory cleanup"""
        memory_manager = MemoryManager(device_manager)
        tensor_ops = TensorOperations(device_manager)

        # Create large tensors
        tensors = []
        for i in range(10):
            a = np.random.randn(100, 100).astype(np.float32)
            b = np.random.randn(100, 100).astype(np.float32)
            result = tensor_ops.matrix_multiply(a, b)
            tensors.append(result)

        stats_before = memory_manager.get_memory_stats()

        # Cleanup
        tensors.clear()
        memory_manager.cleanup()

        stats_after = memory_manager.get_memory_stats()

        print(f"\nMemory Cleanup:")
        print(f"  Before: {stats_before.current_allocated / 1024**2:.2f} MB")
        print(f"  After: {stats_after.current_allocated / 1024**2:.2f} MB")
        print(f"  Freed: {(stats_before.current_allocated - stats_after.current_allocated) / 1024**2:.2f} MB")

    @pytest.mark.performance
    def test_tensor_operations_correctness(self, device_manager):
        """Test tensor operations produce correct results"""
        tensor_ops = TensorOperations(device_manager)

        # Test matrix multiplication
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)

        result = tensor_ops.matrix_multiply(a, b)

        # Expected: [[19, 22], [43, 50]]
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.performance
    def test_device_switching(self, device_manager):
        """Test switching between devices"""
        original_device = device_manager.current_device

        print(f"\nOriginal Device: {original_device.device_type.value}")

        # Try switching to CPU
        device_manager.set_device(DeviceType.CPU)
        assert device_manager.current_device.device_type == DeviceType.CPU

        # Switch back
        device_manager.set_device(
            original_device.device_type,
            original_device.device_id
        )

        assert device_manager.current_device.device_type == original_device.device_type

    @pytest.mark.performance
    def test_synchronization(self, device_manager):
        """Test device synchronization"""
        tensor_ops = TensorOperations(device_manager)

        start = time.perf_counter()

        # Perform operations
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)
        result = tensor_ops.matrix_multiply(a, b)

        # Synchronize
        device_manager.synchronize()

        elapsed = time.perf_counter() - start

        print(f"\nSynchronization test completed in {elapsed * 1000:.2f} ms")
        assert elapsed < 1.0  # Should be fast

    @pytest.mark.performance
    @pytest.mark.parametrize("matrix_size", [100, 500, 1000, 2000])
    def test_scaling_with_matrix_size(self, perf_benchmark, matrix_size):
        """Test performance scaling with matrix size"""
        results = perf_benchmark.benchmark_matrix_operations(
            matrix_size=matrix_size,
            iterations=5
        )

        result = results["matmul"]

        print(f"\nMatrix size {matrix_size}x{matrix_size}:")
        print(f"  CPU Time: {result.cpu_time * 1000:.2f} ms")
        if result.speedup is not None:
            print(f"  GPU Speedup: {result.speedup:.2f}x")

        assert result.cpu_time > 0

    @pytest.mark.performance
    def test_warmup_effect(self, perf_benchmark, device_manager):
        """Test effect of warmup iterations"""
        tensor_ops = TensorOperations(device_manager)

        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)

        # First run (cold)
        start = time.perf_counter()
        tensor_ops.matrix_multiply(a, b)
        device_manager.synchronize()
        cold_time = time.perf_counter() - start

        # Warmup
        for _ in range(5):
            tensor_ops.matrix_multiply(a, b)
            device_manager.synchronize()

        # Warmed run
        start = time.perf_counter()
        tensor_ops.matrix_multiply(a, b)
        device_manager.synchronize()
        warm_time = time.perf_counter() - start

        print(f"\nWarmup Effect:")
        print(f"  Cold: {cold_time * 1000:.2f} ms")
        print(f"  Warm: {warm_time * 1000:.2f} ms")
        print(f"  Improvement: {cold_time / warm_time:.2f}x")

        # Warm run should typically be faster or similar
        assert warm_time <= cold_time * 2  # Allow some variance

    @pytest.mark.performance
    def test_batch_operations_efficiency(self, device_manager):
        """Test efficiency of batch operations"""
        tensor_ops = TensorOperations(device_manager)

        # Single large operation
        start = time.perf_counter()
        a_large = np.random.randn(10000, 128).astype(np.float32)
        tensor_ops.normalize(a_large)
        device_manager.synchronize()
        batch_time = time.perf_counter() - start

        # Multiple small operations
        start = time.perf_counter()
        for i in range(100):
            a_small = np.random.randn(100, 128).astype(np.float32)
            tensor_ops.normalize(a_small)
            device_manager.synchronize()
        individual_time = time.perf_counter() - start

        print(f"\nBatch vs Individual:")
        print(f"  Batch (10k): {batch_time * 1000:.2f} ms")
        print(f"  Individual (100x100): {individual_time * 1000:.2f} ms")
        print(f"  Efficiency gain: {individual_time / batch_time:.2f}x")

        # Batch should be more efficient
        assert batch_time < individual_time

    @pytest.mark.performance
    def test_memory_pressure(self, device_manager):
        """Test behavior under memory pressure"""
        memory_manager = MemoryManager(device_manager)
        tensor_ops = TensorOperations(device_manager)

        memory_manager.reset_peak_stats()

        # Allocate increasingly large tensors
        tensors = []
        for size in [100, 500, 1000, 2000]:
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            result = tensor_ops.matrix_multiply(a, b)
            tensors.append(result)

            stats = memory_manager.get_memory_stats()
            print(f"\nSize {size}x{size}:")
            print(f"  Memory: {stats.current_allocated / 1024**2:.2f} MB")

        # Should handle without errors
        assert len(tensors) == 4

        # Cleanup
        tensors.clear()
        memory_manager.cleanup()
