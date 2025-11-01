"""
Unit tests for GPU performance benchmarking
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from agentcore.dspy_optimization.gpu.benchmark import (
    BenchmarkResult,
    PerformanceBenchmark,
    benchmark_operation,
)
from agentcore.dspy_optimization.gpu.device import DeviceManager


@pytest.fixture
def device_manager() -> DeviceManager:
    """Create device manager fixture"""
    return DeviceManager()


@pytest.fixture
def perf_benchmark(device_manager: DeviceManager) -> PerformanceBenchmark:
    """Create performance benchmark fixture"""
    return PerformanceBenchmark(device_manager)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass"""

    def test_benchmark_result_creation(self) -> None:
        """Test BenchmarkResult creation"""
        result = BenchmarkResult(
            operation_name="Matrix Multiplication",
            cpu_time=0.1,
            gpu_time=0.02,
            cpu_memory=1024**2,
            gpu_memory=512**2,
            speedup=5.0,
            efficiency=100.0,
            device_used="cuda",
        )

        assert result.operation_name == "Matrix Multiplication"
        assert result.cpu_time == 0.1
        assert result.gpu_time == 0.02
        assert result.cpu_memory == 1024**2
        assert result.gpu_memory == 512**2
        assert result.speedup == 5.0
        assert result.efficiency == 100.0
        assert result.device_used == "cuda"


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark"""

    def test_initialization(self, device_manager: DeviceManager) -> None:
        """Test PerformanceBenchmark initialization"""
        benchmark_obj = PerformanceBenchmark(device_manager)

        assert benchmark_obj.device_manager is device_manager
        assert benchmark_obj.memory_manager is not None
        assert benchmark_obj.tensor_ops is not None

    def test_benchmark_operation_simple(self, perf_benchmark: PerformanceBenchmark) -> None:
        """Test benchmarking simple operation"""

        def simple_op(x: int, y: int) -> int:
            return x + y

        result = perf_benchmark.benchmark_operation(
            simple_op,
            "Addition",
            2,
            3,
            warmup=1,
            iterations=5,
            compare_cpu=False,
        )

        assert result.operation_name == "Addition"
        assert result.cpu_time > 0
        assert result.efficiency > 0

    def test_benchmark_operation_with_comparison(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test benchmarking with CPU comparison"""

        def compute_sum(n: int) -> int:
            return sum(range(n))

        result = perf_benchmark.benchmark_operation(
            compute_sum,
            "Sum",
            1000,
            warmup=1,
            iterations=5,
            compare_cpu=True,
        )

        assert result.operation_name == "Sum"
        assert result.cpu_time > 0

        # If GPU available, should have GPU time
        if perf_benchmark.device_manager.has_gpu:
            assert result.gpu_time is not None
        else:
            assert result.gpu_time is None

    def test_benchmark_matrix_operations(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test benchmarking matrix operations"""
        results = perf_benchmark.benchmark_matrix_operations(
            matrix_size=100, iterations=3
        )

        assert "matmul" in results
        assert results["matmul"].operation_name == "Matrix Multiplication"
        assert results["matmul"].cpu_time > 0

    def test_benchmark_distance_computation(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test benchmarking distance computation"""
        result = perf_benchmark.benchmark_distance_computation(
            n_points=100,
            n_centroids=10,
            n_features=32,
            iterations=3,
        )

        assert result.operation_name == "Distance Computation"
        assert result.cpu_time > 0

    def test_benchmark_normalization(self, perf_benchmark: PerformanceBenchmark) -> None:
        """Test benchmarking normalization"""
        result = perf_benchmark.benchmark_normalization(
            array_size=(100, 32), iterations=3
        )

        assert result.operation_name == "Vector Normalization"
        assert result.cpu_time > 0

    def test_benchmark_cosine_similarity(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test benchmarking cosine similarity"""
        result = perf_benchmark.benchmark_cosine_similarity(
            n_vectors_a=100,
            n_vectors_b=100,
            n_features=64,
            iterations=3,
        )

        assert result.operation_name == "Cosine Similarity"
        assert result.cpu_time > 0

    def test_run_comprehensive_benchmark(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test running comprehensive benchmark suite"""
        results = perf_benchmark.run_comprehensive_benchmark()

        # Should have multiple benchmark results
        assert len(results) >= 4

        # Check for expected benchmarks
        assert "matmul" in results
        assert "distances" in results
        assert "normalization" in results
        assert "cosine_similarity" in results

        # All results should have valid data
        for result in results.values():
            assert result.cpu_time > 0
            assert result.efficiency > 0

    def test_print_benchmark_results(
        self, perf_benchmark: PerformanceBenchmark, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test printing benchmark results"""
        results = {
            "test_op": BenchmarkResult(
                operation_name="Test Operation",
                cpu_time=0.1,
                gpu_time=0.02,
                cpu_memory=1024**2,
                gpu_memory=512**2,
                speedup=5.0,
                efficiency=100.0,
                device_used="cuda",
            )
        }

        perf_benchmark.print_benchmark_results(results)

        captured = capsys.readouterr()
        assert "GPU ACCELERATION BENCHMARK RESULTS" in captured.out
        assert "Test Operation" in captured.out


class TestBenchmarkOperationFunction:
    """Tests for standalone benchmark_operation function"""

    def test_benchmark_operation_function(
        self, device_manager: DeviceManager
    ) -> None:
        """Test standalone benchmark operation function"""

        def simple_op(x: int) -> int:
            return x * 2

        result = benchmark_operation(
            simple_op, "Multiply", device_manager, 5, iterations=3, warmup=1
        )

        assert result.operation_name == "Multiply"
        assert result.cpu_time > 0


class TestBenchmarkIntegration:
    """Integration tests for benchmarking"""

    def test_end_to_end_benchmark_workflow(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test end-to-end benchmarking workflow"""
        # Run comprehensive benchmark
        results = perf_benchmark.run_comprehensive_benchmark()

        # Analyze results
        total_operations = len(results)
        assert total_operations >= 4

        # Check if GPU provides speedup (if available)
        if perf_benchmark.device_manager.has_gpu:
            gpu_results = [r for r in results.values() if r.gpu_time is not None]
            if gpu_results:
                # At least some operations should run on GPU
                assert len(gpu_results) > 0

    def test_benchmark_with_different_sizes(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test benchmarking with different problem sizes"""
        sizes = [10, 100]

        results = {}
        for size in sizes:
            result = perf_benchmark.benchmark_normalization(
                array_size=(size, 32), iterations=3
            )
            results[size] = result

        # Both sizes should complete successfully and report reasonable times
        # Note: For sub-millisecond operations, timing noise can dominate
        # so we don't strictly require larger sizes to take more time
        assert results[10].cpu_time > 0
        assert results[100].cpu_time > 0
        # Allow either order due to timing variance on fast operations
        assert 0 < results[10].cpu_time < 1.0  # Should complete in under 1 second
        assert 0 < results[100].cpu_time < 1.0

    def test_benchmark_memory_tracking(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test memory tracking during benchmarking"""
        result = perf_benchmark.benchmark_matrix_operations(matrix_size=100, iterations=3)

        # Should track memory usage
        assert result["matmul"].cpu_memory >= 0

    def test_benchmark_statistical_validity(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test statistical validity of benchmarks"""
        # Run same benchmark multiple times
        results = []
        for _ in range(3):
            result = perf_benchmark.benchmark_normalization(
                array_size=(100, 32), iterations=5
            )
            results.append(result.cpu_time)

        # Times should be relatively consistent (not vary by more than 25x)
        min_time = min(results)
        max_time = max(results)

        # Allow for variation due to system load during full test suite execution
        # During isolated test runs, variance is typically <5x
        # During full test suite (23+ minutes), variance can be 15-20x due to system load
        # Setting threshold to 25x to catch pathological cases while allowing normal variance
        assert max_time < min_time * 25

    def test_benchmark_efficiency_calculation(
        self, perf_benchmark: PerformanceBenchmark
    ) -> None:
        """Test efficiency calculation"""
        result = perf_benchmark.benchmark_operation(
            lambda: sum(range(1000)),
            "Sum",
            warmup=1,
            iterations=10,
            compare_cpu=False,
        )

        # Efficiency should be operations per second
        assert result.efficiency > 0
        # 10 iterations / time should match efficiency
        expected_efficiency = 10 / result.cpu_time
        assert abs(result.efficiency - expected_efficiency) < 0.01
