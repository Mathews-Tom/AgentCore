"""
Unit tests for GPU tensor operations
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType
from agentcore.dspy_optimization.gpu.tensor_ops import BatchProcessor, TensorOperations


@pytest.fixture
def device_manager() -> DeviceManager:
    """Create device manager fixture"""
    return DeviceManager()


@pytest.fixture
def tensor_ops(device_manager: DeviceManager) -> TensorOperations:
    """Create tensor operations fixture"""
    return TensorOperations(device_manager)


class TestTensorOperations:
    """Tests for TensorOperations"""

    def test_initialization(self, device_manager: DeviceManager) -> None:
        """Test TensorOperations initialization"""
        ops = TensorOperations(device_manager)
        assert ops.device_manager is device_manager

    def test_to_numpy_from_numpy(self, tensor_ops: TensorOperations) -> None:
        """Test converting numpy array to numpy"""
        arr = np.array([[1, 2], [3, 4]])
        result = tensor_ops.to_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_matrix_multiply_cpu(self, tensor_ops: TensorOperations) -> None:
        """Test matrix multiplication on CPU"""
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)

        result = tensor_ops.matrix_multiply(a, b)

        expected = np.matmul(a, b)
        np.testing.assert_array_almost_equal(result, expected)

    def test_batch_matrix_multiply_cpu(self, tensor_ops: TensorOperations) -> None:
        """Test batched matrix multiplication on CPU"""
        matrices_a = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        ]
        matrices_b = [
            np.array([[2, 0], [1, 2]], dtype=np.float32),
            np.array([[1, 1], [1, 1]], dtype=np.float32),
        ]

        results = tensor_ops.batch_matrix_multiply(matrices_a, matrices_b)

        assert len(results) == 2
        for i, result in enumerate(results):
            expected = np.matmul(matrices_a[i], matrices_b[i])
            np.testing.assert_array_almost_equal(result, expected)

    def test_compute_distances_cpu(self, tensor_ops: TensorOperations) -> None:
        """Test distance computation on CPU"""
        points = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        centroids = np.array([[0, 0], [3, 3]], dtype=np.float32)

        distances = tensor_ops.compute_distances(points, centroids)

        assert distances.shape == (3, 2)
        # Distance from [0,0] to [0,0] should be 0
        assert abs(distances[0, 0]) < 0.01
        # Distance from [1,1] to [0,0] should be sqrt(2)
        assert abs(distances[1, 0] - np.sqrt(2)) < 0.01

    def test_normalize_cpu(self, tensor_ops: TensorOperations) -> None:
        """Test vector normalization on CPU"""
        data = np.array([[3, 4], [6, 8]], dtype=np.float32)

        normalized = tensor_ops.normalize(data)

        # Check unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_cosine_similarity_cpu(self, tensor_ops: TensorOperations) -> None:
        """Test cosine similarity on CPU"""
        a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        b = np.array([[1, 0], [1, 1]], dtype=np.float32)

        similarity = tensor_ops.cosine_similarity(a, b)

        assert similarity.shape == (2, 2)
        # Similarity of a[0]=[1,0] with b[0]=[1,0] should be 1
        assert abs(similarity[0, 0] - 1.0) < 0.01
        # Similarity of a[0]=[1,0] with b[1]=[1,1] should be 1/sqrt(2) ≈ 0.707
        assert abs(similarity[0, 1] - 0.707) < 0.01


class TestBatchProcessor:
    """Tests for BatchProcessor"""

    def test_initialization(self, device_manager: DeviceManager) -> None:
        """Test BatchProcessor initialization"""
        processor = BatchProcessor(device_manager)

        assert processor.device_manager is device_manager
        assert processor.batch_size > 0

    def test_initialization_with_batch_size(
        self, device_manager: DeviceManager
    ) -> None:
        """Test BatchProcessor with specified batch size"""
        processor = BatchProcessor(device_manager, batch_size=64)

        assert processor.batch_size == 64

    def test_determine_batch_size_cpu(self, device_manager: DeviceManager) -> None:
        """Test batch size determination for CPU"""
        device_manager.set_device(DeviceType.CPU, 0)
        processor = BatchProcessor(device_manager, batch_size=None)

        # Should use conservative batch size for CPU
        assert processor.batch_size >= 16

    def test_process_batches(self, device_manager: DeviceManager) -> None:
        """Test batch processing"""
        processor = BatchProcessor(device_manager, batch_size=2)

        data = [1, 2, 3, 4, 5]

        def operation(batch: list[int]) -> list[int]:
            return [x * 2 for x in batch]

        results = processor.process_batches(data, operation)

        assert results == [2, 4, 6, 8, 10]

    def test_process_batches_with_error_recovery(
        self, device_manager: DeviceManager
    ) -> None:
        """Test batch processing with error recovery"""
        processor = BatchProcessor(device_manager, batch_size=2)

        data = [1, 2, 3, 4]
        call_count = [0]

        def operation(batch: list[int]) -> list[int]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Batch error")
            return [x * 2 for x in batch]

        results = processor.process_batches(data, operation)

        # Should recover and process items individually
        assert len(results) == 4

    def test_parallel_compute_sum(self, device_manager: DeviceManager) -> None:
        """Test parallel computation of sum"""
        processor = BatchProcessor(device_manager)

        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        ]

        results = processor.parallel_compute(arrays, "sum")

        assert len(results) == 3
        assert results[0] == 6
        assert results[1] == 15
        assert results[2] == 24

    def test_parallel_compute_mean(self, device_manager: DeviceManager) -> None:
        """Test parallel computation of mean"""
        processor = BatchProcessor(device_manager)

        arrays = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ]

        results = processor.parallel_compute(arrays, "mean")

        assert len(results) == 2
        assert abs(results[0] - 2.0) < 0.01
        assert abs(results[1] - 5.0) < 0.01

    def test_parallel_compute_std(self, device_manager: DeviceManager) -> None:
        """Test parallel computation of standard deviation"""
        processor = BatchProcessor(device_manager)

        arrays = [
            np.array([1, 2, 3], dtype=np.float32),
        ]

        results = processor.parallel_compute(arrays, "std")

        assert len(results) == 1
        assert results[0] > 0

    def test_parallel_compute_invalid_operation(
        self, device_manager: DeviceManager
    ) -> None:
        """Test parallel compute with invalid operation"""
        processor = BatchProcessor(device_manager)

        arrays = [np.array([1, 2, 3])]

        with pytest.raises(ValueError, match="Unknown operation"):
            processor.parallel_compute(arrays, "invalid")


class TestTensorOperationsIntegration:
    """Integration tests for tensor operations"""

    def test_end_to_end_matrix_workflow(self, tensor_ops: TensorOperations) -> None:
        """Test end-to-end matrix computation workflow"""
        # Create matrices
        a = np.random.randn(10, 10).astype(np.float32)
        b = np.random.randn(10, 10).astype(np.float32)

        # Multiply
        result = tensor_ops.matrix_multiply(a, b)

        # Normalize
        normalized = tensor_ops.normalize(result)

        # Check result
        assert normalized.shape == (10, 10)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10), decimal=5)

    def test_large_batch_processing(self, device_manager: DeviceManager) -> None:
        """Test processing large batches"""
        processor = BatchProcessor(device_manager, batch_size=100)

        # Create large dataset
        data = list(range(1000))

        def operation(batch: list[int]) -> list[int]:
            return [x * 2 for x in batch]

        results = processor.process_batches(data, operation)

        assert len(results) == 1000
        assert results[0] == 0
        assert results[999] == 1998
