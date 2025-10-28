"""
GPU-accelerated tensor operations for optimization algorithms

Provides high-performance tensor operations on GPU with automatic batching
and graceful CPU fallback when GPU is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agentcore.dspy_optimization.gpu.device import DeviceManager, DeviceType

logger = logging.getLogger(__name__)


class TensorOperations:
    """
    GPU-accelerated tensor operations

    Provides common tensor operations with automatic GPU acceleration
    and CPU fallback. All operations handle both numpy arrays and PyTorch tensors.

    Key features:
    - Automatic device management
    - CPU fallback when GPU unavailable
    - Type conversion handling
    - Memory-efficient operations
    """

    def __init__(self, device_manager: DeviceManager) -> None:
        """
        Initialize tensor operations

        Args:
            device_manager: Device manager for GPU/CPU selection
        """
        self.device_manager = device_manager
        self._torch = None

        if device_manager.torch_available:
            import torch

            self._torch = torch

    def to_tensor(self, data: np.ndarray | Any) -> Any:
        """
        Convert numpy array to tensor on current device

        Args:
            data: Input data (numpy array or tensor)

        Returns:
            Tensor on current device
        """
        if not self._torch:
            # No PyTorch, return numpy array
            return np.asarray(data)

        # Convert to tensor
        if isinstance(data, np.ndarray):
            tensor = self._torch.from_numpy(data)
        else:
            tensor = self._torch.as_tensor(data)

        # Move to device
        device = self.device_manager.get_torch_device()
        return tensor.to(device)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert tensor to numpy array

        Args:
            tensor: Input tensor

        Returns:
            Numpy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor

        if not self._torch:
            return np.asarray(tensor)

        # Move to CPU first if on GPU
        if tensor.is_cuda or (hasattr(tensor, "is_mps") and tensor.is_mps):
            tensor = tensor.cpu()

        return tensor.numpy()

    def matrix_multiply(self, a: np.ndarray | Any, b: np.ndarray | Any) -> np.ndarray:
        """
        Perform matrix multiplication on GPU

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Result matrix as numpy array
        """
        if not self._torch or self.device_manager.current_device.device_type == DeviceType.CPU:
            # CPU fallback
            return np.matmul(np.asarray(a), np.asarray(b))

        # GPU computation
        a_tensor = self.to_tensor(a)
        b_tensor = self.to_tensor(b)
        result = self._torch.matmul(a_tensor, b_tensor)
        return self.to_numpy(result)

    def batch_matrix_multiply(
        self, matrices_a: list[np.ndarray], matrices_b: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Perform batched matrix multiplication on GPU

        Args:
            matrices_a: List of first matrices
            matrices_b: List of second matrices

        Returns:
            List of result matrices
        """
        if not self._torch or self.device_manager.current_device.device_type == DeviceType.CPU:
            # CPU fallback
            return [
                np.matmul(np.asarray(a), np.asarray(b))
                for a, b in zip(matrices_a, matrices_b)
            ]

        # GPU batched computation
        batch_a = self._torch.stack([self.to_tensor(a) for a in matrices_a])
        batch_b = self._torch.stack([self.to_tensor(b) for b in matrices_b])
        results = self._torch.bmm(batch_a, batch_b)

        return [self.to_numpy(result) for result in results]

    def compute_distances(
        self, points: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between points and centroids

        Args:
            points: Array of shape (n_points, n_features)
            centroids: Array of shape (n_centroids, n_features)

        Returns:
            Distance matrix of shape (n_points, n_centroids)
        """
        if not self._torch or self.device_manager.current_device.device_type == DeviceType.CPU:
            # CPU fallback using numpy
            points_arr = np.asarray(points)
            centroids_arr = np.asarray(centroids)

            # Compute using broadcasting
            diff = points_arr[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            return distances

        # GPU computation
        points_tensor = self.to_tensor(points)
        centroids_tensor = self.to_tensor(centroids)

        # Efficient distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        points_norm = (points_tensor ** 2).sum(dim=1, keepdim=True)
        centroids_norm = (centroids_tensor ** 2).sum(dim=1, keepdim=True)
        distances_sq = (
            points_norm
            + centroids_norm.t()
            - 2 * self._torch.mm(points_tensor, centroids_tensor.t())
        )

        # Clamp to avoid numerical issues with sqrt
        distances_sq = self._torch.clamp(distances_sq, min=0.0)
        distances = self._torch.sqrt(distances_sq)

        return self.to_numpy(distances)

    def normalize(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Normalize vectors to unit length

        Args:
            data: Input data
            axis: Axis along which to normalize

        Returns:
            Normalized data
        """
        if not self._torch or self.device_manager.current_device.device_type == DeviceType.CPU:
            # CPU fallback
            data_arr = np.asarray(data)
            norm = np.linalg.norm(data_arr, axis=axis, keepdims=True)
            return data_arr / (norm + 1e-8)

        # GPU computation
        tensor = self.to_tensor(data)
        norm = self._torch.norm(tensor, dim=axis, keepdim=True)
        normalized = tensor / (norm + 1e-8)
        return self.to_numpy(normalized)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between vectors

        Args:
            a: First set of vectors (n_samples_a, n_features)
            b: Second set of vectors (n_samples_b, n_features)

        Returns:
            Similarity matrix (n_samples_a, n_samples_b)
        """
        if not self._torch or self.device_manager.current_device.device_type == DeviceType.CPU:
            # CPU fallback
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)

            a_norm = a_arr / (np.linalg.norm(a_arr, axis=1, keepdims=True) + 1e-8)
            b_norm = b_arr / (np.linalg.norm(b_arr, axis=1, keepdims=True) + 1e-8)

            return np.dot(a_norm, b_norm.T)

        # GPU computation
        a_tensor = self.to_tensor(a)
        b_tensor = self.to_tensor(b)

        a_norm = self._torch.nn.functional.normalize(a_tensor, p=2, dim=1)
        b_norm = self._torch.nn.functional.normalize(b_tensor, p=2, dim=1)

        similarity = self._torch.mm(a_norm, b_norm.t())
        return self.to_numpy(similarity)


class BatchProcessor:
    """
    Efficient batch processing on GPU

    Handles automatic batching of operations for optimal GPU utilization
    with memory-aware batch sizing.

    Key features:
    - Automatic batch size optimization
    - Memory-aware processing
    - Progress tracking
    - Error handling and recovery
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        batch_size: int | None = None,
        max_memory_usage: float = 0.8,
    ) -> None:
        """
        Initialize batch processor

        Args:
            device_manager: Device manager
            batch_size: Fixed batch size (None for automatic)
            max_memory_usage: Maximum memory usage fraction (0.0-1.0)
        """
        self.device_manager = device_manager
        self.tensor_ops = TensorOperations(device_manager)
        self.batch_size = batch_size
        self.max_memory_usage = max_memory_usage

        # Determine optimal batch size if not specified
        if self.batch_size is None:
            self.batch_size = self._determine_batch_size()

    def _determine_batch_size(self) -> int:
        """
        Determine optimal batch size based on available memory

        Returns:
            Optimal batch size
        """
        device = self.device_manager.current_device

        if device.device_type == DeviceType.CPU:
            # Conservative batch size for CPU
            return 32

        # GPU batch size based on available memory
        available_memory = device.available_memory
        memory_per_batch = 1024 * 1024  # 1MB per sample estimate

        optimal_batch = int(
            (available_memory * self.max_memory_usage) / memory_per_batch
        )

        # Clamp to reasonable range
        return max(16, min(optimal_batch, 512))

    def process_batches(
        self,
        data: list[Any],
        operation: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Process data in batches

        Args:
            data: List of data items to process
            operation: Callable operation to apply
            **kwargs: Additional arguments for operation

        Returns:
            List of results
        """
        results: list[Any] = []

        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]

            try:
                # Process batch
                batch_results = operation(batch, **kwargs)
                results.extend(batch_results)

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Try processing items individually on error
                for item in batch:
                    try:
                        result = operation([item], **kwargs)
                        results.extend(result)
                    except Exception as item_error:
                        logger.error(f"Error processing item: {item_error}")
                        results.append(None)

            # Clear cache periodically
            if i % (self.batch_size * 10) == 0:
                self.device_manager.clear_cache()

        return results

    def parallel_compute(
        self,
        arrays: list[np.ndarray],
        operation: str,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Perform parallel computation on arrays

        Args:
            arrays: List of arrays to process
            operation: Operation name ('sum', 'mean', 'std', etc.)
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if not self.tensor_ops._torch:
            # CPU fallback - process sequentially
            results = []
            for arr in arrays:
                if operation == "sum":
                    results.append(np.sum(arr, **kwargs))
                elif operation == "mean":
                    results.append(np.mean(arr, **kwargs))
                elif operation == "std":
                    results.append(np.std(arr, **kwargs))
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            return results

        # GPU batch processing
        tensors = [self.tensor_ops.to_tensor(arr) for arr in arrays]

        results = []
        for tensor in tensors:
            if operation == "sum":
                result = self.tensor_ops._torch.sum(tensor, **kwargs)
            elif operation == "mean":
                result = self.tensor_ops._torch.mean(tensor, **kwargs)
            elif operation == "std":
                result = self.tensor_ops._torch.std(tensor, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            results.append(self.tensor_ops.to_numpy(result))

        return results
