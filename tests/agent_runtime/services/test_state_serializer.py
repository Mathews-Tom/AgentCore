"""Tests for StateSerializer service."""

from __future__ import annotations

import gzip
import zlib
from datetime import datetime, timezone
from typing import Any

import pytest

from agentcore.agent_runtime.models.state_persistence import (
    AgentStateSnapshot,
    CompressionType,
    StateVersion,
)
from agentcore.agent_runtime.services.state_serializer import (
    CURRENT_STATE_VERSION,
    StateDeserializationError,
    StateSerializationError,
    StateSerializer,
    get_serializer,
)


@pytest.fixture
def serializer() -> StateSerializer:
    """Create a StateSerializer instance."""
    return StateSerializer(compression=CompressionType.GZIP)


@pytest.fixture
def sample_state_data() -> dict[str, Any]:
    """Create sample agent state data."""
    return {
        "status": "running",
        "container_id": "container-123",
        "current_step": "reasoning",
        "execution_context": {
            "task_id": "task-001",
            "iteration": 5,
            "mode": "interactive",
        },
        "reasoning_chain": [
            {"step": 1, "thought": "Analyze problem", "confidence": 0.9},
            {"step": 2, "thought": "Identify solution", "confidence": 0.85},
        ],
        "decision_history": [
            {"timestamp": "2025-01-01T00:00:00Z", "action": "start", "result": "success"},
        ],
        "tool_usage_log": [
            {"tool": "calculator", "input": "2+2", "output": "4"},
        ],
        "performance_metrics": {
            "cpu_percent": 45.2,
            "memory_mb": 512.0,
            "throughput": 100.5,
        },
        "execution_time": 123.45,
        "working_memory": {
            "current_goal": "solve problem",
            "active_variables": {"x": 10, "y": 20},
        },
        "long_term_memory": {
            "learned_patterns": ["pattern1", "pattern2"],
            "knowledge_base": {"domain": "mathematics"},
        },
        "philosophy": "react",
        "tags": ["experiment", "production"],
        "metadata": {"version": "1.0", "priority": "high"},
    }


class TestStateSerializer:
    """Test suite for StateSerializer."""

    def test_init_default_compression(self) -> None:
        """Test initializing serializer with default compression."""
        serializer = StateSerializer()
        assert serializer._compression == CompressionType.GZIP

    def test_init_custom_compression(self) -> None:
        """Test initializing serializer with custom compression."""
        serializer = StateSerializer(compression=CompressionType.ZLIB)
        assert serializer._compression == CompressionType.ZLIB

    def test_serialize_state_gzip(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test serializing state with GZIP compression."""
        agent_id = "agent-001"

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.GZIP,
        )

        # Verify compressed bytes
        assert isinstance(compressed_bytes, bytes)
        assert len(compressed_bytes) > 0

        # Verify snapshot metadata
        assert snapshot.agent_id == agent_id
        assert snapshot.snapshot_id is not None
        assert snapshot.version == CURRENT_STATE_VERSION
        assert snapshot.compression == CompressionType.GZIP
        assert snapshot.status == "running"
        assert snapshot.container_id == "container-123"
        assert snapshot.compressed_size is not None
        assert snapshot.uncompressed_size is not None
        assert snapshot.compressed_size < snapshot.uncompressed_size  # GZIP should compress

    def test_serialize_state_zlib(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test serializing state with ZLIB compression."""
        agent_id = "agent-002"

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.ZLIB,
        )

        # Verify compression type
        assert snapshot.compression == CompressionType.ZLIB
        assert len(compressed_bytes) > 0
        assert snapshot.compressed_size < snapshot.uncompressed_size

    def test_serialize_state_no_compression(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test serializing state without compression."""
        agent_id = "agent-003"

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.NONE,
        )

        # Verify no compression
        assert snapshot.compression == CompressionType.NONE
        assert snapshot.compressed_size == snapshot.uncompressed_size

    def test_serialize_state_lz4_fallback(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test serializing state with LZ4 (should fallback to GZIP)."""
        agent_id = "agent-004"

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.LZ4,
        )

        # LZ4 should fallback to GZIP
        assert snapshot.compression == CompressionType.LZ4
        assert len(compressed_bytes) > 0

        # Data should be GZIP compressed (can decompress with gzip)
        json_bytes = gzip.decompress(compressed_bytes)
        assert len(json_bytes) > 0

    def test_serialize_state_default_compression(
        self,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test serializing state uses default compression when not specified."""
        serializer = StateSerializer(compression=CompressionType.ZLIB)
        agent_id = "agent-005"

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=None,  # Use default
        )

        # Should use ZLIB from default
        assert snapshot.compression == CompressionType.ZLIB

    def test_serialize_state_empty_data(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test serializing empty state data."""
        agent_id = "agent-006"
        state_data = {
            "status": "created",
            "philosophy": "react",
        }

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=state_data,
        )

        # Should still work with minimal data
        assert len(compressed_bytes) > 0
        assert snapshot.agent_id == agent_id
        assert snapshot.status == "created"

    def test_deserialize_state_gzip(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deserializing state with GZIP compression."""
        agent_id = "agent-007"

        # Serialize first
        compressed_bytes, original_snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.GZIP,
        )

        # Deserialize
        deserialized_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=CompressionType.GZIP,
        )

        # Verify deserialized data matches original
        assert deserialized_snapshot.agent_id == agent_id
        assert deserialized_snapshot.snapshot_id == original_snapshot.snapshot_id
        assert deserialized_snapshot.status == "running"
        assert deserialized_snapshot.container_id == "container-123"
        assert deserialized_snapshot.execution_context == sample_state_data["execution_context"]
        assert deserialized_snapshot.reasoning_chain == sample_state_data["reasoning_chain"]
        assert deserialized_snapshot.performance_metrics == sample_state_data["performance_metrics"]

    def test_deserialize_state_zlib(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deserializing state with ZLIB compression."""
        agent_id = "agent-008"

        # Serialize first
        compressed_bytes, _ = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.ZLIB,
        )

        # Deserialize
        deserialized_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=CompressionType.ZLIB,
        )

        assert deserialized_snapshot.agent_id == agent_id
        assert deserialized_snapshot.status == "running"

    def test_deserialize_state_no_compression(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deserializing state without compression."""
        agent_id = "agent-009"

        # Serialize first
        compressed_bytes, _ = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.NONE,
        )

        # Deserialize
        deserialized_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=CompressionType.NONE,
        )

        assert deserialized_snapshot.agent_id == agent_id

    def test_deserialize_state_lz4_fallback(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deserializing state with LZ4 (should fallback to GZIP)."""
        agent_id = "agent-010"

        # Serialize first
        compressed_bytes, _ = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.LZ4,
        )

        # Deserialize with LZ4 (fallback to GZIP)
        deserialized_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=CompressionType.LZ4,
        )

        assert deserialized_snapshot.agent_id == agent_id

    def test_serialize_deserialize_roundtrip(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test complete serialize-deserialize roundtrip preserves data."""
        agent_id = "agent-011"

        # Serialize
        compressed_bytes, original_snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Deserialize
        restored_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=original_snapshot.compression,
        )

        # Verify all fields preserved
        assert restored_snapshot.agent_id == original_snapshot.agent_id
        assert restored_snapshot.snapshot_id == original_snapshot.snapshot_id
        assert restored_snapshot.status == original_snapshot.status
        assert restored_snapshot.container_id == original_snapshot.container_id
        assert restored_snapshot.current_step == original_snapshot.current_step
        assert restored_snapshot.execution_context == original_snapshot.execution_context
        assert restored_snapshot.reasoning_chain == original_snapshot.reasoning_chain
        assert restored_snapshot.decision_history == original_snapshot.decision_history
        assert restored_snapshot.tool_usage_log == original_snapshot.tool_usage_log
        assert restored_snapshot.performance_metrics == original_snapshot.performance_metrics
        assert restored_snapshot.execution_time == original_snapshot.execution_time
        assert restored_snapshot.working_memory == original_snapshot.working_memory
        assert restored_snapshot.long_term_memory == original_snapshot.long_term_memory
        assert restored_snapshot.philosophy == original_snapshot.philosophy
        assert restored_snapshot.tags == original_snapshot.tags
        assert restored_snapshot.metadata == original_snapshot.metadata

    def test_calculate_checksum(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test calculating SHA256 checksum."""
        data = b"test data for checksum"

        checksum = serializer.calculate_checksum(data)

        # Verify checksum format
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest is 64 characters
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_calculate_checksum_deterministic(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test checksum calculation is deterministic."""
        data = b"deterministic test data"

        checksum1 = serializer.calculate_checksum(data)
        checksum2 = serializer.calculate_checksum(data)

        assert checksum1 == checksum2

    def test_calculate_checksum_different_data(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test different data produces different checksums."""
        data1 = b"data one"
        data2 = b"data two"

        checksum1 = serializer.calculate_checksum(data1)
        checksum2 = serializer.calculate_checksum(data2)

        assert checksum1 != checksum2

    def test_verify_checksum_valid(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test verifying valid checksum."""
        data = b"test data for verification"
        checksum = serializer.calculate_checksum(data)

        is_valid = serializer.verify_checksum(data, checksum)

        assert is_valid is True

    def test_verify_checksum_invalid(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test verifying invalid checksum."""
        data = b"test data"
        wrong_checksum = "0" * 64  # Invalid checksum

        is_valid = serializer.verify_checksum(data, wrong_checksum)

        assert is_valid is False

    def test_verify_checksum_corrupted_data(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test verifying checksum with corrupted data."""
        original_data = b"original data"
        checksum = serializer.calculate_checksum(original_data)

        corrupted_data = b"corrupted data"
        is_valid = serializer.verify_checksum(corrupted_data, checksum)

        assert is_valid is False

    def test_serialize_state_error_handling(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test serialization error handling with invalid data."""
        agent_id = "agent-error"

        # Create state data with invalid field type that will cause validation error
        # AgentStateSnapshot requires 'status' to be a string, but we can't pass
        # invalid types through state_data. Instead, we'll create a circular reference
        # that will cause JSON serialization to fail.
        circular_dict: dict[str, Any] = {"status": "running", "philosophy": "react"}
        circular_dict["self_reference"] = circular_dict

        state_data = {
            "status": "running",
            "philosophy": "react",
            "execution_context": circular_dict,
        }

        with pytest.raises(StateSerializationError) as exc_info:
            serializer.serialize_state(
                agent_id=agent_id,
                state_data=state_data,
            )

        assert "Failed to serialize state" in str(exc_info.value)
        assert agent_id in str(exc_info.value)

    def test_deserialize_state_error_invalid_data(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test deserialization error with invalid compressed data."""
        invalid_data = b"this is not valid compressed data"

        with pytest.raises(StateDeserializationError) as exc_info:
            serializer.deserialize_state(
                compressed_data=invalid_data,
                compression=CompressionType.GZIP,
            )

        assert "Failed to deserialize state" in str(exc_info.value)

    def test_deserialize_state_error_corrupted_json(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test deserialization error with corrupted JSON data."""
        # Create valid GZIP data but with invalid JSON
        invalid_json = b"{invalid json}"
        compressed_data = gzip.compress(invalid_json)

        with pytest.raises(StateDeserializationError) as exc_info:
            serializer.deserialize_state(
                compressed_data=compressed_data,
                compression=CompressionType.GZIP,
            )

        assert "Failed to deserialize state" in str(exc_info.value)

    def test_compress_unsupported_type(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test compression with unsupported type raises error."""
        data = b"test data"

        with pytest.raises(ValueError) as exc_info:
            serializer._compress(data, "invalid_compression")  # type: ignore

        assert "Unsupported compression type" in str(exc_info.value)

    def test_decompress_unsupported_type(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test decompression with unsupported type raises error."""
        data = b"test data"

        with pytest.raises(ValueError) as exc_info:
            serializer._decompress(data, "invalid_compression")  # type: ignore

        assert "Unsupported compression type" in str(exc_info.value)

    def test_serialize_large_state(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test serializing large state data."""
        agent_id = "agent-large"

        # Create large state with many items
        large_state_data = {
            "status": "running",
            "philosophy": "react",
            "reasoning_chain": [
                {"step": i, "thought": f"thought_{i}" * 100, "confidence": 0.9}
                for i in range(1000)
            ],
            "tool_usage_log": [
                {"tool": f"tool_{i}", "input": f"input_{i}" * 50, "output": f"output_{i}" * 50}
                for i in range(500)
            ],
            "performance_metrics": {f"metric_{i}": float(i) for i in range(100)},
        }

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=large_state_data,
            compression=CompressionType.GZIP,
        )

        # Verify large data handled correctly
        assert len(compressed_bytes) > 0
        assert snapshot.compressed_size < snapshot.uncompressed_size
        assert snapshot.uncompressed_size > 100000  # Should be large

        # Verify compression ratio is reasonable
        compression_ratio = snapshot.compressed_size / snapshot.uncompressed_size
        assert compression_ratio < 0.5  # GZIP should compress significantly

    def test_serialize_nested_complex_data(
        self,
        serializer: StateSerializer,
    ) -> None:
        """Test serializing complex nested data structures."""
        agent_id = "agent-complex"

        complex_state_data = {
            "status": "running",
            "philosophy": "react",
            "execution_context": {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "deep_value": "nested_data",
                                "deep_list": [1, 2, 3, [4, 5, [6, 7]]],
                            }
                        }
                    }
                }
            },
            "reasoning_chain": [
                {
                    "step": 1,
                    "analysis": {
                        "factors": ["factor1", "factor2"],
                        "weights": {"factor1": 0.7, "factor2": 0.3},
                        "sub_analysis": {
                            "details": ["detail1", "detail2"],
                        },
                    },
                }
            ],
        }

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id=agent_id,
            state_data=complex_state_data,
        )

        # Deserialize and verify structure preserved
        restored_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=snapshot.compression,
        )

        assert restored_snapshot.execution_context == complex_state_data["execution_context"]
        assert restored_snapshot.reasoning_chain == complex_state_data["reasoning_chain"]

    def test_compression_ratios(
        self,
        serializer: StateSerializer,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test and compare compression ratios for different compression types."""
        agent_id = "agent-compression-test"

        # Test all compression types
        results = {}
        for compression_type in [
            CompressionType.NONE,
            CompressionType.GZIP,
            CompressionType.ZLIB,
        ]:
            compressed_bytes, snapshot = serializer.serialize_state(
                agent_id=agent_id,
                state_data=sample_state_data,
                compression=compression_type,
            )
            results[compression_type] = {
                "uncompressed": snapshot.uncompressed_size,
                "compressed": snapshot.compressed_size,
                "ratio": snapshot.compressed_size / snapshot.uncompressed_size,
            }

        # Verify NONE has no compression
        assert results[CompressionType.NONE]["ratio"] == 1.0

        # Verify GZIP and ZLIB achieve compression
        assert results[CompressionType.GZIP]["ratio"] < 1.0
        assert results[CompressionType.ZLIB]["ratio"] < 1.0

        # All should have same uncompressed size
        assert (
            results[CompressionType.NONE]["uncompressed"]
            == results[CompressionType.GZIP]["uncompressed"]
            == results[CompressionType.ZLIB]["uncompressed"]
        )


class TestGlobalSerializer:
    """Test global serializer instance."""

    def test_get_serializer_singleton(self) -> None:
        """Test get_serializer returns singleton instance."""
        serializer1 = get_serializer()
        serializer2 = get_serializer()

        assert serializer1 is serializer2

    def test_get_serializer_default_compression(self) -> None:
        """Test get_serializer uses default GZIP compression."""
        serializer = get_serializer()

        assert serializer._compression == CompressionType.GZIP


class TestStateVersion:
    """Test StateVersion model functionality."""

    def test_state_version_string_representation(self) -> None:
        """Test StateVersion string representation."""
        version = StateVersion(major=1, minor=2, patch=3)

        assert str(version) == "1.2.3"

    def test_state_version_parse(self) -> None:
        """Test StateVersion parse method."""
        version = StateVersion.parse("2.5.10")

        assert version.major == 2
        assert version.minor == 5
        assert version.patch == 10

    def test_state_version_parse_invalid(self) -> None:
        """Test StateVersion parse with invalid string."""
        with pytest.raises(ValueError) as exc_info:
            StateVersion.parse("invalid.version")

        assert "Invalid version string" in str(exc_info.value)

    def test_state_version_is_compatible_same_major(self) -> None:
        """Test StateVersion compatibility with same major version."""
        version1 = StateVersion(major=1, minor=0, patch=0)
        version2 = StateVersion(major=1, minor=5, patch=10)

        assert version1.is_compatible(version2) is True
        assert version2.is_compatible(version1) is True

    def test_state_version_is_compatible_different_major(self) -> None:
        """Test StateVersion incompatibility with different major version."""
        version1 = StateVersion(major=1, minor=0, patch=0)
        version2 = StateVersion(major=2, minor=0, patch=0)

        assert version1.is_compatible(version2) is False
        assert version2.is_compatible(version1) is False

    def test_current_state_version(self) -> None:
        """Test CURRENT_STATE_VERSION is properly defined."""
        assert CURRENT_STATE_VERSION.major == 1
        assert CURRENT_STATE_VERSION.minor == 0
        assert CURRENT_STATE_VERSION.patch == 0
