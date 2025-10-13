"""State serialization and compression service."""

import gzip
import hashlib
import json
import zlib
from typing import Any
from uuid import uuid4

import structlog

from ..models.state_persistence import (
    AgentStateSnapshot,
    CompressionType,
    StateVersion,
)

logger = structlog.get_logger()

# Current state schema version
CURRENT_STATE_VERSION = StateVersion(major=1, minor=0, patch=0)


class StateSerializationError(Exception):
    """Raised when state serialization fails."""


class StateDeserializationError(Exception):
    """Raised when state deserialization fails."""


class StateSerializer:
    """Handles agent state serialization and compression."""

    def __init__(self, compression: CompressionType = CompressionType.GZIP) -> None:
        """
        Initialize state serializer.

        Args:
            compression: Default compression type to use
        """
        self._compression = compression

    def serialize_state(
        self,
        agent_id: str,
        state_data: dict[str, Any],
        compression: CompressionType | None = None,
    ) -> tuple[bytes, AgentStateSnapshot]:
        """
        Serialize agent state to bytes with compression.

        Args:
            agent_id: Agent identifier
            state_data: State data dictionary
            compression: Compression type (or use default)

        Returns:
            Tuple of (compressed bytes, snapshot metadata)

        Raises:
            StateSerializationError: If serialization fails
        """
        compression = compression or self._compression

        try:
            # Create snapshot metadata
            snapshot = AgentStateSnapshot(
                agent_id=agent_id,
                snapshot_id=str(uuid4()),
                version=CURRENT_STATE_VERSION,
                compression=compression,
                **state_data,
            )

            # Serialize to JSON
            json_data = snapshot.model_dump_json(indent=None)
            json_bytes = json_data.encode("utf-8")
            snapshot.uncompressed_size = len(json_bytes)

            # Compress data
            compressed_bytes = self._compress(json_bytes, compression)
            snapshot.compressed_size = len(compressed_bytes)

            logger.info(
                "state_serialized",
                agent_id=agent_id,
                snapshot_id=snapshot.snapshot_id,
                uncompressed_size=snapshot.uncompressed_size,
                compressed_size=snapshot.compressed_size,
                compression_ratio=round(
                    snapshot.compressed_size / snapshot.uncompressed_size, 2
                ),
            )

            return compressed_bytes, snapshot

        except Exception as e:
            logger.error(
                "state_serialization_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise StateSerializationError(
                f"Failed to serialize state for agent {agent_id}: {e}"
            ) from e

    def deserialize_state(
        self,
        compressed_data: bytes,
        compression: CompressionType,
    ) -> AgentStateSnapshot:
        """
        Deserialize agent state from compressed bytes.

        Args:
            compressed_data: Compressed state data
            compression: Compression type used

        Returns:
            Deserialized agent state snapshot

        Raises:
            StateDeserializationError: If deserialization fails
        """
        try:
            # Decompress data
            json_bytes = self._decompress(compressed_data, compression)

            # Parse JSON to snapshot
            json_data = json_bytes.decode("utf-8")
            snapshot = AgentStateSnapshot.model_validate_json(json_data)

            logger.info(
                "state_deserialized",
                agent_id=snapshot.agent_id,
                snapshot_id=snapshot.snapshot_id,
                version=str(snapshot.version),
            )

            return snapshot

        except Exception as e:
            logger.error(
                "state_deserialization_failed",
                error=str(e),
            )
            raise StateDeserializationError(
                f"Failed to deserialize state: {e}"
            ) from e

    def calculate_checksum(self, data: bytes) -> str:
        """
        Calculate SHA256 checksum of data.

        Args:
            data: Data to checksum

        Returns:
            Hex-encoded SHA256 checksum
        """
        return hashlib.sha256(data).hexdigest()

    def verify_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """
        Verify data checksum.

        Args:
            data: Data to verify
            expected_checksum: Expected checksum

        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_checksum(data)
        return actual_checksum == expected_checksum

    def _compress(self, data: bytes, compression: CompressionType) -> bytes:
        """
        Compress data using specified compression type.

        Args:
            data: Data to compress
            compression: Compression type

        Returns:
            Compressed data
        """
        if compression == CompressionType.NONE:
            return data
        if compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=6)
        if compression == CompressionType.ZLIB:
            return zlib.compress(data, level=6)
        if compression == CompressionType.LZ4:
            # LZ4 requires external library, fallback to gzip for now
            logger.warning(
                "lz4_not_available",
                message="LZ4 compression not available, using gzip",
            )
            return gzip.compress(data, compresslevel=6)
        raise ValueError(f"Unsupported compression type: {compression}")

    def _decompress(self, data: bytes, compression: CompressionType) -> bytes:
        """
        Decompress data using specified compression type.

        Args:
            data: Compressed data
            compression: Compression type

        Returns:
            Decompressed data
        """
        if compression == CompressionType.NONE:
            return data
        if compression == CompressionType.GZIP:
            return gzip.decompress(data)
        if compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        if compression == CompressionType.LZ4:
            # LZ4 requires external library, fallback to gzip for now
            logger.warning(
                "lz4_not_available",
                message="LZ4 decompression not available, using gzip",
            )
            return gzip.decompress(data)
        raise ValueError(f"Unsupported compression type: {compression}")


# Global serializer instance
_serializer: StateSerializer | None = None


def get_serializer() -> StateSerializer:
    """Get global state serializer instance."""
    global _serializer
    if _serializer is None:
        _serializer = StateSerializer()
    return _serializer
