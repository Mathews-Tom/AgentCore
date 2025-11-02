"""Tests for StatePersistenceService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.agent_runtime.models.state_persistence import (
    CompressionType,
    StateRestoreRequest,
    StateVersion,
)
from agentcore.agent_runtime.services.state_persistence import (
    StateNotFoundError,
    StatePersistenceError,
    StatePersistenceService,
)
from agentcore.agent_runtime.services.state_serializer import (
    StateDeserializationError,
    StateSerializationError,
)


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Create temporary storage directory."""
    storage_dir = tmp_path / "state_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def persistence_service(tmp_storage: Path) -> StatePersistenceService:
    """Create StatePersistenceService instance with temp storage."""
    return StatePersistenceService(
        storage_path=tmp_storage,
        compression=CompressionType.GZIP,
        retention_days=30,
    )


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
        },
        "reasoning_chain": [
            {"step": 1, "thought": "Analyze problem"},
            {"step": 2, "thought": "Identify solution"},
        ],
        "decision_history": [
            {"action": "start", "result": "success"},
        ],
        "tool_usage_log": [
            {"tool": "calculator", "input": "2+2", "output": "4"},
        ],
        "performance_metrics": {
            "cpu_percent": 45.2,
            "memory_mb": 512.0,
        },
        "execution_time": 123.45,
        "working_memory": {
            "current_goal": "solve problem",
        },
        "long_term_memory": {
            "learned_patterns": ["pattern1"],
        },
        "philosophy": "react",
        "metadata": {"version": "1.0"},
    }


@pytest.mark.asyncio
class TestStatePersistenceServiceInit:
    """Test suite for StatePersistenceService initialization."""

    async def test_init_creates_storage_directory(
        self,
        tmp_storage: Path,
    ) -> None:
        """Test initialization creates storage directory."""
        # Remove directory to test creation
        storage_dir = tmp_storage / "new_storage"
        assert not storage_dir.exists()

        service = StatePersistenceService(
            storage_path=storage_dir,
            compression=CompressionType.GZIP,
            retention_days=30,
        )

        # Verify directory was created
        assert storage_dir.exists()
        assert storage_dir.is_dir()
        assert service._storage_path == storage_dir

    async def test_init_with_string_path(
        self,
        tmp_storage: Path,
    ) -> None:
        """Test initialization with string path."""
        storage_dir = tmp_storage / "string_path"
        service = StatePersistenceService(
            storage_path=str(storage_dir),
            compression=CompressionType.ZLIB,
            retention_days=15,
        )

        assert service._storage_path == storage_dir
        assert service._compression == CompressionType.ZLIB
        assert service._retention_days == 15

    async def test_init_with_existing_directory(
        self,
        tmp_storage: Path,
    ) -> None:
        """Test initialization with existing directory."""
        # Directory already exists from fixture
        service = StatePersistenceService(
            storage_path=tmp_storage,
        )

        # Should work without errors
        assert service._storage_path == tmp_storage

    async def test_init_default_values(
        self,
        tmp_storage: Path,
    ) -> None:
        """Test initialization with default values."""
        service = StatePersistenceService(storage_path=tmp_storage)

        assert service._compression == CompressionType.GZIP
        assert service._retention_days == 30
        assert service._state_cache == {}
        assert service._backup_metadata == {}


@pytest.mark.asyncio
class TestStatePersistenceServiceSave:
    """Test suite for save_state method."""

    async def test_save_state_success(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test successful state save."""
        agent_id = "agent-001"

        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Verify snapshot metadata
        assert snapshot.agent_id == agent_id
        assert snapshot.snapshot_id is not None
        assert snapshot.status == "running"
        assert snapshot.compressed_size is not None
        assert snapshot.uncompressed_size is not None

        # Verify file was created
        storage_file = persistence_service._get_snapshot_path(
            agent_id, snapshot.snapshot_id
        )
        assert storage_file.exists()

        # Verify cache was updated
        assert agent_id in persistence_service._state_cache
        assert snapshot.snapshot_id in persistence_service._backup_metadata

    async def test_save_state_with_tags(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test saving state with tags."""
        agent_id = "agent-002"
        tags = ["production", "experiment"]

        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            tags=tags,
        )

        # Verify tags were added
        assert snapshot.tags == tags

        # Verify backup metadata has tags
        metadata = persistence_service._backup_metadata[snapshot.snapshot_id]
        assert metadata.tags == tags

    async def test_save_state_custom_compression(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test saving state with custom compression."""
        agent_id = "agent-003"

        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=CompressionType.ZLIB,
        )

        # Verify compression type
        assert snapshot.compression == CompressionType.ZLIB

    async def test_save_state_default_compression(
        self,
        tmp_storage: Path,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test saving state uses default compression."""
        service = StatePersistenceService(
            storage_path=tmp_storage,
            compression=CompressionType.NONE,
        )
        agent_id = "agent-004"

        snapshot = await service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
            compression=None,  # Use default
        )

        # Should use NONE from default
        assert snapshot.compression == CompressionType.NONE

    async def test_save_state_creates_agent_directory(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test saving state creates agent-specific directory."""
        agent_id = "agent-005"
        agent_dir = persistence_service._storage_path / agent_id

        assert not agent_dir.exists()

        await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Verify agent directory was created
        assert agent_dir.exists()
        assert agent_dir.is_dir()

    async def test_save_state_backup_metadata_complete(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test backup metadata is complete after save."""
        agent_id = "agent-006"

        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        metadata = persistence_service._backup_metadata[snapshot.snapshot_id]

        # Verify all metadata fields
        assert metadata.backup_id == snapshot.snapshot_id
        assert metadata.agent_id == agent_id
        assert metadata.snapshot_id == snapshot.snapshot_id
        assert metadata.version == snapshot.version
        assert metadata.backup_type == "full"
        assert metadata.compression == snapshot.compression
        assert metadata.size_bytes > 0
        assert metadata.checksum is not None
        assert len(metadata.checksum) == 64  # SHA256 hex
        assert metadata.retention_days == 30
        assert metadata.expires_at is not None

    async def test_save_state_serialization_error(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test save_state handles serialization errors."""
        agent_id = "agent-error"

        # Create circular reference to cause serialization error
        circular_dict: dict[str, Any] = {"status": "running", "philosophy": "react"}
        circular_dict["self_reference"] = circular_dict

        state_data = {
            "status": "running",
            "philosophy": "react",
            "execution_context": circular_dict,
        }

        with pytest.raises(StatePersistenceError) as exc_info:
            await persistence_service.save_state(
                agent_id=agent_id,
                state_data=state_data,
            )

        assert "Failed to save state" in str(exc_info.value)
        assert agent_id in str(exc_info.value)

    async def test_save_state_oserror(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test save_state handles OS errors."""
        agent_id = "agent-oserror"

        # Make storage path read-only to cause write error
        persistence_service._storage_path.chmod(0o444)

        try:
            with pytest.raises(StatePersistenceError) as exc_info:
                await persistence_service.save_state(
                    agent_id=agent_id,
                    state_data=sample_state_data,
                )

            assert "Failed to save state" in str(exc_info.value)
        finally:
            # Restore permissions
            persistence_service._storage_path.chmod(0o755)


@pytest.mark.asyncio
class TestStatePersistenceServiceLoad:
    """Test suite for load_state method."""

    async def test_load_state_latest_snapshot(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test loading latest snapshot."""
        agent_id = "agent-007"

        # Save multiple snapshots
        snapshot1 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )
        await asyncio.sleep(0.01)  # Ensure different modification times

        sample_state_data["status"] = "paused"
        snapshot2 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Clear cache to force load from storage
        persistence_service._state_cache.clear()

        # Load latest (should be snapshot2)
        loaded_snapshot = await persistence_service.load_state(agent_id=agent_id)

        assert loaded_snapshot.snapshot_id == snapshot2.snapshot_id
        assert loaded_snapshot.status == "paused"

    async def test_load_state_specific_snapshot(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test loading specific snapshot."""
        agent_id = "agent-008"

        # Save multiple snapshots
        snapshot1 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        sample_state_data["status"] = "completed"
        snapshot2 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Load specific snapshot (snapshot1)
        loaded_snapshot = await persistence_service.load_state(
            agent_id=agent_id,
            snapshot_id=snapshot1.snapshot_id,
        )

        assert loaded_snapshot.snapshot_id == snapshot1.snapshot_id
        assert loaded_snapshot.status == "running"

    async def test_load_state_from_cache(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test loading state from cache."""
        agent_id = "agent-009"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Load without snapshot_id should use cache
        loaded_snapshot = await persistence_service.load_state(agent_id=agent_id)

        # Should return cached snapshot (same object reference)
        assert loaded_snapshot is persistence_service._state_cache[agent_id]
        assert loaded_snapshot.snapshot_id == snapshot.snapshot_id

    async def test_load_state_checksum_verification(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test load_state verifies checksums."""
        agent_id = "agent-010"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Clear cache
        persistence_service._state_cache.clear()

        # Corrupt checksum in metadata
        persistence_service._backup_metadata[snapshot.snapshot_id].checksum = "0" * 64

        # Load should fail checksum verification
        with pytest.raises(StatePersistenceError) as exc_info:
            await persistence_service.load_state(
                agent_id=agent_id,
                snapshot_id=snapshot.snapshot_id,
            )

        assert "Checksum mismatch" in str(exc_info.value)

    async def test_load_state_not_found_no_snapshots(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test load_state raises error when no snapshots exist."""
        agent_id = "agent-nonexistent"

        with pytest.raises(StateNotFoundError) as exc_info:
            await persistence_service.load_state(agent_id=agent_id)

        assert "No state found" in str(exc_info.value)
        assert agent_id in str(exc_info.value)

    async def test_load_state_not_found_specific_snapshot(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test load_state raises error when specific snapshot not found."""
        agent_id = "agent-011"

        # Save one snapshot
        await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Try to load non-existent snapshot
        with pytest.raises(StateNotFoundError) as exc_info:
            await persistence_service.load_state(
                agent_id=agent_id,
                snapshot_id="nonexistent-snapshot-id",
            )

        assert "not found" in str(exc_info.value)

    async def test_load_state_deserialization_error(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test load_state handles deserialization errors."""
        agent_id = "agent-012"

        # Save valid snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Corrupt the file
        storage_file = persistence_service._get_snapshot_path(
            agent_id, snapshot.snapshot_id
        )
        storage_file.write_bytes(b"corrupted data")

        # Clear metadata to skip checksum verification
        persistence_service._backup_metadata.clear()

        # Load should fail deserialization
        with pytest.raises(StatePersistenceError) as exc_info:
            await persistence_service.load_state(
                agent_id=agent_id,
                snapshot_id=snapshot.snapshot_id,
            )

        assert "Failed to load state" in str(exc_info.value)

    async def test_load_state_updates_cache(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test load_state updates cache."""
        agent_id = "agent-013"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Clear cache
        persistence_service._state_cache.clear()
        assert agent_id not in persistence_service._state_cache

        # Load snapshot
        loaded_snapshot = await persistence_service.load_state(
            agent_id=agent_id,
            snapshot_id=snapshot.snapshot_id,
        )

        # Verify cache was updated
        assert agent_id in persistence_service._state_cache
        assert persistence_service._state_cache[agent_id].snapshot_id == snapshot.snapshot_id


@pytest.mark.asyncio
class TestStatePersistenceServiceRestore:
    """Test suite for restore_state method."""

    async def test_restore_state_full_restore(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test full state restore."""
        agent_id = "agent-014"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Create restore request
        request = StateRestoreRequest(
            agent_id=agent_id,
            snapshot_id=snapshot.snapshot_id,
            restore_memory=True,
            restore_execution_context=True,
            restore_tools=True,
        )

        # Restore state
        result = await persistence_service.restore_state(request)

        # Verify result
        assert result.success is True
        assert result.agent_id == agent_id
        assert result.restored_snapshot_id == snapshot.snapshot_id
        assert result.error_message is None

    async def test_restore_state_selective_memory_only(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test selective restore (memory only)."""
        agent_id = "agent-015"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Create restore request (memory only)
        request = StateRestoreRequest(
            agent_id=agent_id,
            snapshot_id=snapshot.snapshot_id,
            restore_memory=True,
            restore_execution_context=False,
            restore_tools=False,
        )

        # Restore state
        result = await persistence_service.restore_state(request)

        assert result.success is True

    async def test_restore_state_with_target_status(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test restore with target status override."""
        agent_id = "agent-016"

        # Save snapshot with "running" status
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Create restore request with target status
        request = StateRestoreRequest(
            agent_id=agent_id,
            snapshot_id=snapshot.snapshot_id,
            target_status="paused",
        )

        # Restore state
        result = await persistence_service.restore_state(request)

        assert result.success is True

    async def test_restore_state_with_backup_id(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test restore using backup_id."""
        agent_id = "agent-017"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Create restore request with backup_id
        request = StateRestoreRequest(
            agent_id=agent_id,
            backup_id=snapshot.snapshot_id,
        )

        # Restore state
        result = await persistence_service.restore_state(request)

        assert result.success is True
        assert result.restored_from_backup == snapshot.snapshot_id

    async def test_restore_state_latest_snapshot(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test restore without specific snapshot (use latest)."""
        agent_id = "agent-018"

        # Save snapshots
        await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )
        await asyncio.sleep(0.01)

        sample_state_data["status"] = "completed"
        snapshot2 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Create restore request without snapshot_id or backup_id
        request = StateRestoreRequest(agent_id=agent_id)

        # Should restore latest
        persistence_service._state_cache.clear()
        result = await persistence_service.restore_state(request)

        assert result.success is True
        assert result.restored_snapshot_id == snapshot2.snapshot_id

    async def test_restore_state_error_handling(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test restore_state handles errors gracefully."""
        agent_id = "agent-nonexistent"

        # Create restore request for non-existent agent
        request = StateRestoreRequest(agent_id=agent_id)

        # Should return error result, not raise exception
        result = await persistence_service.restore_state(request)

        assert result.success is False
        assert result.error_message is not None
        assert "No state found" in result.error_message


@pytest.mark.asyncio
class TestStatePersistenceServiceSnapshotManagement:
    """Test suite for snapshot management methods."""

    async def test_list_snapshots_with_snapshots(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test listing snapshots for agent."""
        agent_id = "agent-020"

        # Save multiple snapshots
        snapshot1 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )
        await asyncio.sleep(0.01)

        snapshot2 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )
        await asyncio.sleep(0.01)

        snapshot3 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # List snapshots
        snapshots = await persistence_service.list_snapshots(agent_id)

        # Should return all 3 snapshots, newest first
        assert len(snapshots) == 3
        assert snapshots[0].snapshot_id == snapshot3.snapshot_id
        assert snapshots[1].snapshot_id == snapshot2.snapshot_id
        assert snapshots[2].snapshot_id == snapshot1.snapshot_id

    async def test_list_snapshots_with_limit(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test listing snapshots with limit."""
        agent_id = "agent-021"

        # Save 5 snapshots
        for i in range(5):
            await persistence_service.save_state(
                agent_id=agent_id,
                state_data=sample_state_data,
            )
            await asyncio.sleep(0.01)

        # List with limit of 3
        snapshots = await persistence_service.list_snapshots(agent_id, limit=3)

        # Should return only 3 most recent
        assert len(snapshots) == 3

    async def test_list_snapshots_empty_directory(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test listing snapshots for non-existent agent."""
        agent_id = "agent-nonexistent"

        snapshots = await persistence_service.list_snapshots(agent_id)

        # Should return empty list
        assert snapshots == []

    async def test_list_snapshots_no_metadata(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test listing snapshots when metadata is missing."""
        agent_id = "agent-022"

        # Save snapshot
        await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Clear metadata
        persistence_service._backup_metadata.clear()

        # List snapshots
        snapshots = await persistence_service.list_snapshots(agent_id)

        # Should return empty list when metadata missing
        assert snapshots == []

    async def test_delete_snapshot_success(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deleting snapshot."""
        agent_id = "agent-023"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Verify file exists
        storage_file = persistence_service._get_snapshot_path(
            agent_id, snapshot.snapshot_id
        )
        assert storage_file.exists()

        # Delete snapshot
        await persistence_service.delete_snapshot(agent_id, snapshot.snapshot_id)

        # Verify file deleted
        assert not storage_file.exists()

        # Verify cache cleared
        assert snapshot.snapshot_id not in persistence_service._backup_metadata

    async def test_delete_snapshot_clears_state_cache(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deleting snapshot clears state cache."""
        agent_id = "agent-024"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Verify cache exists
        assert agent_id in persistence_service._state_cache

        # Delete snapshot
        await persistence_service.delete_snapshot(agent_id, snapshot.snapshot_id)

        # Verify state cache cleared
        assert agent_id not in persistence_service._state_cache

    async def test_delete_snapshot_not_in_cache(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test deleting snapshot when different snapshot is cached."""
        agent_id = "agent-025"

        # Save two snapshots
        snapshot1 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        snapshot2 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Cache has snapshot2
        assert persistence_service._state_cache[agent_id].snapshot_id == snapshot2.snapshot_id

        # Delete snapshot1
        await persistence_service.delete_snapshot(agent_id, snapshot1.snapshot_id)

        # Cache should still have snapshot2
        assert agent_id in persistence_service._state_cache
        assert persistence_service._state_cache[agent_id].snapshot_id == snapshot2.snapshot_id

    async def test_delete_snapshot_not_found(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test deleting non-existent snapshot."""
        agent_id = "agent-nonexistent"

        with pytest.raises(StateNotFoundError) as exc_info:
            await persistence_service.delete_snapshot(agent_id, "nonexistent-snapshot")

        assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
class TestStatePersistenceServiceCleanup:
    """Test suite for cleanup operations."""

    async def test_cleanup_expired_snapshots(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test cleaning up expired snapshots."""
        agent_id = "agent-026"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Manually expire the snapshot
        metadata = persistence_service._backup_metadata[snapshot.snapshot_id]
        metadata.expires_at = datetime.now(UTC) - timedelta(days=1)

        # Run cleanup
        deleted_count = await persistence_service.cleanup_expired_snapshots()

        # Verify snapshot was deleted
        assert deleted_count == 1
        assert snapshot.snapshot_id not in persistence_service._backup_metadata

        storage_file = persistence_service._get_snapshot_path(
            agent_id, snapshot.snapshot_id
        )
        assert not storage_file.exists()

    async def test_cleanup_keeps_non_expired_snapshots(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test cleanup keeps non-expired snapshots."""
        agent_id = "agent-027"

        # Save snapshot with future expiration
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Verify not expired
        metadata = persistence_service._backup_metadata[snapshot.snapshot_id]
        assert metadata.expires_at > datetime.now(UTC)

        # Run cleanup
        deleted_count = await persistence_service.cleanup_expired_snapshots()

        # Verify snapshot was NOT deleted
        assert deleted_count == 0
        assert snapshot.snapshot_id in persistence_service._backup_metadata

        storage_file = persistence_service._get_snapshot_path(
            agent_id, snapshot.snapshot_id
        )
        assert storage_file.exists()

    async def test_cleanup_multiple_expired_snapshots(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test cleaning up multiple expired snapshots."""
        agent_id = "agent-028"

        # Save 3 snapshots and expire 2 of them
        snapshots = []
        for i in range(3):
            snapshot = await persistence_service.save_state(
                agent_id=f"{agent_id}-{i}",
                state_data=sample_state_data,
            )
            snapshots.append(snapshot)

        # Expire first two
        for i in range(2):
            metadata = persistence_service._backup_metadata[snapshots[i].snapshot_id]
            metadata.expires_at = datetime.now(UTC) - timedelta(days=1)

        # Run cleanup
        deleted_count = await persistence_service.cleanup_expired_snapshots()

        # Verify 2 deleted, 1 kept
        assert deleted_count == 2
        assert snapshots[0].snapshot_id not in persistence_service._backup_metadata
        assert snapshots[1].snapshot_id not in persistence_service._backup_metadata
        assert snapshots[2].snapshot_id in persistence_service._backup_metadata

    async def test_cleanup_no_expired_snapshots(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test cleanup with no expired snapshots."""
        agent_id = "agent-029"

        # Save snapshot
        await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Run cleanup
        deleted_count = await persistence_service.cleanup_expired_snapshots()

        # No snapshots should be deleted
        assert deleted_count == 0

    async def test_cleanup_handles_deletion_errors(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test cleanup handles deletion errors gracefully."""
        agent_id = "agent-030"

        # Save snapshot
        snapshot = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Manually expire
        metadata = persistence_service._backup_metadata[snapshot.snapshot_id]
        metadata.expires_at = datetime.now(UTC) - timedelta(days=1)

        # Delete file manually to cause error
        storage_file = persistence_service._get_snapshot_path(
            agent_id, snapshot.snapshot_id
        )
        storage_file.unlink()

        # Run cleanup (should not raise exception)
        deleted_count = await persistence_service.cleanup_expired_snapshots()

        # Should handle error and continue (count is 0 since delete failed)
        assert deleted_count == 0


@pytest.mark.asyncio
class TestStatePersistenceServiceHelpers:
    """Test suite for helper methods."""

    async def test_get_snapshot_path(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test _get_snapshot_path method."""
        agent_id = "agent-031"
        snapshot_id = "snapshot-123"

        path = persistence_service._get_snapshot_path(agent_id, snapshot_id)

        expected_path = persistence_service._storage_path / agent_id / f"{snapshot_id}.snapshot"
        assert path == expected_path

    async def test_find_latest_snapshot_with_snapshots(
        self,
        persistence_service: StatePersistenceService,
        sample_state_data: dict[str, Any],
    ) -> None:
        """Test _find_latest_snapshot with existing snapshots."""
        agent_id = "agent-032"

        # Save multiple snapshots
        snapshot1 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )
        await asyncio.sleep(0.01)

        snapshot2 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )
        await asyncio.sleep(0.01)

        snapshot3 = await persistence_service.save_state(
            agent_id=agent_id,
            state_data=sample_state_data,
        )

        # Find latest
        latest_id = await persistence_service._find_latest_snapshot(agent_id)

        # Should return most recent (snapshot3)
        assert latest_id == snapshot3.snapshot_id

    async def test_find_latest_snapshot_no_directory(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test _find_latest_snapshot with non-existent directory."""
        agent_id = "agent-nonexistent"

        latest_id = await persistence_service._find_latest_snapshot(agent_id)

        assert latest_id is None

    async def test_find_latest_snapshot_empty_directory(
        self,
        persistence_service: StatePersistenceService,
    ) -> None:
        """Test _find_latest_snapshot with empty directory."""
        agent_id = "agent-033"

        # Create empty directory
        agent_dir = persistence_service._storage_path / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        latest_id = await persistence_service._find_latest_snapshot(agent_id)

        assert latest_id is None
