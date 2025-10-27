"""Tests for agent state persistence."""

import tempfile
from pathlib import Path

import pytest

from agentcore.agent_runtime.models.state_persistence import (
    CompressionType,
    StateRestoreRequest,
    StateVersion)
from agentcore.agent_runtime.services.state_migration import (
    StateMigrationService,
    get_migration_service)
from agentcore.agent_runtime.services.state_persistence import (
    StateNotFoundError,
    StatePersistenceError,
    StatePersistenceService)
from agentcore.agent_runtime.services.state_serializer import (
    CURRENT_STATE_VERSION,
    StateDeserializationError,
    StateSerializationError,
    StateSerializer)


class TestStateSerializer:
    """Tests for state serialization."""

    def test_serialize_state_success(self) -> None:
        """Test successful state serialization."""
        serializer = StateSerializer(compression=CompressionType.GZIP)

        state_data = {
            "status": "running",
            "philosophy": "react",
            "execution_context": {"step": 1},
            "tool_usage_log": [],
            "performance_metrics": {"cpu": 0.5},
            "working_memory": {"key": "value"},
            "long_term_memory": {},
        }

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id="test-agent",
            state_data=state_data,
            compression=CompressionType.GZIP)

        assert compressed_bytes
        assert snapshot.agent_id == "test-agent"
        assert snapshot.philosophy == "react"
        assert snapshot.compression == CompressionType.GZIP
        assert snapshot.compressed_size is not None
        assert snapshot.uncompressed_size is not None
        assert snapshot.compressed_size < snapshot.uncompressed_size

    def test_serialize_with_no_compression(self) -> None:
        """Test serialization without compression."""
        serializer = StateSerializer(compression=CompressionType.NONE)

        state_data = {
            "status": "running",
            "philosophy": "cot",
            "execution_context": {},
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {},
            "long_term_memory": {},
        }

        compressed_bytes, snapshot = serializer.serialize_state(
            agent_id="test-agent",
            state_data=state_data,
            compression=CompressionType.NONE)

        assert compressed_bytes
        assert snapshot.compressed_size == snapshot.uncompressed_size

    def test_deserialize_state_success(self) -> None:
        """Test successful state deserialization."""
        serializer = StateSerializer()

        state_data = {
            "status": "paused",
            "philosophy": "autonomous",
            "execution_context": {"goal": "test"},
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {},
            "long_term_memory": {},
        }

        compressed_bytes, original_snapshot = serializer.serialize_state(
            agent_id="test-agent",
            state_data=state_data)

        deserialized_snapshot = serializer.deserialize_state(
            compressed_data=compressed_bytes,
            compression=CompressionType.GZIP)

        assert deserialized_snapshot.agent_id == original_snapshot.agent_id
        assert deserialized_snapshot.snapshot_id == original_snapshot.snapshot_id
        assert deserialized_snapshot.philosophy == "autonomous"
        assert deserialized_snapshot.status == "paused"

    def test_deserialize_invalid_data(self) -> None:
        """Test deserialization with invalid data."""
        serializer = StateSerializer()

        with pytest.raises(StateDeserializationError):
            serializer.deserialize_state(
                compressed_data=b"invalid data",
                compression=CompressionType.GZIP)

    def test_checksum_calculation(self) -> None:
        """Test checksum calculation and verification."""
        serializer = StateSerializer()
        data = b"test data for checksum"

        checksum = serializer.calculate_checksum(data)
        assert len(checksum) == 64  # SHA256 hex length
        assert serializer.verify_checksum(data, checksum)

    def test_checksum_mismatch(self) -> None:
        """Test checksum verification failure."""
        serializer = StateSerializer()
        data = b"test data"

        checksum = serializer.calculate_checksum(data)
        modified_data = b"modified data"

        assert not serializer.verify_checksum(modified_data, checksum)


@pytest.mark.asyncio
class TestStatePersistenceService:
    """Tests for state persistence service."""

    @pytest.fixture
    def temp_storage(self) -> Path:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def persistence_service(self, temp_storage: Path) -> StatePersistenceService:
        """Create persistence service with temp storage."""
        return StatePersistenceService(
            storage_path=temp_storage,
            compression=CompressionType.GZIP,
            retention_days=7)

    async def test_save_state_success(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test successful state save."""
        state_data = {
            "status": "running",
            "philosophy": "react",
            "execution_context": {"step": 1},
            "reasoning_chain": [],
            "decision_history": [],
            "tool_usage_log": [],
            "performance_metrics": {"cpu": 0.5},
            "working_memory": {"key": "value"},
            "long_term_memory": {},
        }

        snapshot = await persistence_service.save_state(
            agent_id="agent-1",
            state_data=state_data,
            tags=["test", "react"])

        assert snapshot.agent_id == "agent-1"
        assert snapshot.philosophy == "react"
        assert "test" in snapshot.tags

    async def test_load_state_success(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test successful state load."""
        state_data = {
            "status": "paused",
            "philosophy": "cot",
            "execution_context": {},
            "reasoning_chain": [],
            "decision_history": [],
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {},
            "long_term_memory": {},
        }

        # Save state
        saved_snapshot = await persistence_service.save_state(
            agent_id="agent-2",
            state_data=state_data)

        # Load state
        loaded_snapshot = await persistence_service.load_state(
            agent_id="agent-2",
            snapshot_id=saved_snapshot.snapshot_id)

        assert loaded_snapshot.agent_id == "agent-2"
        assert loaded_snapshot.snapshot_id == saved_snapshot.snapshot_id
        assert loaded_snapshot.philosophy == "cot"

    async def test_load_latest_state(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test loading latest state without specific snapshot ID."""
        state_data = {
            "status": "running",
            "philosophy": "multi_agent",
            "execution_context": {},
            "reasoning_chain": [],
            "decision_history": [],
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {},
            "long_term_memory": {},
        }

        # Save multiple snapshots
        await persistence_service.save_state(agent_id="agent-3", state_data=state_data)
        await persistence_service.save_state(agent_id="agent-3", state_data=state_data)
        latest_snapshot = await persistence_service.save_state(
            agent_id="agent-3", state_data=state_data
        )

        # Load latest
        loaded_snapshot = await persistence_service.load_state(agent_id="agent-3")

        assert loaded_snapshot.snapshot_id == latest_snapshot.snapshot_id

    async def test_load_nonexistent_state(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test loading state that doesn't exist."""
        with pytest.raises(StateNotFoundError):
            await persistence_service.load_state(agent_id="nonexistent")

    async def test_restore_state_full(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test full state restore."""
        state_data = {
            "status": "completed",
            "philosophy": "autonomous",
            "execution_context": {"goal": "completed"},
            "reasoning_chain": [{"step": 1}],
            "decision_history": [{"decision": "A"}],
            "tool_usage_log": [{"tool": "calculator"}],
            "performance_metrics": {"time": 1.5},
            "working_memory": {"data": "test"},
            "long_term_memory": {"learned": "something"},
        }

        # Save state
        snapshot = await persistence_service.save_state(
            agent_id="agent-4",
            state_data=state_data)

        # Restore state
        restore_request = StateRestoreRequest(
            agent_id="agent-4",
            snapshot_id=snapshot.snapshot_id,
            restore_memory=True,
            restore_execution_context=True,
            restore_tools=True,
            target_status="running")

        result = await persistence_service.restore_state(restore_request)

        assert result.success
        assert result.agent_id == "agent-4"
        assert result.restored_snapshot_id == snapshot.snapshot_id

    async def test_restore_state_partial(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test partial state restore (memory only)."""
        state_data = {
            "status": "paused",
            "philosophy": "react",
            "execution_context": {},
            "reasoning_chain": [],
            "decision_history": [],
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {"key": "value"},
            "long_term_memory": {},
        }

        snapshot = await persistence_service.save_state(
            agent_id="agent-5",
            state_data=state_data)

        # Restore only memory
        restore_request = StateRestoreRequest(
            agent_id="agent-5",
            snapshot_id=snapshot.snapshot_id,
            restore_memory=True,
            restore_execution_context=False,
            restore_tools=False)

        result = await persistence_service.restore_state(restore_request)

        assert result.success

    async def test_list_snapshots(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test listing snapshots for an agent."""
        state_data = {
            "status": "running",
            "philosophy": "cot",
            "execution_context": {},
            "reasoning_chain": [],
            "decision_history": [],
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {},
            "long_term_memory": {},
        }

        # Create multiple snapshots
        for i in range(3):
            await persistence_service.save_state(
                agent_id="agent-6",
                state_data=state_data,
                tags=[f"snapshot-{i}"])

        # List snapshots
        snapshots = await persistence_service.list_snapshots(agent_id="agent-6")

        assert len(snapshots) == 3

    async def test_delete_snapshot(
        self, persistence_service: StatePersistenceService
    ) -> None:
        """Test snapshot deletion."""
        state_data = {
            "status": "running",
            "philosophy": "react",
            "execution_context": {},
            "reasoning_chain": [],
            "decision_history": [],
            "tool_usage_log": [],
            "performance_metrics": {},
            "working_memory": {},
            "long_term_memory": {},
        }

        snapshot = await persistence_service.save_state(
            agent_id="agent-7",
            state_data=state_data)

        # Delete snapshot
        await persistence_service.delete_snapshot(
            agent_id="agent-7",
            snapshot_id=snapshot.snapshot_id)

        # Verify deletion
        with pytest.raises(StateNotFoundError):
            await persistence_service.load_state(
                agent_id="agent-7",
                snapshot_id=snapshot.snapshot_id)


class TestStateMigration:
    """Tests for state migration service."""

    def test_state_version_parsing(self) -> None:
        """Test state version parsing."""
        version = StateVersion.parse("1.2.3")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"

    def test_state_version_compatibility(self) -> None:
        """Test version compatibility check."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)
        v3 = StateVersion(major=2, minor=0, patch=0)

        assert v1.is_compatible(v2)  # Same major version
        assert not v1.is_compatible(v3)  # Different major version

    def test_register_migration(self) -> None:
        """Test migration registration."""
        service = StateMigrationService()

        def test_migration(data: dict) -> dict:
            data["migrated"] = True
            return data

        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=1, minor=1, patch=0)

        service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=test_migration,
            description="Test migration",
            breaking_change=False)

        assert service.can_migrate(from_version, to_version)

    def test_migration_not_needed(self) -> None:
        """Test when no migration is needed."""
        service = StateMigrationService()
        version = CURRENT_STATE_VERSION

        from agentcore.agent_runtime.models.state_persistence import AgentStateSnapshot

        snapshot = AgentStateSnapshot(
            agent_id="test",
            snapshot_id="test-snapshot",
            version=version,
            status="running",
            philosophy="react")

        migrated_snapshot, migrations = service.migrate_state(snapshot, version)

        assert len(migrations) == 0
        assert migrated_snapshot.version == version
