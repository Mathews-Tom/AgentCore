"""Agent state persistence service with database integration."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from ..models.state_persistence import (
    AgentStateSnapshot,
    CompressionType,
    StateBackupMetadata,
    StateRestoreRequest,
    StateRestoreResult,
    StateVersion,
)
from .state_serializer import (
    CURRENT_STATE_VERSION,
    StateDeserializationError,
    StateSerializationError,
    get_serializer,
)

logger = structlog.get_logger()


class StatePersistenceError(Exception):
    """Base exception for state persistence errors."""


class StateNotFoundError(StatePersistenceError):
    """Raised when requested state is not found."""


class StatePersistenceService:
    """
    Service for persisting and recovering agent state.

    This service handles:
    - State serialization and compression
    - State backup and recovery
    - State versioning and migration
    - Integration with storage backend
    """

    def __init__(
        self,
        storage_path: Path | str,
        compression: CompressionType = CompressionType.GZIP,
        retention_days: int = 30,
    ) -> None:
        """
        Initialize state persistence service.

        Args:
            storage_path: Base path for state storage
            compression: Default compression type
            retention_days: Default retention period in days
        """
        self._storage_path = Path(storage_path)
        self._compression = compression
        self._retention_days = retention_days
        self._serializer = get_serializer()

        # In-memory cache for recent states
        self._state_cache: dict[str, AgentStateSnapshot] = {}
        self._backup_metadata: dict[str, StateBackupMetadata] = {}

        # Ensure storage directory exists
        self._storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "state_persistence_initialized",
            storage_path=str(self._storage_path),
            compression=compression.value,
            retention_days=retention_days,
        )

    async def save_state(
        self,
        agent_id: str,
        state_data: dict[str, Any],
        tags: list[str] | None = None,
        compression: CompressionType | None = None,
    ) -> AgentStateSnapshot:
        """
        Save agent state to persistent storage.

        Args:
            agent_id: Agent identifier
            state_data: State data to persist
            tags: Optional tags for categorization
            compression: Compression type (or use default)

        Returns:
            State snapshot metadata

        Raises:
            StatePersistenceError: If save fails
        """
        try:
            # Add tags if provided
            if tags:
                state_data["tags"] = tags

            # Serialize and compress state
            compressed_data, snapshot = self._serializer.serialize_state(
                agent_id=agent_id,
                state_data=state_data,
                compression=compression or self._compression,
            )

            # Calculate checksum
            checksum = self._serializer.calculate_checksum(compressed_data)

            # Write to storage
            storage_file = self._get_snapshot_path(agent_id, snapshot.snapshot_id)
            storage_file.parent.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(storage_file.write_bytes, compressed_data)

            # Create backup metadata
            backup_metadata = StateBackupMetadata(
                backup_id=snapshot.snapshot_id,
                agent_id=agent_id,
                snapshot_id=snapshot.snapshot_id,
                version=snapshot.version,
                backup_type="full",
                storage_path=str(storage_file),
                compression=snapshot.compression,
                size_bytes=len(compressed_data),
                checksum=checksum,
                tags=snapshot.tags,
                retention_days=self._retention_days,
                expires_at=datetime.now() + timedelta(days=self._retention_days),
            )

            # Cache metadata
            self._backup_metadata[snapshot.snapshot_id] = backup_metadata
            self._state_cache[agent_id] = snapshot

            logger.info(
                "state_saved",
                agent_id=agent_id,
                snapshot_id=snapshot.snapshot_id,
                storage_path=str(storage_file),
                size_bytes=len(compressed_data),
            )

            return snapshot

        except (StateSerializationError, OSError) as e:
            logger.error(
                "state_save_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise StatePersistenceError(
                f"Failed to save state for agent {agent_id}: {e}"
            ) from e

    async def load_state(
        self,
        agent_id: str,
        snapshot_id: str | None = None,
    ) -> AgentStateSnapshot:
        """
        Load agent state from persistent storage.

        Args:
            agent_id: Agent identifier
            snapshot_id: Specific snapshot to load (or latest if None)

        Returns:
            Loaded agent state snapshot

        Raises:
            StateNotFoundError: If state not found
            StatePersistenceError: If load fails
        """
        try:
            # Check cache first
            if snapshot_id is None and agent_id in self._state_cache:
                cached_state = self._state_cache[agent_id]
                logger.debug("state_loaded_from_cache", agent_id=agent_id)
                return cached_state

            # Find snapshot
            if snapshot_id is None:
                snapshot_id = await self._find_latest_snapshot(agent_id)
                if snapshot_id is None:
                    raise StateNotFoundError(f"No state found for agent {agent_id}")

            # Read from storage
            storage_file = self._get_snapshot_path(agent_id, snapshot_id)
            if not storage_file.exists():
                raise StateNotFoundError(
                    f"Snapshot {snapshot_id} not found for agent {agent_id}"
                )

            compressed_data = await asyncio.to_thread(storage_file.read_bytes)

            # Verify checksum if metadata available
            if snapshot_id in self._backup_metadata:
                metadata = self._backup_metadata[snapshot_id]
                if not self._serializer.verify_checksum(
                    compressed_data, metadata.checksum
                ):
                    raise StatePersistenceError(
                        f"Checksum mismatch for snapshot {snapshot_id}"
                    )

            # Deserialize state
            snapshot = self._serializer.deserialize_state(
                compressed_data=compressed_data,
                compression=CompressionType.GZIP,  # Default for now
            )

            # Update cache
            self._state_cache[agent_id] = snapshot

            logger.info(
                "state_loaded",
                agent_id=agent_id,
                snapshot_id=snapshot.snapshot_id,
                version=str(snapshot.version),
            )

            return snapshot

        except StateNotFoundError:
            raise
        except (StateDeserializationError, OSError) as e:
            logger.error(
                "state_load_failed",
                agent_id=agent_id,
                snapshot_id=snapshot_id,
                error=str(e),
            )
            raise StatePersistenceError(
                f"Failed to load state for agent {agent_id}: {e}"
            ) from e

    async def restore_state(
        self,
        request: StateRestoreRequest,
    ) -> StateRestoreResult:
        """
        Restore agent state from backup.

        Args:
            request: Restore request parameters

        Returns:
            Restore operation result

        Raises:
            StatePersistenceError: If restore fails
        """
        try:
            # Load snapshot
            snapshot = await self.load_state(
                agent_id=request.agent_id,
                snapshot_id=request.snapshot_id or request.backup_id,
            )

            # Check version compatibility
            migrations_applied = []
            if not snapshot.version.is_compatible(CURRENT_STATE_VERSION):
                logger.warning(
                    "state_version_incompatible",
                    agent_id=request.agent_id,
                    snapshot_version=str(snapshot.version),
                    current_version=str(CURRENT_STATE_VERSION),
                )
                # TODO: Apply migrations if needed

            # Build filtered state based on restore options
            restored_state: dict[str, Any] = {
                "agent_id": snapshot.agent_id,
                "status": request.target_status or snapshot.status,
                "philosophy": snapshot.philosophy,
            }

            if request.restore_execution_context:
                restored_state["execution_context"] = snapshot.execution_context
                restored_state["reasoning_chain"] = snapshot.reasoning_chain
                restored_state["decision_history"] = snapshot.decision_history
                restored_state["current_step"] = snapshot.current_step

            if request.restore_memory:
                restored_state["working_memory"] = snapshot.working_memory
                restored_state["long_term_memory"] = snapshot.long_term_memory

            if request.restore_tools:
                restored_state["tool_usage_log"] = snapshot.tool_usage_log

            result = StateRestoreResult(
                agent_id=request.agent_id,
                restored_snapshot_id=snapshot.snapshot_id,
                restored_from_backup=request.backup_id,
                version=snapshot.version,
                migrations_applied=migrations_applied,
                success=True,
            )

            logger.info(
                "state_restored",
                agent_id=request.agent_id,
                snapshot_id=snapshot.snapshot_id,
            )

            return result

        except Exception as e:
            logger.error(
                "state_restore_failed",
                agent_id=request.agent_id,
                error=str(e),
            )
            return StateRestoreResult(
                agent_id=request.agent_id,
                restored_snapshot_id="",
                version=CURRENT_STATE_VERSION,
                success=False,
                error_message=str(e),
            )

    async def list_snapshots(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> list[StateBackupMetadata]:
        """
        List available state snapshots for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of snapshots to return

        Returns:
            List of backup metadata sorted by creation time (newest first)
        """
        agent_dir = self._storage_path / agent_id
        if not agent_dir.exists():
            return []

        # Find all snapshot files
        snapshot_files = list(agent_dir.glob("*.snapshot"))
        snapshot_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Build metadata list
        metadata_list = []
        for snapshot_file in snapshot_files[:limit]:
            snapshot_id = snapshot_file.stem
            if snapshot_id in self._backup_metadata:
                metadata_list.append(self._backup_metadata[snapshot_id])

        return metadata_list

    async def delete_snapshot(
        self,
        agent_id: str,
        snapshot_id: str,
    ) -> None:
        """
        Delete a specific state snapshot.

        Args:
            agent_id: Agent identifier
            snapshot_id: Snapshot to delete

        Raises:
            StateNotFoundError: If snapshot not found
        """
        storage_file = self._get_snapshot_path(agent_id, snapshot_id)
        if not storage_file.exists():
            raise StateNotFoundError(
                f"Snapshot {snapshot_id} not found for agent {agent_id}"
            )

        await asyncio.to_thread(storage_file.unlink)

        # Remove from cache
        if snapshot_id in self._backup_metadata:
            del self._backup_metadata[snapshot_id]
        if agent_id in self._state_cache:
            if self._state_cache[agent_id].snapshot_id == snapshot_id:
                del self._state_cache[agent_id]

        logger.info(
            "snapshot_deleted",
            agent_id=agent_id,
            snapshot_id=snapshot_id,
        )

    async def cleanup_expired_snapshots(self) -> int:
        """
        Clean up expired state snapshots.

        Returns:
            Number of snapshots deleted
        """
        deleted_count = 0
        now = datetime.now()

        for snapshot_id, metadata in list(self._backup_metadata.items()):
            if metadata.expires_at and metadata.expires_at < now:
                try:
                    await self.delete_snapshot(
                        metadata.agent_id,
                        snapshot_id,
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(
                        "snapshot_cleanup_failed",
                        snapshot_id=snapshot_id,
                        error=str(e),
                    )

        if deleted_count > 0:
            logger.info(
                "snapshots_cleaned_up",
                deleted_count=deleted_count,
            )

        return deleted_count

    def _get_snapshot_path(self, agent_id: str, snapshot_id: str) -> Path:
        """Get storage path for a snapshot."""
        return self._storage_path / agent_id / f"{snapshot_id}.snapshot"

    async def _find_latest_snapshot(self, agent_id: str) -> str | None:
        """Find the latest snapshot ID for an agent."""
        agent_dir = self._storage_path / agent_id
        if not agent_dir.exists():
            return None

        snapshot_files = list(agent_dir.glob("*.snapshot"))
        if not snapshot_files:
            return None

        # Get most recent file
        latest_file = max(snapshot_files, key=lambda p: p.stat().st_mtime)
        return latest_file.stem
