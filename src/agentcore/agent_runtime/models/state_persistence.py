"""Agent state persistence models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StateVersion(BaseModel):
    """State schema version information."""

    major: int = Field(description="Major version number (breaking changes)")
    minor: int = Field(description="Minor version number (backward compatible)")
    patch: int = Field(description="Patch version number (bug fixes)")

    def __str__(self) -> str:
        """String representation of version."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, version_string: str) -> "StateVersion":
        """Parse version string into StateVersion."""
        parts = version_string.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_string}")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
        )

    def is_compatible(self, other: "StateVersion") -> bool:
        """Check if this version is compatible with another."""
        return self.major == other.major


class CompressionType(str, Enum):
    """Supported compression types for state data."""

    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


class AgentStateSnapshot(BaseModel):
    """Complete agent state snapshot for persistence."""

    agent_id: str = Field(description="Agent identifier")
    snapshot_id: str = Field(description="Unique snapshot identifier")
    version: StateVersion = Field(description="State schema version")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Snapshot creation time",
    )

    # Agent execution state
    status: str = Field(description="Agent execution status")
    container_id: str | None = Field(
        default=None,
        description="Container ID if running",
    )
    current_step: str | None = Field(
        default=None,
        description="Current execution step",
    )

    # Execution context and history
    execution_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Philosophy-specific execution context",
    )
    reasoning_chain: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chain of reasoning steps",
    )
    decision_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of agent decisions",
    )
    tool_usage_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of tool executions",
    )

    # Performance and metrics
    performance_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics",
    )
    execution_time: float = Field(
        default=0.0,
        description="Total execution time in seconds",
    )

    # Memory state
    working_memory: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent working memory",
    )
    long_term_memory: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent long-term memory",
    )

    # Metadata
    philosophy: str = Field(description="Agent philosophy type")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    # Compression info
    compression: CompressionType = Field(
        default=CompressionType.GZIP,
        description="Compression type used",
    )
    compressed_size: int | None = Field(
        default=None,
        description="Size after compression in bytes",
    )
    uncompressed_size: int | None = Field(
        default=None,
        description="Original size in bytes",
    )


class StateBackupMetadata(BaseModel):
    """Metadata for state backups."""

    backup_id: str = Field(description="Unique backup identifier")
    agent_id: str = Field(description="Agent identifier")
    snapshot_id: str = Field(description="Associated snapshot ID")
    version: StateVersion = Field(description="State schema version")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Backup creation time",
    )
    backup_type: str = Field(
        description="Type of backup (full, incremental, differential)",
    )
    storage_path: str = Field(description="Storage location path")
    compression: CompressionType = Field(description="Compression type")
    size_bytes: int = Field(description="Backup size in bytes")
    checksum: str = Field(description="SHA256 checksum for integrity")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    retention_days: int | None = Field(
        default=None,
        description="Days to retain backup",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Expiration timestamp",
    )


class StateMigration(BaseModel):
    """State migration information for schema changes."""

    from_version: StateVersion = Field(description="Source version")
    to_version: StateVersion = Field(description="Target version")
    migration_function: str = Field(description="Migration function name")
    description: str = Field(description="Migration description")
    breaking_change: bool = Field(
        default=False,
        description="Whether migration involves breaking changes",
    )
    applied_at: datetime | None = Field(
        default=None,
        description="When migration was applied",
    )


class StateRestoreRequest(BaseModel):
    """Request to restore agent state from backup."""

    agent_id: str = Field(description="Agent identifier")
    backup_id: str | None = Field(
        default=None,
        description="Specific backup to restore (or latest if None)",
    )
    snapshot_id: str | None = Field(
        default=None,
        description="Specific snapshot to restore",
    )
    restore_memory: bool = Field(
        default=True,
        description="Whether to restore memory state",
    )
    restore_execution_context: bool = Field(
        default=True,
        description="Whether to restore execution context",
    )
    restore_tools: bool = Field(
        default=True,
        description="Whether to restore tool usage log",
    )
    target_status: str | None = Field(
        default=None,
        description="Target status after restore",
    )


class StateRestoreResult(BaseModel):
    """Result of state restore operation."""

    agent_id: str = Field(description="Agent identifier")
    restored_snapshot_id: str = Field(description="Restored snapshot ID")
    restored_from_backup: str | None = Field(
        default=None,
        description="Backup ID used for restore",
    )
    version: StateVersion = Field(description="State schema version")
    migrations_applied: list[StateMigration] = Field(
        default_factory=list,
        description="Migrations applied during restore",
    )
    restored_at: datetime = Field(
        default_factory=datetime.now,
        description="Restore timestamp",
    )
    success: bool = Field(description="Whether restore was successful")
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )
