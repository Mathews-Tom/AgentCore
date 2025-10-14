"""Sandbox security models for isolated agent execution."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class SandboxPermission(str, Enum):
    """Permission types for sandbox resource access."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    EXTERNAL_API = "external_api"
    PROCESS = "process"
    MEMORY = "memory"


class ResourcePolicy(BaseModel):
    """Policy defining allowed operations on resources."""

    resource_pattern: str = Field(
        description="Resource pattern (glob-style: /data/*, *.py, api.example.com/*)",
    )
    allowed_permissions: list[SandboxPermission] = Field(
        description="Permissions granted for this resource",
    )
    denied_permissions: list[SandboxPermission] = Field(
        default_factory=list,
        description="Explicitly denied permissions",
    )
    description: str = Field(
        default="",
        description="Human-readable policy description",
    )


class ExecutionLimits(BaseModel):
    """Resource limits for code execution in sandbox."""

    max_execution_time_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Maximum execution time in seconds",
    )
    max_memory_mb: int = Field(
        default=256,
        ge=64,
        le=4096,
        description="Maximum memory usage in MB",
    )
    max_cpu_percent: float = Field(
        default=50.0,
        ge=1.0,
        le=100.0,
        description="Maximum CPU usage percentage",
    )
    max_processes: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of processes/threads",
    )
    max_file_descriptors: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Maximum number of file descriptors",
    )
    max_network_requests: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Maximum network requests (0=disabled)",
    )


class SandboxConfig(BaseModel):
    """Complete sandbox configuration with permissions and limits."""

    sandbox_id: str = Field(
        description="Unique sandbox identifier",
    )
    agent_id: str = Field(
        description="Agent that owns this sandbox",
    )
    permissions: list[SandboxPermission] = Field(
        default_factory=lambda: [
            SandboxPermission.READ,
            SandboxPermission.EXECUTE,
        ],
        description="Global permissions for sandbox",
    )
    resource_policies: list[ResourcePolicy] = Field(
        default_factory=list,
        description="Fine-grained resource access policies",
    )
    execution_limits: ExecutionLimits = Field(
        default_factory=ExecutionLimits,
        description="Resource limits for execution",
    )
    strict_mode: bool = Field(
        default=True,
        description="Strict mode: fail on any permission violation",
    )
    allow_network: bool = Field(
        default=False,
        description="Allow network access",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Allowed network hosts (if network enabled)",
    )
    read_only_paths: list[str] = Field(
        default_factory=lambda: ["/app", "/usr", "/lib", "/etc"],
        description="Paths mounted read-only",
    )
    writable_paths: list[str] = Field(
        default_factory=lambda: ["/tmp", "/workspace"],
        description="Paths with write access",
    )
    environment_variables: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for sandbox",
    )


class AuditEventType(str, Enum):
    """Types of auditable security events."""

    PERMISSION_CHECK = "permission_check"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_ACCESS = "resource_access"
    EXECUTION_START = "execution_start"
    EXECUTION_COMPLETE = "execution_complete"
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_ERROR = "execution_error"
    LIMIT_EXCEEDED = "limit_exceeded"
    SECURITY_VIOLATION = "security_violation"
    SANDBOX_CREATED = "sandbox_created"
    SANDBOX_DESTROYED = "sandbox_destroyed"


class AuditLogEntry(BaseModel):
    """Security audit log entry for sandbox operations."""

    model_config = ConfigDict(
        # No need for json_encoders - Pydantic v2 handles datetime serialization automatically
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp",
    )
    event_type: AuditEventType = Field(
        description="Type of security event",
    )
    sandbox_id: str = Field(
        description="Sandbox identifier",
    )
    agent_id: str = Field(
        description="Agent identifier",
    )
    operation: str = Field(
        description="Operation attempted (e.g., 'read', 'write', 'execute')",
    )
    resource: str = Field(
        default="",
        description="Resource accessed or affected",
    )
    permission: SandboxPermission | None = Field(
        default=None,
        description="Permission checked",
    )
    result: bool = Field(
        description="Operation result (True=allowed, False=denied)",
    )
    reason: str = Field(
        default="",
        description="Reason for result (especially denials)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context information",
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return value.isoformat()


class SandboxStats(BaseModel):
    """Real-time statistics for sandbox execution."""

    sandbox_id: str = Field(
        description="Sandbox identifier",
    )
    cpu_usage_percent: float = Field(
        default=0.0,
        description="Current CPU usage percentage",
    )
    memory_usage_mb: float = Field(
        default=0.0,
        description="Current memory usage in MB",
    )
    execution_time_seconds: float = Field(
        default=0.0,
        description="Total execution time in seconds",
    )
    network_requests_count: int = Field(
        default=0,
        description="Number of network requests made",
    )
    file_operations_count: int = Field(
        default=0,
        description="Number of file operations",
    )
    process_count: int = Field(
        default=0,
        description="Number of active processes",
    )
    is_running: bool = Field(
        default=False,
        description="Whether sandbox is currently executing",
    )


class SecurityViolationError(Exception):
    """Exception raised when security policy is violated."""

    def __init__(
        self,
        message: str,
        permission: SandboxPermission | None = None,
        resource: str = "",
    ) -> None:
        """Initialize security violation error."""
        super().__init__(message)
        self.permission = permission
        self.resource = resource


class ResourceLimitExceededError(Exception):
    """Exception raised when resource limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        current_value: float,
        max_value: float,
    ) -> None:
        """Initialize resource limit exceeded error."""
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.max_value = max_value
