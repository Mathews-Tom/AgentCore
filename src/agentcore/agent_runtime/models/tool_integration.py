"""Tool integration models for agent runtime."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ToolCategory(str, Enum):
    """Categories for tool classification."""

    SEARCH = "search"
    CODE_EXECUTION = "code_execution"
    API_CLIENT = "api_client"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    MONITORING = "monitoring"
    SECURITY = "security"
    CUSTOM = "custom"


class AuthMethod(str, Enum):
    """Authentication methods for tools."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    JWT = "jwt"
    CUSTOM = "custom"


class ToolParameter(BaseModel):
    """Detailed parameter definition for tools."""

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, number, boolean, object, array)")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not provided")
    enum: list[Any] | None = Field(default=None, description="Allowed values (if constrained)")
    min_value: float | None = Field(default=None, description="Minimum value for numbers")
    max_value: float | None = Field(default=None, description="Maximum value for numbers")
    min_length: int | None = Field(default=None, description="Minimum length for strings/arrays")
    max_length: int | None = Field(default=None, description="Maximum length for strings/arrays")
    pattern: str | None = Field(default=None, description="Regex pattern for string validation")


class ToolDefinition(BaseModel):
    """Definition of an external tool available to agents."""

    tool_id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Human-readable tool name")
    description: str = Field(description="Tool functionality description")
    version: str = Field(default="1.0.0", description="Tool version (semver format)")
    category: ToolCategory = Field(description="Tool category for discovery")

    # Parameters
    parameters: dict[str, ToolParameter] = Field(
        default_factory=dict,
        description="Detailed parameter definitions",
    )

    # Authentication
    auth_method: AuthMethod = Field(
        default=AuthMethod.NONE,
        description="Required authentication method",
    )
    auth_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Authentication configuration (e.g., token_url, scopes)",
    )

    # Execution Configuration
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Execution timeout in seconds",
    )
    is_retryable: bool = Field(
        default=True,
        description="Whether tool supports retries on failure",
    )
    is_idempotent: bool = Field(
        default=True,
        description="Whether multiple executions with same params produce same result",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )

    # Rate Limiting & Cost
    rate_limits: dict[str, int] = Field(
        default_factory=dict,
        description="Rate limits (e.g., calls_per_minute: 60)",
    )
    cost_per_execution: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per tool execution in USD",
    )

    # Capabilities & Requirements
    capabilities: list[str] = Field(
        default_factory=list,
        description="Tool capabilities (e.g., 'parallel_execution', 'streaming')",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="System requirements (e.g., 'docker', 'gpu')",
    )
    security_requirements: list[str] = Field(
        default_factory=list,
        description="Required security permissions",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional tool metadata",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for search and discovery",
    )

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate semantic versioning format."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must be in semver format (x.y.z)")
        for part in parts:
            if not part.isdigit():
                raise ValueError("Version parts must be numeric")
        return v


class ToolExecutionRequest(BaseModel):
    """Request to execute a tool on behalf of an agent."""

    tool_id: str = Field(description="Tool to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool execution parameters",
    )
    execution_context: dict[str, str] = Field(
        default_factory=dict,
        description="Execution context metadata (trace_id, session_id, etc.)",
    )
    agent_id: str = Field(description="Requesting agent ID")
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier",
    )
    timeout_override: int | None = Field(
        default=None,
        description="Override tool's default timeout (seconds)",
    )
    retry_override: int | None = Field(
        default=None,
        description="Override tool's default max retries",
    )


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolResult(BaseModel):
    """Result of tool execution with comprehensive metadata."""

    request_id: str = Field(description="Original request ID")
    tool_id: str = Field(description="Tool that was executed")
    status: ToolExecutionStatus = Field(description="Execution status")

    # Result Data
    result: Any = Field(default=None, description="Execution result data")
    error: str | None = Field(default=None, description="Error message if failed")
    error_type: str | None = Field(default=None, description="Error type/code")

    # Execution Metadata
    execution_time_ms: float = Field(description="Execution duration in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Result timestamp",
    )
    retry_count: int = Field(default=0, description="Number of retries attempted")

    # Resource Usage
    memory_mb: float | None = Field(default=None, description="Peak memory usage in MB")
    cpu_percent: float | None = Field(default=None, description="CPU utilization percentage")

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata",
    )

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolExecutionStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if execution failed."""
        return self.status in (
            ToolExecutionStatus.FAILED,
            ToolExecutionStatus.TIMEOUT,
            ToolExecutionStatus.CANCELLED,
        )
