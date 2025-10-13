"""Agent configuration models for runtime execution."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AgentPhilosophy(str, Enum):
    """Supported agent execution philosophies."""

    REACT = "react"
    CHAIN_OF_THOUGHT = "cot"
    MULTI_AGENT = "multi_agent"
    AUTONOMOUS = "autonomous"


class ResourceLimits(BaseModel):
    """Resource limits for agent container execution."""

    max_memory_mb: int = Field(
        default=512,
        ge=128,
        le=8192,
        description="Maximum memory in MB",
    )
    max_cpu_cores: float = Field(
        default=1.0,
        ge=0.1,
        le=8.0,
        description="Maximum CPU cores (fractional)",
    )
    max_execution_time_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Maximum execution time in seconds",
    )
    max_file_descriptors: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum file descriptors",
    )
    network_access: Literal["none", "restricted", "full"] = Field(
        default="restricted",
        description="Network access level",
    )
    storage_quota_mb: int = Field(
        default=1024,
        ge=100,
        le=10240,
        description="Storage quota in MB",
    )


class SecurityProfile(BaseModel):
    """Security profile for agent container isolation."""

    profile_name: Literal["minimal", "standard", "privileged"] = Field(
        default="standard",
        description="Predefined security profile",
    )
    allowed_syscalls: list[str] = Field(
        default_factory=list,
        description="Explicitly allowed system calls",
    )
    blocked_syscalls: list[str] = Field(
        default_factory=lambda: ["mount", "umount", "chroot", "pivot_root"],
        description="Blocked dangerous system calls",
    )
    user_namespace: bool = Field(
        default=True,
        description="Enable user namespace remapping",
    )
    read_only_filesystem: bool = Field(
        default=True,
        description="Read-only root filesystem",
    )
    no_new_privileges: bool = Field(
        default=True,
        description="Prevent privilege escalation",
    )


class AgentConfig(BaseModel):
    """Complete agent configuration for runtime deployment."""

    agent_id: str = Field(
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique agent identifier",
    )
    philosophy: AgentPhilosophy = Field(
        description="Agent execution philosophy",
    )
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Resource constraints",
    )
    security_profile: SecurityProfile = Field(
        default_factory=SecurityProfile,
        description="Security isolation settings",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Allowed tool IDs",
    )
    environment_variables: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for agent",
    )
    image_tag: str = Field(
        default="agentcore/agent-runtime:latest",
        description="Docker image tag for agent execution",
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        """Validate tool permissions and existence."""
        if len(v) > 100:
            raise ValueError("Maximum 100 tools allowed per agent")
        return v

    @field_validator("environment_variables")
    @classmethod
    def validate_env_vars(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate environment variables for security."""
        restricted_keys = ["PATH", "HOME", "USER", "SHELL"]
        for key in v:
            if key in restricted_keys:
                raise ValueError(f"Cannot override system variable: {key}")
        return v
