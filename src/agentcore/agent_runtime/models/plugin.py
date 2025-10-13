"""Plugin system models for dynamic agent capabilities."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PluginType(str, Enum):
    """Types of plugins supported by the system."""

    TOOL = "tool"
    ENGINE = "engine"
    MIDDLEWARE = "middleware"
    INTEGRATION = "integration"
    CUSTOM = "custom"


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    UNLOADING = "unloading"


class PluginCapability(BaseModel):
    """Capability provided by a plugin."""

    name: str = Field(description="Capability name")
    description: str = Field(description="Capability description")
    version: str = Field(description="Capability version")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Capability parameters schema",
    )


class PluginDependency(BaseModel):
    """Plugin dependency specification."""

    plugin_id: str = Field(description="Required plugin identifier")
    version_constraint: str = Field(
        default="*",
        description="Version constraint (semver pattern)",
    )
    optional: bool = Field(
        default=False,
        description="Whether dependency is optional",
    )


class PluginPermissions(BaseModel):
    """Permissions required by plugin."""

    filesystem_read: list[str] = Field(
        default_factory=list,
        description="Filesystem paths requiring read access",
    )
    filesystem_write: list[str] = Field(
        default_factory=list,
        description="Filesystem paths requiring write access",
    )
    network_hosts: list[str] = Field(
        default_factory=list,
        description="Network hosts requiring access",
    )
    external_apis: list[str] = Field(
        default_factory=list,
        description="External APIs requiring access",
    )
    environment_variables: list[str] = Field(
        default_factory=list,
        description="Environment variables requiring access",
    )
    system_resources: list[str] = Field(
        default_factory=list,
        description="System resources requiring access (cpu, memory, etc)",
    )


class PluginMetadata(BaseModel):
    """Plugin metadata and manifest."""

    plugin_id: str = Field(
        description="Unique plugin identifier (reverse DNS style)",
    )
    name: str = Field(description="Human-readable plugin name")
    version: str = Field(description="Plugin version (semver)")
    description: str = Field(description="Plugin description")
    author: str = Field(description="Plugin author")
    license: str = Field(default="MIT", description="Plugin license")
    homepage: str = Field(default="", description="Plugin homepage URL")
    plugin_type: PluginType = Field(description="Plugin type")
    capabilities: list[PluginCapability] = Field(
        default_factory=list,
        description="Capabilities provided by plugin",
    )
    dependencies: list[PluginDependency] = Field(
        default_factory=list,
        description="Plugin dependencies",
    )
    permissions: PluginPermissions = Field(
        default_factory=PluginPermissions,
        description="Required permissions",
    )
    entry_point: str = Field(
        description="Entry point module path (e.g., 'my_plugin.main')",
    )
    config_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration schema (JSON Schema)",
    )
    min_runtime_version: str = Field(
        default="0.1.0",
        description="Minimum required runtime version",
    )
    max_runtime_version: str = Field(
        default="*",
        description="Maximum compatible runtime version",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for plugin discovery",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Plugin creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Plugin last update timestamp",
    )

    @field_validator("plugin_id")
    @classmethod
    def validate_plugin_id(cls, v: str) -> str:
        """Validate plugin ID follows reverse DNS pattern."""
        if not v or "." not in v:
            raise ValueError(
                "Plugin ID must follow reverse DNS pattern (e.g., 'com.example.plugin')"
            )
        return v

    @field_validator("version", "min_runtime_version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate semantic version format."""
        if v != "*" and not all(
            part.isdigit() for part in v.split(".") if part != "*"
        ):
            raise ValueError(f"Invalid semantic version: {v}")
        return v


class PluginConfig(BaseModel):
    """Runtime configuration for plugin instance."""

    plugin_id: str = Field(description="Plugin identifier")
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    auto_load: bool = Field(
        default=False,
        description="Load plugin automatically on startup",
    )
    priority: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Plugin loading priority (higher = earlier)",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration",
    )
    sandbox_config_override: dict[str, Any] = Field(
        default_factory=dict,
        description="Sandbox configuration overrides for plugin",
    )


class PluginState(BaseModel):
    """Runtime state of loaded plugin."""

    plugin_id: str = Field(description="Plugin identifier")
    status: PluginStatus = Field(
        default=PluginStatus.UNLOADED,
        description="Current plugin status",
    )
    metadata: PluginMetadata = Field(description="Plugin metadata")
    config: PluginConfig = Field(description="Plugin configuration")
    instance: Any | None = Field(
        default=None,
        exclude=True,
        description="Loaded plugin instance (not serialized)",
    )
    load_time: datetime | None = Field(
        default=None,
        description="Time when plugin was loaded",
    )
    error_message: str = Field(
        default="",
        description="Error message if status is FAILED",
    )
    usage_count: int = Field(
        default=0,
        description="Number of times plugin has been used",
    )
    last_used: datetime | None = Field(
        default=None,
        description="Last time plugin was used",
    )


class PluginValidationResult(BaseModel):
    """Result of plugin security validation."""

    valid: bool = Field(description="Whether plugin passed validation")
    errors: list[str] = Field(
        default_factory=list,
        description="Validation errors",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings",
    )
    security_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Security score (0-100)",
    )
    risk_level: str = Field(
        default="unknown",
        description="Risk level assessment (low, medium, high, critical)",
    )
    scanned_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Validation timestamp",
    )


class PluginMarketplaceInfo(BaseModel):
    """Plugin information from marketplace."""

    plugin_id: str = Field(description="Plugin identifier")
    marketplace_url: str = Field(description="Marketplace listing URL")
    download_url: str = Field(description="Plugin download URL")
    version: str = Field(description="Available version")
    checksum: str = Field(description="SHA-256 checksum of plugin package")
    signature: str = Field(
        default="",
        description="Cryptographic signature (if signed)",
    )
    downloads_count: int = Field(
        default=0,
        description="Total download count",
    )
    rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="User rating (0-5)",
    )
    verified: bool = Field(
        default=False,
        description="Whether plugin is verified by marketplace",
    )
    last_updated: datetime = Field(description="Last update timestamp")


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails."""

    def __init__(
        self,
        plugin_id: str,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize plugin load error."""
        super().__init__(f"Failed to load plugin {plugin_id}: {message}")
        self.plugin_id = plugin_id
        self.message = message
        self.original_error = original_error


class PluginValidationError(Exception):
    """Exception raised when plugin validation fails."""

    def __init__(
        self,
        plugin_id: str,
        validation_result: PluginValidationResult,
    ) -> None:
        """Initialize plugin validation error."""
        super().__init__(
            f"Plugin {plugin_id} failed validation: {', '.join(validation_result.errors)}"
        )
        self.plugin_id = plugin_id
        self.validation_result = validation_result


class PluginVersionConflictError(Exception):
    """Exception raised when plugin version conflicts occur."""

    def __init__(
        self,
        plugin_id: str,
        required_version: str,
        available_version: str,
    ) -> None:
        """Initialize version conflict error."""
        super().__init__(
            f"Plugin {plugin_id} version conflict: "
            f"required {required_version}, available {available_version}"
        )
        self.plugin_id = plugin_id
        self.required_version = required_version
        self.available_version = available_version
