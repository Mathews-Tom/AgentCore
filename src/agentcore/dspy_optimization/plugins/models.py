"""
Plugin metadata models for custom optimizer registration
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PluginStatus(str, Enum):
    """Plugin lifecycle status"""

    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class PluginCapability(str, Enum):
    """Plugin capabilities"""

    MULTI_OBJECTIVE = "multi_objective"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_FREE = "gradient_free"
    GRADIENT_BASED = "gradient_based"
    BAYESIAN = "bayesian"
    REINFORCEMENT = "reinforcement"
    HYBRID = "hybrid"


class PluginMetadata(BaseModel):
    """Metadata for optimizer plugin"""

    name: str = Field(description="Unique plugin name")
    version: str = Field(description="Semantic version (e.g., 1.0.0)")
    author: str = Field(description="Plugin author")
    description: str = Field(description="Plugin description")
    capabilities: list[PluginCapability] = Field(default_factory=list)
    requires_python: str = Field(default=">=3.12")
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    documentation_url: str | None = None
    license: str = Field(default="MIT")


class PluginConfig(BaseModel):
    """Configuration for plugin instance"""

    plugin_name: str
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    timeout_seconds: int = Field(default=7200, ge=60)
    max_memory_mb: int = Field(default=4096, ge=128)
    parameters: dict[str, Any] = Field(default_factory=dict)


class PluginValidationResult(BaseModel):
    """Result of plugin validation"""

    plugin_name: str
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0


class PluginRegistration(BaseModel):
    """Plugin registration record"""

    metadata: PluginMetadata
    config: PluginConfig
    status: PluginStatus = PluginStatus.REGISTERED
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None
    usage_count: int = 0
    error_message: str | None = None
