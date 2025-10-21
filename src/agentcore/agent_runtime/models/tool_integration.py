"""Tool integration models for agent runtime."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Definition of an external tool available to agents."""

    tool_id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Human-readable tool name")
    description: str = Field(description="Tool functionality description")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameter schema (JSON Schema format)",
    )
    security_requirements: list[str] = Field(
        default_factory=list,
        description="Required security permissions",
    )
    rate_limits: dict[str, int] = Field(
        default_factory=dict,
        description="Rate limits (e.g., calls_per_minute: 60)",
    )
    cost_per_execution: float = Field(
        default=0.0,
        description="Cost per tool execution in USD",
    )


class ToolExecutionRequest(BaseModel):
    """Request to execute a tool on behalf of an agent."""

    tool_id: str = Field(description="Tool to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool execution parameters",
    )
    execution_context: dict[str, str] = Field(
        default_factory=dict,
        description="Execution context metadata",
    )
    agent_id: str = Field(description="Requesting agent ID")
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier",
    )
