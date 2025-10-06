"""Agent execution state models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .agent_config import AgentPhilosophy


class AgentExecutionState(BaseModel):
    """Current execution state of an agent."""

    agent_id: str = Field(description="Agent identifier")
    container_id: str | None = Field(
        default=None,
        description="Container ID if running",
    )
    status: Literal[
        "initializing",
        "running",
        "paused",
        "completed",
        "failed",
        "terminated",
    ] = Field(description="Current execution status")
    current_step: str | None = Field(
        default=None,
        description="Current execution step description",
    )
    execution_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Philosophy-specific execution context",
    )
    tool_usage_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of tool executions",
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics (CPU, memory, latency)",
    )
    checkpoint_data: bytes | None = Field(
        default=None,
        description="Serialized checkpoint for state recovery",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp",
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Failure reason if status is failed",
    )


class PhilosophyExecutionContext(BaseModel):
    """Philosophy-specific execution context and metadata."""

    philosophy: AgentPhilosophy = Field(description="Agent philosophy type")
    execution_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Philosophy-specific parameters",
    )
    prompt_templates: dict[str, str] = Field(
        default_factory=dict,
        description="Prompt templates for LLM interaction",
    )
    reasoning_chain: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chain of reasoning steps",
    )
    decision_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of agent decisions",
    )
    optimization_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for DSPy optimization",
    )
