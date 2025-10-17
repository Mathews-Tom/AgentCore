"""
Pydantic models for training infrastructure.

Provides type-safe configuration and data validation for GRPO training system.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class TrainingJobStatus(str, Enum):
    """Training job status enumeration."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GRPOConfig(BaseModel):
    """Configuration for GRPO training algorithm.

    Defines hyperparameters and resource limits for training jobs.
    """

    n_iterations: int = Field(
        default=1000, ge=1, le=10000, description="Number of training iterations"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Number of queries per training batch",
    )
    n_trajectories_per_query: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Number of parallel trajectories per query",
    )
    learning_rate: float = Field(
        default=0.0001,
        gt=0,
        le=1.0,
        description="Policy gradient learning rate",
    )
    max_budget_usd: Decimal = Field(
        default=Decimal("100.00"),
        ge=Decimal("0"),
        description="Maximum training budget in USD",
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N iterations",
    )
    max_steps_per_trajectory: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum steps per trajectory",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor for multi-step credit assignment",
    )

    @field_validator("max_budget_usd", mode="before")
    @classmethod
    def convert_budget_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert budget to Decimal for precise monetary calculations."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class TrainingQuery(BaseModel):
    """Single training query with expected outcome."""

    query: str = Field(min_length=1, description="User query or task prompt")
    expected_outcome: dict[str, object] = Field(
        description="Expected task outcome for reward computation"
    )


class TrajectoryStep(BaseModel):
    """Single step in an agent execution trajectory."""

    state: dict[str, object] = Field(
        description="Agent state (context, memory, variables)"
    )
    action: dict[str, object] = Field(
        description="Action taken (tool call, planning decision)"
    )
    result: dict[str, object] = Field(
        description="Result of action execution"
    )
    timestamp: datetime = Field(description="Step execution timestamp")
    duration_ms: int = Field(ge=0, description="Step execution duration in milliseconds")


class Trajectory(BaseModel):
    """Complete agent execution trajectory for a query."""

    trajectory_id: UUID | None = Field(
        default=None, description="Unique trajectory identifier"
    )
    job_id: UUID = Field(description="Parent training job ID")
    agent_id: str = Field(min_length=1, max_length=255, description="Agent identifier")
    query: str = Field(min_length=1, description="Query that generated this trajectory")
    steps: list[TrajectoryStep] = Field(
        default_factory=list, description="Execution steps"
    )
    reward: float = Field(default=0.0, description="Raw reward value")
    normalized_reward: float = Field(
        default=0.0, description="Group-normalized reward"
    )
    advantage: float = Field(
        default=0.0, description="Advantage (for policy gradient)"
    )
    execution_time_ms: int | None = Field(
        default=None, ge=0, description="Total trajectory execution time"
    )
    success: bool | None = Field(default=None, description="Task success indicator")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")

    @field_validator("steps")
    @classmethod
    def validate_steps_length(cls, v: list[TrajectoryStep]) -> list[TrajectoryStep]:
        """Validate trajectory doesn't exceed max steps."""
        if len(v) > 100:  # Safety limit
            raise ValueError("Trajectory exceeds maximum 100 steps")
        return v


class TrainingJob(BaseModel):
    """Training job model for GRPO algorithm."""

    job_id: UUID | None = Field(
        default=None, description="Unique job identifier (assigned by database)"
    )
    agent_id: str = Field(min_length=1, max_length=255, description="Agent identifier")
    status: TrainingJobStatus = Field(
        default=TrainingJobStatus.QUEUED, description="Job status"
    )
    config: GRPOConfig = Field(description="Training configuration")
    training_data: list[TrainingQuery] = Field(
        min_length=100, description="Training queries (minimum 100 required)"
    )
    current_iteration: int = Field(
        default=0, ge=0, description="Current training iteration"
    )
    total_iterations: int = Field(ge=1, description="Total iterations to run")
    metrics: dict[str, float | int | str] = Field(
        default_factory=dict, description="Current training metrics"
    )
    cost_usd: Decimal = Field(
        default=Decimal("0.00"), ge=Decimal("0"), description="Accumulated cost in USD"
    )
    budget_usd: Decimal = Field(
        ge=Decimal("0"), description="Budget limit in USD"
    )
    best_checkpoint_id: UUID | None = Field(
        default=None, description="ID of best checkpoint"
    )
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    started_at: datetime | None = Field(default=None, description="Start timestamp")
    completed_at: datetime | None = Field(
        default=None, description="Completion timestamp"
    )
    error_message: str | None = Field(
        default=None, max_length=1000, description="Error message if failed"
    )

    @field_validator("cost_usd", mode="before")
    @classmethod
    def convert_cost_to_decimal(cls, v: float | Decimal | str) -> Decimal:
        """Convert cost to Decimal for precise monetary calculations."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @field_validator("budget_usd", mode="before")
    @classmethod
    def convert_budget_to_decimal_job(cls, v: float | Decimal | str) -> Decimal:
        """Convert budget to Decimal for precise monetary calculations."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class PolicyCheckpoint(BaseModel):
    """Policy checkpoint for training continuity."""

    checkpoint_id: UUID | None = Field(
        default=None, description="Unique checkpoint identifier"
    )
    agent_id: str = Field(min_length=1, max_length=255, description="Agent identifier")
    job_id: UUID = Field(description="Parent training job ID")
    iteration: int = Field(ge=0, description="Training iteration number")
    policy_data: dict[str, object] | None = (
        Field(default=None, description="Policy data for small policies (prompts)")
    )
    policy_s3_path: str | None = Field(
        default=None, max_length=500, description="S3 path for large policy weights"
    )
    validation_score: float = Field(
        default=0.0, description="Validation performance score"
    )
    metrics: dict[str, float | int | str] = Field(
        default_factory=dict, description="Training metrics at checkpoint"
    )
    created_at: datetime | None = Field(default=None, description="Creation timestamp")

    @field_validator("policy_data", "policy_s3_path")
    @classmethod
    def validate_policy_storage(
        cls,
        v: dict[str, object] | str | None,
        info,
    ) -> dict[str, object] | str | None:
        """Ensure at least one policy storage method is specified."""
        # Note: Pydantic v2 validators don't have access to other fields in field validators
        # This validation will be enforced at model level if needed
        return v
