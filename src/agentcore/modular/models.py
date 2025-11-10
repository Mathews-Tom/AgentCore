"""
Execution Plan Data Models for Modular Agent Core

Provides comprehensive Pydantic models for execution planning, tracking,
and verification with database compatibility and rich validation.

These models extend the basic interface models with additional fields
for execution tracking, success criteria, and module transitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enumerations
# ============================================================================


class PlanStatus(str, Enum):
    """Status of an execution plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModuleType(str, Enum):
    """Type of module in the four-module architecture."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    GENERATOR = "generator"


class VerificationLevel(str, Enum):
    """Level of verification rigor."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


# ============================================================================
# Tool and Dependency Models
# ============================================================================


class ToolRequirement(BaseModel):
    """Requirements for a specific tool."""

    tool_name: str = Field(..., description="Name of the required tool")
    version: str | None = Field(None, description="Required tool version")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool configuration parameters"
    )
    optional: bool = Field(
        default=False, description="Whether tool is optional"
    )


class StepDependency(BaseModel):
    """Dependency relationship between plan steps."""

    step_id: str = Field(..., description="ID of dependent step")
    dependency_type: str = Field(
        default="sequential", description="Type of dependency (sequential, data, conditional)"
    )
    required: bool = Field(
        default=True, description="Whether dependency must complete successfully"
    )


# ============================================================================
# Success Criteria Models
# ============================================================================


class SuccessCriterion(BaseModel):
    """Individual success criterion for plan execution."""

    criterion_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique criterion ID"
    )
    description: str = Field(..., description="Human-readable criterion description")
    metric_name: str = Field(..., description="Name of metric to evaluate")
    operator: str = Field(
        ..., description="Comparison operator (gt, lt, eq, gte, lte, in, contains)"
    )
    threshold: Any = Field(..., description="Threshold value for comparison")
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relative weight of this criterion"
    )
    required: bool = Field(
        default=True, description="Whether this criterion must be met"
    )

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v: str) -> str:
        """Validate operator is supported."""
        valid_operators = {"gt", "lt", "eq", "gte", "lte", "in", "contains", "ne"}
        if v not in valid_operators:
            raise ValueError(
                f"Invalid operator '{v}'. Must be one of: {valid_operators}"
            )
        return v


class SuccessCriteria(BaseModel):
    """Collection of success criteria for plan evaluation."""

    criteria: list[SuccessCriterion] = Field(
        default_factory=list, description="List of success criteria"
    )
    aggregation_method: str = Field(
        default="weighted_average",
        description="Method to aggregate criteria (all, weighted_average, majority)",
    )
    minimum_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum aggregate score for success",
    )

    @field_validator('aggregation_method')
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        """Validate aggregation method."""
        valid_methods = {"all", "weighted_average", "majority", "any"}
        if v not in valid_methods:
            raise ValueError(
                f"Invalid aggregation method '{v}'. Must be one of: {valid_methods}"
            )
        return v


# ============================================================================
# Enhanced Plan Step Model
# ============================================================================


class EnhancedPlanStep(BaseModel):
    """
    Enhanced plan step with execution tracking and dependencies.

    Extends the basic PlanStep interface model with additional fields
    for status tracking, dependencies, tool requirements, and results.
    """

    # Core fields (from interface)
    step_id: str = Field(..., description="Unique identifier for this step")
    action: str = Field(..., description="Action to perform")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action"
    )

    # Enhanced fields
    status: StepStatus = Field(
        default=StepStatus.PENDING, description="Current step status"
    )
    dependencies: list[StepDependency] = Field(
        default_factory=list, description="Step dependencies"
    )
    tool_requirements: list[ToolRequirement] = Field(
        default_factory=list, description="Required tools for this step"
    )

    # Execution tracking
    started_at: str | None = Field(None, description="Execution start time")
    completed_at: str | None = Field(None, description="Execution completion time")
    duration_seconds: float | None = Field(None, description="Execution duration")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Results
    result: Any = Field(None, description="Step execution result")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional step metadata"
    )

    # Cost tracking
    estimated_cost: float = Field(default=0.0, description="Estimated execution cost")
    actual_cost: float | None = Field(None, description="Actual execution cost")

    def mark_started(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, result: Any = None) -> None:
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if result is not None:
            self.result = result
        if self.started_at:
            started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            completed = datetime.now(timezone.utc)
            self.duration_seconds = (completed - started).total_seconds()

    def mark_failed(self, error: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()


# ============================================================================
# Enhanced Execution Plan Model
# ============================================================================


class EnhancedExecutionPlan(BaseModel):
    """
    Enhanced execution plan with success criteria and iteration control.

    Extends the basic ExecutionPlan interface model with additional fields
    for success criteria, iteration limits, and comprehensive tracking.
    """

    # Core fields (from interface)
    plan_id: str = Field(..., description="Unique identifier for this plan")
    steps: list[EnhancedPlanStep] = Field(
        ..., description="Ordered list of execution steps"
    )
    total_estimated_cost: float = Field(
        default=0.0, description="Total estimated cost for entire plan"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional plan metadata"
    )

    # Enhanced fields
    status: PlanStatus = Field(
        default=PlanStatus.PENDING, description="Current plan status"
    )
    success_criteria: SuccessCriteria | None = Field(
        None, description="Success criteria for plan evaluation"
    )
    max_iterations: int = Field(
        default=10, ge=1, le=100, description="Maximum refinement iterations"
    )
    current_iteration: int = Field(
        default=0, description="Current iteration number"
    )

    # Execution tracking
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Plan creation timestamp",
    )
    started_at: str | None = Field(None, description="Execution start time")
    completed_at: str | None = Field(None, description="Execution completion time")
    duration_seconds: float | None = Field(None, description="Total execution duration")

    # Results
    final_result: Any = Field(None, description="Final execution result")
    error: str | None = Field(None, description="Error message if failed")
    actual_cost: float | None = Field(None, description="Actual total cost")

    # Context
    query: str | None = Field(None, description="Original query that generated this plan")
    parent_plan_id: str | None = Field(
        None, description="ID of parent plan if this is a refinement"
    )

    def mark_started(self) -> None:
        """Mark plan as started."""
        self.status = PlanStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, result: Any = None) -> None:
        """Mark plan as completed."""
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if result is not None:
            self.final_result = result
        if self.started_at:
            started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            completed = datetime.now(timezone.utc)
            self.duration_seconds = (completed - started).total_seconds()

    def mark_failed(self, error: str) -> None:
        """Mark plan as failed."""
        self.status = PlanStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def get_next_step(self) -> EnhancedPlanStep | None:
        """Get next pending step with satisfied dependencies."""
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_satisfied = True
            for dep in step.dependencies:
                dep_step = self.get_step_by_id(dep.step_id)
                if not dep_step:
                    deps_satisfied = False
                    break
                if dep.required and dep_step.status != StepStatus.COMPLETED:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                return step

        return None

    def get_step_by_id(self, step_id: str) -> EnhancedPlanStep | None:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None


# ============================================================================
# Module Transition Model
# ============================================================================


class ModuleTransition(BaseModel):
    """
    Tracks transitions between modules in the four-module flow.

    Records when control passes from one module to another, enabling
    analysis of execution flow and bottleneck identification.
    """

    transition_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique transition ID"
    )
    plan_id: str = Field(..., description="ID of execution plan")
    iteration: int = Field(..., description="Iteration number")

    # Transition details
    from_module: ModuleType = Field(..., description="Source module")
    to_module: ModuleType = Field(..., description="Destination module")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Transition timestamp",
    )

    # Transition reason
    reason: str = Field(..., description="Reason for transition")
    trigger: str | None = Field(None, description="Event that triggered transition")

    # Data transferred
    data: dict[str, Any] = Field(
        default_factory=dict, description="Data passed between modules"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional transition metadata"
    )

    # Performance metrics
    duration_in_from_module: float | None = Field(
        None, description="Time spent in source module (seconds)"
    )


# ============================================================================
# Enhanced Verification Result Model
# ============================================================================


class EnhancedVerificationResult(BaseModel):
    """
    Enhanced verification result with confidence scores and detailed feedback.

    Extends the basic VerificationResult interface model with additional
    fields for confidence tracking, verification levels, and detailed metrics.
    """

    # Core fields (from interface)
    valid: bool = Field(..., description="Whether results are valid")
    errors: list[str] = Field(
        default_factory=list, description="Validation errors found"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    feedback: str | None = Field(
        None, description="Feedback for improvement"
    )

    # Enhanced fields
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in validation (0.0-1.0)"
    )
    verification_level: VerificationLevel = Field(
        default=VerificationLevel.STANDARD,
        description="Level of verification applied",
    )

    # Detailed metrics
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Verification metrics"
    )
    checked_criteria: list[str] = Field(
        default_factory=list, description="Criteria that were checked"
    )
    passed_criteria: list[str] = Field(
        default_factory=list, description="Criteria that passed"
    )
    failed_criteria: list[str] = Field(
        default_factory=list, description="Criteria that failed"
    )

    # Timing
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Verification timestamp",
    )
    duration_seconds: float | None = Field(
        None, description="Verification duration"
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    suggested_refinements: dict[str, Any] = Field(
        default_factory=dict, description="Suggested plan refinements"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def calculate_success_rate(self) -> float:
        """Calculate success rate based on passed/checked criteria."""
        if not self.checked_criteria:
            return 1.0 if self.valid else 0.0
        return len(self.passed_criteria) / len(self.checked_criteria)
