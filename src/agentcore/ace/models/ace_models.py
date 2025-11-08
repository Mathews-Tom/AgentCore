"""
ACE (Agentic Context Engineering) Pydantic Models

COMPASS-enhanced models for context evolution and performance monitoring.
Implements self-supervised context improvement for long-running agents.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvolutionStatusType(str, Enum):
    """Evolution processing status."""

    IDLE = "idle"
    PROCESSING = "processing"
    FAILED = "failed"


class ContextPlaybook(BaseModel):
    """Agent context playbook with versioning.

    Stores agent-specific context that evolves over time based on
    LLM-generated improvements from execution traces.
    """

    playbook_id: UUID = Field(default_factory=uuid4, description="Unique playbook ID")
    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    context: dict[str, Any] = Field(..., description="Context playbook data (JSONB)")
    version: int = Field(default=1, description="Playbook version number", ge=1)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (JSONB)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "playbook_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "agent-001",
                "context": {
                    "goals": ["optimize performance"],
                    "constraints": ["max_tokens: 4096"],
                    "preferences": {"temperature": 0.7},
                },
                "version": 1,
                "metadata": {"source": "initial_config"},
            }
        }
    )

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate context is not empty."""
        if not v:
            raise ValueError("Context cannot be empty")
        return v


class ContextDelta(BaseModel):
    """LLM-generated context improvement suggestion.

    Represents a proposed change to agent context based on
    execution trace analysis.
    """

    delta_id: UUID = Field(default_factory=uuid4, description="Unique delta ID")
    playbook_id: UUID = Field(..., description="Target playbook ID")
    changes: dict[str, Any] = Field(..., description="Proposed changes (JSONB)")
    confidence: float = Field(
        ..., description="Confidence score (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(..., description="LLM reasoning for changes", min_length=1)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Generation timestamp",
    )
    applied: bool = Field(default=False, description="Whether delta was applied")
    applied_at: datetime | None = Field(None, description="Application timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "delta_id": "660e8400-e29b-41d4-a716-446655440001",
                "playbook_id": "550e8400-e29b-41d4-a716-446655440000",
                "changes": {"preferences.temperature": 0.8},
                "confidence": 0.85,
                "reasoning": "Based on traces #1-5, increasing temperature improves creativity without sacrificing accuracy.",
                "applied": False,
            }
        }
    )

    @field_validator("changes")
    @classmethod
    def validate_changes(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate changes is not empty."""
        if not v:
            raise ValueError("Changes cannot be empty")
        return v

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        """Validate reasoning has minimum length."""
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters")
        return v.strip()


class ExecutionTrace(BaseModel):
    """Agent execution performance trace.

    Captures metrics from agent task execution for LLM analysis
    and context evolution.
    """

    trace_id: UUID = Field(default_factory=uuid4, description="Unique trace ID")
    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    task_id: str | None = Field(None, description="Task identifier", max_length=255)
    execution_time: float = Field(..., description="Execution time in seconds", ge=0.0)
    success: bool = Field(..., description="Whether execution succeeded")
    output_quality: float | None = Field(
        None, description="Output quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    error_message: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional trace metadata (JSONB)"
    )
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Capture timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "770e8400-e29b-41d4-a716-446655440002",
                "agent_id": "agent-001",
                "task_id": "task-123",
                "execution_time": 2.5,
                "success": True,
                "output_quality": 0.92,
                "metadata": {"tokens_used": 1024},
            }
        }
    )

    @model_validator(mode="after")
    def validate_error_message(self) -> "ExecutionTrace":
        """Validate error message is present for failed executions."""
        if self.success is False and not self.error_message:
            raise ValueError("Error message required for failed executions")
        return self


class EvolutionStatus(BaseModel):
    """Agent evolution processing status.

    Tracks evolution progress, costs, and state per agent.
    """

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    last_evolution: datetime | None = Field(None, description="Last evolution timestamp")
    pending_traces: int = Field(default=0, description="Pending trace count", ge=0)
    deltas_generated: int = Field(default=0, description="Total deltas generated", ge=0)
    deltas_applied: int = Field(default=0, description="Total deltas applied", ge=0)
    total_cost: float = Field(default=0.0, description="Total LLM cost (USD)", ge=0.0)
    status: EvolutionStatusType = Field(
        default=EvolutionStatusType.IDLE, description="Current status"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_id": "agent-001",
                "last_evolution": "2025-11-08T02:00:00Z",
                "pending_traces": 15,
                "deltas_generated": 10,
                "deltas_applied": 7,
                "total_cost": 0.25,
                "status": "idle",
            }
        }
    )

    @field_validator("deltas_applied")
    @classmethod
    def validate_deltas_applied(cls, v: int, info: Any) -> int:
        """Validate applied deltas <= generated deltas."""
        generated = info.data.get("deltas_generated", 0)
        if v > generated:
            raise ValueError("Applied deltas cannot exceed generated deltas")
        return v


# Request/Response Models for API


class CreatePlaybookRequest(BaseModel):
    """Request to create a new context playbook."""

    agent_id: str = Field(..., description="Agent identifier")
    initial_context: dict[str, Any] = Field(..., description="Initial context data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class CreatePlaybookResponse(BaseModel):
    """Response after creating a playbook."""

    playbook: ContextPlaybook
    message: str = Field(default="Playbook created successfully")


class PlaybookCreateRequest(BaseModel):
    """Request to create a new playbook."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    context: dict[str, Any] = Field(..., description="Initial context data")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class PlaybookUpdateRequest(BaseModel):
    """Request to update playbook context."""

    context: dict[str, Any] = Field(..., description="Updated context data")


class PlaybookResponse(BaseModel):
    """Response containing playbook data."""

    playbook_id: UUID
    agent_id: str
    context: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class ApplyDeltaRequest(BaseModel):
    """Request to apply a context delta."""

    delta_id: UUID = Field(..., description="Delta to apply")
    force: bool = Field(default=False, description="Force apply regardless of confidence")


class ApplyDeltaResponse(BaseModel):
    """Response after applying a delta."""

    playbook: ContextPlaybook
    delta: ContextDelta
    message: str = Field(default="Delta applied successfully")


class CaptureTraceRequest(BaseModel):
    """Request to capture an execution trace."""

    agent_id: str = Field(..., description="Agent identifier")
    task_id: str | None = Field(None, description="Task identifier")
    execution_time: float = Field(..., description="Execution time (seconds)")
    success: bool = Field(..., description="Execution success")
    output_quality: float | None = Field(None, description="Output quality score")
    error_message: str | None = Field(None, description="Error message")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class CaptureTraceResponse(BaseModel):
    """Response after capturing a trace."""

    trace: ExecutionTrace
    message: str = Field(default="Trace captured successfully")


class TriggerEvolutionRequest(BaseModel):
    """Request to trigger context evolution."""

    agent_id: str = Field(..., description="Agent identifier")
    force: bool = Field(default=False, description="Force evolution even if threshold not met")


class TriggerEvolutionResponse(BaseModel):
    """Response after triggering evolution."""

    status: EvolutionStatus
    deltas_generated: int = Field(..., description="Number of new deltas generated")
    message: str = Field(default="Evolution triggered successfully")


# COMPASS-Enhanced Models (Performance Monitoring)


class PerformanceMetrics(BaseModel):
    """Stage-aware performance metrics (COMPASS ACE-1).

    Tracks agent performance across reasoning stages (planning, execution,
    reflection, verification) with baseline comparison capabilities.
    """

    metric_id: UUID = Field(default_factory=uuid4, description="Unique metric ID")
    task_id: UUID = Field(..., description="Task identifier")
    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    stage: str = Field(..., description="Reasoning stage", min_length=1, max_length=50)

    # Stage-specific metrics
    stage_success_rate: float = Field(..., description="Stage success rate (0-1)", ge=0.0, le=1.0)
    stage_error_rate: float = Field(..., description="Stage error rate (0-1)", ge=0.0, le=1.0)
    stage_duration_ms: int = Field(..., description="Stage duration in milliseconds", ge=0)
    stage_action_count: int = Field(..., description="Number of actions in stage", ge=0)

    # Cross-stage metrics
    overall_progress_velocity: float = Field(
        ..., description="Actions per minute", ge=0.0
    )
    error_accumulation_rate: float = Field(
        ..., description="Errors per stage", ge=0.0
    )
    context_staleness_score: float = Field(
        ..., description="Context staleness (0-1, higher = staler)", ge=0.0, le=1.0
    )
    intervention_effectiveness: float | None = Field(
        None, description="Effectiveness of last intervention (0-1)", ge=0.0, le=1.0
    )

    # Baseline comparison
    baseline_delta: dict[str, float] = Field(
        default_factory=dict,
        description="Deviation from baseline (metric_name -> delta)",
    )

    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Recording timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metric_id": "880e8400-e29b-41d4-a716-446655440003",
                "task_id": "990e8400-e29b-41d4-a716-446655440004",
                "agent_id": "agent-001",
                "stage": "execution",
                "stage_success_rate": 0.85,
                "stage_error_rate": 0.15,
                "stage_duration_ms": 2500,
                "stage_action_count": 12,
                "overall_progress_velocity": 4.8,
                "error_accumulation_rate": 0.3,
                "context_staleness_score": 0.2,
                "intervention_effectiveness": 0.75,
                "baseline_delta": {"stage_success_rate": -0.05, "error_rate": 0.03},
            }
        }
    )

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        """Validate stage is one of the allowed values."""
        allowed_stages = {"planning", "execution", "reflection", "verification"}
        if v not in allowed_stages:
            raise ValueError(
                f"Stage must be one of {allowed_stages}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_error_rates(self) -> "PerformanceMetrics":
        """Validate success and error rates sum to <= 1.0."""
        total_rate = self.stage_success_rate + self.stage_error_rate
        if total_rate > 1.0:
            raise ValueError(
                f"Success rate ({self.stage_success_rate}) + error rate "
                f"({self.stage_error_rate}) cannot exceed 1.0 (got {total_rate})"
            )
        return self


class PerformanceBaseline(BaseModel):
    """Performance baseline for comparison (COMPASS ACE-1).

    Stores rolling baseline metrics for detecting performance degradation.
    """

    baseline_id: UUID = Field(default_factory=uuid4, description="Unique baseline ID")
    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    stage: str = Field(..., description="Reasoning stage", min_length=1, max_length=50)
    task_type: str | None = Field(None, description="Task type (optional)", max_length=100)

    # Baseline metrics (mean values)
    mean_success_rate: float = Field(..., description="Mean success rate", ge=0.0, le=1.0)
    mean_error_rate: float = Field(..., description="Mean error rate", ge=0.0, le=1.0)
    mean_duration_ms: float = Field(..., description="Mean duration in ms", ge=0.0)
    mean_action_count: float = Field(..., description="Mean action count", ge=0.0)

    # Statistical measures
    std_dev: dict[str, float] = Field(
        default_factory=dict,
        description="Standard deviations (metric_name -> std_dev)",
    )
    confidence_interval: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="95% confidence intervals (metric_name -> (low, high))",
    )

    # Metadata
    sample_size: int = Field(..., description="Number of samples in baseline", ge=1)
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "baseline_id": "aa0e8400-e29b-41d4-a716-446655440005",
                "agent_id": "agent-001",
                "stage": "execution",
                "task_type": "data_analysis",
                "mean_success_rate": 0.90,
                "mean_error_rate": 0.10,
                "mean_duration_ms": 2000.0,
                "mean_action_count": 10.0,
                "std_dev": {"success_rate": 0.05, "duration_ms": 500.0},
                "confidence_interval": {"success_rate": (0.85, 0.95)},
                "sample_size": 50,
            }
        }
    )

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        """Validate stage is one of the allowed values."""
        allowed_stages = {"planning", "execution", "reflection", "verification"}
        if v not in allowed_stages:
            raise ValueError(
                f"Stage must be one of {allowed_stages}, got '{v}'"
            )
        return v


# COMPASS-Enhanced Models (Strategic Intervention - ACE-2)


class TriggerType(str, Enum):
    """Intervention trigger types (COMPASS ACE-2)."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_ACCUMULATION = "error_accumulation"
    CONTEXT_STALENESS = "context_staleness"
    CAPABILITY_MISMATCH = "capability_mismatch"


class InterventionType(str, Enum):
    """Intervention action types (COMPASS ACE-2)."""

    CONTEXT_REFRESH = "context_refresh"
    REPLAN = "replan"
    REFLECT = "reflect"
    CAPABILITY_SWITCH = "capability_switch"


class ExecutionStatus(str, Enum):
    """Intervention execution status."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"


class TriggerSignal(BaseModel):
    """Intervention trigger signal (COMPASS ACE-2).

    Represents a detected signal that may trigger strategic intervention.
    Used by TriggerDetector to communicate detected conditions.
    """

    trigger_type: TriggerType = Field(..., description="Type of trigger signal")
    signals: list[str] = Field(
        ..., description="Specific signals detected", min_length=1
    )
    rationale: str = Field(
        ..., description="Human-readable rationale for trigger", min_length=10
    )
    confidence: float = Field(
        ..., description="Detection confidence (0-1)", ge=0.0, le=1.0
    )
    metric_values: dict[str, float] = Field(
        default_factory=dict,
        description="Metric values that triggered detection",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Detection timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trigger_type": "performance_degradation",
                "signals": ["velocity_drop_50pct", "error_rate_2x"],
                "rationale": "Task velocity dropped 50% below baseline (0.6 -> 0.3 actions/min) and error rate increased 2x (0.1 -> 0.2)",
                "confidence": 0.92,
                "metric_values": {
                    "velocity_ratio": 0.5,
                    "error_rate_ratio": 2.0,
                    "baseline_velocity": 0.6,
                    "current_velocity": 0.3,
                },
            }
        }
    )

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Validate rationale has minimum length."""
        if len(v.strip()) < 10:
            raise ValueError("Trigger rationale must be at least 10 characters")
        return v.strip()

    @field_validator("signals")
    @classmethod
    def validate_signals(cls, v: list[str]) -> list[str]:
        """Validate signals list is not empty."""
        if not v:
            raise ValueError("At least one signal required")
        return v


class InterventionRecord(BaseModel):
    """Record of strategic intervention (COMPASS ACE-2).

    Tracks intervention trigger, decision, execution, and outcome
    for meta-learning and effectiveness analysis.
    """

    intervention_id: UUID = Field(default_factory=uuid4, description="Unique intervention ID")
    task_id: UUID = Field(..., description="Task identifier")
    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)

    # Trigger information
    trigger_type: TriggerType = Field(..., description="Type of trigger signal")
    trigger_signals: list[str] = Field(
        ..., description="Specific signals that triggered intervention", min_length=1
    )
    trigger_metric_id: UUID | None = Field(
        None, description="Performance metric that triggered intervention"
    )

    # Decision information
    intervention_type: InterventionType = Field(..., description="Type of intervention to execute")
    intervention_rationale: str = Field(
        ..., description="Reasoning for intervention decision", min_length=10
    )
    decision_confidence: float = Field(
        ..., description="Confidence in decision (0-1)", ge=0.0, le=1.0
    )

    # Execution information
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Execution timestamp",
    )
    execution_duration_ms: int = Field(
        default=0, description="Execution duration in milliseconds", ge=0
    )
    execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING, description="Execution status"
    )
    execution_error: str | None = Field(None, description="Error message if execution failed")

    # Outcome tracking
    pre_metric_id: UUID | None = Field(
        None, description="Performance metric before intervention"
    )
    post_metric_id: UUID | None = Field(
        None, description="Performance metric after intervention"
    )
    effectiveness_delta: float | None = Field(
        None, description="Improvement score (-1 to 1)", ge=-1.0, le=1.0
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intervention_id": "bb0e8400-e29b-41d4-a716-446655440006",
                "task_id": "cc0e8400-e29b-41d4-a716-446655440007",
                "agent_id": "agent-001",
                "trigger_type": "performance_degradation",
                "trigger_signals": ["velocity_drop_50pct", "error_rate_2x"],
                "intervention_type": "replan",
                "intervention_rationale": "Task velocity dropped 50% below baseline with 2x error rate increase",
                "decision_confidence": 0.92,
                "execution_status": "success",
                "effectiveness_delta": 0.35,
            }
        }
    )

    @field_validator("intervention_rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Validate rationale has minimum length."""
        if len(v.strip()) < 10:
            raise ValueError("Intervention rationale must be at least 10 characters")
        return v.strip()

    @field_validator("trigger_signals")
    @classmethod
    def validate_signals(cls, v: list[str]) -> list[str]:
        """Validate trigger signals list is not empty."""
        if not v:
            raise ValueError("At least one trigger signal required")
        return v


class StrategicContext(BaseModel):
    """Strategic context for intervention decision making (COMPASS ACE-2).

    Provides context from COMPASS stages for informed intervention decisions.
    Used by DecisionMaker to select appropriate intervention type.
    """

    relevant_stage_summaries: list[str] = Field(
        default_factory=list,
        description="Summaries from relevant COMPASS stages",
    )
    critical_facts: list[str] = Field(
        default_factory=list,
        description="Critical facts extracted from context",
    )
    error_patterns: list[str] = Field(
        default_factory=list,
        description="Error patterns identified in execution",
    )
    successful_patterns: list[str] = Field(
        default_factory=list,
        description="Successful patterns identified in execution",
    )
    context_health_score: float = Field(
        ..., description="Context health score (0-1, higher is better)", ge=0.0, le=1.0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relevant_stage_summaries": [
                    "Planning stage completed with 85% confidence",
                    "Execution stage showing 2x error rate increase",
                ],
                "critical_facts": [
                    "Task requires data transformation capabilities",
                    "Agent has limited data processing tools",
                ],
                "error_patterns": [
                    "Repeated file parsing failures",
                    "Memory retrieval returning stale results",
                ],
                "successful_patterns": [
                    "API calls completing successfully",
                    "Context refresh improved performance",
                ],
                "context_health_score": 0.65,
            }
        }
    )

    @field_validator("context_health_score")
    @classmethod
    def validate_health_score(cls, v: float) -> float:
        """Validate health score is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"context_health_score must be in [0, 1], got {v}")
        return v


class InterventionDecision(BaseModel):
    """Intervention decision from DecisionMaker (COMPASS ACE-2).

    Represents the LLM-generated decision for which intervention to execute
    based on trigger signals and strategic context.
    """

    intervention_type: InterventionType = Field(
        ..., description="Selected intervention type"
    )
    rationale: str = Field(
        ..., description="Reasoning for intervention decision", min_length=10
    )
    confidence: float = Field(
        ..., description="Decision confidence (0-1)", ge=0.0, le=1.0
    )
    expected_impact: str = Field(
        ..., description="Predicted outcome description", min_length=10
    )
    alternative_interventions: list[str] = Field(
        default_factory=list,
        description="Considered alternative interventions",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional decision context",
    )
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Decision timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intervention_type": "replan",
                "rationale": "Task velocity dropped 50% with 2x error rate increase. Replanning will help agent reassess approach and break down tasks differently.",
                "confidence": 0.88,
                "expected_impact": "Velocity should return to baseline levels within 2-3 stages after replanning. Error rate should drop to <15%.",
                "alternative_interventions": [
                    "reflect - considered but less urgent than replan",
                    "context_refresh - useful but won't address root planning issues",
                ],
                "metadata": {
                    "trigger_confidence": 0.92,
                    "context_health": 0.65,
                    "decision_latency_ms": 145,
                },
            }
        }
    )

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Validate rationale has minimum length."""
        if len(v.strip()) < 10:
            raise ValueError("Decision rationale must be at least 10 characters")
        return v.strip()

    @field_validator("expected_impact")
    @classmethod
    def validate_expected_impact(cls, v: str) -> str:
        """Validate expected impact has minimum length."""
        if len(v.strip()) < 10:
            raise ValueError("Expected impact must be at least 10 characters")
        return v.strip()
