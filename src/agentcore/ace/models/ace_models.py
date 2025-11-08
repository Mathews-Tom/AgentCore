"""
ACE Pydantic Models - COMPASS Enhanced

Data models for capability evaluation, fitness scoring, and recommendations.
Follows COMPASS ACE-4 specification for dynamic capability evaluation.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CapabilityType(str, Enum):
    """Types of agent capabilities."""

    API = "api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    SEARCH = "search"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


class TaskRequirement(BaseModel):
    """Task requirement specification."""

    requirement_id: str = Field(..., description="Unique requirement identifier")
    capability_type: CapabilityType = Field(
        ..., description="Required capability type"
    )
    capability_name: str = Field(..., description="Specific capability name")
    required: bool = Field(
        default=True, description="Whether this capability is mandatory"
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Importance weight (0-1)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional requirement metadata"
    )


class FitnessMetrics(BaseModel):
    """Detailed fitness metrics for a capability."""

    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    error_correlation: float = Field(
        ..., ge=0.0, le=1.0, description="Correlation with errors (0-1)"
    )
    usage_frequency: int = Field(..., ge=0, description="Number of times used")
    avg_execution_time_ms: float = Field(
        ..., ge=0.0, description="Average execution time in milliseconds"
    )
    resource_efficiency: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Resource usage efficiency (0-1)"
    )


class CapabilityFitness(BaseModel):
    """Capability fitness evaluation result (COMPASS ACE-4)."""

    capability_id: str = Field(..., description="Capability identifier")
    capability_name: str = Field(..., description="Capability display name")
    agent_id: str = Field(..., description="Agent identifier")
    task_type: str | None = Field(None, description="Task type if task-specific")

    # Fitness scoring
    fitness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall fitness score (0-1)"
    )
    coverage_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well capability covers task requirements (0-1)",
    )
    performance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Performance history score (0-1)",
    )

    # Detailed metrics
    metrics: FitnessMetrics = Field(..., description="Detailed fitness metrics")

    # Context
    sample_size: int = Field(..., ge=0, description="Number of executions analyzed")
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Evaluation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def is_fit(self) -> bool:
        """Check if capability meets fitness threshold (>=0.5)."""
        return self.fitness_score >= 0.5

    @property
    def fitness_level(self) -> str:
        """Get fitness level category."""
        if self.fitness_score >= 0.8:
            return "excellent"
        elif self.fitness_score >= 0.6:
            return "good"
        elif self.fitness_score >= 0.4:
            return "acceptable"
        else:
            return "poor"


class CapabilityGap(BaseModel):
    """Identified capability gap in agent's current capabilities."""

    gap_id: str = Field(default_factory=lambda: str(uuid4()), description="Gap identifier")
    required_capability: str = Field(..., description="Missing or underperforming capability")
    capability_type: CapabilityType = Field(..., description="Type of capability")
    current_fitness: float | None = Field(
        None, ge=0.0, le=1.0, description="Current fitness if capability exists"
    )
    required_fitness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Required fitness threshold"
    )
    impact: float = Field(
        ..., ge=0.0, le=1.0, description="Impact on task success (0-1)"
    )
    gap_severity: str = Field(
        ..., description="Severity: critical, high, medium, low"
    )
    mitigation_suggestion: str | None = Field(
        None, description="Suggestion for addressing gap"
    )

    @property
    def gap_size(self) -> float:
        """Calculate gap size."""
        current = self.current_fitness if self.current_fitness is not None else 0.0
        return max(0.0, self.required_fitness - current)


class CapabilityRecommendation(BaseModel):
    """Capability change recommendation (COMPASS ACE-4)."""

    recommendation_id: UUID = Field(
        default_factory=uuid4, description="Unique recommendation identifier"
    )
    agent_id: str = Field(..., description="Target agent identifier")
    task_id: UUID | None = Field(None, description="Related task ID if task-specific")
    task_type: str | None = Field(None, description="Task type for context")

    # Current state
    current_capabilities: list[str] = Field(
        ..., description="Agent's current capabilities"
    )
    underperforming_capabilities: list[str] = Field(
        default_factory=list, description="Capabilities with fitness < 0.5"
    )

    # Recommendations
    capabilities_to_add: list[str] = Field(
        default_factory=list, description="Recommended capabilities to add"
    )
    capabilities_to_remove: list[str] = Field(
        default_factory=list, description="Recommended capabilities to remove"
    )
    capabilities_to_upgrade: list[dict[str, str]] = Field(
        default_factory=list,
        description="Capabilities to upgrade with version info",
    )

    # Analysis
    identified_gaps: list[CapabilityGap] = Field(
        default_factory=list, description="Identified capability gaps"
    )
    fitness_scores: dict[str, float] = Field(
        default_factory=dict, description="Fitness scores for current capabilities"
    )
    alternatives_evaluated: dict[str, float] = Field(
        default_factory=dict,
        description="Alternative capabilities evaluated with scores",
    )

    # Recommendation metadata
    rationale: str = Field(..., description="Detailed rationale for recommendations")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in recommendation (0-1)"
    )
    expected_improvement: float = Field(
        ..., ge=0.0, le=1.0, description="Expected fitness improvement (0-1)"
    )
    risk_level: str = Field(
        default="low", description="Risk level: low, medium, high"
    )

    # Tracking
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Recommendation generation timestamp",
    )
    applied: bool = Field(default=False, description="Whether recommendation was applied")
    applied_at: datetime | None = Field(
        None, description="Application timestamp if applied"
    )
    effectiveness_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Post-application effectiveness (0-1)"
    )

    @property
    def has_critical_gaps(self) -> bool:
        """Check if there are critical capability gaps."""
        return any(gap.gap_severity == "critical" for gap in self.identified_gaps)

    @property
    def recommendation_count(self) -> int:
        """Total number of recommended changes."""
        return (
            len(self.capabilities_to_add)
            + len(self.capabilities_to_remove)
            + len(self.capabilities_to_upgrade)
        )
