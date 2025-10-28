"""
Data models for DSPy optimization engine
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class OptimizationTargetType(str, Enum):
    """Types of optimization targets"""

    AGENT = "agent"
    WORKFLOW = "workflow"
    COMPONENT = "component"


class OptimizationScope(str, Enum):
    """Scope of optimization"""

    INDIVIDUAL = "individual"
    POPULATION = "population"
    CROSS_DOMAIN = "cross_domain"


class OptimizationStatus(str, Enum):
    """Status of optimization"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MetricType(str, Enum):
    """Types of optimization metrics"""

    SUCCESS_RATE = "success_rate"
    COST_EFFICIENCY = "cost_efficiency"
    LATENCY = "latency"
    QUALITY_SCORE = "quality_score"


class OptimizationTarget(BaseModel):
    """Target for optimization"""

    type: OptimizationTargetType
    id: str
    scope: OptimizationScope = OptimizationScope.INDIVIDUAL


class OptimizationObjective(BaseModel):
    """Optimization objective with target and weight"""

    metric: MetricType
    target_value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class OptimizationConstraints(BaseModel):
    """Constraints for optimization"""

    max_optimization_time: int = Field(default=7200, description="Max time in seconds")
    min_improvement_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Minimum improvement required"
    )
    max_resource_usage: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Max resource usage percentage"
    )


class PerformanceMetrics(BaseModel):
    """Performance metrics for baseline and optimized versions"""

    success_rate: float = Field(ge=0.0, le=1.0)
    avg_cost_per_task: float = Field(ge=0.0)
    avg_latency_ms: int = Field(ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class OptimizationDetails(BaseModel):
    """Details about the optimization process"""

    algorithm_used: str
    iterations: int
    key_improvements: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class OptimizationRequest(BaseModel):
    """Request to start optimization"""

    target: OptimizationTarget
    objectives: list[OptimizationObjective]
    algorithms: list[str] = Field(default=["miprov2", "gepa"])
    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints
    )


class OptimizationResult(BaseModel):
    """Result of optimization"""

    optimization_id: str = Field(default_factory=lambda: str(uuid4()))
    status: OptimizationStatus
    baseline_performance: PerformanceMetrics | None = None
    optimized_performance: PerformanceMetrics | None = None
    improvement_percentage: float = 0.0
    statistical_significance: float = 0.0
    optimization_details: OptimizationDetails | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error_message: str | None = None
