"""Coordination Service Data Models

Models for Ripple Effect Protocol (REP) inspired coordination system.
Enables intelligent agent selection based on sensitivity signals (load, capacity, quality, cost).

Key Models:
- SignalType: Enum for signal categories
- SensitivitySignal: Agent capability/load signal with TTL
- AgentCoordinationState: Aggregated coordination state per agent
- CoordinationMetrics: Prometheus metrics for observability
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalType(str, Enum):
    """Types of sensitivity signals agents can report.

    Signal semantics:
    - LOAD: Current task queue depth (0.0=idle, 1.0=saturated)
    - CAPACITY: Available compute/memory resources (0.0=none, 1.0=full)
    - QUALITY: Service quality metric (0.0=poor, 1.0=excellent)
    - COST: Operational cost (0.0=expensive, 1.0=cheap)
    - LATENCY: Response time (0.0=slow, 1.0=fast)
    - AVAILABILITY: Service health (0.0=degraded, 1.0=healthy)
    """

    LOAD = "LOAD"
    CAPACITY = "CAPACITY"
    QUALITY = "QUALITY"
    COST = "COST"
    LATENCY = "LATENCY"
    AVAILABILITY = "AVAILABILITY"


class SensitivitySignal(BaseModel):
    """Sensitivity signal broadcast by agents for coordination.

    Agents report their current state via signals which are:
    - Normalized to 0.0-1.0 range
    - Time-limited with TTL (auto-expire)
    - Confidence-weighted for uncertainty
    - Traceable for debugging

    Attributes:
        signal_id: Unique identifier for this signal
        agent_id: Source agent identifier
        signal_type: Category of signal (load, capacity, etc.)
        value: Normalized signal value [0.0, 1.0]
        timestamp: UTC timestamp when signal generated
        ttl_seconds: Time-to-live before signal expires
        confidence: Confidence in signal accuracy [0.0, 1.0]
        trace_id: Optional distributed tracing ID
    """

    signal_id: UUID = Field(default_factory=uuid4, description="Unique signal identifier")
    agent_id: str = Field(..., min_length=1, description="Source agent ID")
    signal_type: SignalType = Field(..., description="Signal category")
    value: float = Field(..., ge=0.0, le=1.0, description="Normalized signal value [0.0, 1.0]")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signal generation timestamp (UTC)"
    )
    ttl_seconds: int = Field(..., gt=0, description="Time-to-live in seconds")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Signal confidence [0.0, 1.0]"
    )
    trace_id: str | None = Field(default=None, description="Distributed tracing ID")

    @field_validator("value")
    @classmethod
    def validate_value_range(cls, v: float) -> float:
        """Ensure value is in valid range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Signal value must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("ttl_seconds")
    @classmethod
    def validate_ttl_positive(cls, v: int) -> int:
        """Ensure TTL is positive."""
        if v <= 0:
            raise ValueError(f"TTL must be positive, got {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Ensure confidence is in valid range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    def is_expired(self, current_time: datetime | None = None) -> bool:
        """Check if signal has expired based on TTL.

        Args:
            current_time: Reference time (UTC), defaults to now

        Returns:
            True if signal expired, False otherwise
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        age_seconds = (current_time - self.timestamp).total_seconds()
        return age_seconds > self.ttl_seconds

    def decay_factor(self, current_time: datetime | None = None) -> float:
        """Calculate temporal decay factor for aging signals.

        Uses exponential decay: factor = e^(-age / ttl)

        This provides smooth degradation of signal confidence over time:
        - Age = 0: factor = 1.0 (full confidence)
        - Age = ttl: factor ≈ 0.368 (36.8% confidence)
        - Age > ttl: factor = 0.0 (expired, no confidence)

        Args:
            current_time: Reference time (UTC), defaults to now

        Returns:
            Decay factor [0.0, 1.0], 0.0 if expired
        """
        if self.is_expired(current_time):
            return 0.0

        if current_time is None:
            current_time = datetime.now(timezone.utc)

        age_seconds = (current_time - self.timestamp).total_seconds()

        # Exponential decay: e^(-age / ttl)
        import math
        decay = math.exp(-age_seconds / self.ttl_seconds)
        return max(0.0, min(1.0, decay))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signal_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "agent-001",
                "signal_type": "LOAD",
                "value": 0.75,
                "timestamp": "2025-01-15T10:30:00Z",
                "ttl_seconds": 60,
                "confidence": 0.95,
                "trace_id": "trace-abc123"
            }
        }
    )


class AgentCoordinationState(BaseModel):
    """Aggregated coordination state for a single agent.

    Maintains current signal values and computed scores for routing decisions.
    Scores are derived from active (non-expired) signals with temporal decay applied.

    Score Semantics:
    - load_score: Inverted load (1.0=idle, 0.0=overloaded)
    - capacity_score: Available resources (1.0=full capacity, 0.0=exhausted)
    - quality_score: Service quality (1.0=excellent, 0.0=poor)
    - cost_score: Cost efficiency (1.0=cheap, 0.0=expensive)
    - availability_score: Service health (1.0=healthy, 0.0=degraded)
    - routing_score: Composite weighted score for selection

    Attributes:
        agent_id: Target agent identifier
        signals: Active signals by type (latest only)
        load_score: Computed load score [0.0, 1.0]
        capacity_score: Computed capacity score [0.0, 1.0]
        quality_score: Computed quality score [0.0, 1.0]
        cost_score: Computed cost score [0.0, 1.0]
        availability_score: Computed availability score [0.0, 1.0]
        routing_score: Composite routing score [0.0, 1.0]
        last_updated: Last state update timestamp (UTC)
    """

    agent_id: str = Field(..., min_length=1, description="Agent identifier")
    signals: dict[SignalType, SensitivitySignal] = Field(
        default_factory=dict, description="Active signals by type"
    )

    # Individual scores (computed from signals)
    load_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Load score (inverted)")
    capacity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Capacity score")
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score")
    cost_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Cost score")
    availability_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability score")

    # Composite score for routing
    routing_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Composite routing score")

    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_id": "agent-001",
                "signals": {},
                "load_score": 0.8,
                "capacity_score": 0.7,
                "quality_score": 0.9,
                "cost_score": 0.6,
                "availability_score": 1.0,
                "routing_score": 0.8,
                "last_updated": "2025-01-15T10:30:00Z"
            }
        }
    )


class CoordinationMetrics(BaseModel):
    """Prometheus metrics for coordination service observability.

    Tracks operational metrics for monitoring and alerting:
    - Signal throughput and types
    - Routing selections and rationale
    - Performance characteristics
    - Coordination effectiveness vs baseline

    Attributes:
        total_signals: Total signals registered (counter)
        signals_by_type: Signal counts by SignalType (counter per type)
        total_selections: Total optimal agent selections (counter)
        average_selection_time_ms: Average selection latency (gauge)
        average_signal_age_seconds: Average age of signals used (gauge)
        coordination_score_avg: Average routing score across agents (gauge)
        expired_signals_cleaned: Total expired signals removed (counter)
        agents_tracked: Number of agents with coordination state (gauge)
    """

    total_signals: int = Field(default=0, ge=0, description="Total signals registered")
    signals_by_type: dict[SignalType, int] = Field(
        default_factory=dict, description="Signal counts by type"
    )
    total_selections: int = Field(default=0, ge=0, description="Total agent selections")
    average_selection_time_ms: float = Field(
        default=0.0, ge=0.0, description="Avg selection latency (ms)"
    )
    average_signal_age_seconds: float = Field(
        default=0.0, ge=0.0, description="Avg signal age (seconds)"
    )
    coordination_score_avg: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Avg routing score"
    )
    expired_signals_cleaned: int = Field(
        default=0, ge=0, description="Expired signals cleaned"
    )
    agents_tracked: int = Field(default=0, ge=0, description="Agents with coordination state")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_signals": 1500,
                "signals_by_type": {
                    "LOAD": 500,
                    "CAPACITY": 400,
                    "QUALITY": 300,
                    "COST": 200,
                    "AVAILABILITY": 100
                },
                "total_selections": 250,
                "average_selection_time_ms": 3.5,
                "average_signal_age_seconds": 25.0,
                "coordination_score_avg": 0.75,
                "expired_signals_cleaned": 50,
                "agents_tracked": 10
            }
        }
    )
