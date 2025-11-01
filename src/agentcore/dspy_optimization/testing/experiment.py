"""
Experiment design and management

Provides experiment configuration, group management, and result tracking
for A/B testing optimization validation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import OptimizationTarget, PerformanceMetrics


class ExperimentStatus(str, Enum):
    """Status of A/B test experiment"""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentGroup(str, Enum):
    """Experiment group designation"""

    CONTROL = "control"
    TREATMENT = "treatment"


class ExperimentConfig(BaseModel):
    """Configuration for A/B test experiment"""

    name: str
    description: str | None = None
    traffic_percentage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Percentage of traffic to treatment group",
    )
    min_samples_per_group: int = Field(
        default=100,
        description="Minimum samples required per group",
    )
    duration_hours: int = Field(
        default=24,
        description="Experiment duration in hours",
    )
    significance_threshold: float = Field(
        default=0.05,
        description="P-value threshold for significance",
    )
    min_improvement_threshold: float = Field(
        default=0.05,
        description="Minimum improvement required (5%)",
    )
    early_stopping_enabled: bool = Field(
        default=True,
        description="Enable early stopping on significance",
    )
    early_stopping_min_samples: int = Field(
        default=200,
        description="Minimum samples before early stopping",
    )


class ExperimentResult(BaseModel):
    """Result from experiment group"""

    group: ExperimentGroup
    sample_count: int
    metrics: PerformanceMetrics
    samples: list[dict[str, Any]] = Field(default_factory=list)


class Experiment(BaseModel):
    """A/B test experiment"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT
    control_version: str
    treatment_version: str
    results: dict[str, ExperimentResult] = Field(default_factory=dict)
    start_time: datetime | None = None
    end_time: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if experiment is active"""
        return self.status == ExperimentStatus.ACTIVE

    def is_completed(self) -> bool:
        """Check if experiment is completed"""
        return self.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)

    def get_duration_elapsed(self) -> timedelta | None:
        """Get elapsed duration since start"""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now(UTC)
        return end - self.start_time

    def get_control_result(self) -> ExperimentResult | None:
        """Get control group result"""
        return self.results.get(ExperimentGroup.CONTROL.value)

    def get_treatment_result(self) -> ExperimentResult | None:
        """Get treatment group result"""
        return self.results.get(ExperimentGroup.TREATMENT.value)

    def has_minimum_samples(self) -> bool:
        """Check if experiment has minimum required samples"""
        control = self.get_control_result()
        treatment = self.get_treatment_result()

        if not control or not treatment:
            return False

        return (
            control.sample_count >= self.config.min_samples_per_group
            and treatment.sample_count >= self.config.min_samples_per_group
        )

    def should_stop_early(self) -> bool:
        """Check if experiment should stop early"""
        if not self.config.early_stopping_enabled:
            return False

        control = self.get_control_result()
        treatment = self.get_treatment_result()

        if not control or not treatment:
            return False

        # Check if we have enough samples for early stopping
        total_samples = control.sample_count + treatment.sample_count
        return total_samples >= self.config.early_stopping_min_samples


class ExperimentManager:
    """
    Manages A/B test experiments

    Provides experiment lifecycle management including creation,
    activation, data collection, and completion.
    """

    def __init__(self) -> None:
        """Initialize experiment manager"""
        self._experiments: dict[str, Experiment] = {}

    async def create_experiment(
        self,
        target: OptimizationTarget,
        config: ExperimentConfig,
        control_version: str,
        treatment_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> Experiment:
        """
        Create new A/B test experiment

        Args:
            target: Optimization target
            config: Experiment configuration
            control_version: Control version identifier
            treatment_version: Treatment version identifier
            metadata: Optional experiment metadata

        Returns:
            Created experiment
        """
        experiment = Experiment(
            target=target,
            config=config,
            control_version=control_version,
            treatment_version=treatment_version,
            metadata=metadata or {},
        )

        self._experiments[experiment.id] = experiment
        return experiment

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """
        Start experiment

        Args:
            experiment_id: Experiment ID

        Returns:
            Started experiment

        Raises:
            ValueError: If experiment not found or already started
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment already started: {experiment_id}")

        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_time = datetime.now(UTC)
        experiment.updated_at = datetime.now(UTC)

        return experiment

    async def record_result(
        self,
        experiment_id: str,
        group: ExperimentGroup,
        sample: dict[str, Any],
    ) -> None:
        """
        Record result for experiment group

        Args:
            experiment_id: Experiment ID
            group: Experiment group
            sample: Performance sample

        Raises:
            ValueError: If experiment not found or not active
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if not experiment.is_active():
            raise ValueError(f"Experiment not active: {experiment_id}")

        # Get or create result
        result = experiment.results.get(group.value)
        if not result:
            result = ExperimentResult(
                group=group,
                sample_count=0,
                metrics=PerformanceMetrics(
                    success_rate=0.0,
                    avg_cost_per_task=0.0,
                    avg_latency_ms=0,
                    quality_score=0.0,
                ),
            )
            experiment.results[group.value] = result

        # Add sample
        result.samples.append(sample)
        result.sample_count += 1

        # Recalculate metrics
        result.metrics = self._calculate_metrics(result.samples)

        experiment.updated_at = datetime.now(UTC)

    async def pause_experiment(self, experiment_id: str) -> Experiment:
        """
        Pause experiment

        Args:
            experiment_id: Experiment ID

        Returns:
            Paused experiment

        Raises:
            ValueError: If experiment not found or not active
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if not experiment.is_active():
            raise ValueError(f"Experiment not active: {experiment_id}")

        experiment.status = ExperimentStatus.PAUSED
        experiment.updated_at = datetime.now(UTC)

        return experiment

    async def resume_experiment(self, experiment_id: str) -> Experiment:
        """
        Resume paused experiment

        Args:
            experiment_id: Experiment ID

        Returns:
            Resumed experiment

        Raises:
            ValueError: If experiment not found or not paused
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Experiment not paused: {experiment_id}")

        experiment.status = ExperimentStatus.ACTIVE
        experiment.updated_at = datetime.now(UTC)

        return experiment

    async def complete_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
    ) -> Experiment:
        """
        Complete experiment

        Args:
            experiment_id: Experiment ID
            status: Final status (COMPLETED or FAILED)

        Returns:
            Completed experiment

        Raises:
            ValueError: If experiment not found or invalid status
        """
        if status not in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            raise ValueError(f"Invalid completion status: {status}")

        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment.status = status
        experiment.end_time = datetime.now(UTC)
        experiment.updated_at = datetime.now(UTC)

        return experiment

    async def get_experiment(self, experiment_id: str) -> Experiment | None:
        """
        Get experiment by ID

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment or None if not found
        """
        return self._experiments.get(experiment_id)

    async def list_experiments(
        self,
        target: OptimizationTarget | None = None,
        status: ExperimentStatus | None = None,
    ) -> list[Experiment]:
        """
        List experiments with optional filters

        Args:
            target: Filter by optimization target
            status: Filter by experiment status

        Returns:
            List of matching experiments
        """
        experiments = list(self._experiments.values())

        if target:
            experiments = [
                e
                for e in experiments
                if e.target.type == target.type and e.target.id == target.id
            ]

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    def _calculate_metrics(self, samples: list[dict[str, Any]]) -> PerformanceMetrics:
        """
        Calculate aggregate metrics from samples

        Args:
            samples: Performance samples

        Returns:
            Aggregated metrics
        """
        if not samples:
            return PerformanceMetrics(
                success_rate=0.0,
                avg_cost_per_task=0.0,
                avg_latency_ms=0,
                quality_score=0.0,
            )

        success_rates = [s.get("success_rate", 0.0) for s in samples]
        costs = [s.get("avg_cost_per_task", 0.0) for s in samples]
        latencies = [s.get("avg_latency_ms", 0) for s in samples]
        quality_scores = [s.get("quality_score", 0.0) for s in samples]

        return PerformanceMetrics(
            success_rate=sum(success_rates) / len(success_rates),
            avg_cost_per_task=sum(costs) / len(costs),
            avg_latency_ms=int(sum(latencies) / len(latencies)),
            quality_score=sum(quality_scores) / len(quality_scores),
        )
