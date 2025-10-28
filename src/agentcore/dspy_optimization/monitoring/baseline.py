"""
Baseline performance measurement service

Measures and tracks baseline performance metrics for optimization targets,
enabling accurate improvement calculations and validation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    PerformanceMetrics,
    OptimizationTarget,
)


class BaselineConfig(BaseModel):
    """Configuration for baseline measurement"""

    measurement_window_hours: int = Field(
        default=24,
        description="Time window for baseline measurement in hours",
    )
    min_samples: int = Field(
        default=100,
        description="Minimum number of samples required for baseline",
    )
    max_samples: int = Field(
        default=10000,
        description="Maximum number of samples to collect",
    )
    update_frequency_hours: int = Field(
        default=168,
        description="Frequency to update baseline (default: weekly)",
    )


class BaselineMeasurement(BaseModel):
    """Baseline measurement record"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    metrics: PerformanceMetrics
    sample_count: int
    measurement_start: datetime
    measurement_end: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_valid: bool = True


class BaselineService:
    """
    Service for baseline performance measurement

    Manages baseline measurement collection, validation, and retrieval
    for optimization targets. Ensures accurate baseline data for
    calculating performance improvements.
    """

    def __init__(self, config: BaselineConfig | None = None) -> None:
        """
        Initialize baseline service

        Args:
            config: Baseline measurement configuration
        """
        self.config = config or BaselineConfig()
        self._baselines: dict[str, BaselineMeasurement] = {}

    async def measure_baseline(
        self,
        target: OptimizationTarget,
        performance_samples: list[dict[str, Any]],
    ) -> BaselineMeasurement:
        """
        Measure baseline performance from samples

        Args:
            target: Optimization target
            performance_samples: List of performance measurements

        Returns:
            Baseline measurement

        Raises:
            ValueError: If insufficient samples provided
        """
        if len(performance_samples) < self.config.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(performance_samples)} < "
                f"{self.config.min_samples}"
            )

        # Limit to max samples
        samples = performance_samples[: self.config.max_samples]

        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(samples)

        # Create baseline measurement
        now = datetime.utcnow()
        measurement = BaselineMeasurement(
            target=target,
            metrics=metrics,
            sample_count=len(samples),
            measurement_start=now - timedelta(hours=self.config.measurement_window_hours),
            measurement_end=now,
        )

        # Store baseline
        baseline_key = self._get_baseline_key(target)
        self._baselines[baseline_key] = measurement

        return measurement

    async def get_baseline(
        self,
        target: OptimizationTarget,
    ) -> BaselineMeasurement | None:
        """
        Get baseline measurement for target

        Args:
            target: Optimization target

        Returns:
            Baseline measurement or None if not found
        """
        baseline_key = self._get_baseline_key(target)
        return self._baselines.get(baseline_key)

    async def update_baseline(
        self,
        target: OptimizationTarget,
        performance_samples: list[dict[str, Any]],
    ) -> BaselineMeasurement:
        """
        Update existing baseline with new samples

        Args:
            target: Optimization target
            performance_samples: New performance measurements

        Returns:
            Updated baseline measurement
        """
        # Check if baseline needs update
        baseline = await self.get_baseline(target)

        if baseline:
            time_since_update = datetime.utcnow() - baseline.created_at
            if time_since_update.total_seconds() < (
                self.config.update_frequency_hours * 3600
            ):
                # Baseline is still fresh
                return baseline

        # Measure new baseline
        return await self.measure_baseline(target, performance_samples)

    async def invalidate_baseline(self, target: OptimizationTarget) -> None:
        """
        Invalidate baseline for target

        Args:
            target: Optimization target
        """
        baseline = await self.get_baseline(target)
        if baseline:
            baseline.is_valid = False

    async def is_baseline_valid(self, target: OptimizationTarget) -> bool:
        """
        Check if baseline is valid and current

        Args:
            target: Optimization target

        Returns:
            True if baseline is valid
        """
        baseline = await self.get_baseline(target)

        if not baseline or not baseline.is_valid:
            return False

        # Check if baseline is expired
        time_since_measurement = datetime.utcnow() - baseline.created_at
        if time_since_measurement.total_seconds() > (
            self.config.update_frequency_hours * 3600
        ):
            return False

        return True

    def _calculate_aggregate_metrics(
        self,
        samples: list[dict[str, Any]],
    ) -> PerformanceMetrics:
        """
        Calculate aggregate metrics from samples

        Args:
            samples: Performance measurement samples

        Returns:
            Aggregated performance metrics
        """
        # Extract metric values from samples
        success_rates = [s.get("success_rate", 0.0) for s in samples]
        costs = [s.get("avg_cost_per_task", 0.0) for s in samples]
        latencies = [s.get("avg_latency_ms", 0) for s in samples]
        quality_scores = [s.get("quality_score", 0.0) for s in samples]

        # Calculate averages
        avg_success_rate = sum(success_rates) / len(success_rates)
        avg_cost = sum(costs) / len(costs)
        avg_latency = int(sum(latencies) / len(latencies))
        avg_quality = sum(quality_scores) / len(quality_scores)

        return PerformanceMetrics(
            success_rate=avg_success_rate,
            avg_cost_per_task=avg_cost,
            avg_latency_ms=avg_latency,
            quality_score=avg_quality,
        )

    def _get_baseline_key(self, target: OptimizationTarget) -> str:
        """
        Get baseline storage key for target

        Args:
            target: Optimization target

        Returns:
            Baseline key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
