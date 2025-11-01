"""
Performance drift detection

Detects performance degradation and distribution shifts that indicate
the need for model retraining.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import OptimizationTarget, PerformanceMetrics


class DriftStatus(str, Enum):
    """Status of drift detection"""

    NO_DRIFT = "no_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CRITICAL_DRIFT = "critical_drift"


class DriftConfig(BaseModel):
    """Configuration for drift detection"""

    performance_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Performance degradation threshold (10%)",
    )
    critical_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Critical degradation threshold (20%)",
    )
    window_size: int = Field(
        default=100,
        description="Sample window size for comparison",
    )
    comparison_window_hours: int = Field(
        default=24,
        description="Time window for baseline comparison",
    )
    min_samples: int = Field(
        default=50,
        description="Minimum samples required for drift detection",
    )


class DriftResult(BaseModel):
    """Result of drift detection"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    status: DriftStatus
    baseline_metrics: PerformanceMetrics
    current_metrics: PerformanceMetrics
    degradation_percentage: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sample_count: int
    details: dict[str, Any] = Field(default_factory=dict)


class DriftDetector:
    """
    Detects performance drift and distribution shifts

    Monitors performance metrics over time to identify degradation
    requiring model retraining.

    Key features:
    - Performance degradation detection
    - Distribution shift detection
    - Configurable thresholds
    - Historical comparison
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        """
        Initialize drift detector

        Args:
            config: Drift detection configuration
        """
        self.config = config or DriftConfig()
        self._baseline_metrics: dict[str, PerformanceMetrics] = {}
        self._recent_metrics: dict[str, list[tuple[datetime, PerformanceMetrics]]] = {}

    async def set_baseline(
        self,
        target: OptimizationTarget,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Set baseline metrics for target

        Args:
            target: Optimization target
            metrics: Baseline performance metrics
        """
        target_key = self._get_target_key(target)
        self._baseline_metrics[target_key] = metrics

    async def record_metrics(
        self,
        target: OptimizationTarget,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Record metrics for drift tracking

        Args:
            target: Optimization target
            metrics: Current performance metrics
        """
        target_key = self._get_target_key(target)

        if target_key not in self._recent_metrics:
            self._recent_metrics[target_key] = []

        self._recent_metrics[target_key].append((datetime.now(UTC), metrics))

        # Enforce window size limit
        await self._enforce_window_size(target_key)

    async def check_drift(
        self,
        target: OptimizationTarget,
    ) -> DriftResult | None:
        """
        Check for performance drift

        Args:
            target: Optimization target

        Returns:
            DriftResult if drift detected, None otherwise
        """
        target_key = self._get_target_key(target)

        # Get baseline
        baseline = self._baseline_metrics.get(target_key)
        if not baseline:
            return None

        # Get recent metrics
        recent = self._recent_metrics.get(target_key, [])
        if len(recent) < self.config.min_samples:
            return None

        # Calculate current average metrics
        current = await self._calculate_average_metrics(recent)

        # Calculate degradation
        degradation = await self._calculate_degradation(baseline, current)

        # Determine drift status
        status = self._determine_drift_status(degradation)

        # Create result
        result = DriftResult(
            target=target,
            status=status,
            baseline_metrics=baseline,
            current_metrics=current,
            degradation_percentage=degradation,
            sample_count=len(recent),
            details={
                "baseline_success_rate": baseline.success_rate,
                "current_success_rate": current.success_rate,
                "baseline_quality_score": baseline.quality_score,
                "current_quality_score": current.quality_score,
            },
        )

        return result if status != DriftStatus.NO_DRIFT else None

    async def check_distribution_shift(
        self,
        target: OptimizationTarget,
    ) -> bool:
        """
        Check for distribution shift using statistical tests

        Args:
            target: Optimization target

        Returns:
            True if distribution shift detected
        """
        target_key = self._get_target_key(target)
        recent = self._recent_metrics.get(target_key, [])

        if len(recent) < self.config.min_samples * 2:
            return False

        # Split into two windows
        mid = len(recent) // 2
        window1 = recent[:mid]
        window2 = recent[mid:]

        # Calculate metrics for each window
        metrics1 = await self._calculate_average_metrics(window1)
        metrics2 = await self._calculate_average_metrics(window2)

        # Check for significant difference
        # Simplified: check if difference exceeds threshold
        success_rate_diff = abs(metrics1.success_rate - metrics2.success_rate)
        quality_diff = abs(metrics1.quality_score - metrics2.quality_score)

        threshold = self.config.performance_threshold / 2

        return success_rate_diff > threshold or quality_diff > threshold

    async def reset_baseline(
        self,
        target: OptimizationTarget,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Reset baseline after successful retraining

        Args:
            target: Optimization target
            metrics: New baseline metrics
        """
        target_key = self._get_target_key(target)
        self._baseline_metrics[target_key] = metrics

        # Clear recent metrics
        self._recent_metrics[target_key] = []

    async def get_drift_history(
        self,
        target: OptimizationTarget,
        hours: int = 24,
    ) -> list[tuple[datetime, PerformanceMetrics]]:
        """
        Get drift history for target

        Args:
            target: Optimization target
            hours: Time window in hours

        Returns:
            List of (timestamp, metrics) tuples
        """
        target_key = self._get_target_key(target)
        recent = self._recent_metrics.get(target_key, [])

        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return [(ts, m) for ts, m in recent if ts >= cutoff]

    async def _calculate_average_metrics(
        self,
        metrics_list: list[tuple[datetime, PerformanceMetrics]],
    ) -> PerformanceMetrics:
        """
        Calculate average metrics from list

        Args:
            metrics_list: List of (timestamp, metrics) tuples

        Returns:
            Average performance metrics
        """
        if not metrics_list:
            return PerformanceMetrics(
                success_rate=0.0,
                avg_cost_per_task=0.0,
                avg_latency_ms=0,
                quality_score=0.0,
            )

        metrics = [m for _, m in metrics_list]

        return PerformanceMetrics(
            success_rate=sum(m.success_rate for m in metrics) / len(metrics),
            avg_cost_per_task=sum(m.avg_cost_per_task for m in metrics) / len(metrics),
            avg_latency_ms=int(sum(m.avg_latency_ms for m in metrics) / len(metrics)),
            quality_score=sum(m.quality_score for m in metrics) / len(metrics),
        )

    async def _calculate_degradation(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
    ) -> float:
        """
        Calculate performance degradation percentage

        Args:
            baseline: Baseline metrics
            current: Current metrics

        Returns:
            Degradation percentage (0.0 to 1.0)
        """
        # Calculate degradation for success rate and quality score
        # (higher is better, so degradation is negative change)
        success_degradation = 0.0
        quality_degradation = 0.0

        if baseline.success_rate > 0:
            success_degradation = (
                baseline.success_rate - current.success_rate
            ) / baseline.success_rate

        if baseline.quality_score > 0:
            quality_degradation = (
                baseline.quality_score - current.quality_score
            ) / baseline.quality_score

        # Average degradation (weighted)
        degradation = success_degradation * 0.6 + quality_degradation * 0.4

        return max(0.0, degradation)  # Only positive degradation

    def _determine_drift_status(self, degradation: float) -> DriftStatus:
        """
        Determine drift status from degradation

        Args:
            degradation: Degradation percentage

        Returns:
            Drift status
        """
        if degradation >= self.config.critical_threshold:
            return DriftStatus.CRITICAL_DRIFT
        elif degradation >= self.config.performance_threshold:
            return DriftStatus.PERFORMANCE_DEGRADATION
        else:
            return DriftStatus.NO_DRIFT

    async def _enforce_window_size(self, target_key: str) -> None:
        """
        Enforce window size limit

        Args:
            target_key: Target key
        """
        recent = self._recent_metrics.get(target_key, [])

        if len(recent) > self.config.window_size:
            # Keep most recent samples
            recent.sort(key=lambda x: x[0], reverse=True)
            self._recent_metrics[target_key] = recent[: self.config.window_size]

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """
        Get target storage key

        Args:
            target: Optimization target

        Returns:
            Target key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
