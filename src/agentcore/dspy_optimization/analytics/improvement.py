"""
Improvement validation and analysis

Validates optimization improvements against targets (20-30% goal),
provides statistical validation, and analyzes improvement patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationTarget,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.monitoring.statistics import (
    SignificanceResult,
    StatisticalTester,
)


class ImprovementStatus(str, Enum):
    """Status of improvement validation"""

    EXCELLENT = "excellent"  # > 30% improvement
    TARGET_MET = "target_met"  # 20-30% improvement
    ACCEPTABLE = "acceptable"  # 10-20% improvement
    MARGINAL = "marginal"  # 5-10% improvement
    INSUFFICIENT = "insufficient"  # < 5% improvement
    DEGRADATION = "degradation"  # Negative improvement


class ImprovementValidationConfig(BaseModel):
    """Configuration for improvement validation"""

    target_min: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum target improvement (20%)",
    )
    target_max: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Maximum target improvement (30%)",
    )
    acceptable_min: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable improvement (10%)",
    )
    require_statistical_significance: bool = Field(
        default=True,
        description="Require statistical significance",
    )
    significance_threshold: float = Field(
        default=0.05,
        description="P-value threshold for significance",
    )
    min_sample_size: int = Field(
        default=30,
        description="Minimum sample size for validation",
    )


class ImprovementMetrics(BaseModel):
    """Detailed improvement metrics"""

    success_rate_improvement: float
    cost_reduction_percentage: float
    latency_reduction_percentage: float
    quality_improvement: float
    overall_improvement: float
    weighted_improvement: float


class ImprovementValidation(BaseModel):
    """Result of improvement validation"""

    target: OptimizationTarget
    status: ImprovementStatus
    improvement_metrics: ImprovementMetrics
    baseline: PerformanceMetrics
    optimized: PerformanceMetrics
    is_statistically_significant: bool
    significance_result: SignificanceResult | None = None
    meets_target: bool
    exceeds_target: bool
    sample_sizes: dict[str, int] = Field(default_factory=dict)
    validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImprovementAnalyzer:
    """
    Improvement validation and analysis

    Validates optimization improvements against 20-30% target,
    performs statistical validation, and provides recommendations.

    Key features:
    - Target-based validation (20-30% improvement goal)
    - Statistical significance testing
    - Multi-metric improvement analysis
    - Weighted scoring
    - Actionable recommendations
    """

    def __init__(self, config: ImprovementValidationConfig | None = None) -> None:
        """
        Initialize improvement analyzer

        Args:
            config: Validation configuration
        """
        self.config = config or ImprovementValidationConfig()
        self.statistical_tester = StatisticalTester(
            confidence_level=0.95,
            significance_threshold=self.config.significance_threshold,
        )

    async def validate_improvement(
        self,
        result: OptimizationResult,
        baseline_samples: list[dict[str, Any]],
        optimized_samples: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> ImprovementValidation:
        """
        Validate optimization improvement

        Args:
            result: Optimization result to validate
            baseline_samples: Baseline performance samples
            optimized_samples: Optimized performance samples
            weights: Optional metric weights for weighted scoring

        Returns:
            Improvement validation result

        Raises:
            ValueError: If insufficient data provided
        """
        if not result.baseline_performance or not result.optimized_performance:
            raise ValueError("Optimization result missing performance metrics")

        if len(baseline_samples) < self.config.min_sample_size:
            raise ValueError(
                f"Insufficient baseline samples: {len(baseline_samples)} "
                f"< {self.config.min_sample_size}"
            )

        if len(optimized_samples) < self.config.min_sample_size:
            raise ValueError(
                f"Insufficient optimized samples: {len(optimized_samples)} "
                f"< {self.config.min_sample_size}"
            )

        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(
            result.baseline_performance,
            result.optimized_performance,
            weights,
        )

        # Perform statistical validation if required
        significance_result = None
        is_significant = True

        if self.config.require_statistical_significance:
            is_valid, _, significance_result = await self.statistical_tester.validate_improvement(
                result.baseline_performance,
                result.optimized_performance,
                baseline_samples,
                optimized_samples,
            )
            is_significant = is_valid

        # Determine improvement status
        status = self._determine_status(improvement_metrics.overall_improvement)

        # Check if targets met
        meets_target = (
            improvement_metrics.overall_improvement >= self.config.target_min
            and is_significant
        )
        exceeds_target = (
            improvement_metrics.overall_improvement >= self.config.target_max
            and is_significant
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            improvement_metrics,
            status,
            is_significant,
            meets_target,
        )

        return ImprovementValidation(
            target=result.optimization_details.parameters.get("target")
            if result.optimization_details
            else OptimizationTarget(type="agent", id="unknown"),
            status=status,
            improvement_metrics=improvement_metrics,
            baseline=result.baseline_performance,
            optimized=result.optimized_performance,
            is_statistically_significant=is_significant,
            significance_result=significance_result,
            meets_target=meets_target,
            exceeds_target=exceeds_target,
            sample_sizes={
                "baseline": len(baseline_samples),
                "optimized": len(optimized_samples),
            },
            recommendations=recommendations,
            metadata={
                "optimization_id": result.optimization_id,
                "algorithm": result.optimization_details.algorithm_used
                if result.optimization_details
                else "unknown",
            },
        )

    async def validate_multiple_results(
        self,
        results: list[tuple[OptimizationResult, list[dict[str, Any]], list[dict[str, Any]]]],
        weights: dict[str, float] | None = None,
    ) -> list[ImprovementValidation]:
        """
        Validate multiple optimization results

        Args:
            results: List of (result, baseline_samples, optimized_samples) tuples
            weights: Optional metric weights

        Returns:
            List of validation results
        """
        validations = []

        for result, baseline_samples, optimized_samples in results:
            try:
                validation = await self.validate_improvement(
                    result,
                    baseline_samples,
                    optimized_samples,
                    weights,
                )
                validations.append(validation)
            except Exception as e:
                # Log error but continue processing
                continue

        return validations

    async def get_improvement_summary(
        self,
        validations: list[ImprovementValidation],
    ) -> dict[str, Any]:
        """
        Get summary of improvement validations

        Args:
            validations: List of validation results

        Returns:
            Summary statistics
        """
        if not validations:
            return {
                "total_validations": 0,
                "status_distribution": {},
                "avg_improvement": 0.0,
                "target_met_count": 0,
                "target_met_percentage": 0.0,
            }

        # Status distribution
        status_counts: dict[str, int] = {}
        for validation in validations:
            status = validation.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Target metrics
        target_met = sum(1 for v in validations if v.meets_target)
        target_exceeded = sum(1 for v in validations if v.exceeds_target)

        # Average improvements
        avg_improvement = sum(
            v.improvement_metrics.overall_improvement for v in validations
        ) / len(validations)

        avg_success_rate = sum(
            v.improvement_metrics.success_rate_improvement for v in validations
        ) / len(validations)

        return {
            "total_validations": len(validations),
            "status_distribution": status_counts,
            "avg_improvement": avg_improvement,
            "avg_success_rate_improvement": avg_success_rate,
            "target_met_count": target_met,
            "target_met_percentage": target_met / len(validations) * 100,
            "target_exceeded_count": target_exceeded,
            "target_exceeded_percentage": target_exceeded / len(validations) * 100,
            "statistically_significant_count": sum(
                1 for v in validations if v.is_statistically_significant
            ),
        }

    def _calculate_improvement_metrics(
        self,
        baseline: PerformanceMetrics,
        optimized: PerformanceMetrics,
        weights: dict[str, float] | None = None,
    ) -> ImprovementMetrics:
        """
        Calculate detailed improvement metrics

        Args:
            baseline: Baseline metrics
            optimized: Optimized metrics
            weights: Optional metric weights

        Returns:
            Improvement metrics
        """
        # Default weights
        if weights is None:
            weights = {
                "success_rate": 0.4,
                "cost": 0.3,
                "latency": 0.2,
                "quality": 0.1,
            }

        # Calculate individual improvements
        success_rate_improvement = (
            (optimized.success_rate - baseline.success_rate) / baseline.success_rate
            if baseline.success_rate > 0
            else 0.0
        )

        cost_reduction = (
            (baseline.avg_cost_per_task - optimized.avg_cost_per_task)
            / baseline.avg_cost_per_task
            if baseline.avg_cost_per_task > 0
            else 0.0
        )

        latency_reduction = (
            (baseline.avg_latency_ms - optimized.avg_latency_ms)
            / baseline.avg_latency_ms
            if baseline.avg_latency_ms > 0
            else 0.0
        )

        quality_improvement = (
            (optimized.quality_score - baseline.quality_score) / baseline.quality_score
            if baseline.quality_score > 0
            else 0.0
        )

        # Calculate overall improvement (average of all metrics)
        overall_improvement = (
            success_rate_improvement
            + cost_reduction
            + latency_reduction
            + quality_improvement
        ) / 4

        # Calculate weighted improvement
        weighted_improvement = (
            success_rate_improvement * weights.get("success_rate", 0.4)
            + cost_reduction * weights.get("cost", 0.3)
            + latency_reduction * weights.get("latency", 0.2)
            + quality_improvement * weights.get("quality", 0.1)
        )

        return ImprovementMetrics(
            success_rate_improvement=success_rate_improvement,
            cost_reduction_percentage=cost_reduction * 100,
            latency_reduction_percentage=latency_reduction * 100,
            quality_improvement=quality_improvement,
            overall_improvement=overall_improvement,
            weighted_improvement=weighted_improvement,
        )

    def _determine_status(self, improvement: float) -> ImprovementStatus:
        """
        Determine improvement status

        Args:
            improvement: Overall improvement percentage (as decimal, not percentage)

        Returns:
            Improvement status
        """
        if improvement < 0:
            return ImprovementStatus.DEGRADATION
        elif improvement < 0.05:
            return ImprovementStatus.INSUFFICIENT
        elif improvement < 0.10:
            return ImprovementStatus.MARGINAL
        elif improvement < self.config.target_min:
            return ImprovementStatus.ACCEPTABLE
        elif improvement < self.config.target_max:
            return ImprovementStatus.TARGET_MET
        else:
            return ImprovementStatus.EXCELLENT

    def _generate_recommendations(
        self,
        metrics: ImprovementMetrics,
        status: ImprovementStatus,
        is_significant: bool,
        meets_target: bool,
    ) -> list[str]:
        """
        Generate actionable recommendations

        Args:
            metrics: Improvement metrics
            status: Improvement status
            is_significant: Statistical significance
            meets_target: Whether target is met

        Returns:
            List of recommendations
        """
        recommendations = []

        # Statistical significance
        if not is_significant:
            recommendations.append(
                "Collect more samples to achieve statistical significance"
            )

        # Target achievement
        if not meets_target:
            if status == ImprovementStatus.DEGRADATION:
                recommendations.append(
                    "Performance degraded - investigate optimization algorithm or revert changes"
                )
            elif status in (ImprovementStatus.INSUFFICIENT, ImprovementStatus.MARGINAL):
                recommendations.append(
                    "Insufficient improvement - try different optimization algorithm or adjust parameters"
                )
            elif status == ImprovementStatus.ACCEPTABLE:
                recommendations.append(
                    "Acceptable improvement but below 20% target - consider additional optimization iterations"
                )

        # Metric-specific recommendations
        if metrics.success_rate_improvement < 0.10:
            recommendations.append(
                "Success rate improvement below 10% - focus on prompt quality and task decomposition"
            )

        if metrics.cost_reduction_percentage < 10:
            recommendations.append(
                "Limited cost reduction - consider smaller models or better caching strategies"
            )

        if metrics.latency_reduction_percentage < 10:
            recommendations.append(
                "Limited latency improvement - optimize model inference or enable parallel execution"
            )

        # Success recommendations
        if meets_target:
            recommendations.append(
                "Target met - consider deploying optimized version via A/B testing"
            )

        if status == ImprovementStatus.EXCELLENT:
            recommendations.append(
                "Excellent improvement - document optimization strategy for reuse"
            )

        return recommendations
