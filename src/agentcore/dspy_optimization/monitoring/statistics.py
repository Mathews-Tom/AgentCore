"""
Statistical significance testing

Provides statistical tests for validating optimization improvements,
including t-tests, confidence intervals, and effect size calculations.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from scipy import stats

from agentcore.dspy_optimization.models import PerformanceMetrics


class SignificanceTest(str, Enum):
    """Types of statistical significance tests"""

    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY = "mann_whitney"
    PAIRED_T_TEST = "paired_t_test"


class ConfidenceInterval(BaseModel):
    """Confidence interval for metric"""

    metric_name: str
    mean: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = Field(default=0.95)


class EffectSize(BaseModel):
    """Effect size measurement"""

    cohens_d: float
    interpretation: str


class SignificanceResult(BaseModel):
    """Result of statistical significance test"""

    test_type: SignificanceTest
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: EffectSize | None = None
    confidence_intervals: list[ConfidenceInterval] = Field(default_factory=list)
    sample_sizes: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatisticalTester:
    """
    Statistical significance testing for optimization results

    Provides comprehensive statistical analysis including hypothesis testing,
    confidence intervals, and effect size calculations to validate
    optimization improvements.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        significance_threshold: float = 0.05,
    ) -> None:
        """
        Initialize statistical tester

        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            significance_threshold: P-value threshold for significance (default: 0.05)
        """
        self.confidence_level = confidence_level
        self.significance_threshold = significance_threshold

    async def compare_metrics(
        self,
        baseline_samples: list[dict[str, Any]],
        optimized_samples: list[dict[str, Any]],
        test_type: SignificanceTest = SignificanceTest.WELCH_T_TEST,
    ) -> SignificanceResult:
        """
        Compare baseline vs optimized metrics

        Args:
            baseline_samples: Baseline performance samples
            optimized_samples: Optimized performance samples
            test_type: Statistical test to use

        Returns:
            Significance test result

        Raises:
            ValueError: If insufficient samples provided
        """
        if len(baseline_samples) < 2 or len(optimized_samples) < 2:
            raise ValueError(
                f"Insufficient samples: baseline={len(baseline_samples)}, "
                f"optimized={len(optimized_samples)}"
            )

        # Extract success rates for primary comparison
        baseline_values = [s.get("success_rate", 0.0) for s in baseline_samples]
        optimized_values = [s.get("success_rate", 0.0) for s in optimized_samples]

        # Perform statistical test
        if test_type == SignificanceTest.T_TEST:
            statistic, p_value = stats.ttest_ind(
                optimized_values, baseline_values, equal_var=True
            )
        elif test_type == SignificanceTest.WELCH_T_TEST:
            statistic, p_value = stats.ttest_ind(
                optimized_values, baseline_values, equal_var=False
            )
        elif test_type == SignificanceTest.MANN_WHITNEY:
            statistic, p_value = stats.mannwhitneyu(
                optimized_values, baseline_values, alternative="greater"
            )
        elif test_type == SignificanceTest.PAIRED_T_TEST:
            # Require equal sample sizes
            min_size = min(len(baseline_values), len(optimized_values))
            statistic, p_value = stats.ttest_rel(
                optimized_values[:min_size], baseline_values[:min_size]
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Calculate effect size
        effect_size = self._calculate_effect_size(baseline_values, optimized_values)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            baseline_samples, optimized_samples
        )

        # Determine significance
        is_significant = p_value < self.significance_threshold

        return SignificanceResult(
            test_type=test_type,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_intervals=confidence_intervals,
            sample_sizes={
                "baseline": len(baseline_samples),
                "optimized": len(optimized_samples),
            },
            metadata={
                "test_statistic": float(statistic),
            },
        )

    async def validate_improvement(
        self,
        baseline: PerformanceMetrics,
        optimized: PerformanceMetrics,
        baseline_samples: list[dict[str, Any]],
        optimized_samples: list[dict[str, Any]],
    ) -> tuple[bool, float, SignificanceResult]:
        """
        Validate optimization improvement with statistical testing

        Args:
            baseline: Baseline metrics
            optimized: Optimized metrics
            baseline_samples: Baseline samples
            optimized_samples: Optimized samples

        Returns:
            Tuple of (is_valid, improvement_percentage, significance_result)
        """
        # Calculate improvement percentage
        improvement = (
            (optimized.success_rate - baseline.success_rate) / baseline.success_rate
            * 100
        )

        # Perform significance test
        significance = await self.compare_metrics(
            baseline_samples, optimized_samples
        )

        # Validate improvement
        is_valid = (
            significance.is_significant
            and optimized.success_rate > baseline.success_rate
        )

        return is_valid, improvement, significance

    def _calculate_effect_size(
        self,
        baseline_values: list[float],
        optimized_values: list[float],
    ) -> EffectSize:
        """
        Calculate Cohen's d effect size

        Args:
            baseline_values: Baseline values
            optimized_values: Optimized values

        Returns:
            Effect size with interpretation
        """
        # Calculate means
        baseline_mean = sum(baseline_values) / len(baseline_values)
        optimized_mean = sum(optimized_values) / len(optimized_values)

        # Calculate pooled standard deviation
        baseline_var = sum((x - baseline_mean) ** 2 for x in baseline_values) / (
            len(baseline_values) - 1
        )
        optimized_var = sum((x - optimized_mean) ** 2 for x in optimized_values) / (
            len(optimized_values) - 1
        )

        pooled_std = math.sqrt(
            (
                (len(baseline_values) - 1) * baseline_var
                + (len(optimized_values) - 1) * optimized_var
            )
            / (len(baseline_values) + len(optimized_values) - 2)
        )

        # Calculate Cohen's d
        cohens_d = (optimized_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0.0

        # Interpret effect size
        interpretation = self._interpret_effect_size(abs(cohens_d))

        return EffectSize(cohens_d=cohens_d, interpretation=interpretation)

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size

        Args:
            cohens_d: Absolute Cohen's d value

        Returns:
            Interpretation string
        """
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def _calculate_confidence_intervals(
        self,
        baseline_samples: list[dict[str, Any]],
        optimized_samples: list[dict[str, Any]],
    ) -> list[ConfidenceInterval]:
        """
        Calculate confidence intervals for all metrics

        Args:
            baseline_samples: Baseline samples
            optimized_samples: Optimized samples

        Returns:
            List of confidence intervals
        """
        intervals = []

        # Metrics to calculate intervals for
        metrics = [
            "success_rate",
            "avg_cost_per_task",
            "avg_latency_ms",
            "quality_score",
        ]

        for metric_name in metrics:
            # Extract optimized values
            values = [s.get(metric_name, 0.0) for s in optimized_samples]

            if not values:
                continue

            # Calculate confidence interval
            mean = sum(values) / len(values)
            std_err = self._calculate_std_error(values)

            # t-distribution critical value
            df = len(values) - 1
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)

            margin = t_critical * std_err
            lower = mean - margin
            upper = mean + margin

            intervals.append(
                ConfidenceInterval(
                    metric_name=metric_name,
                    mean=mean,
                    lower_bound=lower,
                    upper_bound=upper,
                    confidence_level=self.confidence_level,
                )
            )

        return intervals

    def _calculate_std_error(self, values: list[float]) -> float:
        """
        Calculate standard error

        Args:
            values: List of values

        Returns:
            Standard error
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_dev = math.sqrt(variance)

        return std_dev / math.sqrt(len(values))

    async def calculate_required_sample_size(
        self,
        baseline_std: float,
        min_detectable_effect: float,
        power: float = 0.8,
    ) -> int:
        """
        Calculate required sample size for detecting effect

        Args:
            baseline_std: Baseline standard deviation
            min_detectable_effect: Minimum effect to detect
            power: Statistical power (default: 0.8)

        Returns:
            Required sample size per group
        """
        # Standard normal quantiles
        alpha = 1 - self.confidence_level
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Calculate sample size
        n = (
            2
            * ((z_alpha + z_beta) ** 2)
            * (baseline_std**2)
            / (min_detectable_effect**2)
        )

        return int(math.ceil(n))
