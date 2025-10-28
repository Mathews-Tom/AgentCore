"""
Statistical validation for A/B tests

Integrates with DSP-003 statistical testing infrastructure to validate
experiment results with significance testing and confidence intervals.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.monitoring.statistics import (
    SignificanceResult,
    StatisticalTester,
)
from agentcore.dspy_optimization.testing.experiment import (
    Experiment,
    ExperimentGroup,
)


class ValidationResult(BaseModel):
    """Result of experiment validation"""

    experiment_id: str
    is_valid: bool
    is_significant: bool
    improvement_percentage: float
    significance_result: SignificanceResult
    meets_minimum_improvement: bool
    has_sufficient_samples: bool
    recommendation: str
    warnings: list[str] = Field(default_factory=list)


class ExperimentValidator:
    """
    Validates A/B test experiments

    Integrates with StatisticalTester from DSP-003 to perform
    comprehensive statistical validation of experiment results.
    """

    def __init__(
        self,
        statistical_tester: StatisticalTester | None = None,
    ) -> None:
        """
        Initialize experiment validator

        Args:
            statistical_tester: Statistical tester instance (uses default if None)
        """
        self.tester = statistical_tester or StatisticalTester()

    async def validate_experiment(
        self,
        experiment: Experiment,
    ) -> ValidationResult:
        """
        Validate experiment results

        Args:
            experiment: Experiment to validate

        Returns:
            Validation result with recommendation

        Raises:
            ValueError: If experiment missing required data
        """
        # Check if experiment has results
        control_result = experiment.get_control_result()
        treatment_result = experiment.get_treatment_result()

        if not control_result or not treatment_result:
            raise ValueError(f"Experiment missing results: {experiment.id}")

        # Check sample sizes
        has_sufficient_samples = experiment.has_minimum_samples()

        warnings = []
        if not has_sufficient_samples:
            warnings.append(
                f"Insufficient samples: control={control_result.sample_count}, "
                f"treatment={treatment_result.sample_count}, "
                f"required={experiment.config.min_samples_per_group}"
            )

        # Perform statistical test
        try:
            significance_result = await self.tester.compare_metrics(
                baseline_samples=control_result.samples,
                optimized_samples=treatment_result.samples,
            )
        except ValueError as e:
            # Handle insufficient samples for statistical test
            return ValidationResult(
                experiment_id=experiment.id,
                is_valid=False,
                is_significant=False,
                improvement_percentage=0.0,
                significance_result=SignificanceResult(
                    test_type=self.tester.confidence_level,
                    p_value=1.0,
                    is_significant=False,
                    confidence_level=self.tester.confidence_level,
                ),
                meets_minimum_improvement=False,
                has_sufficient_samples=False,
                recommendation="Continue experiment - insufficient samples for validation",
                warnings=[str(e)],
            )

        # Calculate improvement
        improvement = self._calculate_improvement(
            control_result.metrics.success_rate,
            treatment_result.metrics.success_rate,
        )

        # Check minimum improvement threshold
        meets_minimum_improvement = (
            improvement >= experiment.config.min_improvement_threshold * 100
        )

        # Overall validation
        is_valid = (
            has_sufficient_samples
            and significance_result.is_significant
            and meets_minimum_improvement
            and treatment_result.metrics.success_rate
            > control_result.metrics.success_rate
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_valid=is_valid,
            is_significant=significance_result.is_significant,
            improvement=improvement,
            min_improvement=experiment.config.min_improvement_threshold * 100,
        )

        return ValidationResult(
            experiment_id=experiment.id,
            is_valid=is_valid,
            is_significant=significance_result.is_significant,
            improvement_percentage=improvement,
            significance_result=significance_result,
            meets_minimum_improvement=meets_minimum_improvement,
            has_sufficient_samples=has_sufficient_samples,
            recommendation=recommendation,
            warnings=warnings,
        )

    async def should_stop_early(
        self,
        experiment: Experiment,
    ) -> tuple[bool, str]:
        """
        Determine if experiment should stop early

        Args:
            experiment: Experiment to evaluate

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check if early stopping enabled
        if not experiment.config.early_stopping_enabled:
            return False, "Early stopping disabled"

        # Check if minimum samples reached
        if not experiment.should_stop_early():
            return False, "Insufficient samples for early stopping"

        # Validate experiment
        try:
            validation = await self.validate_experiment(experiment)
        except ValueError:
            return False, "Cannot validate - insufficient data"

        # Stop early if valid and significant
        if validation.is_valid:
            return True, f"Early stop: significant improvement ({validation.improvement_percentage:.1f}%)"

        # Stop early if treatment significantly worse
        treatment = experiment.get_treatment_result()
        control = experiment.get_control_result()

        if treatment and control:
            if treatment.metrics.success_rate < control.metrics.success_rate:
                if validation.is_significant:
                    return (
                        True,
                        f"Early stop: treatment significantly worse ({validation.improvement_percentage:.1f}%)",
                    )

        return False, "Continue experiment"

    async def calculate_required_duration(
        self,
        experiment: Experiment,
        samples_per_hour: int,
    ) -> float:
        """
        Calculate required experiment duration

        Args:
            experiment: Experiment configuration
            samples_per_hour: Expected samples per hour

        Returns:
            Required duration in hours
        """
        # Get control result for baseline std
        control_result = experiment.get_control_result()

        if not control_result or not control_result.samples:
            # Use default estimation
            return float(experiment.config.duration_hours)

        # Calculate baseline standard deviation
        success_rates = [s.get("success_rate", 0.0) for s in control_result.samples]
        mean = sum(success_rates) / len(success_rates)
        variance = sum((x - mean) ** 2 for x in success_rates) / len(success_rates)
        std = variance**0.5

        # Calculate required sample size
        min_detectable_effect = experiment.config.min_improvement_threshold
        required_samples = await self.tester.calculate_required_sample_size(
            baseline_std=std,
            min_detectable_effect=min_detectable_effect,
        )

        # Calculate duration (per group, so divide by 2)
        total_samples_needed = required_samples * 2
        hours_needed = total_samples_needed / samples_per_hour

        return max(hours_needed, experiment.config.duration_hours)

    def _calculate_improvement(
        self,
        control_value: float,
        treatment_value: float,
    ) -> float:
        """
        Calculate percentage improvement

        Args:
            control_value: Control group value
            treatment_value: Treatment group value

        Returns:
            Improvement percentage
        """
        if control_value == 0:
            return 0.0

        return ((treatment_value - control_value) / control_value) * 100

    def _generate_recommendation(
        self,
        is_valid: bool,
        is_significant: bool,
        improvement: float,
        min_improvement: float,
    ) -> str:
        """
        Generate recommendation based on validation

        Args:
            is_valid: Overall validation result
            is_significant: Statistical significance
            improvement: Improvement percentage
            min_improvement: Minimum improvement threshold

        Returns:
            Recommendation string
        """
        if is_valid:
            return f"Deploy treatment - {improvement:.1f}% improvement is significant"

        if not is_significant:
            return "Continue experiment - results not statistically significant"

        if improvement < min_improvement:
            return f"Continue experiment - improvement ({improvement:.1f}%) below threshold ({min_improvement:.1f}%)"

        if improvement < 0:
            return "Rollback treatment - performance degradation detected"

        return "Continue monitoring - validation inconclusive"
