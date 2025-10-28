"""Tests for statistical validation"""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentGroup,
    ExperimentResult,
    ExperimentStatus,
)
from agentcore.dspy_optimization.testing.validation import ExperimentValidator


@pytest.fixture
def target() -> OptimizationTarget:
    """Create test optimization target"""
    return OptimizationTarget(
        type=OptimizationTargetType.AGENT,
        id="test_agent_001",
        scope=OptimizationScope.INDIVIDUAL,
    )


@pytest.fixture
def validator() -> ExperimentValidator:
    """Create experiment validator"""
    return ExperimentValidator()


def create_samples(
    count: int,
    success_rate: float,
    cost: float = 0.1,
    latency: int = 100,
) -> list[dict[str, float]]:
    """Helper to create performance samples"""
    return [
        {
            "success_rate": success_rate,
            "avg_cost_per_task": cost,
            "avg_latency_ms": latency,
            "quality_score": 0.8,
        }
        for _ in range(count)
    ]


class TestExperimentValidator:
    """Test experiment validator"""

    @pytest.mark.asyncio
    async def test_validate_successful_experiment(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test validating successful experiment"""
        config = ExperimentConfig(
            name="Test",
            min_samples_per_group=50,
            min_improvement_threshold=0.05,
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add control results
        control_samples = create_samples(100, 0.75)
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        # Add treatment results (10% improvement)
        treatment_samples = create_samples(100, 0.825)
        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples),
            metrics=PerformanceMetrics(
                success_rate=0.825,
                avg_cost_per_task=0.09,
                avg_latency_ms=90,
            ),
            samples=treatment_samples,
        )

        validation = await validator.validate_experiment(experiment)

        assert validation.experiment_id == experiment.id
        assert validation.is_valid is True
        assert validation.is_significant is True
        assert validation.has_sufficient_samples is True
        assert validation.meets_minimum_improvement is True
        assert validation.improvement_percentage > 5.0
        assert "Deploy" in validation.recommendation

    @pytest.mark.asyncio
    async def test_validate_insufficient_samples(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test validation with insufficient samples"""
        config = ExperimentConfig(
            name="Test",
            min_samples_per_group=100,
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add results with too few samples
        control_samples = create_samples(10, 0.75)
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        treatment_samples = create_samples(10, 0.85)
        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples),
            metrics=PerformanceMetrics(
                success_rate=0.85,
                avg_cost_per_task=0.09,
                avg_latency_ms=90,
            ),
            samples=treatment_samples,
        )

        validation = await validator.validate_experiment(experiment)

        assert validation.is_valid is False
        assert validation.has_sufficient_samples is False
        assert len(validation.warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_insufficient_improvement(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test validation with insufficient improvement"""
        config = ExperimentConfig(
            name="Test",
            min_samples_per_group=50,
            min_improvement_threshold=0.1,  # 10% required
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add results with only 3% improvement
        control_samples = create_samples(100, 0.75)
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        treatment_samples = create_samples(100, 0.7725)  # 3% improvement
        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples),
            metrics=PerformanceMetrics(
                success_rate=0.7725,
                avg_cost_per_task=0.09,
                avg_latency_ms=90,
            ),
            samples=treatment_samples,
        )

        validation = await validator.validate_experiment(experiment)

        assert validation.is_valid is False
        assert validation.meets_minimum_improvement is False

    @pytest.mark.asyncio
    async def test_validate_missing_results(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test validation with missing results"""
        config = ExperimentConfig(name="Test")

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        with pytest.raises(ValueError, match="missing results"):
            await validator.validate_experiment(experiment)

    @pytest.mark.asyncio
    async def test_should_stop_early_success(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test early stopping on success"""
        config = ExperimentConfig(
            name="Test",
            early_stopping_enabled=True,
            early_stopping_min_samples=100,
            min_samples_per_group=50,
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add significant improvement
        control_samples = create_samples(100, 0.75)
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        treatment_samples = create_samples(100, 0.90)  # 20% improvement
        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples),
            metrics=PerformanceMetrics(
                success_rate=0.90,
                avg_cost_per_task=0.09,
                avg_latency_ms=90,
            ),
            samples=treatment_samples,
        )

        should_stop, reason = await validator.should_stop_early(experiment)

        assert should_stop is True
        assert "significant improvement" in reason.lower()

    @pytest.mark.asyncio
    async def test_should_stop_early_degradation(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test early stopping on degradation"""
        config = ExperimentConfig(
            name="Test",
            early_stopping_enabled=True,
            early_stopping_min_samples=100,
            min_samples_per_group=50,
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add degradation
        control_samples = create_samples(100, 0.75)
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        treatment_samples = create_samples(100, 0.60)  # 20% worse
        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples),
            metrics=PerformanceMetrics(
                success_rate=0.60,
                avg_cost_per_task=0.12,
                avg_latency_ms=120,
            ),
            samples=treatment_samples,
        )

        should_stop, reason = await validator.should_stop_early(experiment)

        assert should_stop is True
        assert "worse" in reason.lower()

    @pytest.mark.asyncio
    async def test_should_stop_early_disabled(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test early stopping when disabled"""
        config = ExperimentConfig(
            name="Test",
            early_stopping_enabled=False,
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        should_stop, reason = await validator.should_stop_early(experiment)

        assert should_stop is False
        assert "disabled" in reason.lower()

    @pytest.mark.asyncio
    async def test_calculate_required_duration(
        self,
        validator: ExperimentValidator,
        target: OptimizationTarget,
    ) -> None:
        """Test required duration calculation"""
        config = ExperimentConfig(
            name="Test",
            duration_hours=24,
            min_improvement_threshold=0.05,
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add control samples
        control_samples = create_samples(100, 0.75)
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        duration = await validator.calculate_required_duration(
            experiment=experiment,
            samples_per_hour=50,
        )

        assert duration >= config.duration_hours
        assert isinstance(duration, (int, float))
