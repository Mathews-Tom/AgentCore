"""Tests for experiment design and management"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationScope,
    OptimizationTarget,
    OptimizationTargetType,
)
from agentcore.dspy_optimization.testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentGroup,
    ExperimentManager,
    ExperimentStatus,
)


@pytest.fixture
def target() -> OptimizationTarget:
    """Create test optimization target"""
    return OptimizationTarget(
        type=OptimizationTargetType.AGENT,
        id="test_agent_001",
        scope=OptimizationScope.INDIVIDUAL,
    )


@pytest.fixture
def config() -> ExperimentConfig:
    """Create test experiment config"""
    return ExperimentConfig(
        name="Test A/B Experiment",
        description="Testing optimization improvements",
        traffic_percentage=0.5,
        min_samples_per_group=50,
        duration_hours=24,
    )


@pytest.fixture
def manager() -> ExperimentManager:
    """Create experiment manager"""
    return ExperimentManager()


class TestExperimentConfig:
    """Test experiment configuration"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = ExperimentConfig(name="Test")

        assert config.traffic_percentage == 0.5
        assert config.min_samples_per_group == 100
        assert config.duration_hours == 24
        assert config.significance_threshold == 0.05
        assert config.early_stopping_enabled is True

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = ExperimentConfig(
            name="Custom Test",
            traffic_percentage=0.3,
            min_samples_per_group=200,
            early_stopping_enabled=False,
        )

        assert config.traffic_percentage == 0.3
        assert config.min_samples_per_group == 200
        assert config.early_stopping_enabled is False


class TestExperiment:
    """Test experiment model"""

    def test_create_experiment(
        self, target: OptimizationTarget, config: ExperimentConfig
    ) -> None:
        """Test experiment creation"""
        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        assert experiment.target == target
        assert experiment.config == config
        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.control_version == "v1.0"
        assert experiment.treatment_version == "v1.1"
        assert len(experiment.results) == 0

    def test_is_active(
        self, target: OptimizationTarget, config: ExperimentConfig
    ) -> None:
        """Test active status check"""
        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        assert not experiment.is_active()

        experiment.status = ExperimentStatus.ACTIVE
        assert experiment.is_active()

    def test_is_completed(
        self, target: OptimizationTarget, config: ExperimentConfig
    ) -> None:
        """Test completed status check"""
        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        assert not experiment.is_completed()

        experiment.status = ExperimentStatus.COMPLETED
        assert experiment.is_completed()

        experiment.status = ExperimentStatus.FAILED
        assert experiment.is_completed()

    def test_get_duration_elapsed(
        self, target: OptimizationTarget, config: ExperimentConfig
    ) -> None:
        """Test duration calculation"""
        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        assert experiment.get_duration_elapsed() is None

        start = datetime.now(UTC)
        experiment.start_time = start
        experiment.end_time = start + timedelta(hours=12)

        duration = experiment.get_duration_elapsed()
        assert duration is not None
        assert duration.total_seconds() == pytest.approx(12 * 3600, rel=1)

    def test_has_minimum_samples(
        self, target: OptimizationTarget, config: ExperimentConfig
    ) -> None:
        """Test minimum samples check"""
        config.min_samples_per_group = 50

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        assert not experiment.has_minimum_samples()

        # Add results with insufficient samples
        from agentcore.dspy_optimization.models import PerformanceMetrics
        from agentcore.dspy_optimization.testing.experiment import ExperimentResult

        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=30,
            metrics=PerformanceMetrics(
                success_rate=0.75, avg_cost_per_task=0.1, avg_latency_ms=100
            ),
        )

        assert not experiment.has_minimum_samples()

        # Add treatment with sufficient samples
        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=60,
            metrics=PerformanceMetrics(
                success_rate=0.85, avg_cost_per_task=0.09, avg_latency_ms=90
            ),
        )

        assert not experiment.has_minimum_samples()  # Control still insufficient

        # Update control
        experiment.results[ExperimentGroup.CONTROL.value].sample_count = 50

        assert experiment.has_minimum_samples()


class TestExperimentManager:
    """Test experiment manager"""

    @pytest.mark.asyncio
    async def test_create_experiment(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test creating experiment"""
        experiment = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        assert experiment.target == target
        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.control_version == "v1.0"
        assert experiment.treatment_version == "v1.1"

    @pytest.mark.asyncio
    async def test_start_experiment(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test starting experiment"""
        experiment = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        started = await manager.start_experiment(experiment.id)

        assert started.status == ExperimentStatus.ACTIVE
        assert started.start_time is not None

    @pytest.mark.asyncio
    async def test_start_experiment_errors(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test start experiment error cases"""
        # Non-existent experiment
        with pytest.raises(ValueError, match="not found"):
            await manager.start_experiment("non_existent")

        # Already started
        experiment = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )
        await manager.start_experiment(experiment.id)

        with pytest.raises(ValueError, match="already started"):
            await manager.start_experiment(experiment.id)

    @pytest.mark.asyncio
    async def test_record_result(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test recording experiment results"""
        experiment = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )
        await manager.start_experiment(experiment.id)

        # Record control sample
        sample = {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.1,
            "avg_latency_ms": 100,
            "quality_score": 0.8,
        }

        await manager.record_result(experiment.id, ExperimentGroup.CONTROL, sample)

        updated = await manager.get_experiment(experiment.id)
        assert updated is not None
        assert ExperimentGroup.CONTROL.value in updated.results
        assert updated.results[ExperimentGroup.CONTROL.value].sample_count == 1

    @pytest.mark.asyncio
    async def test_pause_resume_experiment(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test pausing and resuming experiment"""
        experiment = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )
        await manager.start_experiment(experiment.id)

        # Pause
        paused = await manager.pause_experiment(experiment.id)
        assert paused.status == ExperimentStatus.PAUSED

        # Resume
        resumed = await manager.resume_experiment(experiment.id)
        assert resumed.status == ExperimentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_complete_experiment(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test completing experiment"""
        experiment = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )
        await manager.start_experiment(experiment.id)

        completed = await manager.complete_experiment(experiment.id)

        assert completed.status == ExperimentStatus.COMPLETED
        assert completed.end_time is not None

    @pytest.mark.asyncio
    async def test_list_experiments(
        self,
        manager: ExperimentManager,
        target: OptimizationTarget,
        config: ExperimentConfig,
    ) -> None:
        """Test listing experiments"""
        # Create multiple experiments
        exp1 = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
        )

        exp2 = await manager.create_experiment(
            target=target,
            config=config,
            control_version="v1.1",
            treatment_version="v1.2",
        )

        await manager.start_experiment(exp1.id)

        # List all
        all_exps = await manager.list_experiments()
        assert len(all_exps) == 2

        # Filter by status
        active = await manager.list_experiments(status=ExperimentStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].id == exp1.id

        draft = await manager.list_experiments(status=ExperimentStatus.DRAFT)
        assert len(draft) == 1
        assert draft[0].id == exp2.id
