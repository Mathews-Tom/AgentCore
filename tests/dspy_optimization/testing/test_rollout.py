"""Tests for automated rollout management"""

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
from agentcore.dspy_optimization.testing.rollout import (
    RolloutManager,
    RolloutConfig,
    RolloutStrategy,
    RolloutPhase,
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
def manager() -> RolloutManager:
    """Create rollout manager"""
    return RolloutManager()


def create_valid_experiment(target: OptimizationTarget) -> Experiment:
    """Create validated experiment"""
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

    # Add significant improvement
    control_samples = [
        {
            "success_rate": 0.75,
            "avg_cost_per_task": 0.1,
            "avg_latency_ms": 100,
            "quality_score": 0.8,
        }
        for _ in range(100)
    ]

    treatment_samples = [
        {
            "success_rate": 0.85,
            "avg_cost_per_task": 0.09,
            "avg_latency_ms": 90,
            "quality_score": 0.85,
        }
        for _ in range(100)
    ]

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

    return experiment


class TestRolloutConfig:
    """Test rollout configuration"""

    def test_default_config(self) -> None:
        """Test default configuration"""
        config = RolloutConfig()

        assert config.strategy == RolloutStrategy.PROGRESSIVE
        assert config.canary_percentage == 0.05
        assert config.auto_rollback_enabled is True
        assert len(config.progressive_steps) == 5

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = RolloutConfig(
            strategy=RolloutStrategy.CANARY,
            canary_percentage=0.1,
            progressive_steps=[0.2, 0.5, 1.0],
        )

        assert config.strategy == RolloutStrategy.CANARY
        assert config.canary_percentage == 0.1
        assert len(config.progressive_steps) == 3


class TestRolloutManager:
    """Test rollout manager"""

    @pytest.mark.asyncio
    async def test_start_immediate_rollout(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test immediate rollout"""
        experiment = create_valid_experiment(target)

        config = RolloutConfig(strategy=RolloutStrategy.IMMEDIATE)

        state = await manager.start_rollout(experiment, config)

        assert state.experiment_id == experiment.id
        assert state.phase == RolloutPhase.COMPLETE
        assert state.traffic_percentage == 1.0

    @pytest.mark.asyncio
    async def test_start_canary_rollout(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test canary rollout"""
        experiment = create_valid_experiment(target)

        config = RolloutConfig(
            strategy=RolloutStrategy.CANARY,
            canary_percentage=0.05,
        )

        state = await manager.start_rollout(experiment, config)

        assert state.phase == RolloutPhase.CANARY
        assert state.traffic_percentage == 0.05
        assert state.step_start_time is not None

    @pytest.mark.asyncio
    async def test_start_progressive_rollout(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test progressive rollout"""
        experiment = create_valid_experiment(target)

        config = RolloutConfig(
            strategy=RolloutStrategy.PROGRESSIVE,
            progressive_steps=[0.1, 0.25, 0.5, 1.0],
        )

        state = await manager.start_rollout(experiment, config)

        assert state.phase == RolloutPhase.PROGRESSIVE
        assert state.traffic_percentage == 0.1
        assert state.current_step == 0

    @pytest.mark.asyncio
    async def test_start_rollout_invalid_experiment(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test starting rollout with invalid experiment"""
        config = ExperimentConfig(
            name="Test",
            min_samples_per_group=1000,  # Too many required
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Add insufficient samples
        control_samples = [{"success_rate": 0.75} for _ in range(10)]
        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=10,
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples,
        )

        with pytest.raises(ValueError, match="(not valid|missing results)"):
            await manager.start_rollout(experiment)

    @pytest.mark.asyncio
    async def test_evaluate_rollout_hold(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test rollout evaluation - hold decision"""
        experiment = create_valid_experiment(target)

        config = RolloutConfig(
            strategy=RolloutStrategy.PROGRESSIVE,
            step_duration_hours=24,
        )

        state = await manager.start_rollout(experiment, config)

        # Evaluate immediately (duration not reached)
        decision = await manager.evaluate_rollout(experiment)

        assert decision.action == "hold"
        assert decision.phase == RolloutPhase.PROGRESSIVE
        assert "duration not reached" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluate_rollout_complete(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test evaluating completed rollout"""
        experiment = create_valid_experiment(target)

        config = RolloutConfig(strategy=RolloutStrategy.IMMEDIATE)

        state = await manager.start_rollout(experiment, config)

        # Evaluate completed rollout
        decision = await manager.evaluate_rollout(experiment)

        assert decision.action == "hold"
        assert decision.phase == RolloutPhase.COMPLETE
        assert "already complete" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_manual_rollback(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test manual rollback"""
        experiment = create_valid_experiment(target)

        await manager.start_rollout(experiment)

        decision = await manager.rollback(
            experiment.id,
            reason="Manual intervention",
        )

        assert decision.action == "rollback"
        assert decision.phase == RolloutPhase.ROLLED_BACK
        assert decision.traffic_percentage == 0.0
        assert "Manual intervention" in decision.reason

    @pytest.mark.asyncio
    async def test_auto_rollback_on_degradation(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test automatic rollback on degradation"""
        config = ExperimentConfig(
            name="Test",
            min_samples_per_group=50,
            min_improvement_threshold=0.01,  # Lower threshold to allow starting
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        # Start with slight improvement to pass initial validation
        control_samples_init = [
            {
                "success_rate": 0.75,
                "avg_cost_per_task": 0.1,
                "avg_latency_ms": 100,
                "quality_score": 0.8,
            }
            for _ in range(100)
        ]

        treatment_samples_init = [
            {
                "success_rate": 0.76,  # Slight improvement initially
                "avg_cost_per_task": 0.1,
                "avg_latency_ms": 100,
                "quality_score": 0.8,
            }
            for _ in range(100)
        ]

        experiment.results[ExperimentGroup.CONTROL.value] = ExperimentResult(
            group=ExperimentGroup.CONTROL,
            sample_count=len(control_samples_init),
            metrics=PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=control_samples_init,
        )

        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples_init),
            metrics=PerformanceMetrics(
                success_rate=0.76,
                avg_cost_per_task=0.1,
                avg_latency_ms=100,
            ),
            samples=treatment_samples_init,
        )

        rollout_config = RolloutConfig(
            auto_rollback_enabled=True,
            rollback_threshold_percentage=-5.0,
        )

        await manager.start_rollout(experiment, rollout_config)

        # Now update with degradation
        treatment_samples_bad = [
            {
                "success_rate": 0.65,  # 13% worse
                "avg_cost_per_task": 0.12,
                "avg_latency_ms": 120,
                "quality_score": 0.7,
            }
            for _ in range(100)
        ]

        experiment.results[ExperimentGroup.TREATMENT.value] = ExperimentResult(
            group=ExperimentGroup.TREATMENT,
            sample_count=len(treatment_samples_bad),
            metrics=PerformanceMetrics(
                success_rate=0.65,
                avg_cost_per_task=0.12,
                avg_latency_ms=120,
            ),
            samples=treatment_samples_bad,
        )

        decision = await manager.evaluate_rollout(experiment)

        assert decision.action == "rollback"
        assert decision.phase == RolloutPhase.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_get_rollout_state(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test getting rollout state"""
        experiment = create_valid_experiment(target)

        await manager.start_rollout(experiment)

        state = await manager.get_rollout_state(experiment.id)

        assert state is not None
        assert state.experiment_id == experiment.id

    @pytest.mark.asyncio
    async def test_rollout_not_found(
        self,
        manager: RolloutManager,
    ) -> None:
        """Test operations on non-existent rollout"""
        with pytest.raises(ValueError, match="not found"):
            await manager.evaluate_rollout(
                Experiment(
                    target=OptimizationTarget(
                        type=OptimizationTargetType.AGENT,
                        id="unknown",
                    ),
                    config=ExperimentConfig(name="Test"),
                    control_version="v1.0",
                    treatment_version="v1.1",
                )
            )

    @pytest.mark.asyncio
    async def test_rollout_decision_history(
        self,
        manager: RolloutManager,
        target: OptimizationTarget,
    ) -> None:
        """Test rollout decision tracking"""
        experiment = create_valid_experiment(target)

        state = await manager.start_rollout(experiment)

        # Should have initial decision
        assert len(state.decisions) == 1
        assert state.decisions[0].action == "start"

        # Manual rollback adds decision
        await manager.rollback(experiment.id)

        updated_state = await manager.get_rollout_state(experiment.id)
        assert updated_state is not None
        assert len(updated_state.decisions) == 2
        assert updated_state.decisions[-1].action == "rollback"
