"""Tests for online learning"""

import pytest

from agentcore.dspy_optimization.learning.online import (
    OnlineLearner,
    OnlineLearningConfig,
    LearningRate,
)
from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
)


class TestOnlineLearner:
    """Tests for online learner"""

    @pytest.fixture
    def learner(self) -> OnlineLearner:
        """Create online learner for testing"""
        config = OnlineLearningConfig(
            initial_learning_rate=0.01,
            learning_rate_schedule=LearningRate.EXPONENTIAL_DECAY,
            batch_size=10,
            memory_size=100,
            update_frequency=10,
        )
        return OnlineLearner(config)

    @pytest.fixture
    def target(self) -> OptimizationTarget:
        """Create optimization target for testing"""
        return OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent",
            scope=OptimizationScope.INDIVIDUAL,
        )

    @pytest.fixture
    def training_sample(self) -> dict[str, float | int]:
        """Create training sample for testing"""
        return {
            "success_rate": 0.80,
            "avg_cost_per_task": 0.12,
            "avg_latency_ms": 2200,
            "quality_score": 0.75,
        }

    @pytest.mark.asyncio
    async def test_add_training_sample(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test adding training sample"""
        result = await learner.add_training_sample(target, training_sample)

        # First sample should not trigger update
        assert result is None

        target_key = learner._get_target_key(target)
        assert target_key in learner._training_data
        assert len(learner._training_data[target_key]) == 1

    @pytest.mark.asyncio
    async def test_automatic_update_trigger(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test automatic update after update_frequency samples"""
        # Add samples to reach batch size + update frequency
        for _ in range(20):
            result = await learner.add_training_sample(target, training_sample)

        # Last sample should trigger update
        assert result is not None
        assert result.samples_processed == learner.config.batch_size

    @pytest.mark.asyncio
    async def test_manual_update(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test manual update"""
        # Add enough samples
        for _ in range(15):
            await learner.add_training_sample(target, training_sample)

        # Trigger manual update
        update = await learner.update(target)

        assert update is not None
        assert update.samples_processed > 0
        # Learning rate may have decayed slightly
        assert update.learning_rate <= learner.config.initial_learning_rate
        assert update.performance_before is not None
        assert update.performance_after is not None

    @pytest.mark.asyncio
    async def test_learning_rate_decay(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test learning rate decay"""
        # Get initial learning rate
        initial_rate = await learner.get_learning_rate(target)

        # Perform multiple updates
        for _ in range(100):
            await learner.add_training_sample(target, training_sample)

        # Trigger several updates
        for _ in range(5):
            try:
                await learner.update(target)
            except ValueError:
                # Add more samples if needed
                for _ in range(10):
                    await learner.add_training_sample(target, training_sample)
                await learner.update(target)

        # Learning rate should have decayed
        final_rate = await learner.get_learning_rate(target)
        assert final_rate < initial_rate
        assert final_rate >= learner.config.min_learning_rate

    @pytest.mark.asyncio
    async def test_update_history(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test update history tracking"""
        # Perform updates
        for _ in range(50):
            await learner.add_training_sample(target, training_sample)

        # Get history
        history = await learner.get_update_history(target, limit=5)

        assert len(history) <= 5
        for update in history:
            assert update.target == target
            assert update.samples_processed > 0

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test memory limit enforcement"""
        # Add more samples than memory size
        for _ in range(150):
            await learner.add_training_sample(target, training_sample)

        target_key = learner._get_target_key(target)
        assert len(learner._training_data[target_key]) <= learner.config.memory_size

    @pytest.mark.asyncio
    async def test_reset(
        self,
        learner: OnlineLearner,
        target: OptimizationTarget,
        training_sample: dict[str, float | int],
    ) -> None:
        """Test reset functionality"""
        # Add samples and perform update
        for _ in range(20):
            await learner.add_training_sample(target, training_sample)

        # Reset
        await learner.reset(target)

        target_key = learner._get_target_key(target)
        assert len(learner._training_data[target_key]) == 0
        assert learner._update_count[target_key] == 0
        assert learner._current_learning_rate[target_key] == learner.config.initial_learning_rate
