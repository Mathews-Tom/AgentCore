"""Tests for automatic retraining"""

import pytest

from agentcore.dspy_optimization.learning.retraining import (
    RetrainingManager,
    RetrainingConfig,
    TriggerCondition,
    RetrainingStatus,
)
from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
)


class TestRetrainingManager:
    """Tests for retraining manager"""

    @pytest.fixture
    def manager(self) -> RetrainingManager:
        """Create retraining manager for testing"""
        config = RetrainingConfig(
            enable_drift_triggers=True,
            enable_scheduled_triggers=True,
            schedule_interval_hours=24,
            performance_threshold=0.10,
            min_samples_for_retraining=100,
            max_concurrent_retraining=3,
        )
        return RetrainingManager(config)

    @pytest.fixture
    def target(self) -> OptimizationTarget:
        """Create optimization target for testing"""
        return OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent",
            scope=OptimizationScope.INDIVIDUAL,
        )

    @pytest.mark.asyncio
    async def test_drift_trigger(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test drift-based trigger"""
        trigger = await manager.check_triggers(
            target,
            drift_detected=True,
            sample_count=150,
        )

        assert trigger is not None
        assert trigger.condition == TriggerCondition.DRIFT_DETECTED

    @pytest.mark.asyncio
    async def test_performance_threshold_trigger(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test performance threshold trigger"""
        trigger = await manager.check_triggers(
            target,
            performance_degradation=0.15,
            sample_count=150,
        )

        assert trigger is not None
        assert trigger.condition == TriggerCondition.PERFORMANCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_scheduled_trigger_initial(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test initial scheduled trigger"""
        trigger = await manager.check_triggers(
            target,
            sample_count=150,
        )

        assert trigger is not None
        assert trigger.condition == TriggerCondition.SCHEDULED

    @pytest.mark.asyncio
    async def test_sample_count_trigger(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test sample count trigger"""
        # Simulate previous training
        target_key = manager._get_target_key(target)
        manager._sample_counts[f"{target_key}_last_training"] = 100
        manager._last_retraining[target_key] = manager._last_retraining.get(target_key) or __import__('datetime').datetime.utcnow()

        # New samples exceed 2x minimum
        trigger = await manager.check_triggers(
            target,
            sample_count=500,
        )

        assert trigger is not None
        assert trigger.condition in (TriggerCondition.SAMPLE_COUNT, TriggerCondition.SCHEDULED)

    @pytest.mark.asyncio
    async def test_manual_trigger(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test manual trigger creation"""
        trigger = await manager.create_manual_trigger(
            target,
            metadata={"reason": "user_request"},
        )

        assert trigger is not None
        assert trigger.condition == TriggerCondition.MANUAL
        assert trigger.metadata["reason"] == "user_request"

    @pytest.mark.asyncio
    async def test_start_retraining(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test starting retraining job"""
        trigger = await manager.create_manual_trigger(target)

        job = await manager.start_retraining(trigger, sample_count=150)

        assert job is not None
        assert job.status == RetrainingStatus.PENDING
        assert job.trigger.id == trigger.id
        assert job.samples_used == 150

    @pytest.mark.asyncio
    async def test_concurrent_job_limit(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test concurrent job limit enforcement"""
        # Start max concurrent jobs
        for _ in range(manager.config.max_concurrent_retraining):
            trigger = await manager.create_manual_trigger(target)
            job = await manager.start_retraining(trigger, sample_count=150)
            await manager.update_job_status(job.id, RetrainingStatus.RUNNING)

        # Next job should fail
        trigger = await manager.create_manual_trigger(target)
        with pytest.raises(ValueError, match="Maximum concurrent retraining jobs"):
            await manager.start_retraining(trigger, sample_count=150)

    @pytest.mark.asyncio
    async def test_update_job_status(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test updating job status"""
        trigger = await manager.create_manual_trigger(target)
        job = await manager.start_retraining(trigger, sample_count=150)

        # Update to running
        updated = await manager.update_job_status(job.id, RetrainingStatus.RUNNING)
        assert updated.status == RetrainingStatus.RUNNING
        assert updated.started_at is not None

        # Complete job
        updated = await manager.update_job_status(
            job.id,
            RetrainingStatus.COMPLETED,
            validation_improvement=0.15,
        )
        assert updated.status == RetrainingStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.validation_improvement == 0.15

    @pytest.mark.asyncio
    async def test_validate_job(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test job validation"""
        trigger = await manager.create_manual_trigger(target)
        job = await manager.start_retraining(trigger, sample_count=150)

        # Should pass validation
        assert await manager.validate_job(job.id, improvement=0.10)

        # Should fail validation
        assert not await manager.validate_job(job.id, improvement=0.02)

    @pytest.mark.asyncio
    async def test_cancel_job(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test job cancellation"""
        trigger = await manager.create_manual_trigger(target)
        job = await manager.start_retraining(trigger, sample_count=150)

        cancelled = await manager.cancel_job(job.id)
        assert cancelled.status == RetrainingStatus.CANCELLED
        assert cancelled.completed_at is not None

    @pytest.mark.asyncio
    async def test_list_jobs(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test listing jobs"""
        # Create jobs
        for _ in range(3):
            trigger = await manager.create_manual_trigger(target)
            await manager.start_retraining(trigger, sample_count=150)

        # List all jobs
        all_jobs = await manager.list_jobs(target)
        assert len(all_jobs) == 3

        # Update one job
        job_id = all_jobs[0].id
        await manager.update_job_status(job_id, RetrainingStatus.COMPLETED)

        # List completed jobs
        completed = await manager.list_jobs(target, status=RetrainingStatus.COMPLETED)
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_trigger_history(
        self,
        manager: RetrainingManager,
        target: OptimizationTarget,
    ) -> None:
        """Test trigger history"""
        # Create triggers
        for i in range(5):
            await manager.create_manual_trigger(target, metadata={"index": i})

        # Get history
        history = await manager.get_trigger_history(target, limit=3)

        assert len(history) == 3
        # Should be sorted newest first
        assert history[0].metadata["index"] == 4
