"""Tests for model versioning"""

import pytest

from agentcore.dspy_optimization.learning.versioning import (
    ModelVersionManager,
    DeploymentStrategy,
    ModelStatus,
)
from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
    PerformanceMetrics,
)


class TestModelVersionManager:
    """Tests for model version manager"""

    @pytest.fixture
    def manager(self) -> ModelVersionManager:
        """Create version manager for testing"""
        return ModelVersionManager()

    @pytest.fixture
    def target(self) -> OptimizationTarget:
        """Create optimization target for testing"""
        return OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent",
            scope=OptimizationScope.INDIVIDUAL,
        )

    @pytest.fixture
    def performance_metrics(self) -> PerformanceMetrics:
        """Create performance metrics for testing"""
        return PerformanceMetrics(
            success_rate=0.85,
            avg_cost_per_task=0.10,
            avg_latency_ms=2000,
            quality_score=0.80,
        )

    @pytest.mark.asyncio
    async def test_create_version(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test creating model version"""
        version = await manager.create_version(
            target,
            deployment_strategy=DeploymentStrategy.AB_TEST,
            metadata={"test": "data"},
        )

        assert version.version_number == 1
        assert version.target == target
        assert version.status == ModelStatus.TRAINING
        assert version.deployment_strategy == DeploymentStrategy.AB_TEST

    @pytest.mark.asyncio
    async def test_version_numbering(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test version number incrementation"""
        v1 = await manager.create_version(target)
        v2 = await manager.create_version(target)
        v3 = await manager.create_version(target)

        assert v1.version_number == 1
        assert v2.version_number == 2
        assert v3.version_number == 3

    @pytest.mark.asyncio
    async def test_update_version_status(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
        performance_metrics: PerformanceMetrics,
    ) -> None:
        """Test updating version status"""
        version = await manager.create_version(target)

        updated = await manager.update_version_status(
            version.id,
            ModelStatus.VALIDATING,
            performance_metrics,
        )

        assert updated.status == ModelStatus.VALIDATING
        assert updated.performance_metrics == performance_metrics

    @pytest.mark.asyncio
    async def test_deploy_version(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test deploying version"""
        version = await manager.create_version(target)

        deployed = await manager.deploy_version(version.id, traffic_percentage=1.0)

        assert deployed.status == ModelStatus.DEPLOYED
        assert deployed.traffic_percentage == 1.0
        assert deployed.deployed_at is not None
        assert deployed.is_deployed()
        assert deployed.is_active()

    @pytest.mark.asyncio
    async def test_gradual_rollout(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test gradual rollout"""
        version = await manager.create_version(target)
        await manager.deploy_version(version.id, traffic_percentage=0.1)

        # Gradually increase traffic
        updated = await manager.gradual_rollout(version.id, target_percentage=0.5, step_size=0.1)
        assert abs(updated.traffic_percentage - 0.2) < 0.01

        updated = await manager.gradual_rollout(version.id, target_percentage=0.5, step_size=0.1)
        assert abs(updated.traffic_percentage - 0.3) < 0.01

    @pytest.mark.asyncio
    async def test_rollback_version(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test rollback to previous version"""
        # Create two versions
        v1 = await manager.create_version(target)
        await manager.deploy_version(v1.id, traffic_percentage=1.0)

        v2 = await manager.create_version(target, parent_version_id=v1.id)
        await manager.deploy_version(v2.id, traffic_percentage=1.0)

        # Rollback v2 to v1
        rolled_back = await manager.rollback_version(v2.id)

        assert rolled_back.status == ModelStatus.ROLLED_BACK
        assert rolled_back.traffic_percentage == 0.0

        # v1 should be active again
        v1_redeployed = await manager.get_version(v1.id)
        assert v1_redeployed.status == ModelStatus.DEPLOYED

    @pytest.mark.asyncio
    async def test_get_active_version(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test getting active version"""
        version = await manager.create_version(target)
        await manager.deploy_version(version.id, traffic_percentage=1.0)

        active = await manager.get_active_version(target)
        assert active is not None
        assert active.id == version.id

    @pytest.mark.asyncio
    async def test_list_versions(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test listing versions"""
        # Create multiple versions
        v1 = await manager.create_version(target)
        v2 = await manager.create_version(target)
        await manager.deploy_version(v1.id, traffic_percentage=1.0)

        # List all versions
        all_versions = await manager.list_versions(target)
        assert len(all_versions) == 2

        # List deployed versions
        deployed = await manager.list_versions(target, status=ModelStatus.DEPLOYED)
        assert len(deployed) == 1
        assert deployed[0].id == v1.id

    @pytest.mark.asyncio
    async def test_version_history(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test version history"""
        # Create versions
        await manager.create_version(target)
        await manager.create_version(target)
        await manager.create_version(target)

        history = await manager.get_version_history(target)

        assert len(history) == 3
        # Should be sorted by version number (newest first)
        assert history[0].version_number == 3
        assert history[1].version_number == 2
        assert history[2].version_number == 1

    @pytest.mark.asyncio
    async def test_cleanup_old_versions(
        self,
        manager: ModelVersionManager,
        target: OptimizationTarget,
    ) -> None:
        """Test cleanup of old versions"""
        # Create and deprecate versions
        for _ in range(15):
            version = await manager.create_version(target)
            await manager.update_version_status(version.id, ModelStatus.DEPRECATED)

        # Cleanup, keeping only 10
        cleaned = await manager.cleanup_old_versions(target, keep_count=10)

        assert cleaned == 5  # 15 - 10 = 5

        remaining = await manager.list_versions(target)
        assert len(remaining) == 10
