"""Tests for traffic splitting and routing"""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
)
from agentcore.dspy_optimization.testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentGroup,
    ExperimentStatus,
)
from agentcore.dspy_optimization.testing.traffic import (
    TrafficSplitter,
    TrafficSplitConfig,
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
def experiment(target: OptimizationTarget) -> Experiment:
    """Create test experiment"""
    config = ExperimentConfig(
        name="Test Experiment",
        traffic_percentage=0.5,
    )

    exp = Experiment(
        target=target,
        config=config,
        control_version="v1.0",
        treatment_version="v1.1",
        status=ExperimentStatus.ACTIVE,
    )

    return exp


@pytest.fixture
def splitter() -> TrafficSplitter:
    """Create traffic splitter"""
    return TrafficSplitter()


class TestTrafficSplitter:
    """Test traffic splitter"""

    @pytest.mark.asyncio
    async def test_route_request(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test routing single request"""
        decision = await splitter.route_request(
            experiment=experiment,
            request_id="req_001",
        )

        assert decision.experiment_id == experiment.id
        assert decision.group in (ExperimentGroup.CONTROL, ExperimentGroup.TREATMENT)
        assert decision.version in (experiment.control_version, experiment.treatment_version)
        assert decision.confidence == 1.0

    @pytest.mark.asyncio
    async def test_sticky_routing(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test consistent routing for same user"""
        user_id = "user_123"

        # Make multiple requests with same user
        decisions = []
        for i in range(5):
            decision = await splitter.route_request(
                experiment=experiment,
                request_id=f"req_{i}",
                user_id=user_id,
            )
            decisions.append(decision)

        # All should route to same group
        groups = [d.group for d in decisions]
        assert len(set(groups)) == 1

        versions = [d.version for d in decisions]
        assert len(set(versions)) == 1

    @pytest.mark.asyncio
    async def test_non_sticky_routing(
        self,
        experiment: Experiment,
    ) -> None:
        """Test non-sticky routing variation"""
        config = TrafficSplitConfig(use_sticky_routing=False)
        splitter = TrafficSplitter(config=config)

        # Same user_id, different request_ids should vary
        user_id = "user_123"
        decisions = []
        for i in range(100):
            decision = await splitter.route_request(
                experiment=experiment,
                request_id=f"req_{i}",
                user_id=user_id,
            )
            decisions.append(decision)

        # Should have some variation (not all same group)
        groups = [d.group for d in decisions]
        unique_groups = set(groups)

        # With 100 samples and 50/50 split, we expect both groups
        assert len(unique_groups) == 2

    @pytest.mark.asyncio
    async def test_route_batch(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test batch routing"""
        request_ids = [f"req_{i}" for i in range(10)]

        decisions = await splitter.route_batch(
            experiment=experiment,
            request_ids=request_ids,
        )

        assert len(decisions) == 10
        for decision in decisions:
            assert decision.experiment_id == experiment.id
            assert decision.group in (ExperimentGroup.CONTROL, ExperimentGroup.TREATMENT)

    @pytest.mark.asyncio
    async def test_route_batch_with_users(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test batch routing with user IDs"""
        request_ids = [f"req_{i}" for i in range(5)]
        user_ids = [f"user_{i}" for i in range(5)]

        decisions = await splitter.route_batch(
            experiment=experiment,
            request_ids=request_ids,
            user_ids=user_ids,
        )

        assert len(decisions) == 5

    @pytest.mark.asyncio
    async def test_route_inactive_experiment(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test routing to inactive experiment fails"""
        experiment.status = ExperimentStatus.DRAFT

        with pytest.raises(ValueError, match="not active"):
            await splitter.route_request(
                experiment=experiment,
                request_id="req_001",
            )

    @pytest.mark.asyncio
    async def test_traffic_distribution(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test traffic distribution calculation"""
        distribution = await splitter.get_traffic_distribution(
            experiment=experiment,
            sample_size=10000,
        )

        assert "control" in distribution
        assert "treatment" in distribution
        assert "control_count" in distribution
        assert "treatment_count" in distribution

        # Should be close to 50/50 split
        assert distribution["control"] == pytest.approx(0.5, abs=0.02)
        assert distribution["treatment"] == pytest.approx(0.5, abs=0.02)

    @pytest.mark.asyncio
    async def test_traffic_distribution_uneven_split(
        self,
        splitter: TrafficSplitter,
        target: OptimizationTarget,
    ) -> None:
        """Test traffic distribution with uneven split"""
        config = ExperimentConfig(
            name="Uneven Split",
            traffic_percentage=0.3,  # 30% treatment
        )

        experiment = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        distribution = await splitter.get_traffic_distribution(
            experiment=experiment,
            sample_size=10000,
        )

        assert distribution["treatment"] == pytest.approx(0.3, abs=0.02)
        assert distribution["control"] == pytest.approx(0.7, abs=0.02)

    @pytest.mark.asyncio
    async def test_validate_split_accuracy(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test split accuracy validation"""
        is_accurate = await splitter.validate_split_accuracy(
            experiment=experiment,
            sample_size=10000,
            tolerance=0.02,
        )

        assert is_accurate is True

    @pytest.mark.asyncio
    async def test_consistent_hashing(
        self,
        splitter: TrafficSplitter,
        experiment: Experiment,
    ) -> None:
        """Test consistent hashing produces same results"""
        request_id = "req_001"
        user_id = "user_001"

        # Make same request multiple times
        results = []
        for _ in range(10):
            decision = await splitter.route_request(
                experiment=experiment,
                request_id=request_id,
                user_id=user_id,
            )
            results.append((decision.group, decision.version))

        # All should be identical
        assert len(set(results)) == 1

    @pytest.mark.asyncio
    async def test_different_experiments_different_routing(
        self,
        splitter: TrafficSplitter,
        target: OptimizationTarget,
    ) -> None:
        """Test different experiments route independently"""
        config = ExperimentConfig(name="Test", traffic_percentage=0.5)

        exp1 = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        exp2 = Experiment(
            target=target,
            config=config,
            control_version="v1.0",
            treatment_version="v1.1",
            status=ExperimentStatus.ACTIVE,
        )

        request_id = "req_001"

        decision1 = await splitter.route_request(exp1, request_id)
        decision2 = await splitter.route_request(exp2, request_id)

        # Different experiments should potentially route differently
        assert decision1.experiment_id != decision2.experiment_id
