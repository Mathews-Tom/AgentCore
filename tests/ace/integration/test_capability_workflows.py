"""
Integration Tests for Capability Evaluation Workflows - ACE-028

Tests end-to-end capability evaluation workflows combining evaluator,
fitness scorer, and recommender components.
"""

import pytest
from uuid import uuid4

from agentcore.ace.capability.evaluator import CapabilityEvaluator
from agentcore.ace.capability.fitness_scorer import FitnessScorer
from agentcore.ace.capability.recommender import CapabilityRecommender
from agentcore.ace.models.ace_models import (
    CapabilityType,
    TaskRequirement,
)


@pytest.fixture
def evaluator():
    """Create capability evaluator instance."""
    return CapabilityEvaluator()


@pytest.fixture
def scorer():
    """Create fitness scorer instance."""
    return FitnessScorer()


@pytest.fixture
def recommender():
    """Create capability recommender instance."""
    return CapabilityRecommender()


@pytest.mark.asyncio
class TestCapabilityEvaluationWorkflows:
    """Test end-to-end capability evaluation workflows."""

    async def test_complete_evaluation_workflow(
        self, evaluator, recommender
    ):
        """Test complete workflow from evaluation to recommendations."""
        # Step 1: Define task requirements
        task_requirements = [
            TaskRequirement(
                requirement_id="req-1",
                capability_type=CapabilityType.API,
                capability_name="api_client",
                required=True,
                weight=1.0,
            ),
            TaskRequirement(
                requirement_id="req-2",
                capability_type=CapabilityType.DATABASE,
                capability_name="database_query",
                required=True,
                weight=0.9,
            ),
        ]

        # Step 2: Define current agent capabilities
        current_capabilities = [
            {"id": "api_client", "name": "API Client"},
            {"id": "file_reader", "name": "File Reader"},
        ]

        agent_id = "agent-integration-test-001"
        task_type = "data_processing"

        # Step 3: Evaluate all capabilities
        fitness_scores = await evaluator.evaluate_all_capabilities(
            agent_id=agent_id,
            current_capabilities=current_capabilities,
            task_requirements=task_requirements,
            task_type=task_type,
        )

        assert len(fitness_scores) == 2
        assert "api_client" in fitness_scores
        assert "file_reader" in fitness_scores

        # Step 4: Identify capability gaps
        current_cap_ids = [c["id"] for c in current_capabilities]
        gaps = await evaluator.identify_capability_gaps(
            agent_id=agent_id,
            current_capabilities=current_cap_ids,
            task_requirements=task_requirements,
            fitness_scores=fitness_scores,
        )

        # Should identify database_query as missing
        assert len(gaps) > 0
        gap_capabilities = [g.required_capability for g in gaps]
        assert "database_query" in gap_capabilities

        # Step 5: Generate recommendations
        recommendation = await recommender.recommend_capability_changes(
            agent_id=agent_id,
            task_id=uuid4(),
            task_type=task_type,
            current_capabilities=current_cap_ids,
            fitness_scores=fitness_scores,
            capability_gaps=gaps,
        )

        # Should recommend adding database_query
        assert len(recommendation.capabilities_to_add) > 0
        assert "database_query" in recommendation.capabilities_to_add
        assert recommendation.confidence > 0.0
        assert recommendation.expected_improvement > 0.0

    async def test_high_performing_agent_workflow(
        self, evaluator, recommender
    ):
        """Test workflow for high-performing agent with no gaps."""
        # High-performing agent with all required capabilities
        task_requirements = [
            TaskRequirement(
                requirement_id="req-1",
                capability_type=CapabilityType.API,
                capability_name="api_client",
                required=True,
                weight=1.0,
            ),
        ]

        current_capabilities = [
            {"id": "api_client", "name": "API Client"},
            {"id": "database_query", "name": "Database Query"},
        ]

        # Simulate high performance history
        performance_history = {
            "total_executions": 200,
            "successful_executions": 195,
            "total_errors": 5,
            "execution_times": [300] * 200,
            "resource_usage": {
                "cpu_percent": 20.0,
                "memory_percent": 25.0,
            },
        }

        agent_id = "agent-high-performing-001"

        # Evaluate with performance history
        fitness_scores = {}
        for cap in current_capabilities:
            fitness = await evaluator.evaluate_fitness(
                agent_id=agent_id,
                capability_id=cap["id"],
                capability_name=cap["name"],
                task_requirements=task_requirements,
                performance_history=performance_history,
            )
            fitness_scores[cap["id"]] = fitness

        # All should have high fitness
        for fitness in fitness_scores.values():
            assert fitness.fitness_score > 0.5

        # Identify gaps (should be minimal)
        current_cap_ids = [c["id"] for c in current_capabilities]
        gaps = await evaluator.identify_capability_gaps(
            agent_id=agent_id,
            current_capabilities=current_cap_ids,
            task_requirements=task_requirements,
            fitness_scores=fitness_scores,
        )

        # Generate recommendations
        recommendation = await recommender.recommend_capability_changes(
            agent_id=agent_id,
            task_id=None,
            task_type=None,
            current_capabilities=current_cap_ids,
            fitness_scores=fitness_scores,
            capability_gaps=gaps,
        )

        # Should have minimal recommendations
        assert recommendation.recommendation_count == 0 or recommendation.risk_level == "low"

    async def test_underperforming_agent_workflow(
        self, evaluator, recommender
    ):
        """Test workflow for underperforming agent needing changes."""
        task_requirements = [
            TaskRequirement(
                requirement_id="req-1",
                capability_type=CapabilityType.API,
                capability_name="api_client",
                required=True,
                weight=1.0,
            ),
        ]

        current_capabilities = [
            {"id": "old_api_client", "name": "Old API Client"},
            {"id": "legacy_parser", "name": "Legacy Parser"},
        ]

        # Simulate poor performance
        poor_performance = {
            "total_executions": 50,
            "successful_executions": 20,
            "total_errors": 30,
            "execution_times": [5000] * 50,
            "resource_usage": {
                "cpu_percent": 80.0,
                "memory_percent": 75.0,
            },
        }

        agent_id = "agent-underperforming-001"

        # Evaluate with poor performance
        fitness_scores = {}
        for cap in current_capabilities:
            fitness = await evaluator.evaluate_fitness(
                agent_id=agent_id,
                capability_id=cap["id"],
                capability_name=cap["name"],
                performance_history=poor_performance,
            )
            fitness_scores[cap["id"]] = fitness

        # Should have low fitness scores
        low_fitness_count = sum(
            1 for f in fitness_scores.values() if f.fitness_score < 0.5
        )
        assert low_fitness_count > 0

        # Identify gaps
        current_cap_ids = [c["id"] for c in current_capabilities]
        gaps = await evaluator.identify_capability_gaps(
            agent_id=agent_id,
            current_capabilities=current_cap_ids,
            task_requirements=task_requirements,
            fitness_scores=fitness_scores,
        )

        # Generate recommendations
        recommendation = await recommender.recommend_capability_changes(
            agent_id=agent_id,
            task_id=uuid4(),
            task_type="api_integration",
            current_capabilities=current_cap_ids,
            fitness_scores=fitness_scores,
            capability_gaps=gaps,
        )

        # Should have recommendations
        assert recommendation.recommendation_count > 0
        # Should identify underperforming capabilities
        assert len(recommendation.underperforming_capabilities) > 0

    async def test_edge_case_no_capabilities(
        self, evaluator, recommender
    ):
        """Test workflow with agent having no capabilities."""
        task_requirements = [
            TaskRequirement(
                requirement_id="req-1",
                capability_type=CapabilityType.API,
                capability_name="api_client",
                required=True,
                weight=1.0,
            ),
        ]

        agent_id = "agent-no-caps-001"
        current_capabilities = []

        # Evaluate empty capabilities
        fitness_scores = await evaluator.evaluate_all_capabilities(
            agent_id=agent_id,
            current_capabilities=current_capabilities,
            task_requirements=task_requirements,
        )

        assert len(fitness_scores) == 0

        # Identify gaps
        gaps = await evaluator.identify_capability_gaps(
            agent_id=agent_id,
            current_capabilities=[],
            task_requirements=task_requirements,
            fitness_scores=fitness_scores,
        )

        # Should identify all requirements as gaps
        assert len(gaps) >= len(task_requirements)

        # Generate recommendations
        recommendation = await recommender.recommend_capability_changes(
            agent_id=agent_id,
            task_id=None,
            task_type=None,
            current_capabilities=[],
            fitness_scores=fitness_scores,
            capability_gaps=gaps,
        )

        # Should recommend adding all required capabilities
        assert len(recommendation.capabilities_to_add) > 0

    async def test_edge_case_no_requirements(
        self, evaluator, recommender
    ):
        """Test workflow with no task requirements."""
        current_capabilities = [
            {"id": "api_client", "name": "API Client"},
        ]

        agent_id = "agent-no-reqs-001"

        # Evaluate without requirements
        fitness_scores = await evaluator.evaluate_all_capabilities(
            agent_id=agent_id,
            current_capabilities=current_capabilities,
            task_requirements=[],
        )

        # Should evaluate all capabilities
        assert len(fitness_scores) == 1

        # Identify gaps (should be none)
        gaps = await evaluator.identify_capability_gaps(
            agent_id=agent_id,
            current_capabilities=["api_client"],
            task_requirements=[],
            fitness_scores=fitness_scores,
        )

        assert len(gaps) == 0

        # Generate recommendations
        recommendation = await recommender.recommend_capability_changes(
            agent_id=agent_id,
            task_id=None,
            task_type=None,
            current_capabilities=["api_client"],
            fitness_scores=fitness_scores,
            capability_gaps=gaps,
        )

        # Should have no gap-based additions
        assert len(recommendation.capabilities_to_add) == 0

    async def test_performance_latency(self, evaluator):
        """Test that evaluation meets performance targets (<100ms)."""
        import time

        start_time = time.time()

        fitness = await evaluator.evaluate_fitness(
            agent_id="agent-perf-test-001",
            capability_id="test_cap",
            capability_name="Test Capability",
        )

        duration_ms = (time.time() - start_time) * 1000

        # Should be under 100ms target
        assert duration_ms < 100

    async def test_cache_effectiveness(self, evaluator):
        """Test that caching improves performance."""
        import time

        # First evaluation (cold cache)
        start1 = time.time()
        fitness1 = await evaluator.evaluate_fitness(
            agent_id="agent-cache-test-001",
            capability_id="cached_cap",
            capability_name="Cached Capability",
        )
        duration1_ms = (time.time() - start1) * 1000

        # Second evaluation (warm cache)
        start2 = time.time()
        fitness2 = await evaluator.evaluate_fitness(
            agent_id="agent-cache-test-001",
            capability_id="cached_cap",
            capability_name="Cached Capability",
        )
        duration2_ms = (time.time() - start2) * 1000

        # Cached evaluation should be faster
        assert duration2_ms < duration1_ms
        # Results should be identical (from cache)
        assert fitness1.evaluated_at == fitness2.evaluated_at
