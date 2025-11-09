"""
Unit Tests for CapabilityRecommender - ACE-027

Tests capability recommendation generation and risk assessment.
Target: 95%+ code coverage
"""

import pytest
from uuid import uuid4

from agentcore.ace.capability.recommender import CapabilityRecommender
from agentcore.ace.models.ace_models import (
    CapabilityFitness,
    CapabilityGap,
    CapabilityType,
    FitnessMetrics,
)


@pytest.fixture
def recommender():
    """Create capability recommender instance."""
    return CapabilityRecommender()


@pytest.fixture
def high_fitness_capability():
    """Create high-fitness capability."""
    return CapabilityFitness(
        capability_id="api_client",
        capability_name="API Client",
        agent_id="agent-001",
        fitness_score=0.9,
        coverage_score=0.95,
        performance_score=0.85,
        metrics=FitnessMetrics(
            success_rate=0.95,
            error_correlation=0.05,
            usage_frequency=100,
            avg_execution_time_ms=300.0,
        ),
        sample_size=100,
    )


@pytest.fixture
def low_fitness_capability():
    """Create low-fitness capability."""
    return CapabilityFitness(
        capability_id="old_parser",
        capability_name="Old Parser",
        agent_id="agent-001",
        fitness_score=0.25,
        coverage_score=0.3,
        performance_score=0.2,
        metrics=FitnessMetrics(
            success_rate=0.3,
            error_correlation=0.7,
            usage_frequency=5,
            avg_execution_time_ms=5000.0,
        ),
        sample_size=5,
    )


@pytest.fixture
def capability_gaps():
    """Create sample capability gaps."""
    return [
        CapabilityGap(
            required_capability="database_query",
            capability_type=CapabilityType.DATABASE,
            current_fitness=None,
            required_fitness=0.5,
            impact=1.0,
            gap_severity="critical",
            mitigation_suggestion="Add database_query capability",
        ),
        CapabilityGap(
            required_capability="file_reader",
            capability_type=CapabilityType.FILE_SYSTEM,
            current_fitness=None,
            required_fitness=0.5,
            impact=0.6,
            gap_severity="high",
            mitigation_suggestion="Add file_reader capability",
        ),
    ]


@pytest.mark.asyncio
class TestCapabilityRecommender:
    """Test suite for CapabilityRecommender."""

    async def test_recommend_capability_changes_basic(
        self, recommender, high_fitness_capability, capability_gaps
    ):
        """Test basic capability change recommendations."""
        fitness_scores = {"api_client": high_fitness_capability}
        current_capabilities = ["api_client"]

        recommendation = await recommender.recommend_capability_changes(
            agent_id="agent-001",
            task_id=uuid4(),
            task_type="data_processing",
            current_capabilities=current_capabilities,
            fitness_scores=fitness_scores,
            capability_gaps=capability_gaps,
        )

        assert recommendation is not None
        assert recommendation.agent_id == "agent-001"
        assert 0.0 <= recommendation.confidence <= 1.0
        assert 0.0 <= recommendation.expected_improvement <= 1.0
        assert recommendation.risk_level in ["low", "medium", "high"]

    async def test_recommend_additions(
        self, recommender, high_fitness_capability, capability_gaps
    ):
        """Test capability addition recommendations."""
        fitness_scores = {"api_client": high_fitness_capability}
        current_capabilities = ["api_client"]

        recommendation = await recommender.recommend_capability_changes(
            agent_id="agent-001",
            task_id=None,
            task_type=None,
            current_capabilities=current_capabilities,
            fitness_scores=fitness_scores,
            capability_gaps=capability_gaps,
        )

        # Should recommend adding missing capabilities
        assert len(recommendation.capabilities_to_add) > 0
        assert "database_query" in recommendation.capabilities_to_add

    async def test_recommend_removals(
        self, recommender, low_fitness_capability
    ):
        """Test capability removal recommendations."""
        fitness_scores = {"old_parser": low_fitness_capability}
        current_capabilities = ["old_parser"]

        recommendation = await recommender.recommend_capability_changes(
            agent_id="agent-001",
            task_id=None,
            task_type=None,
            current_capabilities=current_capabilities,
            fitness_scores=fitness_scores,
            capability_gaps=[],
        )

        # Should recommend removing underperforming capability
        assert len(recommendation.capabilities_to_remove) > 0 or len(recommendation.underperforming_capabilities) > 0

    async def test_recommend_upgrades(self, recommender):
        """Test capability upgrade recommendations."""
        moderate_fitness = CapabilityFitness(
            capability_id="moderate_cap",
            capability_name="Moderate Capability",
            agent_id="agent-001",
            fitness_score=0.4,  # Between removal and fitness threshold
            coverage_score=0.5,
            performance_score=0.4,
            metrics=FitnessMetrics(
                success_rate=0.5,
                error_correlation=0.3,
                usage_frequency=50,
                avg_execution_time_ms=1000.0,
            ),
            sample_size=50,
        )

        fitness_scores = {"moderate_cap": moderate_fitness}
        current_capabilities = ["moderate_cap"]

        alternatives = {
            "custom": ["moderate_cap_v2", "better_cap"]
        }

        recommendation = await recommender.recommend_capability_changes(
            agent_id="agent-001",
            task_id=None,
            task_type=None,
            current_capabilities=current_capabilities,
            fitness_scores=fitness_scores,
            capability_gaps=[],
            alternatives_available=alternatives,
        )

        # Should recommend upgrade
        assert len(recommendation.capabilities_to_upgrade) > 0 or "moderate_cap" in recommendation.underperforming_capabilities

    async def test_identify_underperforming(
        self, recommender, high_fitness_capability, low_fitness_capability
    ):
        """Test identification of underperforming capabilities."""
        fitness_scores = {
            "high": high_fitness_capability,
            "low": low_fitness_capability,
        }

        underperforming = recommender._identify_underperforming(fitness_scores)

        assert "low" in underperforming
        assert "high" not in underperforming

    async def test_recommend_additions_filters_existing(
        self, recommender, capability_gaps
    ):
        """Test that additions don't recommend existing capabilities."""
        current_capabilities = ["database_query", "api_client"]

        to_add = recommender._recommend_additions(
            capability_gaps, current_capabilities
        )

        # Should not recommend database_query since it exists
        assert "database_query" not in to_add

    async def test_recommend_removals_protects_required(
        self, recommender, low_fitness_capability, capability_gaps
    ):
        """Test that removals don't remove required capabilities."""
        fitness_scores = {"database_query": low_fitness_capability}
        underperforming = ["database_query"]

        to_remove = recommender._recommend_removals(
            underperforming, fitness_scores, capability_gaps
        )

        # Should not remove critical required capability
        # (database_query is in critical gaps)
        assert "database_query" not in to_remove

    async def test_evaluate_alternatives(self, recommender, capability_gaps):
        """Test alternative capability evaluation."""
        alternatives = {
            "database": ["postgres_client", "mysql_client"],
            "file_system": ["advanced_file_reader"],
        }

        evaluated = recommender._evaluate_alternatives(
            capability_gaps, alternatives
        )

        # Should evaluate alternatives
        assert len(evaluated) > 0

        # All scores should be 0-1
        for score in evaluated.values():
            assert 0.0 <= score <= 1.0

    async def test_compute_recommendation_confidence_high(
        self, recommender, high_fitness_capability, capability_gaps
    ):
        """Test confidence computation with high sample size."""
        fitness_scores = {"api_client": high_fitness_capability}

        confidence = recommender._compute_recommendation_confidence(
            fitness_scores, capability_gaps, [], []
        )

        # High sample size should give higher confidence
        assert confidence > 0.4

    async def test_compute_recommendation_confidence_low(
        self, recommender, low_fitness_capability
    ):
        """Test confidence computation with low sample size."""
        fitness_scores = {"old_parser": low_fitness_capability}

        confidence = recommender._compute_recommendation_confidence(
            fitness_scores, [], [], []
        )

        # Low sample size should give lower confidence
        assert confidence < 0.8

    async def test_estimate_improvement(
        self, recommender, low_fitness_capability, capability_gaps
    ):
        """Test expected improvement estimation."""
        fitness_scores = {"low": low_fitness_capability}

        improvement = recommender._estimate_improvement(
            fitness_scores,
            to_add=["database_query"],
            to_remove=["low"],
            gaps=capability_gaps,
        )

        # Should estimate some improvement
        assert 0.0 <= improvement <= 1.0
        assert improvement > 0.0

    async def test_assess_risk_low(self, recommender):
        """Test risk assessment for low-risk changes."""
        risk = recommender._assess_risk(
            to_add=["new_cap"],
            to_remove=[],
            to_upgrade=[],
        )

        assert risk == "low"

    async def test_assess_risk_medium(self, recommender):
        """Test risk assessment for medium-risk changes."""
        risk = recommender._assess_risk(
            to_add=["cap1", "cap2"],
            to_remove=["old_cap"],
            to_upgrade=[],
        )

        assert risk in ["medium", "high"]

    async def test_assess_risk_high(self, recommender):
        """Test risk assessment for high-risk changes."""
        risk = recommender._assess_risk(
            to_add=["cap1", "cap2", "cap3"],
            to_remove=["old1", "old2"],
            to_upgrade=[{"capability": "cap4"}],
        )

        assert risk == "high"

    async def test_generate_rationale(
        self, recommender, high_fitness_capability, low_fitness_capability, capability_gaps
    ):
        """Test rationale generation."""
        fitness_scores = {
            "high": high_fitness_capability,
            "low": low_fitness_capability,
        }

        rationale = recommender._generate_rationale(
            fitness_scores,
            capability_gaps,
            to_add=["database_query"],
            to_remove=["low"],
            to_upgrade=[],
        )

        # Should contain relevant information
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "fitness" in rationale.lower() or "gap" in rationale.lower()

    async def test_generate_rationale_no_changes(
        self, recommender, high_fitness_capability
    ):
        """Test rationale when no changes recommended."""
        fitness_scores = {"high": high_fitness_capability}

        rationale = recommender._generate_rationale(
            fitness_scores,
            gaps=[],
            to_add=[],
            to_remove=[],
            to_upgrade=[],
        )

        # Should indicate no changes needed
        assert "No" in rationale or "no" in rationale

    async def test_recommendation_has_critical_gaps(
        self, recommender, high_fitness_capability, capability_gaps
    ):
        """Test recommendation critical gap detection."""
        fitness_scores = {"api_client": high_fitness_capability}

        recommendation = await recommender.recommend_capability_changes(
            agent_id="agent-001",
            task_id=None,
            task_type=None,
            current_capabilities=["api_client"],
            fitness_scores=fitness_scores,
            capability_gaps=capability_gaps,
        )

        # Should detect critical gaps
        assert recommendation.has_critical_gaps is True

    async def test_recommendation_count(
        self, recommender, high_fitness_capability, capability_gaps
    ):
        """Test recommendation count property."""
        fitness_scores = {"api_client": high_fitness_capability}

        recommendation = await recommender.recommend_capability_changes(
            agent_id="agent-001",
            task_id=None,
            task_type=None,
            current_capabilities=["api_client"],
            fitness_scores=fitness_scores,
            capability_gaps=capability_gaps,
        )

        # Should count all recommended changes
        count = recommendation.recommendation_count
        assert count == (
            len(recommendation.capabilities_to_add)
            + len(recommendation.capabilities_to_remove)
            + len(recommendation.capabilities_to_upgrade)
        )

    async def test_custom_thresholds(self):
        """Test recommender with custom thresholds."""
        custom_recommender = CapabilityRecommender(
            fitness_threshold=0.6,
            removal_threshold=0.2,
            confidence_threshold=0.8,
        )

        assert custom_recommender.fitness_threshold == 0.6
        assert custom_recommender.removal_threshold == 0.2
        assert custom_recommender.confidence_threshold == 0.8

    async def test_recommendation_model_properties(self):
        """Test CapabilityRecommendation model properties."""
        from agentcore.ace.models.ace_models import CapabilityRecommendation

        rec = CapabilityRecommendation(
            agent_id="agent-001",
            current_capabilities=["cap1"],
            identified_gaps=[
                CapabilityGap(
                    required_capability="cap2",
                    capability_type=CapabilityType.API,
                    impact=1.0,
                    gap_severity="critical",
                )
            ],
            rationale="Test rationale",
            confidence=0.9,
            expected_improvement=0.3,
        )

        assert rec.has_critical_gaps is True
        assert rec.recommendation_count == 0  # No changes specified
