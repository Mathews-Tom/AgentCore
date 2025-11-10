"""
Tests for Planner Module Implementation

Validates task decomposition, plan generation, and plan refinement functionality.
"""

from __future__ import annotations

import pytest

from agentcore.modular.interfaces import PlannerQuery, PlanRefinement
from agentcore.modular.models import PlanStatus, StepStatus
from agentcore.modular.planner import Planner


class TestPlannerInitialization:
    """Test Planner module initialization."""

    def test_planner_default_initialization(self) -> None:
        """Test Planner with default configuration."""
        planner = Planner()

        assert planner.module_name == "Planner"
        assert planner.max_steps == 20
        assert planner.enable_parallel is False
        assert planner.state is not None

    def test_planner_custom_configuration(self) -> None:
        """Test Planner with custom configuration."""
        planner = Planner(max_steps=10, enable_parallel=True)

        assert planner.max_steps == 10
        assert planner.enable_parallel is True

    @pytest.mark.asyncio
    async def test_planner_health_check(self) -> None:
        """Test Planner health check."""
        planner = Planner()
        health = await planner.health_check()

        assert health["status"] == "healthy"
        assert health["module"] == "Planner"
        assert health["max_steps"] == 20
        assert "execution_id" in health


class TestQueryAnalysis:
    """Test query analysis and plan generation."""

    @pytest.mark.asyncio
    async def test_analyze_empty_query(self) -> None:
        """Test that empty query raises ValueError."""
        planner = Planner()
        query = PlannerQuery(query="")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await planner.analyze_query(query)

    @pytest.mark.asyncio
    async def test_analyze_whitespace_query(self) -> None:
        """Test that whitespace-only query raises ValueError."""
        planner = Planner()
        query = PlannerQuery(query="   ")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await planner.analyze_query(query)

    @pytest.mark.asyncio
    async def test_analyze_simple_query(self) -> None:
        """Test simple query generates single-step plan."""
        planner = Planner()
        query = PlannerQuery(query="Hello world")

        plan = await planner.analyze_query(query)

        assert plan.plan_id is not None
        assert plan.query == "Hello world"
        assert len(plan.steps) == 1
        assert plan.status == PlanStatus.PENDING
        assert plan.steps[0].step_id == "execute_1"
        assert plan.steps[0].status == StepStatus.PENDING

    @pytest.mark.asyncio
    async def test_analyze_information_retrieval_query(self) -> None:
        """Test information retrieval query generates search plan."""
        planner = Planner()
        query = PlannerQuery(query="What is the capital of France?")

        plan = await planner.analyze_query(query)

        assert plan.plan_id is not None
        assert len(plan.steps) == 2  # Search + Extract
        assert plan.steps[0].action == "search_knowledge"
        assert plan.steps[1].action == "extract_information"
        assert plan.steps[1].dependencies is not None
        assert len(plan.steps[1].dependencies) == 1

    @pytest.mark.asyncio
    async def test_analyze_computation_query(self) -> None:
        """Test computation query generates calculation plan."""
        planner = Planner()
        query = PlannerQuery(query="Calculate the sum of 10 and 20")

        plan = await planner.analyze_query(query)

        assert plan.plan_id is not None
        assert len(plan.steps) == 2  # Gather + Compute
        assert plan.steps[0].action == "gather_data"
        assert plan.steps[1].action == "perform_computation"
        assert len(plan.steps[1].tool_requirements) > 0
        assert plan.steps[1].tool_requirements[0].tool_name == "calculator"

    @pytest.mark.asyncio
    async def test_analyze_multi_step_query(self) -> None:
        """Test multi-step query generates sequential plan."""
        planner = Planner()
        query = PlannerQuery(
            query="First search for data, then analyze it, and finally report results"
        )

        plan = await planner.analyze_query(query)

        assert plan.plan_id is not None
        assert len(plan.steps) == 3  # Multi-step decomposition
        # Check dependencies are sequential
        assert plan.steps[1].dependencies is not None
        assert plan.steps[1].dependencies[0].step_id == "step_1"
        assert plan.steps[2].dependencies is not None
        assert plan.steps[2].dependencies[0].step_id == "step_2"

    @pytest.mark.asyncio
    async def test_plan_exceeds_max_steps(self) -> None:
        """Test that plans exceeding max_steps raise ValueError."""
        planner = Planner(max_steps=1)
        query = PlannerQuery(
            query="What is the capital of France and what is its population?"
        )

        with pytest.raises(ValueError, match="Plan exceeds maximum steps"):
            await planner.analyze_query(query)


class TestPlanGeneration:
    """Test detailed plan generation features."""

    @pytest.mark.asyncio
    async def test_plan_has_success_criteria(self) -> None:
        """Test that generated plans include success criteria."""
        planner = Planner()
        query = PlannerQuery(query="Test query")

        plan = await planner.analyze_query(query)

        assert plan.success_criteria is not None
        assert len(plan.success_criteria.criteria) > 0
        assert any(
            c.criterion_id == "query_answered"
            for c in plan.success_criteria.criteria
        )

    @pytest.mark.asyncio
    async def test_plan_has_timestamps(self) -> None:
        """Test that plans include creation timestamps."""
        planner = Planner()
        query = PlannerQuery(query="Test query")

        plan = await planner.analyze_query(query)

        assert plan.created_at is not None
        assert "T" in plan.created_at  # ISO format

    @pytest.mark.asyncio
    async def test_plan_respects_max_iterations(self) -> None:
        """Test that plans respect max_iterations constraint."""
        planner = Planner()
        query = PlannerQuery(
            query="Test query",
            constraints={"max_iterations": 3},
        )

        plan = await planner.analyze_query(query)

        assert plan.max_iterations == 3

    @pytest.mark.asyncio
    async def test_plan_default_max_iterations(self) -> None:
        """Test default max_iterations value."""
        planner = Planner()
        query = PlannerQuery(query="Test query")

        plan = await planner.analyze_query(query)

        assert plan.max_iterations == 5  # Default value

    @pytest.mark.asyncio
    async def test_steps_have_required_fields(self) -> None:
        """Test that all steps have required fields (step_id, action, parameters)."""
        planner = Planner()
        query = PlannerQuery(query="What is machine learning?")

        plan = await planner.analyze_query(query)

        for step in plan.steps:
            assert step.step_id is not None
            assert len(step.step_id) > 0
            assert step.action is not None
            assert len(step.action) > 0
            assert step.parameters is not None
            assert isinstance(step.parameters, dict)


class TestPlanRefinement:
    """Test plan refinement functionality."""

    @pytest.mark.asyncio
    async def test_refine_plan_basic(self) -> None:
        """Test basic plan refinement."""
        planner = Planner()

        # Create refinement request
        refinement = PlanRefinement(
            plan_id="original-plan-123",
            feedback="Add more validation steps",
            constraints={"original_query": "Test query for refinement"},
        )

        refined_plan = await planner.refine_plan(refinement)

        assert refined_plan.plan_id != "original-plan-123"  # New plan ID
        assert refined_plan.query == "Test query for refinement"

    @pytest.mark.asyncio
    async def test_refine_plan_without_original_query(self) -> None:
        """Test refinement fails without original query."""
        planner = Planner()

        refinement = PlanRefinement(
            plan_id="original-plan-123",
            feedback="Add validation",
            constraints={},  # Missing original_query
        )

        with pytest.raises(ValueError, match="Refinement requires original_query"):
            await planner.refine_plan(refinement)

    @pytest.mark.asyncio
    async def test_refine_plan_includes_feedback_context(self) -> None:
        """Test that refinement includes feedback in context."""
        planner = Planner()

        refinement = PlanRefinement(
            plan_id="plan-456",
            feedback="Increase validation thoroughness",
            constraints={
                "original_query": "Validate data",
                "extra_param": "value",
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Plan should be generated with refinement context
        assert refined_plan.plan_id is not None
        assert refined_plan.query == "Validate data"


class TestConvenienceMethods:
    """Test convenience methods for plan creation."""

    @pytest.mark.asyncio
    async def test_create_plan_from_string(self) -> None:
        """Test create_plan convenience method."""
        planner = Planner()

        plan = await planner.create_plan("Simple query string")

        assert plan.plan_id is not None
        assert plan.query == "Simple query string"
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_create_plan_with_context(self) -> None:
        """Test create_plan with additional context."""
        planner = Planner()

        plan = await planner.create_plan(
            "Query with context",
            query_context={"user_id": "123", "session": "abc"},
        )

        assert plan.plan_id is not None
        assert plan.query == "Query with context"

    @pytest.mark.asyncio
    async def test_create_plan_empty_string(self) -> None:
        """Test create_plan with empty string raises ValueError."""
        planner = Planner()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await planner.create_plan("")


class TestQueryClassification:
    """Test query classification helpers."""

    def test_information_retrieval_classification(self) -> None:
        """Test information retrieval query detection."""
        planner = Planner()

        assert planner._is_information_retrieval("what is python")
        assert planner._is_information_retrieval("find the documentation")
        assert planner._is_information_retrieval("search for examples")
        assert not planner._is_information_retrieval("calculate sum")

    def test_computation_classification(self) -> None:
        """Test computation query detection."""
        planner = Planner()

        assert planner._is_computation_task("calculate the total")
        assert planner._is_computation_task("compute average")
        assert planner._is_computation_task("count the items")
        assert not planner._is_computation_task("find the answer")

    def test_multi_step_classification(self) -> None:
        """Test multi-step query detection."""
        planner = Planner()

        assert planner._is_multi_step_task("first do this and then that")
        assert planner._is_multi_step_task("step 1. do something. step 2. do more.")
        assert planner._is_multi_step_task("after searching, analyze the results")
        assert not planner._is_multi_step_task("simple query")
