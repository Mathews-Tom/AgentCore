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


class TestFeedbackAnalysis:
    """Test feedback analysis for plan refinement."""

    def test_analyze_feedback_error_handling(self) -> None:
        """Test feedback analysis detects error handling needs."""
        planner = Planner()

        feedback = "Step 2 failed with an exception during execution"
        analysis = planner._analyze_feedback(feedback)

        assert "add_error_handling" in analysis
        assert analysis["add_error_handling"]["detected"] is True
        assert "failed" in analysis["add_error_handling"]["keywords"]

    def test_analyze_feedback_parameter_adjustment(self) -> None:
        """Test feedback analysis detects parameter adjustment needs."""
        planner = Planner()

        feedback = "Invalid parameter value provided to step 3"
        analysis = planner._analyze_feedback(feedback)

        assert "adjust_parameters" in analysis
        assert analysis["adjust_parameters"]["detected"] is True

    def test_analyze_feedback_validation_needs(self) -> None:
        """Test feedback analysis detects validation step needs."""
        planner = Planner()

        feedback = "Missing validation for output format"
        analysis = planner._analyze_feedback(feedback)

        assert "add_validation" in analysis
        assert analysis["add_validation"]["detected"] is True
        assert "missing validation" in analysis["add_validation"]["keywords"]

    def test_analyze_feedback_reordering_needs(self) -> None:
        """Test feedback analysis detects step reordering needs."""
        planner = Planner()

        feedback = "Steps executed in wrong order, step 2 should come before step 1"
        analysis = planner._analyze_feedback(feedback)

        assert "reorder_steps" in analysis
        assert analysis["reorder_steps"]["detected"] is True
        assert "before" in analysis["reorder_steps"]["keywords"]

    def test_analyze_feedback_multiple_strategies(self) -> None:
        """Test feedback analysis detects multiple strategies."""
        planner = Planner()

        feedback = "Step failed due to invalid parameter and missing validation"
        analysis = planner._analyze_feedback(feedback)

        assert "add_error_handling" in analysis
        assert "adjust_parameters" in analysis
        assert "add_validation" in analysis


class TestIntelligentPlanRefinement:
    """Test intelligent plan refinement strategies."""

    @pytest.mark.asyncio
    async def test_refine_plan_with_error_handling(self) -> None:
        """Test plan refinement adds error handling for failed steps."""
        planner = Planner()

        # Create initial plan
        initial_plan = await planner.create_plan("Test query")
        plan_dict = initial_plan.model_dump()

        # Mark first step as failed
        plan_dict["steps"][0]["status"] = "failed"
        plan_dict["steps"][0]["error"] = "Execution timeout"

        # Create refinement request
        refinement = PlanRefinement(
            plan_id=plan_dict["plan_id"],
            feedback="Step execute_1 failed with timeout error",
            constraints={
                "original_query": "Test query",
                "existing_plan": plan_dict,
                "failed_step_ids": ["execute_1"],
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify error handling was added
        refined_step = refined_plan.steps[0]
        assert refined_step.max_retries > initial_plan.steps[0].max_retries
        assert refined_step.status == StepStatus.PENDING
        assert refined_step.error is None

    @pytest.mark.asyncio
    async def test_refine_plan_with_parameter_adjustment(self) -> None:
        """Test plan refinement adjusts parameters."""
        planner = Planner()

        # Create initial plan
        initial_plan = await planner.create_plan("Calculate sum")
        plan_dict = initial_plan.model_dump()

        # Create refinement with parameter adjustments
        refinement = PlanRefinement(
            plan_id=plan_dict["plan_id"],
            feedback="Invalid parameter value in step gather_1",
            constraints={
                "original_query": "Calculate sum",
                "existing_plan": plan_dict,
                "parameter_adjustments": {
                    "gather_1": {"timeout": 60, "retry_delay": 5}
                },
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify parameters were adjusted
        gather_step = refined_plan.get_step_by_id("gather_1")
        assert gather_step is not None
        assert gather_step.parameters["timeout"] == 60
        assert gather_step.parameters["retry_delay"] == 5

    @pytest.mark.asyncio
    async def test_refine_plan_with_validation_steps(self) -> None:
        """Test plan refinement inserts validation steps."""
        planner = Planner()

        # Create initial plan
        initial_plan = await planner.create_plan("Search for data")
        plan_dict = initial_plan.model_dump()
        initial_step_count = len(plan_dict["steps"])

        # Create refinement requesting validation
        refinement = PlanRefinement(
            plan_id=plan_dict["plan_id"],
            feedback="Missing validation after search step",
            constraints={
                "original_query": "Search for data",
                "existing_plan": plan_dict,
                "steps_needing_validation": ["search_1"],
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify validation step was inserted
        assert len(refined_plan.steps) > initial_step_count
        validation_steps = [s for s in refined_plan.steps if "validation" in s.step_id]
        assert len(validation_steps) > 0
        assert validation_steps[0].action == "validate_result"

    @pytest.mark.asyncio
    async def test_refine_plan_with_step_reordering(self) -> None:
        """Test plan refinement reorders steps."""
        planner = Planner()

        # Create multi-step plan
        initial_plan = await planner.create_plan(
            "First search, then analyze, finally report"
        )
        plan_dict = initial_plan.model_dump()

        original_order = [s["step_id"] for s in plan_dict["steps"]]
        new_order = list(reversed(original_order))  # Reverse the order

        # Create refinement with new order
        refinement = PlanRefinement(
            plan_id=plan_dict["plan_id"],
            feedback="Steps should be executed in different order",
            constraints={
                "original_query": "First search, then analyze, finally report",
                "existing_plan": plan_dict,
                "step_order": new_order,
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify steps were reordered
        refined_order = [s.step_id for s in refined_plan.steps]
        assert refined_order == new_order

    @pytest.mark.asyncio
    async def test_refine_plan_with_multiple_strategies(self) -> None:
        """Test plan refinement applies multiple strategies."""
        planner = Planner()

        # Create initial plan
        initial_plan = await planner.create_plan("Complex query")
        plan_dict = initial_plan.model_dump()

        # Mark step as failed
        plan_dict["steps"][0]["status"] = "failed"

        # Create refinement with multiple strategies
        refinement = PlanRefinement(
            plan_id=plan_dict["plan_id"],
            feedback="Step failed with invalid parameter and needs validation",
            constraints={
                "original_query": "Complex query",
                "existing_plan": plan_dict,
                "failed_step_ids": ["execute_1"],
                "parameter_adjustments": {"execute_1": {"timeout": 120}},
                "steps_needing_validation": ["execute_1"],
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify all strategies were applied
        # 1. Error handling: max_retries increased
        assert refined_plan.steps[0].max_retries > initial_plan.steps[0].max_retries
        # 2. Parameter adjustment: timeout updated
        assert refined_plan.steps[0].parameters.get("timeout") == 120
        # 3. Validation: validation step inserted
        validation_steps = [s for s in refined_plan.steps if "validation" in s.step_id]
        assert len(validation_steps) > 0

    @pytest.mark.asyncio
    async def test_refine_plan_without_existing_plan_fallback(self) -> None:
        """Test plan refinement falls back to regeneration without existing plan."""
        planner = Planner()

        # Create refinement without existing plan
        refinement = PlanRefinement(
            plan_id="original-plan-123",
            feedback="Plan needs improvement",
            constraints={"original_query": "Test query"},
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify new plan was generated
        assert refined_plan.plan_id != "original-plan-123"
        assert refined_plan.query == "Test query"
        assert len(refined_plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_refine_plan_updates_metadata(self) -> None:
        """Test plan refinement updates metadata correctly."""
        planner = Planner()

        # Create initial plan
        initial_plan = await planner.create_plan("Test query")
        plan_dict = initial_plan.model_dump()

        # Create refinement
        refinement = PlanRefinement(
            plan_id=plan_dict["plan_id"],
            feedback="Needs improvement",
            constraints={
                "original_query": "Test query",
                "existing_plan": plan_dict,
            },
        )

        refined_plan = await planner.refine_plan(refinement)

        # Verify metadata updates
        assert refined_plan.parent_plan_id == plan_dict["plan_id"]
        assert refined_plan.current_iteration == initial_plan.current_iteration + 1
        assert refined_plan.plan_id != plan_dict["plan_id"]
