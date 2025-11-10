"""
Tests for Execution Plan Data Models

Validates enhanced models for execution planning, tracking, and verification.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from agentcore.modular.models import (
    # Enums
    PlanStatus,
    StepStatus,
    ModuleType,
    VerificationLevel,
    # Models
    ToolRequirement,
    StepDependency,
    SuccessCriterion,
    SuccessCriteria,
    EnhancedPlanStep,
    EnhancedExecutionPlan,
    ModuleTransition,
    EnhancedVerificationResult,
)


class TestEnumerations:
    """Test enum definitions."""

    def test_plan_status_values(self) -> None:
        """Test PlanStatus enum values."""
        assert PlanStatus.PENDING == "pending"
        assert PlanStatus.IN_PROGRESS == "in_progress"
        assert PlanStatus.COMPLETED == "completed"
        assert PlanStatus.FAILED == "failed"
        assert PlanStatus.CANCELLED == "cancelled"

    def test_step_status_values(self) -> None:
        """Test StepStatus enum values."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.IN_PROGRESS == "in_progress"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"

    def test_module_type_values(self) -> None:
        """Test ModuleType enum values."""
        assert ModuleType.PLANNER == "planner"
        assert ModuleType.EXECUTOR == "executor"
        assert ModuleType.VERIFIER == "verifier"
        assert ModuleType.GENERATOR == "generator"

    def test_verification_level_values(self) -> None:
        """Test VerificationLevel enum values."""
        assert VerificationLevel.NONE == "none"
        assert VerificationLevel.BASIC == "basic"
        assert VerificationLevel.STANDARD == "standard"
        assert VerificationLevel.STRICT == "strict"
        assert VerificationLevel.PARANOID == "paranoid"


class TestToolRequirement:
    """Test ToolRequirement model."""

    def test_tool_requirement_creation(self) -> None:
        """Test creating a tool requirement."""
        tool = ToolRequirement(
            tool_name="calculator",
            version="1.0.0",
            parameters={"precision": 10},
            optional=False,
        )
        assert tool.tool_name == "calculator"
        assert tool.version == "1.0.0"
        assert tool.parameters["precision"] == 10
        assert tool.optional is False

    def test_tool_requirement_defaults(self) -> None:
        """Test tool requirement default values."""
        tool = ToolRequirement(tool_name="search")
        assert tool.version is None
        assert tool.parameters == {}
        assert tool.optional is False


class TestStepDependency:
    """Test StepDependency model."""

    def test_step_dependency_creation(self) -> None:
        """Test creating a step dependency."""
        dep = StepDependency(
            step_id="step-1",
            dependency_type="data",
            required=True,
        )
        assert dep.step_id == "step-1"
        assert dep.dependency_type == "data"
        assert dep.required is True

    def test_step_dependency_defaults(self) -> None:
        """Test step dependency defaults."""
        dep = StepDependency(step_id="step-2")
        assert dep.dependency_type == "sequential"
        assert dep.required is True


class TestSuccessCriterion:
    """Test SuccessCriterion model."""

    def test_success_criterion_creation(self) -> None:
        """Test creating a success criterion."""
        criterion = SuccessCriterion(
            description="Accuracy must be above threshold",
            metric_name="accuracy",
            operator="gt",
            threshold=0.9,
            weight=1.0,
            required=True,
        )
        assert criterion.description == "Accuracy must be above threshold"
        assert criterion.metric_name == "accuracy"
        assert criterion.operator == "gt"
        assert criterion.threshold == 0.9

    def test_success_criterion_invalid_operator(self) -> None:
        """Test that invalid operator raises error."""
        with pytest.raises(Exception):  # Pydantic validation error
            SuccessCriterion(
                description="Test",
                metric_name="metric",
                operator="invalid",
                threshold=1.0,
            )

    def test_success_criterion_valid_operators(self) -> None:
        """Test all valid operators."""
        valid_ops = ["gt", "lt", "eq", "gte", "lte", "in", "contains", "ne"]
        for op in valid_ops:
            criterion = SuccessCriterion(
                description="Test",
                metric_name="metric",
                operator=op,
                threshold=1.0,
            )
            assert criterion.operator == op


class TestSuccessCriteria:
    """Test SuccessCriteria model."""

    def test_success_criteria_creation(self) -> None:
        """Test creating success criteria."""
        criterion = SuccessCriterion(
            description="Test",
            metric_name="score",
            operator="gt",
            threshold=0.8,
        )
        criteria = SuccessCriteria(
            criteria=[criterion],
            aggregation_method="weighted_average",
            minimum_score=0.7,
        )
        assert len(criteria.criteria) == 1
        assert criteria.aggregation_method == "weighted_average"
        assert criteria.minimum_score == 0.7

    def test_success_criteria_invalid_aggregation(self) -> None:
        """Test invalid aggregation method raises error."""
        with pytest.raises(Exception):
            SuccessCriteria(aggregation_method="invalid")


class TestEnhancedPlanStep:
    """Test EnhancedPlanStep model."""

    def test_enhanced_plan_step_creation(self) -> None:
        """Test creating an enhanced plan step."""
        step = EnhancedPlanStep(
            step_id="step-1",
            action="search",
            parameters={"query": "test"},
        )
        assert step.step_id == "step-1"
        assert step.action == "search"
        assert step.status == StepStatus.PENDING
        assert step.retry_count == 0

    def test_enhanced_plan_step_with_dependencies(self) -> None:
        """Test step with dependencies."""
        dep = StepDependency(step_id="step-0")
        step = EnhancedPlanStep(
            step_id="step-1",
            action="process",
            parameters={},
            dependencies=[dep],
        )
        assert len(step.dependencies) == 1
        assert step.dependencies[0].step_id == "step-0"

    def test_enhanced_plan_step_with_tools(self) -> None:
        """Test step with tool requirements."""
        tool = ToolRequirement(tool_name="calculator")
        step = EnhancedPlanStep(
            step_id="step-1",
            action="calculate",
            parameters={},
            tool_requirements=[tool],
        )
        assert len(step.tool_requirements) == 1
        assert step.tool_requirements[0].tool_name == "calculator"

    def test_enhanced_plan_step_mark_started(self) -> None:
        """Test marking step as started."""
        step = EnhancedPlanStep(
            step_id="step-1",
            action="test",
            parameters={},
        )
        step.mark_started()
        assert step.status == StepStatus.IN_PROGRESS
        assert step.started_at is not None

    def test_enhanced_plan_step_mark_completed(self) -> None:
        """Test marking step as completed."""
        step = EnhancedPlanStep(
            step_id="step-1",
            action="test",
            parameters={},
        )
        step.mark_started()
        step.mark_completed(result={"data": "test"})
        assert step.status == StepStatus.COMPLETED
        assert step.completed_at is not None
        assert step.result["data"] == "test"
        assert step.duration_seconds is not None

    def test_enhanced_plan_step_mark_failed(self) -> None:
        """Test marking step as failed."""
        step = EnhancedPlanStep(
            step_id="step-1",
            action="test",
            parameters={},
        )
        step.mark_failed("Test error")
        assert step.status == StepStatus.FAILED
        assert step.error == "Test error"
        assert step.completed_at is not None


class TestEnhancedExecutionPlan:
    """Test EnhancedExecutionPlan model."""

    def test_enhanced_execution_plan_creation(self) -> None:
        """Test creating an enhanced execution plan."""
        step = EnhancedPlanStep(
            step_id="step-1",
            action="test",
            parameters={},
        )
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[step],
        )
        assert plan.plan_id == "plan-1"
        assert len(plan.steps) == 1
        assert plan.status == PlanStatus.PENDING
        assert plan.max_iterations == 10
        assert plan.current_iteration == 0

    def test_enhanced_execution_plan_with_success_criteria(self) -> None:
        """Test plan with success criteria."""
        criterion = SuccessCriterion(
            description="Test",
            metric_name="score",
            operator="gt",
            threshold=0.8,
        )
        criteria = SuccessCriteria(criteria=[criterion])
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[],
            success_criteria=criteria,
        )
        assert plan.success_criteria is not None
        assert len(plan.success_criteria.criteria) == 1

    def test_enhanced_execution_plan_mark_started(self) -> None:
        """Test marking plan as started."""
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[],
        )
        plan.mark_started()
        assert plan.status == PlanStatus.IN_PROGRESS
        assert plan.started_at is not None

    def test_enhanced_execution_plan_mark_completed(self) -> None:
        """Test marking plan as completed."""
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[],
        )
        plan.mark_started()
        plan.mark_completed(result={"final": "result"})
        assert plan.status == PlanStatus.COMPLETED
        assert plan.completed_at is not None
        assert plan.final_result["final"] == "result"

    def test_enhanced_execution_plan_mark_failed(self) -> None:
        """Test marking plan as failed."""
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[],
        )
        plan.mark_failed("Test error")
        assert plan.status == PlanStatus.FAILED
        assert plan.error == "Test error"

    def test_enhanced_execution_plan_get_next_step(self) -> None:
        """Test getting next pending step."""
        step1 = EnhancedPlanStep(
            step_id="step-1",
            action="first",
            parameters={},
        )
        step2 = EnhancedPlanStep(
            step_id="step-2",
            action="second",
            parameters={},
        )
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[step1, step2],
        )

        # Should get first step
        next_step = plan.get_next_step()
        assert next_step is not None
        assert next_step.step_id == "step-1"

        # Mark first complete, should get second
        step1.mark_completed()
        next_step = plan.get_next_step()
        assert next_step is not None
        assert next_step.step_id == "step-2"

    def test_enhanced_execution_plan_get_next_step_with_dependencies(self) -> None:
        """Test getting next step respects dependencies."""
        step1 = EnhancedPlanStep(
            step_id="step-1",
            action="first",
            parameters={},
        )
        step2 = EnhancedPlanStep(
            step_id="step-2",
            action="second",
            parameters={},
            dependencies=[StepDependency(step_id="step-1")],
        )
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[step1, step2],
        )

        # Should get step-1 first (no dependencies)
        next_step = plan.get_next_step()
        assert next_step.step_id == "step-1"

        # Step-2 should not be available yet
        step1.status = StepStatus.IN_PROGRESS
        next_step = plan.get_next_step()
        assert next_step is None

        # After step-1 completes, step-2 should be available
        step1.mark_completed()
        next_step = plan.get_next_step()
        assert next_step.step_id == "step-2"

    def test_enhanced_execution_plan_get_step_by_id(self) -> None:
        """Test getting step by ID."""
        step = EnhancedPlanStep(
            step_id="step-1",
            action="test",
            parameters={},
        )
        plan = EnhancedExecutionPlan(
            plan_id="plan-1",
            steps=[step],
        )

        found = plan.get_step_by_id("step-1")
        assert found is not None
        assert found.step_id == "step-1"

        not_found = plan.get_step_by_id("step-999")
        assert not_found is None


class TestModuleTransition:
    """Test ModuleTransition model."""

    def test_module_transition_creation(self) -> None:
        """Test creating a module transition."""
        transition = ModuleTransition(
            plan_id="plan-1",
            iteration=1,
            from_module=ModuleType.PLANNER,
            to_module=ModuleType.EXECUTOR,
            reason="Plan completed, starting execution",
        )
        assert transition.plan_id == "plan-1"
        assert transition.iteration == 1
        assert transition.from_module == ModuleType.PLANNER
        assert transition.to_module == ModuleType.EXECUTOR
        assert transition.transition_id is not None
        assert transition.timestamp is not None

    def test_module_transition_with_data(self) -> None:
        """Test transition with data transfer."""
        transition = ModuleTransition(
            plan_id="plan-1",
            iteration=1,
            from_module=ModuleType.EXECUTOR,
            to_module=ModuleType.VERIFIER,
            reason="Execution complete",
            data={"results": [1, 2, 3]},
        )
        assert transition.data["results"] == [1, 2, 3]


class TestEnhancedVerificationResult:
    """Test EnhancedVerificationResult model."""

    def test_enhanced_verification_result_creation(self) -> None:
        """Test creating an enhanced verification result."""
        result = EnhancedVerificationResult(
            valid=True,
            confidence=0.95,
        )
        assert result.valid is True
        assert result.confidence == 0.95
        assert result.verification_level == VerificationLevel.STANDARD
        assert len(result.errors) == 0

    def test_enhanced_verification_result_with_errors(self) -> None:
        """Test verification result with errors."""
        result = EnhancedVerificationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            confidence=0.3,
        )
        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.confidence == 0.3

    def test_enhanced_verification_result_invalid_confidence(self) -> None:
        """Test that invalid confidence raises error."""
        with pytest.raises(Exception):  # Pydantic validation error
            EnhancedVerificationResult(valid=True, confidence=1.5)

        with pytest.raises(Exception):
            EnhancedVerificationResult(valid=True, confidence=-0.1)

    def test_enhanced_verification_result_with_criteria(self) -> None:
        """Test verification result with checked criteria."""
        result = EnhancedVerificationResult(
            valid=True,
            checked_criteria=["criterion-1", "criterion-2", "criterion-3"],
            passed_criteria=["criterion-1", "criterion-2"],
            failed_criteria=["criterion-3"],
        )
        assert len(result.checked_criteria) == 3
        assert len(result.passed_criteria) == 2
        assert len(result.failed_criteria) == 1

    def test_enhanced_verification_result_calculate_success_rate(self) -> None:
        """Test calculating success rate."""
        result = EnhancedVerificationResult(
            valid=True,
            checked_criteria=["c1", "c2", "c3", "c4"],
            passed_criteria=["c1", "c2", "c3"],
            failed_criteria=["c4"],
        )
        success_rate = result.calculate_success_rate()
        assert success_rate == 0.75  # 3/4

    def test_enhanced_verification_result_success_rate_no_criteria(self) -> None:
        """Test success rate when no criteria checked."""
        result = EnhancedVerificationResult(valid=True)
        assert result.calculate_success_rate() == 1.0

        result = EnhancedVerificationResult(valid=False)
        assert result.calculate_success_rate() == 0.0

    def test_enhanced_verification_result_with_recommendations(self) -> None:
        """Test verification result with recommendations."""
        result = EnhancedVerificationResult(
            valid=False,
            recommendations=[
                "Add error handling",
                "Improve validation",
            ],
            suggested_refinements={"retry_count": 3},
        )
        assert len(result.recommendations) == 2
        assert result.suggested_refinements["retry_count"] == 3
