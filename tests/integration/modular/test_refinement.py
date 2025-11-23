"""
Integration Tests for Plan Refinement Loop

Comprehensive test suite for iterative plan refinement scenarios including:
- Basic refinement cycles
- Feedback integration from verifier
- Iteration management and limits
- Complex multi-iteration scenarios
- Confidence improvement tracking
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.modular.coordinator import CoordinationContext, ModuleCoordinator
from agentcore.modular.executor import ExecutorModule
from agentcore.modular.generator import Generator
from agentcore.modular.interfaces import (
    ExecutionResult,
    GeneratedResponse,
    VerificationResult,
)
from agentcore.modular.models import (
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    PlanStatus,
    StepStatus,
)
from agentcore.modular.planner import Planner
from agentcore.modular.verifier import Verifier


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def a2a_context() -> A2AContext:
    """Create A2A context for testing."""
    return A2AContext(
        source_agent="test-refinement",
        target_agent="modular-agent",
        trace_id=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def coordination_context() -> CoordinationContext:
    """Create coordination context for testing."""
    return CoordinationContext(
        execution_id=str(uuid4()),
        trace_id=str(uuid4()),
        session_id=str(uuid4()),
        iteration=0,
    )


@pytest.fixture
def coordinator(coordination_context: CoordinationContext) -> ModuleCoordinator:
    """Create coordinator with context."""
    coordinator = ModuleCoordinator()
    coordinator.set_context(coordination_context)
    return coordinator


# ============================================================================
# Test Suite 1: Basic Refinement
# ============================================================================


@pytest.mark.asyncio
class TestBasicRefinement:
    """Test basic refinement cycles."""

    async def test_single_refinement_cycle(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test a single refinement iteration (fail then succeed)."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock initial plan
        initial_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="initial_action",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test query",
            status=PlanStatus.PENDING,
        )

        # Mock refined plan
        refined_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="refined_action",
                    parameters={"improved": True},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test query",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=initial_plan)
        planner.refine_plan = AsyncMock(return_value=refined_plan)

        # Mock execution results
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "success"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification results (fail first, pass second)
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Validation error"],
            feedback="Plan needs refinement",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Refined answer",
            reasoning=None,
            sources=["step:step-1"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify refinement occurred
        assert result["answer"] == "Refined answer"
        trace = result["execution_trace"]
        assert trace["iterations"] == 2  # Two iterations
        assert trace["verification_passed"] is True
        assert trace["confidence_score"] == 0.9

        # Verify planner was called for initial + refinement
        assert planner.analyze_query.call_count == 1
        assert planner.refine_plan.call_count == 1

    async def test_multiple_refinement_cycles(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test multiple refinement iterations before success."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plans (initial + 2 refinements)
        plan_v1 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="v1",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        plan_v2 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="v2",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        plan_v3 = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="v3",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )

        planner.analyze_query = AsyncMock(return_value=plan_v1)
        planner.refine_plan = AsyncMock(side_effect=[plan_v2, plan_v3])

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (fail, fail, pass)
        verification_fail_1 = VerificationResult(
            valid=False,
            confidence=0.3,
            errors=["Error 1"],
            feedback="Needs improvement",
        )
        verification_fail_2 = VerificationResult(
            valid=False,
            confidence=0.6,
            errors=["Error 2"],
            feedback="Still needs work",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.85,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail_1, verification_fail_2, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Final answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify 3 iterations
        trace = result["execution_trace"]
        assert trace["iterations"] == 3
        assert trace["verification_passed"] is True
        assert trace["confidence_score"] == 0.85

        # Verify refinement calls
        assert planner.refine_plan.call_count == 2

    async def test_max_iteration_limit_enforcement(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test that max iteration limit is enforced."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan (same plan returned each time)
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (always fail)
        verification_result = VerificationResult(
            valid=False,
            confidence=0.3,
            errors=["Persistent error"],
            feedback="Cannot fix",
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Partial answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute with max_iterations=3
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=3,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify stopped at max iterations
        trace = result["execution_trace"]
        assert trace["iterations"] == 3
        assert trace["verification_passed"] is False

        # Verify refinement attempts
        assert planner.refine_plan.call_count == 2  # iterations 2 and 3

    async def test_early_exit_on_high_confidence(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test early exit when confidence threshold is met."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (high confidence first time)
        verification_result = VerificationResult(
            valid=True,
            confidence=0.95,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute with max_iterations=10
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=10,  # High max
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify early exit after iteration 1
        trace = result["execution_trace"]
        assert trace["iterations"] == 1  # Stopped early
        assert trace["verification_passed"] is True

        # Verify no refinement needed
        assert planner.refine_plan.call_count == 0


# ============================================================================
# Test Suite 2: Feedback Integration
# ============================================================================


@pytest.mark.asyncio
class TestFeedbackIntegration:
    """Test feedback integration from verifier to planner."""

    async def test_verifier_feedback_to_planner(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test that verifier feedback is correctly passed to planner."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track feedback received by planner
        received_feedback = None

        async def mock_refine_plan(refinement_request):
            nonlocal received_feedback
            received_feedback = refinement_request.feedback
            return EnhancedExecutionPlan(
                plan_id=str(uuid4()),
                steps=[
                    EnhancedPlanStep(
                        step_id="step-1",
                        action="refined",
                        parameters={},
                        status=StepStatus.PENDING,
                    )
                ],
                query="Test",
                status=PlanStatus.PENDING,
            )

        # Mock initial plan
        initial_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="initial",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=initial_plan)
        planner.refine_plan = mock_refine_plan

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification with specific feedback
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Error A", "Error B"],
            feedback="Specific feedback for refinement",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify feedback was passed to planner
        assert received_feedback == "Specific feedback for refinement"

    async def test_refinement_improves_plan(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test that refined plan addresses verifier feedback."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track plan versions
        plans_created = []

        async def mock_analyze_query(query):
            plan = EnhancedExecutionPlan(
                plan_id=str(uuid4()),
                steps=[
                    EnhancedPlanStep(
                        step_id="step-1",
                        action="initial",
                        parameters={"version": 1},
                        status=StepStatus.PENDING,
                    )
                ],
                query="Test",
                status=PlanStatus.PENDING,
            )
            plans_created.append(plan)
            return plan

        async def mock_refine_plan(refinement_request):
            plan = EnhancedExecutionPlan(
                plan_id=str(uuid4()),
                steps=[
                    EnhancedPlanStep(
                        step_id="step-1",
                        action="refined",
                        parameters={"version": 2, "improved": True},
                        status=StepStatus.PENDING,
                    )
                ],
                query="Test",
                status=PlanStatus.PENDING,
            )
            plans_created.append(plan)
            return plan

        planner.analyze_query = mock_analyze_query
        planner.refine_plan = mock_refine_plan

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Missing improvements"],
            feedback="Add improvements",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify plan was improved
        assert len(plans_created) == 2
        assert plans_created[0].steps[0].parameters["version"] == 1
        assert plans_created[1].steps[0].parameters["version"] == 2
        assert plans_created[1].steps[0].parameters["improved"] is True

    async def test_feedback_includes_error_details(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test that feedback includes detailed error information."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track refinement request
        refinement_constraints = None

        async def mock_refine_plan(refinement_request):
            nonlocal refinement_constraints
            refinement_constraints = refinement_request.constraints
            return EnhancedExecutionPlan(
                plan_id=str(uuid4()),
                steps=[
                    EnhancedPlanStep(
                        step_id="step-1",
                        action="refined",
                        parameters={},
                        status=StepStatus.PENDING,
                    )
                ],
                query="Test",
                status=PlanStatus.PENDING,
            )

        # Mock initial plan
        initial_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="initial",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=initial_plan)
        planner.refine_plan = mock_refine_plan

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification with errors
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Error A", "Error B", "Error C"],
            feedback="Multiple errors detected",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify errors were included in constraints
        assert refinement_constraints is not None
        assert "verification_errors" in refinement_constraints
        assert refinement_constraints["verification_errors"] == [
            "Error A",
            "Error B",
            "Error C",
        ]

    async def test_empty_feedback_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test graceful handling when verifier provides no feedback."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track feedback
        received_feedback = None

        async def mock_refine_plan(refinement_request):
            nonlocal received_feedback
            received_feedback = refinement_request.feedback
            return EnhancedExecutionPlan(
                plan_id=str(uuid4()),
                steps=[
                    EnhancedPlanStep(
                        step_id="step-1",
                        action="refined",
                        parameters={},
                        status=StepStatus.PENDING,
                    )
                ],
                query="Test",
                status=PlanStatus.PENDING,
            )

        # Mock initial plan
        initial_plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="initial",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=initial_plan)
        planner.refine_plan = mock_refine_plan

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification with no feedback (None)
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Some error"],
            feedback=None,  # No feedback
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify feedback fallback was used
        assert received_feedback is not None
        assert "Errors:" in received_feedback


# ============================================================================
# Test Suite 3: Iteration Management
# ============================================================================


@pytest.mark.asyncio
class TestIterationManagement:
    """Test iteration counter and history tracking."""

    async def test_iteration_counter_increments(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test iteration counter increments correctly."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (fail 2 times, pass on 3rd)
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Error"],
            feedback="Needs work",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify iteration counter
        trace = result["execution_trace"]
        assert trace["iterations"] == 3

    async def test_refinement_history_tracked(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test refinement history is recorded in trace."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (2 iterations)
        verification_fail = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Error"],
            feedback="Needs work",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify refinement history
        trace = result["execution_trace"]
        assert "refinement_history" in trace
        assert len(trace["refinement_history"]) == 2  # 2 iterations

    async def test_module_transitions_logged(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test module transitions are tracked correctly."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification
        verification_result = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify transitions
        trace = result["execution_trace"]
        assert "transitions" in trace
        transitions = trace["transitions"]
        assert len(transitions) > 0

        # Verify transition structure
        for transition in transitions:
            assert "from_module" in transition
            assert "to_module" in transition
            assert "reason" in transition


# ============================================================================
# Test Suite 4: Complex Scenarios
# ============================================================================


@pytest.mark.asyncio
class TestComplexScenarios:
    """Test complex multi-iteration scenarios."""

    async def test_alternating_success_failure(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test handling of inconsistent verification results."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (fail, fail, pass)
        verification_fail_1 = VerificationResult(
            valid=False,
            confidence=0.4,
            errors=["Error 1"],
            feedback="Fix this",
        )
        verification_fail_2 = VerificationResult(
            valid=False,
            confidence=0.5,
            errors=["Error 2"],
            feedback="Fix that",
        )
        verification_pass = VerificationResult(
            valid=True,
            confidence=0.8,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_fail_1, verification_fail_2, verification_pass]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify eventually succeeded
        trace = result["execution_trace"]
        assert trace["iterations"] == 3
        assert trace["verification_passed"] is True

    async def test_incremental_improvement(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test confidence score increases over iterations."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (increasing confidence)
        verification_1 = VerificationResult(
            valid=False,
            confidence=0.3,
            errors=["Error"],
            feedback="Needs work",
        )
        verification_2 = VerificationResult(
            valid=False,
            confidence=0.6,
            errors=["Error"],
            feedback="Better",
        )
        verification_3 = VerificationResult(
            valid=True,
            confidence=0.85,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_1, verification_2, verification_3]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify final confidence improved
        trace = result["execution_trace"]
        assert trace["confidence_score"] == 0.85
        assert trace["verification_passed"] is True

    async def test_degradation_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test handling when confidence decreases."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)
        planner.refine_plan = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification (decreasing confidence, then recovery)
        verification_1 = VerificationResult(
            valid=False,
            confidence=0.6,
            errors=["Error"],
            feedback="Needs work",
        )
        verification_2 = VerificationResult(
            valid=False,
            confidence=0.4,  # Decreased!
            errors=["Worse"],
            feedback="Got worse",
        )
        verification_3 = VerificationResult(
            valid=True,
            confidence=0.9,  # Recovered
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(
            side_effect=[verification_1, verification_2, verification_3]
        )

        # Mock generation
        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify eventually recovered
        trace = result["execution_trace"]
        assert trace["iterations"] == 3
        assert trace["confidence_score"] == 0.9
        assert trace["verification_passed"] is True
