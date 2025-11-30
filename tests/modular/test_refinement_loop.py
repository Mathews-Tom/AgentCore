"""
Tests for Plan Refinement Loop

Validates refinement loop functionality including max iterations, convergence detection,
early stopping, refinement history tracking, and metrics collection.
"""

from __future__ import annotations

import pytest
from typing import Any
from uuid import uuid4

from agentcore.modular.coordinator import (
    ModuleCoordinator,
    CoordinationContext,
    RefinementIteration,
)
from agentcore.modular.interfaces import (
    PlannerQuery,
    PlanRefinement,
    VerificationRequest,
    GenerationRequest,
    ExecutionResult,
    VerificationResult,
    GeneratedResponse,
    ExecutionPlan,
    PlanStep,
)


# ============================================================================
# Mock Modules for Testing
# ============================================================================


class MockPlanner:
    """Mock Planner module for testing."""

    def __init__(self, plans: list[ExecutionPlan] | None = None) -> None:
        self.plans = plans or []
        self.call_count = 0
        self.refinement_calls = 0

    async def analyze_query(self, query: PlannerQuery) -> ExecutionPlan:
        """Mock analyze_query."""
        self.call_count += 1
        if self.plans:
            return self.plans[min(self.call_count - 1, len(self.plans) - 1)]

        return ExecutionPlan(
            plan_id=f"plan-{self.call_count}",
            steps=[
                PlanStep(
                    step_id=f"step-{self.call_count}-1",
                    action="test_action",
                    parameters={},
                )
            ],
        )

    async def refine_plan(self, refinement: PlanRefinement) -> ExecutionPlan:
        """Mock refine_plan."""
        self.refinement_calls += 1
        if self.plans and self.refinement_calls < len(self.plans):
            return self.plans[self.refinement_calls]

        return ExecutionPlan(
            plan_id=f"plan-refined-{self.refinement_calls}",
            steps=[
                PlanStep(
                    step_id=f"step-refined-{self.refinement_calls}-1",
                    action="refined_action",
                    parameters={},
                )
            ],
        )


class MockExecutor:
    """Mock Executor module for testing."""

    def __init__(self, results: list[list[ExecutionResult]] | None = None) -> None:
        self.results = results or []
        self.call_count = 0

    async def execute_plan(self, plan: ExecutionPlan) -> list[ExecutionResult]:
        """Mock execute_plan."""
        self.call_count += 1
        if self.results and self.call_count <= len(self.results):
            return self.results[self.call_count - 1]

        return [
            ExecutionResult(
                step_id=step.step_id,
                success=True,
                result={"result": f"output-{self.call_count}"},
                error=None,
                execution_time=1.0,
            )
            for step in plan.steps
        ]


class MockVerifier:
    """Mock Verifier module for testing."""

    def __init__(self, verifications: list[VerificationResult] | None = None) -> None:
        self.verifications = verifications or []
        self.call_count = 0

    async def validate_results(
        self, request: VerificationRequest
    ) -> VerificationResult:
        """Mock validate_results."""
        self.call_count += 1
        if self.verifications and self.call_count <= len(self.verifications):
            return self.verifications[self.call_count - 1]

        return VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            feedback=None,
        )


class MockGenerator:
    """Mock Generator module for testing."""

    def __init__(self, response: str = "Generated response") -> None:
        self.response = response
        self.call_count = 0

    async def synthesize_response(
        self, request: GenerationRequest
    ) -> GeneratedResponse:
        """Mock synthesize_response."""
        self.call_count += 1
        return GeneratedResponse(
            content=self.response,
            format="text",
            reasoning="Test reasoning",
            sources=[],
        )


# ============================================================================
# Test Classes
# ============================================================================


class TestRefinementIteration:
    """Test RefinementIteration model."""

    def test_refinement_iteration_creation(self) -> None:
        """Test creating a refinement iteration record."""
        iteration = RefinementIteration(
            iteration=1,
            plan_id="plan-123",
            confidence=0.75,
            valid=True,
            errors=[],
            feedback="Good results",
        )

        assert iteration.iteration == 1
        assert iteration.plan_id == "plan-123"
        assert iteration.confidence == 0.75
        assert iteration.valid is True
        assert iteration.errors == []
        assert iteration.feedback == "Good results"
        assert iteration.timestamp is not None

    def test_refinement_iteration_with_errors(self) -> None:
        """Test refinement iteration with validation errors."""
        iteration = RefinementIteration(
            iteration=2,
            plan_id="plan-456",
            confidence=0.4,
            valid=False,
            errors=["Error 1", "Error 2"],
            feedback="Needs improvement",
        )

        assert iteration.valid is False
        assert len(iteration.errors) == 2
        assert iteration.confidence == 0.4


class TestCoordinationContextRefinementHistory:
    """Test CoordinationContext with refinement history."""

    def test_context_with_refinement_history(self) -> None:
        """Test coordination context tracks refinement history."""
        context = CoordinationContext(
            execution_id="exec-123",
            plan_id="plan-456",
        )

        assert context.refinement_history == []

        # Add refinement iterations
        context.refinement_history.append(
            RefinementIteration(
                iteration=1,
                plan_id="plan-456",
                confidence=0.6,
                valid=False,
                errors=["Initial error"],
            )
        )

        context.refinement_history.append(
            RefinementIteration(
                iteration=2,
                plan_id="plan-789",
                confidence=0.8,
                valid=True,
                errors=[],
            )
        )

        assert len(context.refinement_history) == 2
        assert context.refinement_history[0].confidence == 0.6
        assert context.refinement_history[1].confidence == 0.8


class TestMaxIterationsLimit:
    """Test max iterations limit enforcement."""

    @pytest.mark.asyncio
    async def test_max_iterations_default(self) -> None:
        """Test default max iterations is 5."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        # Create verifier with gradually improving confidence to avoid convergence detection
        verifier = MockVerifier(
            verifications=[
                VerificationResult(
                    valid=False,
                    confidence=0.3 + (i * 0.05),  # Gradually improving
                    errors=["Test error"],
                    feedback="Try again",
                )
                for i in range(10)  # More than max iterations
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=60.0,
        )

        # Should stop at max_iterations (5)
        assert result["execution_trace"]["iterations"] == 5
        assert verifier.call_count == 5
        assert planner.refinement_calls == 4  # One initial + 4 refinements

    @pytest.mark.asyncio
    async def test_max_iterations_custom(self) -> None:
        """Test custom max iterations limit."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        # Use gradually improving confidence to avoid convergence
        verifier = MockVerifier(
            verifications=[
                VerificationResult(
                    valid=False,
                    confidence=0.3 + (i * 0.06),  # Gradually improving
                    errors=["Error"]
                )
                for i in range(10)
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=3,
            timeout_seconds=60.0,
        )

        assert result["execution_trace"]["iterations"] == 3
        assert verifier.call_count == 3


class TestConvergenceDetection:
    """Test convergence detection (no improvement)."""

    @pytest.mark.asyncio
    async def test_convergence_detected(self) -> None:
        """Test early stopping when no improvement detected."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        # Create verifier with stagnant confidence scores
        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.50, errors=["Error"]),
                VerificationResult(valid=False, confidence=0.50, errors=["Error"]),  # Same confidence
                VerificationResult(valid=False, confidence=0.50, errors=["Error"]),
                VerificationResult(valid=False, confidence=0.50, errors=["Error"]),
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            convergence_window=2,
            convergence_threshold=0.01,  # Very small threshold
            timeout_seconds=60.0,
        )

        # Should stop early due to convergence (at iteration 2 with window=2)
        iterations = result["execution_trace"]["iterations"]
        assert iterations == 2  # Should stop at iteration 2
        assert iterations >= 2  # Need at least convergence_window iterations

        # Check refinement metrics
        metrics = result["execution_trace"]["refinement_metrics"]
        assert metrics["converged"] is True

    @pytest.mark.asyncio
    async def test_no_convergence_with_improvement(self) -> None:
        """Test refinement continues when improvement detected."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        # Create verifier with improving confidence scores
        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.50, errors=["Error"]),
                VerificationResult(valid=False, confidence=0.60, errors=["Error"]),
                VerificationResult(valid=False, confidence=0.70, errors=["Error"]),
                VerificationResult(valid=True, confidence=0.80, errors=[]),  # Success
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            convergence_window=2,
            convergence_threshold=0.05,
            timeout_seconds=60.0,
        )

        # Should reach success without early convergence stopping
        assert result["execution_trace"]["verification_passed"] is True
        assert result["execution_trace"]["iterations"] == 4


class TestEarlyStopping:
    """Test early stopping when confidence threshold met."""

    @pytest.mark.asyncio
    async def test_early_stop_on_success(self) -> None:
        """Test early stopping when verification succeeds."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        # Create verifier that succeeds on second iteration
        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.60, errors=["Error"]),
                VerificationResult(valid=True, confidence=0.85, errors=[]),  # Success
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            confidence_threshold=0.7,
            timeout_seconds=60.0,
        )

        # Should stop at iteration 2
        assert result["execution_trace"]["iterations"] == 2
        assert result["execution_trace"]["verification_passed"] is True
        assert verifier.call_count == 2

    @pytest.mark.asyncio
    async def test_early_stop_with_high_confidence(self) -> None:
        """Test early stopping when confidence threshold exceeded."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=True, confidence=0.95, errors=[]),
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            confidence_threshold=0.8,
            timeout_seconds=60.0,
        )

        # Should stop at iteration 1 (first success)
        assert result["execution_trace"]["iterations"] == 1
        assert verifier.call_count == 1


class TestRefinementHistoryTracking:
    """Test refinement history tracking."""

    @pytest.mark.asyncio
    async def test_refinement_history_populated(self) -> None:
        """Test refinement history is tracked correctly."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.5, errors=["Error 1"]),
                VerificationResult(valid=False, confidence=0.6, errors=["Error 2"]),
                VerificationResult(valid=True, confidence=0.8, errors=[]),
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=60.0,
        )

        # Check refinement history in execution trace
        history = result["execution_trace"]["refinement_history"]
        assert len(history) == 3

        # Validate history records
        assert history[0]["iteration"] == 1
        assert history[0]["confidence"] == 0.5
        assert history[0]["valid"] is False
        assert "Error 1" in history[0]["errors"]

        assert history[1]["iteration"] == 2
        assert history[1]["confidence"] == 0.6

        assert history[2]["iteration"] == 3
        assert history[2]["confidence"] == 0.8
        assert history[2]["valid"] is True

    @pytest.mark.asyncio
    async def test_refinement_history_in_context(self) -> None:
        """Test refinement history stored in coordination context."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.5, errors=[]),
                VerificationResult(valid=True, confidence=0.8, errors=[]),
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=60.0,
        )

        # Check context refinement history
        assert len(context.refinement_history) == 2
        assert isinstance(context.refinement_history[0], RefinementIteration)
        assert context.refinement_history[0].confidence == 0.5
        assert context.refinement_history[1].confidence == 0.8


class TestRefinementMetrics:
    """Test refinement effectiveness metrics."""

    @pytest.mark.asyncio
    async def test_refinement_metrics_calculation(self) -> None:
        """Test refinement metrics are calculated correctly."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.4, errors=["Error"]),
                VerificationResult(valid=False, confidence=0.6, errors=["Error"]),
                VerificationResult(valid=True, confidence=0.85, errors=[]),
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            confidence_threshold=0.7,
            timeout_seconds=60.0,
        )

        metrics = result["execution_trace"]["refinement_metrics"]

        assert metrics["total_refinements"] == 2  # 3 iterations - 1
        assert metrics["initial_confidence"] == 0.4
        assert metrics["final_confidence"] == 0.85
        assert metrics["confidence_improvement"] == pytest.approx(0.45)
        assert metrics["max_confidence_reached"] == 0.85
        assert metrics["iterations_to_success"] == 3
        assert metrics["confidence_progression"] == [0.4, 0.6, 0.85]

    @pytest.mark.asyncio
    async def test_refinement_metrics_no_improvement(self) -> None:
        """Test refinement metrics when no improvement occurs."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=False, confidence=0.5, errors=["Error"])
                for _ in range(5)
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=60.0,
        )

        metrics = result["execution_trace"]["refinement_metrics"]

        assert metrics["confidence_improvement"] == 0.0
        assert metrics["iterations_to_success"] is None  # Never succeeded
        assert all(c == 0.5 for c in metrics["confidence_progression"])


class TestIterationLimitValidation:
    """Test iteration limit validation and edge cases."""

    @pytest.mark.asyncio
    async def test_single_iteration_success(self) -> None:
        """Test immediate success on first iteration."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        verifier = MockVerifier(
            verifications=[
                VerificationResult(valid=True, confidence=0.95, errors=[]),
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=60.0,
        )

        assert result["execution_trace"]["iterations"] == 1
        assert result["execution_trace"]["refinement_metrics"]["total_refinements"] == 0

    @pytest.mark.asyncio
    async def test_max_iterations_reached_without_success(self) -> None:
        """Test max iterations reached without achieving success."""
        coordinator = ModuleCoordinator()
        context = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
        )
        coordinator.set_context(context)

        # Gradually improving but never reaching threshold - avoid convergence
        verifier = MockVerifier(
            verifications=[
                VerificationResult(
                    valid=False,
                    confidence=0.3 + (i * 0.07),  # Improving enough to avoid convergence
                    errors=["Persistent error"]
                )
                for i in range(10)
            ]
        )

        planner = MockPlanner()
        executor = MockExecutor()
        generator = MockGenerator()

        result = await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=3,
            timeout_seconds=60.0,
        )

        assert result["execution_trace"]["iterations"] == 3
        assert result["execution_trace"]["verification_passed"] is False
        assert result["execution_trace"]["refinement_metrics"]["iterations_to_success"] is None
