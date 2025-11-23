"""
Integration tests for Module Coordination Loop

Tests the full PEVG workflow with iterative refinement, including:
- Successful single-iteration execution
- Multi-iteration refinement scenarios
- Max iteration limit enforcement
- Early exit on verification success
- Timeout handling
- State persistence
- Module transition events
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
    PlanStep,
    VerificationResult,
)
from agentcore.modular.models import EnhancedExecutionPlan, EnhancedPlanStep, StepStatus
from agentcore.modular.planner import Planner
from agentcore.modular.verifier import Verifier


@pytest.fixture
def a2a_context() -> A2AContext:
    """Create A2A context for testing."""
    return A2AContext(
        source_agent="test-agent",
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
# Test: Successful Single-Iteration Execution
# ============================================================================


@pytest.mark.asyncio
async def test_single_iteration_success(
    coordinator: ModuleCoordinator,
    a2a_context: A2AContext,
) -> None:
    """Test successful execution in single iteration (no refinement needed)."""
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
                action="test_action",
                parameters={"param": "value"},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )
    planner.analyze_query = AsyncMock(return_value=plan)

    # Mock execution results
    execution_results = [
        ExecutionResult(
            step_id="step-1",
            success=True,
            result={"result": "success"},
            execution_time=1.5,
        )
    ]
    executor.execute_plan = AsyncMock(return_value=execution_results)

    # Mock verification result (passed with high confidence)
    verification_result = VerificationResult(
        valid=True,
        confidence=0.95,
        errors=[],
        warnings=[],
    )
    verifier.validate_results = AsyncMock(return_value=verification_result)

    # Mock generation response
    generation_response = GeneratedResponse(
        format="text",
        content="Test answer",
        reasoning="Test reasoning",
        sources=["source1"],
    )
    generator.synthesize_response = AsyncMock(return_value=generation_response)

    # Execute coordination loop
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

    # Verify result structure
    assert "answer" in result
    assert "execution_trace" in result
    assert result["answer"] == "Test answer"

    # Verify execution trace
    trace = result["execution_trace"]
    assert trace["iterations"] == 1
    assert trace["verification_passed"] is True
    assert trace["confidence_score"] == 0.95
    assert trace["step_count"] == 1
    assert trace["successful_steps"] == 1
    assert trace["failed_steps"] == 0

    # Verify module invocations
    assert planner.analyze_query.call_count == 1
    assert executor.execute_plan.call_count == 1
    assert verifier.validate_results.call_count == 1
    assert generator.synthesize_response.call_count == 1

    # Verify planner.refine_plan was NOT called (no refinement needed)
    assert not hasattr(planner, "refine_plan") or planner.refine_plan.call_count == 0


# ============================================================================
# Test: Multi-Iteration Refinement
# ============================================================================


@pytest.mark.asyncio
async def test_multi_iteration_refinement(
    coordinator: ModuleCoordinator,
    a2a_context: A2AContext,
) -> None:
    """Test execution with multiple refinement iterations."""
    # Create mock modules
    planner = MagicMock(spec=Planner)
    executor = MagicMock(spec=ExecutorModule)
    verifier = MagicMock(spec=Verifier)
    generator = MagicMock(spec=Generator)

    # Mock initial plan
    plan_v1 = EnhancedExecutionPlan(
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
    )

    # Mock refined plan
    plan_v2 = EnhancedExecutionPlan(
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
    )

    planner.analyze_query = AsyncMock(return_value=plan_v1)
    planner.refine_plan = AsyncMock(return_value=plan_v2)

    # Mock execution results
    execution_results = [
        ExecutionResult(
            step_id="step-1",
            success=True,
            result={"result": "success"},
            execution_time=1.5,
        )
    ]
    executor.execute_plan = AsyncMock(return_value=execution_results)

    # Mock verification results (fail first, pass second)
    verification_fail = VerificationResult(
        valid=False,
        confidence=0.5,
        errors=["Validation error"],
        feedback="Need to improve plan",
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

    # Mock generation response
    generation_response = GeneratedResponse(
        format="text",
        content="Refined answer",
        reasoning="Refined reasoning",
        sources=["source1"],
    )
    generator.synthesize_response = AsyncMock(return_value=generation_response)

    # Execute coordination loop
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

    # Verify result
    assert result["answer"] == "Refined answer"

    # Verify execution trace
    trace = result["execution_trace"]
    assert trace["iterations"] == 2  # Two iterations
    assert trace["verification_passed"] is True
    assert trace["confidence_score"] == 0.9

    # Verify module invocations
    assert planner.analyze_query.call_count == 1  # Initial planning
    assert planner.refine_plan.call_count == 1  # One refinement
    assert executor.execute_plan.call_count == 2  # Two executions
    assert verifier.validate_results.call_count == 2  # Two verifications
    assert generator.synthesize_response.call_count == 1  # Final generation

    # Verify refinement history
    assert len(trace["refinement_history"]) == 2


# ============================================================================
# Test: Max Iteration Limit Enforcement
# ============================================================================


@pytest.mark.asyncio
async def test_max_iteration_limit(
    coordinator: ModuleCoordinator,
    a2a_context: A2AContext,
) -> None:
    """Test that max iteration limit is enforced."""
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
                action="test_action",
                parameters={},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )

    planner.analyze_query = AsyncMock(return_value=plan)
    planner.refine_plan = AsyncMock(return_value=plan)  # Always return same plan

    # Mock execution results
    execution_results = [
        ExecutionResult(
            step_id="step-1",
            success=True,
            result={"result": "success"},
            execution_time=1.5,
        )
    ]
    executor.execute_plan = AsyncMock(return_value=execution_results)

    # Mock verification result (always fail with low confidence)
    verification_result = VerificationResult(
        valid=False,
        confidence=0.3,
        errors=["Persistent error"],
        feedback="Cannot fix",
    )
    verifier.validate_results = AsyncMock(return_value=verification_result)

    # Mock generation response
    generation_response = GeneratedResponse(
        format="text",
        content="Partial answer after max iterations",
        reasoning="Could not fully solve",
        sources=[],
    )
    generator.synthesize_response = AsyncMock(return_value=generation_response)

    # Execute coordination loop with max_iterations=3
    result = await coordinator.execute_with_refinement(
        query="Test query",
        planner=planner,
        executor=executor,
        verifier=verifier,
        generator=generator,
        max_iterations=3,
        timeout_seconds=30.0,
        confidence_threshold=0.7,
    )

    # Verify max iterations reached
    trace = result["execution_trace"]
    assert trace["iterations"] == 3  # Stopped at max
    assert trace["verification_passed"] is False
    assert trace["confidence_score"] == 0.3

    # Verify module invocations
    assert planner.analyze_query.call_count == 1
    assert planner.refine_plan.call_count == 2  # 2 refinements (iterations 2, 3)
    assert executor.execute_plan.call_count == 3  # 3 executions
    assert verifier.validate_results.call_count == 3  # 3 verifications


# ============================================================================
# Test: Early Exit on Verification Success
# ============================================================================


@pytest.mark.asyncio
async def test_early_exit_on_success(
    coordinator: ModuleCoordinator,
    a2a_context: A2AContext,
) -> None:
    """Test early exit when verification passes before max iterations."""
    # Create mock modules
    planner = MagicMock(spec=Planner)
    executor = MagicMock(spec=ExecutorModule)
    verifier = MagicMock(spec=Verifier)
    generator = MagicMock(spec=Generator)

    # Mock plans
    plan_v1 = EnhancedExecutionPlan(
        plan_id=str(uuid4()),
        steps=[
            EnhancedPlanStep(
                step_id="step-1",
                action="action_v1",
                parameters={},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )

    plan_v2 = EnhancedExecutionPlan(
        plan_id=str(uuid4()),
        steps=[
            EnhancedPlanStep(
                step_id="step-1",
                action="action_v2",
                parameters={},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )

    planner.analyze_query = AsyncMock(return_value=plan_v1)
    planner.refine_plan = AsyncMock(return_value=plan_v2)

    # Mock execution results
    execution_results = [
        ExecutionResult(
            step_id="step-1",
            success=True,
            result={"result": "success"},
            execution_time=1.5,
        )
    ]
    executor.execute_plan = AsyncMock(return_value=execution_results)

    # Mock verification results (fail iteration 1, pass iteration 2)
    verification_fail = VerificationResult(
        valid=False,
        confidence=0.5,
        errors=["Error"],
        feedback="Refinement needed",
    )
    verification_pass = VerificationResult(
        valid=True,
        confidence=0.85,
        errors=[],
        warnings=[],
    )

    verifier.validate_results = AsyncMock(
        side_effect=[verification_fail, verification_pass]
    )

    # Mock generation response
    generation_response = GeneratedResponse(
        format="text",
        content="Success after refinement",
        reasoning="Improved plan worked",
        sources=["source1"],
    )
    generator.synthesize_response = AsyncMock(return_value=generation_response)

    # Execute coordination loop with max_iterations=10
    result = await coordinator.execute_with_refinement(
        query="Test query",
        planner=planner,
        executor=executor,
        verifier=verifier,
        generator=generator,
        max_iterations=10,  # Large max, but should exit early
        timeout_seconds=30.0,
        confidence_threshold=0.7,
    )

    # Verify early exit after iteration 2
    trace = result["execution_trace"]
    assert trace["iterations"] == 2  # Stopped early
    assert trace["verification_passed"] is True
    assert trace["confidence_score"] == 0.85

    # Verify module invocations (should only run twice, not 10 times)
    assert executor.execute_plan.call_count == 2
    assert verifier.validate_results.call_count == 2


# ============================================================================
# Test: Timeout Handling
# ============================================================================


@pytest.mark.asyncio
async def test_timeout_handling(
    coordinator: ModuleCoordinator,
    a2a_context: A2AContext,
) -> None:
    """Test timeout handling for long-running executions."""
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
                action="slow_action",
                parameters={},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )

    planner.analyze_query = AsyncMock(return_value=plan)

    # Mock slow execution (sleeps for 5 seconds)
    async def slow_execution(plan):
        await asyncio.sleep(5.0)
        return [
            ExecutionResult(
                step_id="step-1",
                success=True,
                output={"result": "slow"},
            )
        ]

    executor.execute_plan = slow_execution

    # Execute coordination loop with short timeout
    with pytest.raises(asyncio.TimeoutError):
        await coordinator.execute_with_refinement(
            query="Test query",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=1.0,  # Short timeout
            confidence_threshold=0.7,
        )


# ============================================================================
# Test: State Persistence
# ============================================================================


@pytest.mark.asyncio
async def test_state_persistence(
    coordinator: ModuleCoordinator,
    coordination_context: CoordinationContext,
    a2a_context: A2AContext,
) -> None:
    """Test that state is persisted at each module transition."""
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
                action="test_action",
                parameters={},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )

    planner.analyze_query = AsyncMock(return_value=plan)

    # Mock execution results
    execution_results = [
        ExecutionResult(
            step_id="step-1",
            success=True,
            result={"result": "success"},
            execution_time=1.5,
        )
    ]
    executor.execute_plan = AsyncMock(return_value=execution_results)

    # Mock verification result
    verification_result = VerificationResult(
        valid=True,
        confidence=0.9,
        errors=[],
        warnings=[],
    )
    verifier.validate_results = AsyncMock(return_value=verification_result)

    # Mock generation response
    generation_response = GeneratedResponse(
        format="text",
        content="Test answer",
        reasoning="Test reasoning",
        sources=["source1"],
    )
    generator.synthesize_response = AsyncMock(return_value=generation_response)

    # Execute coordination loop
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

    # Verify state persistence in coordination context
    context = coordinator.get_context()
    assert context is not None
    assert context.plan_id == plan.plan_id
    assert context.iteration == 1

    # Verify metadata includes verification info
    assert "last_verification" in context.metadata
    verification_meta = context.metadata["last_verification"]
    assert verification_meta["valid"] is True
    assert verification_meta["confidence"] == 0.9
    assert verification_meta["iteration"] == 1


# ============================================================================
# Test: Module Transition Events
# ============================================================================


@pytest.mark.asyncio
async def test_module_transition_events(
    coordinator: ModuleCoordinator,
    a2a_context: A2AContext,
) -> None:
    """Test that module transition events are emitted."""
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
                action="test_action",
                parameters={},
                status=StepStatus.PENDING,
            )
        ],
        query="Test query",
    )

    planner.analyze_query = AsyncMock(return_value=plan)

    # Mock execution results
    execution_results = [
        ExecutionResult(
            step_id="step-1",
            success=True,
            result={"result": "success"},
            execution_time=1.5,
        )
    ]
    executor.execute_plan = AsyncMock(return_value=execution_results)

    # Mock verification result
    verification_result = VerificationResult(
        valid=True,
        confidence=0.9,
        errors=[],
        warnings=[],
    )
    verifier.validate_results = AsyncMock(return_value=verification_result)

    # Mock generation response
    generation_response = GeneratedResponse(
        format="text",
        content="Test answer",
        reasoning="Test reasoning",
        sources=["source1"],
    )
    generator.synthesize_response = AsyncMock(return_value=generation_response)

    # Execute coordination loop
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

    # Verify transitions are recorded in execution trace
    trace = result["execution_trace"]
    assert "transitions" in trace
    assert len(trace["transitions"]) > 0

    # Verify transition structure
    transitions = trace["transitions"]
    for transition in transitions:
        assert "from_module" in transition
        assert "to_module" in transition
        assert "reason" in transition
        assert "timestamp" in transition
        assert "iteration" in transition
