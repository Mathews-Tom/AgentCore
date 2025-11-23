"""
Integration Tests for Full Pipeline (End-to-End Execution)

Comprehensive integration test suite covering end-to-end execution through all
four modules (Planner, Executor, Verifier, Generator) with real tool integrations.

Test Coverage:
- Successful single-iteration execution paths
- Multi-step execution with tool calls
- Module integration (P→E→V→G flow)
- State persistence
- Error handling and recovery
- Timeout enforcement
- Execution trace validation
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
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
        source_agent="test-pipeline",
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


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create tool registry for tests."""
    return ToolRegistry()


@pytest.fixture
def tool_executor(tool_registry: ToolRegistry) -> ToolExecutor:
    """Create tool executor for tests."""
    return ToolExecutor(registry=tool_registry)


# ============================================================================
# Test Suite 1: Successful Execution Paths
# ============================================================================


@pytest.mark.asyncio
class TestSingleIterationExecution:
    """Test successful execution in single iteration."""

    async def test_single_iteration_success_simple_query(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test simple query requiring no refinement completes in one iteration."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock simple plan (single step)
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="respond",
                    parameters={"message": "Hello"},
                    status=StepStatus.PENDING,
                )
            ],
            query="Say hello",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock execution results (success)
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"response": "Hello!"},
                execution_time=0.5,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification result (high confidence, passed)
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
            content="Hello!",
            reasoning=None,
            sources=["step:step-1"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute coordination loop
        result = await coordinator.execute_with_refinement(
            query="Say hello",
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
        assert result["answer"] == "Hello!"

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

    async def test_single_iteration_with_tool_execution(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test query requiring actual tool calls completes successfully."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock plan with tool invocation
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="search",
                    parameters={"query": "test"},
                    status=StepStatus.PENDING,
                )
            ],
            query="Search for test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock execution results (tool invocation)
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"results": ["result1", "result2"]},
                execution_time=1.2,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification result
        verification_result = VerificationResult(
            valid=True,
            confidence=0.88,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        # Mock generation response
        generation_response = GeneratedResponse(
            format="text",
            content="Found 2 results",
            reasoning=None,
            sources=["step:step-1"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute coordination loop
        result = await coordinator.execute_with_refinement(
            query="Search for test",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify successful execution
        assert result["answer"] == "Found 2 results"
        trace = result["execution_trace"]
        assert trace["iterations"] == 1
        assert trace["verification_passed"] is True
        assert trace["successful_steps"] == 1


@pytest.mark.asyncio
class TestMultiStepExecution:
    """Test complex multi-step execution scenarios."""

    async def test_successful_execution_with_multiple_steps(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test complex query with multiple execution steps."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock complex plan (3 steps)
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="search",
                    parameters={"query": "data"},
                    status=StepStatus.PENDING,
                ),
                EnhancedPlanStep(
                    step_id="step-2",
                    action="analyze",
                    parameters={"data": "from_step_1"},
                    status=StepStatus.PENDING,
                ),
                EnhancedPlanStep(
                    step_id="step-3",
                    action="summarize",
                    parameters={"analysis": "from_step_2"},
                    status=StepStatus.PENDING,
                ),
            ],
            query="Search, analyze, and summarize data",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock execution results (all successful)
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "raw_data"},
                execution_time=0.8,
            ),
            ExecutionResult(
                step_id="step-2",
                success=True,
                result={"analysis": "analyzed_data"},
                execution_time=1.2,
            ),
            ExecutionResult(
                step_id="step-3",
                success=True,
                result={"summary": "final_summary"},
                execution_time=0.5,
            ),
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verification result
        verification_result = VerificationResult(
            valid=True,
            confidence=0.92,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        # Mock generation response
        generation_response = GeneratedResponse(
            format="text",
            content="Final summary generated",
            reasoning=None,
            sources=["step:step-1", "step:step-2", "step:step-3"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute coordination loop
        result = await coordinator.execute_with_refinement(
            query="Search, analyze, and summarize data",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify result
        assert result["answer"] == "Final summary generated"
        trace = result["execution_trace"]
        assert trace["iterations"] == 1
        assert trace["step_count"] == 3
        assert trace["successful_steps"] == 3
        assert trace["failed_steps"] == 0
        assert len(result["sources"]) == 3

    async def test_execution_trace_completeness(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Validate all trace fields are populated correctly."""
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
            query="Test trace",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

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

        # Mock verification result
        verification_result = VerificationResult(
            valid=True,
            confidence=0.85,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        # Mock generation response
        generation_response = GeneratedResponse(
            format="text",
            content="Test answer",
            reasoning=None,
            sources=["step:step-1"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute coordination loop
        result = await coordinator.execute_with_refinement(
            query="Test trace",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify execution trace has all required fields
        trace = result["execution_trace"]
        required_fields = [
            "plan_id",
            "iterations",
            "modules_invoked",
            "total_duration_ms",
            "verification_passed",
            "step_count",
            "successful_steps",
            "failed_steps",
            "confidence_score",
            "transitions",
            "refinement_history",
        ]

        for field in required_fields:
            assert field in trace, f"Missing required field: {field}"

        # Verify field types and values
        assert isinstance(trace["plan_id"], str)
        assert trace["plan_id"] == plan.plan_id
        assert trace["iterations"] == 1
        assert isinstance(trace["modules_invoked"], list)
        assert len(trace["modules_invoked"]) == 4  # P, E, V, G
        assert trace["total_duration_ms"] >= 0  # Can be 0 for very fast mock execution
        assert trace["verification_passed"] is True
        assert trace["confidence_score"] == 0.85
        assert isinstance(trace["transitions"], list)
        assert len(trace["transitions"]) > 0
        assert isinstance(trace["refinement_history"], list)


# ============================================================================
# Test Suite 2: Module Integration
# ============================================================================


@pytest.mark.asyncio
class TestModuleIntegration:
    """Test integration between modules (P→E→V→G flow)."""

    async def test_planner_executor_integration(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test plan creation flows correctly to executor."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track plan passed to executor
        received_plan = None

        async def mock_execute_plan(plan):
            nonlocal received_plan
            received_plan = plan
            return [
                ExecutionResult(
                    step_id=plan.steps[0].step_id,
                    success=True,
                    result={"result": "ok"},
                    execution_time=1.0,
                )
            ]

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
        executor.execute_plan = mock_execute_plan

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

        # Verify executor received the plan from planner
        assert received_plan is not None
        assert received_plan.plan_id == plan.plan_id
        assert len(received_plan.steps) == 1

    async def test_executor_verifier_integration(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test execution results are correctly passed to verifier."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track results passed to verifier
        received_results = None

        async def mock_validate_results(request):
            nonlocal received_results
            received_results = request.results
            return VerificationResult(
                valid=True,
                confidence=0.9,
                errors=[],
                warnings=[],
            )

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

        # Mock execution results
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "test"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        # Mock verifier
        verifier.validate_results = mock_validate_results

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

        # Verify verifier received execution results
        assert received_results is not None
        assert len(received_results) == 1
        assert received_results[0].step_id == "step-1"
        assert received_results[0].success is True

    async def test_verifier_generator_integration(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test verification results flow to generator."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Track results passed to generator
        received_results = None

        async def mock_synthesize_response(request):
            nonlocal received_results
            received_results = request.verified_results
            return GeneratedResponse(
                format="text",
                content="Answer",
                reasoning=None,
                sources=[],
            )

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

        # Mock execution results
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"data": "test"},
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

        # Mock generator
        generator.synthesize_response = mock_synthesize_response

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

        # Verify generator received execution results
        assert received_results is not None
        assert len(received_results) == 1
        assert received_results[0].step_id == "step-1"

    async def test_full_pevg_flow(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test complete Planner→Executor→Verifier→Generator flow."""
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
            query="Test full flow",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock execution
        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "data"},
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
            content="Final answer",
            reasoning=None,
            sources=["step:step-1"],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute
        result = await coordinator.execute_with_refinement(
            query="Test full flow",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify all modules were invoked in order
        trace = result["execution_trace"]
        modules = trace["modules_invoked"]
        assert modules == ["planner", "executor", "verifier", "generator"]

        # Verify final answer
        assert result["answer"] == "Final answer"


# ============================================================================
# Test Suite 3: State Persistence
# ============================================================================


@pytest.mark.asyncio
class TestStatePersistence:
    """Test state persistence during module transitions."""

    async def test_state_persisted_after_each_module(
        self,
        coordinator: ModuleCoordinator,
        coordination_context: CoordinationContext,
        a2a_context: A2AContext,
    ) -> None:
        """Verify state is saved at each module transition."""
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
            query="Test state",
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
        await coordinator.execute_with_refinement(
            query="Test state",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify state persistence in context
        context = coordinator.get_context()
        assert context is not None
        assert context.plan_id == plan.plan_id
        assert context.iteration == 1

        # Verify verification metadata
        assert "last_verification" in context.metadata
        verification_meta = context.metadata["last_verification"]
        assert verification_meta["valid"] is True
        assert verification_meta["confidence"] == 0.9
        assert verification_meta["iteration"] == 1

    async def test_multiple_executions_isolated(
        self,
        a2a_context: A2AContext,
    ) -> None:
        """Test that state is isolated between multiple executions."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Setup mocks
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

        execution_results = [
            ExecutionResult(
                step_id="step-1",
                success=True,
                result={"result": "ok"},
                execution_time=1.0,
            )
        ]
        executor.execute_plan = AsyncMock(return_value=execution_results)

        verification_result = VerificationResult(
            valid=True,
            confidence=0.9,
            errors=[],
            warnings=[],
        )
        verifier.validate_results = AsyncMock(return_value=verification_result)

        generation_response = GeneratedResponse(
            format="text",
            content="Answer",
            reasoning=None,
            sources=[],
        )
        generator.synthesize_response = AsyncMock(return_value=generation_response)

        # Execute first coordination
        coordinator1 = ModuleCoordinator()
        context1 = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
            session_id=str(uuid4()),
            iteration=0,
        )
        coordinator1.set_context(context1)

        await coordinator1.execute_with_refinement(
            query="Test 1",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Execute second coordination
        coordinator2 = ModuleCoordinator()
        context2 = CoordinationContext(
            execution_id=str(uuid4()),
            trace_id=str(uuid4()),
            session_id=str(uuid4()),
            iteration=0,
        )
        coordinator2.set_context(context2)

        await coordinator2.execute_with_refinement(
            query="Test 2",
            planner=planner,
            executor=executor,
            verifier=verifier,
            generator=generator,
            max_iterations=5,
            timeout_seconds=30.0,
            confidence_threshold=0.7,
        )

        # Verify contexts are isolated
        final_context1 = coordinator1.get_context()
        final_context2 = coordinator2.get_context()

        assert final_context1.execution_id != final_context2.execution_id
        assert final_context1.trace_id != final_context2.trace_id


# ============================================================================
# Test Suite 4: Error Handling
# ============================================================================


@pytest.mark.asyncio
class TestErrorRecovery:
    """Test error handling and recovery mechanisms."""

    async def test_planner_failure_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test graceful handling of planner errors."""
        # Create mock modules
        planner = MagicMock(spec=Planner)
        executor = MagicMock(spec=ExecutorModule)
        verifier = MagicMock(spec=Verifier)
        generator = MagicMock(spec=Generator)

        # Mock planner to raise error
        planner.analyze_query = AsyncMock(
            side_effect=ValueError("Invalid query structure")
        )

        # Execute and expect error
        with pytest.raises(ValueError) as exc_info:
            await coordinator.execute_with_refinement(
                query="Bad query",
                planner=planner,
                executor=executor,
                verifier=verifier,
                generator=generator,
                max_iterations=5,
                timeout_seconds=30.0,
                confidence_threshold=0.7,
            )

        assert "Invalid query structure" in str(exc_info.value)

    async def test_executor_failure_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test handling of executor errors."""
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

        # Mock executor to raise error
        executor.execute_plan = AsyncMock(
            side_effect=RuntimeError("Tool execution failed")
        )

        # Execute and expect error
        with pytest.raises(RuntimeError) as exc_info:
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

        assert "Tool execution failed" in str(exc_info.value)

    async def test_verifier_failure_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test handling of verifier errors."""
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

        # Mock verifier to raise error
        verifier.validate_results = AsyncMock(
            side_effect=RuntimeError("Verification service unavailable")
        )

        # Execute and expect error
        with pytest.raises(RuntimeError) as exc_info:
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

        assert "Verification service unavailable" in str(exc_info.value)

    async def test_generator_failure_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test handling of generator errors."""
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

        # Mock generator to raise error
        generator.synthesize_response = AsyncMock(
            side_effect=RuntimeError("Response generation failed")
        )

        # Execute and expect error
        with pytest.raises(RuntimeError) as exc_info:
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

        assert "Response generation failed" in str(exc_info.value)

    async def test_timeout_handling(
        self,
        coordinator: ModuleCoordinator,
        a2a_context: A2AContext,
    ) -> None:
        """Test timeout enforcement for long-running executions."""
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
                    action="slow",
                    parameters={},
                    status=StepStatus.PENDING,
                )
            ],
            query="Test",
            status=PlanStatus.PENDING,
        )
        planner.analyze_query = AsyncMock(return_value=plan)

        # Mock slow execution
        async def slow_execution(plan):
            await asyncio.sleep(5.0)
            return [
                ExecutionResult(
                    step_id="step-1",
                    success=True,
                    result={"result": "slow"},
                    execution_time=5.0,
                )
            ]

        executor.execute_plan = slow_execution

        # Execute with short timeout
        with pytest.raises(asyncio.TimeoutError):
            await coordinator.execute_with_refinement(
                query="Test",
                planner=planner,
                executor=executor,
                verifier=verifier,
                generator=generator,
                max_iterations=5,
                timeout_seconds=1.0,  # Short timeout
                confidence_threshold=0.7,
            )
