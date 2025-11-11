"""
Tests for Base Module Classes

Validates that base classes provide:
- Structured logging with A2A context
- Error handling and tracking
- State management
- Context propagation
"""

from __future__ import annotations

import pytest
from typing import Any
from datetime import datetime, timezone

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.modular.base import (
    BaseModule,
    BasePlanner,
    BaseExecutor,
    BaseVerifier,
    BaseGenerator,
    ModuleState,
)
from agentcore.modular.interfaces import (
    PlannerQuery,
    ExecutionPlan,
    PlanStep,
    PlanRefinement,
    ExecutionContext,
    ExecutionResult,
    RetryPolicy,
    VerificationRequest,
    VerificationResult,
    ConsistencyCheck,
    GenerationRequest,
    GeneratedResponse,
    OutputFormat,
)


class TestModuleState:
    """Test ModuleState data model."""

    def test_module_state_creation(self) -> None:
        """Test ModuleState initialization."""
        state = ModuleState(
            module_name="TestModule",
            completed_at=None,
            error=None,
        )
        assert state.module_name == "TestModule"
        assert state.execution_id is not None
        assert state.started_at is not None
        assert state.completed_at is None
        assert state.error is None

    def test_module_state_with_error(self) -> None:
        """Test ModuleState with error."""
        state = ModuleState(
            module_name="TestModule",
            completed_at=None,
            error="Test error",
        )
        assert state.error == "Test error"


class TestBaseModule:
    """Test BaseModule abstract class."""

    def test_base_module_initialization(self) -> None:
        """Test BaseModule initialization with defaults."""

        class ConcreteModule(BaseModule):
            async def health_check(self) -> dict[str, Any]:
                return {"status": "healthy"}

        module = ConcreteModule("TestModule")

        assert module.module_name == "TestModule"
        assert module.a2a_context is not None
        assert module.a2a_context.source_agent == "modular-agent-core"
        assert module.state is not None
        assert module.state.module_name == "TestModule"

    def test_base_module_with_custom_context(self) -> None:
        """Test BaseModule with custom A2A context."""

        class ConcreteModule(BaseModule):
            async def health_check(self) -> dict[str, Any]:
                return {"status": "healthy"}

        context = A2AContext(
            source_agent="test-agent",
            target_agent="target-agent",
            trace_id="test-trace-123",
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="session-123",
            conversation_id="conv-123",
        )

        module = ConcreteModule("TestModule", a2a_context=context)

        assert module.a2a_context.source_agent == "test-agent"
        assert module.a2a_context.trace_id == "test-trace-123"
        assert module.a2a_context.session_id == "session-123"

    def test_base_module_error_logging(self) -> None:
        """Test error logging functionality."""

        class ConcreteModule(BaseModule):
            async def health_check(self) -> dict[str, Any]:
                return {"status": "healthy"}

        module = ConcreteModule("TestModule")

        error = ValueError("Test error")
        module._log_error(error, "test_operation", extra_field="value")

        assert module.state.error is not None
        assert "ValueError" in module.state.error
        assert "Test error" in module.state.error


class TestBasePlanner:
    """Test BasePlanner class."""

    @pytest.mark.asyncio
    async def test_base_planner_initialization(self) -> None:
        """Test BasePlanner initialization."""

        class ConcretePlanner(BasePlanner):
            async def _analyze_query_impl(
                self, query: PlannerQuery
            ) -> ExecutionPlan:
                return ExecutionPlan(
                    plan_id="test-plan",
                    steps=[
                        PlanStep(
                            step_id="1",
                            action="test",
                            parameters={},
                        )
                    ],
                )

            async def _refine_plan_impl(
                self, refinement: PlanRefinement
            ) -> ExecutionPlan:
                return ExecutionPlan(plan_id=refinement.plan_id, steps=[])

        planner = ConcretePlanner()

        assert planner.module_name == "Planner"
        health = await planner.health_check()
        assert health["status"] == "healthy"
        assert health["module"] == "Planner"

    @pytest.mark.asyncio
    async def test_base_planner_analyze_query(self) -> None:
        """Test analyze_query method."""

        class ConcretePlanner(BasePlanner):
            async def _analyze_query_impl(
                self, query: PlannerQuery
            ) -> ExecutionPlan:
                return ExecutionPlan(
                    plan_id="test-plan",
                    steps=[
                        PlanStep(
                            step_id="1",
                            action="search",
                            parameters={"query": query.query},
                        )
                    ],
                )

            async def _refine_plan_impl(
                self, refinement: PlanRefinement
            ) -> ExecutionPlan:
                return ExecutionPlan(plan_id=refinement.plan_id, steps=[])

        planner = ConcretePlanner()
        query = PlannerQuery(query="Test query")

        plan = await planner.analyze_query(query)

        assert plan.plan_id == "test-plan"
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "search"

    @pytest.mark.asyncio
    async def test_base_planner_create_plan(self) -> None:
        """Test create_plan convenience method."""

        class ConcretePlanner(BasePlanner):
            async def _analyze_query_impl(
                self, query: PlannerQuery
            ) -> ExecutionPlan:
                return ExecutionPlan(
                    plan_id="simple-plan",
                    steps=[
                        PlanStep(step_id="1", action="execute", parameters={})
                    ],
                )

            async def _refine_plan_impl(
                self, refinement: PlanRefinement
            ) -> ExecutionPlan:
                return ExecutionPlan(plan_id=refinement.plan_id, steps=[])

        planner = ConcretePlanner()
        plan = await planner.create_plan("Simple query")

        assert plan.plan_id == "simple-plan"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_base_planner_create_plan_empty_query(self) -> None:
        """Test create_plan with empty query raises ValueError."""

        class ConcretePlanner(BasePlanner):
            async def _analyze_query_impl(
                self, query: PlannerQuery
            ) -> ExecutionPlan:
                return ExecutionPlan(plan_id="test", steps=[])

            async def _refine_plan_impl(
                self, refinement: PlanRefinement
            ) -> ExecutionPlan:
                return ExecutionPlan(plan_id=refinement.plan_id, steps=[])

        planner = ConcretePlanner()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await planner.create_plan("")


class TestBaseExecutor:
    """Test BaseExecutor class."""

    @pytest.mark.asyncio
    async def test_base_executor_initialization(self) -> None:
        """Test BaseExecutor initialization."""

        class ConcreteExecutor(BaseExecutor):
            async def _execute_step_impl(
                self, context: ExecutionContext
            ) -> ExecutionResult:
                return ExecutionResult(
                    step_id=context.step.step_id,
                    success=True,
                    result={},
                    execution_time=0.1,
                )

            async def _execute_with_retry_impl(
                self, context: ExecutionContext, policy: RetryPolicy
            ) -> ExecutionResult:
                return await self._execute_step_impl(context)

            async def _handle_tool_invocation_impl(
                self, tool_name: str, parameters: dict[str, Any]
            ) -> Any:
                return {"tool": tool_name, "result": "success"}

        executor = ConcreteExecutor()

        assert executor.module_name == "Executor"
        health = await executor.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_base_executor_execute_step(self) -> None:
        """Test execute_step method."""

        class ConcreteExecutor(BaseExecutor):
            async def _execute_step_impl(
                self, context: ExecutionContext
            ) -> ExecutionResult:
                return ExecutionResult(
                    step_id=context.step.step_id,
                    success=True,
                    result={"data": "test"},
                    execution_time=0.05,
                )

            async def _execute_with_retry_impl(
                self, context: ExecutionContext, policy: RetryPolicy
            ) -> ExecutionResult:
                return await self._execute_step_impl(context)

            async def _handle_tool_invocation_impl(
                self, tool_name: str, parameters: dict[str, Any]
            ) -> Any:
                return {}

        executor = ConcreteExecutor()
        step = PlanStep(step_id="1", action="test", parameters={})
        context = ExecutionContext(step=step)

        result = await executor.execute_step(context)

        assert result.step_id == "1"
        assert result.success is True
        assert result.result["data"] == "test"


class TestBaseVerifier:
    """Test BaseVerifier class."""

    @pytest.mark.asyncio
    async def test_base_verifier_initialization(self) -> None:
        """Test BaseVerifier initialization."""

        class ConcreteVerifier(BaseVerifier):
            async def _validate_results_impl(
                self, request: VerificationRequest
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def _check_consistency_impl(
                self, check: ConsistencyCheck
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def _provide_feedback_impl(
                self, results: list[ExecutionResult]
            ) -> str:
                return "All good"

        verifier = ConcreteVerifier()

        assert verifier.module_name == "Verifier"
        health = await verifier.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_base_verifier_validate_results(self) -> None:
        """Test validate_results method."""

        class ConcreteVerifier(BaseVerifier):
            async def _validate_results_impl(
                self, request: VerificationRequest
            ) -> VerificationResult:
                if len(request.results) > 0:
                    return VerificationResult(valid=True)
                return VerificationResult(
                    valid=False, errors=["No results provided"]
                )

            async def _check_consistency_impl(
                self, check: ConsistencyCheck
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def _provide_feedback_impl(
                self, results: list[ExecutionResult]
            ) -> str:
                return "Feedback"

        verifier = ConcreteVerifier()
        result = ExecutionResult(
            step_id="1", success=True, result={}, execution_time=0.1
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.valid is True

    @pytest.mark.asyncio
    async def test_base_verifier_provide_feedback_empty_list(self) -> None:
        """Test provide_feedback with empty list raises ValueError."""

        class ConcreteVerifier(BaseVerifier):
            async def _validate_results_impl(
                self, request: VerificationRequest
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def _check_consistency_impl(
                self, check: ConsistencyCheck
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def _provide_feedback_impl(
                self, results: list[ExecutionResult]
            ) -> str:
                return "Feedback"

        verifier = ConcreteVerifier()

        with pytest.raises(ValueError, match="Results list cannot be empty"):
            await verifier.provide_feedback([])


class TestBaseGenerator:
    """Test BaseGenerator class."""

    @pytest.mark.asyncio
    async def test_base_generator_initialization(self) -> None:
        """Test BaseGenerator initialization."""

        class ConcreteGenerator(BaseGenerator):
            async def _synthesize_response_impl(
                self, request: GenerationRequest
            ) -> GeneratedResponse:
                return GeneratedResponse(
                    content="Test response", format="text"
                )

            async def _format_output_impl(
                self, content: str, format_spec: OutputFormat
            ) -> str:
                return content

            async def _include_reasoning_impl(
                self, response: GeneratedResponse, reasoning: str
            ) -> GeneratedResponse:
                response.reasoning = reasoning
                return response

        generator = ConcreteGenerator()

        assert generator.module_name == "Generator"
        health = await generator.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_base_generator_synthesize_response(self) -> None:
        """Test synthesize_response method."""

        class ConcreteGenerator(BaseGenerator):
            async def _synthesize_response_impl(
                self, request: GenerationRequest
            ) -> GeneratedResponse:
                content = f"Generated from {len(request.verified_results)} results"
                return GeneratedResponse(
                    content=content, format=request.format
                )

            async def _format_output_impl(
                self, content: str, format_spec: OutputFormat
            ) -> str:
                return content

            async def _include_reasoning_impl(
                self, response: GeneratedResponse, reasoning: str
            ) -> GeneratedResponse:
                response.reasoning = reasoning
                return response

        generator = ConcreteGenerator()
        result = ExecutionResult(
            step_id="1", success=True, result={}, execution_time=0.1
        )
        request = GenerationRequest(
            verified_results=[result], format="markdown"
        )

        response = await generator.synthesize_response(request)

        assert "Generated from 1 results" in response.content
        assert response.format == "markdown"
