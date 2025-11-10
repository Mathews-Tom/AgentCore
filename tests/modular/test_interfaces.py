"""
Tests for Module Interface Protocols

Validates that interface protocols are properly defined with:
- Correct method signatures
- Pydantic model validation
- Type safety with mypy strict mode
"""

from __future__ import annotations

import pytest
from typing import Any

from agentcore.modular.interfaces import (
    # Models
    PlannerQuery,
    PlanStep,
    ExecutionPlan,
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
    # Protocols
    PlannerInterface,
    ExecutorInterface,
    VerifierInterface,
    GeneratorInterface,
)


class TestPlannerModels:
    """Test Planner module Pydantic models."""

    def test_planner_query_creation(self) -> None:
        """Test PlannerQuery model validation."""
        query = PlannerQuery(
            query="Test query",
            context={"user_id": "123"},
            constraints={"max_cost": 10.0},
        )
        assert query.query == "Test query"
        assert query.context["user_id"] == "123"
        assert query.constraints["max_cost"] == 10.0

    def test_plan_step_creation(self) -> None:
        """Test PlanStep model validation."""
        step = PlanStep(
            step_id="step-1",
            action="search",
            parameters={"query": "test"},
            dependencies=[],
            estimated_cost=0.5,
        )
        assert step.step_id == "step-1"
        assert step.action == "search"
        assert len(step.dependencies) == 0

    def test_execution_plan_creation(self) -> None:
        """Test ExecutionPlan model validation."""
        step1 = PlanStep(step_id="1", action="search", parameters={})
        step2 = PlanStep(step_id="2", action="process", parameters={}, dependencies=["1"])

        plan = ExecutionPlan(
            plan_id="plan-123",
            steps=[step1, step2],
            total_estimated_cost=1.5,
            metadata={"priority": "high"},
        )
        assert plan.plan_id == "plan-123"
        assert len(plan.steps) == 2
        assert plan.steps[1].dependencies == ["1"]

    def test_plan_refinement_creation(self) -> None:
        """Test PlanRefinement model validation."""
        refinement = PlanRefinement(
            plan_id="plan-123",
            feedback="Add error handling",
            constraints={"timeout": 30},
        )
        assert refinement.plan_id == "plan-123"
        assert refinement.feedback == "Add error handling"


class TestExecutorModels:
    """Test Executor module Pydantic models."""

    def test_execution_context_creation(self) -> None:
        """Test ExecutionContext model validation."""
        step = PlanStep(step_id="1", action="search", parameters={})
        context = ExecutionContext(
            step=step,
            previous_results={"step-0": "result"},
            timeout_seconds=10.0,
        )
        assert context.step.step_id == "1"
        assert context.previous_results["step-0"] == "result"
        assert context.timeout_seconds == 10.0

    def test_execution_result_creation(self) -> None:
        """Test ExecutionResult model validation."""
        result = ExecutionResult(
            step_id="1",
            success=True,
            result={"data": "test"},
            error=None,
            execution_time=0.5,
            metadata={"attempts": 1},
        )
        assert result.success is True
        assert result.result["data"] == "test"
        assert result.error is None

    def test_retry_policy_creation(self) -> None:
        """Test RetryPolicy model validation."""
        policy = RetryPolicy(
            max_attempts=5,
            backoff_seconds=2.0,
            exponential=True,
        )
        assert policy.max_attempts == 5
        assert policy.backoff_seconds == 2.0
        assert policy.exponential is True

    def test_retry_policy_defaults(self) -> None:
        """Test RetryPolicy default values."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_seconds == 1.0
        assert policy.exponential is True


class TestVerifierModels:
    """Test Verifier module Pydantic models."""

    def test_verification_request_creation(self) -> None:
        """Test VerificationRequest model validation."""
        result = ExecutionResult(
            step_id="1", success=True, result={"value": 42}, execution_time=0.1
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={"type": "object"},
            consistency_rules=["rule1", "rule2"],
        )
        assert len(request.results) == 1
        assert request.expected_json_schema["type"] == "object"
        assert len(request.consistency_rules) == 2

    def test_verification_result_creation(self) -> None:
        """Test VerificationResult model validation."""
        verification = VerificationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            feedback="Needs improvement",
            confidence=0.8,
        )
        assert verification.valid is False
        assert len(verification.errors) == 2
        assert len(verification.warnings) == 1
        assert verification.confidence == 0.8

    def test_verification_result_confidence_bounds(self) -> None:
        """Test VerificationResult confidence validation."""
        # Valid confidence values
        VerificationResult(valid=True, confidence=0.0)
        VerificationResult(valid=True, confidence=1.0)
        VerificationResult(valid=True, confidence=0.5)

        # Invalid confidence values should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            VerificationResult(valid=True, confidence=-0.1)
        with pytest.raises(Exception):
            VerificationResult(valid=True, confidence=1.1)

    def test_consistency_check_creation(self) -> None:
        """Test ConsistencyCheck model validation."""
        check = ConsistencyCheck(
            result_ids=["result-1", "result-2"],
            rule="values_match",
            parameters={"threshold": 0.9},
        )
        assert len(check.result_ids) == 2
        assert check.rule == "values_match"


class TestGeneratorModels:
    """Test Generator module Pydantic models."""

    def test_generation_request_creation(self) -> None:
        """Test GenerationRequest model validation."""
        result = ExecutionResult(
            step_id="1", success=True, result={"data": "test"}, execution_time=0.1
        )
        request = GenerationRequest(
            verified_results=[result],
            format="markdown",
            include_reasoning=True,
            max_length=1000,
        )
        assert len(request.verified_results) == 1
        assert request.format == "markdown"
        assert request.include_reasoning is True

    def test_generation_request_defaults(self) -> None:
        """Test GenerationRequest default values."""
        result = ExecutionResult(
            step_id="1", success=True, result={}, execution_time=0.1
        )
        request = GenerationRequest(verified_results=[result])
        assert request.format == "text"
        assert request.include_reasoning is False
        assert request.max_length is None

    def test_generated_response_creation(self) -> None:
        """Test GeneratedResponse model validation."""
        response = GeneratedResponse(
            content="Generated content",
            format="text",
            reasoning="Step 1, Step 2",
            sources=["source1", "source2"],
            metadata={"model": "gpt-4"},
        )
        assert response.content == "Generated content"
        assert response.reasoning == "Step 1, Step 2"
        assert len(response.sources) == 2

    def test_output_format_creation(self) -> None:
        """Test OutputFormat model validation."""
        format_spec = OutputFormat(
            type="json",
            json_schema={"type": "object", "properties": {}},
            template=None,
        )
        assert format_spec.type == "json"
        assert format_spec.json_schema["type"] == "object"


class TestProtocolInterfaces:
    """Test Protocol interface definitions."""

    def test_planner_interface_protocol(self) -> None:
        """Test PlannerInterface can be used as a Protocol."""

        class MockPlanner:
            async def analyze_query(self, query: PlannerQuery) -> ExecutionPlan:
                return ExecutionPlan(plan_id="test", steps=[])

            async def create_plan(
                self, query: str, query_context: dict[str, Any] | None = None
            ) -> ExecutionPlan:
                return ExecutionPlan(plan_id="test", steps=[])

            async def refine_plan(self, refinement: PlanRefinement) -> ExecutionPlan:
                return ExecutionPlan(plan_id="test", steps=[])

        planner: PlannerInterface = MockPlanner()
        assert planner is not None

    def test_executor_interface_protocol(self) -> None:
        """Test ExecutorInterface can be used as a Protocol."""

        class MockExecutor:
            async def execute_step(self, context: ExecutionContext) -> ExecutionResult:
                return ExecutionResult(
                    step_id="1", success=True, result={}, execution_time=0.1
                )

            async def execute_with_retry(
                self, context: ExecutionContext, policy: RetryPolicy
            ) -> ExecutionResult:
                return ExecutionResult(
                    step_id="1", success=True, result={}, execution_time=0.1
                )

            async def handle_tool_invocation(
                self, tool_name: str, parameters: dict[str, Any]
            ) -> Any:
                return {"result": "success"}

        executor: ExecutorInterface = MockExecutor()
        assert executor is not None

    def test_verifier_interface_protocol(self) -> None:
        """Test VerifierInterface can be used as a Protocol."""

        class MockVerifier:
            async def validate_results(
                self, request: VerificationRequest
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def check_consistency(
                self, check: ConsistencyCheck
            ) -> VerificationResult:
                return VerificationResult(valid=True)

            async def provide_feedback(self, results: list[ExecutionResult]) -> str:
                return "Feedback"

        verifier: VerifierInterface = MockVerifier()
        assert verifier is not None

    def test_generator_interface_protocol(self) -> None:
        """Test GeneratorInterface can be used as a Protocol."""

        class MockGenerator:
            async def synthesize_response(
                self, request: GenerationRequest
            ) -> GeneratedResponse:
                return GeneratedResponse(content="Test", format="text")

            async def format_output(
                self, content: str, format_spec: OutputFormat
            ) -> str:
                return content

            async def include_reasoning(
                self, response: GeneratedResponse, reasoning: str
            ) -> GeneratedResponse:
                response.reasoning = reasoning
                return response

        generator: GeneratorInterface = MockGenerator()
        assert generator is not None


class TestInterfaceImports:
    """Test that all interfaces can be imported."""

    def test_all_models_importable(self) -> None:
        """Test that all Pydantic models are importable."""
        # If we got this far, imports succeeded
        assert PlannerQuery is not None
        assert PlanStep is not None
        assert ExecutionPlan is not None
        assert PlanRefinement is not None
        assert ExecutionContext is not None
        assert ExecutionResult is not None
        assert RetryPolicy is not None
        assert VerificationRequest is not None
        assert VerificationResult is not None
        assert ConsistencyCheck is not None
        assert GenerationRequest is not None
        assert GeneratedResponse is not None
        assert OutputFormat is not None

    def test_all_protocols_importable(self) -> None:
        """Test that all Protocol interfaces are importable."""
        # If we got this far, imports succeeded
        assert PlannerInterface is not None
        assert ExecutorInterface is not None
        assert VerifierInterface is not None
        assert GeneratorInterface is not None
