"""
Module Interface Protocols for Modular Agent Core

This module defines Protocol interfaces for the four-module architecture:
- PlannerInterface: Query analysis and execution planning
- ExecutorInterface: Plan execution and tool orchestration
- VerifierInterface: Result validation and consistency checking
- GeneratorInterface: Response synthesis and formatting

All interfaces use async methods with Pydantic models for type safety
and comprehensive error handling contracts.
"""

from __future__ import annotations

from typing import Protocol, Any
from pydantic import BaseModel, Field


# ============================================================================
# Planner Module Models
# ============================================================================


class PlannerQuery(BaseModel):
    """Input query for the planner module."""

    query: str = Field(..., description="User query to analyze and plan for")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context for planning"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Constraints on plan generation"
    )


class PlanStep(BaseModel):
    """Single step in an execution plan."""

    step_id: str = Field(..., description="Unique identifier for this step")
    action: str = Field(..., description="Action to perform")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Step IDs that must complete first"
    )
    estimated_cost: float = Field(default=0.0, description="Estimated execution cost")


class ExecutionPlan(BaseModel):
    """Complete execution plan with ordered steps."""

    plan_id: str = Field(..., description="Unique identifier for this plan")
    steps: list[PlanStep] = Field(..., description="Ordered list of execution steps")
    total_estimated_cost: float = Field(
        default=0.0, description="Total estimated cost for entire plan"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional plan metadata"
    )


class PlanRefinement(BaseModel):
    """Refinement request for an existing plan."""

    plan_id: str = Field(..., description="ID of plan to refine")
    feedback: str = Field(..., description="Feedback for refinement")
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Additional constraints"
    )


# ============================================================================
# Executor Module Models
# ============================================================================


class ExecutionContext(BaseModel):
    """Context for executing a plan step."""

    step: PlanStep = Field(..., description="Step to execute")
    previous_results: dict[str, Any] = Field(
        default_factory=dict, description="Results from previous steps"
    )
    timeout_seconds: float = Field(
        default=30.0, description="Maximum execution time"
    )


class ExecutionResult(BaseModel):
    """Result of executing a plan step."""

    step_id: str = Field(..., description="ID of executed step")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(None, description="Execution result data")
    error: str | None = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )


class RetryPolicy(BaseModel):
    """Policy for retrying failed executions."""

    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    backoff_seconds: float = Field(default=1.0, description="Initial backoff delay")
    exponential: bool = Field(
        default=True, description="Use exponential backoff"
    )


# ============================================================================
# Verifier Module Models
# ============================================================================


class VerificationRequest(BaseModel):
    """Request to verify execution results."""

    results: list[ExecutionResult] = Field(
        ..., description="Results to verify"
    )
    expected_json_schema: dict[str, Any] | None = Field(
        None, description="Expected result schema"
    )
    consistency_rules: list[str] = Field(
        default_factory=list, description="Consistency rules to check"
    )


class VerificationResult(BaseModel):
    """Result of verification checks."""

    valid: bool = Field(..., description="Whether results are valid")
    errors: list[str] = Field(
        default_factory=list, description="Validation errors found"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    feedback: str | None = Field(
        None, description="Feedback for improvement"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in validation"
    )


class ConsistencyCheck(BaseModel):
    """Consistency check between results."""

    result_ids: list[str] = Field(
        ..., description="IDs of results to check"
    )
    rule: str = Field(..., description="Consistency rule to apply")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Rule parameters"
    )


# ============================================================================
# Generator Module Models
# ============================================================================


class GenerationRequest(BaseModel):
    """Request to generate final response."""

    verified_results: list[ExecutionResult] = Field(
        ..., description="Verified execution results"
    )
    format: str = Field(
        default="text", description="Output format (text, json, markdown)"
    )
    include_reasoning: bool = Field(
        default=False, description="Include reasoning chain"
    )
    max_length: int | None = Field(
        None, description="Maximum response length"
    )


class GeneratedResponse(BaseModel):
    """Generated response with metadata."""

    content: str = Field(..., description="Generated response content")
    format: str = Field(..., description="Response format")
    reasoning: str | None = Field(
        None, description="Reasoning chain if requested"
    )
    sources: list[str] = Field(
        default_factory=list, description="Sources used in generation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )


class OutputFormat(BaseModel):
    """Output format specification."""

    type: str = Field(..., description="Format type (text, json, markdown, html)")
    json_schema: dict[str, Any] | None = Field(
        None, description="JSON schema if applicable"
    )
    template: str | None = Field(
        None, description="Template string if applicable"
    )


# ============================================================================
# Protocol Interfaces
# ============================================================================


class PlannerInterface(Protocol):
    """
    Protocol for the Planner module.

    The Planner is responsible for analyzing user queries and generating
    structured execution plans that can be executed by the Executor module.

    Example:
        >>> planner: PlannerInterface = MyPlannerImplementation()
        >>> query = PlannerQuery(query="Calculate fibonacci(10)")
        >>> plan = await planner.analyze_query(query)
        >>> print(f"Plan has {len(plan.steps)} steps")
    """

    async def analyze_query(self, query: PlannerQuery) -> ExecutionPlan:
        """
        Analyze a user query and generate an execution plan.

        Args:
            query: Query with context and constraints

        Returns:
            Execution plan with ordered steps

        Raises:
            ValueError: If query is invalid or cannot be planned
            RuntimeError: If planning fails due to internal error

        Example:
            >>> query = PlannerQuery(
            ...     query="Search for Python tutorials",
            ...     context={"user_level": "beginner"}
            ... )
            >>> plan = await planner.analyze_query(query)
        """
        ...

    async def create_plan(
        self, query: str, query_context: dict[str, Any] | None = None
    ) -> ExecutionPlan:
        """
        Create an execution plan from a simple query string.

        Convenience method for creating plans without full PlannerQuery.

        Args:
            query: User query string
            query_context: Optional context dictionary

        Returns:
            Execution plan with ordered steps

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If planning fails

        Example:
            >>> plan = await planner.create_plan("Calculate 2 + 2")
        """
        ...

    async def refine_plan(self, refinement: PlanRefinement) -> ExecutionPlan:
        """
        Refine an existing plan based on feedback.

        Args:
            refinement: Refinement request with feedback

        Returns:
            Refined execution plan

        Raises:
            ValueError: If plan_id not found or feedback invalid
            RuntimeError: If refinement fails

        Example:
            >>> refinement = PlanRefinement(
            ...     plan_id="plan-123",
            ...     feedback="Add error handling to step 2"
            ... )
            >>> refined = await planner.refine_plan(refinement)
        """
        ...


class ExecutorInterface(Protocol):
    """
    Protocol for the Executor module.

    The Executor is responsible for executing plan steps, managing tool
    invocations, handling retries, and collecting execution results.

    Example:
        >>> executor: ExecutorInterface = MyExecutorImplementation()
        >>> context = ExecutionContext(step=plan.steps[0])
        >>> result = await executor.execute_step(context)
        >>> print(f"Step completed in {result.execution_time}s")
    """

    async def execute_step(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute a single plan step.

        Args:
            context: Execution context with step and previous results

        Returns:
            Execution result with success status and data

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails after retries

        Example:
            >>> context = ExecutionContext(
            ...     step=PlanStep(step_id="1", action="search", parameters={}),
            ...     timeout_seconds=10.0
            ... )
            >>> result = await executor.execute_step(context)
        """
        ...

    async def execute_with_retry(
        self, context: ExecutionContext, policy: RetryPolicy
    ) -> ExecutionResult:
        """
        Execute a step with retry policy.

        Args:
            context: Execution context
            policy: Retry policy configuration

        Returns:
            Execution result after retries if needed

        Raises:
            TimeoutError: If all attempts exceed timeout
            RuntimeError: If all retry attempts fail

        Example:
            >>> policy = RetryPolicy(max_attempts=3, backoff_seconds=2.0)
            >>> result = await executor.execute_with_retry(context, policy)
        """
        ...

    async def handle_tool_invocation(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """
        Handle invocation of a specific tool.

        Args:
            tool_name: Name of tool to invoke
            parameters: Tool parameters

        Returns:
            Tool invocation result

        Raises:
            ValueError: If tool not found or parameters invalid
            RuntimeError: If tool invocation fails

        Example:
            >>> result = await executor.handle_tool_invocation(
            ...     "calculator",
            ...     {"operation": "add", "a": 2, "b": 3}
            ... )
        """
        ...


class VerifierInterface(Protocol):
    """
    Protocol for the Verifier module.

    The Verifier is responsible for validating execution results, checking
    consistency, and providing feedback for plan refinement.

    Example:
        >>> verifier: VerifierInterface = MyVerifierImplementation()
        >>> request = VerificationRequest(results=[result1, result2])
        >>> verification = await verifier.validate_results(request)
        >>> if not verification.valid:
        ...     print(f"Errors: {verification.errors}")
    """

    async def validate_results(
        self, request: VerificationRequest
    ) -> VerificationResult:
        """
        Validate execution results against expected schema.

        Args:
            request: Verification request with results and rules

        Returns:
            Verification result with errors and feedback

        Raises:
            ValueError: If verification request is invalid
            RuntimeError: If verification process fails

        Example:
            >>> request = VerificationRequest(
            ...     results=[result],
            ...     expected_json_schema={"type": "number"}
            ... )
            >>> verification = await verifier.validate_results(request)
        """
        ...

    async def check_consistency(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Check consistency between multiple results.

        Args:
            check: Consistency check specification

        Returns:
            Verification result for consistency check

        Raises:
            ValueError: If check specification is invalid
            RuntimeError: If consistency check fails

        Example:
            >>> check = ConsistencyCheck(
            ...     result_ids=["result-1", "result-2"],
            ...     rule="values_match"
            ... )
            >>> verification = await verifier.check_consistency(check)
        """
        ...

    async def provide_feedback(
        self, results: list[ExecutionResult]
    ) -> str:
        """
        Provide feedback for improving execution results.

        Args:
            results: Execution results to analyze

        Returns:
            Feedback string for plan refinement

        Raises:
            ValueError: If results list is empty
            RuntimeError: If feedback generation fails

        Example:
            >>> feedback = await verifier.provide_feedback([result1, result2])
            >>> print(f"Feedback: {feedback}")
        """
        ...


class GeneratorInterface(Protocol):
    """
    Protocol for the Generator module.

    The Generator is responsible for synthesizing final responses from
    verified execution results, with support for multiple output formats
    and optional reasoning chains.

    Example:
        >>> generator: GeneratorInterface = MyGeneratorImplementation()
        >>> request = GenerationRequest(verified_results=[result])
        >>> response = await generator.synthesize_response(request)
        >>> print(response.content)
    """

    async def synthesize_response(
        self, request: GenerationRequest
    ) -> GeneratedResponse:
        """
        Synthesize final response from verified results.

        Args:
            request: Generation request with results and format

        Returns:
            Generated response with content and metadata

        Raises:
            ValueError: If generation request is invalid
            RuntimeError: If response synthesis fails

        Example:
            >>> request = GenerationRequest(
            ...     verified_results=[result],
            ...     format="markdown",
            ...     include_reasoning=True
            ... )
            >>> response = await generator.synthesize_response(request)
        """
        ...

    async def format_output(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Format content according to output specification.

        Args:
            content: Content to format
            format_spec: Output format specification

        Returns:
            Formatted content string

        Raises:
            ValueError: If format specification is invalid
            RuntimeError: If formatting fails

        Example:
            >>> format_spec = OutputFormat(type="json", json_schema={"type": "object"})
            >>> formatted = await generator.format_output(content, format_spec)
        """
        ...

    async def include_reasoning(
        self, response: GeneratedResponse, reasoning: str
    ) -> GeneratedResponse:
        """
        Add reasoning chain to generated response.

        Args:
            response: Generated response to enhance
            reasoning: Reasoning chain to include

        Returns:
            Enhanced response with reasoning

        Raises:
            ValueError: If response or reasoning is invalid
            RuntimeError: If reasoning inclusion fails

        Example:
            >>> enhanced = await generator.include_reasoning(
            ...     response,
            ...     "Step 1: Analyzed query\\nStep 2: Executed plan"
            ... )
        """
        ...
