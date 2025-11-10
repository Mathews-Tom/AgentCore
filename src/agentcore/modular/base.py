"""
Base Module Classes for Modular Agent Core

Provides abstract base classes with common functionality for all modules:
- Structured logging with A2A context propagation
- Error handling with detailed error tracking
- State management for module execution
- A2A context (trace_id, source_agent, session_id) propagation

All modules inherit from these base classes to ensure consistent
behavior across the four-module architecture.
"""

from __future__ import annotations

import structlog
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.modular.interfaces import (
    ExecutionPlan,
    PlannerQuery,
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


# ============================================================================
# Module State Management
# ============================================================================


class ModuleState(BaseModel):
    """State tracking for module execution."""

    module_name: str = Field(..., description="Name of the module")
    execution_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique execution ID"
    )
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Execution start time",
    )
    completed_at: str | None = Field(None, description="Execution completion time")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional state metadata"
    )


# ============================================================================
# Base Module Class
# ============================================================================


class BaseModule(ABC):
    """
    Abstract base class for all modular agent modules.

    Provides common functionality:
    - Structured logging with A2A context
    - Error handling and tracking
    - State management
    - Context propagation

    All modules (Planner, Executor, Verifier, Generator) inherit from this.
    """

    def __init__(
        self,
        module_name: str,
        a2a_context: A2AContext | None = None,
        logger: Any | None = None,
    ) -> None:
        """
        Initialize base module.

        Args:
            module_name: Name of the module (e.g., "Planner", "Executor")
            a2a_context: A2A protocol context for tracing
            logger: Optional structlog logger instance
        """
        self.module_name = module_name
        self.a2a_context = a2a_context or self._create_default_context()

        # Initialize structured logger
        if logger:
            self.logger = logger
        else:
            self.logger = structlog.get_logger()

        # Bind A2A context to logger
        self.logger = self.logger.bind(
            module=self.module_name,
            trace_id=self.a2a_context.trace_id,
            source_agent=self.a2a_context.source_agent,
            session_id=self.a2a_context.session_id,
        )

        # Initialize state
        self.state = ModuleState(
            module_name=module_name,
            completed_at=None,
            error=None,
        )

        self.logger.info(
            "module_initialized",
            execution_id=self.state.execution_id,
            module=self.module_name,
        )

    def _create_default_context(self) -> A2AContext:
        """Create default A2A context if none provided."""
        return A2AContext(
            source_agent="modular-agent-core",
            target_agent=None,
            trace_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=None,
            conversation_id=None,
        )

    def _log_error(
        self, error: Exception, operation: str, **kwargs: Any
    ) -> None:
        """
        Log error with structured context.

        Args:
            error: Exception that occurred
            operation: Operation that failed
            **kwargs: Additional context fields
        """
        self.logger.error(
            "module_error",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            execution_id=self.state.execution_id,
            **kwargs,
        )

        # Update state with error
        self.state.error = f"{type(error).__name__}: {str(error)}"

    def _log_operation(
        self, operation: str, status: str, **kwargs: Any
    ) -> None:
        """
        Log module operation with structured context.

        Args:
            operation: Operation name
            status: Operation status (started, completed, failed)
            **kwargs: Additional context fields
        """
        self.logger.info(
            "module_operation",
            operation=operation,
            status=status,
            execution_id=self.state.execution_id,
            **kwargs,
        )

    def _complete_execution(self) -> None:
        """Mark execution as completed."""
        self.state.completed_at = datetime.now(timezone.utc).isoformat()
        self.logger.info(
            "module_execution_complete",
            execution_id=self.state.execution_id,
            duration=self._calculate_duration(),
        )

    def _calculate_duration(self) -> float:
        """Calculate execution duration in seconds."""
        if not self.state.completed_at:
            return 0.0

        started = datetime.fromisoformat(self.state.started_at.replace("Z", "+00:00"))
        completed = datetime.fromisoformat(
            self.state.completed_at.replace("Z", "+00:00")
        )
        return (completed - started).total_seconds()

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check module health and readiness.

        Returns:
            Health status dict with module-specific metrics

        Example:
            {
                "status": "healthy",
                "module": "Planner",
                "uptime": 3600.0,
                "execution_count": 42
            }
        """
        ...


# ============================================================================
# Planner Base Class
# ============================================================================


class BasePlanner(BaseModule):
    """
    Base class for Planner module implementations.

    Provides common planning functionality with logging and error handling.
    """

    def __init__(
        self,
        a2a_context: A2AContext | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__("Planner", a2a_context, logger)

    async def analyze_query(self, query: PlannerQuery) -> ExecutionPlan:
        """
        Analyze query and generate execution plan.

        Args:
            query: Query to analyze

        Returns:
            Generated execution plan

        Raises:
            ValueError: If query is invalid
            RuntimeError: If planning fails
        """
        self._log_operation("analyze_query", "started", query=query.query)

        try:
            plan = await self._analyze_query_impl(query)
            self._log_operation(
                "analyze_query",
                "completed",
                plan_id=plan.plan_id,
                steps=len(plan.steps),
            )
            return plan

        except Exception as e:
            self._log_error(e, "analyze_query", query=query.query)
            raise

    async def create_plan(
        self, query: str, query_context: dict[str, Any] | None = None
    ) -> ExecutionPlan:
        """
        Create execution plan from query string.

        Args:
            query: Query string
            query_context: Optional context

        Returns:
            Generated execution plan

        Raises:
            ValueError: If query is empty
            RuntimeError: If planning fails
        """
        if not query:
            raise ValueError("Query cannot be empty")

        planner_query = PlannerQuery(
            query=query,
            context=query_context or {},
        )
        return await self.analyze_query(planner_query)

    async def refine_plan(self, refinement: PlanRefinement) -> ExecutionPlan:
        """
        Refine existing plan based on feedback.

        Args:
            refinement: Refinement request

        Returns:
            Refined execution plan

        Raises:
            ValueError: If plan_id not found
            RuntimeError: If refinement fails
        """
        self._log_operation(
            "refine_plan", "started", plan_id=refinement.plan_id
        )

        try:
            refined_plan = await self._refine_plan_impl(refinement)
            self._log_operation(
                "refine_plan",
                "completed",
                plan_id=refined_plan.plan_id,
                steps=len(refined_plan.steps),
            )
            return refined_plan

        except Exception as e:
            self._log_error(e, "refine_plan", plan_id=refinement.plan_id)
            raise

    @abstractmethod
    async def _analyze_query_impl(self, query: PlannerQuery) -> ExecutionPlan:
        """Implementation-specific query analysis."""
        ...

    @abstractmethod
    async def _refine_plan_impl(
        self, refinement: PlanRefinement
    ) -> ExecutionPlan:
        """Implementation-specific plan refinement."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check planner health."""
        return {
            "status": "healthy",
            "module": self.module_name,
            "execution_id": self.state.execution_id,
            "has_error": self.state.error is not None,
        }


# ============================================================================
# Executor Base Class
# ============================================================================


class BaseExecutor(BaseModule):
    """
    Base class for Executor module implementations.

    Provides common execution functionality with logging and error handling.
    """

    def __init__(
        self,
        a2a_context: A2AContext | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__("Executor", a2a_context, logger)

    async def execute_step(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute a single plan step.

        Args:
            context: Execution context

        Returns:
            Execution result

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails
        """
        self._log_operation(
            "execute_step", "started", step_id=context.step.step_id
        )

        try:
            result = await self._execute_step_impl(context)
            self._log_operation(
                "execute_step",
                "completed",
                step_id=result.step_id,
                success=result.success,
                execution_time=result.execution_time,
            )
            return result

        except Exception as e:
            self._log_error(e, "execute_step", step_id=context.step.step_id)
            raise

    async def execute_with_retry(
        self, context: ExecutionContext, policy: RetryPolicy
    ) -> ExecutionResult:
        """
        Execute step with retry policy.

        Args:
            context: Execution context
            policy: Retry policy

        Returns:
            Execution result after retries

        Raises:
            RuntimeError: If all retries fail
        """
        self._log_operation(
            "execute_with_retry",
            "started",
            step_id=context.step.step_id,
            max_attempts=policy.max_attempts,
        )

        try:
            result = await self._execute_with_retry_impl(context, policy)
            self._log_operation(
                "execute_with_retry",
                "completed",
                step_id=result.step_id,
                success=result.success,
            )
            return result

        except Exception as e:
            self._log_error(
                e,
                "execute_with_retry",
                step_id=context.step.step_id,
                max_attempts=policy.max_attempts,
            )
            raise

    async def handle_tool_invocation(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """
        Handle tool invocation.

        Args:
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            Tool invocation result

        Raises:
            ValueError: If tool not found
            RuntimeError: If invocation fails
        """
        self._log_operation(
            "handle_tool_invocation", "started", tool=tool_name
        )

        try:
            result = await self._handle_tool_invocation_impl(
                tool_name, parameters
            )
            self._log_operation(
                "handle_tool_invocation", "completed", tool=tool_name
            )
            return result

        except Exception as e:
            self._log_error(e, "handle_tool_invocation", tool=tool_name)
            raise

    @abstractmethod
    async def _execute_step_impl(
        self, context: ExecutionContext
    ) -> ExecutionResult:
        """Implementation-specific step execution."""
        ...

    @abstractmethod
    async def _execute_with_retry_impl(
        self, context: ExecutionContext, policy: RetryPolicy
    ) -> ExecutionResult:
        """Implementation-specific retry execution."""
        ...

    @abstractmethod
    async def _handle_tool_invocation_impl(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """Implementation-specific tool invocation."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check executor health."""
        return {
            "status": "healthy",
            "module": self.module_name,
            "execution_id": self.state.execution_id,
            "has_error": self.state.error is not None,
        }


# ============================================================================
# Verifier Base Class
# ============================================================================


class BaseVerifier(BaseModule):
    """
    Base class for Verifier module implementations.

    Provides common verification functionality with logging and error handling.
    """

    def __init__(
        self,
        a2a_context: A2AContext | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__("Verifier", a2a_context, logger)

    async def validate_results(
        self, request: VerificationRequest
    ) -> VerificationResult:
        """
        Validate execution results.

        Args:
            request: Verification request

        Returns:
            Verification result

        Raises:
            ValueError: If request invalid
            RuntimeError: If validation fails
        """
        self._log_operation(
            "validate_results", "started", results_count=len(request.results)
        )

        try:
            result = await self._validate_results_impl(request)
            self._log_operation(
                "validate_results",
                "completed",
                valid=result.valid,
                errors_count=len(result.errors),
            )
            return result

        except Exception as e:
            self._log_error(e, "validate_results")
            raise

    async def check_consistency(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Check consistency between results.

        Args:
            check: Consistency check

        Returns:
            Verification result

        Raises:
            ValueError: If check invalid
            RuntimeError: If check fails
        """
        self._log_operation(
            "check_consistency", "started", rule=check.rule
        )

        try:
            result = await self._check_consistency_impl(check)
            self._log_operation(
                "check_consistency", "completed", valid=result.valid
            )
            return result

        except Exception as e:
            self._log_error(e, "check_consistency", rule=check.rule)
            raise

    async def provide_feedback(
        self, results: list[ExecutionResult]
    ) -> str:
        """
        Provide feedback for results.

        Args:
            results: Results to analyze

        Returns:
            Feedback string

        Raises:
            ValueError: If results empty
            RuntimeError: If feedback generation fails
        """
        if not results:
            raise ValueError("Results list cannot be empty")

        self._log_operation(
            "provide_feedback", "started", results_count=len(results)
        )

        try:
            feedback = await self._provide_feedback_impl(results)
            self._log_operation(
                "provide_feedback",
                "completed",
                feedback_length=len(feedback),
            )
            return feedback

        except Exception as e:
            self._log_error(e, "provide_feedback")
            raise

    @abstractmethod
    async def _validate_results_impl(
        self, request: VerificationRequest
    ) -> VerificationResult:
        """Implementation-specific result validation."""
        ...

    @abstractmethod
    async def _check_consistency_impl(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """Implementation-specific consistency check."""
        ...

    @abstractmethod
    async def _provide_feedback_impl(
        self, results: list[ExecutionResult]
    ) -> str:
        """Implementation-specific feedback generation."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check verifier health."""
        return {
            "status": "healthy",
            "module": self.module_name,
            "execution_id": self.state.execution_id,
            "has_error": self.state.error is not None,
        }


# ============================================================================
# Generator Base Class
# ============================================================================


class BaseGenerator(BaseModule):
    """
    Base class for Generator module implementations.

    Provides common generation functionality with logging and error handling.
    """

    def __init__(
        self,
        a2a_context: A2AContext | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__("Generator", a2a_context, logger)

    async def synthesize_response(
        self, request: GenerationRequest
    ) -> GeneratedResponse:
        """
        Synthesize response from results.

        Args:
            request: Generation request

        Returns:
            Generated response

        Raises:
            ValueError: If request invalid
            RuntimeError: If synthesis fails
        """
        self._log_operation(
            "synthesize_response",
            "started",
            results_count=len(request.verified_results),
            format=request.format,
        )

        try:
            response = await self._synthesize_response_impl(request)
            self._log_operation(
                "synthesize_response",
                "completed",
                format=response.format,
                content_length=len(response.content),
            )
            return response

        except Exception as e:
            self._log_error(e, "synthesize_response", format=request.format)
            raise

    async def format_output(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """
        Format content according to spec.

        Args:
            content: Content to format
            format_spec: Format specification

        Returns:
            Formatted content

        Raises:
            ValueError: If format spec invalid
            RuntimeError: If formatting fails
        """
        self._log_operation(
            "format_output", "started", format_type=format_spec.type
        )

        try:
            formatted = await self._format_output_impl(content, format_spec)
            self._log_operation(
                "format_output",
                "completed",
                format_type=format_spec.type,
                output_length=len(formatted),
            )
            return formatted

        except Exception as e:
            self._log_error(e, "format_output", format_type=format_spec.type)
            raise

    async def include_reasoning(
        self, response: GeneratedResponse, reasoning: str
    ) -> GeneratedResponse:
        """
        Add reasoning to response.

        Args:
            response: Generated response
            reasoning: Reasoning to include

        Returns:
            Enhanced response

        Raises:
            ValueError: If inputs invalid
            RuntimeError: If inclusion fails
        """
        self._log_operation("include_reasoning", "started")

        try:
            enhanced = await self._include_reasoning_impl(response, reasoning)
            self._log_operation("include_reasoning", "completed")
            return enhanced

        except Exception as e:
            self._log_error(e, "include_reasoning")
            raise

    @abstractmethod
    async def _synthesize_response_impl(
        self, request: GenerationRequest
    ) -> GeneratedResponse:
        """Implementation-specific response synthesis."""
        ...

    @abstractmethod
    async def _format_output_impl(
        self, content: str, format_spec: OutputFormat
    ) -> str:
        """Implementation-specific output formatting."""
        ...

    @abstractmethod
    async def _include_reasoning_impl(
        self, response: GeneratedResponse, reasoning: str
    ) -> GeneratedResponse:
        """Implementation-specific reasoning inclusion."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check generator health."""
        return {
            "status": "healthy",
            "module": self.module_name,
            "execution_id": self.state.execution_id,
            "has_error": self.state.error is not None,
        }
