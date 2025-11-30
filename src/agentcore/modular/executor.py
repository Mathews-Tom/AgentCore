"""
Executor Module Implementation

Implements plan step execution with tool invocation for the modular agent core.
Provides tool orchestration, parameter formatting, execution monitoring, and parallel execution
where dependencies allow.

Features:
- Configurable retry strategies with exponential backoff and jitter
- Circuit breaker pattern for fault tolerance
- Error categorization (retryable vs non-retryable)
- Graceful degradation when tools unavailable
- Structured error reporting and recovery

This module follows the PEVG (Planner, Executor, Verifier, Generator) architecture pattern.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.agent_runtime.models.error_types import CircuitBreakerConfig
from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionStatus,
    ToolResult,
)
from agentcore.agent_runtime.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)
from agentcore.agent_runtime.tools.base import ExecutionContext as ToolExecutionContext
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.modular.base import BaseExecutor
from agentcore.modular.interfaces import (
    ExecutionContext,
    ExecutionResult,
    RetryPolicy,
)
from agentcore.modular.models import (
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    StepStatus,
)

logger = structlog.get_logger()


class ErrorCategory(str, Enum):
    """Error categories for retry classification."""

    TRANSIENT = "transient"  # Temporary errors that can be retried
    PERMANENT = "permanent"  # Permanent errors that should not be retried
    TIMEOUT = "timeout"  # Timeout errors
    VALIDATION = "validation"  # Validation errors (non-retryable)
    TOOL_NOT_FOUND = "tool_not_found"  # Tool not found (non-retryable)
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker open (non-retryable)


class ExecutorModule(BaseExecutor):
    """
    Concrete Executor module implementation with retry and error recovery.

    Executes plan steps by invoking tools from the Tool Integration Framework.
    Provides:
    1. Tool parameter formatting and validation
    2. Execution monitoring with timeout handling
    3. Result collection and formatting
    4. Support for parallel tool execution where dependencies allow
    5. Integration with Tool Registry and ToolExecutor
    6. Configurable retry strategies with exponential backoff
    7. Circuit breaker pattern for failing tools
    8. Error categorization and graceful degradation
    9. Structured error reporting

    Integrates with Tool Integration Framework (TOOL-001).
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        a2a_context: A2AContext | None = None,
        logger_instance: Any | None = None,
        max_parallel_steps: int = 5,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        default_retry_policy: RetryPolicy | None = None,
    ) -> None:
        """
        Initialize Executor module.

        Args:
            tool_registry: Tool registry for tool lookup
            tool_executor: Tool executor for tool invocation
            a2a_context: A2A protocol context for tracing
            logger_instance: Structured logger instance
            max_parallel_steps: Maximum number of steps to execute in parallel
            enable_circuit_breaker: Enable circuit breaker for tool fault tolerance
            circuit_breaker_config: Configuration for circuit breakers
            default_retry_policy: Default retry policy for failed executions
        """
        super().__init__(a2a_context, logger_instance)
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.max_parallel_steps = max_parallel_steps
        self.enable_circuit_breaker = enable_circuit_breaker

        # Circuit breaker configuration and instances
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Default retry policy
        self.default_retry_policy = default_retry_policy or RetryPolicy(
            max_attempts=3,
            backoff_seconds=1.0,
            exponential=True,
        )

        self.logger.info(
            "executor_initialized",
            max_parallel_steps=max_parallel_steps,
            registry_tool_count=len(tool_registry),
            circuit_breaker_enabled=enable_circuit_breaker,
            default_max_retries=self.default_retry_policy.max_attempts,
        )

    def _get_circuit_breaker(self, tool_id: str) -> CircuitBreaker:
        """
        Get or create circuit breaker for a tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Circuit breaker instance for the tool
        """
        if tool_id not in self._circuit_breakers:
            self._circuit_breakers[tool_id] = CircuitBreaker(
                name=f"tool_{tool_id}",
                config=self.circuit_breaker_config,
            )
            self.logger.debug(
                "circuit_breaker_created",
                tool_id=tool_id,
                config=self.circuit_breaker_config.model_dump(),
            )
        return self._circuit_breakers[tool_id]

    def _categorize_error(
        self,
        error: Exception,
        error_type: str | None = None,
    ) -> ErrorCategory:
        """
        Categorize error to determine if it's retryable.

        Args:
            error: Exception that occurred
            error_type: Error type from metadata (if available)

        Returns:
            Error category classification
        """
        # Circuit breaker errors are non-retryable
        if isinstance(error, CircuitBreakerError):
            return ErrorCategory.CIRCUIT_OPEN

        # Timeout errors are retryable
        if isinstance(error, asyncio.TimeoutError) or error_type == "TimeoutError":
            return ErrorCategory.TIMEOUT

        # Check error type from metadata
        if error_type:
            if error_type == "ToolNotFoundError":
                return ErrorCategory.TOOL_NOT_FOUND
            elif error_type == "ParameterValidationError":
                return ErrorCategory.VALIDATION

        # Check exception type
        exception_name = type(error).__name__

        # Non-retryable errors
        if exception_name in {
            "ValueError",
            "TypeError",
            "KeyError",
            "AttributeError",
        }:
            return ErrorCategory.PERMANENT

        # Retryable errors (network, temporary failures)
        if exception_name in {
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "RuntimeError",
        }:
            return ErrorCategory.TRANSIENT

        # Default to transient (optimistic retry)
        return ErrorCategory.TRANSIENT

    def _is_retryable(self, error_category: ErrorCategory) -> bool:
        """
        Determine if an error category should be retried.

        Args:
            error_category: Error category

        Returns:
            True if error should be retried
        """
        non_retryable = {
            ErrorCategory.PERMANENT,
            ErrorCategory.VALIDATION,
            ErrorCategory.TOOL_NOT_FOUND,
            ErrorCategory.CIRCUIT_OPEN,
        }
        return error_category not in non_retryable

    def _calculate_backoff(
        self,
        attempt: int,
        base_delay: float,
        exponential: bool,
        max_delay: float = 60.0,
    ) -> float:
        """
        Calculate backoff delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay in seconds
            exponential: Use exponential backoff
            max_delay: Maximum delay cap

        Returns:
            Delay in seconds with jitter applied
        """
        if exponential:
            # Exponential backoff: base_delay * 2^attempt
            delay = base_delay * (2**attempt)
        else:
            # Linear backoff: base_delay * (attempt + 1)
            delay = base_delay * (attempt + 1)

        # Cap at max_delay
        delay = min(delay, max_delay)

        # Add jitter (±25% random variation)
        jitter_factor = random.uniform(0.75, 1.25)
        delay = delay * jitter_factor

        # Ensure positive delay
        return max(0.1, delay)

    async def _execute_step_impl(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute a single plan step using tool invocation.

        Implementation process:
        1. Extract action and parameters from step
        2. Validate tool exists in registry
        3. Format parameters for tool execution
        4. Create tool execution context with A2A tracing
        5. Invoke tool via ToolExecutor
        6. Collect and format results
        7. Handle errors and timeouts

        Args:
            context: Execution context with step and previous results

        Returns:
            Execution result with success status and data

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails
        """
        # Convert basic PlanStep to step data
        step = context.step
        step_id = step.step_id
        action = step.action
        parameters = dict(step.parameters) if hasattr(step, 'parameters') else {}

        start_time = time.time()

        self.logger.info(
            "executing_step",
            step_id=step_id,
            action=action,
            timeout=context.timeout_seconds,
        )

        try:
            # Extract tool name from action
            tool_id = self._extract_tool_id(action)

            # Validate tool exists
            tool = self.tool_registry.get(tool_id)
            if not tool:
                error_msg = f"Tool '{tool_id}' not found in registry"
                execution_time = time.time() - start_time
                return ExecutionResult(
                    step_id=step_id,
                    success=False,
                    result=None,
                    error=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "error_type": "ToolNotFoundError",
                        "error_category": ErrorCategory.TOOL_NOT_FOUND.value,
                        "retryable": False,
                        "action": action,
                    },
                )

            # Format parameters using previous results
            formatted_params = self._format_parameters(
                parameters, context.previous_results
            )

            # Validate parameters
            is_valid, validation_error = await tool.validate_parameters(formatted_params)
            if not is_valid:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    step_id=step_id,
                    success=False,
                    result=None,
                    error=f"Parameter validation failed: {validation_error}",
                    execution_time=execution_time,
                    metadata={
                        "error_type": "ParameterValidationError",
                        "error_category": ErrorCategory.VALIDATION.value,
                        "retryable": False,
                        "validation_error": validation_error,
                    },
                )

            # Create tool execution context with A2A tracing
            tool_context = ToolExecutionContext(
                user_id=self.a2a_context.source_agent,
                agent_id=self.a2a_context.target_agent or "executor-module",
                trace_id=self.a2a_context.trace_id,
                session_id=self.a2a_context.session_id,
                request_id=step_id,
                metadata={
                    "step_id": step_id,
                    "action": action,
                    "conversation_id": self.a2a_context.conversation_id,
                },
            )

            # Execute tool with timeout and circuit breaker protection
            try:
                # Wrap tool execution with circuit breaker if enabled
                if self.enable_circuit_breaker:
                    circuit_breaker = self._get_circuit_breaker(tool_id)

                    async def execute_tool_fn() -> ToolResult:
                        return await self.tool_executor.execute_tool(
                            tool_id=tool_id,
                            parameters=formatted_params,
                            context=tool_context,
                        )

                    tool_result = await asyncio.wait_for(
                        circuit_breaker.call(execute_tool_fn),
                        timeout=context.timeout_seconds,
                    )
                else:
                    tool_result = await asyncio.wait_for(
                        self.tool_executor.execute_tool(
                            tool_id=tool_id,
                            parameters=formatted_params,
                            context=tool_context,
                        ),
                        timeout=context.timeout_seconds,
                    )

            except CircuitBreakerError as e:
                execution_time = time.time() - start_time
                error_msg = f"Circuit breaker open for tool '{tool_id}': {str(e)}"
                self.logger.warning(
                    "circuit_breaker_blocked_execution",
                    tool_id=tool_id,
                    step_id=step_id,
                    circuit_state=circuit_breaker.state.value if self.enable_circuit_breaker else "disabled",
                )
                return ExecutionResult(
                    step_id=step_id,
                    success=False,
                    result=None,
                    error=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "error_type": "CircuitBreakerError",
                        "error_category": ErrorCategory.CIRCUIT_OPEN.value,
                        "retryable": False,
                        "tool_id": tool_id,
                        "circuit_state": circuit_breaker.state.value if self.enable_circuit_breaker else "disabled",
                    },
                )

            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                error_msg = f"Tool execution exceeded timeout of {context.timeout_seconds}s"
                return ExecutionResult(
                    step_id=step_id,
                    success=False,
                    result=None,
                    error=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "error_type": "TimeoutError",
                        "error_category": ErrorCategory.TIMEOUT.value,
                        "retryable": True,
                        "timeout_seconds": context.timeout_seconds,
                    },
                )

            # Convert ToolResult to ExecutionResult
            execution_time = time.time() - start_time
            success = tool_result.status == ToolExecutionStatus.SUCCESS

            # Categorize error if execution failed
            error_category = None
            retryable = True
            if not success:
                error_category = self._categorize_error(
                    Exception(tool_result.error) if tool_result.error else Exception("Unknown error"),
                    tool_result.error_type,
                )
                retryable = self._is_retryable(error_category)

            # Build metadata with error categorization
            metadata = {
                "tool_id": tool_id,
                "tool_execution_time_ms": tool_result.execution_time_ms,
                "tool_status": tool_result.status.value,
                "error_type": tool_result.error_type,
                "retry_count": tool_result.retry_count,
            }

            if error_category:
                metadata["error_category"] = error_category.value
                metadata["retryable"] = retryable

            if self.enable_circuit_breaker:
                circuit_breaker = self._get_circuit_breaker(tool_id)
                metadata["circuit_breaker_state"] = circuit_breaker.state.value
                metadata["circuit_breaker_failures"] = circuit_breaker.failure_count

            result = ExecutionResult(
                step_id=step_id,
                success=success,
                result=tool_result.result if success else None,
                error=tool_result.error,
                execution_time=execution_time,
                metadata=metadata,
            )

            self.logger.info(
                "step_executed",
                step_id=step_id,
                success=success,
                execution_time=execution_time,
                tool_id=tool_id,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error during step execution: {str(e)}"
            self.logger.error(
                "step_execution_error",
                step_id=step_id,
                error=error_msg,
                error_type=type(e).__name__,
            )
            return ExecutionResult(
                step_id=step_id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "error_type": type(e).__name__,
                    "exception": str(e),
                },
            )

    async def _execute_with_retry_impl(
        self, context: ExecutionContext, policy: RetryPolicy
    ) -> ExecutionResult:
        """
        Execute step with intelligent retry policy.

        Enhanced Implementation:
        1. Execute step with timeout
        2. On failure, categorize error (retryable vs non-retryable)
        3. Skip retry for non-retryable errors (validation, tool not found, circuit open)
        4. Apply exponential backoff with jitter for retryable errors
        5. Track retry attempts in metadata
        6. Return last result (success or final failure)

        Args:
            context: Execution context
            policy: Retry policy configuration

        Returns:
            Execution result after retries

        Raises:
            RuntimeError: If all retry attempts fail for retryable errors
        """
        self.logger.info(
            "executing_with_retry",
            step_id=context.step.step_id,
            max_attempts=policy.max_attempts,
            backoff=policy.backoff_seconds,
            exponential=policy.exponential,
        )

        last_result: ExecutionResult | None = None
        retry_count = 0

        for attempt in range(1, policy.max_attempts + 1):
            self.logger.debug(
                "retry_attempt",
                step_id=context.step.step_id,
                attempt=attempt,
                max_attempts=policy.max_attempts,
            )

            result = await self._execute_step_impl(context)
            last_result = result

            # Add retry count to metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata["retry_attempt"] = attempt
            result.metadata["max_retry_attempts"] = policy.max_attempts

            if result.success:
                self.logger.info(
                    "retry_succeeded",
                    step_id=context.step.step_id,
                    attempt=attempt,
                    retry_count=retry_count,
                )
                return result

            # Check if error is retryable
            error_category_str = result.metadata.get("error_category")
            retryable = result.metadata.get("retryable", True)

            if not retryable:
                self.logger.warning(
                    "non_retryable_error",
                    step_id=context.step.step_id,
                    error_category=error_category_str,
                    error=result.error,
                    attempt=attempt,
                )
                # Return immediately for non-retryable errors
                raise RuntimeError(
                    f"Step {context.step.step_id} failed with non-retryable error "
                    f"({error_category_str}): {result.error}"
                )

            retry_count += 1

            # If not last attempt, wait with backoff
            if attempt < policy.max_attempts:
                backoff_delay = self._calculate_backoff(
                    attempt=attempt - 1,
                    base_delay=policy.backoff_seconds,
                    exponential=policy.exponential,
                    max_delay=60.0,
                )

                self.logger.warning(
                    "retry_failed_waiting",
                    step_id=context.step.step_id,
                    attempt=attempt,
                    backoff_seconds=backoff_delay,
                    error=result.error,
                    error_category=error_category_str,
                    retryable=retryable,
                )
                await asyncio.sleep(backoff_delay)

        # All retry attempts exhausted
        if last_result:
            self.logger.error(
                "all_retries_exhausted",
                step_id=context.step.step_id,
                max_attempts=policy.max_attempts,
                total_retries=retry_count,
                final_error=last_result.error,
                error_category=last_result.metadata.get("error_category"),
            )
            raise RuntimeError(
                f"Step {context.step.step_id} failed after {policy.max_attempts} attempts: {last_result.error}"
            )
        else:
            raise RuntimeError(
                f"Step {context.step.step_id} failed with no result"
            )

    async def _handle_tool_invocation_impl(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """
        Handle invocation of a specific tool.

        Direct tool invocation without plan step wrapping.
        Useful for ad-hoc tool calls.

        Args:
            tool_name: Name of tool to invoke
            parameters: Tool parameters

        Returns:
            Tool invocation result

        Raises:
            ValueError: If tool not found or parameters invalid
            RuntimeError: If tool invocation fails
        """
        self.logger.info(
            "handling_tool_invocation",
            tool=tool_name,
            param_count=len(parameters),
        )

        # Validate tool exists
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        # Create execution context
        tool_context = ToolExecutionContext(
            user_id=self.a2a_context.source_agent,
            agent_id=self.a2a_context.target_agent or "executor-module",
            trace_id=self.a2a_context.trace_id,
            session_id=self.a2a_context.session_id,
            metadata={"tool_name": tool_name},
        )

        # Execute tool
        tool_result = await self.tool_executor.execute_tool(
            tool_id=tool_name,
            parameters=parameters,
            context=tool_context,
        )

        if tool_result.status != ToolExecutionStatus.SUCCESS:
            raise RuntimeError(
                f"Tool invocation failed: {tool_result.error}"
            )

        return tool_result.result

    async def execute_plan(
        self, plan: EnhancedExecutionPlan
    ) -> list[ExecutionResult]:
        """
        Execute a complete execution plan.

        Orchestrates execution of all plan steps:
        1. Build dependency graph from steps
        2. Execute steps in topological order
        3. Enable parallel execution where dependencies allow
        4. Collect results from all steps
        5. Handle step failures and propagation

        Args:
            plan: Enhanced execution plan with steps and dependencies

        Returns:
            List of execution results for all steps

        Raises:
            RuntimeError: If plan execution fails
        """
        self.logger.info(
            "executing_plan",
            plan_id=plan.plan_id,
            step_count=len(plan.steps),
            max_parallel=self.max_parallel_steps,
        )

        results: dict[str, ExecutionResult] = {}
        executed_steps: set[str] = set()

        try:
            # Execute steps in dependency order
            while len(executed_steps) < len(plan.steps):
                # Find steps ready to execute (dependencies satisfied)
                ready_steps = self._get_ready_steps(
                    plan.steps, executed_steps, results
                )

                if not ready_steps:
                    # No steps ready - check for circular dependencies
                    remaining = [
                        s.step_id
                        for s in plan.steps
                        if s.step_id not in executed_steps
                    ]
                    if remaining:
                        raise RuntimeError(
                            f"Circular dependency detected. Remaining steps: {remaining}"
                        )
                    break

                # Execute ready steps in parallel (up to max_parallel_steps)
                step_results = await self._execute_parallel_steps(
                    ready_steps, results
                )

                # Update results and executed set
                for step_id, result in step_results.items():
                    results[step_id] = result
                    executed_steps.add(step_id)

                    # Check for failures
                    if not result.success:
                        self.logger.warning(
                            "step_failed_in_plan",
                            plan_id=plan.plan_id,
                            step_id=step_id,
                            error=result.error,
                        )

            self.logger.info(
                "plan_executed",
                plan_id=plan.plan_id,
                total_steps=len(plan.steps),
                executed_steps=len(executed_steps),
                successful_steps=sum(
                    1 for r in results.values() if r.success
                ),
            )

            # Return results in original step order
            return [results[step.step_id] for step in plan.steps]

        except Exception as e:
            self.logger.error(
                "plan_execution_failed",
                plan_id=plan.plan_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Plan execution failed: {str(e)}") from e

    def _extract_tool_id(self, action: str) -> str:
        """
        Extract tool ID from action string.

        Supports formats:
        - "tool_name" (direct tool name)
        - "tool:tool_name" (prefixed format)
        - "invoke_tool_name" (verb format)

        Args:
            action: Action string from plan step

        Returns:
            Extracted tool ID
        """
        # Remove common prefixes
        action = action.lower().strip()

        if action.startswith("tool:"):
            return action[5:]
        elif action.startswith("invoke_"):
            return action[7:]
        elif action.startswith("call_"):
            return action[5:]
        else:
            return action

    def _format_parameters(
        self, parameters: dict[str, Any], previous_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Format parameters using previous step results.

        Supports parameter value substitution:
        - "${step_id}" -> result from step_id
        - "${step_id.field}" -> specific field from step_id result

        Args:
            parameters: Raw parameters from plan step
            previous_results: Results from previous steps

        Returns:
            Formatted parameters with substitutions applied
        """
        formatted = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Parameter reference: ${step_id} or ${step_id.field}
                ref = value[2:-1]  # Remove ${ and }

                if "." in ref:
                    # Field reference: ${step_id.field}
                    step_id, field = ref.split(".", 1)
                    if step_id in previous_results:
                        result_value = previous_results[step_id]
                        if isinstance(result_value, dict) and field in result_value:
                            formatted[key] = result_value[field]
                        else:
                            formatted[key] = value  # Keep original if not found
                    else:
                        formatted[key] = value
                else:
                    # Step reference: ${step_id}
                    if ref in previous_results:
                        formatted[key] = previous_results[ref]
                    else:
                        formatted[key] = value
            else:
                formatted[key] = value

        return formatted

    def _get_ready_steps(
        self,
        steps: list[EnhancedPlanStep],
        executed: set[str],
        results: dict[str, ExecutionResult],
    ) -> list[EnhancedPlanStep]:
        """
        Get steps that are ready to execute (dependencies satisfied).

        Args:
            steps: All plan steps
            executed: Set of already executed step IDs
            results: Execution results so far

        Returns:
            List of steps ready to execute
        """
        ready = []

        for step in steps:
            if step.step_id in executed:
                continue

            # Check if all dependencies are satisfied
            deps_satisfied = True
            for dep in step.dependencies:
                dep_step_id = dep.step_id

                # Check if dependency executed
                if dep_step_id not in executed:
                    deps_satisfied = False
                    break

                # Check if dependency succeeded (if required)
                if dep.required:
                    dep_result = results.get(dep_step_id)
                    if not dep_result or not dep_result.success:
                        deps_satisfied = False
                        break

            if deps_satisfied:
                ready.append(step)

        return ready

    async def _execute_parallel_steps(
        self,
        steps: list[EnhancedPlanStep],
        previous_results: dict[str, ExecutionResult],
    ) -> dict[str, ExecutionResult]:
        """
        Execute multiple steps in parallel.

        Args:
            steps: Steps to execute
            previous_results: Results from previous steps for parameter substitution

        Returns:
            Dictionary mapping step_id to ExecutionResult
        """
        # Limit parallelism
        steps_to_execute = steps[: self.max_parallel_steps]

        self.logger.info(
            "executing_parallel_steps",
            step_count=len(steps_to_execute),
            max_parallel=self.max_parallel_steps,
        )

        # Create execution contexts - convert EnhancedPlanStep to basic PlanStep
        from agentcore.modular.interfaces import PlanStep

        contexts = [
            ExecutionContext(
                step=PlanStep(
                    step_id=step.step_id,
                    action=step.action,
                    parameters=step.parameters,
                    dependencies=[dep.step_id for dep in step.dependencies],
                    estimated_cost=step.estimated_cost,
                ),
                previous_results={
                    step_id: result.result
                    for step_id, result in previous_results.items()
                },
                timeout_seconds=30.0,  # Default timeout
            )
            for step in steps_to_execute
        ]

        # Execute in parallel
        results_list = await asyncio.gather(
            *[self._execute_step_impl(ctx) for ctx in contexts],
            return_exceptions=True,
        )

        # Build results dictionary
        results: dict[str, ExecutionResult] = {}
        for step, result_item in zip(steps_to_execute, results_list):
            if isinstance(result_item, Exception):
                # Convert exception to ExecutionResult
                results[step.step_id] = ExecutionResult(
                    step_id=step.step_id,
                    success=False,
                    result=None,
                    error=str(result_item),
                    execution_time=0.0,
                    metadata={"error_type": type(result_item).__name__},
                )
            else:
                # Type narrowing: result_item is ExecutionResult
                results[step.step_id] = result_item

        return results

    async def health_check(self) -> dict[str, Any]:
        """
        Check executor health and readiness.

        Returns:
            Health status dict with executor-specific metrics including circuit breaker states
        """
        # Collect circuit breaker statistics
        circuit_breaker_stats = {}
        for tool_id, breaker in self._circuit_breakers.items():
            circuit_breaker_stats[tool_id] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
            }

        return {
            "status": "healthy",
            "module": self.module_name,
            "execution_id": self.state.execution_id,
            "has_error": self.state.error is not None,
            "tool_registry_size": len(self.tool_registry),
            "max_parallel_steps": self.max_parallel_steps,
            "circuit_breaker_enabled": self.enable_circuit_breaker,
            "circuit_breakers": circuit_breaker_stats,
            "default_max_retries": self.default_retry_policy.max_attempts,
        }
