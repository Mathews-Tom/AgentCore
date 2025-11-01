"""Tool execution service with lifecycle management and observability."""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog

from ..models.tool_integration import (
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionStatus,
    ToolResult,
)
from .tool_registry import ToolNotFoundError, ToolRegistry

logger = structlog.get_logger()


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""


class ToolAuthenticationError(Exception):
    """Raised when tool authentication fails."""


class ToolTimeoutError(Exception):
    """Raised when tool execution times out."""


class ToolValidationError(Exception):
    """Raised when tool parameter validation fails."""


class ToolExecutor:
    """
    Tool execution service with comprehensive lifecycle management.

    Separates execution concerns from registry:
    - Lifecycle hooks (before, after, on_error)
    - Authentication and authorization
    - Timeout and retry management
    - Comprehensive observability
    - Resource tracking
    """

    def __init__(
        self,
        registry: ToolRegistry,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize tool executor.

        Args:
            registry: Tool registry for tool definitions and executors
            enable_metrics: Enable metrics collection
        """
        self._registry = registry
        self._enable_metrics = enable_metrics
        self._before_hooks: list[Callable] = []
        self._after_hooks: list[Callable] = []
        self._error_hooks: list[Callable] = []

        logger.info(
            "tool_executor_initialized",
            enable_metrics=enable_metrics,
        )

    def add_before_hook(self, hook: Callable[[ToolExecutionRequest], None]) -> None:
        """
        Add a hook to run before tool execution.

        Args:
            hook: Function to call before execution
        """
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable[[ToolResult], None]) -> None:
        """
        Add a hook to run after tool execution.

        Args:
            hook: Function to call after execution
        """
        self._after_hooks.append(hook)

    def add_error_hook(self, hook: Callable[[ToolExecutionRequest, Exception], None]) -> None:
        """
        Add a hook to run when tool execution fails.

        Args:
            hook: Function to call on error
        """
        self._error_hooks.append(hook)

    async def execute(self, request: ToolExecutionRequest) -> ToolResult:
        """
        Execute a tool with full lifecycle management.

        Args:
            request: Tool execution request

        Returns:
            ToolResult with execution metadata
        """
        start_time = time.time()
        timestamp = datetime.now()

        try:
            # Get tool definition
            tool = self._registry.get_tool(request.tool_id)
            if not tool:
                raise ToolNotFoundError(f"Tool '{request.tool_id}' not found")

            # Run before hooks
            await self._run_before_hooks(request)

            # Validate authentication
            await self._validate_authentication(tool, request)

            # Validate parameters
            self._validate_parameters(tool, request.parameters)

            # Execute with timeout and retry
            result_data = await self._execute_with_retry(tool, request)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Create successful result
            result = ToolResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result=result_data,
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
                retry_count=0,
            )

            # Run after hooks
            await self._run_after_hooks(result)

            logger.info(
                "tool_execution_success",
                tool_id=request.tool_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                execution_time_ms=execution_time_ms,
            )

            return result

        except ToolTimeoutError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            result = ToolResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolExecutionStatus.TIMEOUT,
                error=str(e),
                error_type="ToolTimeoutError",
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
            )
            await self._run_error_hooks(request, e)
            logger.error(
                "tool_execution_timeout",
                tool_id=request.tool_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                error=str(e),
            )
            return result

        except (ToolNotFoundError, ToolAuthenticationError, ToolValidationError) as e:
            execution_time_ms = (time.time() - start_time) * 1000
            result = ToolResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
            )
            await self._run_error_hooks(request, e)
            logger.error(
                "tool_execution_failed",
                tool_id=request.tool_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            result = ToolResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
            )
            await self._run_error_hooks(request, e)
            logger.exception(
                "tool_execution_error",
                tool_id=request.tool_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
            )
            return result

    async def _run_before_hooks(self, request: ToolExecutionRequest) -> None:
        """Run all before hooks."""
        for hook in self._before_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(request)
                else:
                    hook(request)
            except Exception as e:
                logger.warning(
                    "before_hook_failed",
                    hook=hook.__name__,
                    error=str(e),
                )

    async def _run_after_hooks(self, result: ToolResult) -> None:
        """Run all after hooks."""
        for hook in self._after_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(result)
                else:
                    hook(result)
            except Exception as e:
                logger.warning(
                    "after_hook_failed",
                    hook=hook.__name__,
                    error=str(e),
                )

    async def _run_error_hooks(self, request: ToolExecutionRequest, error: Exception) -> None:
        """Run all error hooks."""
        for hook in self._error_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(request, error)
                else:
                    hook(request, error)
            except Exception as e:
                logger.warning(
                    "error_hook_failed",
                    hook=hook.__name__,
                    error=str(e),
                )

    async def _validate_authentication(
        self,
        tool: ToolDefinition,
        request: ToolExecutionRequest,
    ) -> None:
        """
        Validate authentication for tool execution.

        Args:
            tool: Tool definition
            request: Execution request

        Raises:
            ToolAuthenticationError: If authentication fails
        """
        # TODO: Implement authentication validation based on tool.auth_method
        # For now, just check if auth is required
        if tool.auth_method.value != "none":
            # Check if credentials are provided in execution context
            if "auth_token" not in request.execution_context:
                raise ToolAuthenticationError(
                    f"Tool '{tool.tool_id}' requires authentication but no token provided"
                )

    def _validate_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> None:
        """
        Validate tool parameters against definition.

        Args:
            tool: Tool definition
            parameters: Provided parameters

        Raises:
            ToolValidationError: If parameters are invalid
        """
        # Check required parameters
        for param_name, param_def in tool.parameters.items():
            if param_def.required and param_name not in parameters:
                raise ToolValidationError(
                    f"Required parameter '{param_name}' missing for tool '{tool.tool_id}'"
                )

        # Validate parameter types and constraints
        for param_name, value in parameters.items():
            if param_name not in tool.parameters:
                # Allow extra parameters (lenient validation)
                continue

            param_def = tool.parameters[param_name]

            # Type validation
            if param_def.type == "string":
                if not isinstance(value, str):
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be a string"
                    )
                if param_def.min_length and len(value) < param_def.min_length:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be at least {param_def.min_length} characters"
                    )
                if param_def.max_length and len(value) > param_def.max_length:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be at most {param_def.max_length} characters"
                    )

            elif param_def.type == "number":
                if not isinstance(value, (int, float)):
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be a number"
                    )
                if param_def.min_value is not None and value < param_def.min_value:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be at least {param_def.min_value}"
                    )
                if param_def.max_value is not None and value > param_def.max_value:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be at most {param_def.max_value}"
                    )

            elif param_def.type == "boolean":
                if not isinstance(value, bool):
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be a boolean"
                    )

            # Enum validation
            if param_def.enum and value not in param_def.enum:
                raise ToolValidationError(
                    f"Parameter '{param_name}' must be one of {param_def.enum}"
                )

    async def _execute_with_retry(
        self,
        tool: ToolDefinition,
        request: ToolExecutionRequest,
    ) -> Any:
        """
        Execute tool with timeout and retry logic.

        Args:
            tool: Tool definition
            request: Execution request

        Returns:
            Execution result

        Raises:
            ToolTimeoutError: If execution times out
            Exception: If execution fails after all retries
        """
        # Get executor from registry
        executor = self._registry._executors.get(tool.tool_id)
        if not executor:
            raise ToolExecutionError(
                f"No executor found for tool '{tool.tool_id}'"
            )

        # Determine timeout
        timeout = (
            request.timeout_override
            if request.timeout_override is not None
            else tool.timeout_seconds
        )

        # Determine max retries
        max_retries = (
            request.retry_override
            if request.retry_override is not None
            else tool.max_retries
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(executor):
                    result = await asyncio.wait_for(
                        executor(**request.parameters),
                        timeout=timeout,
                    )
                else:
                    result = executor(**request.parameters)

                return result

            except asyncio.TimeoutError:
                raise ToolTimeoutError(
                    f"Tool '{tool.tool_id}' execution timed out after {timeout}s"
                )

            except Exception as e:
                last_error = e
                if not tool.is_retryable or attempt >= max_retries:
                    raise

                # Log retry
                logger.warning(
                    "tool_execution_retry",
                    tool_id=tool.tool_id,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )

                # Wait before retry (exponential backoff)
                await asyncio.sleep(2**attempt)

        # Should never reach here, but just in case
        if last_error:
            raise last_error
        raise ToolExecutionError("Execution failed after all retries")


# Global tool executor instance
_global_executor: ToolExecutor | None = None


def get_tool_executor(registry: ToolRegistry | None = None) -> ToolExecutor:
    """
    Get global tool executor instance.

    Args:
        registry: Tool registry (uses global if not provided)

    Returns:
        ToolExecutor instance
    """
    global _global_executor

    if _global_executor is None:
        from .tool_registry import get_tool_registry

        reg = registry or get_tool_registry()
        _global_executor = ToolExecutor(reg)

    return _global_executor
