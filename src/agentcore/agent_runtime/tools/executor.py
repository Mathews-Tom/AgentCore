"""Tool executor for managing tool invocation lifecycle.

This module implements the ToolExecutor class that manages the complete tool execution
lifecycle including authentication, validation, error handling, logging, and result
formatting. Following specification from docs/specs/tool-integration/spec.md.
"""

import time
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.tool_integration import ToolExecutionStatus, ToolResult
from .base import ExecutionContext, Tool
from .registry import ToolRegistry

logger = structlog.get_logger()


class ToolExecutionError(Exception):
    """Base exception for tool execution errors."""

    pass


class ToolAuthenticationError(ToolExecutionError):
    """Raised when tool authentication fails."""

    pass


class ToolValidationError(ToolExecutionError):
    """Raised when tool parameter validation fails."""

    pass


class ToolTimeoutError(ToolExecutionError):
    """Raised when tool execution exceeds timeout."""

    pass


class ToolExecutor:
    """Tool executor managing tool invocation lifecycle.

    Implements comprehensive tool execution with:
    - Tool lookup via ToolRegistry
    - Parameter validation
    - Authentication handling (basic via ExecutionContext)
    - Error handling and categorization
    - Database logging of executions
    - Distributed tracing support
    - Timeout management

    Implements TOOL-006 requirements from docs/specs/tool-integration/tasks.md.

    Attributes:
        registry: ToolRegistry instance for tool lookup
        db_session: Optional database session for execution logging

    Example:
        ```python
        registry = ToolRegistry()
        registry.register(GoogleSearchTool())

        executor = ToolExecutor(registry, db_session)

        context = ExecutionContext(
            user_id="user123",
            agent_id="agent456",
            trace_id="trace789"
        )

        result = await executor.execute_tool(
            tool_id="google_search",
            parameters={"query": "AgentCore"},
            context=context
        )

        if result.status == ToolExecutionStatus.SUCCESS:
            print(result.result)
        ```
    """

    def __init__(
        self,
        registry: ToolRegistry,
        db_session: AsyncSession | None = None,
    ):
        """Initialize tool executor.

        Args:
            registry: ToolRegistry instance for tool lookup
            db_session: Optional database session for execution logging
        """
        self.registry = registry
        self.db_session = db_session
        self.logger = logger.bind(component="tool_executor")
        self.logger.info("tool_executor_initialized")

    async def execute_tool(
        self,
        tool_id: str,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute a tool with full lifecycle management.

        This is the main entry point for tool execution. It handles:
        1. Tool lookup from registry
        2. Authentication validation
        3. Parameter validation
        4. Tool execution with timeout
        5. Error handling and categorization
        6. Result logging to database
        7. Trace ID propagation

        Implements all TOOL-006 acceptance criteria.

        Args:
            tool_id: Unique identifier of the tool to execute
            parameters: Dictionary of parameter values for the tool
            context: Execution context with user_id, agent_id, trace_id, etc.

        Returns:
            ToolResult containing execution status, result data, errors, and metadata

        Example:
            ```python
            result = await executor.execute_tool(
                tool_id="python_repl",
                parameters={"code": "print('hello')"},
                context=ExecutionContext(user_id="user123")
            )
            ```
        """
        start_time = time.time()
        timestamp = datetime.utcnow()

        self.logger.info(
            "tool_execution_started",
            tool_id=tool_id,
            user_id=context.user_id,
            agent_id=context.agent_id,
            trace_id=context.trace_id,
            request_id=context.request_id,
        )

        try:
            # 1. Tool lookup from registry
            tool = self.registry.get(tool_id)
            if tool is None:
                execution_time_ms = (time.time() - start_time) * 1000
                result = ToolResult(
                    request_id=context.request_id,
                    tool_id=tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=f"Tool '{tool_id}' not found in registry",
                    error_type="ToolNotFoundError",
                    execution_time_ms=execution_time_ms,
                    timestamp=timestamp,
                )
                await self._log_execution(result, context, parameters)
                return result

            # 2. Authentication validation (basic via ExecutionContext)
            auth_error = await self._validate_authentication(tool, context)
            if auth_error:
                execution_time_ms = (time.time() - start_time) * 1000
                result = ToolResult(
                    request_id=context.request_id,
                    tool_id=tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=auth_error,
                    error_type="ToolAuthenticationError",
                    execution_time_ms=execution_time_ms,
                    timestamp=timestamp,
                )
                await self._log_execution(result, context, parameters)
                return result

            # 3. Parameter validation
            is_valid, validation_error = await tool.validate_parameters(parameters)
            if not is_valid:
                execution_time_ms = (time.time() - start_time) * 1000
                result = ToolResult(
                    request_id=context.request_id,
                    tool_id=tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=f"Parameter validation failed: {validation_error}",
                    error_type="ToolValidationError",
                    execution_time_ms=execution_time_ms,
                    timestamp=timestamp,
                )
                await self._log_execution(result, context, parameters)
                return result

            # 4. Tool execution with timeout
            result = await self._execute_with_timeout(
                tool, parameters, context, start_time, timestamp
            )

            # 5. Log execution to database
            await self._log_execution(result, context, parameters)

            # Log success
            if result.status == ToolExecutionStatus.SUCCESS:
                self.logger.info(
                    "tool_execution_completed",
                    tool_id=tool_id,
                    status="success",
                    execution_time_ms=result.execution_time_ms,
                    trace_id=context.trace_id,
                )
            else:
                self.logger.warning(
                    "tool_execution_failed",
                    tool_id=tool_id,
                    status=result.status.value,
                    error=result.error,
                    error_type=result.error_type,
                    trace_id=context.trace_id,
                )

            return result

        except Exception as e:
            # Catch-all for unexpected errors
            execution_time_ms = (time.time() - start_time) * 1000
            result = ToolResult(
                request_id=context.request_id,
                tool_id=tool_id,
                status=ToolExecutionStatus.FAILED,
                error=f"Unexpected error: {str(e)}",
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
            )
            await self._log_execution(result, context, parameters)
            self.logger.error(
                "tool_execution_error",
                tool_id=tool_id,
                error=str(e),
                error_type=type(e).__name__,
                trace_id=context.trace_id,
            )
            return result

    async def _validate_authentication(
        self, tool: Tool, context: ExecutionContext
    ) -> str | None:
        """Validate authentication requirements for tool execution.

        Basic authentication check via ExecutionContext. Currently validates that
        required identity fields (user_id, agent_id) are present when tool requires
        authentication.

        Args:
            tool: Tool instance to validate authentication for
            context: Execution context with authentication info

        Returns:
            Error message if authentication fails, None if successful
        """
        # Basic authentication: check that user_id and agent_id are present
        # for tools that require authentication
        if tool.metadata.auth_method.value != "none":
            if not context.user_id:
                return "Authentication required: user_id missing from context"
            if not context.agent_id:
                return "Authentication required: agent_id missing from context"

        return None

    async def _execute_with_timeout(
        self,
        tool: Tool,
        parameters: dict[str, Any],
        context: ExecutionContext,
        start_time: float,
        timestamp: datetime,
    ) -> ToolResult:
        """Execute tool with timeout management.

        Args:
            tool: Tool instance to execute
            parameters: Validated parameters
            context: Execution context
            start_time: Execution start time for metrics
            timestamp: Execution timestamp

        Returns:
            ToolResult from tool execution or timeout error
        """
        import asyncio

        timeout = tool.metadata.timeout_seconds

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(parameters, context),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult(
                request_id=context.request_id,
                tool_id=tool.metadata.tool_id,
                status=ToolExecutionStatus.TIMEOUT,
                error=f"Tool execution exceeded timeout of {timeout} seconds",
                error_type="ToolTimeoutError",
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
            )

    async def _log_execution(
        self,
        result: ToolResult,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> None:
        """Log tool execution to database.

        Writes execution record to tool_executions table with all metadata
        for observability and auditing.

        Args:
            result: ToolResult from execution
            context: Execution context with user/agent/trace info
            parameters: Tool parameters that were executed
        """
        if self.db_session is None:
            return

        try:
            # Determine success boolean from status
            success = result.status == ToolExecutionStatus.SUCCESS

            # Insert execution record
            await self.db_session.execute(
                text("""
                    INSERT INTO tool_executions (
                        request_id, tool_id, agent_id, user_id, trace_id,
                        status, result, error, error_type, execution_time_ms,
                        parameters, execution_context, execution_metadata,
                        timestamp, created_at, retry_count, success
                    )
                    VALUES (
                        :request_id, :tool_id, :agent_id, :user_id, :trace_id,
                        :status::toolexecutionstatus, :result::jsonb, :error, :error_type,
                        :execution_time_ms, :parameters::jsonb, :execution_context::jsonb,
                        :execution_metadata::jsonb, :timestamp, :created_at, :retry_count,
                        :success
                    )
                """),
                {
                    "request_id": result.request_id,
                    "tool_id": result.tool_id,
                    "agent_id": context.agent_id,
                    "user_id": context.user_id,
                    "trace_id": context.trace_id,
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "error_type": result.error_type,
                    "execution_time_ms": result.execution_time_ms,
                    "parameters": parameters,
                    "execution_context": context.to_dict(),
                    "execution_metadata": result.metadata or {},
                    "timestamp": result.timestamp,
                    "created_at": datetime.utcnow(),
                    "retry_count": result.retry_count or 0,
                    "success": success,
                },
            )
            await self.db_session.commit()

            self.logger.debug(
                "tool_execution_logged",
                request_id=result.request_id,
                tool_id=result.tool_id,
            )

        except Exception as e:
            self.logger.error(
                "tool_execution_logging_failed",
                error=str(e),
                request_id=result.request_id,
                tool_id=result.tool_id,
            )
            # Don't fail the execution if logging fails
            await self.db_session.rollback()
