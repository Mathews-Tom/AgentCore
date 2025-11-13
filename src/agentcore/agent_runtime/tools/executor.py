"""Tool executor for managing tool invocation lifecycle.

This module implements the ToolExecutor class that manages the complete tool execution
lifecycle including authentication, validation, error handling, logging, and result
formatting. Following specification from docs/specs/tool-integration/spec.md.

Enhanced with:
- Rate limiting using Redis-based sliding window algorithm
- Retry logic with exponential backoff and jitter
- Lifecycle hooks (before/after/error) for observability
- OpenTelemetry distributed tracing for observability (TOOL-020)
"""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.tool_integration import ToolExecutionStatus, ToolResult
from ..monitoring.tracing import (
    add_span_attributes,
    add_span_event,
    get_tracer,
    record_exception,
)
from ..services.rate_limiter import RateLimitExceeded, RateLimiter
from ..services.quota_manager import QuotaExceeded, QuotaManager
from ..services.retry_handler import BackoffStrategy, RetryHandler
from .base import ExecutionContext, Tool
from .errors import categorize_error, get_error_metadata
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
    - Rate limiting (Redis-based sliding window)
    - Quota management (daily/monthly limits)
    - Retry logic with exponential backoff
    - Lifecycle hooks (before/after/error)

    Implements TOOL-006 requirements from docs/specs/tool-integration/tasks.md.

    Attributes:
        registry: ToolRegistry instance for tool lookup
        db_session: Optional database session for execution logging
        rate_limiter: Optional rate limiter for tool execution
        quota_manager: Optional quota manager for execution quotas
        retry_handler: Retry handler with configurable backoff strategy

    Example:
        ```python
        registry = ToolRegistry()
        registry.register(GoogleSearchTool())

        rate_limiter = RateLimiter()
        retry_handler = RetryHandler(max_retries=3, strategy=BackoffStrategy.EXPONENTIAL)

        executor = ToolExecutor(
            registry=registry,
            db_session=db_session,
            rate_limiter=rate_limiter,
            retry_handler=retry_handler
        )

        # Add lifecycle hooks
        executor.add_before_hook(lambda ctx: print(f"Starting {ctx.request_id}"))
        executor.add_after_hook(lambda result: print(f"Completed with {result.status}"))

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
        rate_limiter: RateLimiter | None = None,
        quota_manager: QuotaManager | None = None,
        retry_handler: RetryHandler | None = None,
    ):
        """Initialize tool executor.

        Args:
            registry: ToolRegistry instance for tool lookup
            db_session: Optional database session for execution logging
            rate_limiter: Optional rate limiter for tool execution
            quota_manager: Optional quota manager for tool execution quotas
            retry_handler: Optional retry handler (defaults to 3 retries with exponential backoff)
        """
        self.registry = registry
        self.db_session = db_session
        self.rate_limiter = rate_limiter
        self.quota_manager = quota_manager
        self.retry_handler = retry_handler or RetryHandler(
            max_retries=3,
            base_delay=1.0,
            strategy=BackoffStrategy.EXPONENTIAL,
        )

        # Lifecycle hooks
        self._before_hooks: list[Callable[[ExecutionContext], None]] = []
        self._after_hooks: list[Callable[[ToolResult], None]] = []
        self._error_hooks: list[Callable[[ExecutionContext, Exception], None]] = []

        self.logger = logger.bind(component="tool_executor")
        self.logger.info(
            "tool_executor_initialized",
            rate_limiter_enabled=rate_limiter is not None,
            max_retries=self.retry_handler.max_retries,
        )

    def add_before_hook(self, hook: Callable[[ExecutionContext], None]) -> None:
        """Add a hook to run before tool execution.

        Args:
            hook: Function to call before execution (sync or async)
        """
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable[[ToolResult], None]) -> None:
        """Add a hook to run after tool execution.

        Args:
            hook: Function to call after execution (sync or async)
        """
        self._after_hooks.append(hook)

    def add_error_hook(self, hook: Callable[[ExecutionContext, Exception], None]) -> None:
        """Add a hook to run when tool execution fails.

        Args:
            hook: Function to call on error (sync or async)
        """
        self._error_hooks.append(hook)

    def _enrich_result_with_error_metadata(
        self,
        result: ToolResult,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ToolResult:
        """Enrich ToolResult with error categorization metadata.

        Args:
            result: Original ToolResult
            error_type: Error type string (e.g., "ToolNotFoundError")
            error_message: Error message for additional context

        Returns:
            ToolResult with enriched metadata including error category, code, and recovery guidance
        """
        if result.status != ToolExecutionStatus.FAILED or not error_type:
            return result

        # Categorize the error
        category, error_code, recovery_strategy = categorize_error(error_type, error_message)

        # Get user-friendly error metadata
        error_meta = get_error_metadata(category, error_code, recovery_strategy)

        # Enrich existing metadata
        enriched_metadata = result.metadata or {}
        enriched_metadata.update(
            {
                "error_category": error_meta["category"],
                "error_code": error_meta["error_code"],
                "user_message": error_meta["user_message"],
                "is_retryable": error_meta["is_retryable"],
                "recovery_strategy": error_meta["recovery_strategy"],
                "recovery_guidance": error_meta["recovery_guidance"],
            }
        )

        # Create new ToolResult with enriched metadata
        return ToolResult(
            request_id=result.request_id,
            tool_id=result.tool_id,
            status=result.status,
            result=result.result,
            error=result.error,
            error_type=result.error_type,
            execution_time_ms=result.execution_time_ms,
            timestamp=result.timestamp,
            metadata=enriched_metadata,
        )

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
        8. Distributed tracing with OpenTelemetry

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
        # Create OpenTelemetry span for distributed tracing
        tracer = get_tracer("agentcore.agent_runtime.tools.executor")

        with tracer.start_as_current_span(f"tool.execute.{tool_id}") as span:
            # Add span attributes for observability
            add_span_attributes(
                tool_id=tool_id,
                user_id=context.user_id or "unknown",
                agent_id=context.agent_id or "unknown",
                request_id=context.request_id,
                trace_id=context.trace_id,
            )

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
                # 0. Run before hooks
                await self._run_before_hooks(context)
                add_span_event("hooks.before_completed")

                # 1. Tool lookup from registry
                tool = self.registry.get(tool_id)
                if tool is None:
                    execution_time_ms = (time.time() - start_time) * 1000
                    error_msg = f"Tool '{tool_id}' not found in registry"
                    result = ToolResult(
                        request_id=context.request_id,
                        tool_id=tool_id,
                        status=ToolExecutionStatus.FAILED,
                        error=error_msg,
                        error_type="ToolNotFoundError",
                        execution_time_ms=execution_time_ms,
                        timestamp=timestamp,
                    )
                    # Enrich with error categorization metadata
                    result = self._enrich_result_with_error_metadata(
                        result, error_type="ToolNotFoundError", error_message=error_msg
                    )
                    await self._log_execution(result, context, parameters)
                    await self._run_error_hooks(context, Exception(result.error))
                    record_exception(Exception(result.error))
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
                    # Enrich with error categorization metadata
                    result = self._enrich_result_with_error_metadata(
                        result, error_type="ToolAuthenticationError", error_message=auth_error
                    )
                    await self._log_execution(result, context, parameters)
                    await self._run_error_hooks(context, ToolAuthenticationError(auth_error))
                    auth_exception = ToolAuthenticationError(auth_error)
                    record_exception(auth_exception)
                    return result

                add_span_event("authentication.validated")

                # 3. Parameter validation
                is_valid, validation_error = await tool.validate_parameters(parameters)
                if not is_valid:
                    execution_time_ms = (time.time() - start_time) * 1000
                    error_msg = f"Parameter validation failed: {validation_error}"
                    result = ToolResult(
                        request_id=context.request_id,
                        tool_id=tool_id,
                        status=ToolExecutionStatus.FAILED,
                        error=error_msg,
                        error_type="ToolValidationError",
                        execution_time_ms=execution_time_ms,
                        timestamp=timestamp,
                    )
                    # Enrich with error categorization metadata
                    result = self._enrich_result_with_error_metadata(
                        result, error_type="ToolValidationError", error_message=error_msg
                    )
                    await self._log_execution(result, context, parameters)
                    await self._run_error_hooks(context, ToolValidationError(validation_error))
                    validation_exception = ToolValidationError(validation_error)
                    record_exception(validation_exception)
                    return result

                add_span_event("parameters.validated", parameter_count=len(parameters))

                # 4. Check rate limits
                rate_limit_config = getattr(tool.metadata, "rate_limits", None)
                if self.rate_limiter and rate_limit_config:
                    requests_per_minute = rate_limit_config.get("requests_per_minute")
                    if requests_per_minute:
                        try:
                            await self.rate_limiter.check_rate_limit(
                                tool_id=tool_id,
                                limit=requests_per_minute,
                                window_seconds=60,
                                identifier=context.agent_id,
                            )
                            add_span_event("rate_limit.checked", limit=requests_per_minute)
                        except RateLimitExceeded as e:
                            execution_time_ms = (time.time() - start_time) * 1000
                            result = ToolResult(
                                request_id=context.request_id,
                                tool_id=tool_id,
                                status=ToolExecutionStatus.FAILED,
                                error=str(e),
                                error_type="RateLimitExceeded",
                                execution_time_ms=execution_time_ms,
                                timestamp=timestamp,
                                metadata={"retry_after": e.retry_after},
                            )
                            # Enrich with error categorization metadata
                            result = self._enrich_result_with_error_metadata(
                                result, error_type="RateLimitExceeded", error_message=str(e)
                            )
                            await self._log_execution(result, context, parameters)
                            await self._run_error_hooks(context, e)
                            record_exception(e)
                            self.logger.warning(
                                "tool_execution_rate_limited",
                                tool_id=tool_id,
                                agent_id=context.agent_id,
                                retry_after=e.retry_after,
                            )
                            return result

                # 5. Check quota limits
                daily_quota = getattr(tool.metadata, "daily_quota", None)
                monthly_quota = getattr(tool.metadata, "monthly_quota", None)
                if self.quota_manager and (daily_quota or monthly_quota):
                    try:
                        await self.quota_manager.check_quota(
                            tool_id=tool_id,
                            daily_quota=daily_quota,
                            monthly_quota=monthly_quota,
                            identifier=context.agent_id,
                        )
                        add_span_event(
                            "quota.checked",
                            daily_quota=daily_quota,
                            monthly_quota=monthly_quota,
                        )
                    except QuotaExceeded as e:
                        execution_time_ms = (time.time() - start_time) * 1000
                        result = ToolResult(
                            request_id=context.request_id,
                            tool_id=tool_id,
                            status=ToolExecutionStatus.FAILED,
                            error=str(e),
                            error_type="QuotaExceeded",
                            execution_time_ms=execution_time_ms,
                            timestamp=timestamp,
                            metadata={
                                "quota_type": e.quota_type,
                                "limit": e.limit,
                                "reset_at": e.reset_at.isoformat(),
                            },
                        )
                        # Enrich with error categorization metadata
                        result = self._enrich_result_with_error_metadata(
                            result, error_type="QuotaExceeded", error_message=str(e)
                        )
                        await self._log_execution(result, context, parameters)
                        await self._run_error_hooks(context, e)
                        record_exception(e)
                        self.logger.warning(
                            "tool_execution_quota_exceeded",
                            tool_id=tool_id,
                            agent_id=context.agent_id,
                            quota_type=e.quota_type,
                            reset_at=e.reset_at.isoformat(),
                        )
                        return result

                # 6. Tool execution with timeout and retry
                add_span_event("tool.execution_started")
                result, retry_count = await self._execute_with_retry(
                    tool, parameters, context, start_time, timestamp
                )
                add_span_event("tool.execution_completed", retry_count=retry_count)

                # Update retry count in result
                result.retry_count = retry_count

                # 6. Log execution to database
                await self._log_execution(result, context, parameters)

                # 7. Run after hooks
                await self._run_after_hooks(result)
                add_span_event("hooks.after_completed")

                # Log success
                if result.status == ToolExecutionStatus.SUCCESS:
                    self.logger.info(
                        "tool_execution_completed",
                        tool_id=tool_id,
                        status="success",
                        execution_time_ms=result.execution_time_ms,
                        retry_count=retry_count,
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

            except RateLimitExceeded as e:
                record_exception(e)
                execution_time_ms = (time.time() - start_time) * 1000
                result = ToolResult(
                    request_id=context.request_id,
                    tool_id=tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=str(e),
                    error_type="RateLimitExceeded",
                    execution_time_ms=execution_time_ms,
                    timestamp=timestamp,
                    metadata={"retry_after": e.retry_after},
                )
                # Enrich with error categorization metadata
                result = self._enrich_result_with_error_metadata(
                    result, error_type="RateLimitExceeded", error_message=str(e)
                )
                await self._log_execution(result, context, parameters)
                await self._run_error_hooks(context, e)
                return result

            except Exception as e:
                # Catch-all for unexpected errors
                record_exception(e)
                execution_time_ms = (time.time() - start_time) * 1000
                error_msg = f"Unexpected error: {str(e)}"
                result = ToolResult(
                    request_id=context.request_id,
                    tool_id=tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=error_msg,
                    error_type=type(e).__name__,
                    execution_time_ms=execution_time_ms,
                    timestamp=timestamp,
                )
                # Enrich with error categorization metadata
                result = self._enrich_result_with_error_metadata(
                    result, error_type=type(e).__name__, error_message=error_msg
                )
                await self._log_execution(result, context, parameters)
                await self._run_error_hooks(context, e)
                self.logger.error(
                    "tool_execution_error",
                    tool_id=tool_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=context.trace_id,
                )
                return result

    async def _run_before_hooks(self, context: ExecutionContext) -> None:
        """Run all before hooks."""
        for hook in self._before_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context)
                else:
                    hook(context)
            except Exception as e:
                self.logger.warning(
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
                self.logger.warning(
                    "after_hook_failed",
                    hook=hook.__name__,
                    error=str(e),
                )

    async def _run_error_hooks(self, context: ExecutionContext, error: Exception) -> None:
        """Run all error hooks."""
        for hook in self._error_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context, error)
                else:
                    hook(context, error)
            except Exception as e:
                self.logger.warning(
                    "error_hook_failed",
                    hook=hook.__name__,
                    error=str(e),
                )

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

    async def _execute_with_retry(
        self,
        tool: Tool,
        parameters: dict[str, Any],
        context: ExecutionContext,
        start_time: float,
        timestamp: datetime,
    ) -> tuple[ToolResult, int]:
        """Execute tool with timeout and retry management.

        Args:
            tool: Tool instance to execute
            parameters: Validated parameters
            context: Execution context
            start_time: Execution start time for metrics
            timestamp: Execution timestamp

        Returns:
            Tuple of (ToolResult from execution, retry count)
        """
        timeout = tool.metadata.timeout_seconds
        retry_count = 0

        # Determine if tool is retryable (check if metadata has is_retryable attribute)
        is_retryable = getattr(tool.metadata, "is_retryable", True)
        max_retries = getattr(tool.metadata, "max_retries", self.retry_handler.max_retries)

        async def execute_with_timeout() -> ToolResult:
            """Execute tool with timeout."""
            try:
                result = await asyncio.wait_for(
                    tool.execute(parameters, context),
                    timeout=timeout,
                )
                return result
            except asyncio.TimeoutError:
                execution_time_ms = (time.time() - start_time) * 1000
                error_msg = f"Tool execution exceeded timeout of {timeout} seconds"
                result = ToolResult(
                    request_id=context.request_id,
                    tool_id=tool.metadata.tool_id,
                    status=ToolExecutionStatus.TIMEOUT,
                    error=error_msg,
                    error_type="ToolTimeoutError",
                    execution_time_ms=execution_time_ms,
                    timestamp=timestamp,
                )
                # Enrich with error categorization metadata
                return self._enrich_result_with_error_metadata(
                    result, error_type="ToolTimeoutError", error_message=error_msg
                )

        def on_retry_callback(exception: Exception, attempt: int, delay: float) -> None:
            """Callback for retry attempts."""
            nonlocal retry_count
            retry_count = attempt
            self.logger.warning(
                "tool_execution_retry",
                tool_id=tool.metadata.tool_id,
                attempt=attempt,
                max_retries=max_retries,
                error=str(exception),
                delay=delay,
            )

        # Execute with retry logic if tool is retryable
        if is_retryable and max_retries > 0:
            # Create a retry handler for this execution
            retry_handler = RetryHandler(
                max_retries=max_retries,
                base_delay=self.retry_handler.base_delay,
                max_delay=self.retry_handler.max_delay,
                strategy=self.retry_handler.strategy,
                jitter=self.retry_handler.jitter,
            )

            result = await retry_handler.retry(
                execute_with_timeout,
                retryable_exceptions=(Exception,),
                on_retry=on_retry_callback,
            )
        else:
            result = await execute_with_timeout()

        return result, retry_count

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
