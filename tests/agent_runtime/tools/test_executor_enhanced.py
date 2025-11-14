"""Tests for enhanced tool executor features (rate limiting, retry, hooks)."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from agentcore.agent_runtime.services.rate_limiter import RateLimiter
from agentcore.agent_runtime.services.retry_handler import BackoffStrategy, RetryHandler
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry


class MockToolWithRateLimit(Tool):
    """Mock tool that has rate limiting configured."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="rate_limited_tool",
            name="Rate Limited Tool",
            description="A tool with rate limiting",
            category=ToolCategory.UTILITY,
            parameters={
                "input": ToolParameter(
                    name="input",
                    type="string",
                    description="Input parameter",
                    required=True,
                )
            },
            auth_method=AuthMethod.NONE,
            timeout_seconds=5,
        )
        # Add rate limit configuration
        metadata.rate_limits = {"requests_per_minute": 10}
        super().__init__(metadata)

    async def execute(
        self, parameters: dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result={"output": f"Processed: {parameters['input']}"},
            execution_time_ms=100.0,
            timestamp=datetime.now(UTC),
        )


class MockRetryableTool(Tool):
    """Mock tool that fails twice then succeeds (for retry testing)."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="retryable_tool",
            name="Retryable Tool",
            description="A tool that can be retried",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.NONE,
            timeout_seconds=5,
        )
        metadata.is_retryable = True
        metadata.max_retries = 3
        super().__init__(metadata)
        self.attempt_count = 0

    async def execute(
        self, parameters: dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        self.attempt_count += 1
        if self.attempt_count < 3:
            # Fail first two attempts
            raise Exception(f"Attempt {self.attempt_count} failed")

        # Succeed on third attempt
        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result={"attempts": self.attempt_count},
            execution_time_ms=100.0,
            timestamp=datetime.now(UTC),
        )


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled_by_default(self):
        """Test that rate limiting is disabled when no rate limiter provided."""
        registry = ToolRegistry()
        tool = MockToolWithRateLimit()
        registry.register(tool)

        # No rate limiter provided
        executor = ToolExecutor(registry)

        context = ExecutionContext(user_id="user123", agent_id="agent456")
        parameters = {"input": "test"}

        # Should execute without rate limiting
        result = await executor.execute_tool("rate_limited_tool", parameters, context)
        assert result.status == ToolExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_rate_limiting_with_rate_limiter(self):
        """Test rate limiting when rate limiter is provided."""
        registry = ToolRegistry()
        tool = MockToolWithRateLimit()
        registry.register(tool)

        # Create mock rate limiter
        mock_rate_limiter = Mock(spec=RateLimiter)
        mock_rate_limiter.check_rate_limit = AsyncMock()

        executor = ToolExecutor(registry, rate_limiter=mock_rate_limiter)

        context = ExecutionContext(user_id="user123", agent_id="agent456")
        parameters = {"input": "test"}

        result = await executor.execute_tool("rate_limited_tool", parameters, context)

        # Verify rate limiter was called
        assert mock_rate_limiter.check_rate_limit.called
        assert result.status == ToolExecutionStatus.SUCCESS


class TestRetryLogic:
    """Test retry logic functionality."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that tool is retried on failure."""
        registry = ToolRegistry()
        tool = MockRetryableTool()
        registry.register(tool)

        retry_handler = RetryHandler(
            max_retries=3,
            base_delay=0.01,  # Very short delay for testing
            strategy=BackoffStrategy.FIXED,
        )
        executor = ToolExecutor(registry, retry_handler=retry_handler)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("retryable_tool", parameters, context)

        # Should succeed after retries
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["attempts"] == 3
        assert result.retry_count >= 2  # At least 2 retries

    @pytest.mark.asyncio
    async def test_retry_disabled_for_non_retryable_tool(self):
        """Test that non-retryable tools don't retry."""
        registry = ToolRegistry()

        # Create tool that always fails and is not retryable
        class NonRetryableTool(Tool):
            def __init__(self):
                metadata = ToolDefinition(
                    tool_id="non_retryable",
                    name="Non Retryable",
                    description="Non retryable tool",
                    category=ToolCategory.UTILITY,
                    parameters={},
                    auth_method=AuthMethod.NONE,
                    timeout_seconds=5,
                )
                metadata.is_retryable = False
                super().__init__(metadata)

            async def execute(
                self, parameters: dict[str, Any], context: ExecutionContext
            ) -> ToolResult:
                raise Exception("Always fails")

        tool = NonRetryableTool()
        registry.register(tool)

        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("non_retryable", parameters, context)

        # Should fail without retry
        assert result.status == ToolExecutionStatus.FAILED
        assert result.retry_count == 0


class TestLifecycleHooks:
    """Test lifecycle hooks functionality."""

    @pytest.mark.asyncio
    async def test_before_hook_called(self):
        """Test that before hooks are called before execution."""
        registry = ToolRegistry()

        # Create simple success tool
        class SimpleTool(Tool):
            def __init__(self):
                metadata = ToolDefinition(
                    tool_id="simple_tool",
                    name="Simple Tool",
                    description="Simple test tool",
                    category=ToolCategory.UTILITY,
                    parameters={},
                    auth_method=AuthMethod.NONE,
                    timeout_seconds=5,
                )
                super().__init__(metadata)

            async def execute(
                self, parameters: dict[str, Any], context: ExecutionContext
            ) -> ToolResult:
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result={},
                    execution_time_ms=100.0,
                    timestamp=datetime.now(UTC),
                )

        tool = SimpleTool()
        registry.register(tool)

        executor = ToolExecutor(registry)

        # Track hook calls
        before_called = {"called": False}

        def before_hook(context: ExecutionContext):
            before_called["called"] = True

        executor.add_before_hook(before_hook)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("simple_tool", parameters, context)

        assert before_called["called"] is True
        assert result.status == ToolExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_after_hook_called(self):
        """Test that after hooks are called after successful execution."""
        registry = ToolRegistry()

        class SimpleTool(Tool):
            def __init__(self):
                metadata = ToolDefinition(
                    tool_id="simple_tool",
                    name="Simple Tool",
                    description="Simple test tool",
                    category=ToolCategory.UTILITY,
                    parameters={},
                    auth_method=AuthMethod.NONE,
                    timeout_seconds=5,
                )
                super().__init__(metadata)

            async def execute(
                self, parameters: dict[str, Any], context: ExecutionContext
            ) -> ToolResult:
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result={},
                    execution_time_ms=100.0,
                    timestamp=datetime.now(UTC),
                )

        tool = SimpleTool()
        registry.register(tool)

        executor = ToolExecutor(registry)

        # Track hook calls
        after_called = {"called": False, "result": None}

        def after_hook(result: ToolResult):
            after_called["called"] = True
            after_called["result"] = result

        executor.add_after_hook(after_hook)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("simple_tool", parameters, context)

        assert after_called["called"] is True
        assert after_called["result"] == result
        assert result.status == ToolExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_error_hook_called_on_failure(self):
        """Test that error hooks are called on execution failure."""
        registry = ToolRegistry()

        class FailingTool(Tool):
            def __init__(self):
                metadata = ToolDefinition(
                    tool_id="failing_tool",
                    name="Failing Tool",
                    description="Tool that always fails",
                    category=ToolCategory.UTILITY,
                    parameters={},
                    auth_method=AuthMethod.NONE,
                    timeout_seconds=5,
                )
                metadata.is_retryable = False
                super().__init__(metadata)

            async def execute(
                self, parameters: dict[str, Any], context: ExecutionContext
            ) -> ToolResult:
                raise Exception("Tool failure")

        tool = FailingTool()
        registry.register(tool)

        executor = ToolExecutor(registry)

        # Track hook calls
        error_called = {"called": False, "error": None}

        def error_hook(context: ExecutionContext, error: Exception):
            error_called["called"] = True
            error_called["error"] = error

        executor.add_error_hook(error_hook)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("failing_tool", parameters, context)

        assert error_called["called"] is True
        assert isinstance(error_called["error"], Exception)
        assert result.status == ToolExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_async_hooks_supported(self):
        """Test that async hooks are properly supported."""
        registry = ToolRegistry()

        class SimpleTool(Tool):
            def __init__(self):
                metadata = ToolDefinition(
                    tool_id="simple_tool",
                    name="Simple Tool",
                    description="Simple test tool",
                    category=ToolCategory.UTILITY,
                    parameters={},
                    auth_method=AuthMethod.NONE,
                    timeout_seconds=5,
                )
                super().__init__(metadata)

            async def execute(
                self, parameters: dict[str, Any], context: ExecutionContext
            ) -> ToolResult:
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result={},
                    execution_time_ms=100.0,
                    timestamp=datetime.now(UTC),
                )

        tool = SimpleTool()
        registry.register(tool)

        executor = ToolExecutor(registry)

        # Track async hook calls
        async_called = {"called": False}

        async def async_before_hook(context: ExecutionContext):
            await asyncio.sleep(0.001)  # Simulate async operation
            async_called["called"] = True

        executor.add_before_hook(async_before_hook)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("simple_tool", parameters, context)

        assert async_called["called"] is True
        assert result.status == ToolExecutionStatus.SUCCESS
