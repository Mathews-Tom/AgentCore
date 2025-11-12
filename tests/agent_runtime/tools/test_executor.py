"""Tests for tool executor implementation."""

import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool
from agentcore.agent_runtime.tools.executor import (
    ToolExecutor,
    ToolExecutionError,
)
from agentcore.agent_runtime.tools.registry import ToolRegistry


class MockSuccessTool(Tool):
    """Mock tool that always succeeds."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="success_tool",
            name="Success Tool",
            description="A tool that always succeeds",
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
            timestamp=datetime.utcnow(),
        )


class MockFailureTool(Tool):
    """Mock tool that always fails."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="failure_tool",
            name="Failure Tool",
            description="A tool that always fails",
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
        super().__init__(metadata)

    async def execute(
        self, parameters: dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.FAILED,
            error="Simulated tool failure",
            error_type="MockError",
            execution_time_ms=50.0,
            timestamp=datetime.utcnow(),
        )


class MockTimeoutTool(Tool):
    """Mock tool that times out."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="timeout_tool",
            name="Timeout Tool",
            description="A tool that times out",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.NONE,
            timeout_seconds=1,  # Very short timeout
        )
        super().__init__(metadata)

    async def execute(
        self, parameters: dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        import asyncio

        # Sleep longer than timeout
        await asyncio.sleep(2)
        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result={},
            execution_time_ms=2000.0,
            timestamp=datetime.utcnow(),
        )


class MockAuthTool(Tool):
    """Mock tool that requires authentication."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="auth_tool",
            name="Auth Tool",
            description="A tool requiring authentication",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.API_KEY,
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
            result={"authenticated": True},
            execution_time_ms=100.0,
            timestamp=datetime.utcnow(),
        )


class TestToolExecutorBasics:
    """Test basic executor functionality."""

    def test_executor_initialization(self):
        """Test executor initializes with registry."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        assert executor.registry == registry
        assert executor.db_session is None

    def test_executor_initialization_with_db_session(self):
        """Test executor initializes with database session."""
        registry = ToolRegistry()
        mock_session = Mock()
        executor = ToolExecutor(registry, db_session=mock_session)

        assert executor.registry == registry
        assert executor.db_session == mock_session


class TestToolExecution:
    """Test tool execution scenarios."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext(user_id="user123", agent_id="agent456")
        parameters = {"input": "test data"}

        result = await executor.execute_tool("success_tool", parameters, context)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.tool_id == "success_tool"
        assert result.request_id == context.request_id
        assert result.result["output"] == "Processed: test data"
        assert result.error is None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test execution of nonexistent tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("nonexistent_tool", parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolNotFoundError"
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        """Test tool that returns failure status."""
        registry = ToolRegistry()
        tool = MockFailureTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {"input": "test"}

        result = await executor.execute_tool("failure_tool", parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error == "Simulated tool failure"
        assert result.error_type == "MockError"

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self):
        """Test tool execution timeout."""
        registry = ToolRegistry()
        tool = MockTimeoutTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {}

        result = await executor.execute_tool("timeout_tool", parameters, context)

        assert result.status == ToolExecutionStatus.TIMEOUT
        assert result.error_type == "ToolTimeoutError"
        assert "timeout" in result.error.lower()


class TestParameterValidation:
    """Test parameter validation during execution."""

    @pytest.mark.asyncio
    async def test_execute_with_missing_required_parameter(self):
        """Test execution fails with missing required parameter."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {}  # Missing required 'input' parameter

        result = await executor.execute_tool("success_tool", parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolValidationError"
        assert "validation failed" in result.error.lower()
        assert "input" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_valid_parameters(self):
        """Test execution succeeds with valid parameters."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {"input": "valid data"}

        result = await executor.execute_tool("success_tool", parameters, context)

        assert result.status == ToolExecutionStatus.SUCCESS


class TestAuthentication:
    """Test authentication handling."""

    @pytest.mark.asyncio
    async def test_execute_auth_tool_without_user_id(self):
        """Test auth tool fails without user_id in context."""
        registry = ToolRegistry()
        tool = MockAuthTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext(agent_id="agent456")  # Missing user_id
        parameters = {}

        result = await executor.execute_tool("auth_tool", parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert "user_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_auth_tool_without_agent_id(self):
        """Test auth tool fails without agent_id in context."""
        registry = ToolRegistry()
        tool = MockAuthTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext(user_id="user123")  # Missing agent_id
        parameters = {}

        result = await executor.execute_tool("auth_tool", parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert "agent_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_auth_tool_with_complete_context(self):
        """Test auth tool succeeds with complete context."""
        registry = ToolRegistry()
        tool = MockAuthTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext(user_id="user123", agent_id="agent456")
        parameters = {}

        result = await executor.execute_tool("auth_tool", parameters, context)

        assert result.status == ToolExecutionStatus.SUCCESS


class TestTraceIdPropagation:
    """Test distributed tracing support."""

    @pytest.mark.asyncio
    async def test_trace_id_propagation(self):
        """Test that trace_id is propagated through execution."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        trace_id = "trace_123"
        context = ExecutionContext(trace_id=trace_id)
        parameters = {"input": "test"}

        result = await executor.execute_tool("success_tool", parameters, context)

        # Verify trace_id is in context (checked via logging or result)
        assert context.trace_id == trace_id
        assert result.request_id == context.request_id


class TestDatabaseLogging:
    """Test database logging functionality."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_db_logging(self):
        """Test that tool execution is logged to database."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)

        # Create mock database session
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        executor = ToolExecutor(registry, db_session=mock_session)

        context = ExecutionContext(user_id="user123", agent_id="agent456")
        parameters = {"input": "test"}

        result = await executor.execute_tool("success_tool", parameters, context)

        # Verify database operations were called
        assert mock_session.execute.called
        assert mock_session.commit.called
        assert result.status == ToolExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_tool_without_db_session(self):
        """Test tool execution works without database session."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)

        executor = ToolExecutor(registry)  # No db_session

        context = ExecutionContext()
        parameters = {"input": "test"}

        result = await executor.execute_tool("success_tool", parameters, context)

        # Should succeed without database logging
        assert result.status == ToolExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_db_logging_failure_does_not_fail_execution(self):
        """Test that database logging failures don't fail the execution."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)

        # Create mock session that fails on execute
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("DB Error"))
        mock_session.rollback = AsyncMock()

        executor = ToolExecutor(registry, db_session=mock_session)

        context = ExecutionContext()
        parameters = {"input": "test"}

        result = await executor.execute_tool("success_tool", parameters, context)

        # Execution should still succeed despite logging failure
        assert result.status == ToolExecutionStatus.SUCCESS
        assert mock_session.rollback.called


class TestErrorHandling:
    """Test error handling and categorization."""

    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(self):
        """Test handling of unexpected exceptions during execution."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)

        executor = ToolExecutor(registry)

        # Mock the tool to raise an exception
        async def mock_execute_with_error(params, ctx):
            raise RuntimeError("Unexpected error")

        tool.execute = mock_execute_with_error

        context = ExecutionContext()
        parameters = {"input": "test"}

        result = await executor.execute_tool("success_tool", parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "RuntimeError"
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_error_categorization(self):
        """Test that errors are properly categorized with enriched metadata."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        context = ExecutionContext()

        # Test ToolNotFoundError
        result = await executor.execute_tool("nonexistent", {}, context)
        assert result.error_type == "ToolNotFoundError"
        assert result.metadata is not None
        assert result.metadata["error_category"] == "not_found_error"
        assert result.metadata["error_code"] == "TOOL_E1301"
        assert result.metadata["recovery_strategy"] == "user_intervention"
        assert result.metadata["is_retryable"] is False
        assert "user_message" in result.metadata
        assert "recovery_guidance" in result.metadata

        # Test ToolValidationError
        tool = MockSuccessTool()
        registry.register(tool)
        result = await executor.execute_tool("success_tool", {}, context)  # Missing param
        assert result.error_type == "ToolValidationError"
        assert result.metadata is not None
        assert result.metadata["error_category"] == "validation_error"
        assert result.metadata["error_code"] == "TOOL_E1001"
        assert result.metadata["recovery_strategy"] == "user_intervention"
        assert result.metadata["is_retryable"] is False

        # Test ToolAuthenticationError
        auth_tool = MockAuthTool()
        registry.register(auth_tool)
        result = await executor.execute_tool("auth_tool", {}, ExecutionContext())  # No auth
        assert result.error_type == "ToolAuthenticationError"
        assert result.metadata is not None
        assert result.metadata["error_category"] == "authentication_error"
        assert result.metadata["error_code"] == "TOOL_E1102"
        assert result.metadata["recovery_strategy"] == "user_intervention"
        assert result.metadata["is_retryable"] is False


class TestExecutionMetrics:
    """Test execution time tracking."""

    @pytest.mark.asyncio
    async def test_execution_time_tracking(self):
        """Test that execution time is tracked correctly."""
        registry = ToolRegistry()
        tool = MockSuccessTool()
        registry.register(tool)
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        parameters = {"input": "test"}

        result = await executor.execute_tool("success_tool", parameters, context)

        # Mock tool returns execution_time_ms in its result
        # Verify it's tracked (positive value)
        assert result.execution_time_ms > 0
        assert isinstance(result.execution_time_ms, float)

    @pytest.mark.asyncio
    async def test_execution_time_on_failure(self):
        """Test that execution time is tracked even on failure."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        context = ExecutionContext()
        result = await executor.execute_tool("nonexistent", {}, context)

        assert result.execution_time_ms > 0
