"""Unit tests for tool JSON-RPC methods with mocked ToolExecutor."""

from unittest.mock import AsyncMock, Mock, patch
from datetime import UTC, datetime

import pytest

from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest
from agentcore.agent_runtime.jsonrpc.tools_jsonrpc import (
    handle_tools_execute,
    handle_tools_list,
    handle_tools_get,
)
from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionStatus,
    ToolResult,
    ToolCategory,
)


@pytest.mark.asyncio
class TestToolsExecuteJSONRPC:
    """Unit tests for tools.execute JSON-RPC method."""

    async def test_tools_execute_success(self):
        """Test successful tool execution."""
        # Mock ToolResult
        mock_result = ToolResult(
            request_id="test-request-1",
            tool_id="test_tool",
            status=ToolExecutionStatus.SUCCESS,
            result={"output": "test output"},
            error=None,
            execution_time_ms=150.0,
            timestamp=datetime.now(UTC),
        )

        # Mock ToolExecutor
        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            # Create request
            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {"input": "test input"},
                    "agent_id": "test-agent-001",
                },
                id="1",
            )

            # Execute
            result = await handle_tools_execute(request)

            # Verify
            assert result["tool_id"] == "test_tool"
            assert result["status"] == "success"
            assert result["result"] == {"output": "test output"}
            assert result["error"] is None
            assert result["execution_time_ms"] == 150.0
            assert "timestamp" in result

            # Verify executor was called correctly
            mock_executor.execute_tool.assert_called_once()
            call_args = mock_executor.execute_tool.call_args
            assert call_args[0][0] == "test_tool"
            assert call_args[0][1] == {"input": "test input"}
            assert call_args[0][2].agent_id == "test-agent-001"

    async def test_tools_execute_with_a2a_context(self):
        """Test tool execution with A2A context for trace_id and source_agent."""
        mock_result = ToolResult(
            request_id="test-request-2",
            tool_id="test_tool",
            status=ToolExecutionStatus.SUCCESS,
            result={"output": "test"},
            execution_time_ms=100.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            # Create request with A2A context
            a2a_context = A2AContext(
                trace_id="trace-123",
                source_agent="agent-source",
                target_agent="agent-target",
                session_id="session-456",
                timestamp=datetime.now(UTC).isoformat(),
            )

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {},
                },
                id="2",
                a2a_context=a2a_context,
            )

            # Execute
            result = await handle_tools_execute(request)

            # Verify A2A context was used
            assert result["status"] == "success"
            call_args = mock_executor.execute_tool.call_args
            context = call_args[0][2]
            assert context.trace_id == "trace-123"
            assert context.session_id == "session-456"
            assert context.agent_id == "agent-source"

    async def test_tools_execute_missing_tool_id(self):
        """Test tools.execute with missing tool_id parameter (400)."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.execute",
            params={
                "parameters": {},
                "agent_id": "test-agent",
            },
            id="3",
        )

        with pytest.raises(ValueError, match="tool_id parameter required"):
            await handle_tools_execute(request)

    async def test_tools_execute_missing_agent_id(self):
        """Test tools.execute without agent_id or A2A context (400)."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.execute",
            params={
                "tool_id": "test_tool",
                "parameters": {},
            },
            id="4",
        )

        with pytest.raises(
            ValueError,
            match="agent_id parameter required or must be provided via A2A context",
        ):
            await handle_tools_execute(request)

    async def test_tools_execute_tool_not_found(self):
        """Test tools.execute with non-existent tool (404)."""
        # Mock ToolExecutor to raise error for tool not found
        mock_result = ToolResult(
            request_id="test-request-3",
            tool_id="nonexistent_tool",
            status=ToolExecutionStatus.FAILED,
            result=None,
            error="Tool not found: nonexistent_tool",
            error_type="ToolNotFoundError",
            execution_time_ms=5.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "nonexistent_tool",
                    "parameters": {},
                    "agent_id": "test-agent",
                },
                id="5",
            )

            result = await handle_tools_execute(request)

            # Verify error response
            assert result["status"] == "failed"
            assert result["error"] == "Tool not found: nonexistent_tool"
            assert result["result"] is None

    async def test_tools_execute_invalid_parameters(self):
        """Test tools.execute with invalid parameters (400)."""
        mock_result = ToolResult(
            request_id="test-request-4",
            tool_id="test_tool",
            status=ToolExecutionStatus.FAILED,
            result=None,
            error="Invalid parameter: 'required_param' is required",
            error_type="ValidationError",
            execution_time_ms=10.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {"invalid_param": "value"},
                    "agent_id": "test-agent",
                },
                id="6",
            )

            result = await handle_tools_execute(request)

            assert result["status"] == "failed"
            assert "required" in result["error"].lower()

    async def test_tools_execute_rate_limit_exceeded(self):
        """Test tools.execute when rate limit is exceeded (429)."""
        mock_result = ToolResult(
            request_id="test-request-5",
            tool_id="test_tool",
            status=ToolExecutionStatus.FAILED,
            result=None,
            error="Rate limit exceeded for tool: test_tool",
            error_type="RateLimitError",
            execution_time_ms=2.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {},
                    "agent_id": "test-agent",
                },
                id="7",
            )

            result = await handle_tools_execute(request)

            assert result["status"] == "failed"
            assert "rate limit" in result["error"].lower()

    async def test_tools_execute_timeout(self):
        """Test tools.execute when execution times out (408)."""
        mock_result = ToolResult(
            request_id="test-request-6",
            tool_id="test_tool",
            status=ToolExecutionStatus.TIMEOUT,
            result=None,
            error="Tool execution timed out after 30 seconds",
            error_type="TimeoutError",
            execution_time_ms=30000.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {},
                    "agent_id": "test-agent",
                },
                id="8",
            )

            result = await handle_tools_execute(request)

            assert result["status"] == "timeout"
            assert result["error"] is not None
            assert "timed out" in result["error"].lower()
            assert result["execution_time_ms"] == 30000.0

    async def test_tools_execute_execution_error(self):
        """Test tools.execute when tool execution fails (500)."""
        mock_result = ToolResult(
            request_id="test-request-7",
            tool_id="test_tool",
            status=ToolExecutionStatus.FAILED,
            result=None,
            error="Internal execution error: Division by zero",
            error_type="ExecutionError",
            execution_time_ms=50.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {"divisor": 0},
                    "agent_id": "test-agent",
                },
                id="9",
            )

            result = await handle_tools_execute(request)

            assert result["status"] == "failed"
            assert "error" in result["error"].lower()

    async def test_tools_execute_with_timeout_override(self):
        """Test tools.execute with timeout override parameter."""
        mock_result = ToolResult(
            request_id="test-request-8",
            tool_id="test_tool",
            status=ToolExecutionStatus.SUCCESS,
            result={"data": "success"},
            execution_time_ms=25000.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {},
                    "agent_id": "test-agent",
                    "timeout_override": 60,
                },
                id="10",
            )

            result = await handle_tools_execute(request)

            assert result["status"] == "success"
            assert result["execution_time_ms"] == 25000.0

    async def test_tools_execute_with_execution_context(self):
        """Test tools.execute with legacy execution_context parameter."""
        mock_result = ToolResult(
            request_id="test-request-9",
            tool_id="test_tool",
            status=ToolExecutionStatus.SUCCESS,
            result={"data": "success"},
            execution_time_ms=100.0,
            timestamp=datetime.now(UTC),
        )

        with patch(
            "agentcore.agent_runtime.jsonrpc.tools_jsonrpc.get_tool_executor"
        ) as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tool = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="tools.execute",
                params={
                    "tool_id": "test_tool",
                    "parameters": {},
                    "agent_id": "test-agent",
                    "execution_context": {
                        "trace_id": "legacy-trace-123",
                        "session_id": "legacy-session-456",
                    },
                },
                id="11",
            )

            result = await handle_tools_execute(request)

            assert result["status"] == "success"
            call_args = mock_executor.execute_tool.call_args
            context = call_args[0][2]
            assert context.trace_id == "legacy-trace-123"
            assert context.session_id == "legacy-session-456"


@pytest.mark.asyncio
class TestToolsListJSONRPC:
    """Unit tests for tools.list JSON-RPC method."""

    async def test_tools_list_all(self):
        """Test listing all tools."""
        # Use real registry since tools.list doesn't need mocking
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.list",
            params={},
            id="1",
        )

        result = await handle_tools_list(request)

        assert "tools" in result
        assert "count" in result
        assert isinstance(result["tools"], list)
        assert result["count"] == len(result["tools"])
        assert result["count"] > 0

    async def test_tools_list_with_category_filter(self):
        """Test listing tools filtered by category."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.list",
            params={"category": "utility"},
            id="2",
        )

        result = await handle_tools_list(request)

        assert "tools" in result
        for tool in result["tools"]:
            assert tool["category"] == "utility"

    async def test_tools_list_with_invalid_category(self):
        """Test listing tools with invalid category."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.list",
            params={"category": "INVALID_CATEGORY"},
            id="3",
        )

        with pytest.raises(ValueError, match="Invalid category"):
            await handle_tools_list(request)

    async def test_tools_list_with_capabilities_filter(self):
        """Test listing tools filtered by capabilities."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.list",
            params={"capabilities": ["external_api"]},
            id="4",
        )

        result = await handle_tools_list(request)

        assert "tools" in result
        for tool in result["tools"]:
            assert "external_api" in tool["capabilities"]

    async def test_tools_list_with_tags_filter(self):
        """Test listing tools filtered by tags."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.list",
            params={"tags": ["search"]},
            id="5",
        )

        result = await handle_tools_list(request)

        assert "tools" in result
        for tool in result["tools"]:
            assert "search" in tool["tags"]


@pytest.mark.asyncio
class TestToolsGetJSONRPC:
    """Unit tests for tools.get JSON-RPC method."""

    async def test_tools_get_existing_tool(self):
        """Test getting detailed information for an existing tool."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.get",
            params={"tool_id": "calculator"},
            id="1",
        )

        result = await handle_tools_get(request)

        assert result["tool_id"] == "calculator"
        assert "name" in result
        assert "description" in result
        assert "version" in result
        assert "category" in result
        assert "parameters" in result
        assert "auth_method" in result
        assert "timeout_seconds" in result

    async def test_tools_get_nonexistent_tool(self):
        """Test getting information for non-existent tool."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.get",
            params={"tool_id": "nonexistent_tool_xyz"},
            id="2",
        )

        with pytest.raises(ValueError, match="Tool not found"):
            await handle_tools_get(request)

    async def test_tools_get_missing_tool_id(self):
        """Test tools.get without tool_id parameter."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools.get",
            params={},
            id="3",
        )

        with pytest.raises(ValueError, match="tool_id parameter required"):
            await handle_tools_get(request)
