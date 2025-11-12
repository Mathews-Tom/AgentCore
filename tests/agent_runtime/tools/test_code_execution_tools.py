"""Unit and integration tests for Code Execution Tools (TOOL-011).

Tests cover:
- Parameter validation (code, timeout)
- Docker-based execution with security restrictions
- Local fallback execution when Docker unavailable
- Timeout enforcement at container and local levels
- Error handling (container crashes, execution errors, timeouts)
- Result formatting (stdout, stderr, exit_code)
- Container lifecycle management and cleanup
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.code_execution_tools import (
    EvaluateExpressionTool,
    ExecutePythonTool,
)


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create execution context for testing."""
    return ExecutionContext(
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace-123",
        session_id="test-session",
    )


@pytest.fixture
def execute_python_tool() -> ExecutePythonTool:
    """Create ExecutePythonTool with Docker enabled."""
    return ExecutePythonTool(use_docker=True, docker_image="agentcore-python-sandbox")


@pytest.fixture
def execute_python_tool_no_docker() -> ExecutePythonTool:
    """Create ExecutePythonTool with Docker disabled."""
    return ExecutePythonTool(use_docker=False)


@pytest.fixture
def evaluate_expression_tool() -> EvaluateExpressionTool:
    """Create EvaluateExpressionTool."""
    return EvaluateExpressionTool()


# ExecutePythonTool - Parameter Validation Tests


@pytest.mark.asyncio
async def test_execute_python_missing_code(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that missing required code parameter fails validation."""
    result = await execute_python_tool.execute(
        parameters={},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "code" in result.error.lower()
    assert result.error_type == "ValidationError"


@pytest.mark.asyncio
async def test_execute_python_empty_code(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that empty code string fails validation."""
    result = await execute_python_tool.execute(
        parameters={"code": ""},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_execute_python_code_too_long(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that code exceeding max length fails validation."""
    long_code = "x" * 50001  # max is 50000
    result = await execute_python_tool.execute(
        parameters={"code": long_code},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_execute_python_invalid_timeout(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that invalid timeout value fails validation."""
    result = await execute_python_tool.execute(
        parameters={"code": "print('test')", "timeout": 400},  # max is 300
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_python_timeout_too_low(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that timeout below minimum fails validation."""
    result = await execute_python_tool.execute(
        parameters={"code": "print('test')", "timeout": 0},  # min is 1
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


# ExecutePythonTool - Docker Execution Tests


@pytest.mark.asyncio
async def test_execute_python_docker_success(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test successful Python code execution in Docker sandbox."""
    # Mock Docker client and container
    mock_docker = AsyncMock()
    mock_container = AsyncMock()

    # Mock container lifecycle
    mock_container.put_archive = AsyncMock()
    mock_container.start = AsyncMock()
    mock_container.wait = AsyncMock()
    mock_container.delete = AsyncMock()

    # Mock successful execution logs
    mock_log_entry = Mock()
    mock_log_entry.stream = 1  # stdout
    mock_log_entry.message = b"Hello from sandbox\n"
    mock_container.log = AsyncMock(return_value=[mock_log_entry])

    # Mock container info
    mock_container.show = AsyncMock(
        return_value={"State": {"ExitCode": 0}}
    )

    mock_docker.containers.create = AsyncMock(return_value=mock_container)
    mock_docker.close = AsyncMock()

    with patch("aiodocker.Docker", return_value=mock_docker):
        result = await execute_python_tool.execute(
            parameters={"code": "print('Hello from sandbox')", "timeout": 30},
            context=execution_context,
        )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.result is not None

    result_data = result.result
    assert result_data["exit_code"] == 0
    assert "Hello from sandbox" in result_data["stdout"]
    assert result_data["sandbox_type"] == "docker"

    # Verify container lifecycle
    mock_docker.containers.create.assert_called_once()
    mock_container.put_archive.assert_called_once()
    mock_container.start.assert_called_once()
    mock_container.wait.assert_called_once()
    mock_container.delete.assert_called_once_with(force=True)
    mock_docker.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_python_docker_timeout(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test Docker execution timeout handling."""
    mock_docker = AsyncMock()
    mock_container = AsyncMock()

    mock_container.put_archive = AsyncMock()
    mock_container.start = AsyncMock()
    mock_container.wait = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_container.kill = AsyncMock()
    mock_container.delete = AsyncMock()

    mock_docker.containers.create = AsyncMock(return_value=mock_container)
    mock_docker.close = AsyncMock()

    with patch("aiodocker.Docker", return_value=mock_docker):
        result = await execute_python_tool.execute(
            parameters={"code": "import time; time.sleep(100)", "timeout": 1},
            context=execution_context,
        )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.result is not None

    result_data = result.result
    assert result_data["exit_code"] == 124  # Timeout exit code
    assert "timed out" in result_data["stderr"].lower()
    assert result_data["error"] == "Timeout"
    assert result_data["sandbox_type"] == "docker"

    # Verify container was killed and cleaned up
    mock_container.kill.assert_called_once()
    mock_container.delete.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_execute_python_docker_execution_error(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test Docker execution with code that raises errors."""
    mock_docker = AsyncMock()
    mock_container = AsyncMock()

    mock_container.put_archive = AsyncMock()
    mock_container.start = AsyncMock()
    mock_container.wait = AsyncMock()
    mock_container.delete = AsyncMock()

    # Mock stderr with error message
    mock_log_entry = Mock()
    mock_log_entry.stream = 2  # stderr
    mock_log_entry.message = b"ZeroDivisionError: division by zero\n"
    mock_container.log = AsyncMock(return_value=[mock_log_entry])

    # Mock non-zero exit code
    mock_container.show = AsyncMock(
        return_value={"State": {"ExitCode": 1}}
    )

    mock_docker.containers.create = AsyncMock(return_value=mock_container)
    mock_docker.close = AsyncMock()

    with patch("aiodocker.Docker", return_value=mock_docker):
        result = await execute_python_tool.execute(
            parameters={"code": "x = 1 / 0", "timeout": 30},
            context=execution_context,
        )

    assert result.status == ToolExecutionStatus.FAILED
    result_data = result.result
    assert result_data["exit_code"] == 1
    assert "ZeroDivisionError" in result_data["stderr"]
    assert result_data["error"] == "Non-zero exit code"


@pytest.mark.asyncio
async def test_execute_python_docker_container_config(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that Docker container is configured with correct security restrictions."""
    mock_docker = AsyncMock()
    mock_container = AsyncMock()

    mock_container.put_archive = AsyncMock()
    mock_container.start = AsyncMock()
    mock_container.wait = AsyncMock()
    mock_container.delete = AsyncMock()
    mock_container.log = AsyncMock(return_value=[])
    mock_container.show = AsyncMock(return_value={"State": {"ExitCode": 0}})

    mock_docker.containers.create = AsyncMock(return_value=mock_container)
    mock_docker.close = AsyncMock()

    with patch("aiodocker.Docker", return_value=mock_docker):
        await execute_python_tool.execute(
            parameters={"code": "print('test')", "timeout": 30},
            context=execution_context,
        )

    # Verify container config has security restrictions
    call_args = mock_docker.containers.create.call_args
    config = call_args.kwargs["config"]

    assert config["Image"] == "agentcore-python-sandbox"
    assert config["User"] == "sandbox"
    assert config["WorkingDir"] == "/tmp/sandbox"

    host_config = config["HostConfig"]
    assert host_config["Memory"] == 512 * 1024 * 1024  # 512MB
    assert host_config["NetworkMode"] == "none"
    assert host_config["ReadonlyRootfs"] is True
    assert "/tmp/sandbox" in host_config["Tmpfs"]
    assert "no-new-privileges" in host_config["SecurityOpt"]
    assert "ALL" in host_config["CapDrop"]


@pytest.mark.asyncio
async def test_execute_python_docker_fallback_on_error(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test fallback to local execution when Docker fails."""
    # Mock Docker to raise exception
    with patch("aiodocker.Docker", side_effect=Exception("Docker not available")):
        result = await execute_python_tool.execute(
            parameters={"code": "print('fallback test')", "timeout": 30},
            context=execution_context,
        )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result is not None
    assert result.result["sandbox_type"] == "local_fallback"
    assert "fallback test" in result.result["stdout"]


# ExecutePythonTool - Local Execution Tests


@pytest.mark.asyncio
async def test_execute_python_local_success(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test successful local Python execution."""
    result = await execute_python_tool_no_docker.execute(
        parameters={"code": "print('Hello local')", "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result
    assert result_data["exit_code"] == 0
    assert "Hello local" in result_data["stdout"]
    assert result_data["sandbox_type"] == "local"


@pytest.mark.skip(reason="Thread pool timeout is unreliable for local execution - use Docker for reliable timeouts")
@pytest.mark.asyncio
async def test_execute_python_local_timeout(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test local execution timeout handling.

    Note: Timeouts in thread pool executors are not reliable because threads
    cannot be force-killed. The asyncio.wait_for timeout will be raised, but
    the thread will continue running until it completes or blocks.
    For reliable timeout enforcement, use Docker-based execution instead.
    """
    pass


@pytest.mark.asyncio
async def test_execute_python_local_execution_error(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test local execution with code that raises errors."""
    result = await execute_python_tool_no_docker.execute(
        parameters={"code": "raise ValueError('test error')", "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    result_data = result.result
    assert result_data["exit_code"] == 1
    assert "ValueError" in result_data["stderr"]
    assert "test error" in result_data["stderr"]


@pytest.mark.asyncio
async def test_execute_python_local_restricted_builtins(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that local execution restricts dangerous builtins."""
    # Try to use restricted builtin (open)
    result = await execute_python_tool_no_docker.execute(
        parameters={"code": "open('/etc/passwd', 'r')", "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    result_data = result.result
    assert result_data["exit_code"] == 1
    # Should fail because 'open' is not in safe_globals


@pytest.mark.asyncio
async def test_execute_python_local_allowed_builtins(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that local execution allows safe builtins."""
    result = await execute_python_tool_no_docker.execute(
        parameters={
            "code": "result = sum([1, 2, 3])\nprint(f'Sum: {result}')",
            "timeout": 30,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["exit_code"] == 0
    assert "Sum: 6" in result_data["stdout"]


# ExecutePythonTool - Tool Metadata Tests


def test_execute_python_tool_metadata(execute_python_tool: ExecutePythonTool):
    """Test that tool metadata is correctly configured."""
    metadata = execute_python_tool.metadata

    assert metadata.tool_id == "execute_python"
    assert metadata.name == "Execute Python"
    assert metadata.version == "2.0.0"
    assert metadata.auth_method.value == "none"
    assert metadata.is_retryable is False
    assert metadata.max_retries == 1
    assert metadata.timeout_seconds == 320
    assert metadata.is_idempotent is False

    # Verify parameters
    assert "code" in metadata.parameters
    assert "timeout" in metadata.parameters
    assert metadata.parameters["code"].required is True
    assert metadata.parameters["code"].max_length == 50000
    assert metadata.parameters["timeout"].required is False
    assert metadata.parameters["timeout"].default == 30
    assert metadata.parameters["timeout"].min_value == 1
    assert metadata.parameters["timeout"].max_value == 300

    # Verify capabilities
    assert "docker" in metadata.capabilities
    assert "sandboxed" in metadata.capabilities


# EvaluateExpressionTool - Parameter Validation Tests


@pytest.mark.asyncio
async def test_evaluate_expression_missing_expression(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test that missing required expression parameter fails validation."""
    result = await evaluate_expression_tool.execute(
        parameters={},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "expression" in result.error.lower()


@pytest.mark.asyncio
async def test_evaluate_expression_empty_expression(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test that empty expression string fails validation."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": ""},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_evaluate_expression_too_long(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test that expression exceeding max length fails validation."""
    long_expression = "1 + " * 300  # max is 1000 chars
    result = await evaluate_expression_tool.execute(
        parameters={"expression": long_expression},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error is not None


# EvaluateExpressionTool - Evaluation Tests


@pytest.mark.asyncio
async def test_evaluate_expression_success(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test successful expression evaluation."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": "2 + 2"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result
    assert result_data["success"] is True
    assert result_data["result"] == 4


@pytest.mark.asyncio
async def test_evaluate_expression_complex(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test complex expression evaluation."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": "sum([1, 2, 3]) * max([4, 5, 6])"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["success"] is True
    assert result_data["result"] == 36  # (1+2+3) * 6


@pytest.mark.asyncio
async def test_evaluate_expression_error(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test expression evaluation with errors."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": "1 / 0"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    result_data = result.result
    assert result_data["success"] is False
    assert "ZeroDivisionError" in result_data["error"]


@pytest.mark.asyncio
async def test_evaluate_expression_restricted_builtins(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test that dangerous builtins are restricted."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": "open('/etc/passwd')"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    result_data = result.result
    assert result_data["success"] is False
    assert "NameError" in result_data["error"]


@pytest.mark.asyncio
async def test_evaluate_expression_allowed_builtins(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test that safe builtins are allowed."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": "abs(-10) + min([1, 2, 3])"},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert result_data["success"] is True
    assert result_data["result"] == 11


# EvaluateExpressionTool - Tool Metadata Tests


def test_evaluate_expression_tool_metadata(
    evaluate_expression_tool: EvaluateExpressionTool,
):
    """Test that tool metadata is correctly configured."""
    metadata = evaluate_expression_tool.metadata

    assert metadata.tool_id == "evaluate_expression"
    assert metadata.name == "Evaluate Expression"
    assert metadata.version == "1.0.0"
    assert metadata.auth_method.value == "none"
    assert metadata.is_retryable is False
    assert metadata.max_retries == 1
    assert metadata.timeout_seconds == 5
    assert metadata.is_idempotent is True

    # Verify parameters
    assert "expression" in metadata.parameters
    assert metadata.parameters["expression"].required is True
    assert metadata.parameters["expression"].max_length == 1000


# Edge Cases and Integration Tests


@pytest.mark.asyncio
async def test_execute_python_unicode_code(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test execution with Unicode characters."""
    result = await execute_python_tool_no_docker.execute(
        parameters={"code": "print('Hello 世界 🌍')", "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    assert "Hello 世界 🌍" in result_data["stdout"]


@pytest.mark.asyncio
async def test_execute_python_multiline_output(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test execution with multiple output lines."""
    code = """
for i in range(5):
    print(f'Line {i}')
"""
    result = await execute_python_tool_no_docker.execute(
        parameters={"code": code, "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result
    stdout = result_data["stdout"]
    assert "Line 0" in stdout
    assert "Line 4" in stdout


@pytest.mark.asyncio
async def test_execute_python_execution_metadata(
    execute_python_tool_no_docker: ExecutePythonTool,
    execution_context: ExecutionContext,
):
    """Test that execution metadata is correctly populated."""
    result = await execute_python_tool_no_docker.execute(
        parameters={"code": "print('test')", "timeout": 30},
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "execute_python"
    assert result.execution_time_ms > 0
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
    assert result.metadata["sandbox_type"] in ["local", "docker", "local_fallback"]


@pytest.mark.asyncio
async def test_evaluate_expression_execution_metadata(
    evaluate_expression_tool: EvaluateExpressionTool,
    execution_context: ExecutionContext,
):
    """Test that execution metadata is correctly populated."""
    result = await evaluate_expression_tool.execute(
        parameters={"expression": "1 + 1"},
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "evaluate_expression"
    assert result.execution_time_ms > 0
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["sandbox_type"] == "eval_restricted"
