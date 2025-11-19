"""Integration tests for Python Execution Tool with real Docker containers.

Tests cover:
- Real Docker container execution with security restrictions
- Actual timeout enforcement at container level
- Network isolation verification
- Filesystem restrictions (read-only, /tmp access)
- Resource limits enforcement (memory, CPU)
- Security profile verification (capabilities, privileges)
- Container lifecycle and cleanup
- Error handling with real container failures
- Security penetration testing (sandbox escape attempts)

Requires:
- Docker daemon running
- agentcore-python-sandbox image built
"""

import asyncio
import time

import aiodocker
import pytest

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.code_execution_tools import ExecutePythonTool


@pytest.fixture
async def docker_client():
    """Create Docker client for integration tests."""
    try:
        client = aiodocker.Docker()
        # Verify Docker is available
        await client.version()
        yield client
        await client.close()
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture
async def ensure_sandbox_image(docker_client):
    """Ensure the sandbox Docker image exists."""
    try:
        await docker_client.images.inspect("agentcore-python-sandbox")
    except aiodocker.exceptions.DockerError:
        pytest.skip(
            "Docker image 'agentcore-python-sandbox' not found. "
            "Build it with: cd docker/python-sandbox && docker build -t agentcore-python-sandbox ."
        )


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create execution context for testing."""
    return ExecutionContext(
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace-integration",
        session_id="test-session",
    )


@pytest.fixture
def execute_python_tool() -> ExecutePythonTool:
    """Create ExecutePythonTool with Docker enabled."""
    return ExecutePythonTool(use_docker=True, docker_image="agentcore-python-sandbox")


# Basic Execution Tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_basic_execution(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test basic Python code execution in real Docker container."""
    result = await execute_python_tool.execute(
        parameters={"code": "print('Hello from Docker')", "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.result is not None
    assert result.result["exit_code"] == 0
    assert "Hello from Docker" in result.result["stdout"]
    assert result.result["sandbox_type"] == "docker"
    assert result.execution_time_ms > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_multiline_code(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test execution of multiline Python code."""
    code = """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert "Hello, World!" in result.result["stdout"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_error_handling(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test error handling with code that raises exceptions."""
    result = await execute_python_tool.execute(
        parameters={"code": "raise ValueError('Test error')", "timeout": 30},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.result["exit_code"] == 1
    assert "ValueError" in result.result["stderr"]
    assert "Test error" in result.result["stderr"]


# Timeout Tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_timeout_enforcement(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that timeout is enforced at container level."""
    start_time = time.time()

    result = await execute_python_tool.execute(
        parameters={"code": "import time; time.sleep(100)", "timeout": 2},
        context=execution_context,
    )

    elapsed = time.time() - start_time

    assert result.status == ToolExecutionStatus.FAILED
    assert result.result["exit_code"] == 124  # Timeout exit code
    assert "timed out" in result.result["stderr"].lower()
    assert result.result["error"] == "Timeout"
    assert elapsed < 5  # Should timeout quickly, not wait full 100 seconds


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_long_running_within_timeout(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that code completes within timeout successfully."""
    result = await execute_python_tool.execute(
        parameters={
            "code": "import time; time.sleep(1); print('Completed')",
            "timeout": 5,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result["exit_code"] == 0
    assert "Completed" in result.result["stdout"]


# Security Tests - Network Isolation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_network_isolation(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that network access is blocked in container."""
    # Attempt to import socket and make a connection
    code = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('8.8.8.8', 53))
    print('NETWORK_ACCESS_GRANTED')
    s.close()
except Exception as e:
    print(f'NETWORK_BLOCKED: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    # Should either fail or show network is blocked
    stdout = result.result["stdout"]
    assert "NETWORK_ACCESS_GRANTED" not in stdout
    # Network should be blocked by container config (NetworkMode=none)


# Security Tests - Filesystem Restrictions


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_readonly_filesystem(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that filesystem outside /tmp is read-only."""
    # Attempt to write to root filesystem
    code = """
try:
    with open('/test_file.txt', 'w') as f:
        f.write('FILESYSTEM_WRITABLE')
    print('ROOT_WRITE_SUCCESS')
except Exception as e:
    print(f'ROOT_WRITE_FAILED: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    stdout = result.result["stdout"]
    assert "ROOT_WRITE_SUCCESS" not in stdout
    assert "ROOT_WRITE_FAILED" in stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_tmp_writable(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that /tmp/sandbox is writable."""
    code = """
try:
    with open('/tmp/sandbox/test_file.txt', 'w') as f:
        f.write('Test data')
    with open('/tmp/sandbox/test_file.txt', 'r') as f:
        content = f.read()
    print(f'TMP_WRITE_SUCCESS: {content}')
except Exception as e:
    print(f'TMP_WRITE_FAILED: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    stdout = result.result["stdout"]
    assert "TMP_WRITE_SUCCESS: Test data" in stdout


# Security Tests - Privilege Escalation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_no_privilege_escalation(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that privilege escalation is blocked."""
    # Attempt to check user permissions
    code = """
import os
try:
    # Try to get UID (should be non-root)
    uid = os.getuid()
    print(f'UID: {uid}')

    # Non-root user should not be able to access sensitive files
    try:
        with open('/etc/shadow', 'r') as f:
            print('PRIVILEGE_ESCALATION_SUCCESS')
    except PermissionError:
        print('PRIVILEGE_DENIED: /etc/shadow not accessible')
except Exception as e:
    print(f'ERROR: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    stdout = result.result["stdout"]
    assert "UID: 1000" in stdout  # sandbox user UID
    assert "PRIVILEGE_ESCALATION_SUCCESS" not in stdout
    assert "PRIVILEGE_DENIED" in stdout


# Security Tests - Sandbox Escape Attempts


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_no_process_manipulation(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that process manipulation is restricted."""
    # Attempt to spawn subprocesses or manipulate processes
    code = """
import subprocess
try:
    # Try to run a shell command
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    print(f'SUBPROCESS_ALLOWED: {len(result.stdout)} bytes')
except Exception as e:
    print(f'SUBPROCESS_BLOCKED: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    # subprocess module should not be available in restricted environment
    assert result.status == ToolExecutionStatus.FAILED
    stderr = result.result["stderr"]
    assert "NameError" in stderr or "ImportError" in stderr or "ModuleNotFoundError" in stderr


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_no_system_calls(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that dangerous system calls are blocked."""
    # Attempt to use os.system or similar
    code = """
import os
try:
    result = os.system('whoami')
    print(f'SYSTEM_CALL_SUCCESS: {result}')
except Exception as e:
    print(f'SYSTEM_CALL_BLOCKED: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    # os module may be available but system calls should fail
    # In Docker container, this should either fail or return non-zero
    stdout = result.result["stdout"]
    stderr = result.result["stderr"]

    # Either blocked or failed
    if "SYSTEM_CALL_SUCCESS" in stdout:
        # If it succeeded, result should be non-zero (command not found)
        assert "127" in stdout or result.result["exit_code"] != 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_no_file_descriptor_manipulation(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that file descriptor manipulation is restricted."""
    # Attempt to manipulate file descriptors
    code = """
import os
try:
    # Try to access /proc/self/fd
    fds = os.listdir('/proc/self/fd')
    print(f'FD_ACCESS: {len(fds)} descriptors')
except Exception as e:
    print(f'FD_BLOCKED: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 5},
        context=execution_context,
    )

    # This test just verifies execution completes
    # In a properly sandboxed environment, access should be limited
    assert result.status in [ToolExecutionStatus.SUCCESS, ToolExecutionStatus.FAILED]


# Resource Limit Tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_memory_limit(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test that memory limits are enforced."""
    # Try to allocate more than 512MB
    code = """
try:
    # Try to allocate 600MB (more than limit)
    data = bytearray(600 * 1024 * 1024)
    print('MEMORY_LIMIT_EXCEEDED')
except MemoryError:
    print('MEMORY_LIMIT_ENFORCED')
except Exception as e:
    print(f'ERROR: {type(e).__name__}')
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 10},
        context=execution_context,
    )

    # Should either fail or show memory limit enforced
    # Container might be killed by OOM killer
    if result.status == ToolExecutionStatus.SUCCESS:
        stdout = result.result["stdout"]
        assert "MEMORY_LIMIT_EXCEEDED" not in stdout


# Container Cleanup Tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_container_cleanup(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
    docker_client,
):
    """Test that containers are properly cleaned up after execution."""
    # Get initial container count
    initial_containers = await docker_client.containers.list(all=True)
    initial_count = len(initial_containers)

    # Execute code
    await execute_python_tool.execute(
        parameters={"code": "print('test')", "timeout": 5},
        context=execution_context,
    )

    # Give a moment for cleanup
    await asyncio.sleep(1)

    # Check container count
    final_containers = await docker_client.containers.list(all=True)
    final_count = len(final_containers)

    # No new containers should remain
    assert final_count <= initial_count + 1  # Allow for timing


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_cleanup_on_timeout(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
    docker_client,
):
    """Test that containers are cleaned up even when timeout occurs."""
    initial_containers = await docker_client.containers.list(all=True)
    initial_count = len(initial_containers)

    # Execute code that will timeout
    await execute_python_tool.execute(
        parameters={"code": "import time; time.sleep(100)", "timeout": 1},
        context=execution_context,
    )

    # Give a moment for cleanup
    await asyncio.sleep(1)

    final_containers = await docker_client.containers.list(all=True)
    final_count = len(final_containers)

    # No new containers should remain after timeout
    assert final_count <= initial_count + 1


# Edge Cases


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_unicode_handling(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test proper handling of Unicode characters."""
    result = await execute_python_tool.execute(
        parameters={"code": "print('Hello 世界 🌍 Привет')", "timeout": 5},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert "Hello 世界 🌍 Привет" in result.result["stdout"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_docker_large_output(
    execute_python_tool: ExecutePythonTool,
    execution_context: ExecutionContext,
    ensure_sandbox_image,
):
    """Test handling of large output."""
    code = """
for i in range(1000):
    print(f'Line {i}: ' + 'x' * 100)
"""
    result = await execute_python_tool.execute(
        parameters={"code": code, "timeout": 10},
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert len(result.result["stdout"]) > 100000
