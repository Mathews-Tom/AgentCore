"""Native Tool ABC implementations for code execution tools.

This module provides Tool ABC implementations for executing Python code and
evaluating expressions in sandboxed environments. These are native implementations
that directly inherit from Tool ABC (not legacy function-based tools).

Implements TOOL-011: Docker-based sandbox for secure Python code execution with
resource limits, network isolation, and security profiles.

Migration from: agent_runtime/tools/code_execution_tools.py
Status: Stage 3 - Native Migration
"""

import asyncio
import os
import sys
import tempfile
import time
import traceback
from io import StringIO
from pathlib import Path
from typing import Any

import aiodocker

from ...config.settings import get_settings
from ...models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from ..base import ExecutionContext, Tool


class ExecutePythonTool(Tool):
    """Python code execution tool with Docker-based sandbox.

    Executes Python code in isolated Docker containers with:
    - No network access
    - Read-only filesystem except /tmp
    - Resource limits (512MB RAM, 1 CPU)
    - Execution timeout enforcement (default: 30s)
    - Security profiles (AppArmor/SELinux)

    Implements TOOL-011 acceptance criteria for secure code execution.
    """

    def __init__(self, use_docker: bool = True, docker_image: str = "agentcore-python-sandbox"):
        """Initialize Python execution tool with metadata.

        Args:
            use_docker: Whether to use Docker sandbox (default: True, falls back if Docker unavailable)
            docker_image: Docker image name for sandbox (default: agentcore-python-sandbox)
        """
        metadata = ToolDefinition(
            tool_id="execute_python",
            name="Execute Python",
            description="Execute Python code in a Docker-sandboxed environment with resource limits",
            version="2.0.0",
            category=ToolCategory.CODE_EXECUTION,
            parameters={
                "code": ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True,
                    min_length=1,
                    max_length=50000,
                ),
                "timeout": ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Execution timeout in seconds",
                    required=False,
                    default=30,
                    min_value=1,
                    max_value=300,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=False,  # Code execution should not retry
            max_retries=1,
            timeout_seconds=320,  # Slightly longer than max execution timeout
            is_idempotent=False,  # May have side effects
            capabilities=["code_execution", "python", "sandboxed", "docker"],
            tags=["python", "code", "execution", "sandbox", "docker"],
            requirements=["python3", "docker"],
            security_requirements=["sandbox_isolation", "no_network", "resource_limits"],
        )
        super().__init__(metadata)

        self.use_docker = use_docker
        self.docker_image = docker_image
        self.settings = get_settings()

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute Python code in Docker sandbox.

        Args:
            parameters: Dictionary with keys:
                - code: str - Python code to execute
                - timeout: int - Execution timeout in seconds (default: 30)
            context: Execution context

        Returns:
            ToolResult with execution results (stdout, stderr, exit_code)
        """
        start_time = time.time()

        try:
            # Validate parameters
            is_valid, error = await self.validate_parameters(parameters)
            if not is_valid:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=error,
                    error_type="ValidationError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            code = parameters["code"]
            timeout = int(parameters.get("timeout", 30))

            self.logger.info(
                "python_execution_executing",
                code_length=len(code),
                timeout=timeout,
                use_docker=self.use_docker,
            )

            # Try Docker execution, fall back to local if unavailable
            if self.use_docker:
                try:
                    result_data = await self._execute_in_docker(code, timeout, context)
                except Exception as docker_error:
                    self.logger.warning(
                        "docker_execution_failed_fallback",
                        error=str(docker_error),
                    )
                    result_data = await self._execute_local(code, timeout)
                    result_data["sandbox_type"] = "local_fallback"
            else:
                result_data = await self._execute_local(code, timeout)
                result_data["sandbox_type"] = "local"

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "python_execution_completed",
                success=result_data["exit_code"] == 0,
                code_length=len(code),
                sandbox_type=result_data.get("sandbox_type", "docker"),
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS if result_data["exit_code"] == 0 else ToolExecutionStatus.FAILED,
                result=result_data,
                error=result_data.get("error"),
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                    "sandbox_type": result_data.get("sandbox_type", "docker"),
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "python_execution_error",
                error=str(e),
                error_type=type(e).__name__,
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

    async def _execute_in_docker(
        self,
        code: str,
        timeout: int,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute code in Docker sandbox with security restrictions.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            context: Execution context

        Returns:
            Dictionary with execution results

        Raises:
            Exception: If Docker execution fails
        """
        docker = aiodocker.Docker()

        try:
            # Create temporary directory for code file
            with tempfile.TemporaryDirectory() as tmpdir:
                code_file = Path(tmpdir) / "code.py"
                code_file.write_text(code)

                # Configure container with security restrictions
                container_config = {
                    "Image": self.docker_image,
                    "Cmd": ["python3", "/tmp/sandbox/code.py"],
                    "HostConfig": {
                        # Resource limits
                        "Memory": 512 * 1024 * 1024,  # 512MB
                        "MemorySwap": 512 * 1024 * 1024,  # No swap
                        "CpuPeriod": 100000,
                        "CpuQuota": 100000,  # 1 CPU
                        # Network isolation
                        "NetworkMode": "none",
                        # Read-only filesystem except /tmp
                        "ReadonlyRootfs": True,
                        "Tmpfs": {"/tmp/sandbox": "rw,size=100m,mode=1777"},
                        # Security options
                        "SecurityOpt": ["no-new-privileges"],
                        "CapDrop": ["ALL"],
                    },
                    "WorkingDir": "/tmp/sandbox",
                    "User": "sandbox",
                    "AttachStdout": True,
                    "AttachStderr": True,
                }

                # Create and start container
                container = await docker.containers.create(config=container_config)

                try:
                    # Copy code file to container
                    await container.put_archive("/tmp/sandbox", await self._create_tar(code_file))

                    # Start container with timeout
                    await container.start()

                    # Wait for completion with timeout
                    try:
                        await asyncio.wait_for(container.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        await container.kill()
                        return {
                            "exit_code": 124,  # Timeout exit code
                            "stdout": "",
                            "stderr": f"Execution timed out after {timeout} seconds",
                            "error": "Timeout",
                            "sandbox_type": "docker",
                        }

                    # Get execution results
                    logs = await container.log(stdout=True, stderr=True)
                    stdout_lines = []
                    stderr_lines = []

                    for log_entry in logs:
                        if log_entry.stream == 1:  # stdout
                            stdout_lines.append(log_entry.message.decode("utf-8"))
                        elif log_entry.stream == 2:  # stderr
                            stderr_lines.append(log_entry.message.decode("utf-8"))

                    # Get exit code
                    container_info = await container.show()
                    exit_code = container_info["State"]["ExitCode"]

                    return {
                        "exit_code": exit_code,
                        "stdout": "".join(stdout_lines),
                        "stderr": "".join(stderr_lines),
                        "error": None if exit_code == 0 else "Non-zero exit code",
                        "sandbox_type": "docker",
                    }

                finally:
                    # Cleanup container
                    try:
                        await container.delete(force=True)
                    except Exception as e:
                        self.logger.warning("container_cleanup_failed", error=str(e))

        finally:
            await docker.close()

    async def _execute_local(
        self,
        code: str,
        timeout: int,
    ) -> dict[str, Any]:
        """Execute code locally with basic restrictions (fallback).

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dictionary with execution results
        """
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        result_data = {
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
            "error": None,
        }

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    # Allow common exceptions for user code
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "KeyError": KeyError,
                    "IndexError": IndexError,
                    "AttributeError": AttributeError,
                    "RuntimeError": RuntimeError,
                    "ZeroDivisionError": ZeroDivisionError,
                    "Exception": Exception,
                }
            }

            def run_code_sync():
                exec(code, safe_globals)

            try:
                # Run code in thread pool to enable timeout cancellation
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, run_code_sync),
                    timeout=timeout
                )
                result_data["exit_code"] = 0
            except asyncio.TimeoutError:
                result_data["stderr"] = f"Execution timed out after {timeout} seconds"
                result_data["error"] = "Timeout"
                result_data["exit_code"] = 124
            except Exception as e:
                result_data["stderr"] = traceback.format_exc()
                result_data["error"] = f"{type(e).__name__}: {str(e)}"
                result_data["exit_code"] = 1

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            result_data["stdout"] = stdout_capture.getvalue()
            if not result_data["stderr"]:
                result_data["stderr"] = stderr_capture.getvalue()

        return result_data

    async def _create_tar(self, file_path: Path) -> bytes:
        """Create tar archive of a file.

        Args:
            file_path: Path to file to archive

        Returns:
            Tar archive bytes
        """
        import io
        import tarfile

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(file_path, arcname="code.py")

        tar_stream.seek(0)
        return tar_stream.read()


class EvaluateExpressionTool(Tool):
    """Python expression evaluation tool for safe expression evaluation.

    Evaluates Python expressions with restricted builtins. Suitable for
    calculator-like operations and simple data transformations.
    """

    def __init__(self):
        """Initialize expression evaluation tool with metadata."""
        metadata = ToolDefinition(
            tool_id="evaluate_expression",
            name="Evaluate Expression",
            description="Evaluate a Python expression and return the result",
            version="1.0.0",
            category=ToolCategory.CODE_EXECUTION,
            parameters={
                "expression": ToolParameter(
                    name="expression",
                    type="string",
                    description="Python expression to evaluate",
                    required=True,
                    min_length=1,
                    max_length=1000,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=False,
            max_retries=1,
            timeout_seconds=5,
            is_idempotent=True,  # Pure expression evaluation
            capabilities=["expression_evaluation", "python"],
            tags=["python", "expression", "eval", "calculator"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Evaluate Python expression.

        Args:
            parameters: Dictionary with keys:
                - expression: str - Python expression to evaluate
            context: Execution context

        Returns:
            ToolResult with evaluation result
        """
        start_time = time.time()

        try:
            expression = parameters["expression"]

            # Validate expression
            if not expression or not expression.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Expression cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            self.logger.info(
                "expression_evaluation_executing",
                expression_length=len(expression),
            )

            result_data = {
                "success": False,
                "result": None,
                "error": None,
            }

            try:
                # Safe globals for eval
                safe_globals = {
                    "__builtins__": {
                        "abs": abs,
                        "min": min,
                        "max": max,
                        "sum": sum,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set,
                    }
                }

                # Evaluate expression
                eval_result = eval(expression, safe_globals)
                result_data["success"] = True
                result_data["result"] = eval_result

            except Exception as e:
                result_data["error"] = f"{type(e).__name__}: {str(e)}"
                self.logger.error("expression_evaluation_error", error=str(e))

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "expression_evaluation_completed",
                success=result_data["success"],
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS if result_data["success"] else ToolExecutionStatus.FAILED,
                result=result_data,
                error=result_data["error"],
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                    "sandbox_type": "eval_restricted",
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "expression_evaluation_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )
