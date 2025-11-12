"""Native Tool ABC implementations for code execution tools.

This module provides Tool ABC implementations for executing Python code and
evaluating expressions in sandboxed environments. These are native implementations
that directly inherit from Tool ABC (not legacy function-based tools).

Migration from: agent_runtime/tools/code_execution_tools.py
Status: Stage 3 - Native Migration
"""

import asyncio
import sys
import time
import traceback
from io import StringIO
from typing import Any

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
    """Python code execution tool for running code in sandboxed environment.

    Executes Python code with restricted builtins for basic sandboxing.
    In production, should use more sophisticated sandboxing mechanisms.
    """

    def __init__(self):
        """Initialize Python execution tool with metadata."""
        metadata = ToolDefinition(
            tool_id="execute_python",
            name="Execute Python",
            description="Execute Python code in a sandboxed environment",
            version="1.0.0",
            category=ToolCategory.CODE_EXECUTION,
            parameters={
                "code": ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True,
                    min_length=1,
                    max_length=10000,
                ),
                "timeout": ToolParameter(
                    name="timeout",
                    type="number",
                    description="Execution timeout in seconds",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=60,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=False,  # Code execution should not retry
            max_retries=1,
            timeout_seconds=65,  # Slightly longer than max execution timeout
            is_idempotent=False,  # May have side effects
            capabilities=["code_execution", "python", "sandboxed"],
            tags=["python", "code", "execution", "sandbox"],
            requirements=["python3"],
            security_requirements=["sandbox_isolation"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute Python code.

        Args:
            parameters: Dictionary with keys:
                - code: str - Python code to execute
                - timeout: int - Execution timeout in seconds (default: 10)
            context: Execution context

        Returns:
            ToolResult with execution results (stdout, stderr, result)
        """
        start_time = time.time()

        try:
            code = parameters["code"]
            timeout = int(parameters.get("timeout", 10))

            # Validate parameters
            if not code or not code.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Code cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            if timeout < 1 or timeout > 60:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Timeout must be between 1 and 60 seconds",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            self.logger.info(
                "python_execution_executing",
                code_length=len(code),
                timeout=timeout,
            )

            # Capture stdout and stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            # Save original stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            result_data = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": None,
            }

            try:
                # Redirect stdout/stderr
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Create restricted globals (basic sandboxing)
                # In production, would use more sophisticated sandboxing
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
                        "map": map,
                        "filter": filter,
                        "any": any,
                        "all": all,
                    }
                }

                # Execute code with timeout
                async def run_code():
                    # Use exec to run code
                    exec(code, safe_globals)
                    return safe_globals.get("result", None)

                try:
                    exec_result = await asyncio.wait_for(run_code(), timeout=timeout)
                    result_data["success"] = True
                    result_data["result"] = str(exec_result) if exec_result is not None else None
                except asyncio.TimeoutError:
                    result_data["error"] = f"Execution timed out after {timeout} seconds"
                    self.logger.warning("python_execution_timeout", timeout=timeout)
                except Exception as e:
                    result_data["error"] = f"{type(e).__name__}: {str(e)}"
                    result_data["stderr"] = traceback.format_exc()
                    self.logger.error("python_execution_error", error=str(e))

            finally:
                # Restore original stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                # Capture output
                result_data["stdout"] = stdout_capture.getvalue()
                if not result_data["stderr"]:
                    result_data["stderr"] = stderr_capture.getvalue()

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "python_execution_completed",
                success=result_data["success"],
                code_length=len(code),
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
                    "sandbox_type": "restricted_globals",
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "python_execution_error",
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
