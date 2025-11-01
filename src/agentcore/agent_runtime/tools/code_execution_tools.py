"""Code execution tool adapters for running code in sandboxed environments."""

import asyncio
import sys
import traceback
from io import StringIO
from typing import Any

import structlog

from ..models.tool_integration import (
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from ..services.tool_registry import ToolRegistry

logger = structlog.get_logger()


async def execute_python(code: str, timeout: int = 10) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with execution result
    """
    logger.info(
        "python_execution_called",
        code_length=len(code),
        timeout=timeout,
    )

    # Capture stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    result = {
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
            result["success"] = True
            result["result"] = str(exec_result) if exec_result is not None else None
        except asyncio.TimeoutError:
            result["error"] = f"Execution timed out after {timeout} seconds"
            logger.warning("python_execution_timeout", timeout=timeout)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["stderr"] = traceback.format_exc()
            logger.error("python_execution_error", error=str(e))

    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Capture output
        result["stdout"] = stdout_capture.getvalue()
        if not result["stderr"]:
            result["stderr"] = stderr_capture.getvalue()

    return result


async def evaluate_expression(expression: str) -> dict[str, Any]:
    """
    Evaluate a Python expression and return the result.

    Args:
        expression: Python expression to evaluate

    Returns:
        Dictionary with evaluation result
    """
    logger.info(
        "expression_evaluation_called",
        expression_length=len(expression),
    )

    result = {
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
        result["success"] = True
        result["result"] = eval_result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error("expression_evaluation_error", error=str(e))

    return result


async def run_shell_command(command: str, timeout: int = 30) -> dict[str, Any]:
    """
    Run a shell command in a controlled environment.

    SECURITY WARNING: This tool should only be enabled in trusted environments.

    Args:
        command: Shell command to execute
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with command output
    """
    logger.warning(
        "shell_command_called",
        command=command[:100],  # Log only first 100 chars
        timeout=timeout,
    )

    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "return_code": None,
        "error": None,
    }

    try:
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            result["success"] = process.returncode == 0
            result["return_code"] = process.returncode
            result["stdout"] = stdout.decode("utf-8", errors="replace")
            result["stderr"] = stderr.decode("utf-8", errors="replace")

        except asyncio.TimeoutError:
            # Kill process if timeout
            process.kill()
            await process.wait()
            result["error"] = f"Command timed out after {timeout} seconds"
            logger.warning("shell_command_timeout", timeout=timeout)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error("shell_command_error", error=str(e))

    return result


def register_code_execution_tools(registry: ToolRegistry) -> None:
    """
    Register code execution tools with the tool registry.

    Args:
        registry: ToolRegistry instance
    """
    # Python Execution tool
    python_exec_def = ToolDefinition(
        tool_id="execute_python",
        name="execute_python",
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
        timeout_seconds=65,  # Slightly longer than max execution timeout
        is_retryable=False,  # Code execution should not retry
        is_idempotent=False,  # May have side effects
        capabilities=["code_execution", "python", "sandboxed"],
        tags=["python", "code", "execution", "sandbox"],
        requirements=["python3"],
        security_requirements=["sandbox_isolation"],
        metadata={
            "language": "python",
            "sandbox_type": "restricted_globals",
            "warning": "Limited built-in functions available",
        },
    )
    registry.register_tool(python_exec_def, execute_python)

    # Expression Evaluation tool
    eval_expr_def = ToolDefinition(
        tool_id="evaluate_expression",
        name="evaluate_expression",
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
        timeout_seconds=5,
        is_retryable=False,
        is_idempotent=True,  # Pure expression evaluation
        capabilities=["expression_evaluation", "python"],
        tags=["python", "expression", "eval", "calculator"],
        metadata={
            "language": "python",
            "sandbox_type": "eval_restricted",
        },
    )
    registry.register_tool(eval_expr_def, evaluate_expression)

    # Shell Command tool (disabled by default for security)
    # Uncomment to enable in trusted environments only
    # shell_cmd_def = ToolDefinition(
    #     tool_id="run_shell_command",
    #     name="run_shell_command",
    #     description="Run a shell command (SECURITY WARNING: Use in trusted environments only)",
    #     version="1.0.0",
    #     category=ToolCategory.CODE_EXECUTION,
    #     parameters={
    #         "command": ToolParameter(
    #             name="command",
    #             type="string",
    #             description="Shell command to execute",
    #             required=True,
    #             min_length=1,
    #             max_length=1000,
    #         ),
    #         "timeout": ToolParameter(
    #             name="timeout",
    #             type="number",
    #             description="Execution timeout in seconds",
    #             required=False,
    #             default=30,
    #             min_value=1,
    #             max_value=300,
    #         ),
    #     },
    #     timeout_seconds=305,
    #     is_retryable=False,
    #     is_idempotent=False,
    #     capabilities=["shell_execution", "system_commands"],
    #     tags=["shell", "command", "system", "bash"],
    #     requirements=["shell"],
    #     security_requirements=["trusted_environment", "command_validation"],
    #     metadata={
    #         "warning": "SECURITY RISK: Only enable in trusted environments",
    #     },
    # )
    # registry.register_tool(shell_cmd_def, run_shell_command)

    logger.info(
        "code_execution_tools_registered",
        tools=["execute_python", "evaluate_expression"],
    )
