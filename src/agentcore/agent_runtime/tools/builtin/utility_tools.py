"""Native Tool ABC implementations for utility tools.

This module provides Tool ABC implementations for basic utility operations
like calculator, time, and echo. These are native implementations that
directly inherit from Tool ABC (not legacy function-based tools).

Migration from: agent_runtime/tools/utility_tools.py
Status: Stage 3 - Native Migration
"""

import time
from datetime import UTC, datetime
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


class CalculatorTool(Tool):
    """Calculator tool for basic arithmetic operations.

    Performs basic arithmetic operations: addition, subtraction, multiplication,
    division, modulo, and exponentiation.
    """

    def __init__(self):
        """Initialize calculator tool with metadata."""
        metadata = ToolDefinition(
            tool_id="calculator",
            name="Calculator",
            description="Perform basic arithmetic operations (+, -, *, /, %, **)",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "operation": ToolParameter(
                    name="operation",
                    type="string",
                    description="Arithmetic operation to perform",
                    required=True,
                    enum=["+", "-", "*", "/", "%", "**", "^"],
                ),
                "a": ToolParameter(
                    name="a",
                    type="number",
                    description="First operand",
                    required=True,
                ),
                "b": ToolParameter(
                    name="b",
                    type="number",
                    description="Second operand",
                    required=True,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=5,
            tags=["math", "calculator", "arithmetic", "utility"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute arithmetic operation.

        Args:
            parameters: Dictionary with keys:
                - operation: str - Operation symbol (+, -, *, /, %, **, ^)
                - a: float - First operand
                - b: float - Second operand
            context: Execution context

        Returns:
            ToolResult with calculation result
        """
        start_time = time.time()

        try:
            operation = parameters["operation"]
            a = float(parameters["a"])
            b = float(parameters["b"])

            # Define operations
            operations = {
                "+": lambda x, y: x + y,
                "-": lambda x, y: x - y,
                "*": lambda x, y: x * y,
                "/": lambda x, y: x / y,
                "%": lambda x, y: x % y,
                "**": lambda x, y: x**y,
                "^": lambda x, y: x**y,  # Alternative power notation
            }

            # Validate operation
            if operation not in operations:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error=f"Invalid operation: {operation}. Supported: {', '.join(operations.keys())}",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Check for division/modulo by zero
            if operation == "/" and b == 0:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Division by zero is not allowed",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            if operation == "%" and b == 0:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Modulo by zero is not allowed",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Perform calculation
            result = operations[operation](a, b)
            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "calculation_completed",
                operation=operation,
                a=a,
                b=b,
                result=result,
            )

            return ToolResult(
                    request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result={
                    "result": result,
                    "operation": operation,
                    "operands": {"a": a, "b": b},
                    "expression": f"{a} {operation} {b} = {result}",
                },
                error=None,
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "calculation_error",
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


class GetCurrentTimeTool(Tool):
    """Get current time tool with optional timezone and formatting.

    Returns the current time in various formats. Currently only supports UTC timezone.
    """

    def __init__(self):
        """Initialize get current time tool with metadata."""
        metadata = ToolDefinition(
            tool_id="get_current_time",
            name="Get Current Time",
            description="Get current time with optional timezone and formatting",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "timezone": ToolParameter(
                    name="timezone",
                    type="string",
                    description="Timezone name (currently only UTC supported)",
                    required=False,
                    default="UTC",
                ),
                "format": ToolParameter(
                    name="format",
                    type="string",
                    description="Output format",
                    required=False,
                    default="iso",
                    enum=["iso", "unix", "human"],
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=5,
            tags=["time", "datetime", "clock", "utility"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Get current time.

        Args:
            parameters: Dictionary with keys:
                - timezone: str - Timezone name (default: "UTC")
                - format: str - Output format: "iso", "unix", or "human" (default: "iso")
            context: Execution context

        Returns:
            ToolResult with current time information
        """
        start_time = time.time()

        try:
            timezone = parameters.get("timezone", "UTC")
            format_type = parameters.get("format", "iso")

            # Validate timezone (only UTC supported for now)
            if timezone.upper() != "UTC":
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error=f"Timezone '{timezone}' not supported. Currently only UTC is supported.",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Get current time
            now = datetime.now(UTC)

            # Format time
            formats = {
                "iso": now.isoformat(),
                "unix": now.timestamp(),
                "human": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            }

            if format_type not in formats:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error=f"Invalid format: {format_type}. Supported: {', '.join(formats.keys())}",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.debug(
                "time_retrieved",
                timezone=timezone,
                format=format_type,
            )

            return ToolResult(
                    request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result={
                    "current_time": formats[format_type],
                    "timezone": timezone,
                    "format": format_type,
                    "utc_timestamp": now.timestamp(),
                    "iso_string": now.isoformat(),
                },
                error=None,
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "time_retrieval_error",
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


class EchoTool(Tool):
    """Echo tool that returns the input message with optional transformations.

    Useful for testing and debugging tool execution pipelines.
    """

    def __init__(self):
        """Initialize echo tool with metadata."""
        metadata = ToolDefinition(
            tool_id="echo",
            name="Echo",
            description="Echo back the input message with optional transformations",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "message": ToolParameter(
                    name="message",
                    type="string",
                    description="Message to echo",
                    required=True,
                ),
                "uppercase": ToolParameter(
                    name="uppercase",
                    type="boolean",
                    description="Convert to uppercase",
                    required=False,
                    default=False,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=1,  # Echo doesn't really need retries
            timeout_seconds=5,
            tags=["echo", "test", "debug", "utility"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Echo the input message.

        Args:
            parameters: Dictionary with keys:
                - message: str - Message to echo
                - uppercase: bool - Convert to uppercase (default: False)
            context: Execution context

        Returns:
            ToolResult with echoed message and metadata
        """
        start_time = time.time()

        try:
            message = parameters["message"]
            uppercase = parameters.get("uppercase", False)

            # Transform message
            output_message = message.upper() if uppercase else message

            # Handle word count correctly for empty strings
            word_count = len(message.split()) if message.strip() else 0

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.debug(
                "echo_executed",
                message_length=len(message),
                uppercase=uppercase,
            )

            return ToolResult(
                    request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result={
                    "echo": output_message,
                    "original": message,
                    "length": len(message),
                    "uppercase": uppercase,
                    "word_count": word_count,
                },
                error=None,
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "echo_error",
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
