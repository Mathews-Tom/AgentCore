"""Utility tools for basic operations.

Provides essential utility functions like calculator, time, and echo.
"""

from datetime import UTC, datetime
from typing import Any

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from agentcore.agent_runtime.services.tool_registry import ToolRegistry


async def calculator_impl(operation: str, a: float, b: float) -> dict[str, Any]:
    """Perform basic arithmetic operations.

    Args:
        operation: Operation to perform (+, -, *, /, %, **)
        a: First operand
        b: Second operand

    Returns:
        Dictionary with result and operation details

    Raises:
        ValueError: If operation is invalid or division by zero
    """
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "%": lambda x, y: x % y,
        "**": lambda x, y: x**y,
        "^": lambda x, y: x**y,  # Alternative power notation
    }

    if operation not in operations:
        raise ValueError(
            f"Invalid operation: {operation}. "
            f"Supported: {', '.join(operations.keys())}"
        )

    if operation == "/" and b == 0:
        raise ValueError("Division by zero is not allowed")

    if operation == "%" and b == 0:
        raise ValueError("Modulo by zero is not allowed")

    result = operations[operation](a, b)

    return {
        "result": result,
        "operation": operation,
        "operands": {"a": a, "b": b},
        "expression": f"{a} {operation} {b} = {result}",
    }


async def get_current_time_impl(
    timezone: str = "UTC", format: str = "iso"
) -> dict[str, Any]:
    """Get current time with optional timezone and formatting.

    Args:
        timezone: Timezone name (default: UTC). Currently only UTC supported.
        format: Output format - 'iso', 'unix', 'human' (default: iso)

    Returns:
        Dictionary with current time in requested format

    Raises:
        ValueError: If format is invalid
    """
    # For now, only UTC is supported
    # Future enhancement: support pytz/zoneinfo for other timezones
    if timezone.upper() != "UTC":
        raise ValueError(
            f"Timezone '{timezone}' not supported. Currently only UTC is supported."
        )

    now = datetime.now(UTC)

    formats = {
        "iso": now.isoformat(),
        "unix": now.timestamp(),
        "human": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    if format not in formats:
        raise ValueError(
            f"Invalid format: {format}. Supported: {', '.join(formats.keys())}"
        )

    return {
        "current_time": formats[format],
        "timezone": timezone,
        "format": format,
        "utc_timestamp": now.timestamp(),
        "iso_string": now.isoformat(),
    }


async def echo_impl(message: str, uppercase: bool = False) -> dict[str, Any]:
    """Echo back the input message with optional transformations.

    Args:
        message: Message to echo
        uppercase: Convert to uppercase (default: False)

    Returns:
        Dictionary with echoed message and metadata
    """
    output_message = message.upper() if uppercase else message

    # Handle word count correctly for empty strings
    word_count = len(message.split()) if message.strip() else 0

    return {
        "echo": output_message,
        "original": message,
        "length": len(message),
        "uppercase": uppercase,
        "word_count": word_count,
    }


def register_utility_tools(registry: ToolRegistry) -> None:
    """Register all utility tools with the registry.

    Args:
        registry: Tool registry instance
    """
    # Calculator Tool
    calculator_def = ToolDefinition(
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

    # Get Current Time Tool
    get_current_time_def = ToolDefinition(
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

    # Echo Tool
    echo_def = ToolDefinition(
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

    # Register all tools
    registry.register_tool(calculator_def, calculator_impl)
    registry.register_tool(get_current_time_def, get_current_time_impl)
    registry.register_tool(echo_def, echo_impl)
