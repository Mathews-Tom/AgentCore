"""Base tool interface and abstract classes for tool integration framework.

This module defines the core Tool abstract base class that all tools must implement,
along with execution context and validation utilities. Following the Tool Interface
specification from docs/specs/tool-integration/spec.md (FR-2).
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import structlog

from ..models.tool_integration import (
    ToolDefinition,
    ToolResult,
)

logger = structlog.get_logger()


class ExecutionContext:
    """Context information for tool execution.

    Provides execution environment details including user identity, agent identity,
    distributed tracing information, and execution options.

    Attributes:
        user_id: User initiating the tool execution
        agent_id: Agent requesting tool execution
        trace_id: Distributed tracing identifier for observability
        session_id: Session identifier for grouping related executions
        request_id: Unique identifier for this specific execution request
        metadata: Additional context-specific metadata
    """

    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.trace_id = trace_id or str(uuid4())
        self.session_id = session_id
        self.request_id = request_id or str(uuid4())
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary representation."""
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """Abstract base class for all tools in the AgentCore tool framework.

    All tool implementations must inherit from this class and implement the execute()
    method. The Tool interface provides standardized parameter validation, error
    handling, and result formatting.

    Implements specification FR-2: Tool Interface from docs/specs/tool-integration/spec.md

    Attributes:
        metadata: ToolDefinition containing tool configuration, parameters, auth, etc.

    Example:
        ```python
        class GoogleSearchTool(Tool):
            def __init__(self):
                metadata = ToolDefinition(
                    tool_id="google_search",
                    name="Google Search",
                    description="Search the web using Google",
                    category=ToolCategory.SEARCH,
                    parameters={
                        "query": ToolParameter(
                            name="query",
                            type="string",
                            description="Search query",
                            required=True
                        )
                    }
                )
                super().__init__(metadata)

            async def execute(
                self,
                parameters: dict[str, Any],
                context: ExecutionContext
            ) -> ToolResult:
                # Implementation here
                pass
        ```
    """

    def __init__(self, metadata: ToolDefinition):
        """Initialize tool with metadata definition.

        Args:
            metadata: ToolDefinition containing tool configuration
        """
        self.metadata = metadata
        self.logger = logger.bind(
            tool_id=metadata.tool_id,
            tool_name=metadata.name,
        )

    @abstractmethod
    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute the tool with given parameters and context.

        This is the core method that all tools must implement. It receives validated
        parameters and execution context, performs the tool's functionality, and
        returns a standardized ToolResult.

        Implementations should:
        - Perform the tool's core functionality
        - Handle errors gracefully and return them in ToolResult
        - Track execution time
        - Log important events using self.logger
        - Not raise exceptions (return errors in ToolResult instead)

        Args:
            parameters: Dictionary of validated parameter values
            context: Execution context with user_id, agent_id, trace_id, etc.

        Returns:
            ToolResult containing success status, result data, error information,
            and execution metadata

        Example:
            ```python
            async def execute(
                self,
                parameters: dict[str, Any],
                context: ExecutionContext
            ) -> ToolResult:
                start_time = time.time()

                try:
                    query = parameters["query"]
                    results = await self._perform_search(query)

                    execution_time_ms = (time.time() - start_time) * 1000

                    return ToolResult(
                        request_id=context.request_id,
                        tool_id=self.metadata.tool_id,
                        status=ToolExecutionStatus.SUCCESS,
                        result=results,
                        execution_time_ms=execution_time_ms
                    )
                except Exception as e:
                    execution_time_ms = (time.time() - start_time) * 1000

                    return ToolResult(
                        request_id=context.request_id,
                        tool_id=self.metadata.tool_id,
                        status=ToolExecutionStatus.FAILED,
                        error=str(e),
                        error_type=type(e).__name__,
                        execution_time_ms=execution_time_ms
                    )
            ```
        """

    async def validate_parameters(
        self,
        parameters: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate parameters against tool's parameter definitions.

        Enhanced validation implementing TOOL-007 Parameter Validation Framework.

        Checks that:
        - All required parameters are present
        - Parameter types match expected types (strict type checking)
        - Values are within allowed ranges/enums
        - String lengths are within bounds
        - Numbers are within min/max values
        - Strings match regex patterns (if defined)
        - Complex types (objects, arrays) are properly structured

        Implements specification FR-2.2: Parameter validation before execution.

        Args:
            parameters: Dictionary of parameter values to validate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
            If invalid, error_message contains detailed validation error with
            parameter name, expected type/format, and received value.

        Example:
            ```python
            is_valid, error = await tool.validate_parameters({"query": "test"})
            if not is_valid:
                return ToolResult(
                    status=ToolExecutionStatus.FAILED,
                    error=error
                )
            ```
        """
        # Check required parameters
        for param_name, param_def in self.metadata.parameters.items():
            if param_def.required and param_name not in parameters:
                error_msg = (
                    f"Missing required parameter: '{param_name}' "
                    f"(type: {param_def.type}, description: {param_def.description})"
                )
                self.logger.warning(
                    "parameter_validation_failed",
                    error=error_msg,
                    parameter=param_name,
                    expected_type=param_def.type,
                )
                return False, error_msg

        # Type and constraint validation
        for param_name, value in parameters.items():
            if param_name not in self.metadata.parameters:
                # Allow extra parameters (may be used by specific implementations)
                self.logger.debug(
                    "extra_parameter_provided",
                    parameter=param_name,
                    value_type=type(value).__name__,
                )
                continue

            param_def = self.metadata.parameters[param_name]

            # Strict type checking (TOOL-007 enhancement)
            type_valid, type_error = self._validate_type(param_name, value, param_def)
            if not type_valid:
                self.logger.warning(
                    "parameter_type_validation_failed",
                    error=type_error,
                    parameter=param_name,
                    expected_type=param_def.type,
                    actual_type=type(value).__name__,
                )
                return False, type_error

            # Enum validation
            if param_def.enum and value not in param_def.enum:
                error_msg = (
                    f"Parameter '{param_name}' must be one of {param_def.enum}, "
                    f"got: {value!r} (type: {type(value).__name__})"
                )
                self.logger.warning(
                    "parameter_validation_failed",
                    error=error_msg,
                    parameter=param_name,
                    value=value,
                    allowed_values=param_def.enum,
                )
                return False, error_msg

            # String-specific validation
            if param_def.type == "string" and isinstance(value, str):
                # Length validation
                if param_def.min_length and len(value) < param_def.min_length:
                    error_msg = (
                        f"Parameter '{param_name}' must be at least "
                        f"{param_def.min_length} characters, got {len(value)} characters"
                    )
                    return False, error_msg

                if param_def.max_length and len(value) > param_def.max_length:
                    error_msg = (
                        f"Parameter '{param_name}' must be at most "
                        f"{param_def.max_length} characters, got {len(value)} characters"
                    )
                    return False, error_msg

                # Regex pattern validation (TOOL-007 enhancement)
                if param_def.pattern:
                    pattern_valid, pattern_error = self._validate_pattern(
                        param_name, value, param_def.pattern
                    )
                    if not pattern_valid:
                        self.logger.warning(
                            "parameter_pattern_validation_failed",
                            error=pattern_error,
                            parameter=param_name,
                            pattern=param_def.pattern,
                            value=value,
                        )
                        return False, pattern_error

            # Number range validation
            if param_def.type in ("number", "integer") and isinstance(value, (int, float)):
                if param_def.min_value is not None and value < param_def.min_value:
                    error_msg = (
                        f"Parameter '{param_name}' must be >= {param_def.min_value}, "
                        f"got {value}"
                    )
                    return False, error_msg

                if param_def.max_value is not None and value > param_def.max_value:
                    error_msg = (
                        f"Parameter '{param_name}' must be <= {param_def.max_value}, "
                        f"got {value}"
                    )
                    return False, error_msg

            # Array length validation
            if param_def.type == "array" and isinstance(value, list):
                if param_def.min_length and len(value) < param_def.min_length:
                    error_msg = (
                        f"Parameter '{param_name}' must have at least "
                        f"{param_def.min_length} items, got {len(value)} items"
                    )
                    return False, error_msg

                if param_def.max_length and len(value) > param_def.max_length:
                    error_msg = (
                        f"Parameter '{param_name}' must have at most "
                        f"{param_def.max_length} items, got {len(value)} items"
                    )
                    return False, error_msg

        self.logger.debug(
            "parameter_validation_passed",
            parameter_count=len(parameters),
            tool_id=self.metadata.tool_id,
        )
        return True, None

    def _validate_type(
        self,
        param_name: str,
        value: Any,
        param_def: Any,
    ) -> tuple[bool, str | None]:
        """Validate parameter type matches expected type.

        TOOL-007 enhancement: Strict type checking with comprehensive error messages.

        Args:
            param_name: Name of the parameter being validated
            value: Actual value provided
            param_def: Parameter definition with expected type

        Returns:
            Tuple of (is_valid, error_message)
        """
        expected_type = param_def.type
        actual_type = type(value).__name__

        type_mapping: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type not in type_mapping:
            # Unknown type, skip validation
            return True, None

        expected_python_type = type_mapping[expected_type]

        # Special case: integers can be provided as floats if they're whole numbers
        if expected_type == "integer" and isinstance(value, float):
            if value.is_integer():
                # Allow whole number floats for integers
                return True, None
            else:
                # Non-integer float for integer parameter - specific error
                error_msg = (
                    f"Parameter '{param_name}' must be an integer, "
                    f"got float: {value}"
                )
                return False, error_msg

        if not isinstance(value, expected_python_type):
            error_msg = (
                f"Parameter '{param_name}' has incorrect type: "
                f"expected {expected_type}, got {actual_type} "
                f"(value: {value!r})"
            )
            return False, error_msg

        return True, None

    def _validate_pattern(
        self,
        param_name: str,
        value: str,
        pattern: str,
    ) -> tuple[bool, str | None]:
        """Validate string matches regex pattern.

        TOOL-007 enhancement: Pattern validation for strings.

        Args:
            param_name: Name of the parameter being validated
            value: String value to validate
            pattern: Regex pattern to match

        Returns:
            Tuple of (is_valid, error_message)
        """
        import re

        try:
            if not re.match(pattern, value):
                error_msg = (
                    f"Parameter '{param_name}' does not match required pattern. "
                    f"Pattern: '{pattern}', Value: '{value}'"
                )
                return False, error_msg
        except re.error as e:
            error_msg = (
                f"Invalid regex pattern for parameter '{param_name}': {pattern}. "
                f"Error: {e}"
            )
            return False, error_msg

        return True, None

    def __repr__(self) -> str:
        """String representation of tool."""
        return (
            f"<{self.__class__.__name__}("
            f"tool_id={self.metadata.tool_id}, "
            f"name={self.metadata.name}, "
            f"category={self.metadata.category}"
            f")>"
        )
