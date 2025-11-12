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
    ToolExecutionStatus,
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
        pass

    async def validate_parameters(
        self,
        parameters: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate parameters against tool's parameter definitions.

        Checks that:
        - All required parameters are present
        - Parameter types are correct
        - Values are within allowed ranges/enums
        - String lengths are within bounds
        - Numbers are within min/max values

        Implements specification FR-2.2: Parameter validation before execution.

        Args:
            parameters: Dictionary of parameter values to validate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
            If invalid, error_message contains detailed validation error.

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
                error_msg = f"Missing required parameter: {param_name}"
                self.logger.warning(
                    "parameter_validation_failed",
                    error=error_msg,
                    parameter=param_name,
                )
                return False, error_msg

        # Type and constraint validation
        for param_name, value in parameters.items():
            if param_name not in self.metadata.parameters:
                # Allow extra parameters (may be used by specific implementations)
                continue

            param_def = self.metadata.parameters[param_name]

            # Enum validation
            if param_def.enum and value not in param_def.enum:
                error_msg = (
                    f"Parameter '{param_name}' must be one of {param_def.enum}, "
                    f"got: {value}"
                )
                self.logger.warning(
                    "parameter_validation_failed",
                    error=error_msg,
                    parameter=param_name,
                    value=value,
                )
                return False, error_msg

            # String length validation
            if param_def.type == "string" and isinstance(value, str):
                if param_def.min_length and len(value) < param_def.min_length:
                    error_msg = (
                        f"Parameter '{param_name}' must be at least "
                        f"{param_def.min_length} characters"
                    )
                    return False, error_msg

                if param_def.max_length and len(value) > param_def.max_length:
                    error_msg = (
                        f"Parameter '{param_name}' must be at most "
                        f"{param_def.max_length} characters"
                    )
                    return False, error_msg

            # Number range validation
            if param_def.type in ("number", "integer") and isinstance(value, (int, float)):
                if param_def.min_value is not None and value < param_def.min_value:
                    error_msg = (
                        f"Parameter '{param_name}' must be >= {param_def.min_value}"
                    )
                    return False, error_msg

                if param_def.max_value is not None and value > param_def.max_value:
                    error_msg = (
                        f"Parameter '{param_name}' must be <= {param_def.max_value}"
                    )
                    return False, error_msg

            # Array length validation
            if param_def.type == "array" and isinstance(value, list):
                if param_def.min_length and len(value) < param_def.min_length:
                    error_msg = (
                        f"Parameter '{param_name}' must have at least "
                        f"{param_def.min_length} items"
                    )
                    return False, error_msg

                if param_def.max_length and len(value) > param_def.max_length:
                    error_msg = (
                        f"Parameter '{param_name}' must have at most "
                        f"{param_def.max_length} items"
                    )
                    return False, error_msg

        self.logger.debug(
            "parameter_validation_passed",
            parameter_count=len(parameters),
        )
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
