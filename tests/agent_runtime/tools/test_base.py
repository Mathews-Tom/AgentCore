"""Tests for base tool interface and execution context."""

import time
from typing import Any

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool


class MockTool(Tool):
    """Mock tool implementation for testing base functionality."""

    def __init__(self, should_succeed: bool = True):
        """Initialize mock tool with configurable behavior."""
        metadata = ToolDefinition(
            tool_id="mock_tool",
            name="Mock Tool",
            description="A mock tool for testing",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "required_param": ToolParameter(
                    name="required_param",
                    type="string",
                    description="A required parameter",
                    required=True,
                ),
                "optional_param": ToolParameter(
                    name="optional_param",
                    type="integer",
                    description="An optional parameter",
                    required=False,
                    default=10,
                ),
                "enum_param": ToolParameter(
                    name="enum_param",
                    type="string",
                    description="Parameter with enum constraint",
                    required=False,
                    enum=["option1", "option2", "option3"],
                ),
                "string_length_param": ToolParameter(
                    name="string_length_param",
                    type="string",
                    description="String with length constraints",
                    required=False,
                    min_length=5,
                    max_length=20,
                ),
                "number_range_param": ToolParameter(
                    name="number_range_param",
                    type="number",
                    description="Number with range constraints",
                    required=False,
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
            auth_method=AuthMethod.NONE,
            timeout_seconds=30,
            is_retryable=True,
            is_idempotent=True,
        )
        super().__init__(metadata)
        self.should_succeed = should_succeed
        self.execution_count = 0

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute mock tool with configurable success/failure."""
        start_time = time.time()
        self.execution_count += 1

        if not self.should_succeed:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error="Mock tool configured to fail",
                error_type="MockError",
                execution_time_ms=execution_time_ms,
            )

        result_data = {
            "parameters": parameters,
            "context": context.to_dict(),
            "execution_count": self.execution_count,
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result=result_data,
            execution_time_ms=execution_time_ms,
        )


class TestExecutionContext:
    """Test cases for ExecutionContext."""

    def test_execution_context_initialization(self):
        """Test basic context initialization."""
        context = ExecutionContext(
            user_id="user123",
            agent_id="agent456",
            trace_id="trace789",
            session_id="session101",
        )

        assert context.user_id == "user123"
        assert context.agent_id == "agent456"
        assert context.trace_id == "trace789"
        assert context.session_id == "session101"
        assert context.request_id is not None  # Auto-generated
        assert isinstance(context.metadata, dict)

    def test_execution_context_auto_generated_ids(self):
        """Test that trace_id and request_id are auto-generated if not provided."""
        context = ExecutionContext()

        assert context.trace_id is not None
        assert context.request_id is not None
        assert len(context.trace_id) > 0
        assert len(context.request_id) > 0

    def test_execution_context_to_dict(self):
        """Test context serialization to dictionary."""
        context = ExecutionContext(
            user_id="user123",
            agent_id="agent456",
            metadata={"key": "value"},
        )

        context_dict = context.to_dict()

        assert context_dict["user_id"] == "user123"
        assert context_dict["agent_id"] == "agent456"
        assert context_dict["metadata"] == {"key": "value"}
        assert "trace_id" in context_dict
        assert "request_id" in context_dict


class TestToolBase:
    """Test cases for Tool base class."""

    def test_tool_initialization(self):
        """Test tool initialization with metadata."""
        tool = MockTool()

        assert tool.metadata.tool_id == "mock_tool"
        assert tool.metadata.name == "Mock Tool"
        assert tool.metadata.category == ToolCategory.UTILITY
        assert tool.logger is not None

    def test_tool_repr(self):
        """Test tool string representation."""
        tool = MockTool()
        repr_str = repr(tool)

        assert "MockTool" in repr_str
        assert "mock_tool" in repr_str
        assert "Mock Tool" in repr_str

    @pytest.mark.asyncio
    async def test_tool_execute_success(self):
        """Test successful tool execution."""
        tool = MockTool(should_succeed=True)
        context = ExecutionContext(user_id="user123", agent_id="agent456")
        parameters = {"required_param": "test_value"}

        result = await tool.execute(parameters, context)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.tool_id == "mock_tool"
        assert result.error is None
        assert result.result is not None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_tool_execute_failure(self):
        """Test failed tool execution."""
        tool = MockTool(should_succeed=False)
        context = ExecutionContext()
        parameters = {"required_param": "test_value"}

        result = await tool.execute(parameters, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error is not None
        assert "fail" in result.error.lower()
        assert result.error_type == "MockError"

    @pytest.mark.asyncio
    async def test_validate_parameters_success(self):
        """Test parameter validation with valid parameters."""
        tool = MockTool()
        parameters = {"required_param": "test_value", "optional_param": 20}

        is_valid, error = await tool.validate_parameters(parameters)

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_parameters_missing_required(self):
        """Test parameter validation fails when required parameter is missing."""
        tool = MockTool()
        parameters = {"optional_param": 20}  # Missing required_param

        is_valid, error = await tool.validate_parameters(parameters)

        assert is_valid is False
        assert error is not None
        assert "required_param" in error
        assert "Missing" in error or "required" in error

    @pytest.mark.asyncio
    async def test_validate_parameters_enum_constraint(self):
        """Test parameter validation with enum constraints."""
        tool = MockTool()

        # Valid enum value
        parameters_valid = {"required_param": "test", "enum_param": "option1"}
        is_valid, error = await tool.validate_parameters(parameters_valid)
        assert is_valid is True

        # Invalid enum value
        parameters_invalid = {"required_param": "test", "enum_param": "invalid"}
        is_valid, error = await tool.validate_parameters(parameters_invalid)
        assert is_valid is False
        assert error is not None
        assert "enum_param" in error

    @pytest.mark.asyncio
    async def test_validate_parameters_string_length(self):
        """Test parameter validation with string length constraints."""
        tool = MockTool()

        # Valid string length
        parameters_valid = {
            "required_param": "test",
            "string_length_param": "valid_length",
        }
        is_valid, error = await tool.validate_parameters(parameters_valid)
        assert is_valid is True

        # String too short
        parameters_too_short = {
            "required_param": "test",
            "string_length_param": "abc",
        }
        is_valid, error = await tool.validate_parameters(parameters_too_short)
        assert is_valid is False
        assert "characters" in error or "length" in error

        # String too long
        parameters_too_long = {
            "required_param": "test",
            "string_length_param": "a" * 30,
        }
        is_valid, error = await tool.validate_parameters(parameters_too_long)
        assert is_valid is False
        assert "characters" in error or "length" in error

    @pytest.mark.asyncio
    async def test_validate_parameters_number_range(self):
        """Test parameter validation with number range constraints."""
        tool = MockTool()

        # Valid number
        parameters_valid = {"required_param": "test", "number_range_param": 50.0}
        is_valid, error = await tool.validate_parameters(parameters_valid)
        assert is_valid is True

        # Number too small
        parameters_too_small = {"required_param": "test", "number_range_param": -5.0}
        is_valid, error = await tool.validate_parameters(parameters_too_small)
        assert is_valid is False
        assert ">=" in error or "must be" in error

        # Number too large
        parameters_too_large = {"required_param": "test", "number_range_param": 150.0}
        is_valid, error = await tool.validate_parameters(parameters_too_large)
        assert is_valid is False
        assert "<=" in error or "must be" in error

    @pytest.mark.asyncio
    async def test_validate_parameters_allows_extra_parameters(self):
        """Test that validation allows extra parameters not in definition."""
        tool = MockTool()
        parameters = {
            "required_param": "test",
            "extra_param": "extra_value",  # Not in tool definition
        }

        is_valid, error = await tool.validate_parameters(parameters)

        # Should allow extra parameters (may be used by specific implementations)
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_tool_execution_count(self):
        """Test that mock tool tracks execution count."""
        tool = MockTool()
        context = ExecutionContext()
        parameters = {"required_param": "test"}

        assert tool.execution_count == 0

        await tool.execute(parameters, context)
        assert tool.execution_count == 1

        await tool.execute(parameters, context)
        assert tool.execution_count == 2

    @pytest.mark.asyncio
    async def test_tool_execution_with_context_metadata(self):
        """Test that execution context metadata is passed through."""
        tool = MockTool()
        context = ExecutionContext(
            user_id="user123",
            agent_id="agent456",
            metadata={"custom_key": "custom_value"},
        )
        parameters = {"required_param": "test"}

        result = await tool.execute(parameters, context)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["context"]["user_id"] == "user123"
        assert result.result["context"]["agent_id"] == "agent456"
        assert result.result["context"]["metadata"]["custom_key"] == "custom_value"


class TestToolInterfaceContract:
    """Test that Tool interface enforces contract requirements."""

    def test_tool_requires_metadata(self):
        """Test that Tool class requires metadata on initialization."""
        # Tool requires ToolDefinition metadata
        with pytest.raises(TypeError):
            Tool()  # type: ignore

    @pytest.mark.asyncio
    async def test_tool_execute_is_abstract(self):
        """Test that execute() method must be implemented by subclasses."""
        # Create incomplete tool without execute implementation
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            class IncompleteTooldef(Tool):
                pass

            _ = IncompleteTooldef(
                ToolDefinition(
                    tool_id="incomplete",
                    name="Incomplete",
                    description="Test",
                    category=ToolCategory.UTILITY,
                )
            )
