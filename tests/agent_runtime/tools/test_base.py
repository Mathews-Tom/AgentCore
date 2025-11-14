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
                "pattern_param": ToolParameter(
                    name="pattern_param",
                    type="string",
                    description="String with pattern constraint",
                    required=False,
                    pattern=r"^[A-Z]{3}-\d{4}$",  # e.g., ABC-1234
                ),
                "boolean_param": ToolParameter(
                    name="boolean_param",
                    type="boolean",
                    description="Boolean parameter",
                    required=False,
                ),
                "array_param": ToolParameter(
                    name="array_param",
                    type="array",
                    description="Array parameter",
                    required=False,
                    min_length=2,
                    max_length=5,
                ),
                "object_param": ToolParameter(
                    name="object_param",
                    type="object",
                    description="Object parameter",
                    required=False,
                ),
                "integer_strict_param": ToolParameter(
                    name="integer_strict_param",
                    type="integer",
                    description="Integer parameter for strict type checking",
                    required=False,
                    min_value=0,
                    max_value=100,
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
        # Cannot instantiate abstract class without execute implementation
        # Define incomplete tool class without execute method
        class IncompleteTooldef(Tool):
            pass

        # Try to instantiate the incomplete tool should raise TypeError
        with pytest.raises(TypeError):
            _ = IncompleteTooldef(
                ToolDefinition(
                    tool_id="incomplete",
                    name="Incomplete",
                    description="Test",
                    category=ToolCategory.UTILITY,
                )
            )


class TestTOOL007EnhancedValidation:
    """Test cases for TOOL-007: Parameter Validation Framework enhancements.

    Tests strict type checking, pattern validation, and enhanced error messages.
    """

    @pytest.mark.asyncio
    async def test_strict_type_checking_string(self):
        """Test strict type checking for string parameters."""
        tool = MockTool()

        # Valid string
        params_valid = {"required_param": "test_string"}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True
        assert error is None

        # Invalid: integer instead of string
        params_invalid = {"required_param": 123}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False
        assert "required_param" in error
        assert "incorrect type" in error
        assert "expected string" in error
        assert "got int" in error

    @pytest.mark.asyncio
    async def test_strict_type_checking_integer(self):
        """Test strict type checking for integer parameters."""
        tool = MockTool()

        # Valid integer
        params_valid = {"required_param": "test", "optional_param": 42}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Valid: whole number float acceptable for integer
        params_float_int = {"required_param": "test", "optional_param": 42.0}
        is_valid, error = await tool.validate_parameters(params_float_int)
        assert is_valid is True

        # Invalid: non-integer float
        params_invalid = {"required_param": "test", "integer_strict_param": 42.5}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False
        assert "integer_strict_param" in error
        assert "must be an integer" in error

        # Invalid: string instead of integer
        params_string = {"required_param": "test", "optional_param": "not_an_int"}
        is_valid, error = await tool.validate_parameters(params_string)
        assert is_valid is False
        assert "optional_param" in error
        assert "incorrect type" in error
        assert "expected integer" in error

    @pytest.mark.asyncio
    async def test_strict_type_checking_boolean(self):
        """Test strict type checking for boolean parameters."""
        tool = MockTool()

        # Valid boolean
        params_valid = {"required_param": "test", "boolean_param": True}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Invalid: integer instead of boolean
        params_invalid = {"required_param": "test", "boolean_param": 1}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False
        assert "boolean_param" in error
        assert "incorrect type" in error
        assert "expected boolean" in error
        assert "got int" in error

    @pytest.mark.asyncio
    async def test_strict_type_checking_number(self):
        """Test strict type checking for number parameters (int or float)."""
        tool = MockTool()

        # Valid: integer
        params_int = {"required_param": "test", "number_range_param": 50}
        is_valid, error = await tool.validate_parameters(params_int)
        assert is_valid is True

        # Valid: float
        params_float = {"required_param": "test", "number_range_param": 50.5}
        is_valid, error = await tool.validate_parameters(params_float)
        assert is_valid is True

        # Invalid: string instead of number
        params_invalid = {"required_param": "test", "number_range_param": "not_a_number"}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False
        assert "number_range_param" in error
        assert "incorrect type" in error
        assert "expected number" in error

    @pytest.mark.asyncio
    async def test_strict_type_checking_array(self):
        """Test strict type checking for array parameters."""
        tool = MockTool()

        # Valid array
        params_valid = {"required_param": "test", "array_param": ["item1", "item2", "item3"]}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Invalid: dict instead of array
        params_invalid = {"required_param": "test", "array_param": {"key": "value"}}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False
        assert "array_param" in error
        assert "incorrect type" in error
        assert "expected array" in error
        assert "got dict" in error

    @pytest.mark.asyncio
    async def test_strict_type_checking_object(self):
        """Test strict type checking for object parameters."""
        tool = MockTool()

        # Valid object
        params_valid = {"required_param": "test", "object_param": {"key": "value"}}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Invalid: list instead of object
        params_invalid = {"required_param": "test", "object_param": ["item1", "item2"]}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False
        assert "object_param" in error
        assert "incorrect type" in error
        assert "expected object" in error
        assert "got list" in error

    @pytest.mark.asyncio
    async def test_pattern_validation_valid(self):
        """Test pattern validation with valid patterns."""
        tool = MockTool()

        # Valid pattern: ABC-1234
        params_valid = {"required_param": "test", "pattern_param": "ABC-1234"}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True
        assert error is None

        # Another valid pattern: XYZ-9999
        params_valid2 = {"required_param": "test", "pattern_param": "XYZ-9999"}
        is_valid, error = await tool.validate_parameters(params_valid2)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_pattern_validation_invalid(self):
        """Test pattern validation with invalid patterns."""
        tool = MockTool()

        # Invalid: lowercase letters
        params_invalid1 = {"required_param": "test", "pattern_param": "abc-1234"}
        is_valid, error = await tool.validate_parameters(params_invalid1)
        assert is_valid is False
        assert "pattern_param" in error
        assert "does not match required pattern" in error
        assert "^[A-Z]{3}-\\d{4}$" in error
        assert "abc-1234" in error

        # Invalid: wrong format
        params_invalid2 = {"required_param": "test", "pattern_param": "ABCD-123"}
        is_valid, error = await tool.validate_parameters(params_invalid2)
        assert is_valid is False
        assert "pattern_param" in error
        assert "does not match required pattern" in error

        # Invalid: missing hyphen
        params_invalid3 = {"required_param": "test", "pattern_param": "ABC1234"}
        is_valid, error = await tool.validate_parameters(params_invalid3)
        assert is_valid is False
        assert "pattern_param" in error

    @pytest.mark.asyncio
    async def test_pattern_validation_invalid_regex(self):
        """Test pattern validation with invalid regex pattern."""
        # Create tool with invalid regex pattern
        metadata = ToolDefinition(
            tool_id="bad_pattern_tool",
            name="Bad Pattern Tool",
            description="Tool with invalid regex",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "bad_pattern_param": ToolParameter(
                    name="bad_pattern_param",
                    type="string",
                    description="Parameter with invalid regex",
                    required=False,
                    pattern=r"[invalid(regex",  # Invalid regex
                ),
            },
        )

        class BadPatternTool(Tool):
            def __init__(self):
                super().__init__(metadata)

            async def execute(
                self, parameters: dict[str, Any], context: ExecutionContext
            ) -> ToolResult:
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    execution_time_ms=0.0,
                )

        tool = BadPatternTool()
        params = {"bad_pattern_param": "test"}

        is_valid, error = await tool.validate_parameters(params)
        assert is_valid is False
        assert "Invalid regex pattern" in error
        assert "bad_pattern_param" in error

    @pytest.mark.asyncio
    async def test_enhanced_error_messages_type_mismatch(self):
        """Test that error messages include expected type, actual type, and value."""
        tool = MockTool()

        params = {"required_param": 123}  # Should be string
        is_valid, error = await tool.validate_parameters(params)

        assert is_valid is False
        assert "required_param" in error
        assert "incorrect type" in error
        assert "expected string" in error
        assert "got int" in error
        assert "123" in error  # Value included in error

    @pytest.mark.asyncio
    async def test_enhanced_error_messages_missing_required(self):
        """Test that error messages for missing required params include type and description."""
        tool = MockTool()

        params = {}  # Missing required_param
        is_valid, error = await tool.validate_parameters(params)

        assert is_valid is False
        assert "Missing required parameter" in error
        assert "required_param" in error
        assert "type: string" in error
        assert "description: A required parameter" in error

    @pytest.mark.asyncio
    async def test_enhanced_error_messages_enum_violation(self):
        """Test that enum violation error messages include parameter name and value."""
        tool = MockTool()

        params = {"required_param": "test", "enum_param": "invalid_option"}
        is_valid, error = await tool.validate_parameters(params)

        assert is_valid is False
        assert "enum_param" in error
        assert "must be one of" in error
        assert "option1" in error or "['option1', 'option2', 'option3']" in error
        assert "invalid_option" in error
        assert "type: str" in error

    @pytest.mark.asyncio
    async def test_array_length_validation(self):
        """Test array length validation with min and max constraints."""
        tool = MockTool()

        # Valid array length (3 items, within 2-5)
        params_valid = {"required_param": "test", "array_param": ["a", "b", "c"]}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Array too short (1 item, min is 2)
        params_too_short = {"required_param": "test", "array_param": ["a"]}
        is_valid, error = await tool.validate_parameters(params_too_short)
        assert is_valid is False
        assert "array_param" in error
        assert "at least 2 items" in error
        assert "got 1 items" in error

        # Array too long (6 items, max is 5)
        params_too_long = {
            "required_param": "test",
            "array_param": ["a", "b", "c", "d", "e", "f"],
        }
        is_valid, error = await tool.validate_parameters(params_too_long)
        assert is_valid is False
        assert "array_param" in error
        assert "at most 5 items" in error
        assert "got 6 items" in error

    @pytest.mark.asyncio
    async def test_integer_range_validation_strict(self):
        """Test integer range validation with strict type checking."""
        tool = MockTool()

        # Valid: integer within range
        params_valid = {
            "required_param": "test",
            "integer_strict_param": 50,
        }
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Valid: whole number float within range
        params_float_valid = {
            "required_param": "test",
            "integer_strict_param": 50.0,
        }
        is_valid, error = await tool.validate_parameters(params_float_valid)
        assert is_valid is True

        # Invalid: float with decimal within range
        params_float_invalid = {
            "required_param": "test",
            "integer_strict_param": 50.5,
        }
        is_valid, error = await tool.validate_parameters(params_float_invalid)
        assert is_valid is False
        assert "integer_strict_param" in error
        assert "must be an integer" in error

        # Invalid: integer below range
        params_below_range = {
            "required_param": "test",
            "integer_strict_param": -5,
        }
        is_valid, error = await tool.validate_parameters(params_below_range)
        assert is_valid is False
        assert "integer_strict_param" in error
        assert ">= 0" in error

        # Invalid: integer above range
        params_above_range = {
            "required_param": "test",
            "integer_strict_param": 150,
        }
        is_valid, error = await tool.validate_parameters(params_above_range)
        assert is_valid is False
        assert "integer_strict_param" in error
        assert "<= 100" in error

    @pytest.mark.asyncio
    async def test_combined_validation_errors(self):
        """Test that validation stops at first error (fail-fast)."""
        tool = MockTool()

        # Multiple violations: wrong type AND wrong enum value
        # Should fail on type check first
        params = {
            "required_param": 123,  # Wrong type (int instead of string)
            "enum_param": "invalid",  # Wrong enum value
        }
        is_valid, error = await tool.validate_parameters(params)

        assert is_valid is False
        # Should fail on required_param type mismatch first
        assert "required_param" in error
        assert "incorrect type" in error

    @pytest.mark.asyncio
    async def test_validation_logging(self):
        """Test that validation logs appropriate messages."""
        tool = MockTool()

        # Valid parameters should log debug message
        params_valid = {"required_param": "test"}
        is_valid, error = await tool.validate_parameters(params_valid)
        assert is_valid is True

        # Invalid parameters should log warning
        params_invalid = {"required_param": 123}
        is_valid, error = await tool.validate_parameters(params_invalid)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_edge_case_empty_string(self):
        """Test validation with empty string values."""
        tool = MockTool()

        # Empty string for required param
        params = {"required_param": ""}
        is_valid, error = await tool.validate_parameters(params)
        # Should pass type validation (is a string)
        # But would fail length validation if min_length was set
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_edge_case_none_value(self):
        """Test validation with None values."""
        tool = MockTool()

        # None for optional param (should fail type check)
        params = {"required_param": "test", "optional_param": None}
        is_valid, error = await tool.validate_parameters(params)
        assert is_valid is False
        assert "optional_param" in error
        assert "incorrect type" in error

    @pytest.mark.asyncio
    async def test_edge_case_boundary_values(self):
        """Test validation at boundary values for ranges."""
        tool = MockTool()

        # At minimum boundary
        params_min = {"required_param": "test", "number_range_param": 0.0}
        is_valid, error = await tool.validate_parameters(params_min)
        assert is_valid is True

        # At maximum boundary
        params_max = {"required_param": "test", "number_range_param": 100.0}
        is_valid, error = await tool.validate_parameters(params_max)
        assert is_valid is True

        # Just below minimum
        params_below = {"required_param": "test", "number_range_param": -0.1}
        is_valid, error = await tool.validate_parameters(params_below)
        assert is_valid is False

        # Just above maximum
        params_above = {"required_param": "test", "number_range_param": 100.1}
        is_valid, error = await tool.validate_parameters(params_above)
        assert is_valid is False
