"""Tests for tool integration models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)


class TestToolParameter:
    """Test cases for ToolParameter model."""

    def test_tool_parameter_minimal(self):
        """Test ToolParameter with minimal required fields."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="A test parameter",
        )

        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is False  # Default
        assert param.default is None
        assert param.enum is None

    def test_tool_parameter_complete(self):
        """Test ToolParameter with all fields."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
            default="",
            enum=["option1", "option2", "option3"],
            min_length=1,
            max_length=100,
            pattern=r"^\w+$",
        )

        assert param.name == "query"
        assert param.type == "string"
        assert param.required is True
        assert param.default == ""
        assert param.enum == ["option1", "option2", "option3"]
        assert param.min_length == 1
        assert param.max_length == 100
        assert param.pattern == r"^\w+$"

    def test_tool_parameter_number_constraints(self):
        """Test ToolParameter with number constraints."""
        param = ToolParameter(
            name="count",
            type="integer",
            description="Number of results",
            required=False,
            default=10,
            min_value=1.0,
            max_value=100.0,
        )

        assert param.min_value == 1.0
        assert param.max_value == 100.0
        assert param.default == 10

    def test_tool_parameter_serialization(self):
        """Test ToolParameter serialization to dict."""
        param = ToolParameter(
            name="test",
            type="string",
            description="Test param",
            required=True,
        )

        param_dict = param.model_dump()
        assert param_dict["name"] == "test"
        assert param_dict["type"] == "string"
        assert param_dict["required"] is True


class TestToolDefinition:
    """Test cases for ToolDefinition model."""

    def test_tool_definition_minimal(self):
        """Test ToolDefinition with minimal required fields."""
        tool = ToolDefinition(
            tool_id="test_tool",
            name="Test Tool",
            description="A test tool",
            category=ToolCategory.UTILITY,
        )

        assert tool.tool_id == "test_tool"
        assert tool.name == "Test Tool"
        assert tool.description == "A test tool"
        assert tool.category == ToolCategory.UTILITY
        assert tool.version == "1.0.0"  # Default
        assert tool.auth_method == AuthMethod.NONE  # Default
        assert tool.timeout_seconds == 30  # Default

    def test_tool_definition_complete(self):
        """Test ToolDefinition with all fields."""
        tool = ToolDefinition(
            tool_id="google_search",
            name="Google Search",
            description="Search the web using Google",
            version="2.1.0",
            category=ToolCategory.SEARCH,
            parameters={
                "query": ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                "num_results": ToolParameter(
                    name="num_results",
                    type="integer",
                    description="Number of results",
                    required=False,
                    default=10,
                ),
            },
            auth_method=AuthMethod.API_KEY,
            auth_config={"api_key_header": "X-API-Key"},
            timeout_seconds=60,
            is_retryable=True,
            is_idempotent=True,
            max_retries=3,
            rate_limits={"calls_per_minute": 100},
            cost_per_execution=0.001,
            capabilities=["parallel_execution", "streaming"],
            requirements=["api_key"],
            security_requirements=["https"],
            metadata={"provider": "Google"},
            tags=["search", "web"],
        )

        assert tool.tool_id == "google_search"
        assert tool.version == "2.1.0"
        assert tool.auth_method == AuthMethod.API_KEY
        assert len(tool.parameters) == 2
        assert tool.rate_limits["calls_per_minute"] == 100
        assert tool.cost_per_execution == 0.001
        assert "parallel_execution" in tool.capabilities
        assert "api_key" in tool.requirements

    def test_tool_definition_semver_validation_valid(self):
        """Test ToolDefinition accepts valid semver versions."""
        valid_versions = ["1.0.0", "2.1.3", "0.0.1", "10.20.30"]

        for version in valid_versions:
            tool = ToolDefinition(
                tool_id="test",
                name="Test",
                description="Test",
                category=ToolCategory.UTILITY,
                version=version,
            )
            assert tool.version == version

    def test_tool_definition_semver_validation_invalid(self):
        """Test ToolDefinition rejects invalid semver versions."""
        invalid_versions = [
            "1.0",  # Missing patch
            "1",  # Missing minor and patch
            "1.0.0.0",  # Too many parts
            "1.0.a",  # Non-numeric
            "v1.0.0",  # Prefix
        ]

        for version in invalid_versions:
            with pytest.raises(ValidationError) as exc_info:
                ToolDefinition(
                    tool_id="test",
                    name="Test",
                    description="Test",
                    category=ToolCategory.UTILITY,
                    version=version,
                )
            assert "Version must be in semver format" in str(exc_info.value) or \
                   "Version parts must be numeric" in str(exc_info.value)

    def test_tool_definition_timeout_validation(self):
        """Test ToolDefinition timeout constraints."""
        # Valid timeouts
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            timeout_seconds=30,
        )
        assert tool.timeout_seconds == 30

        # Minimum valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            timeout_seconds=1,
        )
        assert tool.timeout_seconds == 1

        # Maximum valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            timeout_seconds=3600,
        )
        assert tool.timeout_seconds == 3600

        # Too low
        with pytest.raises(ValidationError):
            ToolDefinition(
                tool_id="test",
                name="Test",
                description="Test",
                category=ToolCategory.UTILITY,
                timeout_seconds=0,
            )

        # Too high
        with pytest.raises(ValidationError):
            ToolDefinition(
                tool_id="test",
                name="Test",
                description="Test",
                category=ToolCategory.UTILITY,
                timeout_seconds=3601,
            )

    def test_tool_definition_max_retries_validation(self):
        """Test ToolDefinition max_retries constraints."""
        # Valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            max_retries=5,
        )
        assert tool.max_retries == 5

        # Minimum valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            max_retries=0,
        )
        assert tool.max_retries == 0

        # Maximum valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            max_retries=10,
        )
        assert tool.max_retries == 10

        # Too high
        with pytest.raises(ValidationError):
            ToolDefinition(
                tool_id="test",
                name="Test",
                description="Test",
                category=ToolCategory.UTILITY,
                max_retries=11,
            )

    def test_tool_definition_cost_validation(self):
        """Test ToolDefinition cost_per_execution constraints."""
        # Valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            cost_per_execution=0.01,
        )
        assert tool.cost_per_execution == 0.01

        # Zero is valid
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            category=ToolCategory.UTILITY,
            cost_per_execution=0.0,
        )
        assert tool.cost_per_execution == 0.0

        # Negative is invalid
        with pytest.raises(ValidationError):
            ToolDefinition(
                tool_id="test",
                name="Test",
                description="Test",
                category=ToolCategory.UTILITY,
                cost_per_execution=-0.01,
            )


class TestToolExecutionRequest:
    """Test cases for ToolExecutionRequest model."""

    def test_tool_execution_request_minimal(self):
        """Test ToolExecutionRequest with minimal fields."""
        request = ToolExecutionRequest(
            tool_id="test_tool",
            agent_id="agent123",
        )

        assert request.tool_id == "test_tool"
        assert request.agent_id == "agent123"
        assert isinstance(request.parameters, dict)
        assert len(request.request_id) > 0  # Auto-generated

    def test_tool_execution_request_complete(self):
        """Test ToolExecutionRequest with all fields."""
        request = ToolExecutionRequest(
            tool_id="google_search",
            parameters={"query": "test", "num_results": 10},
            execution_context={"trace_id": "trace123", "session_id": "session456"},
            agent_id="agent789",
            request_id="req001",
            timeout_override=60,
            retry_override=5,
        )

        assert request.tool_id == "google_search"
        assert request.parameters["query"] == "test"
        assert request.execution_context["trace_id"] == "trace123"
        assert request.request_id == "req001"
        assert request.timeout_override == 60
        assert request.retry_override == 5

    def test_tool_execution_request_auto_generated_id(self):
        """Test that request_id is auto-generated if not provided."""
        request1 = ToolExecutionRequest(tool_id="test", agent_id="agent1")
        request2 = ToolExecutionRequest(tool_id="test", agent_id="agent1")

        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0


class TestToolExecutionStatus:
    """Test cases for ToolExecutionStatus enum."""

    def test_tool_execution_status_values(self):
        """Test all ToolExecutionStatus enum values."""
        assert ToolExecutionStatus.PENDING == "pending"
        assert ToolExecutionStatus.RUNNING == "running"
        assert ToolExecutionStatus.SUCCESS == "success"
        assert ToolExecutionStatus.FAILED == "failed"
        assert ToolExecutionStatus.TIMEOUT == "timeout"
        assert ToolExecutionStatus.CANCELLED == "cancelled"

    def test_tool_execution_status_comparison(self):
        """Test ToolExecutionStatus equality comparison."""
        status1 = ToolExecutionStatus.SUCCESS
        status2 = ToolExecutionStatus.SUCCESS
        status3 = ToolExecutionStatus.FAILED

        assert status1 == status2
        assert status1 != status3


class TestToolResult:
    """Test cases for ToolResult model."""

    def test_tool_result_success(self):
        """Test ToolResult for successful execution."""
        result = ToolResult(
            request_id="req123",
            tool_id="test_tool",
            status=ToolExecutionStatus.SUCCESS,
            result={"data": "test_result"},
            execution_time_ms=123.45,
        )

        assert result.request_id == "req123"
        assert result.tool_id == "test_tool"
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["data"] == "test_result"
        assert result.execution_time_ms == 123.45
        assert result.error is None
        assert result.is_success is True
        assert result.is_failure is False

    def test_tool_result_failure(self):
        """Test ToolResult for failed execution."""
        result = ToolResult(
            request_id="req456",
            tool_id="test_tool",
            status=ToolExecutionStatus.FAILED,
            error="Connection timeout",
            error_type="TimeoutError",
            execution_time_ms=5000.0,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.error_type == "TimeoutError"
        assert result.result is None
        assert result.is_success is False
        assert result.is_failure is True

    def test_tool_result_timeout(self):
        """Test ToolResult for timeout status."""
        result = ToolResult(
            request_id="req789",
            tool_id="test_tool",
            status=ToolExecutionStatus.TIMEOUT,
            error="Execution exceeded timeout",
            execution_time_ms=30000.0,
        )

        assert result.status == ToolExecutionStatus.TIMEOUT
        assert result.is_success is False
        assert result.is_failure is True

    def test_tool_result_with_metadata(self):
        """Test ToolResult with comprehensive metadata."""
        result = ToolResult(
            request_id="req001",
            tool_id="python_executor",
            status=ToolExecutionStatus.SUCCESS,
            result={"output": "Hello, World!"},
            execution_time_ms=250.5,
            retry_count=2,
            memory_mb=128.5,
            cpu_percent=15.3,
            metadata={
                "execution_mode": "sandbox",
                "python_version": "3.12",
            },
        )

        assert result.retry_count == 2
        assert result.memory_mb == 128.5
        assert result.cpu_percent == 15.3
        assert result.metadata["execution_mode"] == "sandbox"
        assert result.metadata["python_version"] == "3.12"

    def test_tool_result_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        result = ToolResult(
            request_id="req123",
            tool_id="test",
            status=ToolExecutionStatus.SUCCESS,
            execution_time_ms=100.0,
        )

        assert isinstance(result.timestamp, datetime)
        assert result.timestamp is not None

    def test_tool_result_is_success_property(self):
        """Test is_success property for various statuses."""
        success = ToolResult(
            request_id="req1",
            tool_id="test",
            status=ToolExecutionStatus.SUCCESS,
            execution_time_ms=100.0,
        )
        assert success.is_success is True

        failed = ToolResult(
            request_id="req2",
            tool_id="test",
            status=ToolExecutionStatus.FAILED,
            execution_time_ms=100.0,
        )
        assert failed.is_success is False

    def test_tool_result_is_failure_property(self):
        """Test is_failure property for various statuses."""
        failed_statuses = [
            ToolExecutionStatus.FAILED,
            ToolExecutionStatus.TIMEOUT,
            ToolExecutionStatus.CANCELLED,
        ]

        for status in failed_statuses:
            result = ToolResult(
                request_id="req",
                tool_id="test",
                status=status,
                execution_time_ms=100.0,
            )
            assert result.is_failure is True

        success = ToolResult(
            request_id="req",
            tool_id="test",
            status=ToolExecutionStatus.SUCCESS,
            execution_time_ms=100.0,
        )
        assert success.is_failure is False


class TestToolCategory:
    """Test cases for ToolCategory enum."""

    def test_tool_category_values(self):
        """Test all ToolCategory enum values."""
        assert ToolCategory.UTILITY == "utility"
        assert ToolCategory.SEARCH == "search"
        assert ToolCategory.CODE_EXECUTION == "code_execution"
        assert ToolCategory.API_CLIENT == "api_client"
        assert ToolCategory.DATA_PROCESSING == "data_processing"
        assert ToolCategory.COMMUNICATION == "communication"
        assert ToolCategory.FILE_SYSTEM == "file_system"
        assert ToolCategory.DATABASE == "database"
        assert ToolCategory.MONITORING == "monitoring"
        assert ToolCategory.SECURITY == "security"
        assert ToolCategory.CUSTOM == "custom"


class TestAuthMethod:
    """Test cases for AuthMethod enum."""

    def test_auth_method_values(self):
        """Test all AuthMethod enum values."""
        assert AuthMethod.NONE == "none"
        assert AuthMethod.API_KEY == "api_key"
        assert AuthMethod.BEARER_TOKEN == "bearer_token"
        assert AuthMethod.OAUTH2 == "oauth2"
        assert AuthMethod.BASIC_AUTH == "basic_auth"
        assert AuthMethod.JWT == "jwt"
        assert AuthMethod.CUSTOM == "custom"


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_tool_parameter_json_roundtrip(self):
        """Test ToolParameter JSON serialization roundtrip."""
        original = ToolParameter(
            name="test",
            type="string",
            description="Test parameter",
            required=True,
            enum=["a", "b", "c"],
        )

        # Serialize to dict
        param_dict = original.model_dump()

        # Deserialize from dict
        restored = ToolParameter(**param_dict)

        assert restored.name == original.name
        assert restored.type == original.type
        assert restored.enum == original.enum

    def test_tool_definition_json_roundtrip(self):
        """Test ToolDefinition JSON serialization roundtrip."""
        original = ToolDefinition(
            tool_id="test",
            name="Test Tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters={
                "param1": ToolParameter(
                    name="param1",
                    type="string",
                    description="Test param",
                )
            },
        )

        # Serialize to dict
        tool_dict = original.model_dump()

        # Deserialize from dict
        restored = ToolDefinition(**tool_dict)

        assert restored.tool_id == original.tool_id
        assert restored.category == original.category
        assert "param1" in restored.parameters

    def test_tool_result_json_roundtrip(self):
        """Test ToolResult JSON serialization roundtrip."""
        original = ToolResult(
            request_id="req123",
            tool_id="test",
            status=ToolExecutionStatus.SUCCESS,
            result={"key": "value"},
            execution_time_ms=100.0,
            metadata={"custom": "data"},
        )

        # Serialize to dict
        result_dict = original.model_dump()

        # Deserialize from dict (need to handle datetime)
        result_dict["timestamp"] = original.timestamp  # Preserve datetime
        restored = ToolResult(**result_dict)

        assert restored.request_id == original.request_id
        assert restored.status == original.status
        assert restored.result == original.result
        assert restored.metadata == original.metadata
