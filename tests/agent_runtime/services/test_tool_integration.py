"""Comprehensive tests for tool integration framework."""

import asyncio

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionStatus,
    ToolParameter,
)
from agentcore.agent_runtime.services.tool_executor import (
    ToolExecutor,
    ToolTimeoutError,
    ToolValidationError,
)
from agentcore.agent_runtime.services.tool_registry import (
    ToolRegistry,
)


# Fixtures


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create fresh tool registry for testing."""
    return ToolRegistry()


@pytest.fixture
def tool_executor(tool_registry: ToolRegistry) -> ToolExecutor:
    """Create tool executor with fresh registry."""
    return ToolExecutor(tool_registry, enable_metrics=True)


# ToolDefinition Model Tests


def test_tool_definition_basic():
    """Test basic tool definition creation."""
    tool = ToolDefinition(
        tool_id="test_tool",
        name="Test Tool",
        description="A test tool",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
    )

    assert tool.tool_id == "test_tool"
    assert tool.name == "Test Tool"
    assert tool.category == ToolCategory.CUSTOM
    assert tool.version == "1.0.0"
    assert tool.timeout_seconds == 30  # Default
    assert tool.is_retryable is True  # Default
    assert tool.max_retries == 3  # Default


def test_tool_definition_semver_validation():
    """Test semantic versioning validation."""
    # Valid versions
    valid_versions = ["1.0.0", "2.1.3", "10.20.30"]
    for version in valid_versions:
        tool = ToolDefinition(
            tool_id="test",
            name="Test",
            description="Test",
            version=version,
            category=ToolCategory.CUSTOM,
        )
        assert tool.version == version

    # Invalid versions
    invalid_versions = ["1.0", "1", "v1.0.0", "1.0.0-beta"]
    for version in invalid_versions:
        with pytest.raises(ValueError):
            ToolDefinition(
                tool_id="test",
                name="Test",
                description="Test",
                version=version,
                category=ToolCategory.CUSTOM,
            )


def test_tool_parameter_model():
    """Test ToolParameter model."""
    param = ToolParameter(
        name="count",
        type="number",
        description="Number of items",
        required=True,
        min_value=1,
        max_value=100,
    )

    assert param.name == "count"
    assert param.type == "number"
    assert param.required is True
    assert param.min_value == 1
    assert param.max_value == 100


# ToolRegistry Search Tests


@pytest.mark.asyncio
async def test_search_by_category(tool_registry: ToolRegistry):
    """Test searching tools by category."""

    # Register tools in different categories
    async def tool1():
        return "result1"

    async def tool2():
        return "result2"

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="search_tool",
            name="Search Tool",
            description="Search tool",
            version="1.0.0",
            category=ToolCategory.SEARCH,
        ),
        tool1,
    )

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="api_tool",
            name="API Tool",
            description="API tool",
            version="1.0.0",
            category=ToolCategory.API_CLIENT,
        ),
        tool2,
    )

    # Search by category
    search_tools = tool_registry.search_by_category(ToolCategory.SEARCH)
    assert len(search_tools) == 1
    assert search_tools[0].tool_id == "search_tool"

    api_tools = tool_registry.search_by_category(ToolCategory.API_CLIENT)
    assert len(api_tools) == 1
    assert api_tools[0].tool_id == "api_tool"


@pytest.mark.asyncio
async def test_search_by_capability(tool_registry: ToolRegistry):
    """Test searching tools by capability."""

    async def parallel_tool():
        return "parallel"

    async def streaming_tool():
        return "streaming"

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="parallel_tool",
            name="Parallel Tool",
            description="Supports parallel execution",
            version="1.0.0",
            category=ToolCategory.DATA_PROCESSING,
            capabilities=["parallel_execution", "batch_processing"],
        ),
        parallel_tool,
    )

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="streaming_tool",
            name="Streaming Tool",
            description="Supports streaming",
            version="1.0.0",
            category=ToolCategory.DATA_PROCESSING,
            capabilities=["streaming", "parallel_execution"],
        ),
        streaming_tool,
    )

    # Search for parallel execution capability
    parallel_tools = tool_registry.search_by_capability("parallel_execution")
    assert len(parallel_tools) == 2

    # Search for streaming capability
    streaming_tools = tool_registry.search_by_capability("streaming")
    assert len(streaming_tools) == 1
    assert streaming_tools[0].tool_id == "streaming_tool"


@pytest.mark.asyncio
async def test_search_by_tags(tool_registry: ToolRegistry):
    """Test searching tools by tags."""

    async def math_tool():
        return 42

    async def text_tool():
        return "text"

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="math_tool",
            name="Math Tool",
            description="Math operations",
            version="1.0.0",
            category=ToolCategory.DATA_PROCESSING,
            tags=["math", "arithmetic", "numeric"],
        ),
        math_tool,
    )

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="text_tool",
            name="Text Tool",
            description="Text operations",
            version="1.0.0",
            category=ToolCategory.DATA_PROCESSING,
            tags=["text", "string", "processing"],
        ),
        text_tool,
    )

    # Search by single tag
    math_tools = tool_registry.search_by_tags(["math"])
    assert len(math_tools) == 1
    assert math_tools[0].tool_id == "math_tool"

    # Search by multiple tags (OR logic)
    tools = tool_registry.search_by_tags(["math", "text"])
    assert len(tools) == 2


@pytest.mark.asyncio
async def test_comprehensive_search(tool_registry: ToolRegistry):
    """Test comprehensive search with multiple filters."""

    async def advanced_search_tool():
        return "results"

    tool_registry.register_tool(
        ToolDefinition(
            tool_id="advanced_search",
            name="Advanced Search Tool",
            description="Advanced search capabilities",
            version="1.0.0",
            category=ToolCategory.SEARCH,
            capabilities=["semantic_search", "filters"],
            tags=["advanced", "search", "powerful"],
        ),
        advanced_search_tool,
    )

    # Search with multiple criteria
    results = tool_registry.search_tools(
        name_query="advanced",
        category=ToolCategory.SEARCH,
        capabilities=["semantic_search"],
        tags=["advanced"],
    )

    assert len(results) == 1
    assert results[0].tool_id == "advanced_search"

    # Search that should find nothing
    results = tool_registry.search_tools(
        category=ToolCategory.API_CLIENT,  # Wrong category
    )
    assert len(results) == 0


# ToolExecutor Tests


@pytest.mark.asyncio
async def test_tool_executor_success(tool_executor: ToolExecutor):
    """Test successful tool execution."""

    async def add_numbers(a: int, b: int) -> int:
        return a + b

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="add",
            name="Add",
            description="Add two numbers",
            version="1.0.0",
            category=ToolCategory.DATA_PROCESSING,
            parameters={
                "a": ToolParameter(name="a", type="number", description="First number", required=True),
                "b": ToolParameter(name="b", type="number", description="Second number", required=True),
            },
        ),
        add_numbers,
    )

    request = ToolExecutionRequest(
        tool_id="add",
        parameters={"a": 5, "b": 3},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result == 8
    assert result.execution_time_ms > 0
    assert result.is_success is True
    assert result.is_failure is False


@pytest.mark.asyncio
async def test_tool_executor_parameter_validation(tool_executor: ToolExecutor):
    """Test parameter validation in executor."""

    async def test_tool(required_param: str) -> str:
        return required_param

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="test_tool",
            name="Test Tool",
            description="Test",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            parameters={
                "required_param": ToolParameter(
                    name="required_param",
                    type="string",
                    description="Required parameter",
                    required=True,
                ),
            },
        ),
        test_tool,
    )

    # Missing required parameter
    request = ToolExecutionRequest(
        tool_id="test_tool",
        parameters={},  # Missing required_param
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error_type == "ToolValidationError"
    assert "required_param" in result.error


@pytest.mark.asyncio
async def test_tool_executor_timeout(tool_executor: ToolExecutor):
    """Test tool execution timeout."""

    async def slow_tool():
        await asyncio.sleep(10)  # Sleep longer than timeout
        return "done"

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="slow_tool",
            name="Slow Tool",
            description="Slow tool",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            timeout_seconds=1,  # 1 second timeout
        ),
        slow_tool,
    )

    request = ToolExecutionRequest(
        tool_id="slow_tool",
        parameters={},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.error_type == "ToolTimeoutError"
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_tool_executor_retry(tool_executor: ToolExecutor):
    """Test tool execution retry logic."""

    call_count = 0

    async def flaky_tool():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Transient error")
        return "success"

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="flaky_tool",
            name="Flaky Tool",
            description="Flaky tool",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            is_retryable=True,
            max_retries=3,
        ),
        flaky_tool,
    )

    request = ToolExecutionRequest(
        tool_id="flaky_tool",
        parameters={},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    # Should succeed after retries
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result == "success"
    assert call_count == 3  # Called 3 times (initial + 2 retries)


@pytest.mark.asyncio
async def test_tool_executor_hooks(tool_executor: ToolExecutor):
    """Test execution lifecycle hooks."""

    before_called = False
    after_called = False

    def before_hook(request):
        nonlocal before_called
        before_called = True

    def after_hook(result):
        nonlocal after_called
        after_called = True

    tool_executor.add_before_hook(before_hook)
    tool_executor.add_after_hook(after_hook)

    async def simple_tool():
        return "result"

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="simple_tool",
            name="Simple Tool",
            description="Simple",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
        ),
        simple_tool,
    )

    request = ToolExecutionRequest(
        tool_id="simple_tool",
        parameters={},
        agent_id="test-agent",
    )

    await tool_executor.execute(request)

    assert before_called is True
    assert after_called is True


# Parameter Validation Tests


@pytest.mark.asyncio
async def test_string_parameter_validation(tool_executor: ToolExecutor):
    """Test string parameter constraints."""

    async def string_tool(text: str) -> str:
        return text

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="string_tool",
            name="String Tool",
            description="String validation",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            parameters={
                "text": ToolParameter(
                    name="text",
                    type="string",
                    description="Text parameter",
                    required=True,
                    min_length=5,
                    max_length=10,
                ),
            },
        ),
        string_tool,
    )

    # Test too short
    request = ToolExecutionRequest(
        tool_id="string_tool",
        parameters={"text": "hi"},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.FAILED
    assert "at least 5 characters" in result.error

    # Test too long
    request = ToolExecutionRequest(
        tool_id="string_tool",
        parameters={"text": "this is way too long"},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.FAILED
    assert "at most 10 characters" in result.error

    # Test valid
    request = ToolExecutionRequest(
        tool_id="string_tool",
        parameters={"text": "valid"},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_number_parameter_validation(tool_executor: ToolExecutor):
    """Test number parameter constraints."""

    async def number_tool(value: float) -> float:
        return value

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="number_tool",
            name="Number Tool",
            description="Number validation",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            parameters={
                "value": ToolParameter(
                    name="value",
                    type="number",
                    description="Number parameter",
                    required=True,
                    min_value=0,
                    max_value=100,
                ),
            },
        ),
        number_tool,
    )

    # Test too small
    request = ToolExecutionRequest(
        tool_id="number_tool",
        parameters={"value": -1},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.FAILED
    assert "at least 0" in result.error

    # Test too large
    request = ToolExecutionRequest(
        tool_id="number_tool",
        parameters={"value": 101},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.FAILED
    assert "at most 100" in result.error

    # Test valid
    request = ToolExecutionRequest(
        tool_id="number_tool",
        parameters={"value": 50},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_enum_parameter_validation(tool_executor: ToolExecutor):
    """Test enum parameter validation."""

    async def enum_tool(operation: str) -> str:
        return operation

    tool_executor._registry.register_tool(
        ToolDefinition(
            tool_id="enum_tool",
            name="Enum Tool",
            description="Enum validation",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            parameters={
                "operation": ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation type",
                    required=True,
                    enum=["add", "subtract", "multiply", "divide"],
                ),
            },
        ),
        enum_tool,
    )

    # Test invalid enum value
    request = ToolExecutionRequest(
        tool_id="enum_tool",
        parameters={"operation": "invalid"},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.FAILED
    assert "must be one of" in result.error

    # Test valid enum value
    request = ToolExecutionRequest(
        tool_id="enum_tool",
        parameters={"operation": "add"},
        agent_id="test-agent",
    )
    result = await tool_executor.execute(request)
    assert result.status == ToolExecutionStatus.SUCCESS
