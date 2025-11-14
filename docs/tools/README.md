# Tool Integration Framework

**Version:** 1.0.0
**Status:** Production
**Component:** agent_runtime/tools

## Overview

The Tool Integration Framework provides a standardized system for agents to interact with external services, execute code, search the web, call APIs, and process data. It implements a layered architecture with comprehensive validation, error handling, rate limiting, and observability.

## Quick Start

### Using an Existing Tool

```python
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.base import ExecutionContext

# Initialize executor
executor = ToolExecutor()

# Create execution context
context = ExecutionContext(
    user_id="user123",
    agent_id="agent456",
    trace_id="trace789"
)

# Execute a tool
result = await executor.execute_tool(
    tool_id="google_search",
    parameters={"query": "AgentCore framework"},
    context=context
)

if result.is_success:
    print(f"Results: {result.result}")
else:
    print(f"Error: {result.error}")
```

### Creating a Custom Tool

```python
from agentcore.agent_runtime.tools.base import Tool, ExecutionContext
from agentcore.agent_runtime.models.tool_integration import (
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    ToolResult,
    ToolExecutionStatus,
    AuthMethod
)
import time

class WeatherTool(Tool):
    """Tool for fetching weather information."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="weather_lookup",
            name="Weather Lookup",
            description="Get current weather for a location",
            version="1.0.0",
            category=ToolCategory.API_CLIENT,
            parameters={
                "location": ToolParameter(
                    name="location",
                    type="string",
                    description="City name or coordinates",
                    required=True,
                    min_length=2,
                    max_length=100
                ),
                "units": ToolParameter(
                    name="units",
                    type="string",
                    description="Temperature units",
                    required=False,
                    default="celsius",
                    enum=["celsius", "fahrenheit", "kelvin"]
                )
            },
            auth_method=AuthMethod.API_KEY,
            timeout_seconds=10,
            is_retryable=True,
            is_idempotent=True,
            max_retries=3
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext
    ) -> ToolResult:
        """Execute weather lookup."""
        start_time = time.time()

        try:
            # Validate parameters
            is_valid, error = await self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=error,
                    error_type="ValidationError",
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # Perform weather API call
            location = parameters["location"]
            units = parameters.get("units", "celsius")

            # Call weather API (example)
            weather_data = await self._fetch_weather(location, units)

            execution_time_ms = (time.time() - start_time) * 1000

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result=weather_data,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "weather_lookup_failed",
                error=str(e),
                location=parameters.get("location")
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms
            )

    async def _fetch_weather(
        self,
        location: str,
        units: str
    ) -> dict[str, Any]:
        """Fetch weather data from API."""
        # Implementation here
        return {
            "location": location,
            "temperature": 22.5,
            "units": units,
            "conditions": "Partly cloudy"
        }
```

## Architecture

The framework follows a **layered architecture pattern** with four distinct layers:

```
┌─────────────────────────────────────────────────────┐
│          Tool Interface Layer (base.py)             │
│  Tool, ExecutionContext, Parameter Validation       │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│         Tool Registry Layer (registry.py)           │
│  Discovery, Metadata, Version Control, Search       │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│        Tool Execution Layer (executor.py)           │
│  Lifecycle, Auth, Rate Limiting, Observability      │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│        Tool Adapters Layer (builtin/*.py)           │
│  Search Tools, Code Execution, API Clients, Utils   │
└─────────────────────────────────────────────────────┘
```

See [architecture.md](./architecture.md) for detailed architecture documentation.

## Core Concepts

### Tool Interface

The `Tool` abstract base class defines the contract all tools must implement:

- **Metadata:** Tool definition with parameters, authentication, constraints
- **Execute Method:** Async method for tool execution logic
- **Parameter Validation:** Automatic validation with type checking and constraints
- **Error Handling:** Structured error responses with detailed messages

### Execution Context

The `ExecutionContext` provides runtime information for tool execution:

- **User Identity:** User ID initiating the tool execution
- **Agent Identity:** Agent ID requesting tool execution
- **Tracing:** Distributed tracing identifiers (trace_id, span_id)
- **Session:** Session identifier for grouping related executions
- **Metadata:** Additional context-specific metadata

### Tool Result

The `ToolResult` model contains comprehensive execution information:

- **Status:** SUCCESS, FAILED, TIMEOUT, CANCELLED
- **Result Data:** Tool execution output
- **Error Information:** Detailed error messages and types
- **Execution Metadata:** Timing, retry count, resource usage
- **Observability:** Structured data for monitoring and debugging

## Key Features

### 1. Parameter Validation Framework

Enhanced validation with strict type checking, pattern validation, and comprehensive error messages:

```python
# String with pattern validation
ToolParameter(
    name="email",
    type="string",
    pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    required=True
)

# Integer with range constraints
ToolParameter(
    name="age",
    type="integer",
    min_value=0,
    max_value=150,
    required=True
)

# Array with length constraints
ToolParameter(
    name="tags",
    type="array",
    min_length=1,
    max_length=10,
    required=False
)
```

### 2. Tool Registry

Centralized tool discovery and management:

```python
from agentcore.agent_runtime.tools.registry import ToolRegistry

registry = ToolRegistry()

# Register a tool
await registry.register(weather_tool)

# Discover tools
search_tools = await registry.search(category=ToolCategory.SEARCH)

# Get tool by ID
tool = await registry.get("google_search")
```

### 3. Tool Execution

Lifecycle management with authentication, rate limiting, and retry logic:

```python
from agentcore.agent_runtime.tools.executor import ToolExecutor

executor = ToolExecutor(
    registry=registry,
    rate_limiter=rate_limiter,
    auth_service=auth_service
)

# Execute with automatic retry and rate limiting
result = await executor.execute_tool(
    tool_id="api_request",
    parameters={"url": "https://api.example.com"},
    context=context
)
```

### 4. Built-in Tools

Production-ready tools for common use cases:

**Search Tools:**
- `GoogleSearchTool`: Web search via Google API
- `WikipediaSearchTool`: Wikipedia article search
- `WebScrapeTool`: Web page content extraction

**Code Execution Tools:**
- `ExecutePythonTool`: Safe Python code execution in sandbox
- `EvaluateExpressionTool`: Mathematical expression evaluation

**API Client Tools:**
- `HttpRequestTool`: Generic HTTP client
- `RestGetTool`: RESTful GET requests
- `RestPostTool`: RESTful POST requests
- `GraphQLQueryTool`: GraphQL query execution

**Utility Tools:**
- `CalculatorTool`: Mathematical calculations
- `GetCurrentTimeTool`: Current timestamp
- `EchoTool`: Parameter echo (testing/debugging)

## Developer Guide

See [developer-guide.md](./developer-guide.md) for:

- Step-by-step tool creation tutorial
- Best practices and design patterns
- Testing strategies
- Deployment guidelines
- Troubleshooting guide

## API Reference

### Tool Interface

```python
class Tool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, metadata: ToolDefinition): ...

    @abstractmethod
    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext
    ) -> ToolResult: ...

    async def validate_parameters(
        self,
        parameters: dict[str, Any]
    ) -> tuple[bool, str | None]: ...
```

### Execution Context

```python
class ExecutionContext:
    """Context information for tool execution."""

    user_id: str | None
    agent_id: str | None
    trace_id: str
    session_id: str | None
    request_id: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: ...
```

### Tool Result

```python
class ToolResult(BaseModel):
    """Result of tool execution."""

    request_id: str
    tool_id: str
    status: ToolExecutionStatus
    result: Any | None
    error: str | None
    error_type: str | None
    execution_time_ms: float
    timestamp: datetime
    retry_count: int
    memory_mb: float | None
    cpu_percent: float | None
    metadata: dict[str, Any]
```

## Configuration

Tool framework configuration in `src/agentcore/agent_runtime/config.py`:

```python
# Tool execution settings
TOOL_EXECUTION_TIMEOUT: int = 30  # seconds
TOOL_MAX_RETRIES: int = 3
TOOL_RETRY_BACKOFF: float = 1.5  # exponential backoff multiplier

# Rate limiting
TOOL_RATE_LIMIT_ENABLED: bool = True
TOOL_RATE_LIMIT_PER_MINUTE: int = 60

# Sandboxing (code execution)
TOOL_SANDBOX_ENABLED: bool = True
TOOL_SANDBOX_MEMORY_LIMIT: str = "512M"
TOOL_SANDBOX_CPU_LIMIT: str = "1.0"
```

## Testing

### Unit Tests

```bash
# Run all tool tests
uv run pytest tests/agent_runtime/tools/ -v

# Run specific tool tests
uv run pytest tests/agent_runtime/tools/test_base.py -v

# Run with coverage
uv run pytest tests/agent_runtime/tools/ --cov=src/agentcore/agent_runtime/tools
```

### Integration Tests

```bash
# Run integration tests
uv run pytest tests/integration/test_tool_integration.py -v
```

## Monitoring

### Metrics

Key metrics exposed via Prometheus:

- `tool_execution_total`: Total tool executions (by tool_id, status)
- `tool_execution_duration_seconds`: Execution duration histogram
- `tool_execution_errors_total`: Error count (by tool_id, error_type)
- `tool_validation_failures_total`: Parameter validation failures
- `tool_rate_limit_exceeded_total`: Rate limit violations

### Logging

Structured logging with context:

```python
self.logger.info(
    "tool_execution_started",
    tool_id=tool_id,
    request_id=context.request_id,
    user_id=context.user_id
)
```

## Security

### Authentication

Tools support multiple authentication methods:

- `NONE`: No authentication required
- `API_KEY`: API key authentication
- `BEARER_TOKEN`: Bearer token authentication
- `OAUTH2`: OAuth 2.0 flow
- `BASIC_AUTH`: Basic HTTP authentication
- `JWT`: JSON Web Token
- `CUSTOM`: Custom authentication logic

### Sandboxing

Code execution tools run in isolated Docker containers:

- Memory limits enforced
- CPU limits enforced
- Network access restricted
- Filesystem access restricted
- Execution timeout enforced

### Input Validation

All tool parameters validated before execution:

- Type checking with strict validation
- Range validation for numbers
- Length validation for strings/arrays
- Pattern validation with regex
- Enum validation for constrained values

## Performance

### Benchmarks

Typical performance characteristics:

- **Framework Overhead:** <5ms per execution
- **Parameter Validation:** <1ms per tool
- **Registry Lookup:** <2ms per query
- **Concurrent Executions:** 1000+ per instance

### Optimization Tips

1. Use connection pooling for API clients
2. Cache tool metadata in registry
3. Implement idempotent operations for retries
4. Use async/await throughout
5. Monitor resource usage per tool

## Troubleshooting

### Common Issues

**Tool execution timeout:**
```python
# Increase timeout in tool definition
ToolDefinition(
    ...,
    timeout_seconds=60  # Increase from default 30
)
```

**Parameter validation failed:**
```python
# Check parameter types match definition
# Enable debug logging to see validation details
self.logger.setLevel("DEBUG")
```

**Rate limit exceeded:**
```python
# Adjust rate limits or implement backoff
ToolDefinition(
    ...,
    rate_limits={"calls_per_minute": 120}  # Increase limit
)
```

## Related Documentation

- [Architecture Documentation](./architecture.md)
- [Developer Guide](./developer-guide.md)
- [Tool Integration Specification](../specs/tool-integration/spec.md)
- [Implementation Plan](../specs/tool-integration/plan.md)
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines

## Support

For questions or issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [developer-guide.md](./developer-guide.md)
3. Open an issue in the AgentCore repository
4. Contact the development team

## Changelog

### Version 1.0.0 (2025-01-13)

- Initial release with core framework
- Tool interface with parameter validation
- Tool registry with search and discovery
- Tool executor with lifecycle management
- Built-in tools (search, code execution, API clients, utilities)
- Comprehensive test coverage (98%+ for core modules)
- Production-ready documentation
