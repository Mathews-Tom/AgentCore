# Tool Integration Framework - Developer Guide

## Overview

The Tool Integration Framework provides a comprehensive system for tool discovery, registration, execution, and management in AgentCore. It enables agents to discover and execute tools with full lifecycle management, rate limiting, retry logic, and parallel execution support.

## Architecture

The framework consists of four main components:

1. **Tool Registry** - Central registry for tool discovery and metadata
2. **Tool Executor** - Execution engine with lifecycle management
3. **Advanced Features** - Rate limiting, retry logic, parallel execution
4. **JSON-RPC Integration** - A2A protocol compliant API

## Configuration

The Tool Integration Framework can be configured via environment variables or settings:

```bash
# Tool Integration Configuration
TOOL_INTEGRATION_ENABLED=true
TOOL_EXECUTION_TIMEOUT=30
TOOL_MAX_RETRIES=3
TOOL_RETRY_BASE_DELAY=1.0
TOOL_RETRY_MAX_DELAY=10.0
TOOL_RETRY_STRATEGY=exponential  # exponential, linear, or fixed
TOOL_RETRY_JITTER=true

# Rate Limiter Configuration
RATE_LIMITER_ENABLED=false
RATE_LIMITER_REDIS_URL=redis://localhost:6379/1
RATE_LIMITER_KEY_PREFIX=agentcore:ratelimit
RATE_LIMITER_DEFAULT_LIMIT=100
RATE_LIMITER_DEFAULT_WINDOW_SECONDS=60

# Parallel Execution Configuration
PARALLEL_EXECUTION_ENABLED=true
PARALLEL_MAX_CONCURRENT=10
PARALLEL_DEFAULT_TIMEOUT=300
```

### Using Configured Executor

```python
from agentcore.agent_runtime.services.tool_executor_factory import create_tool_executor

# Create executor with configuration
executor = create_tool_executor()

# Or override specific settings
executor = create_tool_executor(
    settings_override={
        "tool_max_retries": 5,
        "tool_retry_strategy": "linear",
        "rate_limiter_enabled": True,
    }
)
```

## Quick Start

###  Registering a Tool

```python
from agentcore.agent_runtime.models.tool_integration import (
    ToolDefinition,
    ToolCategory,
    ToolParameter,
    AuthMethod,
)
from agentcore.agent_runtime.services.tool_registry import get_tool_registry

# Define your tool
async def my_tool_function(param1: str, param2: int) -> str:
    """Your tool implementation."""
    return f"Result: {param1} * {param2}"

# Create tool definition
tool_def = ToolDefinition(
    tool_id="my_tool",
    name="My Custom Tool",
    description="A custom tool for demonstration",
    version="1.0.0",
    category=ToolCategory.CUSTOM,
    parameters={
        "param1": ToolParameter(
            name="param1",
            type="string",
            description="First parameter",
            required=True,
        ),
        "param2": ToolParameter(
            name="param2",
            type="integer",
            description="Second parameter",
            required=True,
            min_value=0,
            max_value=100,
        ),
    },
    auth_method=AuthMethod.NONE,
    capabilities=["custom_processing"],
    tags=["demo", "custom"],
)

# Register tool
registry = get_tool_registry()
registry.register_tool(tool_def, my_tool_function)
```

### Executing a Tool

```python
from agentcore.agent_runtime.models.tool_integration import ToolExecutionRequest
from agentcore.agent_runtime.services.tool_executor import get_tool_executor

# Create execution request
request = ToolExecutionRequest(
    tool_id="my_tool",
    parameters={"param1": "hello", "param2": 42},
    agent_id="my-agent",
)

# Execute tool
executor = get_tool_executor()
result = await executor.execute(request)

print(f"Status: {result.status}")
print(f"Result: {result.result}")
```

### Using via JSON-RPC

```python
import httpx

# List available tools
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.list",
    "params": {"category": "search"},
    "id": "1"
})

# Execute a tool
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.execute",
    "params": {
        "tool_id": "my_tool",
        "parameters": {"param1": "hello", "param2": 42},
        "agent_id": "my-agent"
    },
    "id": "2"
})
```

## Advanced Features

### Rate Limiting

Control tool execution frequency with Redis-based rate limiting:

```python
from agentcore.agent_runtime.services.rate_limiter import (
    get_rate_limiter,
    RateLimitExceeded,
)

# Get rate limiter instance
limiter = get_rate_limiter()
await limiter.connect()

# Check rate limit before execution
try:
    await limiter.check_rate_limit(
        tool_id="my_tool",
        limit=100,  # 100 requests
        window_seconds=60,  # per minute
        identifier="agent-123",  # optional per-agent limit
    )
    # Proceed with execution
except RateLimitExceeded as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after}s")
```

### Retry Logic with Exponential Backoff

Automatic retry with configurable backoff strategies:

```python
from agentcore.agent_runtime.services.retry_handler import (
    retry_with_backoff,
    BackoffStrategy,
)

async def unreliable_function():
    # May fail occasionally
    pass

# Retry with exponential backoff
result = await retry_with_backoff(
    unreliable_function,
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
    strategy=BackoffStrategy.EXPONENTIAL,
    jitter=True,
)
```

### Circuit Breaker

Prevent cascading failures with circuit breaker pattern:

```python
from agentcore.agent_runtime.services.retry_handler import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=60.0,  # Try recovery after 60s
)

result = await breaker.call(some_async_function)
```

### Parallel Execution

Execute multiple tools concurrently with dependency management:

```python
from agentcore.agent_runtime.services.parallel_executor import (
    ParallelExecutor,
    ParallelTask,
)

# Create parallel executor
executor = get_tool_executor()
parallel_exec = ParallelExecutor(executor)

# Define tasks with dependencies
tasks = [
    ParallelTask(
        task_id="task1",
        request=ToolExecutionRequest(...),
        dependencies=[],  # No dependencies
    ),
    ParallelTask(
        task_id="task2",
        request=ToolExecutionRequest(...),
        dependencies=["task1"],  # Depends on task1
    ),
]

# Execute in parallel
results = await parallel_exec.execute_parallel(tasks, max_concurrent=10)
```

### Batch Execution

Execute multiple tools in parallel without dependencies:

```python
requests = [
    ToolExecutionRequest(tool_id="tool1", ...),
    ToolExecutionRequest(tool_id="tool2", ...),
    ToolExecutionRequest(tool_id="tool3", ...),
]

results = await parallel_exec.execute_batch(requests, max_concurrent=5)
```

### A2A Authentication Integration

The Tool Integration Framework supports Agent-to-Agent (A2A) protocol context for distributed agent communication and authentication:

```python
from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest

# Create A2A context for agent-to-agent communication
a2a_context = A2AContext(
    source_agent="agent-123",
    target_agent="agent-456",
    trace_id="trace-abc-123",
    session_id="session-xyz-789",
    conversation_id="conv-001",
    timestamp="2024-01-15T10:30:00Z",
)

# Create JSON-RPC request with A2A context
request = JsonRpcRequest(
    method="tools.execute",
    params={
        "tool_id": "python_repl",
        "parameters": {"code": "print('hello')"},
        # agent_id is optional when A2A context is provided
    },
    id="req-001",
    a2a_context=a2a_context,
)

# The executor automatically extracts agent_id from A2A source_agent
# and propagates trace_id/session_id for distributed tracing
```

**A2A Context Propagation:**

- `source_agent` → Used as `agent_id` when not explicitly provided
- `trace_id` → Propagated through all tool executions for distributed tracing
- `session_id` → Maintained across tool execution chain
- `conversation_id` → Preserved for multi-turn agent interactions

**Backward Compatibility:**

The system maintains backward compatibility with legacy `execution_context` parameter:

```python
# Legacy approach (still supported)
result = await executor.execute_tool(
    tool_id="my_tool",
    parameters={},
    context=ExecutionContext(
        agent_id="agent-123",
        trace_id="trace-abc",
        session_id="session-xyz",
    )
)

# Modern A2A approach (preferred)
result = await jsonrpc_processor.process_request(request_with_a2a_context)
```

### Error Categorization and Recovery

Comprehensive error categorization system with hierarchical classification and recovery strategies:

```python
from agentcore.agent_runtime.tools.errors import (
    ToolErrorCategory,
    ToolErrorCode,
    ErrorRecoveryStrategy,
    categorize_error,
    get_error_metadata,
)

# All tool execution errors are automatically categorized
result = await executor.execute_tool("my_tool", {}, context)

if result.status == ToolExecutionStatus.FAILED:
    # Error metadata is automatically enriched
    print(f"Error Category: {result.metadata['error_category']}")
    print(f"Error Code: {result.metadata['error_code']}")
    print(f"User Message: {result.metadata['user_message']}")
    print(f"Recovery Strategy: {result.metadata['recovery_strategy']}")
    print(f"Is Retryable: {result.metadata['is_retryable']}")
    print(f"Recovery Guidance: {result.metadata['recovery_guidance']}")
```

**Error Categories:**

- **Client Errors (4xx equivalent):**
  - `VALIDATION_ERROR` - Invalid parameters, schema mismatch
  - `AUTHENTICATION_ERROR` - Missing or invalid credentials
  - `AUTHORIZATION_ERROR` - Insufficient permissions
  - `NOT_FOUND_ERROR` - Tool or resource not found
  - `RATE_LIMIT_ERROR` - Rate limit exceeded
  - `QUOTA_ERROR` - Usage quota exceeded

- **Execution Errors (5xx equivalent):**
  - `TIMEOUT_ERROR` - Execution timeout
  - `EXECUTION_ERROR` - Tool execution failed
  - `NETWORK_ERROR` - Network/connectivity issues
  - `RESOURCE_ERROR` - Insufficient resources (memory, disk, etc.)
  - `DEPENDENCY_ERROR` - External dependency failed
  - `INTERNAL_ERROR` - Unexpected internal error

- **Configuration Errors:**
  - `CONFIGURATION_ERROR` - Tool misconfiguration
  - `SANDBOX_ERROR` - Sandbox/security constraint violation

**Error Codes:**

Specific error codes organized by ranges:
- `TOOL_E1xxx` - Client errors (validation, auth, not found, rate limits, quotas)
- `TOOL_E2xxx` - Execution errors (timeout, network, resources, dependencies)
- `TOOL_E3xxx` - Configuration errors (misconfiguration, sandbox violations)

Examples:
- `TOOL_E1001` - Invalid parameters
- `TOOL_E1301` - Tool not found
- `TOOL_E1401` - Rate limit exceeded
- `TOOL_E2001` - Execution timeout
- `TOOL_E2201` - Network unreachable
- `TOOL_E3101` - Sandbox violation

**Recovery Strategies:**

- `RETRY` - Retry the operation immediately (for transient errors)
- `RETRY_WITH_BACKOFF` - Retry with exponential backoff (rate limits, network)
- `FALLBACK` - Use fallback tool or method
- `USER_INTERVENTION` - Require user to fix and retry (validation, auth)
- `FAIL` - Fail immediately, no recovery possible (quota, sandbox)

**Custom Error Handling:**

```python
from agentcore.agent_runtime.tools.errors import categorize_error

# Categorize any error
category, code, strategy = categorize_error(
    error_type="ConnectionError",
    error_message="Connection refused"
)

# Get user-friendly metadata
metadata = get_error_metadata(category, code, strategy)
```

### Distributed Tracing with OpenTelemetry

OpenTelemetry integration for comprehensive observability:

```python
from agentcore.agent_runtime.monitoring.tracing import (
    configure_tracing,
    get_tracer,
    add_span_attributes,
    add_span_event,
    record_exception,
    get_trace_id,
    get_span_id,
)

# Configure tracing (typically done at application startup)
configure_tracing(
    service_name="agentcore-runtime",
    service_version="1.0.0",
    otlp_endpoint="http://jaeger:4317",  # Optional OTLP exporter
    sample_rate=1.0,  # 0.0 to 1.0 (1.0 = 100% sampling)
    enable_console_export=False,  # Enable for debugging
)

# Create custom spans
tracer = get_tracer("my.tool.module")

with tracer.start_as_current_span("custom_operation") as span:
    # Add custom attributes
    add_span_attributes(
        tool_id="my_tool",
        agent_id="agent-123",
        custom_metadata="value"
    )

    # Add events to track significant points
    add_span_event("validation_started", parameter_count=5)

    try:
        # Your operation
        result = perform_operation()
        add_span_event("validation_completed", status="success")
    except Exception as e:
        # Record exceptions with context
        record_exception(e)
        raise

# Get current trace/span IDs for logging
trace_id = get_trace_id()
span_id = get_span_id()
logger.info(f"Operation completed", trace_id=trace_id, span_id=span_id)
```

**Automatic Span Creation:**

Tool execution automatically creates spans with:
- Span name: `tool.execute.{tool_id}`
- Attributes: `tool_id`, `user_id`, `agent_id`, `request_id`, `trace_id`
- Events: `hooks.before_completed`, `authentication.validated`, `parameters.validated`,
  `rate_limit.checked`, `tool.execution_started`, `tool.execution_completed`
- Exception recording for all failures

**Nested Spans:**

Spans automatically nest to create execution hierarchy:

```
tool.execute.python_repl
  ├─ hooks.before_completed
  ├─ authentication.validated
  ├─ parameters.validated
  ├─ tool.execution_started
  └─ tool.execution_completed
```

**Integration with A2A Protocol:**

Trace IDs from A2A context are automatically propagated:

```python
# A2A context trace_id becomes OpenTelemetry trace_id
a2a_context = A2AContext(
    source_agent="agent-123",
    trace_id="a2a-trace-xyz",  # Propagated to all spans
    session_id="session-abc",
)

# All tool executions will use the same trace_id
# enabling end-to-end distributed tracing across agents
```

**Viewing Traces:**

Export traces to Jaeger, Zipkin, or any OTLP-compatible backend:

```bash
# Run Jaeger for local development
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# View traces at http://localhost:16686
```

## Built-in Tools

The framework includes several built-in tools:

### Utility Tools
- **calculator** - Basic arithmetic operations
- **get_current_time** - Get current UTC time
- **echo** - Echo input message

### Search Tools
- **google_search** - Web search (mock implementation)
- **wikipedia_search** - Wikipedia article search
- **web_scrape** - Extract content from web pages

### Code Execution Tools
- **execute_python** - Execute Python code in sandbox
- **evaluate_expression** - Evaluate Python expressions

### API Client Tools
- **http_request** - Generic HTTP client
- **rest_get** - REST GET requests
- **rest_post** - REST POST requests
- **graphql_query** - GraphQL queries

## Tool Definition Schema

Complete tool definition includes:

```python
ToolDefinition(
    tool_id="unique_identifier",
    name="Human Readable Name",
    description="Detailed description",
    version="1.0.0",
    category=ToolCategory.CUSTOM,  # SEARCH, CODE_EXECUTION, API_CLIENT, etc.

    # Parameters
    parameters={
        "param_name": ToolParameter(
            name="param_name",
            type="string",  # string, integer, number, boolean, object, array
            description="Parameter description",
            required=True,
            default=None,
            enum=["option1", "option2"],  # Optional enum values
            min_value=0,  # For numbers
            max_value=100,
            min_length=1,  # For strings/arrays
            max_length=1000,
            pattern=r"^[a-z]+$",  # Regex pattern for strings
        ),
    },

    # Authentication
    auth_method=AuthMethod.NONE,  # NONE, API_KEY, OAUTH2, JWT
    auth_config={},  # Auth configuration

    # Execution settings
    timeout_seconds=30,
    is_retryable=True,
    is_idempotent=True,
    max_retries=3,

    # Rate limiting
    rate_limits={"requests_per_minute": 60},
    cost_per_execution=0.001,

    # Discovery
    capabilities=["capability1", "capability2"],
    tags=["tag1", "tag2"],
    requirements=["requirement1"],
    security_requirements=["security1"],

    # Metadata
    metadata={"key": "value"},
)
```

## JSON-RPC API Reference

### tools.list

List available tools with optional filtering.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.list",
    "params": {
        "category": "search",
        "capabilities": ["external_api"],
        "tags": ["research"]
    },
    "id": "1"
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "tools": [...],
        "count": 5
    },
    "id": "1"
}
```

### tools.get

Get detailed information about a specific tool.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.get",
    "params": {
        "tool_id": "calculator"
    },
    "id": "1"
}
```

### tools.execute

Execute a tool.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.execute",
    "params": {
        "tool_id": "calculator",
        "parameters": {
            "operation": "+",
            "a": 10,
            "b": 20
        },
        "agent_id": "my-agent",
        "execution_context": {
            "trace_id": "abc123"
        },
        "timeout_override": 60
    },
    "id": "1"
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "request_id": "...",
        "tool_id": "calculator",
        "status": "success",
        "result": 30,
        "execution_time_ms": 5,
        "timestamp": "2025-11-01T22:00:00Z",
        "retry_count": 0
    },
    "id": "1"
}
```

### tools.search

Search tools with comprehensive filters.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.search",
    "params": {
        "name_query": "search",
        "category": "search",
        "capabilities": ["external_api"],
        "tags": ["research"]
    },
    "id": "1"
}
```

### tools.execute_batch

Execute multiple tools in parallel without dependencies.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.execute_batch",
    "params": {
        "requests": [
            {
                "tool_id": "calculator",
                "parameters": {"operation": "+", "a": 10, "b": 20},
                "agent_id": "my-agent"
            },
            {
                "tool_id": "calculator",
                "parameters": {"operation": "*", "a": 5, "b": 3},
                "agent_id": "my-agent"
            }
        ],
        "max_concurrent": 10
    },
    "id": "1"
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "results": [
            {
                "request_id": "...",
                "tool_id": "calculator",
                "status": "success",
                "result": 30,
                "execution_time_ms": 5
            },
            {
                "request_id": "...",
                "tool_id": "calculator",
                "status": "success",
                "result": 15,
                "execution_time_ms": 4
            }
        ],
        "total_time_ms": 12,
        "successful_count": 2,
        "failed_count": 0
    },
    "id": "1"
}
```

### tools.execute_parallel

Execute multiple tools with dependency management.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.execute_parallel",
    "params": {
        "tasks": [
            {
                "task_id": "task1",
                "tool_id": "calculator",
                "parameters": {"operation": "+", "a": 5, "b": 3},
                "agent_id": "my-agent",
                "dependencies": []
            },
            {
                "task_id": "task2",
                "tool_id": "calculator",
                "parameters": {"operation": "*", "a": 2, "b": 4},
                "agent_id": "my-agent",
                "dependencies": ["task1"]
            }
        ],
        "max_concurrent": 10
    },
    "id": "1"
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "results": {
            "task1": {
                "request_id": "...",
                "tool_id": "calculator",
                "status": "success",
                "result": 8,
                "execution_time_ms": 5
            },
            "task2": {
                "request_id": "...",
                "tool_id": "calculator",
                "status": "success",
                "result": 8,
                "execution_time_ms": 4
            }
        },
        "total_time_ms": 15,
        "successful_count": 2,
        "failed_count": 0,
        "execution_order": ["task1", "task2"]
    },
    "id": "1"
}
```

### tools.execute_with_fallback

Execute a tool with automatic fallback on failure.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools.execute_with_fallback",
    "params": {
        "primary": {
            "tool_id": "google_search",
            "parameters": {"query": "test"},
            "agent_id": "my-agent"
        },
        "fallback": {
            "tool_id": "wikipedia_search",
            "parameters": {"query": "test"},
            "agent_id": "my-agent"
        }
    },
    "id": "1"
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "result": {
            "request_id": "...",
            "tool_id": "google_search",
            "status": "success",
            "result": {...},
            "execution_time_ms": 250
        },
        "used_fallback": false,
        "primary_error": null
    },
    "id": "1"
}
```

## Best Practices

### 1. Tool Design

- **Single Responsibility**: Each tool should do one thing well
- **Idempotency**: Design tools to be idempotent when possible
- **Error Handling**: Provide clear error messages
- **Validation**: Validate parameters thoroughly
- **Documentation**: Write clear descriptions and parameter docs

### 2. Parameter Design

- **Required vs Optional**: Minimize required parameters
- **Defaults**: Provide sensible defaults
- **Validation**: Use min/max values, patterns, enums
- **Types**: Use appropriate types (string, integer, boolean, etc.)

### 3. Performance

- **Timeouts**: Set reasonable timeout values
- **Retries**: Make tools retryable when appropriate
- **Rate Limiting**: Apply rate limits to external API calls
- **Caching**: Cache results when applicable

### 4. Security

- **Input Validation**: Always validate and sanitize inputs
- **Authentication**: Use appropriate auth methods
- **Secrets**: Never log or expose secrets
- **Sandboxing**: Execute untrusted code in sandboxes

### 5. Testing

- **Unit Tests**: Test tool logic independently
- **Integration Tests**: Test with real executor
- **Error Cases**: Test failure scenarios
- **Rate Limits**: Test rate limiting behavior

## Testing

### Unit Testing

```python
import pytest
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_registry import get_tool_registry

@pytest.fixture
def tool_executor():
    registry = get_tool_registry()
    return ToolExecutor(registry, enable_metrics=True)

@pytest.mark.asyncio
async def test_my_tool(tool_executor):
    request = ToolExecutionRequest(
        tool_id="my_tool",
        parameters={"param1": "test", "param2": 42},
        agent_id="test-agent",
    )

    result = await tool_executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result is not None
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_tool_via_jsonrpc(async_client):
    request = {
        "jsonrpc": "2.0",
        "method": "tools.execute",
        "params": {
            "tool_id": "my_tool",
            "parameters": {"param1": "test", "param2": 42},
            "agent_id": "test-agent",
        },
        "id": "1"
    }

    response = await async_client.post("/api/v1/jsonrpc", json=request)

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["status"] == "success"
```

## Troubleshooting

### Common Issues

**Tool not found**
- Ensure tool is registered with `registry.register_tool()`
- Check tool_id matches exactly
- Verify tool registration happens before execution

**Rate limit exceeded**
- Check rate limit configuration
- Use per-agent identifiers for fair limits
- Implement exponential backoff on retry

**Timeout errors**
- Increase `timeout_seconds` in tool definition
- Use `timeout_override` in execution request
- Optimize tool implementation

**Parameter validation failed**
- Check parameter types match definition
- Verify required parameters are provided
- Check min/max constraints

## Contributing

### Adding New Built-in Tools

1. Create tool implementation in `src/agentcore/agent_runtime/tools/`
2. Register in appropriate category (search, code_execution, api_client)
3. Add tests in `tests/agent_runtime/tools/`
4. Update documentation

### Implementing Custom Tool Categories

1. Add category to `ToolCategory` enum
2. Create category-specific adapters
3. Register with tool registry
4. Document usage patterns

## References

- **Models**: `src/agentcore/agent_runtime/models/tool_integration.py`
- **Registry**: `src/agentcore/agent_runtime/services/tool_registry.py`
- **Executor**: `src/agentcore/agent_runtime/services/tool_executor.py`
- **Rate Limiter**: `src/agentcore/agent_runtime/services/rate_limiter.py`
- **Retry Handler**: `src/agentcore/agent_runtime/services/retry_handler.py`
- **Parallel Executor**: `src/agentcore/agent_runtime/services/parallel_executor.py`
- **JSON-RPC**: `src/agentcore/agent_runtime/jsonrpc/tools_jsonrpc.py`

## License

See project LICENSE file for details.
