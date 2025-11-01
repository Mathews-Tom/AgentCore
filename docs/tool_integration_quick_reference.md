# Tool Integration Framework - Quick Reference

## Installation

No additional installation required - all features are built into AgentCore.

Optional: For rate limiting, ensure Redis is available:
```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or use existing Redis instance
export RATE_LIMITER_REDIS_URL=redis://localhost:6379/1
```

## Configuration

Add to `.env` file:
```bash
# Basic Configuration
TOOL_INTEGRATION_ENABLED=true
TOOL_MAX_RETRIES=3
TOOL_RETRY_STRATEGY=exponential

# Rate Limiting (optional)
RATE_LIMITER_ENABLED=false
RATE_LIMITER_REDIS_URL=redis://localhost:6379/1

# Parallel Execution
PARALLEL_MAX_CONCURRENT=10
```

## Quick Examples

### 1. Basic Tool Execution

```python
from agentcore.agent_runtime.services.tool_executor_factory import create_tool_executor
from agentcore.agent_runtime.models.tool_integration import ToolExecutionRequest

# Create executor
executor = create_tool_executor()

# Execute tool
request = ToolExecutionRequest(
    tool_id="calculator",
    parameters={"operation": "+", "a": 10, "b": 20},
    agent_id="my-agent",
)

result = await executor.execute(request)
print(f"Result: {result.result}")  # 30
```

### 2. Register Custom Tool

```python
from agentcore.agent_runtime.models.tool_integration import (
    ToolDefinition, ToolCategory, ToolParameter, AuthMethod
)
from agentcore.agent_runtime.services.tool_registry import get_tool_registry

# Define tool function
async def my_tool(name: str, count: int) -> str:
    return f"Hello {name}! Count: {count}"

# Create definition
tool_def = ToolDefinition(
    tool_id="my_tool",
    name="My Custom Tool",
    description="A custom tool example",
    version="1.0.0",
    category=ToolCategory.CUSTOM,
    parameters={
        "name": ToolParameter(
            name="name",
            type="string",
            description="Name to greet",
            required=True,
        ),
        "count": ToolParameter(
            name="count",
            type="number",
            description="Counter value",
            required=True,
            min_value=0,
            max_value=100,
        ),
    },
    auth_method=AuthMethod.NONE,
    is_retryable=True,
    max_retries=3,
)

# Register tool
registry = get_tool_registry()
registry.register_tool(tool_def, my_tool)
```

### 3. Enable Rate Limiting

```python
# Create executor with rate limiting
executor = create_tool_executor(
    settings_override={
        "rate_limiter_enabled": True,
        "rate_limiter_redis_url": "redis://localhost:6379/1",
    }
)

# Tools with rate_limits defined will be automatically limited
```

### 4. Configure Retry Strategy

```python
# Use linear backoff instead of exponential
executor = create_tool_executor(
    settings_override={
        "tool_retry_strategy": "linear",  # or "exponential", "fixed"
        "tool_max_retries": 5,
        "tool_retry_base_delay": 0.5,
    }
)
```

### 5. Parallel Execution

```python
from agentcore.agent_runtime.services.parallel_executor import (
    ParallelExecutor, ParallelTask
)

# Create parallel executor
parallel_exec = ParallelExecutor(executor)

# Define tasks
tasks = [
    ParallelTask(
        task_id="task1",
        request=ToolExecutionRequest(
            tool_id="tool1",
            parameters={...},
            agent_id="agent",
        ),
        dependencies=[],  # No dependencies
    ),
    ParallelTask(
        task_id="task2",
        request=ToolExecutionRequest(
            tool_id="tool2",
            parameters={...},
            agent_id="agent",
        ),
        dependencies=["task1"],  # Depends on task1
    ),
]

# Execute in parallel
results = await parallel_exec.execute_parallel(tasks, max_concurrent=10)
```

### 6. Batch Execution (No Dependencies)

```python
# Execute multiple tools in parallel without dependencies
requests = [
    ToolExecutionRequest(tool_id="tool1", parameters={...}, agent_id="agent"),
    ToolExecutionRequest(tool_id="tool2", parameters={...}, agent_id="agent"),
    ToolExecutionRequest(tool_id="tool3", parameters={...}, agent_id="agent"),
]

results = await parallel_exec.execute_batch(requests, max_concurrent=5)
```

### 7. Add Lifecycle Hooks

```python
def before_hook(request: ToolExecutionRequest) -> None:
    print(f"Executing tool: {request.tool_id}")

def after_hook(result: ToolResult) -> None:
    print(f"Tool completed: {result.tool_id} in {result.execution_time_ms}ms")

def error_hook(request: ToolExecutionRequest, error: Exception) -> None:
    print(f"Tool failed: {request.tool_id} - {error}")

executor.add_before_hook(before_hook)
executor.add_after_hook(after_hook)
executor.add_error_hook(error_hook)
```

### 8. JSON-RPC API Usage

```python
import httpx

# List available tools
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.list",
    "params": {"category": "custom"},
    "id": "1"
})

# Execute tool
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.execute",
    "params": {
        "tool_id": "my_tool",
        "parameters": {"name": "Alice", "count": 42},
        "agent_id": "my-agent"
    },
    "id": "2"
})

# Batch execution (parallel, no dependencies)
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.execute_batch",
    "params": {
        "requests": [
            {"tool_id": "tool1", "parameters": {...}, "agent_id": "agent"},
            {"tool_id": "tool2", "parameters": {...}, "agent_id": "agent"},
        ],
        "max_concurrent": 10
    },
    "id": "3"
})

# Parallel execution with dependencies
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.execute_parallel",
    "params": {
        "tasks": [
            {
                "task_id": "task1",
                "tool_id": "tool1",
                "parameters": {...},
                "agent_id": "agent",
                "dependencies": []
            },
            {
                "task_id": "task2",
                "tool_id": "tool2",
                "parameters": {...},
                "agent_id": "agent",
                "dependencies": ["task1"]
            }
        ],
        "max_concurrent": 10
    },
    "id": "4"
})

# Execute with fallback
response = await client.post("/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools.execute_with_fallback",
    "params": {
        "primary": {
            "tool_id": "primary_tool",
            "parameters": {...},
            "agent_id": "agent"
        },
        "fallback": {
            "tool_id": "fallback_tool",
            "parameters": {...},
            "agent_id": "agent"
        }
    },
    "id": "5"
})
```

## Built-in Tools

### Utility Tools
- `calculator`: Basic arithmetic operations
- `get_current_time`: Get current UTC time
- `echo`: Echo input message

### Search Tools
- `google_search`: Web search
- `wikipedia_search`: Wikipedia article search
- `web_scrape`: Extract content from web pages

### Code Execution Tools
- `execute_python`: Execute Python code in sandbox
- `evaluate_expression`: Evaluate Python expressions

### API Client Tools
- `http_request`: Generic HTTP client
- `rest_get`: REST GET requests
- `rest_post`: REST POST requests
- `graphql_query`: GraphQL queries

## Common Patterns

### Retry with Custom Strategy

```python
from agentcore.agent_runtime.services.retry_handler import (
    RetryHandler, BackoffStrategy
)

retry_handler = RetryHandler(
    max_retries=5,
    base_delay=1.0,
    max_delay=10.0,
    strategy=BackoffStrategy.EXPONENTIAL,
    jitter=True,
)

executor = ToolExecutor(
    registry=registry,
    retry_handler=retry_handler,
)
```

### Circuit Breaker

```python
from agentcore.agent_runtime.services.retry_handler import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=60.0,  # Try recovery after 60s
)

result = await breaker.call(some_async_function)
```

### Timeout Override

```python
# Override tool's default timeout for this execution
request = ToolExecutionRequest(
    tool_id="slow_tool",
    parameters={...},
    agent_id="agent",
    timeout_override=120,  # 120 seconds instead of default
)
```

## Error Handling

```python
from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus

result = await executor.execute(request)

if result.status == ToolExecutionStatus.SUCCESS:
    print(f"Success: {result.result}")
elif result.status == ToolExecutionStatus.FAILED:
    print(f"Failed: {result.error}")
    print(f"Error type: {result.error_type}")
elif result.status == ToolExecutionStatus.TIMEOUT:
    print(f"Timeout: {result.error}")
```

## Troubleshooting

### Rate Limit Exceeded
```python
if result.error_type == "RateLimitExceeded":
    retry_after = result.metadata.get("retry_after")
    print(f"Rate limited. Retry after {retry_after} seconds")
```

### Tool Not Found
```python
from agentcore.agent_runtime.services.tool_registry import get_tool_registry

registry = get_tool_registry()
available_tools = registry.list_tools()
print("Available tools:", [t.tool_id for t in available_tools])
```

### Check Configuration
```python
from agentcore.agent_runtime.config.settings import get_settings

settings = get_settings()
print(f"Retry strategy: {settings.tool_retry_strategy}")
print(f"Max retries: {settings.tool_max_retries}")
print(f"Rate limiter enabled: {settings.rate_limiter_enabled}")
```

## Testing

```python
import pytest
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_registry import ToolRegistry

@pytest.fixture
def executor() -> ToolExecutor:
    registry = ToolRegistry()
    # Register test tools
    return ToolExecutor(registry)

@pytest.mark.asyncio
async def test_my_tool(executor: ToolExecutor):
    request = ToolExecutionRequest(
        tool_id="my_tool",
        parameters={"name": "test", "count": 1},
        agent_id="test-agent",
    )

    result = await executor.execute(request)
    assert result.status == ToolExecutionStatus.SUCCESS
```

## Performance Tips

1. **Use parallel execution** for independent tools
2. **Enable rate limiting** for external API calls
3. **Configure appropriate timeouts** per tool
4. **Use jitter** in retry logic for distributed systems
5. **Monitor execution metrics** via observability hooks
6. **Batch similar requests** to reduce overhead
7. **Cache tool results** when appropriate

## Further Reading

- Complete Guide: `docs/tool_integration_guide.md`
- Implementation Details: `docs/tool_integration_implementation_summary.md`
- Source Code: `src/agentcore/agent_runtime/services/`
- Tests: `tests/agent_runtime/services/` and `tests/agent_runtime/integration/`
