# Tool Integration Framework Architecture

**Version:** 1.0.0
**Last Updated:** 2025-01-13
**Status:** Production

## Table of Contents

- [Overview](#overview)
- [Architectural Principles](#architectural-principles)
- [Layered Architecture](#layered-architecture)
- [Component Diagrams](#component-diagrams)
- [Data Flow](#data-flow)
- [Security Architecture](#security-architecture)
- [Scalability & Performance](#scalability--performance)
- [Integration Points](#integration-points)

## Overview

The Tool Integration Framework implements a **layered architecture pattern** that provides clear separation of concerns, enabling agents to interact with external services through a standardized, secure, and observable interface.

### Design Goals

1. **Standardization:** Consistent interface for all tool integrations
2. **Extensibility:** Easy addition of new tools without core changes
3. **Reliability:** Comprehensive error handling and retry logic
4. **Security:** Authentication, authorization, and sandboxing
5. **Observability:** Detailed logging, metrics, and tracing
6. **Performance:** Low overhead with high concurrency support

### Technology Stack

- **Runtime:** Python 3.12+ with asyncio
- **Framework:** FastAPI with JSON-RPC 2.0
- **Database:** PostgreSQL (tool execution logs)
- **Cache/Rate Limiting:** Redis
- **Validation:** Pydantic v2
- **Sandboxing:** Docker containers
- **Monitoring:** Prometheus + structlog
- **Tracing:** OpenTelemetry (future)

## Architectural Principles

### 1. Separation of Concerns

Each layer has a distinct responsibility:

- **Interface Layer:** Defines contracts and abstractions
- **Registry Layer:** Manages tool discovery and metadata
- **Execution Layer:** Handles lifecycle and cross-cutting concerns
- **Adapter Layer:** Implements specific tool logic

### 2. Dependency Inversion

Higher layers depend on abstractions, not concrete implementations:

```python
# Good: Depend on abstraction
class ToolExecutor:
    def __init__(self, registry: ToolRegistry): ...

# Not: Depend on concrete implementation
class ToolExecutor:
    def __init__(self, tools: list[GoogleSearchTool, ...]): ...
```

### 3. Open/Closed Principle

Framework is open for extension (add new tools) but closed for modification (core framework stable):

```python
# Extend by implementing Tool interface
class CustomTool(Tool):
    async def execute(self, parameters, context):
        # Custom implementation
        pass
```

### 4. Single Responsibility

Each component has one reason to change:

- `Tool`: Implement tool-specific logic
- `ToolRegistry`: Manage tool catalog
- `ToolExecutor`: Handle execution lifecycle
- `ParameterValidator`: Validate parameters

### 5. Fail-Fast

Validation and error checking happen early:

- Parameter validation before execution
- Authentication check before API calls
- Resource availability check before sandboxing

## Layered Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      Application Layer                         │
│            (Agent Runtime, JSON-RPC Handlers)                  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                    Tool Interface Layer                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Tool (ABC)                                             │  │
│  │  - metadata: ToolDefinition                             │  │
│  │  + execute(parameters, context) → ToolResult            │  │
│  │  + validate_parameters(parameters) → (bool, str?)       │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  ExecutionContext                                        │  │
│  │  - user_id, agent_id, trace_id, session_id, request_id │  │
│  │  - metadata: dict[str, Any]                             │  │
│  │  + to_dict() → dict                                     │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                    Tool Registry Layer                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  ToolRegistry                                           │  │
│  │  + register(tool: Tool)                                 │  │
│  │  + unregister(tool_id: str)                             │  │
│  │  + get(tool_id: str) → Tool                             │  │
│  │  + list() → list[Tool]                                  │  │
│  │  + search(category?, tags?, query?) → list[Tool]        │  │
│  │  + get_capabilities() → dict                            │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                   Tool Execution Layer                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  ToolExecutor                                           │  │
│  │  + execute_tool(tool_id, parameters, context)           │  │
│  │  - _authenticate(tool, context)                         │  │
│  │  - _check_rate_limit(tool, context)                     │  │
│  │  - _execute_with_retry(tool, parameters, context)       │  │
│  │  - _record_execution(result)                            │  │
│  │  - _emit_metrics(result)                                │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌───────────────────┐  ┌──────────────────┐                  │
│  │  RateLimiter      │  │  AuthService     │                  │
│  │  + check_limit()  │  │  + verify_token()│                  │
│  │  + consume_token()│  │  + get_api_key() │                  │
│  └───────────────────┘  └──────────────────┘                  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                    Tool Adapters Layer                         │
│  ┌──────────────────┐  ┌───────────────────┐                  │
│  │  Search Tools    │  │  API Client Tools │                  │
│  │  - GoogleSearch  │  │  - HttpRequest    │                  │
│  │  - Wikipedia     │  │  - RestGet/Post   │                  │
│  │  - WebScrape     │  │  - GraphQLQuery   │                  │
│  └──────────────────┘  └───────────────────┘                  │
│  ┌──────────────────┐  ┌───────────────────┐                  │
│  │  Code Exec Tools │  │  Utility Tools    │                  │
│  │  - ExecutePython │  │  - Calculator     │                  │
│  │  - EvaluateExpr  │  │  - CurrentTime    │                  │
│  │                  │  │  - Echo           │                  │
│  └──────────────────┘  └───────────────────┘                  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                    External Services Layer                     │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  Google API  │  │  REST APIs │  │  Docker (Sandbox)    │  │
│  │  Wikipedia   │  │  GraphQL   │  │  Execution Env       │  │
│  └──────────────┘  └────────────┘  └──────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

#### 1. Tool Interface Layer

**Responsibility:** Define abstractions and contracts

**Components:**
- `Tool` (ABC): Base interface all tools implement
- `ExecutionContext`: Runtime context for execution
- `ToolResult`: Standardized result format
- `ToolDefinition`: Tool metadata and configuration
- `ToolParameter`: Parameter schema definition

**Key Files:**
- `src/agentcore/agent_runtime/tools/base.py`
- `src/agentcore/agent_runtime/models/tool_integration.py`

#### 2. Tool Registry Layer

**Responsibility:** Tool discovery and metadata management

**Components:**
- `ToolRegistry`: Central catalog of available tools
- Tool search and filtering
- Version management
- Capability queries

**Key Files:**
- `src/agentcore/agent_runtime/tools/registry.py`

**Features:**
- In-memory tool catalog
- Fast lookup by tool_id
- Search by category, tags, capabilities
- Thread-safe registration/unregistration

#### 3. Tool Execution Layer

**Responsibility:** Lifecycle management and cross-cutting concerns

**Components:**
- `ToolExecutor`: Main execution orchestrator
- `RateLimiter`: Rate limiting enforcement
- `AuthService`: Authentication handling
- `RetryHandler`: Retry logic with exponential backoff

**Key Files:**
- `src/agentcore/agent_runtime/tools/executor.py`

**Features:**
- Pre-execution validation
- Authentication/authorization
- Rate limiting
- Retry with backoff
- Resource monitoring
- Metrics collection
- Distributed tracing
- Structured logging

#### 4. Tool Adapters Layer

**Responsibility:** Implement specific tool logic

**Components:**
- Search tools (Google, Wikipedia, WebScrape)
- Code execution tools (Python, Expression)
- API client tools (HTTP, REST, GraphQL)
- Utility tools (Calculator, Time, Echo)

**Key Files:**
- `src/agentcore/agent_runtime/tools/builtin/search_tools.py`
- `src/agentcore/agent_runtime/tools/builtin/code_execution_tools.py`
- `src/agentcore/agent_runtime/tools/builtin/api_tools.py`
- `src/agentcore/agent_runtime/tools/builtin/utility_tools.py`

## Component Diagrams

### Tool Execution Sequence

```
Agent → ToolExecutor: execute_tool(tool_id, params, context)
ToolExecutor → ToolRegistry: get(tool_id)
ToolRegistry → ToolExecutor: Tool instance
ToolExecutor → Tool: validate_parameters(params)
Tool → ToolExecutor: (True, None)
ToolExecutor → RateLimiter: check_limit(tool, context)
RateLimiter → ToolExecutor: OK
ToolExecutor → AuthService: authenticate(tool, context)
AuthService → ToolExecutor: credentials
ToolExecutor → Tool: execute(params, context)
Tool → ExternalService: API call / code execution / search
ExternalService → Tool: response
Tool → ToolExecutor: ToolResult(SUCCESS, data)
ToolExecutor → Database: record_execution(result)
ToolExecutor → Metrics: emit_metrics(result)
ToolExecutor → Agent: ToolResult
```

### Parameter Validation Flow

```
Tool.execute() → validate_parameters(params)
  ↓
Check required parameters present
  ↓
For each parameter:
  ↓
  Check type matches definition (_validate_type)
    - string, number, integer, boolean, array, object
    - Special handling for integer vs float
  ↓
  Check enum constraint (if defined)
  ↓
  Check string constraints:
    - min_length, max_length
    - pattern (regex validation via _validate_pattern)
  ↓
  Check number constraints:
    - min_value, max_value
  ↓
  Check array constraints:
    - min_length, max_length
  ↓
Return (is_valid, error_message?)
```

### Error Handling Flow

```
Tool.execute()
  ↓
try:
  validate_parameters()
    ↓ FAIL
    return ToolResult(FAILED, validation_error)
  ↓
  perform_operation()
    ↓ Exception
    catch Exception
      ↓
      log_error()
      ↓
      return ToolResult(FAILED, error_message)
  ↓ SUCCESS
  return ToolResult(SUCCESS, data)
```

## Data Flow

### Tool Registration Flow

```
1. Tool Implementation
   ↓
   class GoogleSearchTool(Tool):
       def __init__(self):
           metadata = ToolDefinition(...)
           super().__init__(metadata)
   ↓
2. Registration
   ↓
   registry = ToolRegistry()
   tool = GoogleSearchTool()
   await registry.register(tool)
   ↓
3. Storage
   ↓
   registry._tools[tool.metadata.tool_id] = tool
   ↓
4. Indexing
   ↓
   registry._by_category[category].append(tool)
   registry._by_tags[tag].append(tool)
```

### Tool Execution Flow

```
1. Request Reception
   ↓
   JSON-RPC Request:
   {
     "method": "tool.execute",
     "params": {
       "tool_id": "google_search",
       "parameters": {"query": "AgentCore"},
       "context": {...}
     }
   }
   ↓
2. Tool Lookup
   ↓
   tool = await registry.get(tool_id)
   ↓
3. Pre-Execution Checks
   ↓
   - Parameter validation
   - Rate limit check
   - Authentication
   ↓
4. Execution
   ↓
   result = await tool.execute(params, context)
   ↓
5. Post-Execution
   ↓
   - Record to database
   - Emit metrics
   - Update rate limiter
   ↓
6. Response
   ↓
   JSON-RPC Response:
   {
     "result": {
       "status": "success",
       "result": {...},
       "execution_time_ms": 245.3
     }
   }
```

## Security Architecture

### Authentication Flow

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐
│  Agent   │────>│ ToolExecutor │────>│ AuthService │
└──────────┘     └──────────────┘     └─────────────┘
                        │                      │
                        │  Tool requires       │
                        │  API_KEY auth        │
                        │                      │
                        │  get_credentials()   │
                        │─────────────────────>│
                        │                      │
                        │  API Key from        │
                        │  env/vault           │
                        │<─────────────────────│
                        │                      │
                        V                      V
                 Add to request          Audit log
                 headers/params          authentication
```

### Sandboxing Architecture

Code execution tools run in isolated Docker containers:

```
┌────────────────────────────────────────────────────┐
│  Host System                                       │
│  ┌──────────────────────────────────────────────┐ │
│  │  ExecutePythonTool                           │ │
│  │  ┌────────────────────────────────────────┐  │ │
│  │  │  Docker Container (Sandbox)            │  │ │
│  │  │  ┌──────────────────────────────────┐  │  │ │
│  │  │  │  Python Runtime                  │  │  │ │
│  │  │  │  - Memory Limit: 512MB           │  │  │ │
│  │  │  │  - CPU Limit: 1.0 core           │  │  │ │
│  │  │  │  - Network: Restricted           │  │  │ │
│  │  │  │  - Filesystem: Read-only         │  │  │ │
│  │  │  │  - Timeout: 30 seconds           │  │  │ │
│  │  │  └──────────────────────────────────┘  │  │ │
│  │  └────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

### Input Validation

Defense in depth with multiple validation layers:

1. **JSON Schema Validation:** Pydantic models validate request structure
2. **Parameter Type Validation:** Strict type checking in `_validate_type()`
3. **Pattern Validation:** Regex validation for strings via `_validate_pattern()`
4. **Business Logic Validation:** Tool-specific validation in `execute()`
5. **Sanitization:** SQL injection, XSS prevention in tool implementations

## Scalability & Performance

### Horizontal Scaling

```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            V              V              V
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │Instance 1│   │Instance 2│   │Instance 3│
     │          │   │          │   │          │
     │ToolExec  │   │ToolExec  │   │ToolExec  │
     │Registry  │   │Registry  │   │Registry  │
     └────┬─────┘   └────┬─────┘   └────┬─────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
                         V
              ┌──────────────────┐
              │  Shared Services │
              │  - Redis (Rate)  │
              │  - PostgreSQL    │
              │  - Metrics       │
              └──────────────────┘
```

### Performance Optimizations

1. **Async I/O:** All operations use async/await
2. **Connection Pooling:** Database and HTTP clients pool connections
3. **Registry Caching:** In-memory tool catalog
4. **Lazy Loading:** Tools loaded on first use
5. **Batching:** Metrics and logs batched for efficiency

### Resource Management

```python
# Resource limits per tool
ToolDefinition(
    timeout_seconds=30,        # Execution timeout
    max_retries=3,             # Retry attempts
    rate_limits={              # Rate limiting
        "calls_per_minute": 60,
        "calls_per_hour": 1000
    },
    cost_per_execution=0.01    # Cost tracking
)
```

## Integration Points

### Agent Runtime Integration

```python
# In ReAct engine
from agentcore.agent_runtime.tools.executor import ToolExecutor

class ReactEngine:
    def __init__(self):
        self.tool_executor = ToolExecutor()

    async def execute_action(self, action: str, params: dict):
        result = await self.tool_executor.execute_tool(
            tool_id=action,
            parameters=params,
            context=self.context
        )
        return result
```

### JSON-RPC Integration

```python
# In JSON-RPC handler
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

@register_jsonrpc_method("tool.execute")
async def handle_tool_execute(request: JsonRpcRequest) -> dict[str, Any]:
    tool_id = request.params["tool_id"]
    parameters = request.params["parameters"]
    context = ExecutionContext(**request.params.get("context", {}))

    result = await tool_executor.execute_tool(
        tool_id=tool_id,
        parameters=parameters,
        context=context
    )

    return result.model_dump()
```

### Database Schema

```sql
CREATE TABLE tool_executions (
    id UUID PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    tool_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    agent_id VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    parameters JSONB,
    result JSONB,
    error TEXT,
    error_type VARCHAR(255),
    execution_time_ms FLOAT,
    retry_count INTEGER DEFAULT 0,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    trace_id VARCHAR(255),
    session_id VARCHAR(255),
    metadata JSONB,
    INDEX idx_tool_id (tool_id),
    INDEX idx_user_id (user_id),
    INDEX idx_agent_id (agent_id),
    INDEX idx_status (status),
    INDEX idx_timestamp (timestamp DESC)
);
```

## Monitoring & Observability

### Metrics

```python
# Prometheus metrics
tool_execution_total = Counter(
    "tool_execution_total",
    "Total tool executions",
    ["tool_id", "status"]
)

tool_execution_duration_seconds = Histogram(
    "tool_execution_duration_seconds",
    "Tool execution duration",
    ["tool_id"]
)

tool_validation_failures_total = Counter(
    "tool_validation_failures_total",
    "Parameter validation failures",
    ["tool_id", "parameter"]
)
```

### Logging

```python
# Structured logging with context
self.logger.info(
    "tool_execution_completed",
    tool_id=tool_id,
    request_id=context.request_id,
    user_id=context.user_id,
    status=result.status,
    execution_time_ms=result.execution_time_ms,
    retry_count=result.retry_count
)
```

## Related Documentation

- [README.md](./README.md) - Framework overview and quick start
- [developer-guide.md](./developer-guide.md) - Developer guide
- [Tool Integration Specification](../specs/tool-integration/spec.md)
- [Implementation Plan](../specs/tool-integration/plan.md)
