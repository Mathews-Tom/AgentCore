# Tool Integration Framework Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-15
**Specification:** `docs/specs/tool-integration/spec.md`
**Research:** `docs/research/multi-tool-integration.md`
**Component ID:** TOOL-001

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → This Plan**

1. **Research Foundation:** docs/research/multi-tool-integration.md
   - Technical architecture and framework design
   - Code examples for Tool interface, ToolRegistry, adapters
   - Implementation patterns for search, code execution, API tools
   - Success metrics and value analysis
   - Integration strategy with phased approach

2. **Formal Specification:** docs/specs/tool-integration/spec.md
   - Functional requirements (FR-1 through FR-5)
   - Non-functional requirements (Performance, Reliability, Scalability, Security, Observability)
   - 5 key features with flows and acceptance criteria
   - API specification and data models
   - Success validation approach

### Related Documentation

**System Context:**

- Project Standards: `CLAUDE.md` - AgentCore development guidelines
- Architecture: `docs/agentcore-architecture-and-development-plan.md`

**Related Specifications:**

- MOD-001: `docs/specs/modular-agent-core/spec.md` - Executor module depends on tool framework
- CTX-001: Context Management System (future integration)
- MEM-001: Memory Management System (future integration)

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable agents to interact with the real world through standardized tool integrations, transforming AgentCore from a pure reasoning system into a capable system that can search the web, execute code, call APIs, and process data.

**Value Proposition:**

- **Increased Agent Capability:** Enable 3-5x more complex workflows through tool access
- **Improved Reliability:** +20-30% task completion rate through standardized error handling
- **Faster Integration:** 50% reduction in time to integrate new tools
- **Better Observability:** 80% reduction in debugging time through centralized logging
- **Cost Control:** Prevent runaway costs through rate limiting and quota management

**Target Users:**

- **Agent Developers:** Building agents that need external tool access
- **Tool Providers:** Integrating services into AgentCore
- **Platform Operators:** Managing tool infrastructure and costs
- **Enterprise Users:** Requiring auditable and secure tool usage

### Technical Approach

**Architecture Pattern:** Layered Architecture with Plugin System

- **Tool Registry Layer:** Centralized discovery, metadata management, version control
- **Tool Interface Layer:** Standardized abstractions for invocation, validation, error handling
- **Tool Execution Layer:** Lifecycle management, authentication, rate limiting, observability
- **Tool Adapters Layer:** Specific implementations (search, code execution, API clients)

**Technology Stack:**

- **Runtime:** Python 3.12+ with asyncio (existing AgentCore stack)
- **Framework:** FastAPI with JSON-RPC 2.0 (A2A protocol)
- **Database:** PostgreSQL with asyncpg (tool execution logs)
- **Rate Limiting:** Redis with token bucket algorithm
- **Validation:** Pydantic v2 (parameter schemas)
- **Sandboxing:** Docker (code execution isolation)
- **Monitoring:** Prometheus + Grafana (existing stack)
- **Tracing:** OpenTelemetry (distributed tracing)

**Implementation Strategy:** 4 phases over 6 weeks

1. **Foundation (Weeks 1-2):** Core framework, interfaces, registry
2. **Built-in Tools (Week 3):** Search, code execution, API adapters
3. **JSON-RPC Integration (Week 4):** A2A protocol endpoints
4. **Advanced Features (Weeks 5-6):** Rate limiting, retry, monitoring, production hardening

### Key Success Metrics

**Service Level Objectives (SLOs):**

- **Availability:** 99.9% uptime for tool execution service
- **Response Time:** <100ms framework overhead (excluding tool execution)
- **Throughput:** 1000 concurrent tool executions per instance
- **Error Rate:** <5% for properly configured tools

**Key Performance Indicators (KPIs):**

- **Tool Adoption:** 80%+ of agent tasks use at least one tool
- **Tool Success Rate:** 95%+ successful executions
- **Integration Time:** <1 day to integrate new tool
- **Error Recovery:** 90%+ of retryable errors recovered automatically
- **Cost Efficiency:** Rate limiting prevents API cost overruns

---

## 💻 Code Examples & Patterns

### Repository Patterns (from Research)

**Pattern 1: Tool Interface (Abstract Base Class)**

- **Source:** `docs/research/multi-tool-integration.md` (lines 119-182)
- **Application:** All tools implement standardized interface
- **Usage Example:**

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    """Base interface for all tools."""

    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata

    @abstractmethod
    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext
    ) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    async def validate_parameters(
        self,
        parameters: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate parameters before execution."""
        for param in self.metadata.parameters:
            if param.required and param.name not in parameters:
                return False, f"Missing required parameter: {param.name}"
        return True, None
```

- **Adaptation Notes:** Integrate with A2A context for trace_id propagation, add async error handling

**Pattern 2: Registry Pattern (Centralized Discovery)**

- **Source:** `docs/research/multi-tool-integration.md` (lines 184-250)
- **Application:** Tool discovery and metadata management
- **Usage Example:**

```python
class ToolRegistry:
    """Central registry for tool discovery and management."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        tool_id = tool.metadata.tool_id
        self._tools[tool_id] = tool

        category = tool.metadata.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool_id)

    def search(self, query: str, category: str | None = None) -> list[Tool]:
        """Search tools by query string."""
        # Implementation with fuzzy matching
        pass
```

- **Adaptation Notes:** Add database persistence for tool metadata, implement version management

**Pattern 3: Adapter Pattern (Tool-Specific Implementations)**

- **Source:** `docs/research/multi-tool-integration.md` (lines 254-344 for GoogleSearchTool, 346-435 for PythonExecutionTool)
- **Application:** Specific tool implementations wrapping external services
- **Usage Example:**

```python
class GoogleSearchTool(Tool):
    """Google search integration."""

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext
    ) -> ToolResult:
        start_time = time.time()

        try:
            # Validate parameters
            is_valid, error = await self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    tool_id=self.metadata.tool_id,
                    success=False,
                    error=error,
                    execution_time_ms=0
                )

            # Execute search
            query = parameters["query"]
            results = await self._perform_search(query)

            return ToolResult(
                tool_id=self.metadata.tool_id,
                success=True,
                result=results,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            return ToolResult(
                tool_id=self.metadata.tool_id,
                success=False,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
```

- **Adaptation Notes:** Add rate limiting checks before execution, integrate with tracing

### Implementation Reference Examples

**From Research (JSON-RPC Integration):**

```python
@register_jsonrpc_method("tools.list")
async def list_tools(request: JsonRpcRequest) -> dict[str, Any]:
    """List available tools."""
    category = request.params.get("category")
    tools = (
        tool_registry.list_by_category(category)
        if category
        else list(tool_registry._tools.values())
    )

    return {
        "tools": [
            {
                "tool_id": tool.metadata.tool_id,
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "category": tool.metadata.category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required
                    }
                    for p in tool.metadata.parameters
                ]
            }
            for tool in tools
        ]
    }
```

### Key Takeaways from Examples

- **Standardization:** All tools follow same interface, enabling polymorphic usage
- **Async-First:** All tool operations are async to avoid blocking
- **Error Handling:** Tools always return ToolResult with success/error status
- **Validation:** Parameter validation happens before execution (fast fail)
- **Observability:** Execution time tracked, trace IDs propagated
- **Anti-patterns to Avoid:**
  - Don't mix sync and async code
  - Don't swallow exceptions without logging
  - Don't hardcode credentials in tool implementations
  - Don't skip parameter validation

### New Patterns to Create

**Patterns This Implementation Will Establish:**

1. **Rate Limiter Pattern**
   - **Purpose:** Distributed rate limiting with token bucket algorithm
   - **Location:** `src/agentcore/tools/rate_limiter.py`
   - **Reusability:** Can be used by any component needing rate limiting

2. **Tool Executor Pattern**
   - **Purpose:** Lifecycle management for tool invocation (auth, retry, logging)
   - **Location:** `src/agentcore/tools/executor.py`
   - **Reusability:** Template for any async execution with retry/timeout

3. **Sandbox Manager Pattern**
   - **Purpose:** Docker container management for code execution
   - **Location:** `src/agentcore/tools/sandbox.py`
   - **Reusability:** Can be used for any containerized execution

---

## 🔧 Technology Stack

### Recommended Stack (from Research & Spec)

**Based on research from:** `docs/research/multi-tool-integration.md` and existing AgentCore stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | Existing AgentCore stack, excellent async support |
| Framework | FastAPI | Latest | Existing stack, JSON-RPC 2.0 support, high performance |
| Database | PostgreSQL | 15+ | Existing stack, JSONB support for flexible tool metadata |
| Caching/Rate Limiting | Redis | 7+ | Distributed state, pub/sub, high performance |
| Validation | Pydantic | 2.x | Existing stack, excellent schema validation |
| Sandboxing | Docker | 24+ | Industry standard, security isolation, resource limits |
| Secret Management | HashiCorp Vault | 1.15+ | Secure credential storage, rotation support |
| Monitoring | Prometheus | Latest | Existing stack, metrics collection |
| Dashboards | Grafana | Latest | Existing stack, visualization |
| Tracing | OpenTelemetry | Latest | Distributed tracing, vendor-neutral |
| Testing | pytest-asyncio | Latest | Async test support, testcontainers for integration |

**Key Technology Decisions:**

1. **Redis for Rate Limiting:**
   - **Rationale:** Atomic operations (INCR), expiration support, distributed state
   - **Alternative Considered:** In-memory (doesn't scale), PostgreSQL (too slow)
   - **Research Citation:** Token bucket algorithm requires sub-millisecond increments

2. **Docker for Code Execution:**
   - **Rationale:** Security isolation, resource limits, widely deployed
   - **Alternative Considered:** Pyodide (browser-based), restricted-python (less secure)
   - **Research Citation:** Industry standard for sandboxing, mature security model

3. **Pydantic for Parameter Validation:**
   - **Rationale:** Already in stack, type-safe, excellent error messages
   - **Alternative Considered:** JSON Schema (more verbose), Marshmallow (legacy)
   - **Research Citation:** Pydantic v2 has 10x validation performance improvement

### Alternatives Considered

**Option 2: gRPC instead of JSON-RPC**

- **Pros:** Higher performance, streaming support, strong typing
- **Cons:** More complex, not web-friendly, requires proto files
- **Why Not Chosen:** AgentCore standardized on JSON-RPC 2.0 for A2A protocol

**Option 3: SQLite instead of PostgreSQL**

- **Pros:** Simpler deployment, no separate server
- **Cons:** No horizontal scaling, limited concurrent writes
- **Why Not Chosen:** AgentCore needs multi-instance deployment for scale

### Alignment with Existing System

**From `CLAUDE.md` and system context:**

**Consistent With:**

- Python 3.12+ with async/await patterns
- FastAPI framework and JSON-RPC 2.0
- PostgreSQL with async SQLAlchemy
- Pydantic v2 for validation
- pytest-asyncio with 90%+ coverage requirement
- Prometheus + Grafana monitoring

**New Additions:**

- **Redis:** Adding Redis for rate limiting (new dependency)
  - Deployment: docker-compose.dev.yml needs Redis service
  - Configuration: REDIS_URL environment variable
- **Docker API:** Using Docker SDK for Python for sandbox management
  - Security: Docker daemon access must be restricted
  - Configuration: Docker socket mount in production
- **OpenTelemetry:** Adding distributed tracing (enhancement to existing monitoring)

**Migration Considerations:**

- None - this is a new component, no existing tool system to migrate

---

## 🏗️ Architecture Design

### System Context (AgentCore Architecture)

**Existing System Architecture:**

AgentCore is an open-source orchestration framework implementing Google's A2A (Agent2Agent) protocol v0.2. It provides JSON-RPC 2.0 compliant infrastructure for agent communication, discovery, task management, and real-time messaging.

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                        AgentCore System                          │
│                                                                   │
│  ┌────────────────┐         ┌────────────────┐                  │
│  │  Agent Manager │◄────────┤  Task Manager  │                  │
│  │  (Discovery)   │         │  (Execution)   │                  │
│  └────────┬───────┘         └────────┬───────┘                  │
│           │                          │                           │
│           │  ┌───────────────────────▼─────┐                    │
│           └──► JSON-RPC Handler            │                    │
│              │ (Method Registry)            │                    │
│              └───────────┬──────────────────┘                    │
│                          │                                       │
│                ┌─────────▼─────────┐                             │
│                │  Database Layer   │                             │
│                │  (PostgreSQL)     │                             │
│                └───────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

**Integration Points for Tool Framework:**

- **JSON-RPC Handler:** Register `tools.list` and `tools.execute` methods
- **Agent Manager:** Tools advertised in agent capabilities
- **Task Manager:** Tools invoked during task execution
- **Database Layer:** Store tool execution logs

### Component Architecture

**Architecture Pattern:** Layered Architecture with Plugin System

- **Rationale:** Separation of concerns between framework and adapters, easy to add new tools without modifying core
- **Alignment:** Fits AgentCore's modular design philosophy

**System Design:**

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                     Tool Integration Framework                   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Tool Registry Layer                    │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │ Discovery  │  │   Metadata   │  │     Versions    │  │   │
│  │  │ & Search   │  │  Management  │  │   & Compat      │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ▲                                     │
│  ┌─────────────────────────┴────────────────────────────────┐   │
│  │                  Tool Interface Layer                     │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │ Tool Base  │  │  Parameter   │  │     Result      │  │   │
│  │  │ Interface  │  │  Validation  │  │   Formatting    │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ▲                                     │
│  ┌─────────────────────────┴────────────────────────────────┐   │
│  │                 Tool Execution Layer                      │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │ Lifecycle  │  │     Auth     │  │  Rate Limiting  │  │   │
│  │  │ Management │  │ & Security   │  │   & Quotas      │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │   Retry    │  │   Logging    │  │    Tracing      │  │   │
│  │  │  & Timeout │  │  & Metrics   │  │  (OpenTel)      │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ▲                                     │
│  ┌─────────────────────────┴────────────────────────────────┐   │
│  │                   Tool Adapters Layer                     │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │  Search    │  │     Code     │  │       API       │  │   │
│  │  │   Tools    │  │  Execution   │  │     Clients     │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │   │
│  │  ┌────────────┐  ┌──────────────┐                       │   │
│  │  │    File    │  │   Database   │       [Extensible]    │   │
│  │  │ Operations │  │    Tools     │                       │   │
│  │  └────────────┘  └──────────────┘                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  External Dependencies:                                          │
│  ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐    │
│  │ PostgreSQL│  │  Redis   │  │   Docker   │  │ External │    │
│  │  (Logs)   │  │ (Limits) │  │ (Sandbox)  │  │   APIs   │    │
│  └───────────┘  └──────────┘  └────────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

```plaintext
Agent Request → JSON-RPC (tools.execute)
                    ↓
              ToolExecutor
                    ↓
       ┌────────────┴─────────────┐
       ↓                          ↓
  Get Tool from Registry    Check Rate Limit (Redis)
       ↓                          ↓
  Validate Parameters        Authenticate
       ↓                          ↓
  Execute Tool Adapter ──────────→ External Service
       ↓
  Log Execution (PostgreSQL)
       ↓
  Emit Metrics (Prometheus)
       ↓
  Return ToolResult → JSON-RPC Response
```

### Architecture Decisions (from Research)

**Decision 1: Layered vs. Microservices**

- **Choice:** Layered Architecture within AgentCore monolith
- **Rationale:**
  - Tool framework is tightly coupled with agent execution
  - Latency critical (<100ms overhead target)
  - Shared database and configuration
- **Trade-offs:** Less independent scalability, but simpler deployment
- **Research Citation:** Research recommends integration over separate service due to latency requirements

**Decision 2: Synchronous vs. Asynchronous Tool Execution**

- **Choice:** Asynchronous with asyncio
- **Rationale:**
  - AgentCore is async-first architecture
  - Tools can take seconds (search, code execution)
  - Need to handle 1000 concurrent executions
- **Implementation:** All tool methods are `async def`, use `await` for I/O
- **Trade-offs:** More complex code, but necessary for performance

**Decision 3: In-Memory vs. Persistent Tool Registry**

- **Choice:** Hybrid - in-memory with database persistence
- **Rationale:**
  - Fast lookups (<10ms) require in-memory
  - Persistence needed for tool metadata across restarts
  - Database used for execution history, not registry
- **Implementation:**
  - ToolRegistry loads from code on startup
  - Database only stores tool_executions (logs)
  - Future: Add database backing for dynamic tool registration
- **Trade-offs:** Tools must be deployed with code, but performance optimal

**Decision 4: Rate Limiting Strategy (Fail Open vs. Fail Closed)**

- **Choice:** Fail Closed (reject requests when Redis unavailable)
- **Rationale:**
  - Cost overruns more damaging than temporary unavailability
  - External APIs will rate limit anyway
  - Fail closed is safer default
- **Implementation:** If Redis error, return 503 Service Unavailable
- **Trade-offs:** Less availability, but prevents runaway costs

**Decision 5: Sandboxing Technology**

- **Choice:** Docker containers with resource limits
- **Rationale:**
  - Industry standard, mature security model
  - Resource limits (CPU, memory, timeout) built-in
  - Easy to deploy in Kubernetes
- **Implementation:** Docker SDK for Python, custom images per language
- **Trade-offs:** Slower startup (~2s), but security is paramount
- **Alternatives:** Pyodide (browser-based, no network), restricted-python (less secure)

### Component Breakdown

**Core Components:**

1. **Tool Interface (base.py)**
   - **Purpose:** Abstract base class defining tool contract
   - **Technology:** Python Protocol/ABC
   - **Pattern:** Abstract base class with template methods
   - **Interfaces:**
     - `async execute(parameters, context) -> ToolResult`
     - `async validate_parameters(parameters) -> tuple[bool, str]`
   - **Dependencies:** Pydantic for ToolMetadata, ToolResult models

2. **Tool Registry (registry.py)**
   - **Purpose:** Centralized catalog for tool discovery
   - **Technology:** In-memory dict with category indexing
   - **Pattern:** Registry pattern with search capabilities
   - **Interfaces:**
     - `register(tool: Tool) -> None`
     - `get(tool_id: str) -> Tool | None`
     - `search(query: str, category: str) -> list[Tool]`
   - **Dependencies:** None (pure Python)

3. **Tool Executor (executor.py)**
   - **Purpose:** Manage tool invocation lifecycle
   - **Technology:** Async Python with error handling
   - **Pattern:** Executor pattern with retry/timeout
   - **Interfaces:**
     - `async execute_tool(tool_id, parameters, context) -> ToolResult`
   - **Dependencies:** ToolRegistry, RateLimiter, database session

4. **Rate Limiter (rate_limiter.py)**
   - **Purpose:** Distributed rate limiting with token bucket
   - **Technology:** Redis with atomic operations
   - **Pattern:** Token bucket algorithm
   - **Interfaces:**
     - `async check_and_consume(tool_id, user_id, tokens=1) -> bool`
   - **Dependencies:** Redis client (aioredis)

5. **Tool Adapters (adapters/)**
   - **Purpose:** Specific tool implementations
   - **Technology:** Varies per tool (HTTP client, Docker SDK, etc.)
   - **Pattern:** Adapter pattern
   - **Interfaces:** Implement Tool base class
   - **Dependencies:** External SDKs (googlesearch, docker, httpx)

### Data Flow & Boundaries

**Request Flow:**

1. **Agent → JSON-RPC → ToolExecutor:**
   - Agent calls `tools.execute` with tool_id and parameters
   - JSON-RPC handler validates request
   - Routes to ToolExecutor

2. **ToolExecutor → Tool Adapter → External Service:**
   - ToolExecutor retrieves tool from registry
   - Checks rate limit (Redis)
   - Validates parameters (Pydantic)
   - Authenticates with credentials
   - Invokes tool adapter
   - Tool adapter calls external service

3. **Response → Agent:**
   - Tool adapter returns ToolResult
   - ToolExecutor logs execution (PostgreSQL)
   - Emits metrics (Prometheus)
   - Returns ToolResult to JSON-RPC
   - Agent receives result

**Component Boundaries:**

**Public Interface:**

- JSON-RPC methods: `tools.list`, `tools.execute`
- Tool base class for extending with custom tools
- ToolResult format for tool outputs

**Internal Implementation:**

- ToolRegistry internals (data structures, caching)
- RateLimiter implementation details (token bucket algorithm)
- Tool adapter implementations (API clients, Docker management)

**Cross-Component Contracts:**

- Tools MUST return ToolResult (standard format)
- Tools MUST validate parameters before execution
- Tools MUST handle auth via ExecutionContext
- Tools MUST respect timeout limits

---

## 📐 Technical Specification

### Data Model

**Entities and Relationships:**

```plaintext
Tool Registration (in-memory):
  ToolRegistry
    ├── tools: dict[str, Tool]
    └── categories: dict[str, list[str]]

Tool Execution (database):
  ToolExecutionLog
    ├── execution_id (PK)
    ├── tool_id (FK → Tool)
    ├── user_id
    ├── agent_id
    ├── parameters (JSONB)
    ├── result (JSONB)
    ├── success (boolean)
    ├── error (text)
    ├── execution_time_ms (integer)
    ├── trace_id (varchar)
    └── created_at (timestamp)

Rate Limiting (Redis):
  rate_limit:{tool_id}:{user_id} → counter
  Expiration: 60 seconds (for per-minute limits)
```

**Validation Rules:**

- `tool_id`: Must be unique, alphanumeric + underscore, max 64 chars
- `parameters`: Must match tool schema (Pydantic validation)
- `timeout_seconds`: Must be > 0 and ≤ 600 (10 minutes max)
- `rate_limit`: Must be > 0, default 100 calls/minute

**Indexing Strategy:**

```sql
-- Tool execution logs
CREATE INDEX idx_tool_executions_tool ON tool_executions(tool_id);
CREATE INDEX idx_tool_executions_user ON tool_executions(user_id);
CREATE INDEX idx_tool_executions_created ON tool_executions(created_at DESC);
CREATE INDEX idx_tool_executions_trace ON tool_executions(trace_id);

-- Composite indexes for analytics
CREATE INDEX idx_tool_executions_tool_user ON tool_executions(tool_id, user_id);
CREATE INDEX idx_tool_executions_success ON tool_executions(success, created_at DESC);
```

**Migration Approach:**

1. **Phase 1 Migration:** Create tool_executions table
2. **Phase 2 Migration:** Add rate_limit_config table (future)
3. **Phase 3 Migration:** Add tool_registry table for persistence (future)

### API Design

**Top 6 Critical Endpoints:**

#### 1. `tools.list` - Tool Discovery

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** List available tools with optional filtering

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools.list",
  "params": {
    "category": "search",     // Optional: filter by category
    "query": "google",        // Optional: search query
    "include_disabled": false // Optional: include disabled tools
  },
  "id": 1
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "tool_id": "google_search",
        "name": "Google Search",
        "description": "Search the web using Google",
        "category": "search",
        "version": "1.0.0",
        "parameters": [
          {
            "name": "query",
            "type": "string",
            "description": "Search query",
            "required": true
          }
        ],
        "authentication": "api_key",
        "rate_limit": {"calls_per_minute": 100},
        "timeout_seconds": 10
      }
    ],
    "total_count": 1
  },
  "id": 1
}
```

**Error Handling:**

- `400 Bad Request`: Invalid category or query parameter
- `500 Internal Server Error`: Registry unavailable

#### 2. `tools.execute` - Tool Invocation

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Execute a tool with validated parameters

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools.execute",
  "params": {
    "tool_id": "google_search",
    "parameters": {
      "query": "AgentCore documentation",
      "num_results": 5
    },
    "context": {
      "user_id": "user-123",
      "agent_id": "agent-456"
    }
  },
  "a2a_context": {
    "trace_id": "trace-789",
    "source_agent": "agent-456"
  },
  "id": 2
}
```

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "tool_id": "google_search",
    "success": true,
    "result": {
      "results": [
        {
          "title": "AgentCore Docs",
          "url": "https://docs.agentcore.dev",
          "snippet": "Complete guide..."
        }
      ]
    },
    "error": null,
    "execution_time_ms": 450,
    "metadata": {
      "query": "AgentCore documentation",
      "num_results": 5
    }
  },
  "id": 2
}
```

**Error Handling:**

- `400 Bad Request`: Invalid parameters, validation failed
- `404 Not Found`: Tool not found
- `429 Too Many Requests`: Rate limit exceeded
- `401 Unauthorized`: Authentication failed
- `408 Request Timeout`: Tool execution timeout
- `500 Internal Server Error`: Tool execution error

#### 3. `tools.search` - Tool Search (Convenience)

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Search tools by capability or description

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools.search",
  "params": {
    "query": "search web",
    "category": "search",
    "limit": 10
  },
  "id": 3
}
```

**Response:** Same as `tools.list`

#### 4. `tools.get_metadata` - Tool Details

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Get detailed metadata for a specific tool

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools.get_metadata",
  "params": {
    "tool_id": "google_search"
  },
  "id": 4
}
```

**Response:** Single tool metadata object

#### 5. `tools.get_execution_history` - Execution Logs

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Retrieve execution history for debugging

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools.get_execution_history",
  "params": {
    "tool_id": "google_search",      // Optional filter
    "user_id": "user-123",           // Optional filter
    "limit": 50,
    "offset": 0
  },
  "id": 5
}
```

**Response:** List of ToolExecutionLog records

#### 6. `tools.get_rate_limit_status` - Quota Check

**Method:** JSON-RPC POST `/api/v1/jsonrpc`

**Purpose:** Check current rate limit status

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools.get_rate_limit_status",
  "params": {
    "tool_id": "google_search",
    "user_id": "user-123"
  },
  "id": 6
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "tool_id": "google_search",
    "user_id": "user-123",
    "limit": 100,
    "remaining": 42,
    "reset_at": "2025-10-15T12:35:00Z"
  },
  "id": 6
}
```

### Security (from Research)

**Based on:** `docs/research/multi-tool-integration.md` Security considerations

**Authentication/Authorization:**

- **Approach:** JWT-based authentication via A2A protocol
- **Implementation:**
  - All tool executions require valid JWT in request
  - User/agent identity extracted from JWT claims
  - Tool access controlled via RBAC policies
- **Standards:** OAuth 2.0 for external API credentials
- **Pattern:** ExecutionContext carries auth information

```python
class ExecutionContext(BaseModel):
    user_id: str
    agent_id: str
    trace_id: str
    jwt_claims: dict[str, Any]
```

**Secrets Management:**

- **Strategy:** Environment variables for development, HashiCorp Vault for production
- **Pattern:**
  - Tool credentials stored in Vault with path: `tools/{tool_id}/credentials`
  - Credentials fetched at tool initialization, cached in memory
  - Automatic rotation via Vault TTL
- **Rotation:** Vault handles automatic rotation, tools reload on cache expiry
- **Implementation:**

```python
class CredentialManager:
    async def get_credentials(self, tool_id: str) -> dict[str, str]:
        # Check cache
        if cached := self._cache.get(tool_id):
            return cached
        # Fetch from Vault
        credentials = await self._vault_client.read(f"tools/{tool_id}/credentials")
        self._cache.set(tool_id, credentials, ttl=3600)
        return credentials
```

**Data Protection:**

- **Encryption in Transit:** All external API calls use TLS 1.3+
- **Encryption at Rest:** Database credentials encrypted via PostgreSQL encryption
- **PII Handling:** Tool execution logs sanitized to remove PII before storage
- **Compliance:** Tool parameters logged without sensitive data (passwords, API keys masked)

**Security Testing:**

- **Approach:**
  - Sandbox penetration testing (Docker escape attempts)
  - Credential leak detection in logs
  - RBAC policy enforcement validation
- **Tools:**
  - SAST: Bandit for Python security issues
  - DAST: OWASP ZAP for API security testing
  - Container scanning: Trivy for Docker image vulnerabilities

**Compliance:**

- **GDPR:** User data (user_id) can be deleted, tool logs support right to erasure
- **SOC 2:** All tool executions auditable with trace IDs, access logs retained
- **ISO 27001:** Secret rotation, encryption, access controls follow standard

**Sandboxing (Code Execution Tools):**

- **Docker Configuration:**
  - No network access by default (unless tool requires)
  - Read-only filesystem except `/tmp`
  - Resource limits: 1 CPU core, 512MB RAM, 30s timeout
  - User namespace mapping (non-root inside container)
- **Security Layers:**
  - AppArmor/SELinux profiles
  - Seccomp filters (syscall restrictions)
  - Drop all capabilities except essential

```python
docker_config = {
    "network_mode": "none",
    "mem_limit": "512m",
    "cpu_quota": 100000,
    "read_only": True,
    "security_opt": ["no-new-privileges", "seccomp=default.json"],
    "cap_drop": ["ALL"]
}
```

### Performance (from Research)

**Based on:** `docs/research/multi-tool-integration.md` Performance section

**Performance Targets (from Research):**

- **Framework Overhead:** <100ms (p95) - Excludes actual tool execution time
  - Rationale: Must be negligible compared to tool execution (typically 500ms-5s)
  - Measurement: `total_time - tool_execution_time`
- **Registry Lookup:** <10ms for 1000+ tools
  - Rationale: Critical path for every tool execution
  - Measurement: Time from `registry.get(tool_id)` call to return
- **Rate Limit Check:** <5ms using Redis
  - Rationale: Sub-millisecond Redis operations critical
  - Measurement: Time for Redis INCR + EXPIRE operations
- **Concurrent Executions:** 1000+ per instance
  - Rationale: Support high-throughput agents
  - Measurement: Load test with 1000 concurrent `tools.execute` calls

**Caching Strategy:**

- **Approach:** Multi-level caching for tool metadata and credentials
- **Pattern:**
  - **L1 Cache (In-Memory):** ToolRegistry loaded at startup, never expires
  - **L2 Cache (Redis):** Rate limit counters, 60s TTL
  - **L3 Cache (Memory):** Credentials from Vault, 1 hour TTL
- **TTL Strategy:**
  - Tool metadata: Never expires (code deployment updates registry)
  - Rate limits: 60s (sliding window)
  - Credentials: 1 hour (Vault rotation period)
- **Invalidation:**
  - Tool metadata: Restart service to reload
  - Rate limits: Automatic expiration via Redis TTL
  - Credentials: Time-based expiration, reload on cache miss

**Database Optimization:**

- **Indexing Strategy:**
  - B-tree indexes on `tool_id`, `user_id`, `created_at` for fast lookups
  - Composite index on `(tool_id, user_id)` for user-specific queries
  - Partial index on `(success = false)` for error analysis
- **Query Patterns:**
  - **INSERT-heavy:** Tool execution logs (write-optimized)
  - **SELECT by tool_id:** Recent executions for debugging
  - **SELECT by user_id:** User quota tracking
  - **Aggregate queries:** Success rate, latency percentiles
- **Connection Pooling:**
  - Min pool size: 5
  - Max pool size: 20
  - Pool timeout: 30s
  - Recycle connections every 3600s
- **Partitioning:**
  - Time-based partitioning on `created_at` (monthly partitions)
  - Archive old partitions (>90 days) to cold storage

```sql
-- Example partition
CREATE TABLE tool_executions_2025_10 PARTITION OF tool_executions
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
```

**Scaling Strategy:**

- **Horizontal Scaling:**
  - Stateless ToolExecutor instances behind load balancer
  - Shared Redis for rate limiting state
  - Shared PostgreSQL for tool execution logs
  - Configuration: Add instances via Kubernetes HPA
- **Vertical Scaling:**
  - Initial: 2 CPU cores, 4GB RAM per instance
  - Scale up for CPU-bound workloads (parameter validation)
  - Scale down for I/O-bound workloads (waiting for external APIs)
- **Auto-scaling:**
  - **Trigger:** CPU >70% for 5 minutes OR request queue >100
  - **Thresholds:**
    - Scale up: Add 1 instance
    - Scale down: Remove 1 instance (min 2)
    - Cool down: 5 minutes between scaling events
- **Performance Monitoring:**
  - **Tools:** Prometheus for metrics, Grafana for dashboards, OpenTelemetry for tracing
  - **Metrics:**
    - `tool_execution_duration_seconds` (histogram)
    - `tool_execution_total` (counter by tool_id, success)
    - `rate_limit_exceeded_total` (counter by tool_id, user_id)
    - `framework_overhead_seconds` (histogram)

---

## 🛠️ Development Setup

### Required Tools and Versions

**Core Dependencies:**

- Python 3.12+ (`python --version`)
- Docker 24+ (`docker --version`)
- PostgreSQL 15+ (via Docker Compose)
- Redis 7+ (via Docker Compose)
- uv package manager (`pip install uv`)

**Development Tools:**

- pytest-asyncio for testing
- Ruff for linting
- mypy for type checking
- testcontainers for integration tests

### Local Environment Setup

**1. Clone and Install Dependencies:**

```bash
# Clone repository
git clone https://github.com/your-org/AgentCore.git
cd AgentCore

# Install dependencies using uv
uv add fastapi pydantic sqlalchemy asyncpg aioredis docker prometheus-client opentelemetry-api opentelemetry-sdk

# Install dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy testcontainers
```

**2. Docker Compose Configuration:**

Update `docker-compose.dev.yml` to include Redis:

```yaml
services:
  postgres:
    # Existing PostgreSQL configuration

  redis:
    image: redis:7-alpine
    container_name: agentcore-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  redis_data:
```

**3. Environment Variables:**

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/agentcore

# Redis
REDIS_URL=redis://localhost:6379/0

# Tool Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id
ENABLE_CODE_EXECUTION=true
CODE_EXECUTION_TIMEOUT=30

# Docker (for code execution)
DOCKER_HOST=unix:///var/run/docker.sock

# Secret Management (development)
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=dev-token

# Monitoring
PROMETHEUS_PORT=9090
```

**4. Database Migration:**

```bash
# Run migration to create tool_executions table
uv run alembic upgrade head
```

**5. Start Services:**

```bash
# Start PostgreSQL and Redis
docker compose -f docker-compose.dev.yml up -d

# Run AgentCore with tool framework
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload
```

**6. Verify Installation:**

```bash
# Test tool listing
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools.list",
    "params": {},
    "id": 1
  }'

# Expected: List of built-in tools (google_search, wikipedia, etc.)
```

### CI/CD Pipeline Requirements

**GitHub Actions Workflow:**

```yaml
name: Tool Framework CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run linter
        run: uv run ruff check src/ tests/

      - name: Run type checker
        run: uv run mypy src/

      - name: Run unit tests
        run: uv run pytest tests/unit/ --cov=agentcore/tools --cov-report=xml

      - name: Run integration tests
        run: uv run pytest tests/integration/tools/
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/agentcore
          REDIS_URL: redis://localhost:6379/0

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Testing Framework and Coverage Targets

**Test Structure:**

```
tests/
├── unit/
│   └── tools/
│       ├── test_base.py          # Tool interface tests
│       ├── test_registry.py      # ToolRegistry tests
│       ├── test_executor.py      # ToolExecutor tests
│       ├── test_rate_limiter.py  # RateLimiter tests
│       └── adapters/
│           ├── test_google_search.py
│           ├── test_python_executor.py
│           └── test_rest_api.py
├── integration/
│   └── tools/
│       ├── test_tool_execution_flow.py
│       ├── test_rate_limiting.py
│       └── test_sandbox_security.py
└── load/
    └── test_tool_concurrency.py
```

**Coverage Targets:**

- **Overall:** 90%+ for tool framework
- **Unit Tests:** 95%+ for core framework (base, registry, executor)
- **Integration Tests:** 85%+ for adapters (external dependencies)
- **Critical Paths:** 100% for security-critical code (auth, sandboxing)

**Test Commands:**

```bash
# Unit tests only
uv run pytest tests/unit/ --cov=agentcore/tools --cov-report=term

# Integration tests (requires Docker)
uv run pytest tests/integration/tools/

# Load tests
uv run locust -f tests/load/test_tool_concurrency.py --host=http://localhost:8001

# All tests with coverage
uv run pytest tests/ --cov=agentcore/tools --cov-report=html
```

---

## ⚠️ Risk Management

| Risk | Impact | Likelihood | Mitigation | Contingency |
|------|--------|------------|------------|-------------|
| **Docker sandbox escape** | Critical | Low | Use AppArmor profiles, drop capabilities, read-only filesystem, regular security audits | Kill all containers, disable code execution tools, incident response |
| **Redis unavailability breaks rate limiting** | High | Medium | Redis clustering with failover, health checks, fail-closed strategy | Manual rate limiting via config, temporary disable high-volume tools |
| **External API rate limits exceeded** | Medium | High | Implement client-side rate limiting, queue requests, caching, multiple API keys | Fallback to alternative APIs, return cached results, notify users |
| **Parameter injection attacks** | High | Medium | Strict Pydantic validation, input sanitization, SQL parameterization | Disable affected tools, review all tool parameters |
| **Tool execution timeout causes resource leak** | Medium | Medium | Timeout enforcement, Docker container auto-cleanup, resource monitoring | Restart service, kill orphaned containers, investigate timeouts |
| **Credential leakage in logs** | Critical | Low | Log sanitization, mask sensitive fields, secret scanning in CI | Rotate all credentials, audit logs, implement secret detection |
| **Database connection pool exhaustion** | High | Medium | Connection pooling with limits, connection recycling, monitoring | Increase pool size, restart service, investigate slow queries |
| **Performance regression in production** | Medium | Medium | Load testing before deployment, canary releases, performance monitoring | Rollback deployment, investigate bottlenecks, optimize hot paths |

**Risk Monitoring:**

- **Security:** Weekly security scans, monthly penetration tests
- **Performance:** Continuous monitoring, alerting on degradation
- **Availability:** 99.9% SLA, incident response within 15 minutes

---

## 📅 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Establish core framework infrastructure

**Tasks:**

1. **Tool Interface and Base Classes** (3 SP)
   - Create `Tool` abstract base class
   - Implement `validate_parameters()` method
   - Define `ExecutionContext` model
   - Unit tests for interface

2. **Data Models** (3 SP)
   - `ToolMetadata` model with Pydantic
   - `ToolParameter` model with validation
   - `ToolResult` model with success/error
   - Unit tests for models

3. **Tool Registry** (5 SP)
   - `ToolRegistry` class with in-memory storage
   - `register()`, `get()`, `search()` methods
   - Category indexing for fast lookups
   - Unit tests for registry operations

4. **Database Schema** (3 SP)
   - Alembic migration for `tool_executions` table
   - Indexes on tool_id, user_id, created_at
   - Test migration up/down

5. **Tool Executor** (5 SP)
   - `ToolExecutor` class for lifecycle management
   - Basic `execute_tool()` method
   - Error handling and logging
   - Unit tests with mocked tools

6. **Parameter Validation Framework** (3 SP)
   - Pydantic schema validation
   - Type checking and required field validation
   - Error message formatting
   - Unit tests for edge cases

7. **Initial Documentation** (2 SP)
   - API documentation for tool interfaces
   - Architecture diagram
   - Developer guide for adding tools

**Deliverables:**

- `src/agentcore/tools/base.py` - Tool interface
- `src/agentcore/tools/models.py` - Data models
- `src/agentcore/tools/registry.py` - ToolRegistry
- `src/agentcore/tools/executor.py` - ToolExecutor
- `alembic/versions/XXX_add_tool_executions.py` - Migration
- `tests/unit/tools/` - Unit tests
- `docs/tools/README.md` - Documentation

**Total:** 24 story points

### Phase 2: Built-in Tools (Week 3)

**Goal:** Implement essential tool adapters

**Tasks:**

1. **Google Search Tool** (5 SP)
   - `GoogleSearchTool` adapter
   - Google Custom Search API integration
   - Result parsing and formatting
   - Unit tests with mocked API

2. **Wikipedia Search Tool** (3 SP)
   - `WikipediaSearchTool` adapter
   - Wikipedia API integration
   - Article summary extraction
   - Unit tests

3. **Python Execution Tool** (8 SP)
   - `PythonExecutionTool` adapter
   - Docker sandbox configuration
   - Security: no network, resource limits
   - Result capture (stdout, stderr, return value)
   - Unit tests + integration tests

4. **REST API Tool** (5 SP)
   - `RESTAPITool` adapter
   - HTTP client with httpx
   - Support GET, POST, PUT, DELETE
   - Auth header support
   - Unit tests with mocked responses

5. **File Operations Tool** (3 SP)
   - `FileOperationsTool` adapter
   - Read, write, list operations
   - Security: path validation, size limits
   - Unit tests

6. **Tool Registration on Startup** (2 SP)
   - `register_builtin_tools()` function
   - Auto-register all tools on app startup
   - Configuration via environment variables

7. **Integration Tests** (5 SP)
   - End-to-end tests for each tool
   - Test with real external services (staging)
   - Validate error handling

**Deliverables:**

- `src/agentcore/tools/adapters/search.py` - Search tools
- `src/agentcore/tools/adapters/code.py` - Code execution
- `src/agentcore/tools/adapters/api.py` - API client
- `src/agentcore/tools/adapters/files.py` - File operations
- `src/agentcore/tools/builtin.py` - Registration
- `docker/python-sandbox/` - Docker configuration
- `tests/integration/tools/` - Integration tests

**Total:** 31 story points

### Phase 3: JSON-RPC Integration (Week 4)

**Goal:** Expose tools via A2A protocol

**Tasks:**

1. **tools.list JSON-RPC Method** (3 SP)
   - Register method with `@register_jsonrpc_method`
   - Implement category filtering
   - Return tool metadata
   - Unit tests

2. **tools.execute JSON-RPC Method** (5 SP)
   - Register method
   - Parameter validation
   - Call ToolExecutor
   - Return ToolResult
   - Error handling (404, 400, 429, etc.)
   - Unit tests

3. **tools.search JSON-RPC Method** (2 SP)
   - Convenience wrapper around tools.list
   - Query-based search
   - Unit tests

4. **A2A Authentication Integration** (3 SP)
   - Extract user_id/agent_id from JWT
   - Pass to ExecutionContext
   - RBAC policy enforcement
   - Unit tests

5. **Distributed Tracing Support** (5 SP)
   - OpenTelemetry instrumentation
   - Trace ID propagation via A2A context
   - Span creation for tool executions
   - Link tool calls to parent trace
   - Integration tests

6. **Error Categorization** (3 SP)
   - Define error types (auth, validation, timeout, execution)
   - Map tool errors to JSON-RPC error codes
   - Structured error responses
   - Unit tests

7. **API Documentation** (3 SP)
   - OpenAPI spec generation
   - JSON-RPC method examples
   - Tool developer guide

**Deliverables:**

- `src/agentcore/tools/jsonrpc.py` - JSON-RPC handlers
- `src/agentcore/tools/tracing.py` - OpenTelemetry integration
- `docs/tools/api.md` - API documentation
- `tests/unit/tools/test_jsonrpc.py` - Unit tests
- `tests/integration/tools/test_jsonrpc_flow.py` - Integration tests

**Total:** 24 story points

### Phase 4: Advanced Features (Weeks 5-6)

**Goal:** Production hardening and optimization

**Tasks:**

1. **Rate Limiting with Redis** (8 SP)
   - `RateLimiter` class with token bucket algorithm
   - Redis INCR + EXPIRE operations
   - Per-tool, per-user limits
   - Handle Redis unavailability (fail closed)
   - Unit tests + integration tests

2. **Automatic Retry with Exponential Backoff** (5 SP)
   - Detect retryable errors (network, 503, timeout)
   - Implement exponential backoff (1s, 2s, 4s)
   - Max retry attempts configurable
   - Non-retryable errors fail immediately
   - Unit tests

3. **Quota Management** (3 SP)
   - Daily/monthly quota tracking
   - `tools.get_rate_limit_status` endpoint
   - Quota exceeded handling
   - Unit tests

4. **Prometheus Metrics** (5 SP)
   - `tool_execution_duration_seconds` histogram
   - `tool_execution_total` counter
   - `rate_limit_exceeded_total` counter
   - Per-tool metrics
   - Integration with existing Prometheus

5. **Grafana Dashboards** (3 SP)
   - Tool usage dashboard
   - Performance dashboard (latency, success rate)
   - Cost dashboard (API calls per tool)
   - Alert configuration

6. **Load Testing** (5 SP)
   - Locust test with 1000 concurrent users
   - Mix of tool types (search, code, API)
   - Sustained load for 1 hour
   - Performance report

7. **Security Audit** (8 SP)
   - Docker sandbox penetration testing
   - Credential leak detection in logs
   - RBAC policy validation
   - Parameter injection testing
   - Security report with findings

8. **Performance Optimization** (5 SP)
   - Profile framework overhead
   - Optimize hot paths (registry lookup, validation)
   - Connection pool tuning
   - Validate <100ms overhead target

9. **Production Runbook** (3 SP)
   - Deployment guide
   - Monitoring and alerting setup
   - Incident response procedures
   - Troubleshooting guide

**Deliverables:**

- `src/agentcore/tools/rate_limiter.py` - Rate limiting
- `src/agentcore/tools/retry.py` - Retry logic
- `src/agentcore/tools/metrics.py` - Prometheus metrics
- `k8s/monitoring/tool-dashboards.yaml` - Grafana dashboards
- `tests/load/test_tool_concurrency.py` - Load tests
- `docs/tools/security-audit.md` - Security report
- `docs/tools/runbook.md` - Operations guide

**Total:** 45 story points

---

## 🎯 Quality Assurance

### Testing Strategy

**Unit Testing:**

- **Target:** 95%+ coverage for core framework
- **Approach:**
  - Test each component in isolation with mocked dependencies
  - Validate parameter validation, error cases, edge conditions
  - Use pytest fixtures for common test data
- **Tools:** pytest-asyncio, pytest-cov, pytest-mock

**Integration Testing:**

- **Target:** 85%+ coverage for tool adapters
- **Approach:**
  - Test full pipeline with real tool integrations (staging APIs)
  - Validate rate limiting with concurrent requests
  - Test retry logic with simulated failures
  - Use testcontainers for PostgreSQL and Redis
- **Tools:** pytest-asyncio, testcontainers, docker-py

**Security Testing:**

- **Approach:**
  - Sandbox penetration testing (Docker escape attempts)
  - Credential leak detection in logs
  - RBAC policy enforcement validation
  - Parameter injection attacks
- **Tools:** Bandit (SAST), OWASP ZAP (DAST), Trivy (container scanning)

**Performance Testing:**

- **Approach:**
  - Benchmark framework overhead (<100ms target)
  - Load test with 1000 concurrent tool executions
  - Measure rate limiting overhead (<5ms target)
  - Profile hot paths
- **Tools:** Locust, pytest-benchmark, cProfile

**Load Testing Scenarios:**

1. **Scenario 1: Sustained Load**
   - 1000 concurrent users
   - Mix of tools: 50% search, 30% API, 20% code execution
   - Duration: 1 hour
   - Success criteria: 95%+ success rate, p95 latency <500ms

2. **Scenario 2: Rate Limit Stress**
   - Single tool, single user
   - Requests at 2x rate limit
   - Verify 429 errors returned
   - Success criteria: Rate limit enforced correctly

3. **Scenario 3: Mixed Workload**
   - Multiple tools, multiple users
   - Realistic distribution of tool types
   - Duration: 30 minutes
   - Success criteria: No resource exhaustion, stable latency

### Code Quality Gates

**Pre-Commit Checks:**

```bash
# Linting with Ruff
uv run ruff check src/ tests/

# Type checking with mypy (strict mode)
uv run mypy src/ --strict

# Unit tests
uv run pytest tests/unit/ --cov=agentcore/tools --cov-fail-under=90
```

**PR Requirements:**

- [ ] All tests passing (unit + integration)
- [ ] Code coverage ≥90%
- [ ] No linting errors (Ruff)
- [ ] No type errors (mypy --strict)
- [ ] Security scan passed (Bandit)
- [ ] Code reviewed by 2+ engineers

**Definition of Done:**

- [ ] Code implemented and reviewed
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Performance benchmarked (meets targets)

### Deployment Verification Checklist

**Pre-Deployment:**

- [ ] All tests passing on CI
- [ ] Database migration tested on staging
- [ ] Redis configured and accessible
- [ ] Docker daemon accessible for code execution
- [ ] Credentials loaded in Vault
- [ ] Monitoring dashboards configured

**Deployment Steps:**

1. [ ] Apply database migration: `uv run alembic upgrade head`
2. [ ] Deploy new service version (blue-green deployment)
3. [ ] Verify health check: `/health` returns 200
4. [ ] Smoke test: Call `tools.list` and verify response
5. [ ] Execute sample tool: `tools.execute` with google_search
6. [ ] Monitor metrics for 15 minutes (error rate, latency)
7. [ ] Switch traffic to new version
8. [ ] Monitor for 1 hour before declaring success

**Post-Deployment:**

- [ ] Verify rate limiting working (check Redis)
- [ ] Verify tracing (check OpenTelemetry backend)
- [ ] Verify logging (check tool_executions table)
- [ ] Verify alerting (trigger test alert)
- [ ] Update runbook with deployment notes

### Monitoring and Alerting Setup

**Key Metrics:**

- `tool_execution_duration_seconds{tool_id, success}` - Histogram
- `tool_execution_total{tool_id, success}` - Counter
- `rate_limit_exceeded_total{tool_id, user_id}` - Counter
- `framework_overhead_seconds` - Histogram

**Alerts:**

- **Critical:**
  - Tool execution success rate <90% for 5 minutes
  - Framework overhead >200ms (p95) for 5 minutes
  - Redis unavailable
- **Warning:**
  - Tool execution success rate <95% for 10 minutes
  - Rate limit exceeded >100 times/minute
  - Docker container failures >5/minute

**Dashboards:**

1. **Tool Usage Dashboard:**
   - Executions per tool (bar chart)
   - Success rate per tool (gauge)
   - Top users by execution count (table)

2. **Performance Dashboard:**
   - Framework overhead (histogram)
   - Tool execution latency (heatmap)
   - Throughput (requests/second)

3. **Cost Dashboard:**
   - API calls per tool (line chart)
   - Rate limit status (gauge)
   - Quota usage per user (table)

---

## ⚠️ Error Handling & Edge Cases

**From:** Research findings and specification requirements

### Error Scenarios

**Critical Error Paths:**

1. **Tool Execution Timeout**
   - **Cause:** Tool runs longer than configured timeout (default: 30s)
   - **Impact:** Blocks agent workflow, consumes resources
   - **Handling:**
     - Enforce timeout at executor level using asyncio.wait_for()
     - Kill Docker container if code execution
     - Return 408 Request Timeout error
   - **Recovery:**
     - Agent can retry with longer timeout if available
     - Log timeout for investigation
   - **User Experience:** "Tool execution timed out after 30 seconds. Consider simplifying the request or increasing timeout."

2. **Authentication Failure**
   - **Cause:** Invalid credentials, expired API key, OAuth token expired
   - **Impact:** Tool cannot access external service
   - **Handling:**
     - Return 401 Unauthorized immediately (non-retryable)
     - Log auth failure with tool_id
     - Do NOT log credentials
   - **Recovery:**
     - Operator must rotate credentials
     - Alert on repeated auth failures
   - **User Experience:** "Authentication failed. Please contact administrator to update credentials."

3. **Rate Limit Exceeded (External API)**
   - **Cause:** Too many requests to external service (Google, Wikipedia, etc.)
   - **Impact:** External API returns 429 error
   - **Handling:**
     - Detect 429 response from external API
     - Extract retry-after header if present
     - Return 429 to agent with retry-after
   - **Recovery:**
     - Automatic retry after backoff period
     - If persistent, reduce rate limit in config
   - **User Experience:** "Rate limit exceeded. Please wait 60 seconds before retrying."

4. **Rate Limit Exceeded (Internal)**
   - **Cause:** User/agent exceeds configured tool rate limit
   - **Impact:** Cost control mechanism triggered
   - **Handling:**
     - Redis counter exceeds limit
     - Return 429 immediately (before tool execution)
     - Log rate limit violation
   - **Recovery:**
     - Wait for rate limit window to reset (60s)
     - Request quota increase if legitimate use case
   - **User Experience:** "You have exceeded the rate limit for this tool (100 calls/minute). Please wait before retrying."

5. **Parameter Validation Failure**
   - **Cause:** Missing required parameter, wrong type, invalid value
   - **Impact:** Tool cannot execute with invalid parameters
   - **Handling:**
     - Validate parameters before execution (fast fail)
     - Return 400 Bad Request with specific error
     - Include parameter name and expected type
   - **Recovery:**
     - Agent must fix parameters and retry
     - No automatic retry
   - **User Experience:** "Invalid parameter 'num_results': expected integer, got string."

6. **Docker Container Crash**
   - **Cause:** Code execution kills container, resource exhaustion, sandbox escape attempt
   - **Impact:** Code execution fails, potential security incident
   - **Handling:**
     - Detect container exit code ≠ 0
     - If security violation detected, log incident
     - Return execution error to agent
   - **Recovery:**
     - Investigate container logs
     - If security issue, block user/agent
     - Kill orphaned containers
   - **User Experience:** "Code execution failed due to container error. Please review your code for issues."

7. **Redis Unavailability**
   - **Cause:** Redis server down, network partition
   - **Impact:** Rate limiting cannot be enforced
   - **Handling:**
     - Fail closed: Return 503 Service Unavailable
     - Log Redis error
     - Alert on Redis failures
   - **Recovery:**
     - Automatic failover if Redis cluster configured
     - Manual intervention if single Redis instance
   - **User Experience:** "Tool execution temporarily unavailable due to infrastructure issue. Please try again shortly."

8. **Malformed External API Response**
   - **Cause:** External service returns unexpected format, parsing error
   - **Impact:** Cannot extract tool result
   - **Handling:**
     - Catch parsing exceptions
     - Log malformed response (with trace_id)
     - Return execution error
   - **Recovery:**
     - Investigate external API changes
     - Update parser if format changed
     - Retry may succeed if transient
   - **User Experience:** "Tool returned unexpected response format. This has been logged for investigation."

### Edge Cases

**Identified in Specification and Research:**

| Edge Case | Detection | Handling | Testing Approach |
|-----------|-----------|----------|------------------|
| **Empty search results** | Check result array length | Return success with empty array, not error | Unit test with mocked empty response |
| **Very large tool response** | Check response size >10MB | Truncate response, log warning | Integration test with large data |
| **Concurrent executions of same tool by same user** | Multiple requests in flight | Each request independent, rate limit shared | Load test with concurrent requests |
| **Tool timeout during result streaming** | Partial results received | Return partial results with timeout flag | Integration test with slow tool |
| **Parameter with special characters** | Validation detects SQL/shell injection | Sanitize or reject parameter | Security test with injection payloads |
| **Tool execution during deployment** | Service restart mid-execution | Graceful shutdown waits for in-flight requests | Manual test during deployment |
| **Rate limit reset during execution** | Counter incremented before execution | Atomic Redis operations prevent race | Concurrency test around reset boundary |
| **Credential rotation during execution** | Cached credential expires mid-execution | Retry with fresh credential | Integration test with expiring token |

### Input Validation

**Validation Rules:**

- `tool_id`: Alphanumeric + underscore, max 64 chars, cannot be empty
- `query` (search tools): Non-empty string, max 500 chars, sanitize HTML
- `code` (execution tools): Non-empty string, max 10KB, no null bytes
- `num_results`: Integer, 1-100 range
- `timeout`: Integer, 1-600 seconds
- `url` (API tools): Valid HTTP(S) URL, whitelist domains for security

**Sanitization:**

- **XSS Prevention:** HTML-escape all user inputs before logging or displaying
- **SQL Injection Prevention:** Use parameterized queries, never string concatenation
- **Shell Injection Prevention:** Never pass user input directly to shell commands
- **Input Normalization:** Trim whitespace, lowercase where appropriate

```python
def sanitize_search_query(query: str) -> str:
    """Sanitize search query to prevent injection."""
    # Remove null bytes
    query = query.replace('\x00', '')
    # Limit length
    query = query[:500]
    # HTML escape
    query = html.escape(query)
    return query.strip()
```

### Graceful Degradation

**Fallback Strategies:**

| Failure Scenario | Degraded Mode | Fallback Approach |
|------------------|---------------|-------------------|
| **Redis unavailable** | No rate limiting | Fail closed (reject requests) OR fail open (allow all) based on config |
| **External API unavailable** | Tool unavailable | Return cached results if available, otherwise error |
| **Docker daemon unavailable** | Code execution unavailable | Disable code execution tools, use alternative if available |
| **High latency from external API** | Slower response | Increase timeout temporarily, warn user |
| **Database write failure** | Logging disabled | Log to file instead, continue tool execution |

**Circuit Breaker Pattern:**

```python
class CircuitBreaker:
    """Circuit breaker for failing tools."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.last_failure_time = time.time()

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def allow_request(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        if self.state == "half-open":
            return True
```

### Monitoring & Alerting

**Error Tracking:**

- **Tool:** Prometheus + Grafana + Alertmanager
- **Metrics:**
  - `tool_errors_total{tool_id, error_type}` - Counter
  - `tool_timeout_total{tool_id}` - Counter
  - `rate_limit_exceeded_total{tool_id, user_id}` - Counter
  - `auth_failures_total{tool_id}` - Counter
- **Thresholds:**
  - Alert if error rate >10% for 5 minutes
  - Alert if timeout rate >20% for 5 minutes
  - Alert if auth failures >5 for single tool

**Incident Response Plan:**

1. **Alert Received:** Incident responder notified via PagerDuty
2. **Triage:** Check monitoring dashboard for affected tools
3. **Investigation:** Review logs with trace_id, check external service status
4. **Mitigation:**
   - If external API down: Enable circuit breaker for tool
   - If auth failure: Rotate credentials
   - If rate limit issue: Increase limits temporarily
5. **Resolution:** Deploy fix, verify metrics return to normal
6. **Post-Mortem:** Document incident, identify root cause, implement prevention

---

## 📚 References & Traceability

### Source Documentation

**Research Foundation:**

- `docs/research/multi-tool-integration.md`
  - Technical architecture and framework design (lines 14-46)
  - Tool interface pattern (lines 117-182)
  - Tool registry implementation (lines 184-250)
  - Code examples: GoogleSearchTool (lines 254-344), PythonExecutionTool (lines 346-435)
  - Success metrics and value analysis (lines 437-468)
  - Integration strategy with phases (lines 532-703)

**Specification:**

- `docs/specs/tool-integration/spec.md`
  - Functional requirements FR-1 through FR-5 (lines 43-104)
  - Non-functional requirements (lines 107-158)
  - 5 key features with flows (lines 161-269)
  - API specification (lines 425-526)
  - Data models (lines 529-583)
  - Success validation approach (lines 586-613)

### System Context

**Architecture & Patterns:**

- Project Standards: `CLAUDE.md`
  - Python 3.12+ with async/await
  - FastAPI framework with JSON-RPC 2.0
  - PostgreSQL with async SQLAlchemy
  - 90%+ test coverage requirement
  - Async-first architecture patterns

**Related Components:**

- MOD-001: `docs/specs/modular-agent-core/spec.md` - Executor module integration
  - Relationship: Executor module depends on tool framework for tool invocation
- CTX-001: Context Management System (future integration)
  - Relationship: Tool context playbooks for tool selection strategies
- MEM-001: Memory Management System (future integration)
  - Relationship: Tool results may be stored in agent memory

### Technology Evaluation

**Framework Selection:**

- **FastAPI:** Existing AgentCore stack, JSON-RPC support, high performance (source: AgentCore architecture)
- **Pydantic v2:** 10x validation performance improvement (source: Pydantic v2 release notes)
- **Redis:** Sub-millisecond atomic operations (source: Redis documentation)
- **Docker:** Industry standard for sandboxing (source: Research analysis)

**Performance Benchmarks:**

- **Framework overhead target:** <100ms (source: Research performance analysis, Spec NFR-1.1)
- **Registry lookup target:** <10ms (source: Spec NFR-1.2)
- **Rate limit check target:** <5ms (source: Spec NFR-1.4)
- **Concurrent executions:** 1000+ per instance (source: Spec NFR-1.3)

### Security Standards

- **OWASP Top 10:** Addressed injection, auth, sensitive data exposure (source: OWASP 2021)
- **Docker Security:** AppArmor/SELinux, seccomp, capability dropping (source: Docker security best practices)
- **Secret Management:** HashiCorp Vault integration (source: Vault documentation)

### Related Components

**Dependencies:**

- MOD-001 (Modular Agent Core): `docs/specs/modular-agent-core/spec.md`
  - **Relationship:** Executor module consumes tool framework
  - **Integration:** Executor calls `tools.execute` to invoke tools
  - **Status:** In progress (MOD-001 spec complete, plan generated)

**Dependents:**

- CTX-001 (Context Management System): Future integration
  - **Relationship:** Tool selection strategies based on context playbooks
  - **Integration:** Context determines which tools agent can use
- MEM-001 (Memory Management System): Future integration
  - **Relationship:** Tool results stored in agent memory
  - **Integration:** Memory provides tool execution history to agents

---

## 📝 Appendix

### Glossary

- **Tool:** External service or capability accessible via standardized interface
- **Tool Adapter:** Specific implementation wrapping external service (e.g., GoogleSearchTool)
- **Tool Registry:** Centralized catalog of available tools
- **Tool Executor:** Manages tool invocation lifecycle (auth, retry, logging)
- **Rate Limiting:** Restricting number of tool calls per time window
- **Sandboxing:** Isolated execution environment for code (Docker containers)
- **A2A Protocol:** Agent-to-Agent communication protocol (JSON-RPC 2.0)
- **Trace ID:** Unique identifier for distributed tracing across system

### Abbreviations

- **A2A:** Agent-to-Agent
- **API:** Application Programming Interface
- **RBAC:** Role-Based Access Control
- **JWT:** JSON Web Token
- **TTL:** Time To Live
- **SLO:** Service Level Objective
- **KPI:** Key Performance Indicator
- **SAST:** Static Application Security Testing
- **DAST:** Dynamic Application Security Testing
- **OWASP:** Open Web Application Security Project

### Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-15 | Architecture Team | Initial PRP-format plan from /sage.plan command |

---

**End of Plan**

This implementation plan provides a comprehensive blueprint for building the Tool Integration Framework following the Product Requirements Prompt (PRP) format. All context sources have been integrated, traceability established, and implementation details specified for successful execution.
