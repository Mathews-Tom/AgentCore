# Reasoning Strategy Framework Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-15 (Revised for strategy framework approach)
**Specification:** `docs/specs/context-reasoning/spec.md` (v2.0)
**Research:** `docs/research/context-reasoning.md`
**Component ID:** RSF (was BCR)
**Epic Ticket:** RSF-001

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → This Plan**

1. **Research Analysis:** docs/research/bounded-context-reasoning.md
   - Bounded context reasoning paradigm with fixed-size context windows
   - Linear O(N) computational scaling vs quadratic O(N²)
   - Carryover compression mechanism for maintaining reasoning continuity
   - 50-98% compute reduction benchmarks
   - Training considerations for carryover generation

2. **Formal Specification:** docs/specs/bounded-context-reasoning/spec.md (v2.0)
   - FR-1: ReasoningStrategy protocol for pluggable strategy architecture
   - FR-2: Strategy configuration system (system/agent/request levels)
   - FR-3: Bounded context strategy as optional implementation
   - FR-4-5: Bounded context iteration and carryover management
   - FR-6: Unified JSON-RPC API (`reasoning.execute`)
   - FR-7: Agent strategy capability advertisement
   - NFR: Strategy extensibility, optional deployment, performance per strategy
   - Success metrics: Strategy flexibility, configuration coverage, compute efficiency

### Related Documentation

**System Context:**

- Architecture: CLAUDE.md - AgentCore A2A protocol implementation
- Tech Stack: Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy (async)
- Patterns: JSON-RPC method registration, async database access, A2A context propagation

**Other Specifications:**

- docs/specs/a2a-protocol/spec.md - JSON-RPC infrastructure dependency
- docs/specs/agent-runtime/spec.md - Agent lifecycle integration point
- docs/specs/integration-layer/spec.md - LLM provider integration dependency

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Provide a pluggable reasoning strategy framework for AgentCore that supports multiple reasoning approaches (Chain of Thought, Bounded Context, ReAct, Tree of Thought). As a generic orchestration framework, AgentCore must offer flexible, optional reasoning capabilities that users can configure based on their specific needs.

**Value Proposition:**

- **Flexibility:** Users choose reasoning strategies appropriate for their use cases
- **Optional Deployment:** No reasoning strategy is mandatory; deploy with zero strategies if desired
- **Cost Optimization:** Bounded context strategy offers 33-98% token reduction for long-form reasoning
- **Extensibility:** Easy to add new strategies without core framework changes
- **User Control:** Configuration at system, agent, and request levels

**Target Users:**

- **AI Application Developers:** Need flexibility to choose reasoning approaches per task type
- **Research Teams:** Require multiple reasoning strategies for different problem domains
- **Enterprise Users:** Want predictable costs with configurable optimization strategies
- **Framework Integrators:** Building on AgentCore need extensible reasoning capabilities

### Technical Approach

**Architecture Pattern:** Pluggable Strategy Framework with Service Layer Components

- ReasoningStrategy protocol for pluggable implementations
- Strategy registry for registration and discovery
- Configuration-driven strategy selection (system/agent/request levels)
- Stateless strategy implementations enable horizontal scaling
- Unified JSON-RPC API (`reasoning.execute`) for all strategies
- Async-first implementation using asyncio

**Technology Stack:**

- **Runtime:** Python 3.12+
- **Framework:** FastAPI (existing AgentCore stack)
- **Strategy Pattern:** Protocol-based (typing.Protocol) for extensibility
- **LLM Integration:** Async LLM client with token counting and stop sequences
- **API Protocol:** JSON-RPC 2.0 (A2A protocol compliance)
- **Validation:** Pydantic v2 for request/response schemas
- **Configuration:** TOML-based with multi-level precedence
- **Monitoring:** Prometheus metrics, distributed tracing via A2A context
- **Testing:** pytest-asyncio, 90%+ coverage requirement

**Implementation Strategy:**

1. **Phase 1 (Week 1-2):** Strategy framework core (protocol, registry, configuration system)
2. **Phase 2 (Week 3):** Unified JSON-RPC API (`reasoning.execute`) with strategy routing
3. **Phase 3 (Week 4):** Bounded context strategy implementation
4. **Phase 4 (Week 5):** Agent capability advertisement and strategy discovery
5. **Phase 5 (Week 6):** Additional strategies (CoT, ReAct) - optional
6. **Phase 6 (Week 7):** Monitoring, optimization, and production readiness

### Key Success Metrics

**Service Level Objectives (SLOs):**

- **Availability:** 99.9% uptime for reasoning endpoint
- **Strategy Selection Overhead:** <50ms per request
- **Response Time:** Strategy-specific (documented per strategy)
- **Throughput:** Support concurrent reasoning requests across multiple strategies
- **Error Rate:** <0.1% for valid reasoning requests

**Key Performance Indicators (KPIs):**

- **Strategy Flexibility:** Support 3+ reasoning strategies (CoT, Bounded Context, ReAct)
- **Configuration Coverage:** Strategy selection at system, agent, and request levels
- **Optional Deployment:** System can deploy with zero strategies enabled
- **Compute Efficiency (Bounded Context):** 50-90% reduction for >10K token reasoning
- **Memory Efficiency (Bounded Context):** Constant O(1) memory usage
- **Extensibility:** New strategies addable without core framework changes

---

## 💻 Code Examples & Patterns

### Repository Patterns (from AgentCore CLAUDE.md)

**Relevant Existing Patterns:**

1. **JSON-RPC Method Registration Pattern:**
   - **Application:** Register `reasoning.bounded_context` method with existing infrastructure
   - **Usage Example:**

     ```python
     from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

     @register_jsonrpc_method("reasoning.bounded_context")
     async def handle_bounded_reasoning(request: JsonRpcRequest) -> dict[str, Any]:
         # Implementation
         return result
     ```

   - **Adaptation Notes:** Create `services/reasoning_jsonrpc.py` following existing `*_jsonrpc.py` pattern

2. **Pydantic Model Pattern for Validation:**
   - **Application:** Define request/response schemas for bounded reasoning API
   - **Usage Example:**

     ```python
     from pydantic import BaseModel, Field

     class BoundedReasoningParams(BaseModel):
         query: str = Field(..., min_length=1, max_length=50000)
         chunk_size: int = Field(default=8192, ge=1024, le=32768)
         carryover_size: int = Field(default=4096, ge=512, le=16384)
         max_iterations: int = Field(default=5, ge=1, le=50)
     ```

   - **Adaptation Notes:** Follow strict typing with modern Python 3.12+ syntax (`list[T]`, `dict[K, V]`)

3. **Async Service Pattern:**
   - **Application:** Implement BoundedContextEngine as async service
   - **Usage Example:**

     ```python
     class BoundedContextEngine:
         async def reason(self, query: str) -> ReasoningResult:
             # Async implementation
             pass
     ```

   - **Adaptation Notes:** All I/O operations use asyncio (LLM calls, potential DB access)

### Implementation Reference Examples

**From Research (bounded-context-reasoning.md):**

```python
async def bounded_context_reasoning(
    query: str,
    chunk_size: int = 8192,
    carryover_size: int = 4096,
    max_iterations: int = 5
) -> str:
    """
    Perform bounded context reasoning with fixed memory.

    Args:
        query: The problem to solve
        chunk_size: Maximum tokens per iteration (C)
        carryover_size: Tokens to carry forward (m)
        max_iterations: Maximum reasoning iterations (I)

    Returns:
        Final answer after reasoning
    """
    prompt = format_prompt(query)
    carryover = ""

    for iteration in range(max_iterations):
        # Build current context
        if iteration == 0:
            context = prompt
            max_new_tokens = chunk_size
        else:
            context = prompt + "\n\nPrevious progress:\n" + carryover
            max_new_tokens = chunk_size - len(tokenize(prompt + carryover))

        # Generate reasoning chunk
        chunk = await generate(
            context=context,
            max_tokens=max_new_tokens,
            stop_on=["<answer>", "<continue>"]
        )

        # Check if answer found
        if "<answer>" in chunk:
            return extract_answer(chunk)

        # Generate carryover for next iteration
        if iteration < max_iterations - 1:
            carryover = await generate_carryover(
                context=context + chunk,
                max_tokens=carryover_size,
                instruction="Summarize the key progress and insights"
            )

    # Fallback if no answer after all iterations
    return generate_final_answer(prompt, carryover)
```

**Key Takeaways from Examples:**

- Fixed context window maintained throughout reasoning process
- Carryover generation is critical for maintaining continuity
- Stop sequences (`<answer>`, `<continue>`) enable answer detection
- Token counting required for accurate chunk size management
- Anti-patterns to avoid: Growing context, unbounded iteration, missing error handling

### New Patterns to Create

**Patterns This Implementation Will Establish:**

1. **Bounded Iteration Pattern**
   - **Purpose:** Execute multi-iteration reasoning with fixed memory footprint
   - **Location:** `agentcore.reasoning.bounded_context.BoundedContextEngine`
   - **Reusability:** Template for other bounded computational patterns (bounded search, bounded planning)

2. **Carryover Compression Pattern**
   - **Purpose:** Generate compressed state summaries for iteration boundaries
   - **Location:** `agentcore.reasoning.bounded_context.CarryoverGenerator`
   - **Reusability:** Applicable to any multi-step process requiring state compression

3. **Compute Savings Metrics Pattern**
   - **Purpose:** Compare actual vs theoretical computational costs
   - **Location:** `agentcore.reasoning.bounded_context.MetricsCalculator`
   - **Reusability:** Template for measuring efficiency gains of optimization techniques

---

## 🔧 Technology Stack

### Recommended Stack (from Research & System Context)

**Based on:** Existing AgentCore stack (CLAUDE.md) + Research recommendations

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | AgentCore standard; modern typing support |
| Framework | FastAPI | Latest | Existing AgentCore infrastructure; async support |
| API Protocol | JSON-RPC 2.0 | A2A v0.2 | AgentCore standard for agent communication |
| Validation | Pydantic | v2.x | AgentCore standard; strict typing validation |
| LLM Client | Async HTTP Client | aiohttp | Async LLM calls with stop sequences and token counting |
| Token Counting | tiktoken or provider API | Latest | Accurate token budget tracking |
| Testing | pytest-asyncio | Latest | AgentCore standard; 90%+ coverage requirement |
| Type Checking | mypy (strict) | Latest | AgentCore standard; strict mode enabled |
| Monitoring | Prometheus | Latest | AgentCore standard metrics collection |
| Tracing | A2A Context | Built-in | Distributed tracing via `trace_id` |

**Key Technology Decisions:**

- **Async-first architecture:** All reasoning operations async for non-blocking execution
- **Stateless design:** No persistent state in engine; enables horizontal scaling
- **Pydantic validation:** Strict input/output validation prevents invalid parameters
- **A2A protocol integration:** Standard JSON-RPC method registration pattern
- **Token counting integration:** Required for accurate chunk size management

**Research Citations:**

- Linear O(N) scaling vs quadratic O(N²) - Source: docs/research/bounded-context-reasoning.md
- Carryover compression mechanism - Source: docs/research/bounded-context-reasoning.md
- 50-98% compute reduction benchmarks - Source: docs/research/bounded-context-reasoning.md

### Alignment with Existing System

**From CLAUDE.md:**

- **Consistent With:**
  - Python 3.12+ runtime
  - FastAPI framework
  - JSON-RPC 2.0 protocol
  - Pydantic validation
  - Async-first patterns
  - pytest-asyncio testing
  - mypy strict type checking

- **New Additions:**
  - LLM client with token counting capability
  - Stop sequence support for answer detection
  - Carryover generation prompting
  - Compute metrics calculation

- **Migration Considerations:**
  - None - pure addition to existing stack
  - No breaking changes to existing components

---

## 🏗️ Architecture Design

### System Context (from CLAUDE.md)

**Existing System Architecture:**
AgentCore implements Google's A2A (Agent2Agent) protocol v0.2 with JSON-RPC 2.0 infrastructure for agent communication, discovery, and task management. The system uses a service-oriented architecture with:

- JSON-RPC Handler: Central request/response processor with method registry
- Services Layer: Agent management, task management, message routing, security
- Database Layer: PostgreSQL with async SQLAlchemy for persistence
- Routers Layer: FastAPI endpoints for JSON-RPC, WebSocket, health checks

**Integration Points:**

- **JSON-RPC Infrastructure:** Register `reasoning.bounded_context` method via `@register_jsonrpc_method` decorator
- **Agent Management:** Extend AgentCard capabilities to advertise bounded reasoning support
- **Message Routing:** Enable routing decisions based on reasoning capabilities
- **Security Service:** JWT authentication for reasoning endpoint access
- **Monitoring:** Prometheus metrics for compute savings and performance tracking

**New Architectural Patterns:**

- Bounded iteration engine (stateless reasoning service)
- Carryover compression mechanism (state summarization)
- Compute metrics calculation (efficiency tracking)

### Component Architecture

**Architecture Pattern:** Service Layer Component (Stateless)

- **Rationale:** Aligns with existing AgentCore service-oriented architecture; stateless design enables horizontal scaling
- **Alignment:** Follows existing patterns in `services/agent_manager.py`, `services/task_manager.py`

**System Design:**

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                      AgentCore System                        │
│                                                              │
│  ┌────────────────┐         ┌──────────────────────────┐   │
│  │  JSON-RPC      │         │  Bounded Context         │   │
│  │  Handler       │────────▶│  Reasoning Service       │   │
│  │  (existing)    │         │  (new)                   │   │
│  └────────────────┘         └──────────────────────────┘   │
│         │                              │                     │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌────────────────┐         ┌──────────────────────────┐   │
│  │  Agent         │         │  LLM Client              │   │
│  │  Manager       │◀────────│  (with token counting)   │   │
│  │  (existing)    │         │  (new)                   │   │
│  └────────────────┘         └──────────────────────────┘   │
│         │                              │                     │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌────────────────┐         ┌──────────────────────────┐   │
│  │  AgentCard     │         │  Prometheus Metrics      │   │
│  │  (enhanced)    │         │  (compute savings)       │   │
│  └────────────────┘         └──────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Data Flow (Bounded Reasoning Request):
1. Client → JSON-RPC Handler → reasoning.bounded_context
2. Handler → BoundedContextEngine.reason(query, params)
3. Engine → LLM Client (iteration 1: prompt)
4. LLM Client → Engine (reasoning chunk 1)
5. Engine → LLM Client (carryover generation)
6. LLM Client → Engine (compressed carryover)
7. Engine → LLM Client (iteration 2: prompt + carryover)
8. ... (repeat until answer or max iterations)
9. Engine → Metrics Calculator (compute savings)
10. Engine → JSON-RPC Handler (result + metrics)
11. JSON-RPC Handler → Client (response)
```

### Architecture Decisions (from Research)

**Decision 1: Stateless Reasoning Engine**

- **Choice:** Stateless service with no persistent state between requests
- **Rationale:** Enables horizontal scaling, simplifies deployment, aligns with existing AgentCore stateless patterns
- **Implementation:** All iteration state maintained in local variables; carryover passed between iterations
- **Trade-offs:** Cannot resume interrupted reasoning (acceptable for P0); future enhancement could add optional persistence

**Decision 2: Synchronous Iteration, Async I/O**

- **Choice:** Sequential iteration execution with async LLM calls
- **Rationale:** Reasoning must be sequential (iteration N+1 depends on N); async I/O prevents blocking
- **Implementation:** `for` loop with `await` for LLM calls
- **Trade-offs:** Cannot parallelize iterations (inherent to reasoning); acceptable given compute savings

**Decision 3: JSON-RPC Protocol Integration**

- **Choice:** Standard JSON-RPC method registration vs custom REST endpoint
- **Rationale:** Consistency with existing AgentCore A2A protocol; enables agent-to-agent reasoning delegation
- **Implementation:** `@register_jsonrpc_method("reasoning.bounded_context")` decorator
- **Trade-offs:** JSON-RPC overhead minimal; consistency benefit outweighs cost

**Decision 4: In-Memory Carryover (No Persistence)**

- **Choice:** Carryover exists only in memory during reasoning execution
- **Rationale:** Simplicity for P0; persistence adds complexity without immediate value
- **Implementation:** Carryover stored in local variable, discarded after response
- **Trade-offs:** Cannot inspect intermediate carryovers post-execution; future enhancement for debugging

### Component Breakdown

**Core Components:**

1. **BoundedContextEngine**
   - **Purpose:** Orchestrate multi-iteration reasoning with fixed context windows
   - **Technology:** Python 3.12+, async/await
   - **Pattern:** Service class with async methods
   - **Interfaces:**
     - `async def reason(query: str, chunk_size: int, carryover_size: int, max_iterations: int) -> ReasoningResult`
   - **Dependencies:** LLMClient, MetricsCalculator

2. **LLMClient**
   - **Purpose:** Async interface to LLM provider with token counting and stop sequences
   - **Technology:** aiohttp, tiktoken (or provider API)
   - **Pattern:** Adapter pattern for provider-agnostic interface
   - **Interfaces:**
     - `async def generate(prompt: str, max_tokens: int, stop_sequences: list[str]) -> GenerationResult`
     - `def count_tokens(text: str) -> int`
   - **Dependencies:** External LLM provider API (OpenAI, Anthropic, etc.)

3. **CarryoverGenerator**
   - **Purpose:** Generate compressed summaries at iteration boundaries
   - **Technology:** Python 3.12+, async/await
   - **Pattern:** Composition within BoundedContextEngine
   - **Interfaces:**
     - `async def generate_carryover(full_context: str, max_tokens: int) -> str`
   - **Dependencies:** LLMClient

4. **MetricsCalculator**
   - **Purpose:** Calculate compute savings and efficiency metrics
   - **Technology:** Python 3.12+
   - **Pattern:** Utility class
   - **Interfaces:**
     - `def calculate_compute_savings(iterations: list[ReasoningIteration], chunk_size: int) -> float`
   - **Dependencies:** None (pure calculation)

5. **ReasoningJSONRPC Handler**
   - **Purpose:** JSON-RPC method handler for `reasoning.bounded_context`
   - **Technology:** FastAPI, Pydantic
   - **Pattern:** Follows existing `*_jsonrpc.py` pattern
   - **Interfaces:**
     - `@register_jsonrpc_method("reasoning.bounded_context")`
     - `async def handle_bounded_reasoning(request: JsonRpcRequest) -> dict[str, Any]`
   - **Dependencies:** BoundedContextEngine, JsonRpcRequest/Response models

### Data Flow & Boundaries

**Request Flow:**

1. **Entry:** Client sends JSON-RPC request to `/api/v1/jsonrpc` with method `reasoning.bounded_context`
2. **Validation:** JsonRpcProcessor validates request format; Pydantic validates params
3. **Authentication:** Security service validates JWT token (if required)
4. **Execution:** Handler invokes BoundedContextEngine.reason()
5. **Iteration Loop:**
   - Build context (prompt + carryover)
   - Call LLM for reasoning chunk
   - Detect answer via stop sequences
   - Generate carryover if continuing
   - Repeat until answer or max iterations
6. **Metrics:** Calculate compute savings
7. **Response:** Return result with iterations, total tokens, compute savings

**Component Boundaries:**

- **Public Interface:** JSON-RPC method `reasoning.bounded_context`
  - Input: `{query, chunk_size?, carryover_size?, max_iterations?}`
  - Output: `{answer, iterations[], total_tokens, compute_savings_pct}`
- **Internal Implementation:** BoundedContextEngine, LLMClient, CarryoverGenerator, MetricsCalculator
  - Hidden from external clients
  - Subject to change without breaking API contract
- **Cross-Component Contracts:**
  - Agent Management: AgentCard capabilities list includes `bounded_context_reasoning`
  - Security Service: JWT authentication for reasoning endpoint
  - Monitoring: Prometheus metrics for compute savings and performance

## 4. Technical Specification

### Data Model

**Core Entities:**

```python
from pydantic import BaseModel, Field

# Request Models
class BoundedReasoningParams(BaseModel):
    """Parameters for bounded context reasoning."""
    query: str = Field(..., min_length=1, max_length=50000, description="The problem to solve")
    chunk_size: int = Field(default=8192, ge=1024, le=32768, description="Max tokens per iteration")
    carryover_size: int = Field(default=4096, ge=512, le=16384, description="Tokens to carry forward")
    max_iterations: int = Field(default=5, ge=1, le=50, description="Max reasoning iterations")

# Response Models
class ReasoningIteration(BaseModel):
    """Details of a single reasoning iteration."""
    iteration: int = Field(..., ge=0, description="Iteration number (0-indexed)")
    tokens: int = Field(..., ge=0, description="Tokens consumed in this iteration")
    has_answer: bool = Field(..., description="Whether answer was found in this iteration")

class BoundedReasoningResult(BaseModel):
    """Result of bounded context reasoning."""
    answer: str = Field(..., description="Final answer after reasoning")
    iterations: list[ReasoningIteration] = Field(..., description="Details of each iteration")
    total_tokens: int = Field(..., ge=0, description="Total tokens processed across all iterations")
    compute_savings_pct: float = Field(..., ge=0, le=100, description="Percentage saved vs traditional")

# Internal Models
class GenerationResult(BaseModel):
    """Result from LLM generation."""
    text: str
    token_count: int
    stop_reason: str | None  # "stop_sequence", "max_tokens", or None

class CarryoverContent(BaseModel):
    """Structured carryover content."""
    current_strategy: str
    key_findings: str
    progress: str
    next_steps: str
    unresolved: str
```

**Validation Rules:**

- Query: 1-50,000 characters (prevent abuse)
- chunk_size: 1024-32,768 tokens (reasonable bounds for LLM context windows)
- carryover_size: 512-16,384 tokens (must be less than chunk_size)
- max_iterations: 1-50 (prevent infinite loops)

**No Database Persistence (P0):**

- Reasoning executions are ephemeral (no storage of results)
- Future enhancement: Optional persistence in PostgreSQL for debugging/analysis
- Migration approach: Add `ReasoningExecutionRecord` table if persistence needed

### API Design

**Top 6 Critical Endpoints:**

1. **`reasoning.bounded_context` (JSON-RPC Method)**
   - **Method:** POST `/api/v1/jsonrpc`
   - **Purpose:** Execute bounded context reasoning
   - **Request Schema:**

     ```json
     {
       "jsonrpc": "2.0",
       "method": "reasoning.bounded_context",
       "params": {
         "query": "string",
         "chunk_size": 8192,
         "carryover_size": 4096,
         "max_iterations": 5
       },
       "id": "unique-request-id"
     }
     ```

   - **Response Schema:**

     ```json
     {
       "jsonrpc": "2.0",
       "result": {
         "answer": "string",
         "iterations": [
           {"iteration": 0, "tokens": 8192, "has_answer": false},
           {"iteration": 1, "tokens": 8192, "has_answer": true}
         ],
         "total_tokens": 16384,
         "compute_savings_pct": 67.5
       },
       "id": "unique-request-id"
     }
     ```

   - **Error Handling:**
     - `INVALID_PARAMS` (-32602): Invalid query or parameter values
     - `INTERNAL_ERROR` (-32603): LLM provider failure
     - Custom error code -32001: Max iterations reached without answer

2. **Agent Capability Advertisement (Extends Existing)**
   - **Method:** Extends `agent.register` JSON-RPC method
   - **Purpose:** Advertise bounded reasoning capability in AgentCard
   - **Enhancement:** Add to `capabilities` array:

     ```json
     {
       "capabilities": ["bounded_context_reasoning", "long_form_reasoning"],
       "supported_methods": ["reasoning.bounded_context"]
     }
     ```

3. **Agent Discovery (Extends Existing)**
   - **Method:** Extends `agent.discover` JSON-RPC method
   - **Purpose:** Filter agents by bounded reasoning capability
   - **Enhancement:** Support capability filtering:

     ```json
     {
       "method": "agent.discover",
       "params": {"capabilities": ["bounded_context_reasoning"]}
     }
     ```

4. **Metrics Endpoint (Prometheus)**
   - **Method:** GET `/metrics`
   - **Purpose:** Expose Prometheus metrics for compute savings
   - **Metrics:**
     - `reasoning_bounded_context_requests_total{status}` - Total requests
     - `reasoning_bounded_context_duration_seconds` - Request duration histogram
     - `reasoning_bounded_context_tokens_total` - Total tokens processed
     - `reasoning_bounded_context_compute_savings_pct` - Compute savings histogram
     - `reasoning_bounded_context_iterations_total` - Iteration count histogram

5. **Health Check (Extends Existing)**
   - **Method:** GET `/health`
   - **Purpose:** Include reasoning service health in overall health check
   - **Enhancement:** Add LLM client health check (ping LLM provider)

6. **Configuration Endpoint (Future)**
   - **Method:** `reasoning.get_config` (future enhancement)
   - **Purpose:** Retrieve current reasoning configuration
   - **Response:** `{chunk_size, carryover_size, max_iterations, max_allowed_iterations}`

### Security (from Research)

**Based on:** AgentCore security patterns (CLAUDE.md) + Spec requirements

**Authentication/Authorization:**

- **Approach:** JWT token authentication (existing AgentCore pattern)
- **Implementation:**
  - Reasoning endpoint requires valid JWT token in request headers
  - Security service validates token before invoking handler
  - RBAC: Only agents with `reasoning:execute` permission can invoke
- **Standards:** JWT (RFC 7519), A2A protocol authentication patterns

**Secrets Management:**

- **Strategy:** Environment variables for LLM provider API keys
- **Pattern:**
  - `LLM_PROVIDER_API_KEY` environment variable
  - Loaded via Pydantic Settings in `config.py`
  - Never logged or exposed in responses
- **Rotation:** API key rotation via environment variable update (no code changes)

**Data Protection:**

- **Encryption in Transit:** TLS/SSL for all LLM provider API calls
- **Encryption at Rest:** Not applicable (no persistent storage in P0)
- **PII Handling:**
  - Query content not logged by default
  - Optional debug logging with PII redaction
  - Compliance: GDPR/CCPA via query content isolation

**Security Testing:**

- **Approach:**
  - Input validation testing (malformed queries, out-of-range parameters)
  - Authentication testing (missing/invalid JWT tokens)
  - Injection testing (SQL injection N/A, prompt injection considered)
- **Tools:**
  - pytest security test suite
  - Manual prompt injection testing

**Compliance:**

- No PII stored (ephemeral execution)
- Query content treated as sensitive data (not logged)
- Token usage metrics do not contain PII

### Performance (from Research)

**Based on:** Research benchmarks (50-98% compute reduction)

**Performance Targets:**

- **Response Time:** <20% latency increase vs traditional reasoning (p95)
  - Traditional 10K token reasoning: ~30s
  - Bounded reasoning target: <36s (acceptable given 50-98% compute savings)
- **Throughput:** >10 concurrent reasoning requests per instance
  - Linear scaling with horizontal instances
- **Resource Usage:**
  - Memory: O(1) per request (constant regardless of reasoning depth)
  - CPU: Minimal (majority of time in LLM API calls)
  - Network: ~50-90% reduction vs traditional (fewer tokens transmitted)

**Caching Strategy:**

- **Approach:** Optional carryover pattern caching (future enhancement)
- **Pattern:** Redis cache for common carryover structures
- **TTL Strategy:** Time-based (1 hour) for pattern cache
- **Invalidation:** LRU eviction for memory-constrained environments
- **P0 Decision:** No caching in initial implementation (complexity vs value)

**LLM Provider Optimization:**

- **Connection Pooling:** aiohttp connection pooling for LLM API calls
- **Batching:** Not applicable (reasoning must be sequential)
- **Timeout Handling:** 60s timeout per LLM call; configurable
- **Retry Logic:** 3 retries with exponential backoff for transient failures

**Scaling Strategy:**

- **Horizontal:** Stateless design enables simple horizontal scaling
  - Kubernetes HPA based on CPU/memory or custom metrics (reasoning queue depth)
  - No shared state between instances
- **Vertical:** Minimal memory requirements (O(1) per request)
- **Auto-scaling:**
  - Trigger: Request queue depth >10 or p95 latency >40s
  - Scale up: Add instances in increments of 2
  - Scale down: Remove instances when queue depth <5 and p95 <30s
- **Performance Monitoring:**
  - Prometheus metrics for latency, throughput, compute savings
  - Grafana dashboards for real-time visualization
  - Alerting on p95 latency >45s or error rate >1%

## 5. Development Setup

**Required Tools:**

- Python 3.12+
- uv package manager (AgentCore standard)
- Docker & Docker Compose (for local FastAPI + PostgreSQL + Redis)
- LLM provider API key (OpenAI, Anthropic, or compatible)

**Local Environment:**

```bash
# Install dependencies
uv add aiohttp tiktoken pydantic pytest-asyncio pytest-cov

# Environment variables (.env)
LLM_PROVIDER_API_KEY=sk-...
REASONING_CHUNK_SIZE=8192
REASONING_CARRYOVER_SIZE=4096
REASONING_MAX_ITERATIONS=5

# Run local development server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# Run tests
uv run pytest tests/unit/reasoning/ tests/integration/reasoning/ -v --cov=agentcore.reasoning --cov-report=term-missing --cov-fail-under=90
```

**CI/CD Pipeline Requirements:**

- **Build:** Python 3.12+ environment with uv
- **Test:** pytest with 90%+ coverage gate
- **Lint:** ruff check (AgentCore standard)
- **Type Check:** mypy --strict (AgentCore standard)
- **Security Scan:** bandit for Python security issues
- **Deploy:** Kubernetes manifest update (if merged to main)

**Testing Framework:**

- **Unit Tests:** pytest-asyncio for async engine logic
  - `tests/unit/reasoning/test_bounded_context_engine.py`
  - `tests/unit/reasoning/test_carryover_generator.py`
  - `tests/unit/reasoning/test_metrics_calculator.py`
- **Integration Tests:** pytest-asyncio for JSON-RPC API
  - `tests/integration/reasoning/test_reasoning_jsonrpc.py`
  - `tests/integration/reasoning/test_agent_capability.py`
- **Performance Tests:** pytest-benchmark for latency and throughput
  - `tests/performance/reasoning/test_compute_savings.py`
  - `tests/performance/reasoning/test_concurrent_requests.py`
- **Coverage Target:** 90%+ (AgentCore standard)

## 6. Risk Management

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **LLM provider API failures** | High (reasoning unavailable) | Medium | Retry logic with exponential backoff; circuit breaker pattern; fallback to degraded mode (fewer iterations) |
| **Token counting inaccuracy** | Medium (budget overruns) | Low | Use official tokenizer (tiktoken); validate against provider API; add 10% safety margin |
| **Carryover quality degradation** | High (reasoning continuity lost) | Medium | Structured carryover format (JSON); validation of carryover content; metrics on carryover information retention |
| **Max iterations without answer** | Medium (user frustration) | High | Clear error message with partial reasoning; configurable max iterations; future: adaptive iteration budgets |
| **Latency regression** | Medium (user experience) | Low | Performance benchmarks in CI/CD; alerting on p95 >45s; horizontal scaling for load spikes |
| **Prompt injection attacks** | Low (no code execution) | Medium | Input sanitization; prompt engineering best practices; future: content filtering |
| **Memory leaks in iteration loop** | Medium (service instability) | Low | Careful resource management; memory profiling; automated restart on memory thresholds |
| **Integration complexity** | Low (delayed launch) | Medium | Incremental integration (JSON-RPC first, agent discovery later); phased rollout |

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Deliverables:**

- Core BoundedContextEngine with iteration loop
- LLMClient abstraction with token counting
- CarryoverGenerator for state compression
- Pydantic models for request/response
- Unit tests (90%+ coverage)

**Tasks:**

1. Create `agentcore/reasoning/` module structure
2. Implement `BoundedContextEngine` class with `reason()` method
3. Implement `LLMClient` adapter for LLM provider API
4. Implement `CarryoverGenerator` for compression
5. Implement `MetricsCalculator` for compute savings
6. Write unit tests for all components
7. Manual testing with sample reasoning queries

**Dependencies:** LLM provider API access

### Phase 2: JSON-RPC Integration (Week 3)

**Deliverables:**

- JSON-RPC method `reasoning.bounded_context` registered
- Integration with existing AgentCore infrastructure
- Request/response validation via Pydantic
- Integration tests for API

**Tasks:**

1. Create `agentcore/a2a_protocol/services/reasoning_jsonrpc.py`
2. Implement `handle_bounded_reasoning()` handler
3. Register method via `@register_jsonrpc_method` decorator
4. Import module in `main.py` for auto-registration
5. Add security middleware for JWT authentication
6. Write integration tests for JSON-RPC method
7. Test via curl/Postman against local server

**Dependencies:** Phase 1 complete

### Phase 3: Agent Integration (Week 4)

**Deliverables:**

- AgentCard capability advertisement for bounded reasoning
- Agent discovery filtering by reasoning capabilities
- Routing enhancements for reasoning tasks
- Integration tests for agent workflows

**Tasks:**

1. Extend `models/agent.py` AgentCard with reasoning capabilities
2. Update `services/agent_manager.py` for capability validation
3. Update `services/message_router.py` for reasoning-aware routing
4. Add agent discovery tests with capability filtering
5. Test end-to-end: agent registration → discovery → reasoning invocation

**Dependencies:** Phase 2 complete

### Phase 4: Hardening (Week 5)

**Deliverables:**

- Prometheus metrics for compute savings
- Performance benchmarks and optimization
- Security hardening (input validation, rate limiting)
- Production-ready documentation

**Tasks:**

1. Add Prometheus metrics collection
2. Create Grafana dashboards for visualization
3. Run performance benchmarks (compute savings, latency, throughput)
4. Optimize LLM client (connection pooling, timeout handling)
5. Add rate limiting for reasoning endpoint
6. Write API documentation (OpenAPI/Swagger)
7. Write configuration guide for tuning parameters
8. Production deployment checklist

**Dependencies:** Phase 3 complete

### Phase 5: Launch (Week 6+)

**Deliverables:**

- Production deployment to staging environment
- A/B testing vs traditional reasoning
- Monitoring and alerting setup
- Post-launch support plan

**Tasks:**

1. Deploy to staging environment
2. Run A/B tests comparing bounded vs traditional reasoning
3. Set up alerting (p95 latency, error rate, compute savings)
4. Gradual rollout (10% → 50% → 100% traffic)
5. Monitor metrics for 48 hours
6. Address any production issues
7. Full rollout to production

**Dependencies:** Phase 4 complete

## 8. Quality Assurance

**Testing Strategy:**

1. **Unit Tests (90%+ coverage):**
   - BoundedContextEngine.reason() with various parameters
   - CarryoverGenerator.generate_carryover() compression quality
   - MetricsCalculator.calculate_compute_savings() accuracy
   - LLMClient mock tests for error handling
   - Edge cases: max iterations, empty query, invalid parameters

2. **Integration Tests:**
   - JSON-RPC method `reasoning.bounded_context` invocation
   - End-to-end reasoning flow (request → iterations → response)
   - Agent capability advertisement and discovery
   - Error handling (LLM failures, invalid inputs)
   - Concurrent requests handling

3. **Performance Tests:**
   - Compute savings benchmarks (vs traditional reasoning)
   - Memory usage validation (constant O(1))
   - Latency targets (p50, p95, p99)
   - Throughput under load (10+ concurrent requests)
   - Scaling tests (horizontal instances)

4. **Quality Tests:**
   - Reasoning accuracy vs traditional approaches
   - Carryover information retention metrics
   - Extended reasoning (50K+ tokens) validation
   - Answer detection reliability

**Code Quality Gates:**

- Ruff linting: Zero errors
- mypy --strict: Zero type errors
- pytest coverage: >=90%
- Bandit security scan: No high-severity issues
- Code review: 2 approvals required

**Deployment Verification Checklist:**

- [ ] All tests passing (unit, integration, performance)
- [ ] Coverage >=90%
- [ ] Linting and type checking clean
- [ ] API documentation complete
- [ ] Configuration guide written
- [ ] Prometheus metrics verified
- [ ] Grafana dashboards created
- [ ] Alerting rules configured
- [ ] Staging deployment successful
- [ ] A/B testing completed
- [ ] Security review passed
- [ ] Performance benchmarks met

**Monitoring and Alerting Setup:**

- **Metrics:** reasoning requests/s, p95 latency, error rate, compute savings %
- **Alerts:**
  - Error rate >1% (critical)
  - p95 latency >45s (warning)
  - Compute savings <50% for >10K token reasoning (warning)
  - LLM provider API failures >5% (critical)
- **Dashboards:**
  - Reasoning overview: requests, latency, errors
  - Compute savings: % savings by query size
  - Iteration analysis: avg iterations, max iterations reached
  - LLM provider health: API latency, failure rate

---

## ⚠️ Error Handling & Edge Cases

**From:** Specification NFR-11 + Research findings

### Error Scenarios

**Critical Error Paths:**

1. **LLM Provider API Failure**
   - **Cause:** Network error, API downtime, rate limiting, authentication failure
   - **Impact:** Reasoning request fails; user unable to get answer
   - **Handling:**
     - Retry with exponential backoff (3 attempts: 1s, 2s, 4s)
     - Circuit breaker pattern (open after 5 consecutive failures, close after 60s)
     - Return JSON-RPC error with code -32603 (Internal Error) and descriptive message
   - **Recovery:** Automatic retry; manual investigation if circuit breaker opens
   - **User Experience:** "Reasoning service temporarily unavailable. Please try again in a moment."

2. **Max Iterations Reached Without Answer**
   - **Cause:** Query too complex for iteration budget; carryover quality degradation
   - **Impact:** User does not receive final answer; partial reasoning wasted
   - **Handling:**
     - Return custom JSON-RPC error code -32001 (Max Iterations Reached)
     - Include partial reasoning trace in error details (iterations array)
     - Suggest increasing max_iterations parameter
   - **Recovery:** User retries with higher max_iterations
   - **User Experience:** "Unable to complete reasoning within 5 iterations. Consider increasing max_iterations to 10."

3. **Token Budget Exceeded**
   - **Cause:** Carryover size misconfigured; token counting inaccuracy
   - **Impact:** LLM API rejects request due to context length limit
   - **Handling:**
     - Validate total context size before LLM call (prompt + carryover <= chunk_size)
     - Truncate carryover if necessary (with warning log)
     - Return JSON-RPC error if truncation insufficient
   - **Recovery:** Reduce carryover_size parameter; improve carryover compression
   - **User Experience:** "Context size exceeded. Try reducing carryover_size."

4. **Invalid Carryover Generation**
   - **Cause:** LLM fails to generate valid carryover; empty response
   - **Impact:** Reasoning continuity lost; subsequent iterations lack context
   - **Handling:**
     - Validate carryover content (non-empty, reasonable length)
     - Use previous iteration's raw output if carryover generation fails
     - Log warning for monitoring
   - **Recovery:** Continue with degraded carryover; reasoning may still succeed
   - **User Experience:** Transparent (no error to user; internal recovery)

### Edge Cases

**Identified in Specification:**

1. **Empty Query String**
   - **Detection:** Pydantic validation (min_length=1)
   - **Handling:** Return JSON-RPC error -32602 (Invalid Params)
   - **Testing:** Unit test with `query=""`

2. **Query Exceeds Maximum Length**
   - **Detection:** Pydantic validation (max_length=50000)
   - **Handling:** Return JSON-RPC error -32602 (Invalid Params)
   - **Testing:** Unit test with 50,001 character query

3. **chunk_size < carryover_size**
   - **Detection:** Custom Pydantic validator
   - **Handling:** Return JSON-RPC error -32602 with message "carryover_size must be less than chunk_size"
   - **Testing:** Unit test with `chunk_size=4096, carryover_size=8192`

4. **max_iterations = 0**
   - **Detection:** Pydantic validation (ge=1)
   - **Handling:** Return JSON-RPC error -32602 (Invalid Params)
   - **Testing:** Unit test with `max_iterations=0`

5. **Answer Found in First Iteration**
   - **Detection:** `<answer>` stop sequence in initial response
   - **Handling:** Return immediately with 1 iteration; no carryover generated
   - **Testing:** Integration test with trivial query

6. **No Stop Sequence in Response**
   - **Detection:** LLM response does not contain `<answer>` or `<continue>`
   - **Handling:** Treat as `<continue>` and proceed to next iteration
   - **Testing:** Mock LLM response without stop sequences

**Identified in Research:**

7. **Carryover Compression Degrades Quality**
   - **Detection:** Reasoning accuracy drops below baseline (requires metrics)
   - **Handling:** Future enhancement: Adaptive carryover size based on compression quality metrics
   - **Testing:** Quality tests comparing accuracy with varying carryover sizes

8. **Very Long Reasoning (>50K Tokens)**
   - **Detection:** User requests high max_iterations and chunk_size
   - **Handling:** Enforce max_allowed_iterations=50 (spec NFR-5)
   - **Testing:** Performance test with max parameters

### Input Validation

**Validation Rules:**

- `query`: String, 1-50,000 characters
- `chunk_size`: Integer, 1024-32,768
- `carryover_size`: Integer, 512-16,384, must be < chunk_size
- `max_iterations`: Integer, 1-50

**Sanitization:**

- Prompt injection prevention: Escape control characters in query
- No SQL injection (no database queries in P0)
- Input normalization: Strip leading/trailing whitespace from query

### Graceful Degradation

**Fallback Strategies:**

1. **LLM Provider Unavailable**
   - **Fallback:** Return error after retries; no degraded mode in P0
   - **Future:** Fallback to simpler reasoning model (e.g., GPT-4o-mini)

2. **Partial Iteration Failure**
   - **Fallback:** Return partial reasoning result with iterations completed before failure
   - **User can retry from last successful iteration (future enhancement)**

3. **High Latency**
   - **Fallback:** No automatic degradation; user can reduce max_iterations for faster (but potentially incomplete) reasoning

### Monitoring & Alerting

**Error Tracking:**

- **Tool:** Prometheus + Grafana (existing AgentCore stack)
- **Metrics:**
  - `reasoning_bounded_context_errors_total{error_type}` - Counter of errors by type
  - `reasoning_bounded_context_llm_failures_total` - LLM provider failure count
  - `reasoning_bounded_context_max_iterations_reached_total` - Count of incomplete reasoning
- **Threshold:**
  - Alert if error rate >1% over 5 minutes (critical)
  - Alert if LLM failure rate >5% over 5 minutes (critical)
  - Alert if max iterations reached >20% of requests (warning)
- **Response:**
  - Critical: Page on-call engineer
  - Warning: Slack notification to team channel
  - Investigate: Review logs, check LLM provider status page

**Incident Response Plan:**

1. Check Grafana dashboard for error spike
2. Review Prometheus metrics for error types
3. Check LLM provider status page
4. Review application logs for stack traces
5. If LLM provider issue: Wait for provider recovery
6. If application issue: Rollback deployment or apply hotfix
7. Post-incident: Update runbook with lessons learned

---

## 📚 References & Traceability

### Source Documentation

**Research:**

- **docs/research/bounded-context-reasoning.md**
  - Bounded context reasoning paradigm with fixed-size context windows
  - Linear O(N) vs quadratic O(N²) computational complexity analysis
  - Carryover compression mechanism for state preservation
  - 50-98% compute reduction benchmarks
  - Training considerations for carryover generation
  - Implementation pseudocode and algorithm details

**Specification:**

- **docs/specs/bounded-context-reasoning/spec.md**
  - FR-1: Core BoundedContextEngine with configurable parameters
  - FR-2: Iteration management with answer detection
  - FR-3: Carryover generation preserving key insights
  - FR-4: JSON-RPC API integration
  - FR-5: Agent capability advertisement
  - NFR: Performance (50-90% compute reduction), security (JWT auth), scalability (stateless)
  - Success metrics and acceptance criteria

### System Context

**Architecture & Patterns:**

- **CLAUDE.md** - AgentCore system architecture and development patterns
  - JSON-RPC method registration pattern
  - Pydantic model validation pattern
  - Async service architecture
  - Database access patterns (not used in P0)
  - A2A context propagation
  - Testing requirements (90%+ coverage, pytest-asyncio)
  - Tech stack: Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy

### Research Citations

**Best Practices Research:**

- Bounded context reasoning paradigm - Source: docs/research/bounded-context-reasoning.md
- Carryover compression techniques - Source: docs/research/bounded-context-reasoning.md
- Computational complexity analysis (O(N) vs O(N²)) - Source: docs/research/bounded-context-reasoning.md

**Technology Evaluation:**

- Python 3.12+ typing (list[T], dict[K,V]) - AgentCore standard
- FastAPI async patterns - AgentCore standard
- Pydantic v2 validation - AgentCore standard
- pytest-asyncio testing - AgentCore standard

**Security Standards:**

- JWT authentication (RFC 7519) - AgentCore standard
- Input validation via Pydantic - AgentCore standard
- A2A protocol security patterns - AgentCore standard

**Performance Benchmarks:**

- 50-98% compute reduction - Source: docs/research/bounded-context-reasoning.md
- Linear scaling vs quadratic - Source: docs/research/bounded-context-reasoning.md
- Constant O(1) memory usage - Source: docs/research/bounded-context-reasoning.md

### Related Components

**Dependencies:**

- **agentcore.a2a_protocol.services.jsonrpc_handler** - JSON-RPC method registration
- **agentcore.a2a_protocol.models.jsonrpc** - Request/response models
- **agentcore.a2a_protocol.services.agent_manager** - Agent capability management
- **agentcore.a2a_protocol.models.agent** - AgentCard model
- **agentcore.a2a_protocol.services.security_service** - JWT authentication (optional for P0)

**Integration Points:**

- Agent Management: AgentCard capabilities, agent discovery
- Message Routing: Capability-based routing to reasoning agents
- Security Service: JWT authentication for reasoning endpoint
- Monitoring: Prometheus metrics, Grafana dashboards

---

## Implementation Checklist

**Phase 1: Foundation (Week 1-2)**

- [ ] Create `agentcore/reasoning/` module structure
- [ ] Implement `BoundedContextEngine` class
- [ ] Implement `LLMClient` adapter
- [ ] Implement `CarryoverGenerator`
- [ ] Implement `MetricsCalculator`
- [ ] Define Pydantic models (request/response)
- [ ] Write unit tests (90%+ coverage)
- [ ] Manual testing with sample queries

**Phase 2: JSON-RPC Integration (Week 3)**

- [ ] Create `reasoning_jsonrpc.py` handler
- [ ] Register `reasoning.bounded_context` method
- [ ] Import module in `main.py`
- [ ] Add JWT authentication middleware
- [ ] Write integration tests
- [ ] Test via curl/Postman

**Phase 3: Agent Integration (Week 4)**

- [ ] Extend AgentCard with reasoning capabilities
- [ ] Update agent registration validation
- [ ] Update message routing for reasoning tasks
- [ ] Write agent discovery tests
- [ ] End-to-end testing

**Phase 4: Hardening (Week 5)**

- [ ] Add Prometheus metrics
- [ ] Create Grafana dashboards
- [ ] Run performance benchmarks
- [ ] Optimize LLM client
- [ ] Add rate limiting
- [ ] Write API documentation
- [ ] Write configuration guide
- [ ] Production deployment checklist

**Phase 5: Launch (Week 6+)**

- [ ] Deploy to staging
- [ ] A/B testing
- [ ] Set up alerting
- [ ] Gradual rollout
- [ ] Monitor for 48 hours
- [ ] Full production rollout

---

---

## Revision History

**Version 2.0 (2025-10-15):**

This plan has been revised to reflect a strategy framework approach rather than a single bounded context implementation:

**Key Changes:**

1. **Architecture:** Shifted from single-purpose bounded context engine to pluggable strategy framework
2. **Priority:** Framework is P0, bounded context is P1 (optional strategy)
3. **API:** Changed from `reasoning.bounded_context` to unified `reasoning.execute` with strategy parameter
4. **Phases:** Restructured to prioritize framework (Phases 1-2), then bounded context (Phase 3), then additional strategies (Phase 5)
5. **Configuration:** Multi-level strategy selection (system > agent > request)
6. **Deployment:** System can deploy with zero reasoning strategies (all optional)

**Rationale:**

- AgentCore is a generic orchestration framework, not a reasoning-specific system
- Users need flexibility to choose reasoning approaches appropriate for their use cases
- Bounded context has limitations (carryover loss, latency overhead, complexity)
- Different problems benefit from different strategies (CoT, ReAct, ToT, Bounded Context)
- No strategy should be mandatory for framework adoption

**Implementation Notes:**

- Phases 1-2 build the framework infrastructure (required)
- Phase 3 implements bounded context as first strategy (recommended but optional)
- Phases 4-5 add agent integration and additional strategies (optional)
- All phase deliverables and tasks should be reviewed against the updated spec.md v2.0

**Updated Timeline:** 7 weeks (was 5 weeks) to accommodate framework development

---

**End of Implementation Blueprint (PRP)**
