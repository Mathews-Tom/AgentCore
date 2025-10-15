# Modular Agent Core Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-15
**Specification:** `docs/specs/modular-agent-core/spec.md`
**Research:** `docs/research/modular-agent-architecture.md`
**Component ID:** MOD-001
**Epic Ticket:** MOD-001

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → This Plan**

1. **Research Analysis:** docs/research/modular-agent-architecture.md
   - Modular architecture with four specialized modules (Planner, Executor, Verifier, Generator)
   - Coordination mechanism using structured message passing
   - Performance benefits: +15-20% task success rate, 30-40% cost reduction
   - Implementation pattern with async/await Python
   - Integration strategy with AgentCore A2A protocol

2. **Formal Specification:** docs/specs/modular-agent-core/spec.md
   - FR-1 through FR-5: Module interfaces and coordination requirements
   - NFR: Performance (15% improvement, <2x latency), scalability, reliability, security
   - Features: Modular execution, registration/discovery, plan refinement, optimization
   - Success metrics: Task success rate, tool call accuracy, latency, cost efficiency
   - Implementation phases: Foundation (1-2 weeks), Implementation (3-4 weeks), Optimization (5-6 weeks)

### Related Documentation

**System Context:**

- Architecture: CLAUDE.md - AgentCore A2A protocol implementation
- Tech Stack: Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy (async)
- Patterns: JSON-RPC method registration, async database access, A2A context propagation

**Existing Code Examples:**

- `src/agentcore/a2a_protocol/services/agent_manager.py` - Agent registration and discovery pattern
- `src/agentcore/a2a_protocol/models/agent.py` - AgentCard and capability models
- `src/agentcore/a2a_protocol/services/jsonrpc_handler.py` - JSON-RPC method registration

**Other Specifications:**

- Tool Integration Framework (TOOL-001) - Required for Executor module
- Memory Management System (MEM-001) - Optional integration for context persistence
- Context Management System (CTX-001) - Optional for bounded reasoning integration

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable AgentCore to handle complex, multi-step workflows through specialized module coordination, achieving superior reliability and performance compared to single-agent approaches.

**Value Proposition:**

- **Improved Reliability:** +15-20% task success rate on complex multi-step tasks through dedicated verification
- **Cost Efficiency:** 30-40% reduction in compute costs by using optimally-sized models per module
- **Better Scalability:** Independent module scaling enables targeted resource allocation
- **Enhanced Maintainability:** Clear separation of concerns simplifies debugging and updates
- **Competitive Differentiation:** Handles workflows that monolithic agents cannot complete reliably

**Target Users:**

- **Agent Developers:** Building complex multi-step agent workflows requiring high reliability
- **Platform Operators:** Managing agent infrastructure at scale with predictable costs
- **Enterprise Users:** Requiring auditable, reliable agent execution for critical workflows
- **Researchers:** Experimenting with advanced agent architectures and optimization strategies

### Technical Approach

**Architecture Pattern:** Microservices / Modular Components (within A2A Protocol)

- Four specialized modules registered as A2A agents (Planner, Executor, Verifier, Generator)
- Coordination through JSON-RPC 2.0 message passing
- State management via PostgreSQL for execution plans and results
- Async-first implementation using Python asyncio
- Event-driven monitoring through AgentCore event streaming

**Technology Stack:**

- **Runtime:** Python 3.12+ (AgentCore standard)
- **Framework:** FastAPI (existing AgentCore stack)
- **Protocol:** JSON-RPC 2.0 (A2A v0.2 compliant)
- **Database:** PostgreSQL with async SQLAlchemy for execution state
- **Validation:** Pydantic v2 for request/response schemas
- **LLM Integration:** Portkey AI gateway for multi-provider support
- **Monitoring:** Prometheus metrics, OpenTelemetry tracing
- **Testing:** pytest-asyncio, 90%+ coverage requirement

**Implementation Strategy:**

1. **Phase 1 (Week 1-2):** Module interfaces, base classes, coordination infrastructure, database schema
2. **Phase 2 (Week 3-4):** Module implementations, A2A registration, JSON-RPC integration, end-to-end flow
3. **Phase 3 (Week 5-6):** Plan refinement loop, distributed tracing, performance optimization, monitoring

### Key Success Metrics

**Service Level Objectives (SLOs):**

- **Availability:** 99.9% uptime for modular execution endpoint
- **Response Time:** <2x single-agent baseline (p95) - acceptable tradeoff for quality improvement
- **Module Transition Latency:** <500ms per module transition (excluding tool execution)
- **Throughput:** 100+ concurrent modular executions per instance

**Key Performance Indicators (KPIs):**

- **Task Success Rate:** +15% improvement over single-agent baseline (target: 15-20%)
- **Tool Call Accuracy:** +10% improvement in correct tool invocations
- **Cost per Task:** -30% reduction through optimized module sizing
- **Error Recovery Rate:** >80% of retryable errors recovered automatically
- **Latency:** <2x single-agent baseline for equivalent tasks

---

## 💻 Code Examples & Patterns

### Repository Patterns (from AgentCore Codebase)

**Relevant Existing Patterns:**

1. **JSON-RPC Method Registration Pattern:**
   - **Application:** Register `modular.solve` method and module-specific methods
   - **Usage Example:**

     ```python
     from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

     @register_jsonrpc_method("modular.solve")
     async def handle_modular_solve(request: JsonRpcRequest) -> dict[str, Any]:
         params = ModularExecutionParams(**request.params)
         result = await modular_agent.solve(params.query, params.config)
         return result.model_dump()
     ```

   - **Adaptation Notes:** Create `services/modular_jsonrpc.py` following existing `*_jsonrpc.py` pattern

2. **Agent Registration Pattern:**
   - **Application:** Register each module as an A2A agent with capabilities
   - **Usage Example:**

     ```python
     from agentcore.a2a_protocol.services.agent_manager import agent_manager
     from agentcore.a2a_protocol.models.agent import AgentCard, AgentCapability

     planner_card = AgentCard(
         agent_name="Planner Module",
         endpoints=[AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)],
         capabilities=[
             AgentCapability(
                 name="task_decomposition",
                 description="Decomposes complex queries into execution plans"
             ),
             AgentCapability(
                 name="strategy_formulation",
                 description="Creates execution strategies for multi-step tasks"
             )
         ],
         authentication=AgentAuthentication(type=AuthenticationType.JWT, required=True)
     )

     await agent_manager.register_agent(
         AgentRegistrationRequest(agent_card=planner_card)
     )
     ```

   - **Adaptation Notes:** Each of the four modules registers similarly with module-specific capabilities

3. **Async Service Pattern:**
   - **Application:** Implement module classes as async services
   - **Usage Example:**

     ```python
     class PlannerModule:
         async def create_plan(self, query: str, memory: Memory) -> Plan:
             # Implementation using async LLM calls
             pass
     ```

   - **Adaptation Notes:** All I/O operations (LLM, database, HTTP) use asyncio

4. **Database Repository Pattern:**
   - **Application:** Store execution plans and results in PostgreSQL
   - **Usage Example:**

     ```python
     from agentcore.a2a_protocol.database import get_session

     async with get_session() as session:
         repo = ModularExecutionRepository(session)
         execution = await repo.create(plan_id=plan.id, query=query)
         await repo.update_status(execution.id, "completed", result=final_result)
     ```

   - **Adaptation Notes:** Follow existing repository patterns in `database/repositories.py`

### Implementation Reference Examples

**From Research (modular-agent-architecture.md):**

```python
class ModularAgent:
    def __init__(self):
        self.planner = PlannerModule()
        self.executor = ExecutorModule()
        self.verifier = VerifierModule()
        self.generator = GeneratorModule()
        self.memory = EvolvingMemory()

    async def solve(self, query: str) -> Response:
        # Planning phase
        plan = await self.planner.create_plan(query, self.memory)

        # Execution loop
        max_iterations = 5
        for iteration in range(max_iterations):
            # Execute current step
            results = await self.executor.execute(plan.current_step)

            # Verify results
            verification = await self.verifier.verify(results, plan.criteria)

            # Update memory
            self.memory.update(results, verification)

            # Check if we're done
            if verification.is_complete and verification.is_correct:
                break

            # Refine plan if needed
            plan = await self.planner.refine(plan, verification.feedback)

        # Generate final response
        response = await self.generator.generate(self.memory, plan)
        return response
```

**Key Takeaways from Examples:**

- Modules communicate through structured interfaces (Protocol pattern)
- Iterative refinement loop is central to reliability improvement
- Memory system maintains state across iterations
- Verification step is critical for error detection and recovery
- Anti-patterns to avoid: Synchronous blocking calls, tight coupling between modules, missing error handling

### New Patterns to Create

**Patterns This Implementation Will Establish:**

1. **Module Coordination Pattern**
   - **Purpose:** Orchestrate multi-module workflows with state management
   - **Location:** `agentcore/modular/coordinator.py`
   - **Reusability:** Template for any multi-agent orchestration workflow

2. **Plan Refinement Loop Pattern**
   - **Purpose:** Iteratively improve execution plans based on feedback
   - **Location:** `agentcore/modular/refinement.py`
   - **Reusability:** Applicable to any self-correcting agent system

3. **Module Interface Protocol Pattern**
   - **Purpose:** Define pluggable module interfaces using Python Protocol
   - **Location:** `agentcore/modular/interfaces.py`
   - **Reusability:** Template for extensible component systems

---

## 🔧 Technology Stack

### Recommended Stack (from Research & System Context)

**Based on:** Existing AgentCore stack (CLAUDE.md) + Research recommendations

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | AgentCore standard; modern typing support (PEP 695) |
| Framework | FastAPI | Latest | Existing AgentCore infrastructure; async support |
| API Protocol | JSON-RPC 2.0 | A2A v0.2 | AgentCore standard for agent communication |
| Validation | Pydantic | v2.x | AgentCore standard; strict typing validation |
| Database | PostgreSQL | 14+ | AgentCore standard; async SQLAlchemy support |
| ORM | SQLAlchemy | 2.0+ | AgentCore standard; async API |
| LLM Gateway | Portkey | Latest | Multi-provider support, fallback, load balancing |
| Testing | pytest-asyncio | Latest | AgentCore standard; 90%+ coverage requirement |
| Type Checking | mypy (strict) | Latest | AgentCore standard; strict mode enabled |
| Monitoring | Prometheus | Latest | AgentCore standard metrics collection |
| Tracing | OpenTelemetry | Latest | Distributed tracing for module transitions |

**Key Technology Decisions:**

- **Async-first architecture:** All modules use asyncio for non-blocking execution
- **Protocol-based interfaces:** Python `Protocol` (typing) for module pluggability
- **JSON-RPC for coordination:** Consistent with A2A protocol; enables remote modules if needed
- **PostgreSQL for state:** Execution plans and results persisted for recovery and analysis
- **Portkey for LLM access:** Unified interface to multiple providers (OpenAI, Anthropic, etc.)

**Research Citations:**

- Modular architecture with specialized modules - Source: docs/research/modular-agent-architecture.md
- 15-20% task success improvement, 30-40% cost reduction - Source: docs/research/modular-agent-architecture.md
- A2A protocol integration for module discovery - Source: docs/research/modular-agent-architecture.md

### Alignment with Existing System

**From CLAUDE.md:**

- **Consistent With:**
  - Python 3.12+ runtime
  - FastAPI framework
  - JSON-RPC 2.0 protocol
  - Pydantic validation
  - Async-first patterns
  - PostgreSQL database
  - pytest-asyncio testing
  - mypy strict type checking

- **New Additions:**
  - Portkey AI gateway (multi-LLM support)
  - OpenTelemetry tracing (distributed module tracing)
  - Module-specific model sizing (cost optimization)
  - Plan refinement loop infrastructure

- **Migration Considerations:**
  - None - pure addition to existing stack
  - No breaking changes to existing components
  - Modules optional; single-agent path remains available

---

## 🏗️ Architecture Design

### System Context (from CLAUDE.md)

**Existing System Architecture:**

AgentCore implements Google's A2A (Agent2Agent) protocol v0.2 with JSON-RPC 2.0 infrastructure for agent communication, discovery, and task management. The system uses a service-oriented architecture with:

- **JSON-RPC Handler:** Central request/response processor with method registry
- **Services Layer:** Agent management, task management, message routing, security
- **Database Layer:** PostgreSQL with async SQLAlchemy for persistence
- **Routers Layer:** FastAPI endpoints for JSON-RPC, WebSocket, health checks

**Integration Points:**

- **JSON-RPC Infrastructure:** Register `modular.solve` method via `@register_jsonrpc_method`
- **Agent Management:** Each module registers as A2A agent with capabilities
- **Task Management:** Map plan steps to AgentCore TaskRecord for tracking
- **Event Streaming:** Emit events for module transitions (SSE/WebSocket)
- **Security Service:** JWT authentication for module access control

**New Architectural Patterns:**

- Modular agent coordination (orchestrator pattern)
- Plan refinement loop (iterative improvement)
- Module-specific model sizing (cost optimization)

### Component Architecture

**Architecture Pattern:** Modular Microservices (within monolith)

- **Rationale:** Enables independent module optimization while leveraging existing AgentCore infrastructure
- **Alignment:** Modules register as A2A agents; uses existing discovery and communication patterns

**System Design:**

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                        AgentCore System                          │
│                                                                  │
│  ┌────────────────┐                                             │
│  │  Client        │                                             │
│  │  (JSON-RPC)    │                                             │
│  └───────┬────────┘                                             │
│          │                                                       │
│          ▼                                                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │  Modular Coordinator (modular.solve)               │         │
│  └──┬───────────────────────────────────────────────┬─┘         │
│     │                                               │           │
│     ▼                                               ▼           │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐         │
│  │  Planner   │────▶│  Executor  │────▶│  Verifier  │         │
│  │  Module    │     │  Module    │     │  Module    │         │
│  └────────────┘     └──────┬─────┘     └──────┬─────┘         │
│       │                    │                   │               │
│       │  [refinement loop] │                   │               │
│       └────────────────────┴───────────────────┘               │
│                                    │                            │
│                                    ▼                            │
│                           ┌────────────────┐                    │
│                           │  Generator     │                    │
│                           │  Module        │                    │
│                           └───────┬────────┘                    │
│                                   │                             │
│          ┌────────────────────────┴────────────────────────┐   │
│          ▼                        ▼                        ▼   │
│   ┌─────────────┐      ┌──────────────┐       ┌──────────────┐│
│   │ PostgreSQL  │      │  Portkey AI  │       │  Event       ││
│   │ (Plans &    │      │  Gateway     │       │  Streaming   ││
│   │  Results)   │      │  (LLMs)      │       │  (Monitor)   ││
│   └─────────────┘      └──────────────┘       └──────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Data Flow (Modular Execution):
1. Client → POST /api/v1/jsonrpc {"method": "modular.solve", "params": {...}}
2. Coordinator → Planner: create_plan(query) → Plan object
3. Coordinator → Executor: execute(plan.current_step) → ExecutionResult
4. Coordinator → Verifier: verify(result, plan.criteria) → Verification
5. IF verification.is_complete AND verification.is_correct:
     Coordinator → Generator: generate(memory, plan) → Response
   ELSE:
     Coordinator → Planner: refine(plan, verification.feedback) → Updated Plan
     REPEAT steps 3-5 (max 5 iterations)
6. Coordinator → Client: Return Response with execution trace
```

### Architecture Decisions (from Research)

**Decision 1: Modules as A2A Agents**

- **Choice:** Each module registers as independent A2A agent
- **Rationale:** Enables discovery, monitoring, and future distributed deployment; consistent with A2A protocol
- **Implementation:** Use existing agent registration; modules advertise specific capabilities
- **Trade-offs:** Slightly more overhead vs. internal function calls; but gains monitoring, flexibility, and consistency

**Decision 2: JSON-RPC for Module Communication**

- **Choice:** Modules communicate via JSON-RPC (same as external clients)
- **Rationale:** Consistency with A2A protocol; enables remote modules if needed; simplifies testing
- **Implementation:** Standard `@register_jsonrpc_method` decorators for module methods
- **Trade-offs:** JSON serialization overhead acceptable given async I/O dominance; can optimize later with direct calls

**Decision 3: PostgreSQL for Execution State**

- **Choice:** Store execution plans, results, and refinement history in PostgreSQL
- **Rationale:** Enables recovery from crashes, post-execution analysis, and debugging; leverages existing DB infrastructure
- **Implementation:** New tables: `modular_executions`, `execution_plans`, `plan_steps`, `refinement_history`
- **Trade-offs:** Database I/O adds latency (~10-20ms per operation); acceptable given total execution time (seconds to minutes)

**Decision 4: Iterative Refinement with Max Iterations**

- **Choice:** Maximum 5 refinement iterations per execution
- **Rationale:** Prevents infinite loops while allowing multiple correction attempts
- **Implementation:** Coordinator tracks iteration count; returns partial result if limit reached
- **Trade-offs:** May not fully resolve complex errors; but 80%+ recovery rate target is achievable within 5 iterations

**Decision 5: Module-Specific Model Sizing**

- **Choice:** Use different LLM models for different modules (e.g., GPT-4 for Planner, GPT-4-mini for Verifier)
- **Rationale:** Optimize cost/quality trade-off per module; Verifier doesn't need full reasoning power
- **Implementation:** Configuration per module in `config.toml`; Portkey handles model routing
- **Trade-offs:** Adds configuration complexity; but enables 30-40% cost reduction

### Component Breakdown

**Core Components:**

1. **ModularCoordinator**
   - **Purpose:** Orchestrates execution flow through all four modules; manages iteration loop
   - **Technology:** Python 3.12+, async/await
   - **Pattern:** Orchestrator/Mediator pattern
   - **Interfaces:**
     - `async def solve(query: str, config: ModularConfig) -> ModularResult`
   - **Dependencies:** All four module interfaces, ExecutionRepository, EventManager

2. **PlannerModule**
   - **Purpose:** Analyzes queries and creates structured execution plans
   - **Technology:** Python 3.12+, Portkey (LLM access), Pydantic (validation)
   - **Pattern:** Strategy pattern (supports multiple planning strategies)
   - **Interfaces:**
     - `async def create_plan(query: str, memory: Memory) -> Plan`
     - `async def refine(plan: Plan, feedback: Feedback) -> Plan`
   - **Dependencies:** Portkey client, Plan/PlanStep models

3. **ExecutorModule**
   - **Purpose:** Executes plan steps by invoking tools and resources
   - **Technology:** Python 3.12+, Tool Integration Framework (TOOL-001)
   - **Pattern:** Command pattern (tool invocation)
   - **Interfaces:**
     - `async def execute(step: PlanStep) -> ExecutionResult`
   - **Dependencies:** Tool registry, PortkeyClient (for LLM-assisted execution)

4. **VerifierModule**
   - **Purpose:** Validates execution results against success criteria
   - **Technology:** Python 3.12+, Portkey (for LLM-based validation)
   - **Pattern:** Validator pattern
   - **Interfaces:**
     - `async def verify(result: ExecutionResult, criteria: Criteria) -> Verification`
   - **Dependencies:** Portkey client, Verification models

5. **GeneratorModule**
   - **Purpose:** Synthesizes final responses from verified results
   - **Technology:** Python 3.12+, Portkey
   - **Pattern:** Builder pattern (response construction)
   - **Interfaces:**
     - `async def generate(memory: Memory, plan: Plan) -> Response`
   - **Dependencies:** Portkey client, Response models

6. **ModularExecutionRepository**
   - **Purpose:** Persist and retrieve execution state from PostgreSQL
   - **Technology:** SQLAlchemy 2.0+ (async)
   - **Pattern:** Repository pattern (following existing AgentCore patterns)
   - **Interfaces:**
     - `async def create(plan_id: UUID, query: str) -> ModularExecutionRecord`
     - `async def update_status(execution_id: UUID, status: str, result: dict) -> bool`
     - `async def get_by_id(execution_id: UUID) -> ModularExecutionRecord | None`
   - **Dependencies:** SQLAlchemy async session

### Data Flow & Boundaries

**Request Flow:**

1. **Entry:** Client sends JSON-RPC request to `/api/v1/jsonrpc` with method `modular.solve`
2. **Validation:** JsonRpcProcessor validates request format; Pydantic validates params
3. **Authentication:** Security service validates JWT token
4. **Coordination:** ModularCoordinator.solve() invoked
5. **Planning:** Planner creates initial execution plan
6. **Iteration Loop (max 5 iterations):**
   - Executor executes current plan step
   - Verifier validates results
   - If complete and correct → break
   - If incomplete/incorrect → Planner refines plan
7. **Generation:** Generator synthesizes final response
8. **Persistence:** Execution record saved to database
9. **Events:** Module transition events emitted
10. **Response:** Return result to client with execution trace

**Component Boundaries:**

- **Public Interface:** JSON-RPC method `modular.solve`
  - Input: `{query: str, config?: {max_iterations, modules}}`
  - Output: `{answer: str, execution_trace: {...}, metrics: {...}}`
- **Internal Implementation:** Four module classes + coordinator
  - Hidden from external clients
  - Subject to change without breaking API contract
- **Cross-Component Contracts:**
  - Agent Management: Modules register as A2A agents
  - Task Management: Optional integration for plan step tracking
  - Event Streaming: Emit events for monitoring
  - Security: JWT authentication for module methods

---

## 4. Technical Specification

### Data Model

**Core Entities:**

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

# Execution Plan Models
class PlanStep(BaseModel):
    """Single step in execution plan."""
    step_id: str = Field(..., description="Unique step identifier")
    step_number: int = Field(..., ge=0, description="Step sequence number")
    description: str = Field(..., description="Step description")
    tool_requirements: list[str] = Field(default_factory=list, description="Required tools")
    dependencies: list[str] = Field(default_factory=list, description="Dependent step IDs")
    status: str = Field(default="pending", description="Step status")

class Plan(BaseModel):
    """Execution plan for multi-step task."""
    plan_id: str = Field(..., description="Unique plan identifier")
    query: str = Field(..., description="Original query")
    steps: list[PlanStep] = Field(..., description="Ordered execution steps")
    success_criteria: list[str] = Field(default_factory=list, description="Success criteria")
    max_iterations: int = Field(default=5, ge=1, le=10, description="Max refinement iterations")
    current_iteration: int = Field(default=0, description="Current iteration number")
    status: str = Field(default="pending", description="Plan status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Execution Results
class ExecutionResult(BaseModel):
    """Result from executing a plan step."""
    step_id: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    status: str  # "success", "failed", "partial"
    error: str | None = None
    execution_time_ms: int
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

# Verification
class Verification(BaseModel):
    """Verification result for execution results."""
    is_complete: bool = Field(..., description="Whether task is complete")
    is_correct: bool = Field(..., description="Whether results are correct")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in verification")
    feedback: str | None = Field(None, description="Feedback for refinement")
    issues: list[str] = Field(default_factory=list, description="Identified issues")

# Final Response
class ModularResult(BaseModel):
    """Final result from modular execution."""
    answer: str
    execution_trace: dict[str, Any]
    metrics: dict[str, Any]
    plan_id: str
    iterations: int
    status: str  # "completed", "partial", "failed"
```

**Validation Rules:**

- Plan steps: Unique step_id, valid step_number sequence (0, 1, 2, ...)
- Dependencies: All referenced step IDs must exist in plan
- max_iterations: 1-10 (prevent infinite loops, default 5)
- Confidence scores: 0.0-1.0 (Verification)

**Database Schema (PostgreSQL):**

```python
from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from agentcore.a2a_protocol.database.models import Base

class ModularExecutionRecord(Base):
    """Execution record for modular agent runs."""
    __tablename__ = "modular_executions"

    id = Column(UUID(as_uuid=True), primary_key=True)
    query = Column(String, nullable=False)
    plan_id = Column(UUID(as_uuid=True), nullable=True)
    iterations = Column(Integer, default=0)
    final_result = Column(JSON, nullable=True)
    status = Column(String, default="pending")  # pending, in_progress, completed, failed
    error = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

class ExecutionPlanRecord(Base):
    """Execution plan storage."""
    __tablename__ = "execution_plans"

    plan_id = Column(UUID(as_uuid=True), primary_key=True)
    execution_id = Column(UUID(as_uuid=True), ForeignKey("modular_executions.id"))
    query = Column(String, nullable=False)
    steps = Column(JSON, nullable=False)  # Serialized PlanStep list
    success_criteria = Column(JSON, nullable=True)
    max_iterations = Column(Integer, default=5)
    current_iteration = Column(Integer, default=0)
    status = Column(String, default="pending")
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
```

**Migration Approach:**

- Alembic migration creating new tables (no changes to existing tables)
- Migrations: `alembic revision --autogenerate -m "add modular execution tables"`
- Indexes: plan_id, execution_id, status, created_at for query performance

### API Design

**Top 6 Critical Endpoints:**

1. **`modular.solve` (JSON-RPC Method)**
   - **Method:** POST `/api/v1/jsonrpc`
   - **Purpose:** Execute query through modular agent pipeline
   - **Request Schema:**

     ```json
     {
       "jsonrpc": "2.0",
       "method": "modular.solve",
       "params": {
         "query": "What is the capital of France and its population?",
         "config": {
           "max_iterations": 5,
           "modules": {
             "planner": "planner-v1",
             "executor": "executor-v1",
             "verifier": "verifier-v1",
             "generator": "generator-v1"
           }
         }
       },
       "id": 1
     }
     ```

   - **Response Schema:**

     ```json
     {
       "jsonrpc": "2.0",
       "result": {
         "answer": "Paris is the capital of France, with approximately 2.2M population.",
         "execution_trace": {
           "plan_id": "plan-123",
           "iterations": 1,
           "steps": [
             {"module": "planner", "duration_ms": 450},
             {"module": "executor", "duration_ms": 1200},
             {"module": "verifier", "duration_ms": 300},
             {"module": "generator", "duration_ms": 400}
           ]
         },
         "metrics": {
           "total_duration_ms": 2350,
           "tokens_used": 1200,
           "cost_usd": 0.024
         }
       },
       "id": 1
     }
     ```

   - **Error Handling:**
     - `INVALID_PARAMS` (-32602): Invalid query or config parameters
     - `INTERNAL_ERROR` (-32603): Module execution failure
     - Custom error -32001: Max iterations reached without completion

2. **`modular.get_execution` (JSON-RPC Method)**
   - **Method:** POST `/api/v1/jsonrpc`
   - **Purpose:** Retrieve execution details by ID
   - **Request:** `{"method": "modular.get_execution", "params": {"execution_id": "uuid"}}`
   - **Response:** Full execution record with plan, results, trace

3. **`modular.list_executions` (JSON-RPC Method)**
   - **Method:** POST `/api/v1/jsonrpc`
   - **Purpose:** List recent executions with filtering
   - **Request:** `{"method": "modular.list_executions", "params": {"status": "completed", "limit": 10}}`
   - **Response:** Paginated list of execution summaries

4. **Module Registration (Existing `agent.register`)**
   - **Method:** POST `/api/v1/jsonrpc`
   - **Purpose:** Register modules as A2A agents
   - **Enhancement:** Add module-specific capabilities to AgentCard

5. **Module Health Checks (Existing `/health`)**
   - **Method:** GET `/health`
   - **Purpose:** Monitor module health status
   - **Enhancement:** Include module availability in health response

6. **Metrics Endpoint (Existing `/metrics`)**
   - **Method:** GET `/metrics`
   - **Purpose:** Expose Prometheus metrics
   - **Metrics:**
     - `modular_solve_requests_total{status}` - Total requests
     - `modular_solve_duration_seconds` - Request duration histogram
     - `modular_iterations_total` - Iteration count distribution
     - `modular_success_rate` - Success rate gauge
     - `modular_module_duration_seconds{module}` - Per-module latency

### Security (from Research & AgentCore Patterns)

**Based on:** AgentCore security patterns (CLAUDE.md) + Modular requirements

**Authentication/Authorization:**

- **Approach:** JWT token authentication (existing AgentCore pattern)
- **Implementation:**
  - Modular execution endpoint requires valid JWT token
  - Each module validates JWT when invoked
  - RBAC: `modular:execute` permission for clients, `modular:module` for modules
- **Standards:** JWT (RFC 7519), A2A protocol authentication patterns

**Secrets Management:**

- **Strategy:** Environment variables for LLM provider API keys (via Portkey)
- **Pattern:**
  - `PORTKEY_API_KEY` environment variable
  - Loaded via Pydantic Settings in `config.py`
  - Never logged or exposed in responses
- **Rotation:** API key rotation via environment variable update (no code changes)

**Data Protection:**

- **Encryption in Transit:** TLS/SSL for all LLM provider API calls (via Portkey)
- **Encryption at Rest:** PostgreSQL encryption for execution plans (optional, via database config)
- **PII Handling:**
  - Query content not logged by default (configurable for debugging)
  - Execution results sanitized before logging
  - Compliance: GDPR/CCPA via query content isolation

**Security Testing:**

- **Approach:**
  - Input validation testing (malformed queries, out-of-range parameters)
  - Authentication testing (missing/invalid JWT tokens)
  - Authorization testing (module access without permissions)
- **Tools:**
  - pytest security test suite
  - Manual penetration testing for module communication

**Compliance:**

- No PII stored unless explicitly in query (user responsibility to sanitize)
- Execution metadata does not contain PII
- Audit logs track all module invocations with trace IDs

### Performance (from Research)

**Based on:** Research benchmarks (15-20% success improvement, 30-40% cost reduction, <2x latency)

**Performance Targets (from Research):**

- **Response Time:** <2x baseline (acceptable given quality improvement)
  - Traditional single-agent: ~5s for multi-step task
  - Modular agent target: <10s (acceptable with +15% success rate)
- **Module Transition:** <500ms per transition (excluding tool execution)
- **Throughput:** >100 concurrent executions per instance
- **Resource Usage:**
  - Memory: O(1) per execution (constant regardless of plan size)
  - CPU: Minimal (majority of time in LLM API calls)

**Caching Strategy:**

- **Approach:** Optional plan caching for repeated queries (future enhancement)
- **Pattern:** Redis cache for similar queries → plan reuse
- **TTL Strategy:** Time-based (1 hour) for plan cache
- **Invalidation:** LRU eviction for memory-constrained environments
- **P0 Decision:** No caching in initial implementation (complexity vs value)

**LLM Provider Optimization (via Portkey):**

- **Connection Pooling:** aiohttp connection pooling for Portkey API calls
- **Load Balancing:** Portkey handles multi-provider load balancing
- **Fallback:** Portkey automatic fallback to secondary providers
- **Timeout Handling:** 60s timeout per LLM call; configurable per module
- **Retry Logic:** Portkey handles retries with exponential backoff

**Scaling Strategy:**

- **Horizontal:** Stateless design enables simple horizontal scaling
  - Kubernetes HPA based on CPU/memory or custom metrics (execution queue depth)
  - No shared state between instances (PostgreSQL handles coordination)
- **Vertical:** Minimal memory requirements per instance (~500MB base + executions)
- **Auto-scaling:**
  - Trigger: Request queue depth >20 or p95 latency >15s
  - Scale up: Add instances in increments of 2
  - Scale down: Remove instances when queue depth <5 and p95 <8s
- **Performance Monitoring:**
  - Prometheus metrics for latency, throughput, success rate
  - Grafana dashboards for real-time visualization
  - Alerting on p95 latency >15s or error rate >5%

---

## 5. Development Setup

**Required Tools:**

- Python 3.12+
- uv package manager (AgentCore standard)
- Docker & Docker Compose (for PostgreSQL + Redis)
- LLM provider API keys (OpenAI, Anthropic via Portkey)
- PostgreSQL 14+ (via Docker or local)

**Local Environment:**

```bash
# Install dependencies
uv add portkey-ai pydantic pytest-asyncio pytest-cov opentelemetry-api

# Environment variables (.env)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/agentcore
PORTKEY_API_KEY=your-portkey-api-key
MODULAR_MAX_ITERATIONS=5
MODULAR_PLANNER_MODEL=gpt-4
MODULAR_EXECUTOR_MODEL=gpt-4-mini
MODULAR_VERIFIER_MODEL=gpt-4-mini
MODULAR_GENERATOR_MODEL=gpt-4

# Run database migrations
uv run alembic upgrade head

# Run local development server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# Run tests
uv run pytest tests/unit/modular/ tests/integration/modular/ -v --cov=agentcore.modular --cov-report=term-missing --cov-fail-under=90
```

**CI/CD Pipeline Requirements:**

- **Build:** Python 3.12+ environment with uv
- **Test:** pytest with 90%+ coverage gate
- **Lint:** ruff check (AgentCore standard)
- **Type Check:** mypy --strict (AgentCore standard)
- **Security Scan:** bandit for Python security issues
- **Deploy:** Kubernetes manifest update (if merged to main)

**Testing Framework:**

- **Unit Tests:** pytest-asyncio for async module logic
  - `tests/unit/modular/test_planner.py`
  - `tests/unit/modular/test_executor.py`
  - `tests/unit/modular/test_verifier.py`
  - `tests/unit/modular/test_generator.py`
  - `tests/unit/modular/test_coordinator.py`
- **Integration Tests:** pytest-asyncio for full pipeline
  - `tests/integration/modular/test_modular_execution.py`
  - `tests/integration/modular/test_refinement_loop.py`
  - `tests/integration/modular/test_module_registration.py`
- **Performance Tests:** pytest-benchmark for latency and throughput
  - `tests/performance/modular/test_baseline_comparison.py`
  - `tests/performance/modular/test_concurrent_executions.py`
- **Coverage Target:** 90%+ (AgentCore standard)

---

## 6. Risk Management

| Risk | Impact | Likelihood | Mitigation | Contingency |
|------|--------|------------|------------|-------------|
| **LLM provider API failures** | High (execution blocked) | Medium | Portkey multi-provider fallback, retry logic | Degrade to single-agent mode |
| **Module coordination overhead** | Medium (latency increase) | High | Async I/O, optimize transitions (<500ms) | Direct function calls if needed |
| **Plan refinement not converging** | High (poor success rate) | Medium | Max iteration limit (5), quality feedback prompts | Return partial result with explanation |
| **Performance targets not met** | Medium (adoption risk) | Low | Early benchmarking in Phase 1, optimization in Phase 3 | Adjust targets or optimize bottlenecks |
| **Tool Integration Framework delay** | High (Executor blocked) | Medium | Mock tool interface for testing, parallel development | Implement basic tool wrapper in-module |
| **Database performance bottleneck** | Medium (latency impact) | Low | Async SQLAlchemy, proper indexing, query optimization | Add read replicas, caching layer |
| **Module state inconsistency** | High (reliability risk) | Low | Transaction boundaries, state validation, error recovery | PostgreSQL rollback, execution replay |
| **Cost overruns from LLM calls** | Medium (budget impact) | Medium | Module-specific model sizing, Portkey cost tracking | Fallback to smaller models, caching |

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Deliverables:**

- Module interface definitions (Protocol-based)
- Coordination infrastructure (ModularCoordinator class)
- Database schema and migrations
- Pydantic models for all data structures
- Unit tests (90%+ coverage)

**Tasks:**

1. Create `agentcore/modular/` module structure
2. Define `interfaces.py` with PlannerInterface, ExecutorInterface, VerifierInterface, GeneratorInterface
3. Implement `coordinator.py` with ModularCoordinator orchestration logic
4. Create Pydantic models in `models.py` (Plan, PlanStep, ExecutionResult, Verification, ModularResult)
5. Database models in `database/models.py` (ModularExecutionRecord, ExecutionPlanRecord)
6. Alembic migration: `alembic revision --autogenerate -m "add modular execution tables"`
7. Write unit tests for coordinator and models
8. Manual testing with mock modules

**Dependencies:** None (foundation work)

### Phase 2: Module Implementation (Week 3-4)

**Deliverables:**

- Working implementation of all four modules
- A2A agent registration for each module
- JSON-RPC method: `modular.solve`
- Portkey integration for LLM access
- Integration tests for end-to-end flow

**Tasks:**

1. Implement `PlannerModule` in `modules/planner.py`
2. Implement `ExecutorModule` in `modules/executor.py` (depends on TOOL-001 mock)
3. Implement `VerifierModule` in `modules/verifier.py`
4. Implement `GeneratorModule` in `modules/generator.py`
5. Create `services/modular_jsonrpc.py` with `@register_jsonrpc_method("modular.solve")`
6. Register modules as A2A agents in module `__init__.py`
7. Integrate Portkey client for LLM calls in all modules
8. Write integration tests for full pipeline
9. Test via curl/Postman against local server

**Dependencies:** Phase 1 complete, Portkey API key available

### Phase 3: Optimization (Week 5-6)

**Deliverables:**

- Plan refinement loop achieving >80% error recovery
- Distributed tracing integration (OpenTelemetry)
- Performance benchmarks vs. baseline
- Module-specific model sizing configuration
- Monitoring dashboards (Grafana)

**Tasks:**

1. Implement refinement loop in `coordinator.py` (iterative plan improvement)
2. Add OpenTelemetry tracing to all module transitions
3. Create `config.toml` section for module-specific LLM models
4. Run performance benchmarks (baseline comparison, concurrent executions)
5. Add Prometheus metrics collection
6. Create Grafana dashboard for modular execution monitoring
7. Optimize module transition latency (<500ms target)
8. Tune Portkey configuration (timeouts, retries, fallbacks)
9. Write performance test suite
10. Production deployment checklist

**Dependencies:** Phase 2 complete

---

## 8. Quality Assurance

**Testing Strategy:**

1. **Unit Tests (90%+ coverage):**
   - ModularCoordinator orchestration logic
   - Each module implementation (Planner, Executor, Verifier, Generator)
   - Plan refinement logic
   - Data model validation (Pydantic)
   - Database repository operations
   - Edge cases: empty plans, max iterations, module failures

2. **Integration Tests:**
   - End-to-end `modular.solve` execution
   - Module registration as A2A agents
   - Plan refinement loop with intentional failures
   - Database persistence and retrieval
   - Event streaming for module transitions
   - Concurrent execution handling

3. **Performance Tests:**
   - Baseline comparison (single-agent vs. modular)
   - Success rate improvement validation (target: +15%)
   - Latency measurement (target: <2x baseline)
   - Module transition latency (target: <500ms)
   - Concurrent execution throughput (target: >100/instance)
   - Cost per task measurement (target: -30%)

4. **Quality Tests:**
   - Task success rate on benchmark dataset
   - Tool call accuracy improvement (target: +10%)
   - Error recovery rate (target: >80%)
   - Response coherence and quality
   - Reasoning trace completeness

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
- [ ] API documentation complete (modular.solve, models)
- [ ] Database migrations applied successfully
- [ ] Module registration working (A2A agents discoverable)
- [ ] Prometheus metrics verified
- [ ] Grafana dashboards deployed
- [ ] Performance benchmarks meet targets (+15% success, <2x latency, -30% cost)
- [ ] Staging deployment successful
- [ ] Security audit passed (JWT auth, audit logging)

**Monitoring and Alerting Setup:**

- **Metrics:** requests/s, p50/p95/p99 latency, success rate, error rate, iterations, cost
- **Alerts:**
  - Error rate >5% (critical)
  - p95 latency >15s (warning)
  - Success rate <baseline+10% (warning)
  - Module unavailable (critical)
- **Dashboards:**
  - Modular execution overview: requests, latency, success rate
  - Per-module performance: latency, success rate, cost
  - Refinement analysis: iterations distribution, convergence rate
  - Cost analysis: tokens used, cost per task, savings vs baseline

---

## ⚠️ Error Handling & Edge Cases

**From:** Specification NFR-3 (Reliability) + Research findings

### Error Scenarios

**Critical Error Paths:**

1. **Module Execution Failure**
   - **Cause:** LLM API failure, timeout, invalid response
   - **Impact:** Execution cannot proceed; task fails
   - **Handling:**
     - Retry with exponential backoff (Portkey handles)
     - If retries exhausted, mark execution as failed
     - Store partial results for debugging
     - Return JSON-RPC error with execution trace
   - **Recovery:** Portkey automatic fallback to secondary provider
   - **User Experience:** "Execution failed due to module error. Partial results available for review."

2. **Plan Refinement Not Converging**
   - **Cause:** Verification repeatedly fails, max iterations reached
   - **Impact:** Task not completed, user doesn't get answer
   - **Handling:**
     - Max iteration limit enforced (5 iterations)
     - Return partial result with refinement history
     - Log convergence failure for analysis
     - Include refinement feedback in response
   - **Recovery:** User can retry with different configuration or simplified query
   - **User Experience:** "Unable to complete task within 5 iterations. Partial progress: [summary]."

3. **Database Connection Failure**
   - **Cause:** PostgreSQL unavailable, connection pool exhausted
   - **Impact:** Cannot persist execution state
   - **Handling:**
     - Retry database operations (SQLAlchemy auto-retry)
     - If persistent failure, continue execution in-memory only
     - Return result but warn about non-persistence
     - Alert monitoring system
   - **Recovery:** Automatic reconnection when database available
   - **User Experience:** "Execution completed successfully (results not persisted due to database error)."

4. **Module Registration Failure**
   - **Cause:** Agent manager unavailable, registration validation fails
   - **Impact:** Modules not discoverable via A2A protocol
   - **Handling:**
     - Retry registration on startup (exponential backoff)
     - Log registration failures
     - Continue without A2A registration (direct invocation fallback)
     - Alert monitoring system
   - **Recovery:** Periodic re-registration attempts
   - **User Experience:** Transparent (modules still functional, just not discoverable)

### Edge Cases

**Identified in Specification:**

- **Empty Query:** Validation error via Pydantic (min_length=1)
- **Query Exceeds Maximum Length:** Validation error via Pydantic (max_length=100000)
- **max_iterations = 0:** Validation error via Pydantic (ge=1)
- **Invalid Module Configuration:** Validation error if module names don't match registered agents
- **Cyclic Plan Dependencies:** Detection in Planner, return error before execution
- **Tool Not Available:** Executor detects missing tool, includes in ExecutionResult.error, triggers refinement

**Handling Strategy:**

| Edge Case | Detection | Handling | Testing Approach |
|-----------|-----------|----------|------------------|
| Empty query | Pydantic validation | Return INVALID_PARAMS error | Unit test with empty string |
| Max iterations=0 | Pydantic validation | Return INVALID_PARAMS error | Unit test with invalid config |
| Module not found | Agent discovery query | Return error with available modules list | Integration test with mock module |
| Cyclic dependencies | Plan validation | Detect cycles, return error | Unit test with circular plan |
| Tool execution timeout | Executor timeout handling | Mark step as failed, trigger refinement | Integration test with slow tool |
| LLM hallucination | Verifier confidence score | Low confidence triggers refinement | Quality test with validation dataset |

### Input Validation

**Validation Rules:**

- query: String, 1-100,000 characters
- max_iterations: Integer, 1-10 (enforced, default 5)
- module names: Must match registered A2A agent IDs
- Plan steps: step_number must be sequential (0, 1, 2, ...)
- Dependencies: Referenced step IDs must exist in plan

**Sanitization:**

- Prompt injection prevention: Escape control characters in query
- No SQL injection (all database queries parameterized via SQLAlchemy)
- Input normalization: Strip leading/trailing whitespace from query

### Graceful Degradation

**Fallback Strategies:**

1. **Portkey Provider Unavailable:**
   - **Fallback:** Automatic failover to secondary provider (Portkey handles)
   - **Degradation:** May use different model, slight quality variation

2. **Database Unavailable:**
   - **Fallback:** In-memory execution without persistence
   - **Degradation:** Execution state not saved, no recovery from crashes

3. **Module Unavailable:**
   - **Fallback:** Return error, suggest retrying or using single-agent mode
   - **Degradation:** No execution possible without all modules

### Monitoring & Alerting

**Error Tracking:**

- **Tool:** Prometheus + Grafana (existing AgentCore stack)
- **Metrics:**
  - `modular_errors_total{error_type}` - Counter of errors by type
  - `modular_module_failures_total{module}` - Module-specific failure count
  - `modular_refinement_convergence_failures_total` - Non-converging refinements
- **Threshold:**
  - Alert if error rate >5% over 5 minutes (critical)
  - Alert if module failure rate >10% over 5 minutes (critical)
  - Alert if convergence failure rate >20% over 15 minutes (warning)
- **Response:**
  - Critical: Page on-call engineer
  - Warning: Slack notification to team channel
  - Investigate: Review logs, check Portkey status, database health

**Incident Response Plan:**

1. Check Grafana dashboard for error spike
2. Review Prometheus metrics for error types
3. Check Portkey status page for LLM provider issues
4. Review application logs for stack traces
5. Check database health and connection pool
6. If Portkey issue: Wait for provider recovery or switch providers
7. If application issue: Rollback deployment or apply hotfix
8. Post-incident: Update runbook with lessons learned

---

## 📚 References & Traceability

### Source Documentation

**Research:**

- **docs/research/modular-agent-architecture.md**
  - Modular architecture with four specialized modules
  - Coordination mechanism using structured message passing
  - Performance benefits: +15-20% task success, 30-40% cost reduction
  - Implementation pattern with async/await Python
  - Integration strategy with AgentCore A2A protocol

**Specification:**

- **docs/specs/modular-agent-core/spec.md**
  - FR-1 through FR-5: Module interfaces and coordination requirements
  - NFR: Performance, scalability, reliability, security, observability
  - Features: Modular execution, registration/discovery, refinement, optimization
  - Success metrics: Task success rate, tool call accuracy, latency, cost efficiency
  - API specification: modular.solve JSON-RPC method

### System Context

**Architecture & Patterns:**

- **CLAUDE.md** - AgentCore system architecture and development patterns
  - JSON-RPC method registration pattern (`@register_jsonrpc_method`)
  - Agent registration pattern (AgentCard, capabilities)
  - Async service architecture
  - Database access patterns (async SQLAlchemy)
  - A2A context propagation (trace_id, source_agent, target_agent)
  - Testing requirements (90%+ coverage, pytest-asyncio)
  - Tech stack: Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy

**Code Examples:**

- `src/agentcore/a2a_protocol/services/agent_manager.py` - Agent registration and discovery
- `src/agentcore/a2a_protocol/models/agent.py` - AgentCard, capabilities, endpoints
- `src/agentcore/a2a_protocol/services/jsonrpc_handler.py` - JSON-RPC method registration
- `src/agentcore/a2a_protocol/database/repositories.py` - Repository pattern examples

### Research Citations

**Best Practices Research:**

- Modular agent architecture with specialized roles - Source: docs/research/modular-agent-architecture.md
- Iterative refinement loop for error recovery - Source: docs/research/modular-agent-architecture.md
- Module-specific model sizing for cost optimization - Source: docs/research/modular-agent-architecture.md

**Technology Evaluation:**

- Python 3.12+ with async/await - AgentCore standard
- FastAPI for async web framework - AgentCore standard
- Pydantic v2 for validation - AgentCore standard
- PostgreSQL with async SQLAlchemy - AgentCore standard
- Portkey for multi-LLM gateway - Recommended for provider flexibility

**Performance Benchmarks:**

- +15-20% task success improvement - Source: docs/research/modular-agent-architecture.md
- 30-40% cost reduction - Source: docs/research/modular-agent-architecture.md
- <2x latency increase acceptable - Source: docs/research/modular-agent-architecture.md

### Related Components

**Dependencies:**

- **Tool Integration Framework (TOOL-001):** Required for Executor module tool invocation
- **A2A Protocol Infrastructure:** Required for module registration and discovery
- **PostgreSQL Database:** Required for execution state persistence
- **Portkey AI Gateway:** Required for LLM provider access

**Optional Integrations:**

- **Memory Management System (MEM-001):** Can enhance context persistence across executions
- **Context Management System (CTX-001):** Can integrate with bounded reasoning strategies
- **Agent Training System (TRAIN-001):** Can train module-specific policies for optimization

---

## Implementation Checklist

**Phase 1: Foundation (Week 1-2)**

- [ ] Create `agentcore/modular/` module structure
- [ ] Define module interfaces (PlannerInterface, ExecutorInterface, VerifierInterface, GeneratorInterface)
- [ ] Implement ModularCoordinator orchestration class
- [ ] Create Pydantic models (Plan, ExecutionResult, Verification, ModularResult)
- [ ] Define database models (ModularExecutionRecord, ExecutionPlanRecord)
- [ ] Create Alembic migration for new tables
- [ ] Write unit tests (90%+ coverage)
- [ ] Manual testing with mock modules

**Phase 2: Module Implementation (Week 3-4)**

- [ ] Implement PlannerModule
- [ ] Implement ExecutorModule (with TOOL-001 integration)
- [ ] Implement VerifierModule
- [ ] Implement GeneratorModule
- [ ] Create `services/modular_jsonrpc.py` handler
- [ ] Register `modular.solve` JSON-RPC method
- [ ] Register modules as A2A agents
- [ ] Integrate Portkey client
- [ ] Write integration tests
- [ ] Test via curl/Postman

**Phase 3: Optimization (Week 5-6)**

- [ ] Implement plan refinement loop
- [ ] Add OpenTelemetry distributed tracing
- [ ] Configure module-specific LLM models
- [ ] Run performance benchmarks
- [ ] Add Prometheus metrics
- [ ] Create Grafana dashboard
- [ ] Optimize module transitions (<500ms)
- [ ] Tune Portkey configuration
- [ ] Write performance tests
- [ ] Production deployment checklist

---

**End of Implementation Blueprint (PRP)**
