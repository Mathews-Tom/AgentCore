# Orchestration Engine - Technical Breakdown

**Created:** 2025-09-27
**Updated:** 2025-10-01
**Sources:** [spec](docs/specs/orchestration-engine/spec.md) | [plan](docs/specs/orchestration-engine/plan.md) | [tasks](docs/specs/orchestration-engine/tasks.md)

---

## Quick Reference

```yaml
complexity: High
risk_level: Medium-High
team_size: 2-3 engineers (1 senior, 1-2 mid-level)
duration: 8 weeks (4 sprints)
story_points: 94 SP (includes hooks system)

dependencies:
  runtime:
    - Python 3.12+ with asyncio
    - Redis Streams 7+ (cluster mode)
    - NetworkX 3+ (graph algorithms)
    - PostgreSQL 14+ with JSONB
    - Kubernetes 1.27+ (distributed execution)
  external:
    - A2A Protocol Layer (agent coordination)
    - Agent Runtime Layer (agent execution)
    - Session Management API (A2A-019, A2A-020) - for session hooks

key_features:
  - Hybrid event-driven + graph-based orchestration
  - 6 built-in patterns (supervisor, hierarchical, handoff, swarm, network, saga)
  - Hooks system for automated workflow enhancement (NEW)
  - CQRS with event sourcing architecture
  - Circuit breaker and saga compensation

performance_targets:
  - Workflow planning: <1s for 1000+ nodes
  - Coordination overhead: <100ms
  - Event processing: 100,000+ events/second
  - Fault tolerance: 99.9% completion rate

phases:
  - Phase 1: Event Processing (Weeks 1-2, 16 SP)
  - Phase 2: Core Orchestration (Weeks 3-4, 21 SP)
  - Phase 3: Advanced Patterns (Weeks 5-6, 26 SP)
  - Phase 4: Production Features (Weeks 7-8, 31 SP)
```

---

## 1. Component Overview

### 1.1 Purpose

The Orchestration Engine provides **hybrid event-driven and graph-based workflow coordination** for complex multi-agent systems. It combines the flexibility of event-driven architectures with the predictability of graph-based workflows, enabling sophisticated agent coordination patterns at enterprise scale.

**Strategic Importance:** The hybrid orchestration approach differentiates AgentCore from pure event-driven (e.g., Temporal) or workflow-only (e.g., Airflow) competitors, enabling both deterministic and reactive coordination in a single system.

### 1.2 Key Capabilities

- **Hybrid Orchestration:** Event-driven + graph-based coordination in one system
- **Built-in Patterns:** Supervisor, hierarchical, handoff, swarm, network, custom
- **Hooks System (NEW):** Automated workflow enhancement via pre/post/session hooks
- **Fault Tolerance:** Circuit breakers, saga compensation, automatic recovery
- **High Performance:** <1s planning for 1000+ node graphs, 100k+ events/second
- **Kubernetes-Native:** Distributed agent allocation and resource management

### 1.3 Success Metrics

```yaml
Workflow Complexity:
  - Support: 1000+ node workflow graphs
  - Planning Time: <1s p95
  - Validation Time: <500ms

Coordination Performance:
  - Coordination Overhead: <100ms per agent interaction
  - Event Processing: 100,000+ events/second
  - State Query Latency: <50ms p95

Fault Tolerance:
  - Workflow Completion Rate: 99.9% (despite agent failures)
  - Circuit Breaker Response: <10ms
  - Recovery Time: <30s from failure detection

Scalability:
  - Concurrent Workflows: 1000+ simultaneous executions
  - Concurrent Agents: 10,000+ across distributed clusters
  - Event Stream Lag: <100 messages p95
```

### 1.4 Target Users

- **Workflow Architects:** Design complex multi-agent coordination patterns
- **AI Engineers:** Build sophisticated agentic systems requiring advanced coordination
- **Platform Operators:** Manage large-scale agent orchestration in production
- **Research Teams:** Experiment with novel multi-agent coordination algorithms

---

## 2. System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         External Systems                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Gateway      │  │ DSPy         │  │ Monitoring   │          │
│  │ Layer        │  │ Optimizer    │  │ Stack        │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │ Workflows        │ Patterns         │ Metrics
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Engine                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Layer (FastAPI)                   │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────┴───────────────────────────────────┐   │
│  │              Workflow Management Layer                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Graph Engine │  │ Event        │  │ Hooks Engine │   │   │
│  │  │ (NetworkX)   │  │ Processor    │  │ (NEW)        │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────┴───────────────────────────────────┐   │
│  │              Pattern Library Layer                       │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┬────────┐ │   │
│  │  │Supervisor│Hierarchic│ Handoff  │  Swarm   │ Custom │ │   │
│  │  └──────────┴──────────┴──────────┴──────────┴────────┘ │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────┴───────────────────────────────────┐   │
│  │           Fault Tolerance Layer                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │Circuit Breaker│  │Saga Pattern  │  │Retry Logic   │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────┬──────────────────┬──────────────────┬───────────────┘
            │                  │                  │
            ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Infrastructure                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Redis        │  │ PostgreSQL   │  │ Kubernetes   │          │
│  │ Streams      │  │ (JSONB)      │  │ (Resources)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────┬──────────────────┬──────────────────┬───────────────┘
            │                  │                  │
            ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Downstream Systems                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ A2A Protocol │  │ Agent        │  │ Integration  │          │
│  │ Layer        │  │ Runtime      │  │ Layer        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Points

**Upstream (Inbound):**
1. Gateway Layer → Workflow definition requests, execution control
2. DSPy Optimizer → Pattern optimization suggestions
3. CLI Layer → Session management commands (via A2A-019, A2A-020)

**Downstream (Outbound):**
1. Orchestration Engine → A2A Protocol Layer (agent coordination)
2. Orchestration Engine → Agent Runtime Layer (agent execution)
3. Orchestration Engine → Integration Layer (external tools)

**Data Stores:**
1. Redis Streams → Event-driven coordination, hooks queue
2. PostgreSQL → Workflow state, definitions, execution history, hooks config
3. Kubernetes → Agent resource allocation, pod scheduling

**Monitoring:**
1. Prometheus → Metrics collection (workflows, events, patterns, hooks)
2. Grafana → Dashboards (execution, performance, fault tolerance)
3. Jaeger/Zipkin → Distributed tracing

---

## 3. Architecture Design

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│  - FastAPI routes for workflow CRUD, pattern registration        │
│  - WebSocket for real-time workflow monitoring                   │
│  - JSON-RPC integration with A2A Protocol                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Business Logic Layer                            │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Graph Engine (NetworkX)                                    │ │
│  │  - Topological sort, cycle detection, critical path       │ │
│  │  - Dependency resolution, parallel execution planning      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Event Processor (Redis Streams)                            │ │
│  │  - Consumer groups, event ordering, dead letter queue      │ │
│  │  - Event sourcing for audit trails                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Hooks Engine (NEW)                                         │ │
│  │  - Pre/post/session hook execution                         │ │
│  │  - Event matching, priority ordering, timeout protection   │ │
│  │  - Async execution via Redis Streams queue                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Pattern Library                                            │ │
│  │  - Supervisor, hierarchical, handoff, swarm, network, saga│ │
│  │  - Custom pattern framework                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Fault Tolerance                                            │ │
│  │  - Circuit breakers, saga compensation, retry logic        │ │
│  │  - Health monitoring, graceful degradation                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                           │
│  - Repositories for workflows, executions, patterns, hooks       │
│  - CQRS: Command handlers (write) + Query processors (read)     │
│  - PostgreSQL + Redis integration                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

```
src/orchestration_engine/
├── main.py                           # FastAPI app initialization
├── api/                              # API Layer
│   ├── routes/
│   │   ├── workflow.py               # Workflow CRUD endpoints
│   │   ├── pattern.py                # Pattern registration
│   │   ├── hooks.py                  # Hooks management (NEW)
│   │   └── websocket.py              # Real-time monitoring
│   └── schemas/
│       ├── workflow.py               # Workflow request/response models
│       └── hooks.py                  # Hook configuration models (NEW)
├── services/                         # Business Logic Layer
│   ├── graph_engine.py               # NetworkX graph algorithms (800 LoC, Very High complexity)
│   ├── workflow_manager.py           # Workflow orchestration core (1200 LoC, Very High complexity)
│   ├── event_processor.py            # Redis Streams integration (500 LoC, High complexity)
│   ├── hooks_engine.py               # Hook execution engine (400 LoC, Medium complexity) (NEW)
│   ├── pattern_registry.py           # Pattern management (300 LoC, Medium complexity)
│   ├── task_allocator.py             # Agent assignment (400 LoC, Medium complexity)
│   └── compensation_handler.py       # Saga compensation (600 LoC, High complexity)
├── patterns/                         # Pattern Library
│   ├── base.py                       # Abstract pattern interface
│   ├── supervisor.py                 # Supervisor pattern (400 LoC, Medium complexity)
│   ├── hierarchical.py               # Hierarchical pattern (500 LoC, High complexity)
│   ├── handoff.py                    # Handoff pattern (250 LoC, Low complexity)
│   ├── swarm.py                      # Swarm pattern (600 LoC, High complexity)
│   ├── network.py                    # Network pattern (400 LoC, Medium complexity)
│   └── saga.py                       # Saga pattern (600 LoC, Very High complexity)
├── fault_tolerance/                  # Fault Tolerance Layer
│   ├── circuit_breaker.py            # Circuit breaker (200 LoC, Medium complexity)
│   ├── retry.py                      # Retry logic (150 LoC, Low complexity)
│   └── health_monitor.py             # Health monitoring (150 LoC, Low complexity)
├── database/                         # Data Access Layer
│   ├── models.py                     # SQLAlchemy ORM models
│   ├── repositories.py               # Repository pattern for data access
│   └── migrations/                   # Alembic migrations
│       └── versions/
│           └── 001_add_hooks_table.py  # Hooks table migration (NEW)
└── utils/
    ├── graph_utils.py                # Graph algorithm helpers
    ├── event_utils.py                # Event formatting
    └── kubernetes_utils.py           # K8s resource management

tests/
├── unit/                             # 95% coverage target
│   ├── test_graph_engine.py          # Graph algorithms
│   ├── test_patterns.py              # Pattern implementations
│   ├── test_hooks_engine.py          # Hook execution (NEW)
│   ├── test_circuit_breaker.py       # Fault tolerance
│   └── test_saga_compensation.py     # Saga patterns
├── integration/                      # Redis, PostgreSQL, K8s
│   ├── test_event_processor.py       # Redis Streams
│   ├── test_workflow_execution.py    # Full workflow
│   ├── test_hooks_integration.py     # Hooks with events (NEW)
│   └── test_k8s_allocation.py        # Kubernetes
├── e2e/                              # End-to-end scenarios
│   ├── test_workflow_lifecycle.py    # Definition → execution → completion
│   ├── test_fault_recovery.py        # Failure → recovery flows
│   └── test_hooks_workflows.py       # Workflows with hooks (NEW)
└── performance/                      # Load testing
    ├── test_graph_performance.py     # 1000+ node graphs
    ├── test_event_throughput.py      # 100k+ events/sec
    └── test_concurrent_workflows.py  # 1000+ workflows

Total Core LoC: ~6000 (includes hooks system)
Complexity: High
Risk: Medium-High
```

**Complexity Assessment:**

| Module | LoC | Complexity | Risk | Owner |
|--------|-----|------------|------|-------|
| graph_engine.py | 800 | Very High | High | Senior Dev |
| workflow_manager.py | 1200 | Very High | High | Senior Dev |
| saga.py (pattern) | 600 | Very High | Very High | Senior Dev |
| event_processor.py | 500 | High | Medium | Senior Dev |
| hierarchical.py | 500 | High | Medium | Senior Dev |
| swarm.py | 600 | High | Medium | Senior Dev |
| compensation_handler.py | 600 | High | High | Senior Dev |
| hooks_engine.py (NEW) | 400 | Medium | Medium | Mid-level Dev |
| task_allocator.py | 400 | Medium | Low | Mid-level Dev |
| pattern_registry.py | 300 | Medium | Low | Mid-level Dev |
| circuit_breaker.py | 200 | Medium | Low | Mid-level Dev |
| **Total** | **~6000** | **High** | **Medium-High** | 2-3 devs |

---

## 4. Interface Contracts

### 4.1 Workflow Management API (JSON-RPC 2.0)

**Method:** `workflow.create`

```yaml
Request:
  - name: string (required)
  - version: string (required)
  - orchestration_pattern: enum (required) - supervisor|hierarchical|handoff|swarm|network|custom
  - agents: object (required)
      - agent_role: object
          - type: string
          - capabilities: array[string]
          - resources: object (cpu, memory)
  - tasks: array[object] (required)
      - id: string
      - agent: string
      - depends_on: array[string]
      - parallel: bool (optional)
      - timeout: int (optional)
      - retry_policy: object (optional)
      - compensation_action: string (optional)
  - coordination: object (required)
      - type: string - hybrid|event_driven|graph_based
      - event_driven: array[string] (optional)
      - graph_based: array[string] (optional)
  - fault_tolerance: object (optional)
      - circuit_breaker: object
      - saga_pattern: object

Response:
  - workflow_id: UUID
  - status: "validated"
  - validation_report:
      - errors: array[string]
      - warnings: array[string]
      - optimizations: array[string]
  - estimated_resources:
      - agents_required: int
      - estimated_duration_seconds: int
      - cpu_cores: float
      - memory_gb: float

Errors:
  - -32602: Invalid workflow definition (validation failed)
  - -32000: Unsupported orchestration pattern
  - -32001: Agent capabilities not available
  - -32002: Resource constraints not satisfiable

Performance: <500ms p95
```

**Method:** `workflow.execute`

```yaml
Request:
  - workflow_id: UUID (required)
  - input_data: object (required) - workflow-specific input
  - execution_options: object (optional)
      - timeout: int (seconds, default: 3600)
      - retry_policy: string (exponential_backoff|linear|none)
      - resource_constraints:
          - max_agents: int
          - max_cpu: string
          - max_memory: string
      - checkpoint_interval: int (seconds, default: 300)

Response:
  - execution_id: UUID
  - workflow_id: UUID
  - status: "planning" | "executing" | "paused" | "completed" | "failed"
  - allocated_agents: array[object]
      - agent_id: string
      - role: string
      - status: string
  - execution_url: string (WebSocket URL for real-time updates)
  - estimated_completion: ISO8601 timestamp

Errors:
  - -32003: Workflow not found
  - -32004: Workflow already executing
  - -32005: Resource allocation failed
  - -32006: Agent allocation timeout

Performance: <1s p95 (planning phase)
```

**Method:** `workflow.status`

```yaml
Request:
  - execution_id: UUID (required)

Response:
  - workflow_id: UUID
  - execution_id: UUID
  - status: "planning" | "executing" | "paused" | "completed" | "failed"
  - current_phase: string (current task or stage)
  - agents: array[object]
      - agent_id: string
      - status: "assigned" | "running" | "completed" | "failed"
      - current_task: string
      - progress: float (0.0-1.0)
  - performance_metrics:
      - total_runtime_seconds: int
      - agents_allocated: int
      - tasks_completed: int
      - tasks_failed: int
      - coordination_overhead: float (percentage)
  - error_details: object (if status = "failed")
      - error_code: string
      - error_message: string
      - failed_task: string
      - compensation_status: string

Errors:
  - -32003: Execution not found

Performance: <50ms p95
```

**Method:** `workflow.pause`

```yaml
Request:
  - execution_id: UUID (required)

Response:
  - execution_id: UUID
  - status: "paused"
  - checkpoint_id: UUID
  - checkpoint_timestamp: ISO8601
  - resume_url: string

Errors:
  - -32003: Execution not found
  - -32007: Cannot pause in current state

Performance: <200ms p95
```

**Method:** `workflow.resume`

```yaml
Request:
  - execution_id: UUID (required)
  - checkpoint_id: UUID (optional, defaults to latest)

Response:
  - execution_id: UUID
  - status: "executing"
  - resumed_at: ISO8601
  - recovered_tasks: int

Errors:
  - -32003: Execution not found
  - -32008: Checkpoint not found
  - -32005: Resource allocation failed

Performance: <1s p95
```

### 4.2 Pattern Management API

**Method:** `pattern.register`

```yaml
Request:
  - name: string (required)
  - version: string (required)
  - description: string (optional)
  - pattern_type: string (required) - custom
  - coordination_logic: object (required)
  - configuration_schema: object (JSON schema)

Response:
  - pattern_id: UUID
  - name: string
  - status: "registered"

Errors:
  - -32009: Pattern name conflict
  - -32010: Invalid coordination logic

Performance: <100ms p95
```

**Method:** `pattern.list`

```yaml
Request:
  - filter: string (optional) - built-in|custom|all (default: all)

Response:
  - patterns: array[object]
      - pattern_id: UUID
      - name: string
      - version: string
      - type: string
      - description: string

Performance: <50ms p95
```

### 4.3 Hooks System API (NEW)

**Method:** `hooks.register`

```yaml
Request:
  - name: string (required)
  - description: string (optional)
  - trigger: enum (required) - pre_task|post_task|pre_workflow|post_workflow|session_start|session_end|agent_assigned|agent_failed
  - event_pattern: string (optional) - regex for event matching
  - action: object (required)
      - type: "command" | "function"
      - command: string (if type=command) - shell command
      - function: string (if type=function) - Python function reference
      - args: array[string] (optional)
  - priority: int (optional, default: 100) - lower numbers run first
  - enabled: bool (optional, default: true)
  - timeout_seconds: int (optional, default: 30)
  - retry_policy: object (optional)
      - max_attempts: int
      - backoff: string - exponential|linear

Response:
  - hook_id: UUID
  - name: string
  - trigger: string
  - status: "registered"
  - created_at: ISO8601

Errors:
  - -32011: Hook name conflict
  - -32012: Invalid trigger type
  - -32013: Invalid action (command validation failed)

Performance: <100ms p95
```

**Method:** `hooks.list`

```yaml
Request:
  - trigger: string (optional) - filter by trigger type
  - enabled: bool (optional) - filter by enabled status

Response:
  - hooks: array[object]
      - hook_id: UUID
      - name: string
      - trigger: string
      - priority: int
      - enabled: bool
      - timeout_seconds: int
      - created_at: ISO8601

Performance: <50ms p95
```

**Method:** `hooks.get`

```yaml
Request:
  - hook_id: UUID (required)

Response:
  - hook_id: UUID
  - name: string
  - description: string
  - trigger: string
  - event_pattern: string
  - action: object
  - priority: int
  - enabled: bool
  - timeout_seconds: int
  - retry_policy: object
  - created_at: ISO8601
  - updated_at: ISO8601
  - execution_stats:
      - total_executions: int
      - successful_executions: int
      - failed_executions: int
      - average_duration_ms: float

Errors:
  - -32014: Hook not found

Performance: <50ms p95
```

**Method:** `hooks.delete`

```yaml
Request:
  - hook_id: UUID (required)

Response:
  - hook_id: UUID
  - status: "deleted"

Errors:
  - -32014: Hook not found

Performance: <100ms p95
```

**Method:** `hooks.execute` (Manual trigger for testing)

```yaml
Request:
  - hook_id: UUID (required)
  - context: object (required) - execution context (simulated)

Response:
  - execution_id: UUID
  - hook_id: UUID
  - status: "completed" | "failed"
  - output: string
  - duration_ms: int

Errors:
  - -32014: Hook not found
  - -32015: Hook execution failed

Performance: <timeout_seconds + 100ms>
```

### 4.4 WebSocket Real-Time Monitoring

```yaml
Connection: /ws/workflows/{execution_id}
Protocol: WebSocket
Authentication: JWT token in query param or header

Message Types (Server → Client):
  1. workflow.status_changed
     - execution_id: UUID
     - previous_status: string
     - new_status: string
     - timestamp: ISO8601

  2. agent.task_assigned
     - execution_id: UUID
     - agent_id: string
     - task_id: string
     - dependencies: array[string]
     - timestamp: ISO8601

  3. task.progress_update
     - execution_id: UUID
     - task_id: string
     - agent_id: string
     - progress: float (0.0-1.0)
     - timestamp: ISO8601

  4. task.completed
     - execution_id: UUID
     - task_id: string
     - agent_id: string
     - result: object
     - duration_seconds: int
     - timestamp: ISO8601

  5. error.occurred
     - execution_id: UUID
     - error_type: string
     - error_message: string
     - task_id: string (if applicable)
     - agent_id: string (if applicable)
     - timestamp: ISO8601

  6. checkpoint.created
     - execution_id: UUID
     - checkpoint_id: UUID
     - checkpoint_timestamp: ISO8601

  7. hook.executed (NEW)
     - execution_id: UUID
     - hook_id: UUID
     - hook_name: string
     - trigger: string
     - status: "completed" | "failed"
     - duration_ms: int
     - timestamp: ISO8601

Message Format:
  {
    "type": "workflow.status_changed",
    "data": { ... },
    "timestamp": "2025-10-01T15:30:00Z"
  }
```

### 4.5 Events Published (Redis Streams)

**Stream:** `workflow_events`

**Event:** `workflow.execution_started`

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - orchestration_pattern: string
  - estimated_duration_seconds: int
  - allocated_agents: array[object]
  - timestamp: ISO8601
  - trace_id: UUID (for distributed tracing)
```

**Event:** `agent.task_assigned`

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - agent_id: string
  - task_id: string
  - dependencies: array[string]
  - estimated_duration_seconds: int
  - timestamp: ISO8601
```

**Event:** `task.completed`

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - task_id: string
  - agent_id: string
  - status: "success" | "failed"
  - result: object
  - duration_seconds: int
  - timestamp: ISO8601
```

**Event:** `workflow.checkpoint_created`

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - checkpoint_id: UUID
  - state_snapshot: object (compressed)
  - recovery_instructions: object
  - timestamp: ISO8601
```

**Event:** `circuit_breaker.opened` (Fault Tolerance)

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - service: string (e.g., "agent_runtime", "a2a_protocol")
  - failure_count: int
  - threshold: int
  - timestamp: ISO8601
```

**Event:** `saga.compensation_started` (Saga Pattern)

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - failed_task_id: string
  - compensation_tasks: array[string]
  - timestamp: ISO8601
```

**Event:** `hook.triggered` (NEW - Hooks System)

```yaml
Schema:
  - workflow_id: UUID
  - execution_id: UUID
  - hook_id: UUID
  - hook_name: string
  - trigger: string
  - event_matched: string (event that triggered hook)
  - timestamp: ISO8601
```

### 4.6 Events Consumed

**From A2A Protocol Layer:**

- `agent.registered` → Update available agents for allocation
- `agent.disconnected` → Trigger circuit breaker, reallocate tasks
- `agent.heartbeat` → Update health monitoring

**From Agent Runtime Layer:**

- `agent.task_started` → Update workflow execution state
- `agent.task_progress` → Stream progress updates to WebSocket clients
- `agent.task_failed` → Trigger compensation logic (saga pattern)

**From Session Management API (A2A-019, A2A-020):**

- `session.saved` → Trigger session hooks (NEW)
- `session.resumed` → Trigger session hooks (NEW)

---

## 5. Data Models

### 5.1 PostgreSQL Schema

**Table:** `workflow_definitions`

```sql
CREATE TABLE workflow_definitions (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    orchestration_pattern VARCHAR(50) NOT NULL,
    definition JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by UUID NOT NULL,
    UNIQUE(name, version)
);

CREATE INDEX idx_workflow_pattern ON workflow_definitions(orchestration_pattern);
CREATE INDEX idx_workflow_created_at ON workflow_definitions(created_at DESC);
CREATE INDEX idx_workflow_name ON workflow_definitions(name);
```

**Table:** `workflow_executions`

```sql
CREATE TABLE workflow_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflow_definitions(workflow_id),
    status VARCHAR(50) NOT NULL,
    current_phase VARCHAR(255),
    allocated_agents JSONB DEFAULT '[]',
    task_states JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    error_details JSONB,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    CONSTRAINT valid_status CHECK (status IN ('planning', 'executing', 'paused', 'completed', 'failed'))
);

CREATE INDEX idx_execution_workflow ON workflow_executions(workflow_id);
CREATE INDEX idx_execution_status ON workflow_executions(status);
CREATE INDEX idx_execution_started_at ON workflow_executions(started_at DESC);
```

**Table:** `workflow_checkpoints`

```sql
CREATE TABLE workflow_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(execution_id),
    state_snapshot JSONB NOT NULL,
    recovery_instructions JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_checkpoint_execution ON workflow_checkpoints(execution_id);
CREATE INDEX idx_checkpoint_created_at ON workflow_checkpoints(created_at DESC);
```

**Table:** `workflow_hooks` (NEW)

```sql
CREATE TABLE workflow_hooks (
    hook_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    trigger VARCHAR(50) NOT NULL,
    event_pattern VARCHAR(500),
    action_type VARCHAR(20) NOT NULL, -- 'command' or 'function'
    action_command TEXT,
    action_function VARCHAR(500),
    action_args JSONB DEFAULT '[]',
    priority INTEGER DEFAULT 100,
    enabled BOOLEAN DEFAULT TRUE,
    timeout_seconds INTEGER DEFAULT 30,
    retry_policy JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_trigger CHECK (trigger IN (
        'pre_task', 'post_task', 'pre_workflow', 'post_workflow',
        'session_start', 'session_end', 'agent_assigned', 'agent_failed'
    )),
    CONSTRAINT valid_action_type CHECK (action_type IN ('command', 'function'))
);

CREATE INDEX idx_hooks_trigger ON workflow_hooks(trigger) WHERE enabled = TRUE;
CREATE INDEX idx_hooks_priority ON workflow_hooks(priority);
```

**Table:** `hook_executions` (NEW)

```sql
CREATE TABLE hook_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hook_id UUID NOT NULL REFERENCES workflow_hooks(hook_id),
    workflow_execution_id UUID REFERENCES workflow_executions(execution_id),
    status VARCHAR(20) NOT NULL, -- 'completed', 'failed', 'timeout'
    output TEXT,
    error_message TEXT,
    duration_ms INTEGER,
    executed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_status CHECK (status IN ('completed', 'failed', 'timeout'))
);

CREATE INDEX idx_hook_exec_hook ON hook_executions(hook_id);
CREATE INDEX idx_hook_exec_workflow ON hook_executions(workflow_execution_id);
CREATE INDEX idx_hook_exec_executed_at ON hook_executions(executed_at DESC);

-- Automatic cleanup: Keep last 7 days only
CREATE INDEX idx_hook_exec_cleanup ON hook_executions(executed_at)
WHERE executed_at < NOW() - INTERVAL '7 days';
```

### 5.2 Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

# Workflow Definition Models

class AgentRequirement(BaseModel):
    """Agent resource and capability requirements"""
    type: str
    count: int = 1
    capabilities: List[str]
    resources: Dict[str, str] = {}  # e.g., {"cpu": "2 cores", "memory": "4GB"}

class TaskDefinition(BaseModel):
    """Individual task in workflow"""
    id: str
    agent: str  # Agent role reference
    depends_on: List[str] = []
    parallel: bool = False
    timeout_seconds: int = 300
    retry_policy: Optional[str] = "exponential_backoff"
    compensation_action: Optional[str] = None

class CoordinationConfig(BaseModel):
    """Coordination configuration"""
    type: Literal["hybrid", "event_driven", "graph_based"]
    event_driven: List[str] = []  # Event types for event-driven coordination
    graph_based: List[str] = []  # Aspects for graph-based coordination

class FaultToleranceConfig(BaseModel):
    """Fault tolerance configuration"""
    circuit_breaker: Dict[str, int] = {
        "failure_threshold": 3,
        "timeout_seconds": 30
    }
    saga_pattern: Dict[str, bool] = {
        "compensation_enabled": True,
        "rollback_on_failure": True
    }

class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    workflow_id: UUID = Field(default_factory=uuid4)
    name: str
    version: str
    orchestration_pattern: Literal["supervisor", "hierarchical", "handoff", "swarm", "network", "custom"]
    agents: Dict[str, AgentRequirement]
    tasks: List[TaskDefinition]
    coordination: CoordinationConfig
    fault_tolerance: Optional[FaultToleranceConfig] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: UUID

# Workflow Execution Models

class TaskState(str, Enum):
    """Task execution state"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowExecution(BaseModel):
    """Workflow execution state"""
    execution_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    status: WorkflowStatus
    current_phase: Optional[str] = None
    allocated_agents: Dict[str, str] = {}  # role → agent_id mapping
    task_states: Dict[str, TaskState] = {}  # task_id → state
    performance_metrics: Dict[str, float] = {}
    error_details: Optional[Dict[str, str]] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

# Hooks System Models (NEW)

class HookTrigger(str, Enum):
    """Hook trigger types"""
    PRE_TASK = "pre_task"
    POST_TASK = "post_task"
    PRE_WORKFLOW = "pre_workflow"
    POST_WORKFLOW = "post_workflow"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    AGENT_ASSIGNED = "agent_assigned"
    AGENT_FAILED = "agent_failed"

class HookActionType(str, Enum):
    """Hook action types"""
    COMMAND = "command"
    FUNCTION = "function"

class HookAction(BaseModel):
    """Hook action configuration"""
    type: HookActionType
    command: Optional[str] = None  # Shell command (if type=command)
    function: Optional[str] = None  # Python function reference (if type=function)
    args: List[str] = []

class RetryPolicy(BaseModel):
    """Retry policy for hook execution"""
    max_attempts: int = 3
    backoff: Literal["exponential", "linear"] = "exponential"
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0

class HookConfig(BaseModel):
    """Hook configuration"""
    hook_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    trigger: HookTrigger
    event_pattern: Optional[str] = None  # Regex for event matching
    action: HookAction
    priority: int = 100  # Lower numbers run first
    enabled: bool = True
    timeout_seconds: int = 30
    retry_policy: Optional[RetryPolicy] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class HookExecutionStatus(str, Enum):
    """Hook execution status"""
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class HookExecution(BaseModel):
    """Hook execution record"""
    execution_id: UUID = Field(default_factory=uuid4)
    hook_id: UUID
    workflow_execution_id: Optional[UUID] = None
    status: HookExecutionStatus
    output: str = ""
    error_message: Optional[str] = None
    duration_ms: int
    executed_at: datetime = Field(default_factory=datetime.utcnow)

# Graph Engine Models

class GraphNode(BaseModel):
    """NetworkX graph node representation"""
    task_id: str
    task_definition: TaskDefinition
    execution_state: TaskState = TaskState.PENDING
    assigned_agent_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class GraphEdge(BaseModel):
    """NetworkX graph edge representation"""
    source: str  # Source task_id
    target: str  # Target task_id
    weight: int = 1  # For optimization algorithms

# Pattern Implementation Models

class PatternConfig(BaseModel):
    """Base pattern configuration"""
    pattern_id: UUID = Field(default_factory=uuid4)
    name: str
    version: str
    pattern_type: str
    coordination_logic: Dict[str, any]
    configuration_schema: Dict[str, any]  # JSON schema
```

### 5.3 Redis Streams Schema

**Stream:** `workflow_events`

```yaml
Key Format: workflow:events:{workflow_id}
Message ID: Auto-generated by Redis (timestamp-sequence)
Consumer Groups:
  - orchestrator (main workflow engine)
  - monitor (metrics collection)
  - hooks_executor (hook system) (NEW)

Message Fields:
  - event_type: string
  - workflow_id: UUID
  - execution_id: UUID
  - timestamp: ISO8601
  - payload: JSON string (event-specific data)
  - trace_id: UUID (distributed tracing)
```

**Stream:** `hook_queue` (NEW)

```yaml
Key Format: hooks:queue
Message ID: Auto-generated by Redis
Consumer Groups:
  - hook_workers (async hook execution)

Message Fields:
  - hook_id: UUID
  - execution_context: JSON string
  - triggered_by: string (event type)
  - priority: int
  - timestamp: ISO8601
```

**Dead Letter Queue:** `workflow_events_dlq`

```yaml
Purpose: Store failed event processing messages
Retention: 7 days
Monitoring: Alert if size > 100 messages
```

---

## 6. Technology Stack

### 6.1 Core Technologies

```yaml
Runtime:
  language: Python 3.12+
  framework: FastAPI 0.100+
  async: asyncio with uvloop
  reason: Modern async support, type hints, excellent performance with uvloop

Event Streaming:
  library: Redis Streams (redis-py 5+)
  cluster: Redis 7+ in cluster mode
  reason: 99.9% availability, 100k+ events/sec, persistent streams, consumer groups
  research: Redis Streams share conceptual ideas with Kafka but simpler operations

Graph Engine:
  library: NetworkX 3+
  backend: nx-cugraph (optional GPU acceleration)
  reason: O(n+m) algorithms, rich graph operations, GPU acceleration up to 500x
  research: NetworkX perfect for DAGs and workflow graphs, GraphScope for distributed graphs

Database:
  primary: PostgreSQL 14+ with JSONB
  extensions: pgvector (not used in this component)
  reason: ACID compliance, JSONB for workflow definitions, mature replication
  research: PostgreSQL JSONB shows 2-3x performance vs traditional JSON storage

Coordination Pattern:
  architecture: CQRS with Event Sourcing
  reason: Separate read/write scaling, complete audit trail, 10x read performance
  research: CQRS recommended for event-driven workflow systems

Container Orchestration:
  platform: Kubernetes 1.27+
  reason: Native distributed agent allocation, auto-scaling, resource management
```

### 6.2 Supporting Libraries

```yaml
Validation & Serialization:
  - pydantic 2.0+ (data validation, JSON schema)
  - pydantic-settings (configuration management)

Database:
  - sqlalchemy 2.0+ (async ORM)
  - asyncpg (PostgreSQL async driver)
  - alembic (database migrations)

Testing:
  - pytest 7.4+ (test framework)
  - pytest-asyncio (async test support)
  - pytest-cov (coverage reporting)
  - testcontainers (Redis, PostgreSQL for integration tests)
  - locust (load testing)

Monitoring:
  - prometheus-client (metrics)
  - opentelemetry (distributed tracing)
  - structlog (structured logging)

Kubernetes:
  - kubernetes-client (K8s API integration)
  - kopf (Kubernetes operator framework - optional)

Fault Tolerance:
  - tenacity (retry logic)
  - aiocircuitbreaker (circuit breaker pattern)
```

### 6.3 Technology Decisions & Research

**Why Redis Streams over Kafka?**
- Simpler operations: No ZooKeeper, easier clustering
- Sufficient throughput: 100k+ events/sec meets requirements
- Persistent streams: Unlike pub/sub, data persists until explicitly deleted
- Consumer groups: Similar to Kafka's consumer groups for fault tolerance
- Research: Redis Streams suitable for 1000s messages/min with minimal infrastructure

**Why NetworkX over igraph/SNAP?**
- Pure Python: Easier debugging and integration
- Rich ecosystem: Extensive algorithms library
- GPU acceleration: nx-cugraph backend for 6-634x speedups
- Workflow-optimized: Perfect for DAGs and multilevel graphs
- Research: NetworkX + nx-cugraph achieves 10-500x speedups on large graphs

**Why CQRS with Event Sourcing?**
- Read/write separation: Optimize query performance independently
- Complete audit trail: Every workflow decision recorded as event
- Scalability: 10x performance improvement for read-heavy monitoring
- Research: Event sourcing enables replayability and state recovery

**Why Saga Pattern over 2PC (Two-Phase Commit)?**
- Long-running workflows: 2PC holds locks, saga uses compensation
- Distributed systems: Saga handles network partitions better
- Flexibility: Saga allows partial rollback and forward recovery
- Research: Saga is industry standard for microservices transactions

---

## 7. Testing Strategy

### 7.1 Unit Tests (Target: 95% coverage)

**Critical Modules:**

1. **Graph Engine (graph_engine.py)**
   ```python
   # tests/unit/test_graph_engine.py

   def test_topological_sort_with_dependencies():
       """Test topological sort produces correct task execution order"""
       # Given: Workflow with tasks A → B → C, A → D → C
       # When: Generate execution plan
       # Then: Order must be A, (B or D in parallel), C

   def test_cycle_detection_raises_error():
       """Test circular dependencies are detected"""
       # Given: Workflow with cycle A → B → C → A
       # When: Validate workflow
       # Then: Raise CycleDetectedError

   def test_critical_path_calculation():
       """Test critical path identifies longest dependency chain"""
       # Given: Complex workflow with multiple paths
       # When: Calculate critical path
       # Then: Return path with longest estimated duration

   def test_parallel_execution_planning():
       """Test parallel task identification"""
       # Given: Workflow with independent tasks
       # When: Generate execution plan
       # Then: Identify tasks that can run in parallel
   ```

2. **Pattern Implementations (patterns/*.py)**
   ```python
   # tests/unit/test_patterns.py

   def test_supervisor_pattern_master_worker_coordination():
       """Test supervisor delegates tasks to workers correctly"""
       # Given: Supervisor pattern with 3 worker agents
       # When: Execute workflow
       # Then: Tasks distributed evenly, master coordinates

   def test_hierarchical_pattern_escalation():
       """Test hierarchical pattern escalates failures upward"""
       # Given: Hierarchical workflow with 3 levels
       # When: L3 task fails
       # Then: Failure escalates to L2, then L1 for resolution

   def test_saga_pattern_compensation():
       """Test saga pattern rolls back on failure"""
       # Given: Saga workflow with T1, T2, T3 (T3 fails)
       # When: T3 fails
       # Then: Compensate T2, then T1 (reverse order)
   ```

3. **Hooks System (hooks_engine.py)** (NEW)
   ```python
   # tests/unit/test_hooks_engine.py

   def test_hook_event_matching():
       """Test hooks match events by trigger type and pattern"""
       # Given: Hook with trigger=post_task, pattern=".*analysis.*"
       # When: Event task.completed with task_id="data_analysis"
       # Then: Hook matches and executes

   def test_hook_priority_ordering():
       """Test hooks execute in priority order (ascending)"""
       # Given: 3 hooks with priorities 100, 50, 200
       # When: Trigger event matches all hooks
       # Then: Execute in order: 50, 100, 200

   def test_hook_timeout_protection():
       """Test hook execution times out after configured duration"""
       # Given: Hook with timeout=5s, action takes 10s
       # When: Execute hook
       # Then: Timeout after 5s, mark as "timeout" status

   def test_hook_retry_policy():
       """Test failed hooks retry with exponential backoff"""
       # Given: Hook with max_attempts=3, backoff=exponential
       # When: Hook fails (e.g., command not found)
       # Then: Retry 3 times with delays 1s, 2s, 4s
   ```

4. **Fault Tolerance (circuit_breaker.py, saga.py)**
   ```python
   # tests/unit/test_circuit_breaker.py

   def test_circuit_breaker_opens_after_threshold():
       """Test circuit breaker opens after failure threshold"""
       # Given: Circuit breaker with threshold=3
       # When: 3 consecutive failures occur
       # Then: Circuit opens, subsequent calls fail fast

   def test_circuit_breaker_half_open_recovery():
       """Test circuit breaker transitions to half-open for recovery"""
       # Given: Circuit in open state for timeout period
       # When: Timeout expires
       # Then: Circuit transitions to half-open, allows test request
   ```

**Tools:**
- pytest with asyncio support
- pytest-cov for coverage reporting (target: 95%)
- pytest-mock for mocking external dependencies
- hypothesis for property-based testing (graph algorithms)

### 7.2 Integration Tests

**Scenarios:**

1. **Redis Streams Event Processing**
   ```python
   # tests/integration/test_event_processor.py

   @pytest.mark.integration
   async def test_event_publishing_and_consumption(redis_container):
       """Test events are published and consumed via Redis Streams"""
       # Given: Redis Streams with consumer group "orchestrator"
       # When: Publish workflow.execution_started event
       # Then: Consumer group receives event, processes successfully

   @pytest.mark.integration
   async def test_consumer_group_coordination(redis_container):
       """Test multiple consumers coordinate via consumer groups"""
       # Given: 3 consumers in same consumer group
       # When: Publish 100 events
       # Then: Events distributed across consumers, each processed once

   @pytest.mark.integration
   async def test_dead_letter_queue_on_failure(redis_container):
       """Test failed event processing moves to DLQ"""
       # Given: Event processing that raises exception
       # When: Process event
       # Then: Event moved to dead letter queue after max retries
   ```

2. **PostgreSQL State Persistence**
   ```python
   # tests/integration/test_workflow_persistence.py

   @pytest.mark.integration
   async def test_workflow_definition_crud(postgres_container):
       """Test workflow definitions are stored and retrieved correctly"""
       # Given: WorkflowDefinition with JSONB definition
       # When: Save to PostgreSQL
       # Then: Retrieved workflow matches original (JSONB integrity)

   @pytest.mark.integration
   async def test_workflow_execution_state_updates(postgres_container):
       """Test execution state updates are persisted"""
       # Given: WorkflowExecution in "executing" status
       # When: Update task states and performance metrics
       # Then: Changes persisted, queryable with JSONB queries

   @pytest.mark.integration
   async def test_checkpoint_creation_and_recovery(postgres_container):
       """Test workflow checkpoints enable recovery"""
       # Given: Workflow execution with checkpoint
       # When: Restore from checkpoint
       # Then: Workflow resumes from checkpoint state
   ```

3. **Hooks Integration** (NEW)
   ```python
   # tests/integration/test_hooks_integration.py

   @pytest.mark.integration
   async def test_hooks_execute_on_workflow_events(redis_container, postgres_container):
       """Test hooks trigger on real workflow events"""
       # Given: Hook registered for post_task trigger
       # When: Workflow completes task
       # Then: Hook executes, result stored in hook_executions table

   @pytest.mark.integration
   async def test_hook_queue_async_execution(redis_container):
       """Test hooks execute asynchronously via Redis queue"""
       # Given: Hook with 5s execution time
       # When: Trigger hook
       # Then: Hook queued immediately, workflow continues without waiting

   @pytest.mark.integration
   async def test_session_hooks_integration(redis_container, postgres_container):
       """Test session hooks integrate with A2A-019, A2A-020"""
       # Given: Hook for session_start trigger
       # When: Session management API sends session.saved event
       # Then: Hook executes with session context
   ```

4. **Kubernetes Resource Allocation**
   ```python
   # tests/integration/test_k8s_allocation.py

   @pytest.mark.integration
   async def test_agent_pod_allocation(k8s_cluster):
       """Test agents are allocated as Kubernetes pods"""
       # Given: Workflow requiring 3 agents
       # When: Execute workflow
       # Then: 3 pods created with correct resources, labels

   @pytest.mark.integration
   async def test_resource_quota_enforcement(k8s_cluster):
       """Test workflow respects resource constraints"""
       # Given: Workflow with max_cpu=4 cores constraint
       # When: Allocate agents
       # Then: Total allocated CPU ≤ 4 cores
   ```

**Tools:**
- testcontainers-python (Redis, PostgreSQL, Kubernetes)
- pytest-asyncio
- docker-compose for multi-service integration tests

### 7.3 End-to-End Tests

**User Flows:**

1. **Complete Workflow Lifecycle**
   ```python
   # tests/e2e/test_workflow_lifecycle.py

   @pytest.mark.e2e
   async def test_supervisor_workflow_end_to_end(full_stack):
       """Test supervisor workflow from definition to completion"""
       # 1. Register 3 agents (1 supervisor, 2 workers)
       # 2. Create supervisor workflow definition
       # 3. Execute workflow with input data
       # 4. Monitor via WebSocket for real-time updates
       # 5. Verify tasks executed in correct order
       # 6. Verify workflow completed successfully
       # 7. Query performance metrics
   ```

2. **Fault Recovery Flow**
   ```python
   # tests/e2e/test_fault_recovery.py

   @pytest.mark.e2e
   async def test_agent_failure_compensation(full_stack):
       """Test saga compensation on agent failure"""
       # 1. Start saga workflow with 3 tasks
       # 2. Simulate task 3 failure (kill agent pod)
       # 3. Verify circuit breaker opens
       # 4. Verify saga compensation executes (T2, T1)
       # 5. Verify workflow marked as "failed" with compensation complete
       # 6. Verify error details include failed task, compensation status
   ```

3. **Hooks-Enabled Workflow** (NEW)
   ```python
   # tests/e2e/test_hooks_workflows.py

   @pytest.mark.e2e
   async def test_workflow_with_code_formatting_hook(full_stack):
       """Test workflow with post-edit hook for code formatting"""
       # 1. Register hook: post_task, command="ruff format {file}"
       # 2. Create workflow that generates code file
       # 3. Execute workflow
       # 4. Verify task completes
       # 5. Verify hook executes after task completion
       # 6. Verify code file is formatted (check file modification time)

   @pytest.mark.e2e
   async def test_session_management_with_hooks(full_stack):
       """Test session save/resume with session hooks"""
       # 1. Register hook: session_end, command="generate_summary.py"
       # 2. Start workflow, execute some tasks
       # 3. Pause workflow, trigger session save (A2A-019)
       # 4. Verify session_end hook executes
       # 5. Resume workflow (A2A-020)
       # 6. Verify session_start hook executes
       # 7. Verify workflow continues from checkpoint
   ```

**Tools:**
- pytest-integration
- docker-compose (full AgentCore stack)
- chaos-monkey for fault injection

### 7.4 Performance Tests

**Load Scenarios:**

```python
# tests/performance/test_graph_performance.py

def test_large_graph_planning_performance():
    """Benchmark workflow planning for 1000+ node graphs"""
    # Target: <1s p95
    # Given: Workflow definition with 1000 tasks, complex dependencies
    # When: Call workflow.create
    # Then: Validation + planning completes in <1s p95

def test_graph_critical_path_calculation():
    """Benchmark critical path algorithm performance"""
    # Target: <500ms for 1000 nodes
    # Given: Graph with 1000 nodes
    # When: Calculate critical path
    # Then: Completes in <500ms

# tests/performance/test_event_throughput.py

def test_event_processing_throughput(locust_runner):
    """Load test event processing with 100k+ events/second"""
    # Target: 100,000+ events/second
    # Given: Redis Streams with multiple consumers
    # When: Publish 100k events/second for 60 seconds
    # Then: All events processed, consumer lag <100 messages

# tests/performance/test_concurrent_workflows.py

def test_concurrent_workflow_execution(locust_runner):
    """Load test with 1000+ concurrent workflows"""
    # Target: 1000+ concurrent workflows without degradation
    # Given: 1000 workflow definitions
    # When: Execute all workflows simultaneously
    # Then: All workflows complete, coordination latency <100ms p95

# tests/performance/test_hooks_performance.py (NEW)

def test_hook_execution_overhead():
    """Benchmark hook execution overhead"""
    # Target: <100ms p95 (not blocking critical path)
    # Given: Hook with 50ms execution time
    # When: Trigger hook
    # Then: Hook queued in <10ms, workflow continues immediately
```

**SLA Targets:**

| Metric | Target | Test Method |
|--------|--------|-------------|
| Workflow planning | <1s p95 for 1000+ nodes | Locust + custom scenarios |
| Coordination latency | <100ms p95 | Distributed tracing analysis |
| Event processing | 100,000+ events/sec | Redis Streams benchmarks |
| State query | <50ms p95 | Database query profiling |
| Hook execution overhead | <100ms p95 | Hook execution benchmarks |

**Tools:**
- k6 or Locust for load testing
- Redis benchmarking tools
- NetworkX performance profiling
- Prometheus query latency metrics

### 7.5 Security Tests

**Attack Scenarios:**

```python
# tests/security/test_workflow_security.py

def test_workflow_definition_injection():
    """Test SQL injection via workflow definition JSONB"""
    # Attempt: Malicious JSONB with SQL injection
    # Expected: Pydantic validation rejects, no DB query executed

def test_hook_command_injection():
    """Test command injection via hook actions"""
    # Attempt: Hook command with shell injection (e.g., "; rm -rf /")
    # Expected: Command sanitization rejects, security error logged

def test_workflow_isolation_multi_tenant():
    """Test workflow isolation between tenants"""
    # Given: Workflows from tenant A and tenant B
    # When: Tenant A queries workflows
    # Then: Only tenant A workflows returned (no data leakage)

def test_unauthorized_workflow_execution():
    """Test RBAC prevents unauthorized workflow execution"""
    # Given: User without workflow:execute permission
    # When: Attempt to execute workflow
    # Then: 403 Forbidden error
```

**Tools:**
- OWASP ZAP for API security scanning
- Bandit for Python security linting
- Trivy for container scanning
- Snyk for dependency vulnerability scanning

### 7.6 Chaos Engineering

**Failure Scenarios:**

```python
# tests/chaos/test_fault_tolerance.py

@pytest.mark.chaos
async def test_redis_node_failure():
    """Test workflow continues despite Redis node failure"""
    # Given: Redis cluster with 3 nodes
    # When: Kill 1 Redis node during workflow execution
    # Then: Workflow continues, event processing recovers

@pytest.mark.chaos
async def test_agent_pod_eviction():
    """Test task recovery on agent pod eviction"""
    # Given: Workflow with running task
    # When: Kubernetes evicts agent pod
    # Then: Circuit breaker opens, task reallocated to new agent

@pytest.mark.chaos
async def test_network_partition():
    """Test workflow behavior under network partition"""
    # Given: Multi-region workflow execution
    # When: Simulate network partition (latency injection)
    # Then: Workflow completes, compensation logic handles timeouts
```

**Tools:**
- Chaos Mesh (Kubernetes-native chaos engineering)
- Litmus (chaos experiments for K8s)
- toxiproxy (network latency injection)

---

## 8. Operational Concerns

### 8.1 Infrastructure Requirements

```yaml
Compute (per instance):
  cpu: 6 vCPU (graph computation overhead)
  memory: 12GB RAM (NetworkX graph in-memory processing)
  storage: 50GB (logs, temporary workflow data)

Auto-Scaling:
  min_instances: 2
  max_instances: 10
  scale_up_trigger: workflow_queue_depth > 100
  scale_down_trigger: workflow_queue_depth < 20 for 5 minutes
  scale_up_cooldown: 60 seconds
  scale_down_cooldown: 300 seconds

Health Checks:
  liveness: GET /health (Redis, PostgreSQL connectivity)
  readiness: GET /ready (can accept workflows, queue not full)
  startup: GET /startup (initial dependency checks)
  capacity: GET /api/v1/capacity (current workflow capacity)

Redis Cluster:
  topology: 5 masters, 5 replicas (high availability)
  memory: 32GB per node
  persistence: AOF + RDB (append-only file + snapshots)
  backup: Automated daily snapshots to S3
  monitoring: Redis Exporter for Prometheus

PostgreSQL:
  storage: 500GB (workflow definitions + execution history)
  backup: Automated daily backups to S3, 30-day retention
  replication: 1 primary, 2 read replicas
  connection_pooling: PgBouncer (max 100 connections)
  monitoring: postgres_exporter for Prometheus

Kubernetes:
  namespace: agentcore-orchestration
  resource_quotas:
    - cpu: 100 cores (agent allocation)
    - memory: 200Gi
    - pods: 500
  pod_security: Restricted mode
  network_policies: Deny all by default, allow specific
  service_mesh: Istio (mTLS, observability)

Networking:
  load_balancer: Nginx Ingress Controller
  routing: Workflow-aware routing (session affinity by execution_id)
  tls: TLS 1.3 for all external communication
  service_mesh: mTLS for inter-service communication
  rate_limiting: 1000 requests/minute per IP
```

### 8.2 Monitoring & Observability

**Metrics (Prometheus):**

```yaml
Workflow Metrics:
  - workflow_executions_total (counter)
      labels: pattern, status
  - workflow_execution_duration_seconds (histogram)
      labels: pattern
      buckets: [1, 5, 10, 30, 60, 300, 600, 1800, 3600]
  - workflow_active_count (gauge)
      labels: pattern
  - workflow_task_count (histogram)
      labels: pattern
      buckets: [1, 5, 10, 50, 100, 500, 1000]
  - workflow_completion_rate (gauge)
      description: Percentage of successful completions

Graph Engine Metrics:
  - graph_planning_duration_seconds (histogram)
      buckets: [0.1, 0.5, 1, 2, 5, 10]
  - graph_nodes_count (histogram)
      buckets: [10, 50, 100, 500, 1000, 5000]
  - graph_critical_path_duration_seconds (gauge)
      description: Predicted workflow duration
  - graph_validation_errors_total (counter)
      labels: error_type

Event Processing Metrics:
  - redis_events_published_total (counter)
      labels: event_type
  - redis_events_consumed_total (counter)
      labels: consumer_group
  - redis_event_processing_duration_seconds (histogram)
  - redis_consumer_lag (gauge)
      labels: consumer_group
      alert: >100 messages
  - redis_dead_letter_queue_size (gauge)
      alert: >100 messages

Pattern Metrics:
  - pattern_executions_total (counter)
      labels: pattern_type
  - pattern_coordination_overhead_seconds (histogram)
      labels: pattern_type
  - pattern_agent_utilization (gauge)
      labels: pattern_type
      description: Percentage of allocated agents actively working

Hooks System Metrics (NEW):
  - hooks_registered_total (gauge)
      labels: trigger_type
  - hooks_executed_total (counter)
      labels: hook_name, trigger, status
  - hooks_execution_duration_seconds (histogram)
      labels: hook_name
      buckets: [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30]
  - hooks_errors_total (counter)
      labels: hook_name, error_type
  - hooks_queue_depth (gauge)
      description: Pending hook executions
      alert: >50 messages
  - hooks_timeout_total (counter)
      labels: hook_name

Fault Tolerance Metrics:
  - circuit_breaker_state (gauge)
      labels: service, state
      values: 0 (closed), 1 (open), 0.5 (half-open)
  - circuit_breaker_failures_total (counter)
      labels: service
  - retry_attempts_total (counter)
      labels: operation
  - compensation_executions_total (counter)
      labels: workflow_id, reason
  - agent_failures_total (counter)
      labels: agent_type

Infrastructure Metrics:
  - cpu_usage_percent (gauge)
  - memory_usage_bytes (gauge)
  - redis_cluster_health (gauge)
  - postgresql_connections_active (gauge)
  - kubernetes_pod_count (gauge)
```

**Alerting Rules (Prometheus Alertmanager):**

```yaml
Critical Alerts (PagerDuty):
  - WorkflowFailureRateHigh:
      expr: rate(workflow_executions_total{status="failed"}[5m]) > 0.05
      for: 5m
      severity: critical
      summary: Workflow failure rate >5% for 5 minutes

  - GraphPlanningTimeout:
      expr: histogram_quantile(0.95, graph_planning_duration_seconds) > 5
      for: 3m
      severity: critical
      summary: Graph planning p95 latency >5s

  - RedisConsumerLagHigh:
      expr: redis_consumer_lag > 10000
      for: 5m
      severity: critical
      summary: Redis consumer lag >10k events for 5 minutes

  - CircuitBreakerOpen:
      expr: circuit_breaker_state{state="open"} == 1
      for: 2m
      severity: critical
      summary: Circuit breaker open for critical service

  - PostgreSQLConnectionPoolExhausted:
      expr: postgresql_connections_active / postgresql_connections_max > 0.9
      for: 2m
      severity: critical
      summary: PostgreSQL connection pool >90% utilization

Warning Alerts (Slack):
  - CoordinationLatencyHigh:
      expr: histogram_quantile(0.95, pattern_coordination_overhead_seconds) > 0.2
      for: 10m
      severity: warning
      summary: Coordination overhead p95 >200ms

  - HooksExecutionTimeoutHigh:
      expr: rate(hooks_timeout_total[5m]) / rate(hooks_executed_total[5m]) > 0.1
      for: 10m
      severity: warning
      summary: Hooks timeout rate >10%

  - DeadLetterQueueGrowing:
      expr: redis_dead_letter_queue_size > 100
      for: 5m
      severity: warning
      summary: DLQ size >100 messages

  - WorkflowQueueDepthHigh:
      expr: workflow_queue_depth > 500
      for: 10m
      severity: warning
      summary: Workflow queue depth >500 (scaling needed)
```

**Dashboards (Grafana):**

1. **Workflow Overview Dashboard**
   - Active workflows by pattern (time series)
   - Workflow completion rate (gauge)
   - Average workflow duration by pattern (time series)
   - Task execution heatmap (tasks vs. time)
   - Agent utilization by pattern (gauge)

2. **Pattern Performance Dashboard**
   - Pattern usage distribution (pie chart)
   - Coordination overhead by pattern (histogram)
   - Pattern efficiency comparison (bar chart)
   - Pattern-specific metrics (custom per pattern)

3. **Event Streaming Dashboard**
   - Event publish/consume rates (time series)
   - Consumer lag by consumer group (time series)
   - Event processing duration histogram
   - Dead letter queue size (time series)

4. **Hooks System Dashboard** (NEW)
   - Registered hooks by trigger type (pie chart)
   - Hook execution success/failure rate (time series)
   - Hook execution duration distribution (histogram)
   - Hook queue depth (time series)
   - Top failing hooks (table)

5. **Fault Tolerance Dashboard**
   - Circuit breaker states (time series with state colors)
   - Retry rates by operation (time series)
   - Compensation execution rate (time series)
   - Agent failure rate (time series)

6. **Infrastructure Health Dashboard**
   - CPU/memory usage (time series)
   - Redis cluster health (multi-stat panel)
   - PostgreSQL performance (query latency, connection pool)
   - Kubernetes pod status (stat panel + table)
   - Network I/O (time series)

**Logging (Structured with Correlation IDs):**

```python
import structlog

logger = structlog.get_logger()

# Workflow lifecycle logging
logger.info(
    "workflow.execution_started",
    workflow_id=str(workflow_id),
    execution_id=str(execution_id),
    pattern=pattern_type,
    trace_id=str(trace_id),  # Distributed tracing
    agent_count=len(allocated_agents)
)

# Hook execution logging (NEW)
logger.info(
    "hook.executed",
    hook_id=str(hook_id),
    hook_name=hook_name,
    trigger=trigger_type,
    workflow_id=str(workflow_id),
    status=execution_status,
    duration_ms=duration,
    trace_id=str(trace_id)
)

# Error logging with context
logger.error(
    "saga.compensation_failed",
    workflow_id=str(workflow_id),
    execution_id=str(execution_id),
    failed_task=task_id,
    compensation_task=comp_task_id,
    error=str(exception),
    trace_id=str(trace_id),
    exc_info=True  # Include stack trace
)
```

**Log Retention:**
- Hot storage: 30 days (Elasticsearch/Loki)
- Cold storage: 1 year (S3)
- Audit logs: 7 years (compliance requirement)

**Distributed Tracing (Jaeger/Zipkin):**
- Trace workflow execution end-to-end
- Trace agent coordination across services
- Trace hook execution (NEW)
- Trace event flow through Redis Streams
- Trace saga compensation flows

### 8.3 Security

**Authentication & Authorization:**

```yaml
API Authentication:
  - JWT tokens via A2A Protocol Layer
  - Token validation on every request
  - Token refresh every 1 hour
  - Secure token storage (never log tokens)

RBAC Permissions:
  - workflow:create - Create workflow definitions
  - workflow:execute - Start workflow execution
  - workflow:read - View workflow status
  - workflow:admin - Pause, cancel, delete workflows
  - pattern:register - Register custom patterns
  - pattern:read - View available patterns
  - hooks:manage - Register, update, delete hooks (NEW)
  - hooks:read - View registered hooks (NEW)
  - hooks:execute - Manually trigger hooks (NEW)

Multi-Tenancy:
  - Workflow isolation by tenant_id
  - Row-level security (RLS) in PostgreSQL
  - Namespace isolation in Kubernetes
  - Redis key prefixing by tenant
```

**Data Protection:**

```yaml
Encryption at Rest:
  - PostgreSQL: Transparent Data Encryption (TDE)
  - Redis: RDB/AOF file encryption
  - Kubernetes Secrets: Encrypted etcd
  - Logs: Encrypted S3 buckets

Encryption in Transit:
  - TLS 1.3 for all API communication
  - mTLS for inter-service communication (service mesh)
  - TLS for Redis cluster communication
  - TLS for PostgreSQL connections

Secrets Management:
  - Kubernetes Secrets for sensitive configuration
  - Hook actions reference secrets by ID, not plaintext (NEW)
  - Secret rotation every 90 days
  - No secrets in logs or metrics
```

**Input Validation & Sanitization:**

```python
# Workflow definition validation
class WorkflowDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9_-]+$")
    orchestration_pattern: Literal["supervisor", "hierarchical", "handoff", "swarm", "network", "custom"]
    # ... Pydantic enforces types and constraints

# Hook command sanitization (NEW)
def sanitize_hook_command(command: str) -> str:
    """Sanitize shell command to prevent injection attacks"""
    # Whitelist allowed commands
    allowed_commands = ["ruff", "pytest", "git", "curl"]
    command_parts = shlex.split(command)

    if command_parts[0] not in allowed_commands:
        raise ValidationError(f"Command '{command_parts[0]}' not allowed")

    # Escape special characters
    return shlex.quote(command)

# Graph validation
def validate_workflow_graph(workflow: WorkflowDefinition):
    """Validate workflow graph for security and correctness"""
    # Check for cycles
    if has_cycle(graph):
        raise ValidationError("Workflow contains circular dependencies")

    # Resource limits
    if len(workflow.tasks) > 10000:
        raise ValidationError("Workflow exceeds maximum task limit (10000)")

    # Agent capability validation
    for task in workflow.tasks:
        if task.agent not in workflow.agents:
            raise ValidationError(f"Task references undefined agent: {task.agent}")
```

**Vulnerability Management:**

```yaml
Dependency Scanning:
  - Snyk: Daily scans for Python dependencies
  - Trivy: Container image scanning on every build
  - Dependabot: Automated dependency updates

Security Audits:
  - Quarterly external penetration testing
  - Annual security architecture review
  - Continuous compliance monitoring (SOC 2, GDPR)

Security Headers:
  - Strict-Transport-Security: max-age=31536000; includeSubDomains
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - Content-Security-Policy: default-src 'self'
```

**Audit Logging:**

```python
# All workflow operations audited
audit_logger.info(
    "workflow.executed",
    user_id=user_id,
    workflow_id=workflow_id,
    execution_id=execution_id,
    ip_address=request.client.host,
    timestamp=datetime.utcnow(),
    action="execute"
)

# Hook operations audited (NEW)
audit_logger.info(
    "hook.registered",
    user_id=user_id,
    hook_id=hook_id,
    hook_name=hook_name,
    trigger=trigger_type,
    action_command=sanitized_command,  # Sanitized for logging
    timestamp=datetime.utcnow(),
    action="register"
)
```

### 8.4 Scaling Strategy

**Horizontal Scaling:**

```yaml
Orchestration Engine:
  - Stateless design (workflow state in PostgreSQL)
  - Load balancer distributes workflows across instances
  - Session affinity by execution_id for WebSocket connections
  - Auto-scaling based on workflow queue depth

Event Processing:
  - Redis Streams partitioning (multiple streams by workflow_id hash)
  - Consumer groups enable parallel processing
  - Add consumers dynamically based on lag

Hooks System (NEW):
  - Hook execution queue in Redis Streams
  - Multiple hook worker consumers for parallel execution
  - Priority-based processing (high-priority hooks first)

Kubernetes Agent Allocation:
  - Agents scheduled across K8s cluster
  - Node affinity for agent co-location (reduce network latency)
  - Resource quotas prevent single workflow monopolizing cluster
```

**Vertical Scaling:**

```yaml
Graph Algorithm Optimization:
  - NetworkX + nx-cugraph GPU backend (6-634x speedups)
  - Graph caching for repeated workflows
  - Incremental graph updates (avoid full recomputation)

Database Query Optimization:
  - JSONB indexing for workflow definition queries
  - Materialized views for analytics queries
  - Connection pooling (PgBouncer)
  - Read replicas for status queries

Event Stream Optimization:
  - Event batching (process multiple events together)
  - Event compression (reduce network I/O)
  - Stream compaction (remove old events)
```

**Geographic Distribution:**

```yaml
Multi-Region Deployment:
  - Orchestration Engine deployed in 3 regions (US-East, US-West, EU-West)
  - PostgreSQL primary in US-East, read replicas in all regions
  - Redis cluster per region (geo-sharding)
  - Agent allocation prefers local region (reduce latency)

Cross-Region Workflows:
  - Workflow execution spans regions if required
  - Inter-region communication via AWS PrivateLink
  - Increased latency tolerance (200ms coordination overhead)
```

---

## 9. Risk Analysis

### 9.1 Technical Risks

| Risk ID | Risk | Impact | Likelihood | Mitigation | Contingency |
|---------|------|--------|------------|------------|-------------|
| ORCH-R1 | Graph algorithm complexity causes planning timeouts (>5s for 1000+ nodes) | High | Medium | NetworkX optimized algorithms (O(n+m)), caching repeated plans, GPU acceleration (nx-cugraph) | Simplify graphs, partition into sub-workflows, increase timeout |
| ORCH-R2 | Redis Streams consumer group coordination failures (lag >10k events) | High | Medium | Proper consumer group config, monitoring lag, auto-scaling consumers | Manual consumer restart, backpressure (pause workflows) |
| ORCH-R3 | Saga compensation logic bugs cause inconsistent state | High | Medium | Extensive testing, idempotent operations, state validation after compensation | Manual rollback procedures, human intervention |
| ORCH-R4 | NetworkX performance degradation with very large graphs (>5000 nodes) | Medium | Medium | Graph size limits (10k nodes), optimization, consider GraphScope for distributed graphs | Reject oversized workflows, suggest decomposition |
| ORCH-R5 | Kubernetes resource allocation failures under heavy load | Medium | High | Resource quotas, graceful degradation, alternative scheduling (spot instances) | Queue workflows, scale cluster, reject new workflows |
| ORCH-R6 | PostgreSQL JSONB queries slow with large workflow definitions (>10MB) | Medium | Low | JSONB indexing, compression, separate blob storage for large definitions | Reference large definitions by URL (S3), lazy loading |
| ORCH-R7 | Event ordering issues cause inconsistent workflow state | Medium | High | Redis Streams guarantees ordering within stream, consumer acknowledgment | Event replay with timestamp ordering, eventual consistency |
| ORCH-R8 | Hooks system introduces security vulnerabilities (command injection) | High | Medium | Command sanitization, whitelist allowed commands, timeout protection | Disable hooks system, manual hook execution |
| ORCH-R9 | Workflow deadlocks in complex dependency graphs | High | Medium | Deadlock detection algorithms, timeout enforcement, dependency validation | Manual workflow cancellation, kill switch |
| ORCH-R10 | Memory leaks in long-running workflows (NetworkX graphs accumulate) | Medium | Medium | Periodic garbage collection, workflow archival, memory monitoring | Auto-restart on memory threshold, workflow checkpointing |

### 9.2 Dependency Risks

| Dependency | Risk | Mitigation | Contingency |
|------------|------|------------|-------------|
| Redis Streams | Message loss or ordering issues | Persistent streams, consumer acknowledgments, monitoring | Fallback to database-backed queue, accept degraded performance |
| NetworkX | Performance limitations or bugs with large graphs | Version pinning, performance testing, GPU acceleration | Alternative: igraph (C-based), SNAP (faster but less Pythonic) |
| Kubernetes API | API rate limits or availability issues | Request batching, retry logic, monitoring | Degraded mode: Manual agent allocation, local execution |
| A2A Protocol Layer | API unavailability blocks agent coordination | Circuit breaker, retry logic, health monitoring | Queue workflow execution, fail gracefully |
| Session Management API (A2A-019, A2A-020) | Session hooks depend on unbuilt API | Mock session API for development, integration tests | Deploy hooks without session hooks initially, defer to Phase 5 |

### 9.3 Operational Risks

| Risk | Impact | Likelihood | Mitigation | Contingency |
|------|--------|------------|------------|-------------|
| Redis cluster split-brain scenario | High | Low | Quorum-based configuration, sentinel monitoring | Failover to backup cluster, data reconciliation |
| PostgreSQL replication lag affects read replicas | Medium | Medium | Monitoring replication lag, read from primary if lag >5s | Scale primary for reads, accept stale data |
| Kubernetes pod evictions disrupt workflows | Medium | High | Pod disruption budgets, anti-affinity rules | Checkpoint workflows, resume after pod recreation |
| Hooks execution causes cascading failures | High | Low | Circuit breaker for hooks, timeout protection, async execution | Disable failing hooks, alert on failures |
| Workflow queue saturation (>1000 pending) | High | Medium | Auto-scaling, workflow prioritization, backpressure | Reject low-priority workflows, scale cluster |

### 9.4 Business Risks

| Risk | Impact | Likelihood | Mitigation | Contingency |
|------|--------|------------|------------|-------------|
| Orchestration patterns insufficient for complex use cases | High | Low | Extensive pattern library (6 built-in), custom pattern framework | Manual orchestration, external workflow engines |
| Performance targets not achieved at scale (1000+ workflows) | Medium | Medium | Early performance testing, optimization, horizontal scaling | Relaxed performance requirements, user education |
| Hooks system adoption low due to complexity | Low | Medium | Comprehensive documentation, example hooks, CLI integration | Simplify hooks interface, provide UI for hook management |

---

## 10. Development Workflow

### 10.1 Local Setup

```bash
# Clone repository
git clone https://github.com/your-org/agentcore.git
cd agentcore

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Start infrastructure dependencies
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
./scripts/wait-for-services.sh

# Setup Kubernetes (minikube for local development)
minikube start --cpus=4 --memory=8192
kubectl apply -f k8s/dev/

# Run database migrations
uv run alembic upgrade head

# Seed database with example workflows and hooks
uv run python scripts/seed_database.py

# Start orchestration engine
uv run uvicorn src.orchestration_engine.main:app --reload --port 8002

# Verify setup
curl http://localhost:8002/health
curl http://localhost:8002/api/v1/patterns
curl http://localhost:8002/api/v1/hooks  # NEW

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/orchestration_engine --cov-report=html
```

### 10.2 Code Quality

```yaml
Linting:
  tool: ruff
  config: pyproject.toml
  rules:
    - E501: Line too long (relaxed to 120 chars)
    - F401: Unused imports
    - async patterns: Enforce async best practices
  command: uv run ruff check src/ tests/

Formatting:
  tool: ruff format
  style: Black-compatible
  command: uv run ruff format src/ tests/

Type Checking:
  tool: mypy
  mode: strict
  plugins: pydantic, sqlalchemy
  command: uv run mypy src/

Pre-commit Hooks:
  - ruff check (linting)
  - ruff format (formatting)
  - mypy (type checking)
  - pytest (unit tests)
  - workflow definition validation
  - hook command sanitization check (NEW)

Documentation:
  - Docstrings: Google style
  - Pattern documentation: Markdown in patterns/README.md
  - Hook examples: docs/hooks-examples.md (NEW)
  - API documentation: Auto-generated from FastAPI + Pydantic
```

### 10.3 Deployment Pipeline

```yaml
Stages:
  1. Build:
      - uv build package
      - Docker build with multi-stage (builder + runtime)
      - Include Kubernetes tools (kubectl, helm)
      - Build hook worker container (NEW)

  2. Test:
      - Unit tests: 95% coverage requirement
      - Integration tests: Redis, PostgreSQL, Kubernetes
      - Hooks integration tests (NEW)
      - Type checking: mypy strict mode

  3. Graph Testing:
      - Large workflow performance validation (1000+ nodes <1s)
      - Graph algorithm correctness tests
      - Critical path accuracy verification

  4. Kubernetes Testing:
      - Agent allocation and resource management
      - Pod scheduling and affinity rules
      - Resource quota enforcement

  5. Security Scanning:
      - Snyk: Dependency vulnerabilities
      - Trivy: Container scanning
      - Bandit: Python security linting
      - Hook command validation (NEW)

  6. Deploy Staging:
      - Helm deploy to staging cluster
      - Run smoke tests (basic workflow execution)
      - Run hooks integration tests with real A2A Protocol (NEW)
      - Performance testing with Locust

  7. Pattern Validation:
      - Test all 6 built-in patterns with real agents
      - Test custom pattern registration and execution
      - Test hooks with all trigger types (NEW)

  8. Deploy Production:
      - Manual approval required
      - Blue-green deployment strategy
      - Gradual rollout (10% → 50% → 100%)
      - Automated rollback on error rate >1%
      - Monitor metrics for 1 hour before final approval

CI/CD Configuration:
  tool: GitHub Actions
  triggers:
    - push to main (automated staging deployment)
    - pull request (run tests only)
    - tag creation (production deployment)
  secrets:
    - KUBECONFIG (Kubernetes access)
    - DOCKER_HUB_TOKEN (container registry)
    - SNYK_TOKEN (security scanning)
```

### 10.4 Branching Strategy

```yaml
Main Branch:
  - Protected branch
  - Requires PR approval from 1+ reviewers
  - All tests must pass
  - Code coverage must not decrease

Feature Branches:
  - naming: feature/<task-id>-<description>
  - example: feature/ORCH-011-hooks-system
  - merge strategy: Squash and merge to main

Release Branches:
  - naming: release/<version>
  - example: release/1.0.0
  - created from main for production releases
  - hotfixes can be applied to release branches

Hotfix Branches:
  - naming: hotfix/<issue-id>-<description>
  - example: hotfix/ORCH-123-redis-connection-leak
  - merge to both main and release branch
```

---

## 11. Implementation Checklist

### Phase 1: Event Processing (Weeks 1-2, 16 SP)

- [ ] **[ORCH-001] Redis Streams Integration** (8 SP)
  - [ ] Redis cluster configuration and deployment
  - [ ] Stream creation and consumer groups
  - [ ] Dead letter queue implementation
  - [ ] Event ordering and deduplication logic
  - [ ] Unit tests (95% coverage)

- [ ] **[ORCH-002] Workflow Graph Engine** (8 SP)
  - [ ] NetworkX integration for graph operations
  - [ ] Workflow definition parsing and validation
  - [ ] Dependency resolution algorithms (topological sort)
  - [ ] Parallel execution planning
  - [ ] Critical path calculation
  - [ ] Cycle detection
  - [ ] Unit tests for graph algorithms

**Phase 1 Exit Criteria:**
- Redis Streams publishing and consuming events
- NetworkX graph operations working (topological sort, cycle detection)
- Workflow definitions validated successfully
- All unit tests passing (95% coverage)

### Phase 2: Core Orchestration (Weeks 3-4, 21 SP)

- [ ] **[ORCH-003] Supervisor Pattern Implementation** (5 SP)
  - [ ] Master-worker coordination logic
  - [ ] Task distribution and monitoring
  - [ ] Worker failure handling and recovery
  - [ ] Load balancing strategies
  - [ ] Integration tests with mock agents

- [ ] **[ORCH-004] Hierarchical Pattern Support** (8 SP)
  - [ ] Multi-level agent hierarchies
  - [ ] Delegation and escalation mechanisms
  - [ ] Authority and permission management
  - [ ] Communication flow optimization
  - [ ] Integration tests

- [ ] **[ORCH-005] CQRS Implementation** (8 SP)
  - [ ] Command and query separation
  - [ ] Event sourcing for audit trails
  - [ ] Read model optimization (PostgreSQL read replicas)
  - [ ] Eventual consistency handling
  - [ ] Integration tests with PostgreSQL

**Phase 2 Exit Criteria:**
- Supervisor and hierarchical patterns working
- CQRS architecture implemented
- Workflow execution engine functional
- Agent allocation via Kubernetes working
- Integration tests passing

### Phase 3: Advanced Patterns (Weeks 5-6, 26 SP)

- [ ] **[ORCH-006] Handoff Pattern Implementation** (5 SP)
  - [ ] Sequential task handoff mechanisms
  - [ ] Context preservation during transfers
  - [ ] Quality gates and validation
  - [ ] Rollback capabilities
  - [ ] Integration tests

- [ ] **[ORCH-007] Swarm Pattern Support** (8 SP)
  - [ ] Distributed coordination algorithms
  - [ ] Emergent behavior management
  - [ ] Consensus and voting mechanisms
  - [ ] Performance optimization for large swarms
  - [ ] Integration tests

- [ ] **[ORCH-008] Saga Pattern & Compensation** (13 SP)
  - [ ] Long-running transaction management
  - [ ] Compensation action definition and execution
  - [ ] State recovery and rollback logic
  - [ ] Consistency guarantees
  - [ ] Idempotent compensation operations
  - [ ] Integration tests with failure injection

**Phase 3 Exit Criteria:**
- All 6 orchestration patterns implemented (supervisor, hierarchical, handoff, swarm, network, saga)
- Saga compensation working
- Pattern library complete
- Integration tests for all patterns passing

### Phase 4: Production Features (Weeks 7-8, 31 SP)

- [ ] **[ORCH-009] Fault Tolerance & Circuit Breakers** (8 SP)
  - [ ] Circuit breaker implementation
  - [ ] Retry policies with exponential backoff
  - [ ] Health monitoring and recovery
  - [ ] Graceful degradation strategies
  - [ ] Integration tests with chaos engineering

- [ ] **[ORCH-010] Performance & Scalability** (8 SP)
  - [ ] Optimize for <1s planning (1000+ node graphs)
  - [ ] Achieve 100,000+ events/second processing
  - [ ] Linear scaling validation (load testing)
  - [ ] GPU acceleration for NetworkX (nx-cugraph)
  - [ ] Load testing with Locust (1000+ workflows)

- [ ] **[ORCH-011] Custom Pattern Framework with Hooks System** (10 SP)
  - [ ] Custom pattern definition interface
  - [ ] Pattern registration and validation
  - [ ] Template system for common patterns
  - [ ] Pattern library management
  - [ ] **Hooks System (NEW):**
    - [ ] Hook configuration model (HookConfig, HookAction)
    - [ ] Hook registration and event matching
    - [ ] PostgreSQL workflow_hooks table and Alembic migration
    - [ ] Async hook execution via Redis Streams queue
    - [ ] Hook error handling and retry logic
    - [ ] Hook execution monitoring and logging
    - [ ] Integration with A2A-007 Event System
    - [ ] Unit tests for hook execution (95% coverage)
    - [ ] Integration tests with real hooks

- [ ] **[ORCH-012] PostgreSQL State Management** (5 SP)
  - [ ] PostgreSQL integration with JSONB
  - [ ] Workflow state persistence
  - [ ] State migration and versioning (Alembic)
  - [ ] Performance optimization for state queries (indexing)
  - [ ] Integration tests

**Phase 4 Exit Criteria:**
- Fault tolerance mechanisms working (circuit breaker, saga)
- Performance targets achieved (<1s planning, 100k+ events/sec)
- Custom pattern framework functional
- Hooks system operational (pre/post/session hooks)
- PostgreSQL state management complete
- All integration tests passing
- E2E tests passing
- Load testing validated (1000+ concurrent workflows)
- Ready for production deployment

---

## 12. References

### 12.1 Internal Documentation

- [Orchestration Engine Specification](../specs/orchestration-engine/spec.md)
- [Implementation Plan](../specs/orchestration-engine/plan.md)
- [Task Breakdown](../specs/orchestration-engine/tasks.md)
- [A2A Protocol Breakdown](./a2a-protocol.md)
- [Agent Templates Library](../agent-templates.md)

### 12.2 External Research & Documentation

**Redis Streams for Event Sourcing:**
- [Redis Streams Documentation](https://redis.io/docs/latest/develop/data-types/streams/) (Official)
- [Event-Driven Architecture Using Redis Streams](https://www.harness.io/blog/event-driven-architecture-redis-streams) (Harness)
- [Microservices Communication with Redis Streams](https://redis.io/learn/howtos/solutions/microservices/interservice-communication) (Redis Labs)
- Research Findings: Redis Streams provide 99.9% availability, persistent data, consumer groups, suitable for 1000s messages/min

**NetworkX Graph Algorithms:**
- [NetworkX Documentation](https://networkx.org/documentation/stable/) (Official)
- [NetworkX GPU Acceleration with cuGraph](https://developer.nvidia.com/blog/accelerating-networkx-on-nvidia-gpus-for-high-performance-graph-analytics) (NVIDIA)
- Research Findings: NetworkX offers O(n+m) algorithms, GPU acceleration provides 6-634x speedups, perfect for DAGs and workflow graphs

**Saga Pattern Implementation:**
- [Microservices Pattern: Saga](https://microservices.io/patterns/data/saga.html) (Chris Richardson)
- [Saga Pattern in Microservices](https://www.baeldung.com/cs/saga-pattern-microservices) (Baeldung)
- [7 Proven Steps to Implement Saga Pattern in Python](https://softwarelogic.co/en/blog/7-proven-steps-to-implement-saga-pattern-in-python-microservices) (SoftwareLogic)
- Research Findings: Saga pattern requires idempotent compensation actions, orchestration approach recommended over choreography

**CQRS & Event Sourcing:**
- [CQRS Pattern](https://microservices.io/patterns/data/cqrs.html) (Microservices.io)
- [Event Sourcing with Redis](https://dev.to/pdambrauskas/event-sourcing-with-redis-45ha) (DEV Community)
- Research Findings: CQRS provides 10x read performance, event sourcing enables complete audit trails

**Kubernetes Orchestration:**
- [Kubernetes Orchestration Best Practices](https://kubernetes.io/docs/concepts/workloads/) (Official)
- [Kubernetes Pod Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/) (Official)

### 12.3 Technology Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Redis-py Documentation](https://redis-py.readthedocs.io/)
- [Kubernetes Python Client](https://github.com/kubernetes-client/python)

---

**End of Technical Breakdown**

**Document Version:** 2.0
**Created:** 2025-09-27
**Updated:** 2025-10-01
**Status:** Ready for Implementation
**Next Steps:**
1. Review and approve this breakdown
2. Begin Phase 1 implementation (Weeks 1-2)
3. Set up CI/CD pipeline
4. Schedule daily standups

**Questions/Clarifications:**
- Session Management API (A2A-019, A2A-020) timeline confirmation for session hooks integration
- GPU acceleration (nx-cugraph) feasibility for production deployment
- Multi-region deployment priority (Phase 4+)
