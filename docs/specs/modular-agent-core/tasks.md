# Tasks: Modular Agent Core

**From:** `spec.md` + `plan.md`
**Timeline:** 6-7 weeks, 3-4 sprints
**Team:** 2 Backend Engineers, 0.5 DevOps, 0.5 QA
**Created:** 2025-10-15

## Summary

- **Total tasks:** 31 (1 epic + 30 stories)
- **Estimated effort:** 162 story points
- **Critical path duration:** 5-8 weeks (1 engineer) or 3-4 weeks (2 engineers)
- **Key risks:**
  1. Complex module coordination and orchestration
  2. Refinement loop feedback mechanism complexity
  3. Baseline data availability for benchmarking

## Phase Breakdown

### Phase 1: Foundation (Weeks 1-2, 29 story points)

**Goal:** Establish modular architecture foundation with interfaces, base classes, and coordination infrastructure
**Deliverable:** Working module interfaces, database schema, and coordination handler

#### Tasks

**[MOD-001] Epic: Modular Agent Core Implementation**

- **Description:** Implement specialized four-module agent architecture (Planner, Executor, Verifier, Generator) with iterative refinement, A2A protocol integration, and performance optimization
- **Acceptance:**
  - [ ] All 30 story tasks completed
  - [ ] Performance targets met (+15% success rate, <2x latency, 30% cost reduction)
  - [ ] 90%+ test coverage
  - [ ] Production-ready with monitoring and documentation
- **Effort:** 162 story points (6-7 weeks)
- **Owner:** Engineering Team
- **Dependencies:** None (Epic)
- **Priority:** P0 (Critical)
- **Type:** Epic

---

**[MOD-002] Define Module Interface Protocols**

- **Description:** Create Protocol definitions for all four modules (PlannerInterface, ExecutorInterface, VerifierInterface, GeneratorInterface) with async method signatures, input/output types, and error handling contracts
- **Acceptance:**
  - [ ] `agentcore/modular/interfaces.py` created with all four Protocol definitions
  - [ ] Type hints using Pydantic models for all parameters
  - [ ] Docstrings with examples for each interface method
  - [ ] Mypy strict mode validation passes
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-001
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/modular/interfaces.py`

**[MOD-003] Implement Base Module Classes**

- **Description:** Create abstract base classes implementing module interfaces with common functionality: logging, error handling, A2A context propagation, state management
- **Acceptance:**
  - [ ] BaseModule abstract class with shared logging/error handling
  - [ ] Each module (Planner, Executor, Verifier, Generator) has base class
  - [ ] A2A context (trace_id, source_agent, session_id) propagated
  - [ ] Structured logging using structlog
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-002
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/modular/base.py`

**[MOD-004] Create Execution Plan Data Models**

- **Description:** Implement Pydantic models for ExecutionPlan, PlanStep, ModuleTransition, VerificationResult with validation, serialization, and database compatibility
- **Acceptance:**
  - [ ] ExecutionPlan model with steps, success_criteria, max_iterations
  - [ ] PlanStep model with dependencies, tool_requirements, status
  - [ ] ModuleTransition model for tracking module flow
  - [ ] VerificationResult model with confidence scores, feedback
  - [ ] All models have comprehensive unit tests
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-002
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/models.py`

**[MOD-005] Database Migration for Modular Executions**

- **Description:** Create Alembic migration for modular_executions, execution_plans, plan_steps, module_transitions tables with proper indexes and foreign keys
- **Acceptance:**
  - [ ] Migration creates modular_executions table (id, query, plan_id, iterations, final_result, status, error, created_at, completed_at)
  - [ ] Migration creates execution_plans table with JSON plan storage
  - [ ] Migration creates plan_steps table with dependencies
  - [ ] Migration creates module_transitions table for tracing
  - [ ] Indexes on frequently queried columns (status, created_at, plan_id)
  - [ ] Migration applies and rolls back successfully
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-004
- **Priority:** P0 (Critical)
- **Files:** `alembic/versions/XXX_add_modular_executions.py`

**[MOD-006] JSON-RPC Coordination Handler**

- **Description:** Implement core coordination handler that manages module-to-module communication using JSON-RPC 2.0, handles message routing, and maintains execution context
- **Acceptance:**
  - [ ] ModuleCoordinator class with async message passing
  - [ ] Support for module registration and discovery
  - [ ] Request routing based on module capabilities
  - [ ] Execution context management (trace_id, session_id)
  - [ ] Error handling and timeout management
  - [ ] Unit tests for coordinator logic
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-003
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/modular/coordinator.py`

**[MOD-007] Module State Management**

- **Description:** Implement state persistence and recovery for module executions, including checkpoint creation, state serialization, and crash recovery
- **Acceptance:**
  - [ ] StateManager class for checkpoint creation/restoration
  - [ ] State serialization using Pydantic models
  - [ ] Database persistence of execution state
  - [ ] Recovery mechanism for crashed executions
  - [ ] State cleanup for completed executions
  - [ ] Tests for crash recovery scenarios
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-005, MOD-006
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/state_manager.py`

**[MOD-008] Unit Tests for Interfaces and Base Classes**

- **Description:** Comprehensive unit test suite for module interfaces, base classes, data models, and coordination handler with 90%+ coverage
- **Acceptance:**
  - [ ] Test coverage ≥90% for modular package
  - [ ] Tests for all interface contracts
  - [ ] Tests for base class functionality (logging, error handling)
  - [ ] Tests for data model validation
  - [ ] Tests for coordinator message routing
  - [ ] Tests use pytest-asyncio for async code
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer
- **Dependencies:** MOD-002, MOD-003, MOD-004, MOD-006
- **Priority:** P0 (Critical)
- **Files:** `tests/unit/modular/test_interfaces.py`, `tests/unit/modular/test_base.py`, `tests/unit/modular/test_models.py`, `tests/unit/modular/test_coordinator.py`

---

### Phase 2: Module Implementation (Weeks 3-4, 71 story points)

**Goal:** Implement all four specialized modules with A2A integration and end-to-end workflow
**Deliverable:** Working modular.solve JSON-RPC method with full pipeline execution

#### Tasks

**[MOD-009] Implement Planner Module with Task Decomposition**

- **Description:** Build Planner module that analyzes queries and creates structured execution plans with ordered steps, tool requirements, and success criteria using LLM-based decomposition
- **Acceptance:**
  - [ ] PlannerModule class implementing PlannerInterface
  - [ ] Query analysis and task decomposition logic
  - [ ] ExecutionPlan generation with dependencies
  - [ ] Tool requirement identification
  - [ ] Success criteria definition
  - [ ] Support for multiple planning strategies (ReAct, Chain-of-Thought)
  - [ ] Unit tests with mocked LLM responses
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-003, MOD-004
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/planner.py`

**[MOD-010] Implement Plan Refinement Logic in Planner**

- **Description:** Add plan refinement capability to Planner module that incorporates verification feedback and creates improved execution plans
- **Acceptance:**
  - [ ] Planner.refine() method processes VerificationResult feedback
  - [ ] Analyzes failure reasons and adjusts strategy
  - [ ] Creates refined ExecutionPlan with corrected steps
  - [ ] Tracks refinement history
  - [ ] Respects max_iterations limit (default: 5)
  - [ ] Tests validate improvement in refined plans
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-009
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/planner.py`

**[MOD-011] Implement Executor Module with Tool Invocation**

- **Description:** Build Executor module that executes plan steps by invoking tools from Tool Integration Framework (TOOL-001) with proper parameter formatting and execution monitoring
- **Acceptance:**
  - [ ] ExecutorModule class implementing ExecutorInterface
  - [ ] Integration with Tool Integration Framework
  - [ ] Tool parameter formatting and validation
  - [ ] Execution monitoring and timeout handling
  - [ ] Result collection and formatting
  - [ ] Support for parallel tool execution where possible
  - [ ] Unit tests with mocked tool responses
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-003, MOD-004, TOOL-001 (Tool Integration Framework)
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/executor.py`

**[MOD-012] Executor Retry and Error Recovery**

- **Description:** Add retry logic and error recovery mechanisms to Executor module for handling transient failures and tool errors
- **Acceptance:**
  - [ ] Configurable retry strategy (exponential backoff)
  - [ ] Differentiation between retryable and non-retryable errors
  - [ ] Circuit breaker for failing tools
  - [ ] Graceful degradation when tools unavailable
  - [ ] Error categorization and reporting
  - [ ] Tests for retry scenarios and error handling
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-011
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/executor.py`

**[MOD-013] Implement Verifier Module with Validation Rules**

- **Description:** Build Verifier module that validates execution results against success criteria using rule-based validation and LLM-based verification
- **Acceptance:**
  - [ ] VerifierModule class implementing VerifierInterface
  - [ ] Rule-based validation for structured outputs
  - [ ] LLM-based verification for semantic correctness
  - [ ] Logical consistency checking
  - [ ] Completeness validation
  - [ ] Hallucination detection
  - [ ] Structured feedback generation for refinement
  - [ ] Unit tests with validation scenarios
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-003, MOD-004
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/verifier.py`

**[MOD-014] Verifier Confidence Scoring**

- **Description:** Add confidence scoring mechanism to Verifier module that assigns numeric confidence to validation results based on multiple signals
- **Acceptance:**
  - [ ] Confidence score calculation (0.0-1.0)
  - [ ] Multiple confidence signals (rule match, LLM certainty, consistency)
  - [ ] Configurable confidence threshold for acceptance
  - [ ] Confidence score included in VerificationResult
  - [ ] Low confidence triggers plan refinement
  - [ ] Tests validate scoring accuracy
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-013
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/verifier.py`

**[MOD-015] Implement Generator Module with Response Synthesis**

- **Description:** Build Generator module that synthesizes final responses from verified execution results with explanations, reasoning traces, and proper formatting
- **Acceptance:**
  - [ ] GeneratorModule class implementing GeneratorInterface
  - [ ] Response synthesis from execution results
  - [ ] Reasoning trace generation
  - [ ] Output formatting per user requirements
  - [ ] Supporting evidence inclusion
  - [ ] Response quality validation
  - [ ] Unit tests with synthesis scenarios
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-003, MOD-004
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/generator.py`

**[MOD-016] Register Modules as A2A Agents**

- **Description:** Register all four modules (Planner, Executor, Verifier, Generator) as discoverable A2A agents with AgentCard specifications and capability advertisements
- **Acceptance:**
  - [ ] Each module has AgentCard with capabilities
  - [ ] Modules registered with AgentManager on startup
  - [ ] Discoverable via /.well-known/agents/{module-id}
  - [ ] Health check endpoints for each module
  - [ ] Capability advertisement includes version, cost, latency
  - [ ] Integration tests validate discovery
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-009, MOD-011, MOD-013, MOD-015
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/registration.py`

**[MOD-017] Implement modular.solve JSON-RPC Method**

- **Description:** Create primary JSON-RPC endpoint (modular.solve) that accepts queries and orchestrates execution through all four modules
- **Acceptance:**
  - [ ] JSON-RPC method registered: modular.solve
  - [ ] Request validation using Pydantic models
  - [ ] Query parameter with optional config (max_iterations, module versions)
  - [ ] Response includes answer and execution_trace
  - [ ] Error handling with JSON-RPC error codes
  - [ ] Registered via @register_jsonrpc_method decorator
  - [ ] Integration tests with end-to-end queries
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-006, MOD-016
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/modular/jsonrpc.py`

**[MOD-018] Module Coordination Loop**

- **Description:** Implement core coordination loop that orchestrates Planner→Executor→Verifier→[refine if needed]→Generator workflow with iteration management
- **Acceptance:**
  - [ ] Coordination loop executes all modules in sequence
  - [ ] Verification failures trigger plan refinement
  - [ ] Max iteration limit enforced (default: 5)
  - [ ] State persistence at each module transition
  - [ ] Module transition events emitted
  - [ ] Early exit on verification success
  - [ ] Timeout handling for long-running executions
  - [ ] Integration tests with refinement scenarios
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-017
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/modular/coordinator.py`

**[MOD-019] Integration Tests for Full Pipeline**

- **Description:** Comprehensive integration test suite covering end-to-end execution through all four modules with real tool integrations
- **Acceptance:**
  - [ ] Tests for successful single-iteration execution
  - [ ] Tests for plan refinement loop (multiple iterations)
  - [ ] Tests for max iteration limit enforcement
  - [ ] Tests for error recovery and retry
  - [ ] Tests for module state persistence
  - [ ] Tests use real PostgreSQL via testcontainers
  - [ ] Tests validate execution traces
  - [ ] Test coverage ≥90% for coordination logic
- **Effort:** 8 story points (5-8 days)
- **Owner:** QA Engineer
- **Dependencies:** MOD-018
- **Priority:** P0 (Critical)
- **Files:** `tests/integration/modular/test_pipeline.py`, `tests/integration/modular/test_refinement.py`

---

### Phase 3: Optimization (Weeks 5-6, 62 story points)

**Goal:** Optimize performance, add monitoring, and achieve production readiness
**Deliverable:** Production-ready modular system with monitoring, tracing, and validated performance

#### Tasks

**[MOD-020] Optimize Module Response Times**

- **Description:** Profile and optimize module execution to reduce latency, focusing on LLM call efficiency, caching, and parallel execution
- **Acceptance:**
  - [ ] Profiling identifies bottlenecks in each module
  - [ ] LLM prompt optimization reduces token usage
  - [ ] Response caching for repeated queries
  - [ ] Parallel execution where dependencies allow
  - [ ] Module transition time <500ms (excluding tool execution)
  - [ ] Overall latency <2x baseline validated
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-018, MOD-019
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/optimizer.py`

**[MOD-021] Implement Plan Refinement Loop with Max Iterations**

- **Description:** Enhance refinement loop with intelligent iteration management, convergence detection, and early stopping
- **Acceptance:**
  - [ ] Max iterations configurable (default: 5)
  - [ ] Convergence detection (no improvement after N iterations)
  - [ ] Early stopping when confidence threshold met
  - [ ] Refinement history tracking
  - [ ] Metrics on refinement effectiveness
  - [ ] Tests validate iteration limits
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-010, MOD-018
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/coordinator.py`

**[MOD-022] Integrate OpenTelemetry Distributed Tracing**

- **Description:** Add OpenTelemetry instrumentation for distributed tracing across all modules with span creation, context propagation, and trace export
- **Acceptance:**
  - [ ] OpenTelemetry SDK integrated
  - [ ] Spans created for each module execution
  - [ ] Trace context propagated via A2A context
  - [ ] Tool executions linked to parent trace
  - [ ] Trace export to collector (Jaeger/Zipkin)
  - [ ] Trace visualization shows execution timeline
- **Effort:** 8 story points (5-8 days)
- **Owner:** DevOps Engineer
- **Dependencies:** MOD-018
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/tracing.py`

**[MOD-023] Add Trace ID Propagation Across Modules**

- **Description:** Ensure trace IDs are propagated through all module transitions and included in logs, metrics, and database records
- **Acceptance:**
  - [ ] Trace ID generated at query entry
  - [ ] Trace ID included in A2A context for all module calls
  - [ ] Trace ID in all log messages (structlog)
  - [ ] Trace ID stored in modular_executions table
  - [ ] Trace ID in error responses
  - [ ] Tests validate trace ID continuity
- **Effort:** 3 story points (2-3 days)
- **Owner:** DevOps Engineer
- **Dependencies:** MOD-022
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/tracing.py`

**[MOD-024] Module-Specific Model Configuration**

- **Description:** Enable independent LLM model configuration for each module to optimize cost/quality trade-offs (larger model for Planner, smaller for Verifier)
- **Acceptance:**
  - [ ] Configuration system for per-module model selection
  - [ ] Planner uses larger model (e.g., GPT-5)
  - [ ] Executor uses medium model (e.g., GPT-4.1)
  - [ ] Verifier uses smaller model (e.g., GPT-4.1-mini)
  - [ ] Generator uses medium model
  - [ ] Cost tracking per module
  - [ ] Quality validation with mixed model sizes
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** MOD-009, MOD-011, MOD-013, MOD-015
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/config.py`

**[MOD-025] Performance Benchmarking vs Baseline**

- **Description:** Conduct comprehensive performance benchmarking against single-agent baseline to validate NFR targets (+15% success rate, <2x latency, 30% cost reduction)
- **Acceptance:**
  - [ ] Benchmark suite with 100 test queries
  - [ ] Baseline measurements from single-agent system
  - [ ] Modular system measurements on same queries
  - [ ] Success rate improvement ≥15%
  - [ ] Latency increase <2x baseline
  - [ ] Cost reduction ≥30%
  - [ ] Benchmark report with statistical analysis
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer
- **Dependencies:** MOD-020, MOD-024
- **Priority:** P0 (Critical)
- **Files:** `tests/benchmarks/test_modular_performance.py`

**[MOD-026] Prometheus Metrics for Each Module**

- **Description:** Add Prometheus metrics for module-level monitoring: latency, success rate, error rate, iteration count, token usage
- **Acceptance:**
  - [ ] Metrics for each module (planner, executor, verifier, generator)
  - [ ] Latency histograms (p50, p95, p99)
  - [ ] Success/failure counters
  - [ ] Error rate by type
  - [ ] Iteration count distribution
  - [ ] Token usage per module
  - [ ] Metrics endpoint at /metrics
- **Effort:** 5 story points (3-5 days)
- **Owner:** DevOps Engineer
- **Dependencies:** MOD-018
- **Priority:** P1 (High)
- **Files:** `src/agentcore/modular/metrics.py`

**[MOD-027] Grafana Dashboards for Module Health**

- **Description:** Create Grafana dashboards for real-time monitoring of module health, performance, and system metrics
- **Acceptance:**
  - [ ] Overview dashboard with key metrics
  - [ ] Per-module dashboards (Planner, Executor, Verifier, Generator)
  - [ ] Performance dashboard (latency, throughput)
  - [ ] Error dashboard (error rates, types, traces)
  - [ ] Cost dashboard (token usage, model costs)
  - [ ] Alerts configured for SLO violations
- **Effort:** 3 story points (2-3 days)
- **Owner:** DevOps Engineer
- **Dependencies:** MOD-026
- **Priority:** P1 (High)
- **Files:** `k8s/monitoring/grafana-dashboards.yaml`

**[MOD-028] Load Testing (100 Concurrent Executions)**

- **Description:** Conduct load testing to validate system handles 100 concurrent modular executions per instance without degradation
- **Acceptance:**
  - [ ] Locust load test with 100 concurrent users
  - [ ] Sustained load for 10 minutes
  - [ ] Success rate >95% under load
  - [ ] p95 latency <3x baseline
  - [ ] No memory leaks or resource exhaustion
  - [ ] Load test report with metrics
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer
- **Dependencies:** MOD-019, MOD-020
- **Priority:** P1 (High)
- **Files:** `tests/load/test_modular_load.py`

**[MOD-029] Error Recovery Testing (>80% Target)**

- **Description:** Test error recovery mechanisms to validate >80% success rate on recoverable errors (retries, plan refinement)
- **Acceptance:**
  - [ ] Test suite for recoverable error scenarios
  - [ ] Transient tool failures (network errors, timeouts)
  - [ ] Invalid tool parameters (fixed via refinement)
  - [ ] Incomplete results (refined via verification)
  - [ ] Recovery success rate ≥80%
  - [ ] Recovery time metrics
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer
- **Dependencies:** MOD-012, MOD-021
- **Priority:** P1 (High)
- **Files:** `tests/integration/modular/test_error_recovery.py`

**[MOD-030] Security Audit (Auth, RBAC, Audit Logging)**

- **Description:** Conduct security audit of modular system covering authentication, authorization, audit logging, and data protection
- **Acceptance:**
  - [ ] All module communications use JWT authentication
  - [ ] RBAC policies control module access
  - [ ] All module interactions auditable with trace IDs
  - [ ] Sensitive data not logged
  - [ ] Input validation on all JSON-RPC methods
  - [ ] Security audit report with findings and fixes
- **Effort:** 8 story points (5-8 days)
- **Owner:** Security/Backend Engineer
- **Dependencies:** MOD-017, MOD-018
- **Priority:** P0 (Critical)
- **Files:** `docs/security/modular-security-audit.md`

**[MOD-031] Production Runbook and Documentation**

- **Description:** Create comprehensive production runbook and documentation for deployment, operation, and troubleshooting of modular system
- **Acceptance:**
  - [ ] Architecture documentation with diagrams
  - [ ] API documentation (modular.solve method)
  - [ ] Deployment guide (Docker, Kubernetes)
  - [ ] Operations runbook (monitoring, alerts, incident response)
  - [ ] Troubleshooting guide (common issues, debug steps)
  - [ ] Configuration reference (all settings documented)
  - [ ] Performance tuning guide
- **Effort:** 5 story points (3-5 days)
- **Owner:** Tech Lead
- **Dependencies:** MOD-025, MOD-026, MOD-030
- **Priority:** P1 (High)
- **Files:** `docs/modular-agent-core/README.md`, `docs/modular-agent-core/runbook.md`

---

## Critical Path

```plaintext
MOD-002 → MOD-003 → MOD-009 → MOD-017 → MOD-018 → MOD-021 → MOD-025
  (3d)      (5d)      (8d)      (8d)      (8d)      (5d)      (5d)
                            [42 story points / 5-8 weeks]
```

**Bottlenecks:**

- **MOD-018 (Module Coordination Loop):** Highest complexity, cross-module orchestration, refinement logic
- **MOD-021 (Plan Refinement Loop):** Complex feedback mechanism, convergence detection
- **MOD-025 (Performance Benchmarking):** Depends on baseline data availability and system stability

**Parallel Tracks:**

- **Module Implementation:** MOD-009 (Planner), MOD-011 (Executor), MOD-013 (Verifier), MOD-015 (Generator) can be developed in parallel after MOD-003
- **Testing:** MOD-008, MOD-019, MOD-028, MOD-029 can run in parallel with development
- **DevOps/Monitoring:** MOD-022, MOD-023, MOD-026, MOD-027 can be done by separate engineer

---

## Quick Wins (Week 1-2)

1. **[MOD-002] Module Interface Protocols** - Unblocks all module development work
2. **[MOD-004] Execution Plan Data Models** - Demonstrates progress, validates approach
3. **[MOD-005] Database Migration** - Validates persistence layer integration

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| MOD-018 | Coordination loop complexity causes delays | Early prototype in week 2, incremental testing | Simplify refinement loop to single iteration |
| MOD-021 | Refinement convergence unreliable | Extensive testing with diverse queries, tunable thresholds | Fallback to fixed iteration count |
| MOD-025 | Baseline data unavailable or incomparable | Establish baseline in week 0, use standardized benchmark suite | Use synthetic benchmarks from research |
| MOD-011 | Tool Integration Framework (TOOL-001) not ready | Mock tool interface for development, coordinate with TOOL team | Implement basic tool registry as temporary solution |
| MOD-020 | Performance targets not met | Profile early (week 4), allocate week 7 buffer | Relax latency target to <3x with user approval |

---

## Testing Strategy

### Automated Testing Tasks

- **[MOD-008] Unit Test Framework** (5 SP) - Week 2
- **[MOD-019] Integration Tests** (8 SP) - Week 4
- **[MOD-028] Load Testing** (5 SP) - Week 6
- **[MOD-029] Error Recovery Testing** (5 SP) - Week 6

### Quality Gates

- **90% code coverage** required (per CLAUDE.md)
- **All P0 features** must have integration tests
- **Performance tests** validate NFR targets before production
- **Security audit** passes before deployment

### Testing Coverage

| Component | Unit Tests | Integration Tests | Load Tests | Security Tests |
|-----------|------------|-------------------|------------|----------------|
| Interfaces | ✓ MOD-008 | ✓ MOD-019 | - | - |
| Modules | ✓ (included in MOD-009-015) | ✓ MOD-019 | ✓ MOD-028 | ✓ MOD-030 |
| Coordination | ✓ MOD-008 | ✓ MOD-019 | ✓ MOD-028 | ✓ MOD-030 |
| API | - | ✓ MOD-019 | ✓ MOD-028 | ✓ MOD-030 |
| Error Recovery | - | ✓ MOD-029 | - | - |

---

## Team Allocation

**Backend Engineers (2 FTE)**

- Core module development (MOD-009, MOD-011, MOD-013, MOD-015)
- Coordination logic (MOD-006, MOD-017, MOD-018)
- Refinement and optimization (MOD-010, MOD-012, MOD-020, MOD-021)
- Module configuration (MOD-024)

**DevOps Engineer (0.5 FTE)**

- Infrastructure (MOD-005, MOD-007)
- Monitoring and tracing (MOD-022, MOD-023, MOD-026, MOD-027)
- Load testing support (MOD-028)

**QA Engineer (0.5 FTE)**

- Unit test framework (MOD-008)
- Integration testing (MOD-019)
- Performance benchmarking (MOD-025)
- Load and error recovery testing (MOD-028, MOD-029)

**Security Engineer (consulting)**

- Security audit (MOD-030)

**Tech Lead (coordination)**

- Architecture decisions
- Code reviews
- Documentation (MOD-031)
- Risk management

---

## Sprint Planning

**2-week sprints, ~40 SP velocity per sprint (2 backend engineers)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| **Sprint 1** (Weeks 1-2) | Foundation | 29 SP | Module interfaces, base classes, database schema, coordination handler |
| **Sprint 2** (Weeks 3-4) | Core Modules Part 1 | 38 SP | Planner, Executor, Verifier implementations with A2A registration |
| **Sprint 3** (Weeks 4-5) | Core Modules Part 2 | 33 SP | Generator, modular.solve API, coordination loop, integration tests |
| **Sprint 4** (Weeks 5-6) | Optimization Part 1 | 35 SP | Performance optimization, tracing, model configuration, metrics |
| **Sprint 5** (Weeks 6-7) | Optimization Part 2 | 27 SP | Dashboards, load testing, security audit, documentation |

**Total:** 162 SP across 5 sprints (6-7 weeks)

---

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint,Type
MOD-001,Epic: Modular Agent Core,Implement four-module architecture,162,P0,Team,,All,Epic
MOD-002,Define Module Interfaces,Create Protocol definitions,3,P0,Backend,MOD-001,1,Story
MOD-003,Implement Base Classes,Base classes with logging/error handling,5,P0,Backend,MOD-002,1,Story
MOD-004,Create Data Models,Pydantic models for execution plans,3,P0,Backend,MOD-002,1,Story
MOD-005,Database Migration,Alembic migration for modular tables,3,P0,Backend,MOD-004,1,Story
MOD-006,Coordination Handler,JSON-RPC message routing,5,P0,Backend,MOD-003,1,Story
MOD-007,State Management,Checkpoint and recovery,5,P1,Backend,"MOD-005,MOD-006",1,Story
MOD-008,Unit Tests Foundation,Test interfaces and base classes,5,P0,QA,"MOD-002,MOD-003,MOD-004,MOD-006",1,Story
MOD-009,Implement Planner,Task decomposition module,8,P0,Backend,"MOD-003,MOD-004",2,Story
MOD-010,Plan Refinement,Feedback-based plan refinement,5,P1,Backend,MOD-009,2,Story
MOD-011,Implement Executor,Tool invocation module,8,P0,Backend,"MOD-003,MOD-004,TOOL-001",2,Story
MOD-012,Executor Retry Logic,Error recovery for Executor,5,P1,Backend,MOD-011,2,Story
MOD-013,Implement Verifier,Result validation module,8,P0,Backend,"MOD-003,MOD-004",2,Story
MOD-014,Verifier Confidence,Confidence scoring,3,P1,Backend,MOD-013,2,Story
MOD-015,Implement Generator,Response synthesis module,5,P0,Backend,"MOD-003,MOD-004",2,Story
MOD-016,A2A Registration,Register modules as agents,5,P0,Backend,"MOD-009,MOD-011,MOD-013,MOD-015",3,Story
MOD-017,modular.solve API,JSON-RPC endpoint,8,P0,Backend,"MOD-006,MOD-016",3,Story
MOD-018,Coordination Loop,Orchestrate module workflow,8,P0,Backend,MOD-017,3,Story
MOD-019,Integration Tests,End-to-end pipeline tests,8,P0,QA,MOD-018,3,Story
MOD-020,Optimize Performance,Reduce latency,5,P1,Backend,"MOD-018,MOD-019",4,Story
MOD-021,Refinement Loop,Iteration management,5,P1,Backend,"MOD-010,MOD-018",4,Story
MOD-022,OpenTelemetry Tracing,Distributed tracing,8,P1,DevOps,MOD-018,4,Story
MOD-023,Trace ID Propagation,Trace continuity,3,P1,DevOps,MOD-022,4,Story
MOD-024,Module Model Config,Per-module LLM selection,5,P1,Backend,"MOD-009,MOD-011,MOD-013,MOD-015",4,Story
MOD-025,Performance Benchmark,Validate NFR targets,5,P0,QA,"MOD-020,MOD-024",4,Story
MOD-026,Prometheus Metrics,Module monitoring,5,P1,DevOps,MOD-018,4,Story
MOD-027,Grafana Dashboards,Health dashboards,3,P1,DevOps,MOD-026,5,Story
MOD-028,Load Testing,100 concurrent executions,5,P1,QA,"MOD-019,MOD-020",5,Story
MOD-029,Error Recovery Testing,Validate >80% recovery,5,P1,QA,"MOD-012,MOD-021",5,Story
MOD-030,Security Audit,Auth and audit logging,8,P0,Security,"MOD-017,MOD-018",5,Story
MOD-031,Production Runbook,Operational documentation,5,P1,Tech Lead,"MOD-025,MOD-026,MOD-030",5,Story
```

---

## Appendix

**Estimation Method:** Planning poker with team, Fibonacci scale
**Story Point Scale:** 1 (trivial), 2 (simple), 3 (moderate), 5 (complex), 8 (very complex), 13 (epic-level)
**Velocity Assumption:** 20 SP per engineer per 2-week sprint (conservative)

**Definition of Done:**

- [ ] Code reviewed and approved by 2+ engineers
- [ ] Unit tests written and passing (90%+ coverage)
- [ ] Integration tests passing (where applicable)
- [ ] Documentation updated (docstrings, README)
- [ ] Deployed to staging environment
- [ ] Performance validated (meets NFR targets)
- [ ] Security reviewed (for P0 tasks)

**Dependencies:**

- **TOOL-001 (Tool Integration Framework):** Required for MOD-011 (Executor module)
- **Baseline System:** Required for MOD-025 (performance benchmarking)
- **PostgreSQL Database:** Required for all persistence tasks
- **Redis (optional):** For distributed coordination if needed

**Technology Stack:**

- Python 3.12+ with async/await
- FastAPI framework
- JSON-RPC 2.0
- PostgreSQL with asyncpg
- Pydantic v2
- Portkey AI gateway
- Prometheus + Grafana
- OpenTelemetry
- pytest-asyncio

**References:**

- Spec: `docs/specs/modular-agent-core/spec.md`
- Plan: `docs/specs/modular-agent-core/plan.md`
- Research: `docs/research/modular-agent-architecture.md`
