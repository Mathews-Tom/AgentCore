# Tasks: Coordination Service (REP-Based Multi-Agent Coordination)

**From:** `spec.md`
**Timeline:** 2 weeks (10 working days), single sprint
**Team:** 1 Backend Engineer (full-time)
**Created:** 2025-10-25

---

## Summary

- **Total tasks:** 16 story-level tasks (COORD-002 through COORD-017)
- **Estimated effort:** 68 story points (~2 weeks)
- **Critical path duration:** 8-9 working days
- **Key risks:**
  1. Multi-objective optimization edge cases (mitigation: comprehensive testing)
  2. Performance at scale (1K agents, 10K signals/sec) (mitigation: early benchmarking)
  3. Overload prediction accuracy (mitigation: configurable algorithm, validation)

**REP Coordination Targets:**

- 41-100% coordination accuracy improvement vs baseline
- <10ms agent selection latency for 100 candidates
- 90%+ even load distribution across agents
- 80%+ overload prediction accuracy

---

## Phase Breakdown

### Phase 1: Foundation (Days 1-3, 20 SP)

**Goal:** Establish data models, configuration, and core signal management
**Deliverable:** Working signal registration with validation and normalization

#### Tasks

**COORD-002: Data Models and Enums**

- **Description:** Create Pydantic models for SensitivitySignal, AgentCoordinationState, CoordinationMetrics with validation
- **Acceptance:**
  - [ ] SignalType enum (LOAD, CAPACITY, QUALITY, COST, LATENCY, AVAILABILITY)
  - [ ] SensitivitySignal model with all fields (agent_id, signal_type, value 0.0-1.0, timestamp, ttl_seconds, confidence)
  - [ ] AgentCoordinationState model with score fields (load, capacity, quality, cost, availability, routing_score)
  - [ ] CoordinationMetrics model for Prometheus
  - [ ] All models have 100% type coverage (mypy strict)
  - [ ] Pydantic validators for value ranges (0.0-1.0, ttl >0)
- **Effort:** 3 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-001 (epic)
- **Priority:** P0 (Blocker for all development)
- **Files:**
  - `src/agentcore/a2a_protocol/models/coordination.py`

**COORD-003: Configuration Management**

- **Description:** Add coordination service configuration to config.py using Pydantic Settings
- **Acceptance:**
  - [ ] COORDINATION_ENABLE_REP bool (default True)
  - [ ] COORDINATION_SIGNAL_TTL int (default 60 seconds)
  - [ ] COORDINATION_MAX_HISTORY_SIZE int (default 100)
  - [ ] COORDINATION_CLEANUP_INTERVAL int (default 300 seconds)
  - [ ] Routing optimization weights (ROUTING_WEIGHT_LOAD, CAPACITY, QUALITY, COST, AVAILABILITY)
  - [ ] Default weights sum to 1.0 (validation)
  - [ ] Settings loadable from .env file
  - [ ] Example .env.template updated
- **Effort:** 2 story points (0.5-1 day)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-001
- **Priority:** P0
- **Files:**
  - `src/agentcore/a2a_protocol/config.py` (extend existing)
  - `.env.template`

**COORD-004: CoordinationService Core**

- **Description:** Implement CoordinationService class with signal registration, validation, and normalization
- **Acceptance:**
  - [ ] CoordinationService class in coordination_service.py
  - [ ] register_signal(signal) validates and normalizes signal values
  - [ ] Signal validation: value in 0.0-1.0, ttl >0, valid agent_id
  - [ ] Signal normalization ensures consistent 0.0-1.0 range
  - [ ] Signals stored in coordination_states dict
  - [ ] UUID generation for signal_id
  - [ ] Timestamp recorded on registration
  - [ ] Unit tests for validation and normalization (90%+ coverage)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-002, COORD-003
- **Priority:** P0 (Critical path)
- **Files:**
  - `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_signal_registration.py`

**COORD-005: Signal Aggregation Logic**

- **Description:** Implement score computation and aggregation for routing decisions
- **Acceptance:**
  - [ ] compute_individual_scores(agent_id) computes load, capacity, quality, cost, availability scores
  - [ ] Scores derived from most recent signal of each type
  - [ ] Load score inverted (high load = low score)
  - [ ] compute_routing_score(agent_id) computes weighted composite score
  - [ ] Weights applied from configuration
  - [ ] Agents without signals receive default score 0.5
  - [ ] Scores cached in AgentCoordinationState
  - [ ] Unit tests for score computation (100% coverage)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-004
- **Priority:** P0 (Critical path)
- **Files:**
  - Enhanced `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_score_aggregation.py`

**COORD-006: Signal History & TTL Management**

- **Description:** Implement signal history tracking and time-based expiry with temporal decay
- **Acceptance:**
  - [ ] Signal history stored per agent (max 100 signals)
  - [ ] Oldest signals evicted when history full
  - [ ] is_expired(signal) checks TTL against current time
  - [ ] get_active_signals(agent_id) filters expired signals
  - [ ] Temporal decay applied to signal values based on age
  - [ ] Decay formula: value * e^(-age / ttl)
  - [ ] Scores recomputed excluding expired signals
  - [ ] Unit tests for TTL and decay (95%+ coverage)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-005
- **Priority:** P0 (Critical path)
- **Files:**
  - Enhanced `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_signal_ttl.py`

---

### Phase 2: Optimization & Prediction (Days 4-6, 18 SP)

**Goal:** Multi-objective optimization and preemptive overload prediction
**Deliverable:** Optimal agent selection and overload warnings functional

#### Tasks

**COORD-007: Multi-Objective Optimization**

- **Description:** Implement optimal agent selection using weighted multi-objective optimization
- **Acceptance:**
  - [ ] select_optimal_agent(candidates, weights) returns best agent
  - [ ] Retrieves coordination state for each candidate
  - [ ] Applies custom weights if provided, defaults otherwise
  - [ ] Sorts candidates by routing score (descending)
  - [ ] Returns top agent with highest composite score
  - [ ] Selection rationale logged (agent_id, score, breakdown)
  - [ ] Handles empty candidates list gracefully
  - [ ] Unit tests for optimization logic (100% coverage)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-006
- **Priority:** P0 (Critical path - blocks integration)
- **Files:**
  - Enhanced `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_optimization.py`

**COORD-008: Overload Prediction**

- **Description:** Implement trend analysis for preemptive overload prediction
- **Acceptance:**
  - [ ] predict_overload(agent_id, forecast_seconds) returns (will_overload, probability)
  - [ ] Retrieves recent load signals (last 10 from history)
  - [ ] Computes load trend using simple linear regression
  - [ ] Extrapolates load at forecast_seconds ahead
  - [ ] Checks if predicted load exceeds threshold (0.8 default)
  - [ ] Returns probability based on trend confidence
  - [ ] Warning logged if overload predicted
  - [ ] Unit tests for prediction algorithm (90%+ coverage)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-006 (needs signal history)
- **Priority:** P1 (Nice-to-have feature)
- **Files:**
  - Enhanced `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_overload_prediction.py`

**COORD-009: Signal Cleanup Service**

- **Description:** Implement background task for periodic cleanup of expired signals and stale states
- **Acceptance:**
  - [ ] cleanup_expired_signals() removes all expired signals
  - [ ] Iterates all agent coordination states
  - [ ] Removes signals where is_expired() == True
  - [ ] Removes agent states with no active signals
  - [ ] Scores recomputed after cleanup
  - [ ] Cleanup statistics logged (signals removed, agents removed)
  - [ ] Background task runs every COORDINATION_CLEANUP_INTERVAL
  - [ ] Integration tests for cleanup behavior
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-006
- **Priority:** P1
- **Files:**
  - Enhanced `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/integration/test_cleanup.py`

**COORD-010: Unit Test Suite**

- **Description:** Comprehensive unit tests covering all core coordination logic
- **Acceptance:**
  - [ ] Tests for signal validation and normalization
  - [ ] Tests for score computation and aggregation
  - [ ] Tests for optimal agent selection
  - [ ] Tests for signal TTL and expiry
  - [ ] Tests for temporal decay
  - [ ] Tests for overload prediction
  - [ ] Tests for cleanup mechanism
  - [ ] 90%+ code coverage for coordination_service.py
  - [ ] All tests run in <5 seconds
  - [ ] CI pipeline integration
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-007, COORD-008, COORD-009
- **Priority:** P0 (Quality gate)
- **Files:**
  - `tests/coordination/unit/test_coordination_service.py`
  - `tests/coordination/unit/test_models.py`

---

### Phase 3: Integration (Days 7-9, 20 SP)

**Goal:** Integrate with MessageRouter, expose JSON-RPC methods, add metrics
**Deliverable:** Full coordination working end-to-end via A2A protocol

#### Tasks

**COORD-011: MessageRouter Integration**

- **Description:** Add RIPPLE_COORDINATION routing strategy to MessageRouter and integrate with CoordinationService
- **Acceptance:**
  - [ ] RoutingStrategy enum extended with RIPPLE_COORDINATION
  - [ ] _ripple_coordination_select(candidates) method in MessageRouter
  - [ ] Integration with coordination_service.select_optimal_agent()
  - [ ] Fallback to RANDOM routing on coordination errors
  - [ ] Selection logged with trace_id
  - [ ] Metrics updated (coordination_routing_selections_total)
  - [ ] Backward compatibility with existing strategies validated
  - [ ] Integration tests for RIPPLE_COORDINATION strategy
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-007
- **Priority:** P0 (Critical path - major integration)
- **Files:**
  - `src/agentcore/a2a_protocol/services/message_router.py` (modify existing)
  - `tests/coordination/integration/test_message_router.py`

**COORD-012: JSON-RPC Methods**

- **Description:** Expose coordination service via JSON-RPC 2.0 protocol for A2A integration
- **Acceptance:**
  - [ ] coordination.signal handler (register signal from agent)
  - [ ] coordination.state handler (get agent coordination state)
  - [ ] coordination.metrics handler (get current metrics snapshot)
  - [ ] coordination.predict_overload handler (get overload prediction)
  - [ ] All methods registered with jsonrpc_processor
  - [ ] A2A context extraction from JsonRpcRequest
  - [ ] Error mapping to JsonRpcErrorCode
  - [ ] Request/response validation with Pydantic
  - [ ] Unit tests for all handlers
  - [ ] Integration test via JSON-RPC endpoint
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-007
- **Priority:** P0 (A2A integration requirement)
- **Files:**
  - `src/agentcore/a2a_protocol/services/coordination_jsonrpc.py`
  - `tests/coordination/integration/test_jsonrpc.py`

**COORD-013: Integration Tests**

- **Description:** End-to-end integration tests with multiple agents and coordination scenarios
- **Acceptance:**
  - [ ] Multi-agent signal registration and routing
  - [ ] Test RIPPLE_COORDINATION vs RANDOM routing
  - [ ] Test signal expiry and cleanup workflows
  - [ ] Test overload prediction accuracy
  - [ ] Test coordination with 10+ agents
  - [ ] Test concurrent signal registration
  - [ ] Test MessageRouter integration
  - [ ] All integration tests pass consistently (>95% success rate)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-011, COORD-012
- **Priority:** P0 (Quality gate)
- **Files:**
  - `tests/coordination/integration/test_multi_agent.py`
  - `tests/coordination/integration/test_end_to_end.py`

**COORD-014: Prometheus Metrics Instrumentation**

- **Description:** Add comprehensive Prometheus metrics for coordination operations
- **Acceptance:**
  - [ ] coordination_signals_total counter (agent_id, signal_type)
  - [ ] coordination_agents_total gauge (active agents)
  - [ ] coordination_routing_selections_total counter (strategy)
  - [ ] coordination_signal_registration_duration_seconds histogram
  - [ ] coordination_agent_selection_duration_seconds histogram
  - [ ] coordination_overload_predictions_total counter (agent_id, predicted)
  - [ ] Metrics exposed at /metrics endpoint
  - [ ] Metrics updated in real-time
  - [ ] Unit tests for metrics (verify counters increment)
- **Effort:** 4 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-011
- **Priority:** P1 (Observability requirement)
- **Files:**
  - `src/agentcore/a2a_protocol/metrics/coordination_metrics.py`
  - `tests/coordination/unit/test_metrics.py`

---

### Phase 4: Validation (Day 10, 10 SP)

**Goal:** Validate performance, document system, prove effectiveness
**Deliverable:** Production-ready coordination service with validated improvements

#### Tasks

**COORD-015: Performance Benchmarks**

- **Description:** Validate performance SLOs with comprehensive benchmarking
- **Acceptance:**
  - [ ] Benchmark signal registration latency: <5ms (p95) achieved
  - [ ] Benchmark routing score retrieval: <2ms (p95) achieved
  - [ ] Benchmark optimal agent selection: <10ms for 100 candidates (p95) achieved
  - [ ] Load test with 1,000 agents and 10,000 signals/sec
  - [ ] Latency histogram published (p50, p90, p95, p99)
  - [ ] Throughput measurement (signals/second, selections/second)
  - [ ] Resource usage profiling (CPU, memory)
  - [ ] Benchmarking script in scripts/benchmark_coordination.py
  - [ ] Results documented in docs/coordination-performance-report.md
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-013
- **Priority:** P0 (NFR validation)
- **Files:**
  - `tests/coordination/load/test_performance.py`
  - `scripts/benchmark_coordination.py`
  - `docs/coordination-performance-report.md`

**COORD-016: Documentation and Examples**

- **Description:** Comprehensive documentation with usage examples and API reference
- **Acceptance:**
  - [ ] README.md in docs/coordination-service/ with overview
  - [ ] API reference (JSON-RPC methods, parameters, return types)
  - [ ] Usage examples: signal registration, agent selection, overload prediction
  - [ ] Configuration guide (optimization weights, TTL settings)
  - [ ] Architecture diagram (components and data flow)
  - [ ] Troubleshooting guide (common issues and solutions)
  - [ ] REP paper references and coordination theory
  - [ ] Migration guide from baseline routing
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-013
- **Priority:** P1 (Documentation requirement)
- **Files:**
  - `docs/coordination-service/README.md`
  - `docs/coordination-service/api-reference.md`
  - `docs/coordination-service/examples.md`

**COORD-017: Effectiveness Validation**

- **Description:** Validate 41-100% coordination accuracy improvement vs baseline routing
- **Acceptance:**
  - [ ] Test dataset: 100 routing decisions with ground truth
  - [ ] Baseline: RANDOM routing accuracy measured
  - [ ] Coordination: RIPPLE_COORDINATION routing accuracy measured
  - [ ] Improvement: 41-100% accuracy gain validated
  - [ ] Load distribution evenness: 90%+ achieved
  - [ ] Overload prediction accuracy: 80%+ achieved
  - [ ] Effectiveness report with statistical significance
  - [ ] Comparison charts (accuracy, load distribution)
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** COORD-015
- **Priority:** P0 (REP validation requirement)
- **Files:**
  - `tests/coordination/validation/test_effectiveness.py`
  - `docs/coordination-effectiveness-report.md`

---

## Critical Path

```plaintext
Foundation (Days 1-3):
COORD-002 → COORD-003 → COORD-004 → COORD-005 → COORD-006
  (1d)        (0.5d)       (2d)         (2d)         (2d)
                          [7.5 days]

Optimization (Days 4-5):
COORD-007
  (2d)

Integration (Days 6-9):
COORD-011 → COORD-013 → COORD-015
  (3d)        (2d)        (2d)
                [7 days]

Total Critical Path: ~16.5 days (compressed with parallel work to 10 days)
```

**Bottlenecks:**

- **COORD-011 (MessageRouter Integration)**: Largest task (8 SP), complex integration
- **COORD-007 (Multi-Objective Optimization)**: Blocks MessageRouter integration
- **COORD-015 (Performance Benchmarks)**: Must validate all NFRs before production

**Parallel Tracks:**

**Week 1 (Days 1-5):**

- Foundation (Eng1, Days 1-3)
- Optimization + Prediction (Eng1, Days 4-5)

**Week 2 (Days 6-10):**

- MessageRouter Integration (Eng1, Days 6-8) || Unit Tests (Eng1, Days 4-6)
- JSON-RPC Methods (Eng1, Day 9) || Metrics (Eng1, Day 8-9)
- Integration Tests (Eng1, Day 9-10)
- Performance + Docs (Eng1, Day 10)

---

## Quick Wins (Week 1, Days 1-3)

1. **COORD-002 (Data Models)** - Day 1: Unblocks all development
2. **COORD-003 (Configuration)** - Day 1: Enables testing setup
3. **COORD-004 (Service Core)** - Days 2-3: First working signal registration, demonstrates feasibility

**Value:** By end of Day 3, signals can be registered and validated. Demonstrates core feasibility.

---

## Risk Mitigation

| Task | Risk | Impact | Likelihood | Mitigation | Contingency |
|------|------|--------|------------|------------|-------------|
| COORD-007 | Multi-objective optimization edge cases | HIGH | MEDIUM | Comprehensive unit tests, A/B testing | Simplify to 3-factor scoring (load, capacity, quality) |
| COORD-011 | MessageRouter breaking changes | MEDIUM | LOW | Integration tests for all strategies, fallback | Revert and isolate coordination |
| COORD-015 | Performance at scale (1K agents, 10K signals/sec) | MEDIUM | MEDIUM | Early benchmarking, profiling | Redis-backed distributed state (Phase 4) |
| COORD-009 | Signal cleanup timing issues | LOW | MEDIUM | Configurable cleanup interval, signal freshness metrics | Manual cleanup triggers |
| COORD-008 | Overload prediction accuracy <80% | MEDIUM | HIGH | Validation testing, configurable algorithm | Disable prediction, focus on reactive balancing |
| COORD-017 | Effectiveness improvement <41% | CRITICAL | LOW | REP paper validation, synthetic workloads | Document achieved improvement, iterate on weights |

---

## Testing Strategy

### Automated Testing Tasks

- **COORD-010 (Unit Tests)** - 5 SP: Mock all components, 90%+ coverage
- **COORD-013 (Integration Tests)** - 5 SP: Multi-agent scenarios, E2E workflows
- **COORD-015 (Performance Benchmarks)** - 5 SP: Validate all NFRs
- **COORD-017 (Effectiveness Validation)** - 2 SP: REP improvement validation

**Total Testing Effort: 17 SP (25% of project)**

### Quality Gates

- **Unit Tests:** 90%+ code coverage required (enforced by CI)
- **Integration Tests:** All critical paths have E2E tests
- **Performance:** Signal registration <5ms (p95), selection <10ms for 100 agents (p95), throughput 10K signals/sec
- **Effectiveness:** 41-100% coordination accuracy improvement vs baseline

---

## Team Allocation

**Backend Engineer (1 FTE, 100% allocated):**

- Foundation: Data models, config, service core (Days 1-3)
- Optimization: Multi-objective, prediction, cleanup (Days 4-5)
- Integration: MessageRouter, JSON-RPC, metrics (Days 6-9)
- Validation: Performance, documentation, effectiveness (Day 10)

**No additional team required:** This is a focused 2-week project with single engineer.

---

## Sprint Planning

**Single 2-week sprint, 68 SP total**

| Week | Focus | Story Points | Key Deliverables |
|------|-------|--------------|------------------|
| Week 1 (Days 1-5) | Foundation + Optimization | 38 SP | Data models, signal management, optimization logic |
| Week 2 (Days 6-10) | Integration + Validation | 30 SP | MessageRouter integration, metrics, validation complete |

**Sprint 1 Total: 68 SP**

Key Milestones:

- Day 3: Signal registration working
- Day 5: Optimization functional
- Day 8: MessageRouter integration complete
- Day 10: All quality gates passed, production-ready

---

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Phase
COORD-002,Data Models,Create SensitivitySignal and AgentCoordinationState models,3,P0,Backend Eng,,Foundation
COORD-003,Configuration,Add coordination settings to config.py,2,P0,Backend Eng,COORD-002,Foundation
COORD-004,Service Core,Implement signal registration and validation,5,P0,Backend Eng,COORD-002;COORD-003,Foundation
COORD-005,Signal Aggregation,Implement score computation,5,P0,Backend Eng,COORD-004,Foundation
COORD-006,Signal TTL,Implement history and expiry,5,P0,Backend Eng,COORD-005,Foundation
COORD-007,Optimization,Multi-objective agent selection,5,P0,Backend Eng,COORD-006,Optimization
COORD-008,Overload Prediction,Trend analysis and prediction,5,P1,Backend Eng,COORD-006,Optimization
COORD-009,Signal Cleanup,Background cleanup task,3,P1,Backend Eng,COORD-006,Optimization
COORD-010,Unit Tests,Comprehensive unit test suite,5,P0,Backend Eng,COORD-007;COORD-008;COORD-009,Testing
COORD-011,MessageRouter Integration,Add RIPPLE_COORDINATION strategy,8,P0,Backend Eng,COORD-007,Integration
COORD-012,JSON-RPC Methods,Expose coordination via A2A,3,P0,Backend Eng,COORD-007,Integration
COORD-013,Integration Tests,End-to-end coordination tests,5,P0,Backend Eng,COORD-011;COORD-012,Testing
COORD-014,Prometheus Metrics,Add observability metrics,4,P1,Backend Eng,COORD-011,Integration
COORD-015,Performance Benchmarks,Validate NFRs,5,P0,Backend Eng,COORD-013,Validation
COORD-016,Documentation,API docs and examples,3,P1,Backend Eng,COORD-013,Documentation
COORD-017,Effectiveness Validation,Validate 41-100% improvement,2,P0,Backend Eng,COORD-015,Validation
```

---

## Appendix

**Estimation Method:** Fibonacci story points based on complexity and effort
**Story Point Scale:** 1 (trivial), 2 (simple), 3 (moderate), 5 (complex), 8 (very complex), 13 (epic-size)

**Definition of Done:**

- Code implemented and reviewed
- Unit tests written and passing (90%+ coverage)
- Integration tests passing (where applicable)
- Documentation updated (inline docstrings + API docs)
- Deployed to staging environment
- Performance validated (if performance-critical task)
- REP targets met (if applicable)

**REP Validation Commitment:**

All tasks contributing to REP coordination effectiveness (COORD-007, COORD-011, COORD-017) will be validated against benchmarks:

- Coordination accuracy improvement measured on synthetic workloads
- Load distribution evenness validated with multiple agents
- Overload prediction accuracy validated on historical load patterns
- Performance measured against NFR targets (<5ms signal registration, <10ms selection)

---

**Document Status:** ✅ Ready for Ticket Generation
**Next Steps:** Generate story tickets (COORD-002 through COORD-017) in `.sage/tickets/`
