# Tasks: ACE Integration (COMPASS-Enhanced Meta-Thinker)

**From:** `spec.md` v2.0 + `plan.md` v2.0 (COMPASS-Enhanced)
**Timeline:** 8 weeks, 4 sprints (2-week sprints)
**Team:** 1 senior backend engineer (full-time)
**Created:** 2025-10-23

---

## Summary

- **Total tasks:** 32 story tasks (ACE-002 through ACE-033)
- **Estimated effort:** 175 story points (~8 weeks)
- **Critical path duration:** 8 weeks (sequential phases with MEM dependency)
- **Key risks:**
  1. MEM dependency timing (Phase 4 requires MEM Phase 5, Week 8)
  2. Intervention decision accuracy (85%+ precision target)
  3. Real-time performance overhead (<5% target)

**COMPASS Targets:**
- +20% long-horizon accuracy improvement
- 90%+ critical error recall
- 85%+ intervention precision
- <200ms ACE-MEM coordination latency
- <5% system overhead

---

## Phase Breakdown

### Phase 1: Foundation + Original ACE Core (Sprint 1, Weeks 1-2, 45 SP)

**Goal:** Establish database schema, models, and context playbook management
**Deliverable:** Working ACE foundation with playbook evolution capability

#### Week 1: Database + Models (18 SP)

**ACE-002: Create COMPASS-Enhanced Database Migration**

- **Description:** Implement Alembic migration for 6 ACE tables including TimescaleDB hypertable for performance metrics
- **Acceptance:**
  - [ ] TimescaleDB extension added to PostgreSQL
  - [ ] All 6 tables created (context_playbooks, playbook_deltas, delta_approvals, performance_metrics, intervention_history, capability_evaluations)
  - [ ] TimescaleDB hypertable configured for performance_metrics (time-series optimized)
  - [ ] Composite indexes on (agent_id, task_id, stage) for metrics queries
  - [ ] GIN index on playbook_deltas.delta_content (JSONB)
  - [ ] Migration reversible (downgrade tested)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-001 (epic)
- **Priority:** P0 (Blocker for all phases)
- **Files:**
  - `alembic/versions/XXX_add_ace_tables.py`
  - `docker-compose.dev.yml` (add TimescaleDB extension)

**ACE-003: Implement COMPASS Pydantic Models**

- **Description:** Create Pydantic models for StageMetrics, TriggerSignal, InterventionDecision, ContextPlaybook with COMPASS enhancements
- **Acceptance:**
  - [ ] StageMetrics model with stage enum validation (planning, execution, reflection, verification)
  - [ ] TriggerSignal model with trigger_type enum (degradation, error_accumulation, staleness, capability_mismatch)
  - [ ] InterventionDecision model with intervention_type enum (replan, reflect, context_refresh, capability_switch)
  - [ ] ContextPlaybook model with self-evolving structure
  - [ ] All models support JSON serialization
  - [ ] Modern typing (use `list[]`, `dict[]`, `|` unions)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-002
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/models.py`

**ACE-004: Implement SQLAlchemy ORM Models**

- **Description:** Create SQLAlchemy models matching database schema with async support and TimescaleDB support
- **Acceptance:**
  - [ ] ContextPlaybookModel with JSONB for sections
  - [ ] PlaybookDeltaModel with foreign key to playbooks
  - [ ] PerformanceMetricsModel as TimescaleDB hypertable
  - [ ] InterventionHistoryModel with JSONB for trigger details
  - [ ] All models use AsyncSession
  - [ ] Relationships configured (playbook → deltas, intervention → metrics)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-003
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/database/models.py`

#### Week 2: Playbook Management + Delta Generation (27 SP)

**ACE-005: Implement Repository Layer**

- **Description:** Create async repositories for Playbook, Delta, Metrics, Intervention with COMPASS query patterns
- **Acceptance:**
  - [ ] PlaybookRepository with get_by_agent_id()
  - [ ] DeltaRepository with get_pending_deltas()
  - [ ] MetricsRepository with time-series queries (last_hour, last_day)
  - [ ] InterventionRepository with get_history()
  - [ ] All methods use async/await
  - [ ] Unit tests for each repository (90%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-004
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/database/repositories.py`
  - `tests/ace/unit/test_repositories.py`

**ACE-006: Implement Playbook Manager**

- **Description:** Create PlaybookManager for context playbook CRUD operations and version management
- **Acceptance:**
  - [ ] create_playbook(agent_id, initial_sections) creates new playbook
  - [ ] get_playbook(agent_id) retrieves current version
  - [ ] apply_deltas(playbook_id, deltas) updates playbook with version increment
  - [ ] Optimistic locking for version conflicts
  - [ ] Playbook versioning tracked
  - [ ] Unit tests for playbook lifecycle (95%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-005
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/context/playbook.py`
  - `tests/ace/unit/test_playbook_manager.py`

**ACE-007: Implement Delta Generator**

- **Description:** Create DeltaGenerator using gpt-4o-mini to generate improvement suggestions from execution traces
- **Acceptance:**
  - [ ] generate_deltas(execution_trace) returns list of PlaybookDelta
  - [ ] Uses gpt-4o-mini (cost-effective for suggestions)
  - [ ] LLM prompts extract learnings from traces
  - [ ] Confidence scores computed per delta
  - [ ] Delta generation latency <5s (p95)
  - [ ] Integration tests with mocked Portkey
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-006
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/context/delta_generator.py`
  - `tests/ace/integration/test_delta_generation.py`

**ACE-008: Implement Simple Curator**

- **Description:** Create SimpleCurator with confidence-threshold filtering for delta approval
- **Acceptance:**
  - [ ] filter_deltas(deltas, threshold=0.8) returns approved deltas
  - [ ] Confidence threshold configurable
  - [ ] Approval/rejection logic based on confidence score
  - [ ] Rejected deltas logged with rationale
  - [ ] Unit tests for curation logic
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-007
- **Priority:** P1
- **Files:**
  - `src/agentcore/ace/context/curator.py`
  - `tests/ace/unit/test_curator.py`

---

### Phase 2: Performance Monitoring (COMPASS ACE-1) (Sprint 2, Week 3, 30 SP)

**Goal:** Implement stage-aware performance tracking and baseline computation
**Deliverable:** Real-time performance monitoring with degradation detection

**ACE-009: Implement PerformanceMonitor Core**

- **Description:** Create PerformanceMonitor service for stage-aware metrics tracking
- **Acceptance:**
  - [ ] update_metrics(agent_id, task_id, stage, metrics) records metrics
  - [ ] Metrics stored in TimescaleDB hypertable
  - [ ] Stage-specific metrics tracked (planning, execution, reflection, verification)
  - [ ] Metrics update latency <50ms (p95)
  - [ ] Metrics batching (buffer 100 updates or 1 second)
  - [ ] Unit tests for monitoring logic (95%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-008
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/monitors/performance_monitor.py`
  - `tests/ace/unit/test_performance_monitor.py`

**ACE-010: Implement Baseline Tracker**

- **Description:** Add baseline computation and drift detection for performance metrics
- **Acceptance:**
  - [ ] compute_baseline(agent_id, stage) from first 10 executions
  - [ ] Rolling baseline updated every 50 executions
  - [ ] Baseline comparison for degradation detection
  - [ ] Baseline drift detection with statistical significance
  - [ ] Baseline reset mechanism for major agent updates
  - [ ] Unit tests for baseline algorithms
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-009
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/monitors/baseline_tracker.py`
  - `tests/ace/unit/test_baseline_tracker.py`

**ACE-011: Implement Error Accumulator**

- **Description:** Add error accumulation tracking and pattern detection
- **Acceptance:**
  - [ ] track_error(agent_id, task_id, stage, error) records errors
  - [ ] Error count per stage computed
  - [ ] Error severity distribution tracked
  - [ ] Compounding error detection (related errors in sequence)
  - [ ] Integration with MEM error pattern detection (future)
  - [ ] Unit tests for accumulation logic
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-010
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/monitors/error_accumulator.py`
  - `tests/ace/unit/test_error_accumulator.py`

**ACE-012: Implement Metrics API (JSON-RPC Handlers)**

- **Description:** Register JSON-RPC methods for ACE performance monitoring
- **Acceptance:**
  - [ ] ace.track_performance(metrics) records metrics
  - [ ] ace.get_baseline(agent_id, stage) retrieves baseline
  - [ ] ace.get_metrics_summary(agent_id, task_id) returns summary
  - [ ] A2A context (agent_id, task_id) handled correctly
  - [ ] JSON-RPC error handling for invalid params
  - [ ] Integration tests for API
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-011
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/jsonrpc.py`
  - `tests/ace/integration/test_metrics_jsonrpc.py`

**ACE-013: Integrate Metrics Dashboard**

- **Description:** Add Prometheus metrics and Grafana dashboard integration
- **Acceptance:**
  - [ ] Prometheus metrics exposed (latency, error rate, throughput)
  - [ ] Grafana dashboard configured for ACE metrics
  - [ ] Metrics retention: 90 days (TimescaleDB compression)
  - [ ] Dashboard shows stage-specific performance
  - [ ] Alert rules configured (degradation thresholds)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer + DevOps
- **Dependencies:** ACE-012
- **Priority:** P1
- **Files:**
  - `prometheus/ace_metrics.yml`
  - `grafana/ace_dashboard.json`

**ACE-014: Integration Tests for Performance Monitoring**

- **Description:** Comprehensive integration tests for monitoring workflows
- **Acceptance:**
  - [ ] End-to-end metrics recording tested
  - [ ] Baseline computation validated
  - [ ] Error accumulation workflows tested
  - [ ] API integration tested
  - [ ] Performance tests validate <50ms latency
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-013
- **Priority:** P0
- **Files:**
  - `tests/ace/integration/test_monitoring_workflows.py`

---

### Phase 3: Strategic Intervention Engine (COMPASS ACE-2) (Sprint 2, Week 4, 35 SP)

**Goal:** Implement trigger detection and intervention orchestration
**Deliverable:** Strategic intervention engine with 4 trigger types

**ACE-015: Implement InterventionEngine Core**

- **Description:** Create InterventionEngine for intervention orchestration
- **Acceptance:**
  - [ ] process_trigger(trigger_signal) orchestrates intervention
  - [ ] Intervention queue management (priority-based)
  - [ ] Intervention execution tracking
  - [ ] Intervention failure handling
  - [ ] Intervention history persisted
  - [ ] Unit tests for engine logic (95%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-014
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/intervention/engine.py`
  - `tests/ace/unit/test_intervention_engine.py`

**ACE-016: Implement Trigger Detection**

- **Description:** Add trigger detection for 4 signal types (degradation, error_accumulation, staleness, capability_mismatch)
- **Acceptance:**
  - [ ] detect_degradation(metrics, baseline) returns TriggerSignal
  - [ ] detect_error_accumulation(errors) detects compounding errors
  - [ ] detect_staleness(context_age, retrieval_quality) detects stale context
  - [ ] detect_capability_mismatch(task_requirements, agent_capabilities)
  - [ ] Trigger detection latency <50ms
  - [ ] False positive rate <15%
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-015
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/intervention/triggers.py`
  - `tests/ace/unit/test_trigger_detection.py`

**ACE-017: Implement Intervention Decision Making**

- **Description:** Add decision logic using gpt-4.1 for intervention type selection
- **Acceptance:**
  - [ ] decide_intervention(trigger, strategic_context) returns InterventionDecision
  - [ ] Uses gpt-4.1 for strategic decisions (accuracy-critical)
  - [ ] Decision includes rationale and expected impact
  - [ ] Decision latency <200ms (p95)
  - [ ] Decision accuracy 85%+ (validated against ground truth)
  - [ ] Integration tests with mocked Portkey
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-016
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/intervention/decision.py`
  - `tests/ace/integration/test_intervention_decision.py`

**ACE-018: Implement Intervention Executor**

- **Description:** Create executor for sending intervention commands to Agent Runtime
- **Acceptance:**
  - [ ] execute_intervention(agent_id, task_id, intervention_type, context)
  - [ ] Agent Runtime integration (intervention commands)
  - [ ] Intervention execution non-blocking
  - [ ] Intervention failures handled gracefully
  - [ ] Intervention effectiveness measured
  - [ ] Integration tests with mocked runtime
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-017
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/intervention/executor.py`
  - `tests/ace/integration/test_intervention_executor.py`

**ACE-019: Integrate with Agent Runtime**

- **Description:** Add Agent Runtime interface for intervention command support
- **Acceptance:**
  - [ ] Agent Runtime accepts replan, reflect, context_refresh, capability_switch commands
  - [ ] Intervention state tracked in runtime
  - [ ] Intervention outcomes reported back to ACE
  - [ ] Runtime integration tested
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-018
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/integration/runtime_interface.py`
  - `tests/ace/integration/test_runtime_integration.py`

**ACE-020: Integration Tests for Intervention Workflows**

- **Description:** End-to-end tests for intervention workflows
- **Acceptance:**
  - [ ] Full intervention workflow tested (trigger → decision → execution)
  - [ ] All 4 trigger types tested
  - [ ] All 4 intervention types tested
  - [ ] Intervention precision validated (85%+ target)
  - [ ] Performance validated (<200ms latency)
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-019
- **Priority:** P0
- **Files:**
  - `tests/ace/integration/test_intervention_workflows.py`

---

### Phase 4: ACE-MEM Integration Layer (COMPASS ACE-3) (Sprint 3, Week 5, 25 SP)

**Goal:** Coordinate with MEM for strategic context queries
**Deliverable:** ACE-MEM coordination with intervention outcome tracking

**ACE-021: Implement MEM Integration Layer**

- **Description:** Create ACEMemoryInterface for strategic context queries to MEM
- **Acceptance:**
  - [ ] get_strategic_context(query_type, agent_id, task_id, context) queries MEM
  - [ ] Query types: strategic_decision, error_analysis, capability_evaluation, context_refresh
  - [ ] Strategic context includes: stage summaries, critical facts, error patterns, successful patterns
  - [ ] Context health score computed (0-1)
  - [ ] Query latency <150ms (p95)
  - [ ] Unit tests for interface (90%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-020, MEM Phase 5 (Week 8) - use mock until then
- **Priority:** P0
- **Files:**
  - `src/agentcore/ace/integration/mem_interface.py`
  - `tests/ace/unit/test_mem_interface.py`

**ACE-022: Implement Strategic Context Queries**

- **Description:** Add query logic for 4 strategic context types
- **Acceptance:**
  - [ ] query_for_strategic_decision(trigger) gets decision context
  - [ ] query_for_error_analysis(errors) gets error patterns
  - [ ] query_for_capability_evaluation(task) gets performance data
  - [ ] query_for_context_refresh(task_id) gets latest compressed context
  - [ ] Query results include relevance scores
  - [ ] Query failures degrade gracefully
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-021
- **Priority:** P0
- **Files:**
  - Enhanced `src/agentcore/ace/integration/mem_interface.py`
  - `tests/ace/integration/test_strategic_queries.py`

**ACE-023: Implement Intervention Outcome Tracking**

- **Description:** Track intervention effectiveness for learning
- **Acceptance:**
  - [ ] record_intervention_outcome(intervention_id, success, delta)
  - [ ] Before-intervention metrics captured
  - [ ] After-intervention metrics captured
  - [ ] Delta computation (improvement or degradation quantified)
  - [ ] Learning updates intervention thresholds
  - [ ] Integration with MEM for outcome storage
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-022
- **Priority:** P1
- **Files:**
  - `src/agentcore/ace/integration/outcome_tracker.py`
  - `tests/ace/unit/test_outcome_tracker.py`

**ACE-024: Cross-Component Integration Tests**

- **Description:** Comprehensive tests for ACE-MEM-Runtime coordination
- **Acceptance:**
  - [ ] Full workflow tested: trigger → MEM query → decision → execution → outcome
  - [ ] MEM query integration validated
  - [ ] Outcome tracking validated
  - [ ] Coordination latency <200ms validated
  - [ ] Error handling tested (MEM unavailable, timeout)
- **Effort:** 4 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-023
- **Priority:** P0
- **Files:**
  - `tests/ace/integration/test_cross_component.py`

---

### Phase 5: Capability Evaluation (COMPASS ACE-4) (Sprint 3, Week 6, 20 SP)

**Goal:** Assess task-capability fitness and recommend changes
**Deliverable:** Dynamic capability evaluation and recommendation engine

**ACE-025: Implement CapabilityEvaluator**

- **Description:** Create CapabilityEvaluator service for task-capability fitness scoring
- **Acceptance:**
  - [ ] evaluate_fitness(agent_id, task_requirements) returns fitness score
  - [ ] Task-capability matching algorithm
  - [ ] Fitness score computation (0-1 scale)
  - [ ] Capability gap identification
  - [ ] Fitness evaluation latency <100ms
  - [ ] Unit tests for evaluator (95%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-024
- **Priority:** P1
- **Files:**
  - `src/agentcore/ace/capability/evaluator.py`
  - `tests/ace/unit/test_capability_evaluator.py`

**ACE-026: Implement Fitness Scoring**

- **Description:** Add multi-factor fitness scoring algorithm
- **Acceptance:**
  - [ ] Capability coverage score (task requirements met)
  - [ ] Performance history score (success rate on similar tasks)
  - [ ] Resource efficiency score (time/cost per task)
  - [ ] Combined fitness score with weighted factors
  - [ ] Fitness trends tracked over time
  - [ ] Unit tests for scoring algorithms
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-025
- **Priority:** P1
- **Files:**
  - Enhanced `src/agentcore/ace/capability/evaluator.py`
  - `tests/ace/unit/test_fitness_scoring.py`

**ACE-027: Implement Recommendation Engine**

- **Description:** Create recommendation engine for capability changes
- **Acceptance:**
  - [ ] recommend_capability_changes(agent_id, fitness_score, gaps)
  - [ ] Recommendations include: add capabilities, remove capabilities, adjust parameters
  - [ ] Recommendation confidence scores
  - [ ] Recommendation rationale with evidence
  - [ ] JSON-RPC API for capability evaluation
  - [ ] Integration tests for recommendations
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-026
- **Priority:** P1
- **Files:**
  - `src/agentcore/ace/capability/recommender.py`
  - Enhanced `src/agentcore/ace/jsonrpc.py`
  - `tests/ace/integration/test_capability_recommendations.py`

**ACE-028: Tests for Capability Evaluation**

- **Description:** Comprehensive tests for capability evaluation workflows
- **Acceptance:**
  - [ ] Fitness scoring validated
  - [ ] Recommendation accuracy tested
  - [ ] API integration tested
  - [ ] Edge cases covered (no capabilities, perfect fit, etc.)
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-027
- **Priority:** P1
- **Files:**
  - `tests/ace/integration/test_capability_workflows.py`

---

### Phase 6: Production Readiness (Sprint 4, Weeks 7-8, 20 SP)

**Goal:** Optimize, validate, and prepare for production deployment
**Deliverable:** Production-ready ACE system with COMPASS validation

**ACE-029: Performance Tuning**

- **Description:** Optimize ACE system for production performance targets
- **Acceptance:**
  - [ ] Metrics batching optimized (100 updates or 1 second)
  - [ ] Redis caching for playbooks (10min TTL), baselines (1hr TTL)
  - [ ] TimescaleDB compression configured (90-day retention)
  - [ ] Connection pooling tuned (min 10, max 50)
  - [ ] System overhead <5% validated
  - [ ] Performance benchmarks documented
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-028
- **Priority:** P1
- **Files:**
  - Database tuning scripts
  - `tests/ace/load/test_performance_tuning.py`

**ACE-030: COMPASS Validation Tests**

- **Description:** Validate against COMPASS benchmarks (accuracy, recall, precision, cost)
- **Acceptance:**
  - [ ] Long-horizon accuracy: +20% improvement on GAIA-style tasks
  - [ ] Critical error recall: 90%+ on test dataset
  - [ ] Intervention precision: 85%+ correct intervention rate
  - [ ] Cost reduction: Within $150/month budget (100 agents)
  - [ ] All COMPASS targets met or documented deviations
  - [ ] Validation report with statistical analysis
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-029
- **Priority:** P0
- **Files:**
  - `tests/ace/validation/test_compass_benchmarks.py`
  - `docs/ace-compass-validation-report.md`

**ACE-031: Load Testing**

- **Description:** Load test ACE with 100 concurrent agents and 1000 tasks
- **Acceptance:**
  - [ ] 100 concurrent agents without errors
  - [ ] 1000 tasks processed successfully
  - [ ] Intervention latency <200ms (p95)
  - [ ] System overhead <5%
  - [ ] No resource exhaustion (memory, CPU, connections)
  - [ ] Load test report generated
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-030
- **Priority:** P1
- **Files:**
  - `tests/ace/load/test_ace_load.py`
  - `docs/ace-load-test-report.md`

**ACE-032: Set Up Monitoring and Alerting**

- **Description:** Configure production monitoring and alerting
- **Acceptance:**
  - [ ] Prometheus metrics: latency, precision, recall, overhead, MEM query latency
  - [ ] Grafana dashboards: ACE system health, intervention analytics, COMPASS metrics
  - [ ] Alerting rules: precision <80%, overhead >7%, MEM failures >5%, recall <85%
  - [ ] Metric retention: 90 days (TimescaleDB)
  - [ ] Alert routing to on-call engineer
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer + DevOps
- **Dependencies:** ACE-031
- **Priority:** P1
- **Files:**
  - Enhanced `prometheus/ace_metrics.yml`
  - Enhanced `grafana/ace_dashboard.json`

**ACE-033: Write Operational Documentation**

- **Description:** Create runbook and API documentation for operations and developers
- **Acceptance:**
  - [ ] Runbook: deployment, troubleshooting, common issues
  - [ ] API docs: JSON-RPC methods, examples, error codes
  - [ ] Architecture diagram: ACE-MEM-Runtime interactions
  - [ ] Configuration guide: all settings explained
  - [ ] COMPASS validation results included
  - [ ] Production readiness checklist completed
- **Effort:** 4 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-032
- **Priority:** P0
- **Files:**
  - `docs/ace-system-runbook.md`
  - `docs/ace-api.md`
  - `docs/ace-architecture.md`

---

## Critical Path

```plaintext
Foundation (Week 1-2):
ACE-002 → ACE-003 → ACE-004 → ACE-005 → ACE-006 → ACE-007 → ACE-008
  (3d)     (2d)       (2d)       (3d)       (3d)       (3d)       (1d)
                                   [17 days]

Performance Monitoring (Week 3):
ACE-009 → ACE-010 → ACE-011 → ACE-012 → ACE-013 → ACE-014
  (3d)      (2d)      (2d)       (2d)       (2d)       (1d)
                              [12 days]

Strategic Intervention (Week 4):
ACE-015 → ACE-016 → ACE-017 → ACE-018 → ACE-019 → ACE-020
  (3d)      (3d)      (3d)       (2d)       (1d)       (1d)
                              [13 days]

ACE-MEM Integration (Week 5, depends on MEM Week 8):
ACE-021 → ACE-022 → ACE-023 → ACE-024
  (3d)      (3d)      (2d)       (1d)
              [9 days]

Capability Evaluation (Week 6):
ACE-025 → ACE-026 → ACE-027 → ACE-028
  (3d)      (2d)      (2d)       (1d)
              [8 days]

Production Readiness (Week 7-8):
ACE-029 → ACE-030 → ACE-031 → ACE-032 → ACE-033
  (2d)      (2d)      (1d)       (1d)       (1d)
                    [7 days]

Total Critical Path: ~66 days (~8 weeks with some parallel work)
```

**Bottlenecks:**

- **ACE-021 (MEM Integration)**: Blocks on MEM Phase 5 (Week 8) - use mock interface until then
- **ACE-017 (Intervention Decision)**: Complex LLM integration (8 SP, highest risk)
- **ACE-030 (COMPASS Validation)**: External benchmark dependency (5 SP)

**Parallel Tracks:**

- **Testing**: Unit tests can be written concurrently with implementation
- **Documentation**: API docs can be written as features complete (ACE-033)
- **Monitoring**: Metrics setup happens alongside optimization (ACE-032)

---

## Quick Wins (Week 1-2)

1. **ACE-002 (Database Migration)** - Unblocks all development
2. **ACE-003/004 (Models)** - Enables parallel repository work
3. **ACE-005 (Repositories)** - Core data access ready early

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| ACE-021 | MEM dependency timing | Mock MEM interface for dev (Weeks 5-7), real integration Week 8 | Defer ACE Phase 4 to Week 9 if MEM delayed |
| ACE-017 | Intervention decision accuracy | A/B testing, threshold tuning, ground truth validation | Lower precision target to 80%, increase human review |
| ACE-009 | Performance overhead | Async processing, metrics batching, caching | Reduce monitoring frequency, sample 50% of actions |
| ACE-030 | COMPASS validation fails | Iterative tuning, benchmark dataset preparation | Document partial achievement (e.g., +15% instead of +20%) |

---

## Testing Strategy

### Automated Testing Tasks

- **ACE-005, ACE-006, ACE-009, etc.** - Unit tests embedded in each story (target 95% coverage)
- **ACE-014, ACE-020, ACE-024** - Integration tests for component interactions
- **ACE-029, ACE-031** - Load tests for performance validation
- **ACE-030** - COMPASS validation tests (benchmark suite)

### Quality Gates

- **90%+ test coverage** for all ACE components
- **All critical paths have integration tests** (intervention workflows, ACE-MEM coordination)
- **Performance tests validate SLOs** (<200ms intervention latency, <5% overhead)
- **COMPASS benchmarks met** (+20% accuracy, 90%+ recall, 85%+ precision)

---

## Team Allocation

**Backend Engineer (1 FTE):**

- Database and models (Week 1)
- Playbook management and delta generation (Week 2)
- Performance monitoring (Week 3)
- Strategic intervention engine (Week 4)
- ACE-MEM integration (Week 5, with mock until Week 8)
- Capability evaluation (Week 6)
- Production readiness (Weeks 7-8)

**DevOps Support (0.2 FTE):**

- TimescaleDB setup (Week 1)
- Prometheus/Grafana configuration (Weeks 3, 8)
- Production deployment (Week 8)

---

## Sprint Planning

**2-week sprints, 44 SP velocity (1 engineer)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 (Week 1-2) | Foundation + Playbook Evolution | 45 SP | Database, models, playbook manager, delta generator |
| Sprint 2 (Week 3-4) | Monitoring + Intervention Engine | 65 SP | Performance monitoring, trigger detection, intervention execution |
| Sprint 3 (Week 5-6) | ACE-MEM Integration + Capability | 45 SP | MEM interface, outcome tracking, capability evaluation |
| Sprint 4 (Week 7-8) | Production Readiness | 20 SP | Tuning, COMPASS validation, monitoring, documentation |

**Total: 175 SP over 8 weeks**

---

## Appendix

**Estimation Method:** Fibonacci story points based on complexity and effort
**Story Point Scale:** 1 (trivial), 2 (simple), 3 (moderate), 5 (complex), 8 (very complex), 13 (epic-size)

**Definition of Done:**

- Code implemented and reviewed
- Unit tests written (90%+ coverage for task)
- Integration tests passing
- Documentation updated (inline + API docs)
- Deployed to staging environment
- Performance validated (if applicable)
- COMPASS targets met (if applicable)

**COMPASS Validation Commitment:**

All tasks contributing to COMPASS targets (ACE-009 through ACE-030) will be validated against benchmarks:
- Long-horizon accuracy measured on GAIA-style evaluation dataset
- Critical error recall validated on annotated error test set
- Intervention precision validated via expert review and A/B testing
- Cost tracked via Portkey token usage monitoring

---

**Document Status:** ✅ Ready for Ticket Generation
**Next Steps:** Generate story tickets (ACE-002 through ACE-033) in `.sage/tickets/`
