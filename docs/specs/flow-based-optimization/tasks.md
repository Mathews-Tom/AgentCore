# Tasks: Flow-Based Optimization Engine

**From:** `spec.md` + `plan.md`
**Timeline:** 8 weeks, 4 sprints (2-week sprints)
**Team:** 2 Backend Engineers
**Created:** 2025-10-15

---

## Summary

- **Total tasks:** 17 story tickets
- **Estimated effort:** 83 story points
- **Critical path duration:** 6-7 weeks (46 SP on critical path)
- **Key risks:**
  1. Integration complexity with Agent Runtime
  2. Budget enforcement bypass vulnerabilities
  3. Database scaling under high trajectory volume

---

## Phase Breakdown

### Phase 1: Core Infrastructure (Sprint 1-2, 41 Story Points)

**Goal:** Implement foundational training infrastructure with trajectory collection and GRPO algorithm
**Deliverable:** End-to-end training job execution (single iteration)

#### Tasks

**[FLOW-002] Database Schema & Models**

- **Description:** Create PostgreSQL tables (training_jobs, trajectories, policy_checkpoints) with Alembic migrations; implement Pydantic models and repository pattern
- **Acceptance:**
  - [ ] Alembic migrations run successfully (`uv run alembic upgrade head`)
  - [ ] Pydantic models validate correctly (GRPOConfig, Trajectory, TrainingJob)
  - [ ] Repository classes implement CRUD operations
  - [ ] Database indexes created for performance (job_id, agent_id, created_at)
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** None (foundation task)
- **Priority:** P0 (Blocker)
- **Files:**
  - `alembic/versions/xxx_add_training_tables.py`
  - `src/agentcore/training/models.py`
  - `src/agentcore/training/repositories.py`
  - `tests/training/unit/test_models.py`

---

**[FLOW-003] Trajectory Collector Implementation**

- **Description:** Implement async parallel trajectory generation (8 concurrent) with middleware wrapper around agent execution
- **Acceptance:**
  - [ ] Generate 8 trajectories in parallel for test query
  - [ ] Complete within 2x baseline execution time (validated)
  - [ ] Capture complete execution state (states, actions, results, timestamps)
  - [ ] Handle execution failures gracefully (timeout, errors)
  - [ ] Integration with Agent Runtime successful
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-002
- **Priority:** P0 (Critical)
- **Files:**
  - `src/agentcore/training/trajectory.py`
  - `src/agentcore/training/middleware/trajectory_recorder.py`
  - `tests/training/unit/test_trajectory_collector.py`
  - `tests/training/integration/test_agent_execution.py`

---

**[FLOW-004] Reward Engine**

- **Description:** Implement reward computation with outcome-based and shaped reward functions; support reward normalization
- **Acceptance:**
  - [ ] Outcome-based rewards computed correctly (success/failure)
  - [ ] Shaped rewards applied (tool usage +0.1, verification +0.05, length -0.01)
  - [ ] Reward normalization using group statistics (mean, std)
  - [ ] Custom reward function registry working
  - [ ] Edge case handled: std_reward == 0
  - [ ] Unit tests achieve 100% coverage (critical path)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-003
- **Priority:** P0 (Critical)
- **Files:**
  - `src/agentcore/training/rewards.py`
  - `tests/training/unit/test_reward_engine.py`

---

**[FLOW-005] GRPO Trainer Core**

- **Description:** Implement GRPO algorithm with policy gradient updates, advantage computation, and training iteration loop
- **Acceptance:**
  - [ ] Policy gradient calculation correct (`loss = -log_prob * advantage`)
  - [ ] Advantage computation using normalized rewards
  - [ ] Training iteration loop executes successfully
  - [ ] Update only positive-advantage trajectories
  - [ ] Track training loss and convergence metrics
  - [ ] Gradient clipping implemented for stability
  - [ ] Unit tests achieve 100% coverage (critical algorithm)
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-004
- **Priority:** P0 (Critical)
- **Files:**
  - `src/agentcore/training/grpo.py`
  - `tests/training/unit/test_grpo_trainer.py`

---

**[FLOW-006] Policy Updater**

- **Description:** Implement prompt-based policy updates with LLM integration via Portkey; update logic for positive-advantage trajectories
- **Acceptance:**
  - [ ] Extract successful patterns from trajectories
  - [ ] Update agent context/prompts based on patterns
  - [ ] Portkey AI integration working (LLM calls)
  - [ ] Update weighted by advantage (higher advantage = stronger update)
  - [ ] Support checkpoint creation after updates
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-005
- **Priority:** P0 (Critical)
- **Files:**
  - `src/agentcore/training/policy.py`
  - `tests/training/unit/test_policy_updater.py`
  - `tests/training/integration/test_portkey_integration.py`

---

**[FLOW-007] Training Job Manager**

- **Description:** Implement job lifecycle management (create, start, cancel) with Redis job queue and background worker pattern
- **Acceptance:**
  - [ ] Create training jobs with configuration validation
  - [ ] Enqueue jobs to Redis `training:jobs` queue
  - [ ] Background worker consumes jobs from queue
  - [ ] Job status tracking in Redis (queued, running, completed, failed, cancelled)
  - [ ] Support job cancellation mid-execution
  - [ ] Checkpoint resumption after worker restart
  - [ ] Integration tests validate full lifecycle
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-002, FLOW-005
- **Priority:** P0 (Critical)
- **Files:**
  - `src/agentcore/training/job_manager.py`
  - `src/agentcore/training/worker.py`
  - `tests/training/integration/test_job_lifecycle.py`

---

**[FLOW-008] Training API Endpoints**

- **Description:** Implement JSON-RPC endpoints (training.start_grpo, training.get_status, training.cancel) with registration and validation
- **Acceptance:**
  - [ ] `training.start_grpo` endpoint creates jobs successfully
  - [ ] `training.get_status` endpoint returns real-time status
  - [ ] `training.cancel` endpoint cancels running jobs
  - [ ] Request/response validation with Pydantic
  - [ ] JWT authentication required (existing security layer)
  - [ ] RBAC authorization enforced (training:start, training:view)
  - [ ] API tests validate all endpoints
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-007
- **Priority:** P0 (Critical)
- **Files:**
  - `src/agentcore/training/training_jsonrpc.py`
  - `tests/training/integration/test_training_api.py`

---

**[FLOW-009] Integration Tests Phase 1**

- **Description:** End-to-end integration tests for Phase 1 deliverables (training job, trajectory generation, API)
- **Acceptance:**
  - [ ] End-to-end training job test (create → execute → complete)
  - [ ] Parallel trajectory generation performance test (<30s for 8 trajectories)
  - [ ] API endpoint integration tests (all endpoints)
  - [ ] Budget enforcement test (simulate exceeding budget)
  - [ ] Worker crash recovery test (resume from checkpoint)
  - [ ] All tests pass with 90%+ coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Both Engineers
- **Dependencies:** FLOW-008
- **Priority:** P0 (Quality gate)
- **Files:**
  - `tests/training/integration/test_end_to_end.py`
  - `tests/training/integration/test_performance.py`
  - `tests/training/integration/test_recovery.py`

---

### Phase 2: Evaluation & Monitoring (Sprint 3, 24 Story Points)

**Goal:** Add evaluation framework, budget enforcement, checkpoints, and monitoring
**Deliverable:** Production-ready training system with cost controls and metrics

#### Tasks

**[FLOW-010] Evaluation Framework**

- **Description:** Implement held-out query evaluation with metrics computation and statistical significance testing
- **Acceptance:**
  - [ ] Evaluation on held-out queries (20% of training data)
  - [ ] Metrics computed: success_rate, avg_reward, avg_steps, tool_accuracy
  - [ ] Baseline comparison (trained vs untrained agent)
  - [ ] Statistical significance testing (p-values, t-tests)
  - [ ] Evaluation runs every N iterations (default: 10)
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-003
- **Priority:** P1 (High)
- **Files:**
  - `src/agentcore/training/evaluation.py`
  - `tests/training/unit/test_evaluation.py`

---

**[FLOW-011] Budget Enforcement**

- **Description:** Implement budget tracking and enforcement with Portkey cost API integration
- **Acceptance:**
  - [ ] Pre-flight budget check before each trajectory batch
  - [ ] Real-time cost monitoring via Portkey API
  - [ ] Abort training when budget exceeded
  - [ ] Budget decorator for cost-sensitive operations
  - [ ] Alert notifications at 75%, 90% thresholds
  - [ ] Unit tests validate budget enforcement
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-007
- **Priority:** P0 (Critical - cost control)
- **Files:**
  - `src/agentcore/training/utils/budget.py`
  - `tests/training/unit/test_budget_enforcement.py`

---

**[FLOW-012] Checkpoint Manager**

- **Description:** Implement save/restore checkpoints with versioning and best-checkpoint selection
- **Acceptance:**
  - [ ] Save checkpoints every N iterations (default: 10)
  - [ ] Store policy parameters, iteration, metrics, optimizer state
  - [ ] Hybrid storage (PostgreSQL metadata + S3 for large weights)
  - [ ] Load checkpoint for resume after interruption
  - [ ] Best checkpoint selection by validation score
  - [ ] Automatic cleanup (keep best 5 checkpoints)
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-005
- **Priority:** P1 (High)
- **Files:**
  - `src/agentcore/training/checkpoint.py`
  - `tests/training/unit/test_checkpoint_manager.py`

---

**[FLOW-013] Prometheus Metrics**

- **Description:** Implement Prometheus metrics export for training jobs, performance, and budget
- **Acceptance:**
  - [ ] Training job metrics (created, completed, failed)
  - [ ] Performance metrics (trajectory_generation_duration, policy_update_duration)
  - [ ] Budget metrics (training_budget_usage)
  - [ ] Metrics exported to Prometheus endpoint
  - [ ] Grafana dashboard configuration documented
  - [ ] Integration test validates metrics collection
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-007
- **Priority:** P1 (High)
- **Files:**
  - `src/agentcore/training/metrics.py`
  - `docs/monitoring/training_metrics.md`
  - `tests/training/integration/test_metrics.py`

---

**[FLOW-014] Data Export API**

- **Description:** Implement training.export_trajectories endpoint with filtering and pagination
- **Acceptance:**
  - [ ] Export trajectories for job with filters (success_only, min_reward)
  - [ ] Pagination support (limit, offset)
  - [ ] JSON export format with trajectory details
  - [ ] Authorization check (data_export permission required)
  - [ ] Size limit enforcement (max 10,000 trajectories)
  - [ ] API tests validate export functionality
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-002
- **Priority:** P1 (Medium)
- **Files:**
  - `src/agentcore/training/training_jsonrpc.py` (extend)
  - `tests/training/integration/test_export_api.py`

---

**[FLOW-019] Integration Tests Phase 2**

- **Description:** Integration tests for Phase 2 deliverables (evaluation, budget, checkpoints, metrics)
- **Acceptance:**
  - [ ] Evaluation framework test (held-out validation)
  - [ ] Budget enforcement test (abort on exceed)
  - [ ] Checkpoint save/restore test (recovery)
  - [ ] Metrics export test (Prometheus integration)
  - [ ] Export API test (data export with filters)
  - [ ] All tests pass with 90%+ coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Both Engineers
- **Dependencies:** FLOW-010, FLOW-011, FLOW-012, FLOW-013, FLOW-014
- **Priority:** P1 (Quality gate)
- **Files:**
  - `tests/training/integration/test_evaluation_e2e.py`
  - `tests/training/integration/test_budget_e2e.py`
  - `tests/training/integration/test_checkpoint_recovery.py`

---

### Phase 3: Advanced Features (Sprint 4, 16 Story Points)

**Goal:** Add advanced training features and complete documentation
**Deliverable:** Production-ready system with advanced capabilities and user documentation

#### Tasks

**[FLOW-015] Multi-Step Credit Assignment**

- **Description:** Implement temporal difference rewards with per-step advantage computation
- **Acceptance:**
  - [ ] Discount factor (gamma=0.99) implemented
  - [ ] Step-wise reward computation (`final_reward * gamma^(n-i-1)`)
  - [ ] Per-step advantage calculation
  - [ ] Integration with GRPO trainer
  - [ ] Convergence speed improvement validated (benchmark test)
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-005
- **Priority:** P1 (Medium)
- **Files:**
  - `src/agentcore/training/grpo.py` (extend)
  - `src/agentcore/training/credit_assignment.py`
  - `tests/training/unit/test_credit_assignment.py`

---

**[FLOW-016] Training Job Scheduling**

- **Description:** Implement job queue prioritization, worker pool management, and auto-scaling support
- **Acceptance:**
  - [ ] Job queue prioritization (P0, P1, P2)
  - [ ] Worker pool management (start, stop, scale)
  - [ ] Auto-scaling support (Kubernetes HPA integration)
  - [ ] Handle 100+ concurrent jobs (load test)
  - [ ] Worker health checks and auto-restart
  - [ ] Integration tests validate scheduling
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-007
- **Priority:** P1 (Medium)
- **Files:**
  - `src/agentcore/training/scheduler.py`
  - `k8s/training-worker-hpa.yaml`
  - `tests/training/performance/test_concurrent_jobs.py`

---

**[FLOW-017] Advanced Reward Shaping**

- **Description:** Implement custom reward function registry and configurable reward strategies
- **Acceptance:**
  - [ ] Custom reward function registry
  - [ ] Configurable reward strategies per agent type
  - [ ] Reward function validation (ensure output range [0, 1])
  - [ ] Example reward functions (domain-specific)
  - [ ] Documentation for creating custom rewards
  - [ ] Unit tests achieve 95% coverage
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer 1
- **Dependencies:** FLOW-004
- **Priority:** P2 (Low)
- **Files:**
  - `src/agentcore/training/rewards.py` (extend)
  - `src/agentcore/training/reward_registry.py`
  - `docs/guides/custom_rewards.md`
  - `tests/training/unit/test_reward_registry.py`

---

**[FLOW-018] Documentation & Guides**

- **Description:** Create comprehensive API documentation, developer guide, and operational runbook
- **Acceptance:**
  - [ ] API documentation (all JSON-RPC endpoints documented)
  - [ ] Developer guide (how to start training jobs, interpret metrics)
  - [ ] Operational runbook (deployment, monitoring, troubleshooting)
  - [ ] Code examples (Python SDK usage)
  - [ ] Architecture diagrams (mermaid)
  - [ ] User acceptance: 3+ developers successfully use documentation
- **Effort:** 5 story points (3-5 days)
- **Owner:** Both Engineers
- **Dependencies:** All previous tasks
- **Priority:** P0 (Blocker for launch)
- **Files:**
  - `docs/api/training-api.md`
  - `docs/guides/training-agents.md`
  - `docs/ops/training-runbook.md`
  - `examples/training/simple_training_job.py`

---

**[FLOW-020] Performance & Load Testing**

- **Description:** Comprehensive performance and load testing to validate SLA targets
- **Acceptance:**
  - [ ] Load test: 100+ concurrent training jobs
  - [ ] Latency test: Trajectory generation <2x baseline (p95)
  - [ ] Throughput test: 8 trajectories in <30s (p95)
  - [ ] Database performance: >100 trajectory writes/sec
  - [ ] API response time: training.get_status <200ms (p95)
  - [ ] All SLA targets met
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer 2
- **Dependencies:** FLOW-019
- **Priority:** P0 (Quality gate)
- **Files:**
  - `tests/training/performance/test_load.py`
  - `tests/training/performance/locustfile.py`
  - `docs/performance/benchmark_results.md`

---

## Critical Path

```plaintext
FLOW-002 → FLOW-003 → FLOW-004 → FLOW-005 → FLOW-006 → FLOW-007 → FLOW-008 → FLOW-009
  (5 SP)    (8 SP)      (5 SP)      (8 SP)      (5 SP)      (5 SP)      (5 SP)      (5 SP)

Total Critical Path: 46 Story Points (~6-7 weeks)
```

**Bottlenecks:**

- **FLOW-005 (GRPO Trainer)**: Most complex algorithm, highest risk (8 SP)
- **FLOW-003 (Trajectory Collector)**: Agent Runtime integration complexity (8 SP)

**Parallel Tracks:**

- **Phase 2:** FLOW-010, FLOW-011, FLOW-012, FLOW-013, FLOW-014 can run in parallel
- **Phase 3:** FLOW-015, FLOW-016, FLOW-017 can run in parallel before FLOW-018

---

## Quick Wins (Sprint 1)

1. **[FLOW-002] Database Schema** - Unblocks all other tasks
2. **[FLOW-011] Budget Enforcement** - Early cost control validation (can start in Sprint 2)
3. **[FLOW-013] Metrics** - Early observability (can start in Sprint 2)

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| FLOW-003 | Agent Runtime integration breaks | Comprehensive integration tests, feature flags | Fallback to mock agent execution |
| FLOW-005 | GRPO algorithm bugs | 100% unit test coverage, mathematical validation | Expert consultation, reference implementation |
| FLOW-007 | Redis queue reliability | Job state persisted to PostgreSQL, health checks | Fallback to database polling |
| FLOW-011 | Budget enforcement bypass | Security review, penetration testing | Emergency budget kill-switch |
| FLOW-012 | Checkpoint storage growth | S3 integration for large checkpoints, cleanup policy | Archive old checkpoints to cold storage |

---

## Testing Strategy

### Automated Testing Tasks

- **[FLOW-009] Integration Tests Phase 1** (5 SP) - Sprint 2
- **[FLOW-019] Integration Tests Phase 2** (5 SP) - Sprint 3
- **[FLOW-020] Performance & Load Testing** (3 SP) - Sprint 4

### Quality Gates

- **90% code coverage** required for all modules
- **100% coverage** for critical paths (GRPO algorithm, reward computation, budget enforcement)
- **All integration tests pass** before phase completion
- **Performance tests validate SLAs** before production deployment

---

## Team Allocation

**Backend Engineer 1 (Focus: Data & Rewards)**

- Sprint 1: FLOW-002 (Database), FLOW-004 (Rewards), FLOW-006 (Policy)
- Sprint 2: FLOW-008 (API), FLOW-009 (Tests)
- Sprint 3: FLOW-011 (Budget), FLOW-013 (Metrics), FLOW-014 (Export)
- Sprint 4: FLOW-015 (Credit), FLOW-017 (Reward Shaping), FLOW-018 (Docs)

**Backend Engineer 2 (Focus: Training & Infrastructure)**

- Sprint 1: FLOW-003 (Trajectory), FLOW-005 (GRPO Trainer)
- Sprint 2: FLOW-007 (Job Manager), FLOW-009 (Tests)
- Sprint 3: FLOW-010 (Evaluation), FLOW-012 (Checkpoints), FLOW-019 (Tests)
- Sprint 4: FLOW-016 (Scheduling), FLOW-018 (Docs), FLOW-020 (Performance)

**Shared Responsibilities:**

- Integration tests (FLOW-009, FLOW-019)
- Documentation (FLOW-018)
- Performance testing (FLOW-020)
- Code reviews and pair programming

---

## Sprint Planning

**2-week sprints, ~21 SP velocity per engineer (42 SP team velocity)**

| Sprint | Focus | Story Points | Key Deliverables | Story Tickets |
|--------|-------|--------------|------------------|---------------|
| **Sprint 1** | Foundation | 41 SP | Database, Trajectory Collection, Rewards, GRPO Core | FLOW-002, FLOW-003, FLOW-004, FLOW-005 |
| **Sprint 2** | Training Loop | 15 SP | Policy Updates, Job Manager, API, Integration Tests | FLOW-006, FLOW-007, FLOW-008, FLOW-009 |
| **Sprint 3** | Production Readiness | 24 SP | Evaluation, Budget, Checkpoints, Metrics, Export | FLOW-010, FLOW-011, FLOW-012, FLOW-013, FLOW-014, FLOW-019 |
| **Sprint 4** | Advanced & Docs | 19 SP | Credit Assignment, Scheduling, Reward Shaping, Docs, Load Tests | FLOW-015, FLOW-016, FLOW-017, FLOW-018, FLOW-020 |

**Total Velocity:** 99 SP across 4 sprints (planned), 83 SP estimated (buffer: 16% for unknowns)

---

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint
FLOW-002,Database Schema & Models,Create PostgreSQL tables and Pydantic models,5,P0,Backend-1,,1
FLOW-003,Trajectory Collector,Implement async parallel trajectory generation,8,P0,Backend-2,FLOW-002,1
FLOW-004,Reward Engine,Implement reward computation and normalization,5,P0,Backend-1,FLOW-003,1
FLOW-005,GRPO Trainer Core,Implement GRPO algorithm and policy gradients,8,P0,Backend-2,FLOW-004,1
FLOW-006,Policy Updater,Implement prompt-based policy updates,5,P0,Backend-1,FLOW-005,2
FLOW-007,Training Job Manager,Implement job lifecycle and Redis queue,5,P0,Backend-2,"FLOW-002,FLOW-005",2
FLOW-008,Training API Endpoints,Implement JSON-RPC training endpoints,5,P0,Backend-1,FLOW-007,2
FLOW-009,Integration Tests Phase 1,End-to-end integration tests,5,P0,Both,FLOW-008,2
FLOW-010,Evaluation Framework,Implement held-out evaluation and metrics,5,P1,Backend-2,FLOW-003,3
FLOW-011,Budget Enforcement,Implement budget tracking via Portkey,3,P0,Backend-1,FLOW-007,3
FLOW-012,Checkpoint Manager,Implement checkpoint save/restore,5,P1,Backend-2,FLOW-005,3
FLOW-013,Prometheus Metrics,Implement metrics export,3,P1,Backend-1,FLOW-007,3
FLOW-014,Data Export API,Implement trajectory export endpoint,3,P1,Backend-1,FLOW-002,3
FLOW-019,Integration Tests Phase 2,Integration tests for Phase 2,5,P1,Both,"FLOW-010,FLOW-011,FLOW-012,FLOW-013,FLOW-014",3
FLOW-015,Multi-Step Credit Assignment,Implement temporal difference rewards,5,P1,Backend-1,FLOW-005,4
FLOW-016,Training Job Scheduling,Implement job prioritization and scaling,3,P1,Backend-2,FLOW-007,4
FLOW-017,Advanced Reward Shaping,Implement custom reward registry,3,P2,Backend-1,FLOW-004,4
FLOW-018,Documentation & Guides,Create API docs and user guides,5,P0,Both,All,4
FLOW-020,Performance & Load Testing,Comprehensive performance validation,3,P0,Backend-2,FLOW-019,4
```

---

## Appendix

### Estimation Method

**Planning Poker** with team using Fibonacci scale

### Story Point Scale

Fibonacci: 1, 2, 3, 5, 8, 13, 21

**Calibration:**
- **1 SP**: Simple configuration change, documentation update
- **3 SP**: Single component implementation, straightforward logic
- **5 SP**: Complex component, multiple files, integration required
- **8 SP**: Core algorithm, high complexity, critical path
- **13 SP**: Large feature, multiple components, high risk

### Definition of Done

- [ ] Code written and reviewed (1+ approver)
- [ ] Unit tests written and passing (90%+ coverage for module)
- [ ] Integration tests passing (where applicable)
- [ ] Documentation updated (API docs, code comments)
- [ ] Deployed to staging environment
- [ ] QA validation complete
- [ ] No critical bugs or security vulnerabilities

### Velocity Tracking

**Historical Velocity** (from similar projects):
- Backend Python development: 20-25 SP per engineer per sprint
- ML/Algorithm implementation: 15-20 SP per engineer per sprint (complexity factor)
- Assumed velocity: **21 SP per engineer per sprint** (conservative)

**Buffer:** 20% buffer included in total estimates for unknowns (83 SP planned vs 99 SP capacity)

---

**Document Status:** Ready for story ticket generation and sprint planning.
