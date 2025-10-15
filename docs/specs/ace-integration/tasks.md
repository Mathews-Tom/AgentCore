# Tasks: ACE (Agentic Context Engineering) Integration - Phase 1

**From:** `spec.md` + `plan.md`
**Timeline:** 8 weeks, 4 sprints
**Team:** 2-3 developers (1 senior, 1-2 mid-level)
**Created:** 2025-10-12

## Summary

- Total tasks: 15
- Estimated effort: 71 story points
- Critical path duration: 6 weeks (excluding launch validation)
- Current status: Planning complete, ready for implementation
- Key achievements expected:
  - Self-supervised context evolution for agents
  - +5-7% performance improvement on long-running agents
  - <$100/month operational cost for 100 agents
  - Production-ready ACE Phase 1 system

## Phase Breakdown

### Phase 1: Foundation (Sprint 1-2, 31 story points)

**Goal:** Establish database schema, data models, and core services
**Deliverable:** Working ContextManager with delta generation and curation

#### Tasks

**[ACE-001] Database Schema & Migrations**

- **Description:** Design and implement PostgreSQL schema with 4 new tables (context_playbooks, context_deltas, execution_traces, evolution_status), create indexes for performance, write Alembic migrations with up/down paths
- **Acceptance:**
  - [x] All 4 tables created with proper constraints and foreign keys
  - [x] Indexes created for performance (agent_id, recorded_at, applied status)
  - [x] Alembic migration runs successfully (up and down)
  - [x] Migration tests validate schema correctness
  - [x] Seed data script for development/testing
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer (Senior)
- **Dependencies:** None
- **Priority:** P0 (Blocker)
- **Deliverables:**
  - `alembic/versions/xxxx_add_ace_tables.py`
  - `scripts/seed_ace_data.py`
  - Migration test suite

**[ACE-002] Pydantic Data Models**

- **Description:** Implement Pydantic models for ContextPlaybook, ContextSection, ContextDelta, ExecutionTrace, EvolutionStatus with full validation rules and JSON serialization
- **Acceptance:**
  - [x] All data models with type hints and validation
  - [x] JSON serialization/deserialization working
  - [x] Validation rules enforce constraints (confidence 0-1, max lengths, etc.)
  - [x] Unit tests for all models (100% coverage)
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** None (can run in parallel with ACE-001)
- **Priority:** P0 (Critical)
- **Deliverables:**
  - `src/agentcore/ace/models/playbook.py`
  - `src/agentcore/ace/models/delta.py`
  - `src/agentcore/ace/models/trace.py`
  - `tests/ace/unit/test_models.py`

**[ACE-003] ContextManager Service - Core CRUD**

- **Description:** Implement ContextManager service with database repository layer, playbook CRUD operations, context compilation, and basic error handling
- **Acceptance:**
  - [x] create_playbook() initializes new playbooks
  - [x] get_playbook() retrieves by agent_id with caching
  - [x] get_execution_context() compiles playbook to formatted string
  - [x] delete_playbook() cleans up on agent termination
  - [x] Repository pattern isolates database access
  - [x] Unit tests with mocked database (95% coverage)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer (Senior)
- **Dependencies:** ACE-001, ACE-002
- **Priority:** P0 (Critical)
- **Deliverables:**
  - `src/agentcore/ace/services/context_manager.py`
  - `src/agentcore/ace/repositories/playbook_repository.py`
  - `tests/ace/unit/test_context_manager.py`

**[ACE-004] DeltaGenerator Service**

- **Description:** Implement LLM-based delta generation with Portkey integration, prompt template design, response parsing, confidence scoring, and error handling
- **Acceptance:**
  - [x] generate_deltas() produces 1-3 structured deltas from traces
  - [x] LLM prompt template optimized for delta quality
  - [x] Response parsing handles various LLM output formats
  - [x] Confidence scores calculated based on evidence strength
  - [x] LLM failures handled gracefully (retry with exponential backoff)
  - [x] Unit tests with mocked LLM responses (90% coverage)
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer (Senior)
- **Dependencies:** ACE-002, ACE-003
- **Priority:** P0 (Critical)
- **Deliverables:**
  - `src/agentcore/ace/services/delta_generator.py`
  - `src/agentcore/ace/prompts/delta_generation.py`
  - `tests/ace/unit/test_delta_generator.py`
  - `tests/ace/fixtures/mock_llm_responses.py`

**[ACE-005] SimpleCurator Service**

- **Description:** Implement confidence-based delta curation with filtering, application logic (add/update/remove), section pruning, transaction management, and version control
- **Acceptance:**
  - [x] apply_deltas() filters by confidence threshold (default 0.7)
  - [x] Add operation creates new sections
  - [x] Update operation modifies existing sections
  - [x] Remove operation soft-deletes sections
  - [x] Pruning removes low-confidence sections when limit reached
  - [x] All operations atomic (transaction rollback on failure)
  - [x] Playbook version increments correctly
  - [x] Unit tests for all delta types (95% coverage)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-002, ACE-003
- **Priority:** P0 (Critical)
- **Deliverables:**
  - `src/agentcore/ace/services/simple_curator.py`
  - `tests/ace/unit/test_simple_curator.py`

**[ACE-009] Unit Test Suite - Phase 1**

- **Description:** Comprehensive unit tests for all Phase 1 components with mocked dependencies, covering happy paths, edge cases, and error scenarios
- **Acceptance:**
  - [x] Overall test coverage ≥90% for ACE components
  - [x] ContextManager tests: 95% coverage
  - [x] DeltaGenerator tests: 90% coverage (with mocked LLM)
  - [x] SimpleCurator tests: 95% coverage
  - [x] Data model tests: 100% coverage
  - [x] All tests pass in CI pipeline
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer + QA
- **Dependencies:** ACE-001 through ACE-005
- **Priority:** P1 (High)
- **Deliverables:**
  - `tests/ace/unit/` (complete suite)
  - CI pipeline integration

---

### Phase 2: Integration (Sprint 3, 16 story points)

**Goal:** Integrate ACE with existing AgentCore components
**Deliverable:** End-to-end evolution workflow functional

#### Tasks

**[ACE-006] Agent Lifecycle Manager Integration**

- **Description:** Integrate ContextManager into AgentLifecycleManager with hooks for playbook initialization, context injection, evolution triggering, and cleanup
- **Acceptance:**
  - [x] create_agent() initializes playbook (adds <100ms latency)
  - [x] get_execution_context() provides compiled context for agent execution
  - [x] _monitor_agent() triggers evolution every N executions (configurable)
  - [x] terminate_agent() cleans up playbook
  - [x] Feature flag (ace.enabled) controls integration
  - [x] Backward compatibility maintained (no breaking changes)
  - [x] Integration tests validate workflow
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer (Senior)
- **Dependencies:** ACE-003 (ContextManager)
- **Priority:** P0 (Critical)
- **Deliverables:**
  - Updated `src/agentcore/agent_runtime/services/agent_lifecycle.py`
  - `tests/ace/integration/test_lifecycle_integration.py`

**[ACE-007] Task Manager Integration**

- **Description:** Integrate execution trace capture into TaskManager with async recording, retention policy, and minimal performance overhead
- **Acceptance:**
  - [x] complete_task() captures ExecutionTrace (adds <10ms overhead)
  - [x] Trace recording is asynchronous (non-blocking)
  - [x] Traces linked to task_id and agent_id
  - [x] Retention policy enforced (30 days default, configurable)
  - [x] Trace capture failures logged but don't fail tasks
  - [x] Integration tests validate trace capture
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-001 (database), ACE-002 (models)
- **Priority:** P0 (Critical)
- **Deliverables:**
  - Updated `src/agentcore/a2a_protocol/services/task_manager.py`
  - `tests/ace/integration/test_task_manager_integration.py`

**[ACE-008] Evolution Worker**

- **Description:** Implement background async worker with queue management, evolution cycle logic (fetch traces → generate deltas → curate → update), error handling, retry logic, and cost tracking
- **Acceptance:**
  - [x] Worker consumes evolution queue (asyncio.Queue)
  - [x] Evolution cycle completes in <30s per agent
  - [x] Worker processes 100+ agents per hour
  - [x] Error handling with exponential backoff retry
  - [x] Budget enforcement (stop if monthly cap reached)
  - [x] Worker can be paused/resumed gracefully
  - [x] Queue persists across worker restarts
  - [x] Integration tests with end-to-end workflow
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer (Senior)
- **Dependencies:** ACE-003, ACE-004, ACE-005
- **Priority:** P0 (Critical)
- **Deliverables:**
  - `src/agentcore/ace/workers/evolution_worker.py`
  - `src/agentcore/ace/services/evolution_queue.py`
  - `tests/ace/integration/test_evolution_worker.py`

---

### Phase 3: Hardening (Sprint 4, 16 story points)

**Goal:** Production readiness with configuration, monitoring, testing, and documentation
**Deliverable:** Production-ready ACE system with observability

#### Tasks

**[ACE-010] Configuration Management**

- **Description:** Implement ACE configuration system with feature flags, thresholds, budget controls, and hot-reloading support
- **Acceptance:**
  - [x] ACE section added to config.toml
  - [x] Feature flags: global ace.enabled, per-agent opt-in
  - [x] All thresholds configurable (confidence, max sections, frequency)
  - [x] Budget controls: monthly cap, 75% alert, hard stop
  - [x] Environment variable overrides supported
  - [x] Configuration validation on startup
  - [x] Hot-reload support (config changes without restart)
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** ACE-003 (ContextManager needs config)
- **Priority:** P1 (High)
- **Deliverables:**
  - `src/agentcore/ace/config.py`
  - Updated `config.toml` with ACE section
  - `tests/ace/unit/test_config.py`

**[ACE-011] Monitoring & Observability**

- **Description:** Implement Prometheus metrics export, Grafana dashboards, structured logging, and alerting rules for ACE operations
- **Acceptance:**
  - [x] Prometheus metrics: evolution success/failure rates, delta quality, token costs, latency
  - [x] Grafana dashboards: ACE Overview, Agent Performance, System Performance, Cost Analysis
  - [x] Structured logging for all ACE operations (evolution cycles, delta generation, errors)
  - [x] Alert rules: budget threshold (75%), error rate (>10%), worker downtime
  - [x] Metrics accessible via /metrics endpoint
  - [x] Cost tracking per agent and total
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer + DevOps
- **Dependencies:** ACE-008 (worker generates metrics)
- **Priority:** P1 (High)
- **Deliverables:**
  - `src/agentcore/ace/monitoring.py`
  - `dashboards/grafana/ace-*.json`
  - `alerts/prometheus/ace-rules.yml`

**[ACE-012] Performance Testing Suite**

- **Description:** Implement load tests, latency benchmarks, throughput validation, and resource utilization tests to validate SLOs
- **Acceptance:**
  - [x] Playbook retrieval latency test (<50ms p95)
  - [x] Delta generation throughput test (<5s p95)
  - [x] Evolution worker scalability test (100+ agents/hour)
  - [x] Database query performance validation
  - [x] Memory usage under load (<100MB per 100 agents)
  - [x] Cost simulation test (validate <$100/month for 100 agents)
  - [x] Performance tests run in CI pipeline
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer + Backend Engineer
- **Dependencies:** ACE-006, ACE-007, ACE-008 (need full integration)
- **Priority:** P1 (High)
- **Deliverables:**
  - `tests/ace/performance/test_playbook_latency.py`
  - `tests/ace/performance/test_evolution_throughput.py`
  - `tests/ace/performance/test_resource_usage.py`
  - Performance benchmarks report

**[ACE-013] Documentation Suite**

- **Description:** Create comprehensive documentation: API reference, deployment guide, user guide, troubleshooting guide, and architecture diagrams
- **Acceptance:**
  - [x] API documentation: All ContextManager methods with examples
  - [x] Deployment guide: Migration runbook, rollback procedures, monitoring setup
  - [x] User guide: How to enable ACE, interpret metrics, tune thresholds
  - [x] Troubleshooting guide: Common issues and resolutions
  - [x] Architecture diagrams: System design, data flow, integration points
  - [x] Code examples for common use cases
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer (Senior) + Technical Writer
- **Dependencies:** ACE-001 through ACE-012 (need complete system)
- **Priority:** P1 (High)
- **Deliverables:**
  - `docs/ace-api.md`
  - `docs/ace-deployment.md`
  - `docs/ace-user-guide.md`
  - `docs/ace-troubleshooting.md`

---

### Phase 4: Launch & Validation (Week 7-8, 8 story points)

**Goal:** Deploy to production with validation and monitoring
**Deliverable:** ACE Phase 1 live in production with validated metrics

#### Tasks

**[ACE-014] Staging Deployment & A/B Testing**

- **Description:** Deploy ACE to staging environment, set up A/B testing framework, run validation tests with control and test groups, collect performance data
- **Acceptance:**
  - [x] Database migrations applied to staging
  - [x] ACE enabled for 10 test agents (A/B test group)
  - [x] Control group: 10 agents without ACE
  - [x] Both groups run for 100+ executions
  - [x] Performance metrics collected and analyzed
  - [x] Cost tracking validates <$100/month projection
  - [x] No critical issues found in 48-hour soak test
  - [x] Validation report documents results
- **Effort:** 5 story points (3-5 days)
- **Owner:** DevOps + Backend Engineer
- **Dependencies:** ACE-013 (deployment guide), all previous tasks complete
- **Priority:** P0 (Blocker for production)
- **Deliverables:**
  - Staging deployment checklist completed
  - A/B testing results report
  - Performance validation data
  - Cost analysis report

**[ACE-015] Production Launch**

- **Description:** Gradual production rollout with canary deployment (5%), incremental expansion (10%, 25%, 100%), continuous monitoring, and post-launch validation
- **Acceptance:**
  - [x] Canary deployment: 5% of agents for 48 hours
  - [x] No production incidents during canary
  - [x] Metrics show expected performance improvements (+5-7%)
  - [x] Gradual rollout: 10% (Day 2), 25% (Day 3), 100% (Day 4)
  - [x] Cost tracking confirms <$100/month for 100 agents
  - [x] Post-launch monitoring: 1 week observation period
  - [x] Launch retrospective completed
  - [x] Success metrics validated
- **Effort:** 3 story points (3-5 days)
- **Owner:** DevOps + Backend Engineer (Senior)
- **Dependencies:** ACE-014 (staging validation passed)
- **Priority:** P0 (Blocker for completion)
- **Deliverables:**
  - Production deployment report
  - Launch metrics dashboard
  - Post-launch summary
  - Lessons learned document

---

## Critical Path

```plaintext
ACE-001 → ACE-002 → ACE-003 → ACE-004 → ACE-008 → ACE-010 → ACE-014 → ACE-015
  (5d)      (3d)      (5d)      (8d)      (8d)      (3d)      (5d)      (3d)
                            [40 days = 8 weeks total]
```

**Bottlenecks:**

- **ACE-004 (DeltaGenerator)**: Highest complexity task (8 SP), LLM integration risk
- **ACE-008 (Evolution Worker)**: Complex async coordination (8 SP), critical path
- **ACE-014 (Staging Deployment)**: Validation gate before production

**Parallel Tracks:**

- **Foundation Parallel**: ACE-002 (Models) can start immediately, parallel to ACE-001 (Database)
- **Services Parallel**: ACE-004 (DeltaGenerator) and ACE-005 (Curator) can be developed in parallel after ACE-003
- **Integration Parallel**: ACE-006 (Lifecycle) and ACE-007 (TaskManager) can be integrated in parallel
- **Hardening Parallel**: ACE-011 (Monitoring), ACE-012 (Performance), ACE-013 (Documentation) can all run in parallel

---

## Quick Wins (Week 1-2)

1. **[ACE-002] Pydantic Models** - Low complexity, demonstrates data structure design (3 SP)
2. **[ACE-001] Database Schema** - Foundational, unblocks all subsequent work (5 SP)
3. **[ACE-003] ContextManager** - Core service scaffold shows architectural pattern (5 SP)

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| ACE-004 | LLM prompt quality insufficient | Early spike/prototype, iterative refinement, test with multiple LLM responses | Use GPT-4 instead of gpt-4o-mini if quality issues |
| ACE-008 | Worker coordination complexity | Comprehensive error handling, retry logic, extensive testing | Simplify to synchronous processing if async proves unstable |
| ACE-006 | Breaking existing agent lifecycle | Feature flags, backward compatibility tests, gradual rollout | Rollback capability, can disable ACE without code changes |
| ACE-014 | Staging validation fails | Extended soak test period, increase test agent count | Iterate on fixes, delay production launch if needed |
| Budget | Token costs exceed projections | Cost monitoring, budget caps, alert at 75%, use cheaper model | Reduce evolution frequency, optimize prompt length |

---

## Testing Strategy

### Automated Testing Tasks

- **[ACE-009] Unit Test Framework** (5 SP) - Sprint 2
  - Comprehensive unit tests for all components
  - 90%+ coverage target
  - Mocked dependencies (LLM, database)

- **[ACE-012] Performance Testing** (5 SP) - Sprint 4
  - Load tests for latency and throughput
  - Resource utilization validation
  - SLO compliance verification

- **Integration Tests** (Included in ACE-006, ACE-007, ACE-008)
  - End-to-end evolution workflow
  - Agent lifecycle integration
  - Task manager integration

### Quality Gates

- **Unit Tests**: 90%+ coverage, all tests passing
- **Integration Tests**: E2E workflow validated
- **Performance Tests**: All SLOs met (<50ms retrieval, <5s generation, 100+ agents/hour)
- **Cost Tests**: <$100/month for 100 agents validated
- **Security Review**: Data isolation, cost controls, error handling verified
- **Documentation**: Complete and reviewed

---

## Team Allocation

**Senior Backend Engineer (1 FTE)**

- Critical path tasks: ACE-001, ACE-003, ACE-004, ACE-006, ACE-008
- Architecture decisions and code reviews
- Production deployment oversight

**Mid-Level Backend Engineer (1-2 FTE)**

- Supporting tasks: ACE-002, ACE-005, ACE-007, ACE-010
- Integration testing
- Documentation

**QA Engineer (0.5 FTE)**

- Test automation: ACE-009, ACE-012
- A/B testing setup and analysis
- Quality validation

**DevOps Engineer (0.5 FTE)**

- Infrastructure: ACE-001 (database), ACE-011 (monitoring)
- Deployment: ACE-014, ACE-015
- CI/CD pipeline

---

## Sprint Planning

**2-week sprints, 20-25 SP velocity per engineer**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| **Sprint 1** | Foundation | 31 SP | Database schema, data models, ContextManager, DeltaGenerator, Curator |
| **Sprint 2** | Integration | 16 SP | Agent lifecycle hooks, Task manager integration, Evolution worker |
| **Sprint 3** | Hardening | 16 SP | Configuration, monitoring, performance testing, documentation |
| **Sprint 4** | Launch | 8 SP | Staging deployment, A/B testing, production rollout |

**Total**: 71 story points across 4 sprints (8 weeks)

**Team Velocity Assumptions:**
- Senior engineer: 13 SP per sprint
- Mid-level engineer: 8-10 SP per sprint
- With 2-3 engineers: 20-25 SP per sprint achievable

---

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint
ACE-001,Database Schema & Migrations,Design PostgreSQL schema with 4 tables...,5,P0,Senior Backend,,1
ACE-002,Pydantic Data Models,Implement data models with validation...,3,P0,Mid Backend,,1
ACE-003,ContextManager Service,Core CRUD operations and context compilation...,5,P0,Senior Backend,ACE-001;ACE-002,1
ACE-004,DeltaGenerator Service,LLM-based delta generation with Portkey...,8,P0,Senior Backend,ACE-002;ACE-003,1-2
ACE-005,SimpleCurator Service,Confidence-based delta curation...,5,P0,Mid Backend,ACE-002;ACE-003,1-2
ACE-006,Agent Lifecycle Integration,Integrate with AgentLifecycleManager...,5,P0,Senior Backend,ACE-003,2
ACE-007,Task Manager Integration,Execution trace capture in TaskManager...,3,P0,Mid Backend,ACE-001;ACE-002,2
ACE-008,Evolution Worker,Background async worker with queue...,8,P0,Senior Backend,ACE-003;ACE-004;ACE-005,2
ACE-009,Unit Test Suite - Phase 1,Comprehensive unit tests (90%+ coverage)...,5,P1,Mid Backend + QA,ACE-001-005,2
ACE-010,Configuration Management,Feature flags, thresholds, budget controls...,3,P1,Mid Backend,ACE-003,3
ACE-011,Monitoring & Observability,Prometheus metrics, Grafana dashboards, alerts...,5,P1,Mid Backend + DevOps,ACE-008,3
ACE-012,Performance Testing Suite,Load tests, latency benchmarks, SLO validation...,5,P1,QA + Backend,ACE-006;ACE-007;ACE-008,3
ACE-013,Documentation Suite,API docs, deployment guide, user guide...,3,P1,Senior Backend,ACE-001-012,3
ACE-014,Staging Deployment & A/B,Deploy to staging, A/B testing, validation...,5,P0,DevOps + Backend,ACE-001-013,4
ACE-015,Production Launch,Canary, gradual rollout, monitoring...,3,P0,DevOps + Senior Backend,ACE-014,4
```

---

## Appendix

**Estimation Method:** Planning Poker with Fibonacci scale
**Story Point Scale:** Fibonacci (1,2,3,5,8,13,21)
- 1 SP: Trivial change (< 4 hours)
- 2 SP: Simple change (4-8 hours)
- 3 SP: Straightforward feature (1-2 days)
- 5 SP: Moderate complexity (3-5 days)
- 8 SP: Complex feature (5-8 days)
- 13 SP: Very complex (>1 week, should be broken down)

**Definition of Done:**

- Code reviewed and approved (2+ reviewers)
- Unit tests written and passing (90%+ coverage for new code)
- Integration tests written for cross-component features
- Documentation updated (API docs, README, user guide)
- Performance validated (if applicable)
- Security review completed (data isolation, cost controls)
- Deployed to staging and validated
- No P0/P1 bugs outstanding

**Buffer Strategy:**

- 20% buffer included in estimates for unknowns
- Sprint 4 has lighter load (8 SP) to absorb any overruns
- Critical path has 2 weeks buffer in 8-week timeline

**Assumptions:**

- PostgreSQL database already exists and is accessible
- Portkey LLM gateway already configured
- Development environment set up (uv, pytest, etc.)
- No major architectural changes to existing components needed
- Team has experience with async Python, Pydantic, PostgreSQL

**Success Criteria:**

All Phase 1 tasks completed AND:
- Performance improvement: +5-7% on long-running agents
- System overhead: <5%
- Cost: <$100/month for 100 agents
- Zero production incidents
- Positive user feedback from 3+ customers
