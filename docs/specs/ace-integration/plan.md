# Implementation Plan: ACE Integration - COMPASS Enhanced Meta-Thinker

**Source:** `docs/specs/ace-integration/spec.md` v2.0 (COMPASS-Enhanced)
**Date:** 2025-10-23
**Version:** 2.0 (COMPASS-Enhanced)
**Status:** Ready for Implementation
**Format:** Product Requirements Prompt (PRP)

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → This Plan**

1. **COMPASS Paper Analysis:** `.docs/research/compass-enhancement-analysis.md`
   - Meta-Thinker role definition
   - Strategic intervention patterns
   - Performance monitoring requirements
   - 20% accuracy improvement validation

2. **Formal Specification:** `docs/specs/ace-integration/spec.md` v2.0
   - Functional requirements (FR-ACE-101 through FR-ACE-404)
   - COMPASS-enhanced success metrics
   - Non-functional requirements

3. **Original ACE Research:** `docs/research/ace-integration-analysis.md`
   - Context playbook foundations
   - Delta generation approach
   - Self-supervised learning

### Related Documentation

**System Context:**
- Architecture: `.sage/agent/system/architecture.md`
- Tech Stack: `.sage/agent/system/tech-stack.md`
- Patterns: `.sage/agent/system/patterns.md`

**Related Specifications:**
- **MEM (Memory System):** `docs/specs/memory-system/spec.md` v2.0 (COMPASS Context Manager)
  - **Critical Dependency:** ACE Phase 4 requires MEM Phase 5 completion
  - Integration: ACE queries MEM for strategic context
- Agent Runtime: Existing orchestration layer
- Task Manager: Existing task execution tracking

---

## 📊 Executive Summary

### Business Alignment

**Problem Statement:**

Long-running agents suffer from three critical failures:
1. **Context degradation** - Performance decays without self-correction
2. **Error compounding** - Small mistakes accumulate into task failures
3. **Lack of strategic oversight** - Agents miss opportunities to replan or reflect

**COMPASS Insight:** Separating tactical execution from strategic oversight (Meta-Thinker) enables 20% accuracy improvements on long-horizon tasks.

**Value Proposition:**

ACE integration addresses these failures by implementing:
1. **Performance monitoring** - Track agent effectiveness across reasoning stages
2. **Strategic interventions** - Trigger replanning, reflection, and context refresh when needed
3. **Error-aware coordination** - Prevent compounding mistakes through MEM integration
4. **Dynamic capability evaluation** - Recommend agent capability changes based on task fitness

### Technical Approach

**Architecture Pattern:** Meta-Thinker + Context Evolution (Hybrid)

1. **Meta-Thinker (COMPASS):** Strategic oversight with performance monitoring and intervention orchestration
2. **Context Evolution (Original ACE):** Self-supervised playbook improvement via delta generation
3. **Integration:** Intervention outcomes inform playbook evolution

**Implementation Strategy:**

- **6 phases over 8 weeks** (4 two-week sprints)
- **Staggered start:** Begin Week 3 (after MEM Phase 2 completes)
- **Hard dependency:** Phase 4 waits for MEM Phase 5 (Week 8)
- **Parallel tracks:** Monitoring, intervention, and capability evaluation can proceed independently

### Key Success Metrics (COMPASS-Enhanced)

**Service Level Objectives (SLOs):**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intervention Latency | <200ms (p95) | ACE-MEM coordination time |
| System Overhead | <5% | End-to-end task latency increase |
| Availability | 99.9% | ACE service uptime |

**Key Performance Indicators (KPIs):**

| Metric | Target | COMPASS Validation |
|--------|--------|-------------------|
| **Long-Horizon Accuracy** | +20% improvement | ✅ COMPASS achieved on GAIA |
| **Critical Error Recall** | 90%+ | 🆕 Error detection accuracy |
| **Intervention Precision** | 85%+ | 🆕 Correct intervention rate |
| **Context Degradation** | 30% reduction | ✅ Original ACE target |
| **Cost Control** | <$150/month (100 agents) | Portkey token tracking |

---

## 💻 Code Examples & Patterns

### Repository Patterns

**Performance Monitoring Pattern:**

```python
# Based on AgentCore async patterns
from agentcore.ace.monitors import PerformanceMonitor
from agentcore.ace.models import StageMetrics, PerformanceBaseline

async def track_stage_performance(
    agent_id: str,
    task_id: str,
    stage: StageType,
    metrics: StageMetrics
) -> None:
    """Track performance metrics for current reasoning stage."""
    monitor = PerformanceMonitor(agent_id, task_id)

    # Compute real-time metrics
    await monitor.update_metrics(stage, metrics)

    # Check against baseline
    baseline = await monitor.get_baseline(stage)
    if monitor.detect_degradation(metrics, baseline):
        # Trigger intervention signal
        await intervention_engine.signal_degradation(
            agent_id, task_id, stage, metrics
        )
```

**Strategic Intervention Pattern:**

```python
from agentcore.ace.intervention import InterventionEngine, InterventionType
from agentcore.memory.ace_integration import ACEMemoryInterface

async def execute_intervention(
    agent_id: str,
    task_id: str,
    trigger: TriggerSignal
) -> InterventionResult:
    """Execute strategic intervention based on trigger signal."""
    engine = InterventionEngine()

    # Query MEM for strategic context
    mem_interface = ACEMemoryInterface()
    strategic_context = await mem_interface.get_strategic_context(
        query_type="strategic_decision",
        agent_id=agent_id,
        task_id=task_id,
        context=trigger.context
    )

    # Decide intervention type
    intervention_type = engine.decide_intervention(
        trigger, strategic_context
    )

    # Execute intervention via agent runtime
    result = await agent_runtime.execute_intervention(
        agent_id, task_id, intervention_type, strategic_context
    )

    # Track outcome for learning
    await mem_interface.record_intervention_outcome(
        intervention_id=result.id,
        success=result.success,
        performance_delta=result.metrics_delta
    )

    return result
```

**Context Evolution Pattern (Original ACE):**

```python
from agentcore.ace.delta import DeltaGenerator
from agentcore.ace.curator import SimpleCurator

async def evolve_context_playbook(
    agent_id: str,
    execution_trace: ExecutionTrace
) -> ContextPlaybook:
    """Self-supervised playbook improvement from execution traces."""
    delta_gen = DeltaGenerator()
    curator = SimpleCurator(confidence_threshold=0.8)

    # Generate improvement deltas from trace
    deltas = await delta_gen.generate_deltas(
        execution_trace,
        model="gpt-4o-mini"  # Cost-effective for suggestions
    )

    # Filter high-confidence deltas
    approved_deltas = curator.filter_deltas(deltas)

    # Apply to playbook
    playbook = await load_playbook(agent_id)
    updated_playbook = playbook.apply_deltas(approved_deltas)

    await save_playbook(agent_id, updated_playbook)
    return updated_playbook
```

### Key Takeaways from Patterns

- **Async-first:** All ACE operations use asyncio for non-blocking execution
- **MEM integration:** Strategic decisions query MEM for context
- **Two-model strategy:** gpt-4o-mini for suggestions, gpt-4.1 for decisions
- **Outcome tracking:** Interventions are measured for effectiveness learning

---

## 🔧 Technology Stack

### Recommended Stack (COMPASS-Enhanced)

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| **Runtime** | Python | 3.12+ | Existing AgentCore language, excellent async support |
| **Data Models** | Pydantic | 2.x | Type safety, validation, modern typing (`list[]`, `dict[]`, `\|`) |
| **Database** | PostgreSQL | 14+ | JSONB support, existing infrastructure |
| **Migrations** | Alembic | Latest | Already integrated, reversible migrations |
| **ORM** | SQLAlchemy | 2.0+ async | Existing AgentCore pattern |
| **LLM Gateway** | Portkey | Latest | Cost control, caching, multi-provider (existing) |
| **Delta Generation** | gpt-4o-mini | Latest | Cost-effective ($0.15/1M tokens) for suggestions |
| **Intervention Decisions** | gpt-4.1 | Latest | Higher accuracy for strategic decisions |
| **Metrics Storage** | TimescaleDB | 2.x | Time-series optimized PostgreSQL extension |
| **Cache** | Redis | 6+ | Existing infrastructure, metrics buffering |
| **Testing** | pytest-asyncio | Latest | Async test support |

**Key Technology Decisions:**

1. **Two-Model Strategy:**
   - **gpt-4o-mini** for delta generation (cost-effective)
   - **gpt-4.1** for intervention decisions (accuracy-critical)
   - **Rationale:** Balance cost and performance based on task criticality

2. **TimescaleDB for Metrics:**
   - PostgreSQL extension for time-series data
   - **Rationale:** Keep metrics in same database, avoid external time-series DB
   - **Performance:** Optimized for high-write workloads

3. **Async Everywhere:**
   - All I/O operations use asyncio
   - **Rationale:** Non-blocking performance monitoring and intervention execution

### Alignment with MEM Component

**Shared Technologies:**
- PostgreSQL + AsyncSession (consistent database access)
- Pydantic models with modern typing
- gpt-4o-mini for LLM operations
- Redis for caching

**New Additions:**
- TimescaleDB extension (time-series metrics)
- gpt-4.1 for strategic decisions (ACE-specific)

---

## 🏗️ Architecture Design

### System Context

**ACE Role in AgentCore:**

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                      Agent Runtime Layer                     │
│  (Tactical Execution: reasoning, tool use, action execution) │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
                │ actions & results           │ interventions
                ▼                             ▼
┌───────────────────────────────┐  ┌─────────────────────────┐
│    MEM (Context Manager)      │  │   ACE (Meta-Thinker)    │
│ - Stage-aware memory org      │◄─┤ - Performance monitoring│
│ - Hierarchical compression    │  │ - Strategic interventions│
│ - Error pattern detection     │──┤ - Capability evaluation │
│ - Test-time scaling (10:1)    │  │ - Context evolution     │
└───────────────────────────────┘  └─────────────────────────┘
         │                                    │
         │ storage & retrieval                │ metrics & playbooks
         ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                PostgreSQL + TimescaleDB + Redis              │
│  (Memories, Stage Summaries, Metrics, Playbooks, Deltas)    │
└─────────────────────────────────────────────────────────────┘
```

**COMPASS Architecture Mapping:**

| COMPASS Component | AgentCore Component | Responsibility |
|-------------------|---------------------|----------------|
| **Main Agent** | Agent Runtime | Tactical execution (reasoning, tools) |
| **Meta-Thinker** | ACE | Strategic oversight (monitoring, interventions) |
| **Context Manager** | MEM | Context organization (compression, retrieval) |

### Component Architecture

**Architecture Pattern:** Modular service within AgentCore monolith

**Design Principles:**
1. **Separation of Concerns:** Monitoring, intervention, and evolution are distinct services
2. **Async Coordination:** ACE-MEM-Agent coordination via async message passing
3. **Event-Driven:** Performance metrics trigger intervention signals
4. **Self-Supervised:** Playbook evolution learns from intervention outcomes

### ACE Component Breakdown

**Core Components:**

```plaintext
src/agentcore/ace/
├── monitors/
│   ├── performance_monitor.py    # Stage-aware metrics tracking
│   ├── baseline_tracker.py       # Baseline computation & drift detection
│   └── error_accumulator.py      # Error pattern tracking
├── intervention/
│   ├── engine.py                 # Intervention orchestration
│   ├── triggers.py               # Signal detection (4 types)
│   ├── decision.py               # Intervention type selection
│   └── executor.py               # Intervention execution via runtime
├── integration/
│   ├── mem_interface.py          # ACE-MEM coordination layer
│   ├── runtime_interface.py      # ACE-Runtime coordination
│   └── outcome_tracker.py        # Intervention effectiveness learning
├── capability/
│   ├── evaluator.py              # Task-capability fitness scoring
│   └── recommender.py            # Capability change recommendations
├── context/
│   ├── playbook.py               # Context Playbook management (original ACE)
│   ├── delta_generator.py        # LLM-based improvement suggestions
│   └── curator.py                # Confidence-threshold filtering
├── models.py                     # Pydantic models
├── jsonrpc.py                    # JSON-RPC handlers (ace.*)
└── database/
    ├── models.py                 # SQLAlchemy models
    └── repositories.py           # Async repositories
```

**Data Model:**

```python
# Key Pydantic Models

class StageMetrics(BaseModel):
    """Performance metrics for a reasoning stage."""
    stage: StageType  # planning, execution, reflection, verification
    task_id: str
    agent_id: str
    timestamp: datetime

    # Stage-specific metrics
    success_rate: float  # 0-1
    error_count: int
    velocity: float  # actions per minute
    confidence: float  # average confidence

    # Computed flags
    is_degraded: bool
    degradation_factors: list[str]

class TriggerSignal(BaseModel):
    """Signal for strategic intervention."""
    trigger_type: TriggerType  # degradation, error_accumulation, staleness, capability_mismatch
    severity: float  # 0-1
    metrics: StageMetrics
    context: dict[str, Any]
    rationale: str

class InterventionDecision(BaseModel):
    """Decision on intervention type."""
    intervention_type: InterventionType  # replan, reflect, context_refresh, capability_switch
    confidence: float
    strategic_context: StrategicContext  # from MEM
    rationale: str
    expected_impact: float

class ContextPlaybook(BaseModel):
    """Self-evolving context structure (original ACE)."""
    agent_id: str
    version: int
    sections: list[PlaybookSection]
    confidence_scores: dict[str, float]
    last_updated: datetime
```

### Data Flow & Integration

**Request Flow:**

1. **Agent Action** → **Performance Monitor**
   - Agent runtime sends action results to ACE monitor
   - Monitor updates stage metrics in real-time

2. **Performance Monitor** → **Intervention Engine**
   - Monitor detects trigger signals (degradation, errors, etc.)
   - Engine receives signals and queries MEM for context

3. **Intervention Engine** ↔ **MEM (Context Manager)**
   - Engine queries MEM for strategic context
   - MEM returns stage summaries, error patterns, successful patterns

4. **Intervention Engine** → **Agent Runtime**
   - Engine sends intervention command (replan, reflect, refresh)
   - Runtime executes intervention with strategic context

5. **Agent Runtime** → **Outcome Tracker**
   - Runtime reports intervention outcome
   - Tracker records effectiveness in MEM for learning

**Async Coordination:**

- Performance monitoring is non-blocking (<5% overhead)
- Intervention execution happens concurrently with ongoing agent work
- MEM queries have 200ms timeout with graceful degradation

---

## 4. Technical Specification

### Database Schema

**New Tables:**

1. **context_playbooks** - Self-evolving context structures
2. **playbook_deltas** - Improvement suggestions from execution traces
3. **delta_approvals** - Curation decisions (approved/rejected deltas)
4. **performance_metrics** - Stage-aware metrics (TimescaleDB hypertable)
5. **intervention_history** - Intervention execution tracking
6. **capability_evaluations** - Task-capability fitness scores

**Key Relationships:**

- `performance_metrics.agent_id` → `agents.id`
- `performance_metrics.task_id` → `tasks.id`
- `intervention_history.trigger_id` → Embedded in JSONB
- `playbook_deltas.playbook_id` → `context_playbooks.id`

**Indexing Strategy:**

- Time-series index on `performance_metrics.timestamp` (TimescaleDB)
- Composite index on `(agent_id, task_id, stage)` for metrics queries
- GIN index on `playbook_deltas.delta_content` (JSONB)

### API Design

**Top 6 Critical Endpoints:**

1. **`ace.track_performance`** - Update stage metrics
   - **Request:** `{agent_id, task_id, stage, metrics}`
   - **Response:** `{recorded: bool, triggers_detected: [...]}`
   - **Purpose:** Real-time performance tracking

2. **`ace.get_intervention_decision`** - Get strategic intervention recommendation
   - **Request:** `{agent_id, task_id, trigger_signal}`
   - **Response:** `{intervention_type, rationale, strategic_context}`
   - **Purpose:** Intervention orchestration

3. **`ace.execute_intervention`** - Execute intervention command
   - **Request:** `{agent_id, task_id, intervention_type, context}`
   - **Response:** `{intervention_id, status, expected_duration}`
   - **Purpose:** Intervention execution tracking

4. **`ace.get_playbook`** - Retrieve context playbook
   - **Request:** `{agent_id}`
   - **Response:** `{playbook: ContextPlaybook, version}`
   - **Purpose:** Context playbook access

5. **`ace.generate_deltas`** - Generate improvement suggestions
   - **Request:** `{agent_id, execution_trace}`
   - **Response:** `{deltas: [PlaybookDelta], confidence_scores}`
   - **Purpose:** Self-supervised playbook evolution

6. **`ace.evaluate_capability`** - Assess task-capability fitness
   - **Request:** `{agent_id, task_requirements}`
   - **Response:** `{fitness_score, recommendations}`
   - **Purpose:** Dynamic capability evaluation

### Security

**Authentication/Authorization:**
- Reuse existing AgentCore JWT authentication
- Agent-scoped data access (filter by `agent_id` from A2A context)
- Admin role for playbook management

**Data Protection:**
- Metrics encrypted at rest (PostgreSQL TDE)
- Playbook content encrypted for sensitive agents
- PII scrubbing in execution traces

**Security Testing:**
- Integration tests with invalid agent IDs (authorization)
- SQL injection tests on JSONB queries
- Rate limiting on LLM-heavy endpoints

### Performance

**Performance Targets:**

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Metrics Update | <50ms (p95) | Non-blocking monitoring |
| Intervention Decision | <200ms (p95) | Real-time coordination with MEM |
| Playbook Retrieval | <100ms (p95) | Frequent access pattern |
| Delta Generation | <5s (p95) | LLM call acceptable latency |

**Caching Strategy:**

- **Redis caching:**
  - Playbooks cached for 10 minutes (frequently accessed)
  - Baselines cached for 1 hour (stable)
  - Strategic context cached for 5 minutes (fresh)

**Database Optimization:**

- TimescaleDB compression for old metrics (retention: 90 days)
- Connection pooling (min 10, max 50)
- Metrics batching (buffer 100 updates or 1 second)

**Scaling Strategy:**

- Horizontal: Multiple ACE service instances (stateless)
- Vertical: TimescaleDB read replicas for metrics queries
- Auto-scaling triggers: CPU >70%, intervention queue >100

---

## 5. Development Setup

**Required Tools:**
- Python 3.12+
- PostgreSQL 14+ with TimescaleDB extension
- Redis 6+
- Portkey API key

**Local Environment:**

```bash
# Install TimescaleDB extension
docker compose -f docker-compose.dev.yml up -d
docker exec -it agentcore-postgres psql -U agentcore -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Environment variables
cat >> .env <<EOF
TIMESCALEDB_ENABLED=true
ACE_DELTA_MODEL=gpt-4o-mini
ACE_INTERVENTION_MODEL=gpt-4.1
ACE_MONITORING_OVERHEAD_TARGET=0.05
EOF

# Run migrations
uv run alembic upgrade head
```

**CI/CD Pipeline:**
- Linting: `uv run ruff check src/agentcore/ace/`
- Type checking: `uv run mypy src/agentcore/ace/`
- Tests: `uv run pytest tests/ace/ --cov=agentcore/ace --cov-report=html`
- Coverage gate: 90%+ required

**Testing Framework:**
- **Unit tests:** pytest with mocked dependencies
- **Integration tests:** testcontainers for PostgreSQL + Redis
- **Load tests:** Locust for intervention throughput
- **COMPASS validation:** Custom benchmark dataset

---

## 6. Risk Management

| Risk | Impact | Likelihood | Mitigation | Contingency |
|------|--------|------------|------------|-------------|
| **MEM Dependency Timing** | HIGH | MEDIUM | Start ACE Phases 1-3 in parallel with MEM; mock MEM interface | Defer Phase 4 to Week 9; reduce MEM integration scope |
| **Intervention Decision Accuracy** | HIGH | MEDIUM | A/B testing, threshold tuning, ground truth validation dataset | Lower precision target to 80%; increase human review |
| **Real-Time Performance Overhead** | MEDIUM | MEDIUM | Async processing, metric batching, caching; load test at 100 agents | Reduce monitoring frequency; sample 50% of actions |
| **Cross-Component Coordination** | MEDIUM | HIGH | Clear state machine, transaction boundaries, integration tests | Simplify to synchronous coordination; accept higher latency |
| **COMPASS Validation Failure** | HIGH | MEDIUM | Early validation tests, iterative tuning, benchmark dataset preparation | Document partial achievement (e.g., +15% instead of +20%); roadmap for full target |

---

## 7. Implementation Roadmap

### Phase 1: Foundation + Original ACE Core (Weeks 1-2, Sprint 1)

**Goal:** Establish database schema, models, and context playbook management

**Deliverables:**
- Database migration with 6 new tables (TimescaleDB hypertable for metrics)
- Pydantic models for all ACE entities
- SQLAlchemy ORM models with async support
- Repository layer for data access
- Context Playbook manager (CRUD operations)
- Delta generator with gpt-4o-mini integration
- Simple curator with confidence-threshold filtering

**Story Points:** 45 SP

**Key Tasks:**
- ACE-002: Database migration (8 SP)
- ACE-003: Pydantic models (5 SP)
- ACE-004: ORM models (5 SP)
- ACE-005: Repositories (8 SP)
- ACE-006: Playbook manager (8 SP)
- ACE-007: Delta generator (8 SP)
- ACE-008: Simple curator (3 SP)

**Dependencies:**
- PostgreSQL + TimescaleDB extension
- Portkey configuration
- LLM model access (gpt-4o-mini)

---

### Phase 2: Performance Monitoring (COMPASS ACE-1) (Week 3, Sprint 2)

**Goal:** Implement stage-aware performance tracking and baseline computation

**Deliverables:**
- PerformanceMonitor service
- Stage-specific metrics tracking (4 stage types)
- Baseline computation and drift detection
- Error accumulation tracking
- Metrics API (JSON-RPC handlers)
- Metrics visualization integration

**Story Points:** 30 SP

**Key Tasks:**
- ACE-009: PerformanceMonitor core (8 SP)
- ACE-010: Baseline tracker (5 SP)
- ACE-011: Error accumulator (5 SP)
- ACE-012: Metrics API (5 SP)
- ACE-013: Metrics dashboard integration (5 SP)
- ACE-014: Integration tests (2 SP)

**Dependencies:**
- Phase 1 complete
- Agent Runtime integration for action results

**Critical Path:** ACE-009 → ACE-010 → ACE-011

---

### Phase 3: Strategic Intervention Engine (COMPASS ACE-2) (Week 4, Sprint 2)

**Goal:** Implement trigger detection and intervention orchestration

**Deliverables:**
- InterventionEngine service
- Trigger detection (4 trigger types: degradation, error_accumulation, staleness, capability_mismatch)
- Intervention decision making with gpt-4.1
- Intervention executor (agent runtime integration)
- Intervention history tracking

**Story Points:** 35 SP

**Key Tasks:**
- ACE-015: InterventionEngine core (8 SP)
- ACE-016: Trigger detection (8 SP)
- ACE-017: Decision making (8 SP)
- ACE-018: Intervention executor (5 SP)
- ACE-019: Agent runtime integration (3 SP)
- ACE-020: Integration tests (3 SP)

**Dependencies:**
- Phase 2 complete (metrics feed triggers)
- Agent Runtime supports intervention commands

**Critical Path:** ACE-015 → ACE-016 → ACE-017 → ACE-018

---

### Phase 4: ACE-MEM Integration Layer (COMPASS ACE-3) (Week 5, Sprint 3)

**Goal:** Coordinate with MEM for strategic context queries

**Deliverables:**
- MEM integration layer (ACEMemoryInterface)
- Strategic context queries (4 query types)
- Intervention outcome tracking
- Cross-component coordination tests

**Story Points:** 25 SP

**Key Tasks:**
- ACE-021: MEM integration layer (8 SP)
- ACE-022: Strategic context queries (8 SP)
- ACE-023: Outcome tracking (5 SP)
- ACE-024: Cross-component tests (4 SP)

**Dependencies:**
- **CRITICAL:** MEM Phase 5 complete (Week 8) - ACE integration API ready
- Phase 3 complete (intervention engine needs strategic context)

**Risk Mitigation:**
- Mock MEM interface for development (Weeks 5-7)
- Real MEM integration in Week 8

---

### Phase 5: Capability Evaluation (COMPASS ACE-4) (Week 6, Sprint 3)

**Goal:** Assess task-capability fitness and recommend changes

**Deliverables:**
- CapabilityEvaluator service
- Task-capability fitness scoring
- Recommendation engine
- Capability evaluation API

**Story Points:** 20 SP

**Key Tasks:**
- ACE-025: CapabilityEvaluator (8 SP)
- ACE-026: Fitness scoring (5 SP)
- ACE-027: Recommendation engine (5 SP)
- ACE-028: Tests (2 SP)

**Dependencies:**
- Phase 2 complete (metrics inform fitness)
- Agent capability registry (existing)

---

### Phase 6: Production Readiness (Weeks 7-8, Sprint 4)

**Goal:** Optimize, validate, and prepare for production deployment

**Deliverables:**
- Performance tuning (metrics batching, caching optimization)
- COMPASS validation tests (+20% accuracy target)
- Load testing (100 agents, 1000 tasks)
- Monitoring setup (Prometheus metrics, Grafana dashboards)
- Operational documentation (runbook, API docs)

**Story Points:** 20 SP

**Key Tasks:**
- ACE-029: Performance tuning (5 SP)
- ACE-030: COMPASS validation tests (5 SP)
- ACE-031: Load testing (3 SP)
- ACE-032: Monitoring setup (3 SP)
- ACE-033: Documentation (4 SP)

**Dependencies:**
- All phases 1-5 complete
- GAIA-style benchmark dataset prepared

---

## 8. Quality Assurance

### Testing Strategy

**Unit Tests (90%+ coverage):**
- All services (PerformanceMonitor, InterventionEngine, etc.)
- Trigger detection logic
- Decision-making algorithms
- Delta generation and curation

**Integration Tests:**
- ACE-MEM coordination
- ACE-Agent Runtime coordination
- End-to-end intervention workflows
- Playbook evolution from traces

**Performance Tests:**
- Metrics update latency (<50ms target)
- Intervention decision latency (<200ms target)
- System overhead (<5% target)
- Load test at 100 concurrent agents

**COMPASS Validation Tests:**
- +20% accuracy improvement on GAIA-style benchmark
- 90%+ critical error recall
- 85%+ intervention precision
- A/B testing (ACE on vs off)

### Code Quality Gates

- Ruff linting (no errors)
- Mypy strict mode (no type errors)
- 90%+ test coverage
- 0 critical security findings (SAST)

### Deployment Verification Checklist

- [ ] All migrations applied successfully
- [ ] TimescaleDB extension enabled
- [ ] Redis cache operational
- [ ] Portkey API key configured
- [ ] MEM integration endpoints responding
- [ ] Agent Runtime supports intervention commands
- [ ] Metrics flowing to Prometheus
- [ ] Grafana dashboards displaying
- [ ] Load test passed (100 agents)
- [ ] COMPASS validation passed (+20% accuracy)

### Monitoring and Alerting

**Prometheus Metrics:**
- `ace_metrics_update_latency_ms` (p95 target: <50ms)
- `ace_intervention_decision_latency_ms` (p95 target: <200ms)
- `ace_intervention_precision_ratio` (target: 85%+)
- `ace_error_recall_ratio` (target: 90%+)
- `ace_system_overhead_ratio` (target: <5%)
- `ace_mem_query_latency_ms` (p95 target: <150ms)

**Grafana Dashboards:**
- ACE System Health (latency, error rates, throughput)
- Intervention Analytics (precision, recall, outcome tracking)
- COMPASS Validation Metrics (accuracy delta, error patterns)

**Alerting Rules:**
- Intervention precision drops below 80% (warning)
- System overhead exceeds 7% (critical)
- MEM query failures exceed 5% (warning)
- Error recall drops below 85% (warning)

---

## ⚠️ Error Handling & Edge Cases

### Error Scenarios

**1. MEM Query Failure**
- **Cause:** MEM service unavailable or timeout
- **Impact:** Strategic context unavailable for intervention decisions
- **Handling:** Degrade gracefully to baseline intervention (no strategic context)
- **Recovery:** Retry with exponential backoff (3 attempts, 100ms → 400ms → 1600ms)
- **User Experience:** Intervention executes with reduced effectiveness; alert logged

**2. Intervention Execution Failure**
- **Cause:** Agent Runtime rejects intervention command
- **Impact:** Agent continues without strategic correction
- **Handling:** Log failure, update intervention history with failure status
- **Recovery:** Schedule retry after 30 seconds if trigger persists
- **User Experience:** Agent may exhibit degraded performance until next trigger

**3. LLM API Failure (Delta Generation)**
- **Cause:** Portkey timeout or model unavailable
- **Impact:** Playbook evolution paused
- **Handling:** Queue delta generation request for retry
- **Recovery:** Circuit breaker pattern (5 failures → pause for 5 minutes)
- **User Experience:** Playbook remains static; alert sent to ops

**4. Metrics Buffer Overflow**
- **Cause:** High-throughput agent execution overwhelms metric ingestion
- **Impact:** Metrics delayed or dropped
- **Handling:** Sample metrics at 50% when buffer >80% full
- **Recovery:** Auto-scale TimescaleDB write capacity
- **User Experience:** Slightly reduced monitoring fidelity; no intervention impact

### Edge Cases

| Edge Case | Detection | Handling | Testing Approach |
|-----------|-----------|----------|------------------|
| Agent switches mid-task | Task ID remains constant, agent ID changes | Track both agent IDs in metrics; baseline reset | Integration test with agent failover |
| Multiple triggers simultaneously | Trigger queue >1 | Prioritize by severity; execute highest-severity intervention first | Unit test with concurrent trigger signals |
| Intervention during reflection stage | Stage type = reflection | Skip replan/reflect interventions; allow context_refresh only | Integration test with stage-aware logic |
| Baseline not yet established | <10 executions | Use fallback thresholds (hardcoded); log warning | Unit test with empty baseline |
| Playbook version conflict | Concurrent delta applications | Optimistic locking with version check; retry on conflict | Integration test with concurrent updates |

### Input Validation

**Validation Rules:**
- `agent_id`: Must exist in agents table
- `task_id`: Must exist in tasks table
- `stage`: Must be one of [planning, execution, reflection, verification]
- `metrics.success_rate`: Range [0, 1]
- `metrics.error_count`: Non-negative integer
- `intervention_type`: Must be one of [replan, reflect, context_refresh, capability_switch]

**Sanitization:**
- Execution traces: Strip PII before delta generation
- Playbook content: Escape special characters in JSONB
- Strategic context: Validate JSON schema from MEM

### Graceful Degradation

**Fallback Strategies:**
- **MEM unavailable:** Use cached strategic context (5-minute TTL)
- **LLM unavailable:** Queue delta generation; playbook remains static
- **TimescaleDB unavailable:** Buffer metrics in Redis (1-hour capacity)

---

## 📚 References & Traceability

### Source Documentation

**COMPASS Paper Analysis:**
- `.docs/research/compass-enhancement-analysis.md`
  - Meta-Thinker role definition
  - Strategic intervention patterns
  - Performance monitoring requirements
  - 20% accuracy improvement validation

**Specification:**
- `docs/specs/ace-integration/spec.md` v2.0 (COMPASS-Enhanced)
  - Functional requirements (FR-ACE-101 through FR-ACE-404)
  - Non-functional requirements (NFR-ACE-001 through NFR-ACE-007)
  - Success metrics and acceptance criteria

**Original ACE Research:**
- `docs/research/ace-integration-analysis.md`
  - Context playbook foundations
  - Delta generation approach
  - Self-supervised learning

### System Context

**Architecture & Patterns:**
- `.sage/agent/system/architecture.md` - AgentCore architecture
- `.sage/agent/system/tech-stack.md` - Existing technology stack
- `.sage/agent/system/patterns.md` - Code patterns

### Related Components

**Dependencies:**
- **MEM (Memory System):** `docs/specs/memory-system/plan.md` v2.0
  - **Relationship:** ACE Phase 4 requires MEM Phase 5 (strategic context API)
  - **Integration:** ACE queries MEM for error patterns, stage summaries, successful patterns
- **Agent Runtime:** Existing orchestration layer
  - **Relationship:** ACE sends intervention commands to runtime
- **Task Manager:** Existing task tracking
  - **Relationship:** ACE queries task metadata and performance history

**Dependents:**
- Future components may query ACE for intervention history and capability evaluations

---

## 📋 Summary

**Implementation Timeline:** 8 weeks (4 two-week sprints)
**Total Effort:** ~175 story points
**Team:** 1 senior backend engineer (full-time)
**Start:** Week 3 (after MEM Phase 2 completes)
**End:** Week 10 (parallel with MEM completion)

**Critical Dependencies:**
- **MEM Phase 5 (Week 8):** Required for ACE Phase 4 (ACE-MEM integration)
- Agent Runtime intervention support
- TimescaleDB extension for time-series metrics

**COMPASS Targets:**
- ✅ +20% long-horizon accuracy improvement
- ✅ 90%+ critical error recall
- ✅ 85%+ intervention precision
- ✅ <200ms ACE-MEM coordination latency
- ✅ <5% system overhead

**Next Steps:**
1. Review and approve this plan
2. Run `/sage.tasks ace-integration` to generate story breakdown
3. Begin Phase 1 implementation in Week 3
4. Coordinate with MEM team for Phase 4 integration (Week 8)

---

**Document Status:** ✅ Ready for Task Breakdown
**Plan Version:** 2.0 (COMPASS-Enhanced)
**Last Updated:** 2025-10-23
