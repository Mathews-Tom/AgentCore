# Specification: ACE (Agent Context Engineering) Integration - COMPASS Enhanced

**Component:** ACE Integration Layer (Meta-Thinker)
**Version:** 2.0 (COMPASS-Enhanced)
**Status:** Draft
**Priority:** P1
**Owner:** Architecture Team
**Created:** 2025-10-12
**Updated:** 2025-10-23
**Based On:**
- Research document `docs/research/ace-integration-analysis.md`
- `.docs/research/compass-enhancement-analysis.md` (COMPASS paper analysis)

---

## 1. Overview

### 1.1 Purpose

Implement **COMPASS-enhanced ACE** as the **Meta-Thinker component** that provides:

1. **Performance Monitoring**: Track agent effectiveness across reasoning stages
2. **Strategic Interventions**: Trigger context refreshes, replanning, and reflection when performance degrades
3. **Error-Aware Context Management**: Coordinate with MEM (Context Manager) to prevent compounding mistakes
4. **Dynamic Capability Evaluation**: Assess and recommend agent capability changes based on task fitness

**COMPASS Insight:** The Meta-Thinker role is critical for long-horizon planning success. Without strategic oversight, agents fail to replan from errors or adjust when context becomes stale.

### 1.2 Scope

**In Scope (Phase 1 - COMPASS Enhanced):**

- **Performance monitoring with stage-aware metrics** (COMPASS ACE-1)
- **Strategic intervention engine with trigger detection** (COMPASS ACE-2)
- **ACE-MEM integration layer for context queries** (COMPASS ACE-3)
- **Dynamic capability evaluation** (COMPASS ACE-4)
- Context Playbook data structures (original ACE Phase 1)
- Basic delta generation from execution traces (original ACE Phase 1)
- Simple curation with confidence-threshold filtering (original ACE Phase 1)
- PostgreSQL storage for contexts, deltas, and metrics
- Configuration and feature flags

**Out of Scope (Deferred to Phase 2+):**

- A2A context sharing protocol extensions
- Full reflection loop (confidence filtering used instead)
- Real-time context streaming
- DSPy + ACE dual optimization
- Advanced analytics dashboards
- Multi-tenant context isolation
- Cross-agent learning networks

### 1.3 Goals & Success Metrics (COMPASS-Enhanced)

**Primary Goals:**

1. **+20% performance improvement on long-horizon tasks** (COMPASS-validated target)
2. **90%+ critical error recall** (prevent compounding mistakes)
3. **Strategic intervention accuracy: 85%+** (correct trigger decisions)
4. **ACE-MEM coordination latency: <200ms** (real-time decision support)
5. Prevent context degradation in long-running agents (original ACE goal)
6. Enable self-supervised context improvement (original ACE goal)

**Success Metrics (COMPASS-Enhanced):**

| Metric | Target | COMPASS Validation |
|--------|--------|-------------------|
| Long-Horizon Accuracy | +20% improvement | ✅ COMPASS achieved on GAIA |
| Critical Error Recall | 90%+ | 🆕 Meta-Thinker error tracking |
| Intervention Precision | 85%+ | 🆕 Correct trigger rate |
| ACE-MEM Coordination | <200ms latency | 🆕 Real-time queries |
| Context Degradation | 30% reduction | ✅ Original ACE target |
| System Overhead | <5% latency | ✅ Original ACE target |
| Cost | <$150/month (100 agents) | ⬆️ Increased for Meta-Thinker |

### 1.4 Dependencies

**Required:**

- PostgreSQL database (existing)
- **MEM (Memory System) Component** - For Context Manager role integration 🆕
- LLM provider via Portkey (existing)
- Agent Runtime Layer (existing)
- Task Manager (existing)
- Python asyncio (existing)

**Optional:**

- DSPy integration (for future dual optimization)
- A2A protocol extensions (for future context sharing)

---

## 2. Functional Requirements

### 2.1 Performance Monitoring (COMPASS ACE-1)

**FR-ACE-101: Stage-Aware Performance Metrics** 🆕

The system SHALL track performance metrics across reasoning stages:

- **Stage-specific metrics**:
  - Planning stage: Goal clarity score, constraint completeness
  - Execution stage: Action success rate, error frequency
  - Reflection stage: Learning extraction rate, insight quality
  - Verification stage: Validation accuracy, goal achievement

- **Cross-stage metrics**:
  - Overall task progress velocity
  - Error accumulation rate
  - Context staleness score
  - Intervention effectiveness

**Acceptance Criteria:**

- Metrics computed in real-time (<100ms per update)
- Metrics persisted to database with stage linkage
- Metrics accessible via JSON-RPC API
- Metrics trigger threshold-based alerts

**FR-ACE-102: Performance Baselines** 🆕

The system SHALL establish and track performance baselines:

- Initial baseline from first 10 task executions
- Rolling baseline updated every 50 executions
- Baseline comparison for degradation detection
- Baseline-relative thresholds for intervention triggers

**Acceptance Criteria:**

- Baseline computation includes confidence intervals
- Baselines stored per agent and task type
- Baseline drift detection with statistical significance
- Baseline reset mechanism for major agent updates

**FR-ACE-103: Error Accumulation Tracking** 🆕

The system SHALL track error accumulation patterns:

- Error count per stage and overall
- Error severity distribution
- Error type clustering
- Compounding error detection (related errors in sequence)

**Acceptance Criteria:**

- Error accumulation computed within 50ms
- Compounding error patterns flagged automatically
- Error trends visualized in monitoring dashboard
- Integration with MEM error pattern detection

### 2.2 Strategic Intervention Engine (COMPASS ACE-2)

**FR-ACE-201: Intervention Trigger Detection** 🆕

The system SHALL detect strategic intervention signals:

1. **Performance Degradation**:
   - Task velocity drops below 50% of baseline
   - Error rate exceeds 2x baseline
   - Success rate drops below 70%

2. **Error Accumulation**:
   - 3+ errors in single stage
   - Compounding error pattern detected
   - Same error type repeats 2+ times

3. **Context Staleness**:
   - No context refresh in 20+ steps
   - Low-confidence sections dominate playbook
   - Memory retrieval returning irrelevant results

4. **Capability Mismatch**:
   - Task requirements exceed agent capabilities
   - 50%+ actions failing due to capability gaps
   - Alternative capabilities show higher fitness

**Acceptance Criteria:**

- Trigger detection latency <50ms
- False positive rate <15%
- Trigger rationale logged for analysis
- Trigger thresholds configurable per agent

**FR-ACE-202: Intervention Decision Making** 🆕

The system SHALL decide on intervention type based on signals:

- **Context Refresh**: Request updated context from MEM when staleness detected
- **Replan**: Trigger replanning when error accumulation or velocity drop
- **Reflect**: Force reflection stage when learning opportunities missed
- **Capability Switch**: Recommend capability changes when mismatch detected

**Acceptance Criteria:**

- Decision latency <100ms
- Decision accuracy 85%+ (validated against ground truth)
- Decision rationale includes metric evidence
- Decision logs for intervention effectiveness analysis

**FR-ACE-203: Intervention Execution** 🆕

The system SHALL execute interventions via Agent Runtime:

- Send intervention command to agent runtime
- Provide strategic context from MEM
- Track intervention start/end time
- Record intervention outcome (success/failure)

**Acceptance Criteria:**

- Intervention execution non-blocking
- Intervention failures handled gracefully
- Intervention effectiveness measured
- Intervention history persisted to database

### 2.3 ACE-MEM Integration Layer (COMPASS ACE-3)

**FR-ACE-301: Strategic Context Queries** 🆕

The system SHALL query MEM for strategic context:

```python
# Query types
- "strategic_decision": Get context for intervention decision
- "error_analysis": Get error patterns and recovery context
- "capability_evaluation": Get performance data for capability assessment
- "context_refresh": Get latest compressed context for agent
```

**Acceptance Criteria:**

- Query latency <150ms (p95)
- Query results include relevance scores
- Query failures degrade gracefully
- Query history tracked for optimization

**FR-ACE-302: Intervention Outcome Tracking** 🆕

The system SHALL track intervention effectiveness:

- **Before-intervention metrics**: Performance state before intervention
- **After-intervention metrics**: Performance state after intervention
- **Delta computation**: Improvement or degradation quantified
- **Learning**: Update intervention thresholds based on effectiveness

**Acceptance Criteria:**

- Outcome tracking automated for all interventions
- Delta computation includes statistical significance
- Learning updates intervention thresholds incrementally
- Outcome data used for intervention model training

**FR-ACE-303: Shared Data Models** 🆕

The system SHALL define shared data models with MEM:

```python
class TaskContext(BaseModel):
    task_id: str
    agent_id: str
    current_stage: str  # planning, execution, reflection, verification
    stage_memory_id: str | None  # Link to MEM stage
    performance_metrics: PerformanceMetrics
    intervention_history: list[InterventionRecord]
    error_summary: ErrorSummary  # From MEM

class ContextQuery(BaseModel):
    query_type: str
    task_id: str
    current_metrics: PerformanceMetrics
    focus_areas: list[str]
    max_context_tokens: int = 2000

class StrategicContext(BaseModel):
    relevant_stage_summaries: list[str]
    critical_facts: list[str]
    error_patterns: list[str]
    successful_patterns: list[str]
    context_health_score: float  # 0-1
```

**Acceptance Criteria:**

- Data models shared via Python package
- Pydantic validation enforced
- Backwards compatibility maintained
- Versioning supported for schema evolution

### 2.4 Dynamic Capability Evaluation (COMPASS ACE-4)

**FR-ACE-401: Capability Fitness Scoring** 🆕

The system SHALL evaluate agent capability fitness:

```python
class CapabilityFitness(BaseModel):
    capability_id: str
    success_rate: float  # 0-1
    error_correlation: float  # 0-1 (high = capability causing errors)
    usage_frequency: int
    fitness_score: float  # success_rate * (1 - error_correlation)
```

**Acceptance Criteria:**

- Fitness computed per capability per task type
- Fitness updated after each task execution
- Fitness scores persisted to database
- Fitness trends tracked over time

**FR-ACE-402: Capability Recommendations** 🆕

The system SHALL recommend capability changes:

- **Identify underperforming capabilities**: Fitness score <0.5
- **Query capability registry**: Find alternatives with similar functions
- **Rank alternatives**: By historical fitness on similar tasks
- **Provide rationale**: Explain why switch recommended

**Acceptance Criteria:**

- Recommendations generated within 500ms
- Recommendations include confidence scores
- Recommendations validated against capability registry
- Recommendations logged for effectiveness tracking

**FR-ACE-403: Capability Switch Execution** 🆕

The system SHALL facilitate capability switching:

- **Validate new capabilities**: Check compatibility and availability
- **Coordinate with Agent Manager**: Update agent capability list
- **Track switch effectiveness**: Measure post-switch performance
- **Rollback mechanism**: Revert if performance degrades further

**Acceptance Criteria:**

- Switch execution atomic (all-or-nothing)
- Switch latency <1 second
- Switch failures handled gracefully
- Switch history persisted for analysis

### 2.5 Context Playbook Management (Original ACE Phase 1)

**FR-ACE-001: Context Playbook Data Structure**

The system SHALL maintain structured context playbooks for each agent containing:

- Agent metadata (ID, philosophy, version)
- Context sections (strategies, patterns, failures, learnings)
- **Performance metrics** (COMPASS-enhanced with stage awareness) 🆕
- **Intervention history** (COMPASS addition) 🆕
- Evolution history (list of applied delta IDs)
- Timestamps (created, last updated)

**Acceptance Criteria:**

- Playbook can be serialized to/from JSON
- Playbook supports CRUD operations
- Playbook version increments on each evolution
- Playbook can be compiled into execution context string
- **Playbook includes stage-aware performance metrics** 🆕
- **Playbook links to MEM stage memories** 🆕

**FR-ACE-002 through FR-ACE-015**: [Original ACE Phase 1 requirements remain unchanged]

---

## 3. Non-Functional Requirements

### 3.1 Performance (COMPASS-Enhanced)

**NFR-ACE-001: Latency Limits**

- **Playbook retrieval: <50ms (p95)**
- **Metric computation: <100ms (p95)** 🆕
- **Intervention trigger detection: <50ms (p95)** 🆕
- **ACE-MEM context query: <150ms (p95)** 🆕
- **Intervention decision: <100ms (p95)** 🆕
- **Capability evaluation: <500ms (p95)** 🆕
- Delta generation: <5s (p95)
- Delta application: <100ms (p95)
- Context injection: <50ms (p95)

**NFR-ACE-002: Throughput**

- Support 100+ concurrent agents
- Handle 1000+ executions per hour
- **Process 50+ interventions per hour** 🆕
- **Track 10K+ performance metrics per hour** 🆕
- Process 100+ evolution cycles per hour
- Store 10K+ execution traces

**NFR-ACE-003: Resource Usage**

- CPU overhead: <8% of baseline (increased from 5% for Meta-Thinker) 🆕
- Memory overhead: <150MB per 100 agents (increased for metrics) 🆕
- Database storage: <15MB per agent (increased for metrics and interventions) 🆕
- LLM token usage: <$150/month for 100 agents (increased for interventions) 🆕

### 3.2 Reliability (COMPASS-Enhanced)

**NFR-ACE-004: Availability**

- ACE failures do not impact core agent execution
- **Meta-Thinker unavailability degrades to baseline performance** 🆕
- **ACE-MEM coordination failures trigger fallback mode** 🆕
- Graceful degradation when LLM unavailable
- Automatic retry with exponential backoff
- Circuit breaker for cost protection

**NFR-ACE-005: Data Integrity**

- Playbook updates are atomic
- Delta application is transactional
- **Metric updates are eventually consistent** 🆕
- **Intervention records are immutable** 🆕
- Trace capture does not lose data
- Database migrations are reversible

**NFR-ACE-006: Fault Tolerance**

- Evolution worker restarts on crash
- **Intervention engine restarts on crash** 🆕
- **Metrics collection resilient to transient failures** 🆕
- Queued evolutions persist across restarts
- Partial updates roll back cleanly
- Error states are recoverable

### 3.3 Security

**NFR-ACE-007: Data Isolation**

- Playbooks isolated per agent
- **Performance metrics isolated per agent** 🆕
- **Intervention logs contain no sensitive data** 🆕
- Traces contain no sensitive data (or are encrypted)
- Delta content sanitized for injection attacks
- Database access follows least-privilege

**NFR-ACE-008: Cost Control**

- Monthly token budget enforced
- **Intervention budget tracked separately** 🆕
- Alert when 75% budget consumed
- Hard cap prevents budget overrun
- Cost tracking per agent and per intervention type 🆕

### 3.4 Maintainability

**NFR-ACE-009: Code Quality**

- 90%+ test coverage for ACE components
- **COMPASS-enhanced components have 95%+ coverage** 🆕
- Comprehensive integration tests
- Clear separation of concerns
- Minimal dependencies

**NFR-ACE-010: Observability**

- Evolution metrics exported to monitoring
- **Performance metrics exported to Prometheus** 🆕
- **Intervention metrics tracked and alerted** 🆕
- **ACE-MEM coordination latency monitored** 🆕
- Delta quality tracked and logged
- Cost metrics per agent

**NFR-ACE-011: Configuration**

- All thresholds configurable
- **Intervention thresholds configurable per agent** 🆕
- **ACE-MEM coordination timeouts configurable** 🆕
- Feature flags for granular control
- Configuration hot-reloadable
- Defaults for common scenarios

---

## 4. Data Models

### 4.1 COMPASS-Enhanced Models 🆕

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class PerformanceMetrics(BaseModel):
    """Stage-aware performance metrics (COMPASS ACE-1)"""

    task_id: UUID
    agent_id: UUID
    stage: str  # planning, execution, reflection, verification

    # Stage-specific metrics
    stage_success_rate: float  # 0-1
    stage_error_rate: float  # 0-1
    stage_duration_ms: int
    stage_action_count: int

    # Cross-stage metrics
    overall_progress_velocity: float  # actions per minute
    error_accumulation_rate: float  # errors per stage
    context_staleness_score: float  # 0-1, higher = staler
    intervention_effectiveness: float  # 0-1, from last intervention

    # Baseline comparison
    baseline_delta: dict[str, float]  # metric_name -> deviation from baseline

    recorded_at: datetime

class InterventionRecord(BaseModel):
    """Record of strategic intervention (COMPASS ACE-2)"""

    intervention_id: UUID
    task_id: UUID
    agent_id: UUID

    # Trigger information
    trigger_type: str  # performance_degradation, error_accumulation, context_staleness, capability_mismatch
    trigger_signals: list[str]  # Specific signals that triggered intervention
    trigger_metrics: PerformanceMetrics

    # Decision information
    intervention_type: str  # context_refresh, replan, reflect, capability_switch
    intervention_rationale: str
    decision_confidence: float  # 0-1

    # Execution information
    executed_at: datetime
    execution_duration_ms: int
    execution_status: str  # success, failure, partial

    # Outcome tracking
    pre_intervention_metrics: PerformanceMetrics
    post_intervention_metrics: PerformanceMetrics | None
    effectiveness_delta: float | None  # Improvement score

class ContextQuery(BaseModel):
    """Query to MEM for strategic context (COMPASS ACE-3)"""

    query_id: UUID
    query_type: str  # strategic_decision, error_analysis, capability_evaluation, context_refresh
    task_id: UUID
    agent_id: UUID
    current_metrics: PerformanceMetrics
    focus_areas: list[str]  # e.g., ["errors", "successful_patterns"]
    max_context_tokens: int = 2000

class StrategicContext(BaseModel):
    """Response from MEM with strategic context (COMPASS ACE-3)"""

    query_id: UUID
    task_id: UUID

    # Context from MEM
    relevant_stage_summaries: list[str]  # Stage summaries from current task
    critical_facts: list[str]  # Critical memories flagged by MEM
    error_patterns: list[str]  # Detected error patterns from MEM
    successful_patterns: list[str]  # Successful action patterns
    context_health_score: float  # 0-1, MEM's assessment of context quality

    # Metadata
    retrieval_duration_ms: int
    token_count: int
    retrieved_at: datetime

class CapabilityFitness(BaseModel):
    """Capability fitness scoring (COMPASS ACE-4)"""

    agent_id: UUID
    capability_id: str
    capability_name: str

    # Fitness metrics
    success_rate: float  # 0-1
    error_correlation: float  # 0-1, high = capability causing errors
    usage_frequency: int
    fitness_score: float  # success_rate * (1 - error_correlation)

    # Context
    task_type: str | None  # Fitness can be task-specific
    sample_size: int  # Number of executions used for scoring

    updated_at: datetime

class CapabilityRecommendation(BaseModel):
    """Capability change recommendation (COMPASS ACE-4)"""

    recommendation_id: UUID
    agent_id: UUID
    task_id: UUID

    # Current state
    current_capabilities: list[str]
    underperforming_capabilities: list[str]

    # Recommendations
    capabilities_to_add: list[str]
    capabilities_to_remove: list[str]
    rationale: str
    confidence: float  # 0-1

    # Alternatives considered
    alternatives_evaluated: dict[str, float]  # capability_id -> fitness_score

    generated_at: datetime
```

### 4.2 Original ACE Models (Enhanced)

```python
class ContextSection(BaseModel):
    section_id: str
    category: str  # "strategies", "patterns", "failures", "learnings"
    content: str
    confidence: float  # 0-1
    created_at: datetime
    last_updated: datetime
    usage_count: int = 0
    stage_relevance_map: dict[str, float] | None = None  # 🆕 Link to MEM stages

class ContextPlaybook(BaseModel):
    playbook_id: str
    agent_id: str
    philosophy: str
    version: int
    sections: list[ContextSection]
    evolution_history: list[str]  # Delta IDs
    performance_baseline: dict[str, float]
    intervention_history: list[UUID] = []  # 🆕 Link to interventions
    mem_task_context_id: UUID | None = None  # 🆕 Link to MEM TaskContext
    created_at: datetime
    updated_at: datetime

class ContextDelta(BaseModel):
    delta_id: str
    playbook_id: str
    agent_id: str
    delta_type: str  # "add", "update", "remove"
    target_section: str
    content: str
    rationale: str
    confidence_score: float  # 0-1
    applied: bool = False
    intervention_triggered: bool = False  # 🆕 Was this from intervention?
    created_at: datetime

class ExecutionTrace(BaseModel):
    trace_id: str
    agent_id: str
    task_id: str | None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    duration_ms: int
    success: bool
    feedback_signals: list[str] = []
    stage: str | None = None  # 🆕 Reasoning stage
    performance_metrics: PerformanceMetrics | None = None  # 🆕 Computed metrics
    recorded_at: datetime

class EvolutionStatus(BaseModel):
    agent_id: str
    last_evolution_at: datetime | None
    evolution_count: int
    success_count: int
    failure_count: int
    total_deltas_generated: int
    total_deltas_applied: int
    average_confidence: float
    intervention_count: int = 0  # 🆕
    last_intervention_at: datetime | None = None  # 🆕
    last_error: str | None
```

---

## 5. Database Schema

### 5.1 COMPASS-Enhanced Tables 🆕

```sql
-- Performance metrics table (COMPASS ACE-1)
CREATE TABLE performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    stage VARCHAR(50) NOT NULL,

    -- Stage-specific metrics
    stage_success_rate FLOAT NOT NULL,
    stage_error_rate FLOAT NOT NULL,
    stage_duration_ms INT NOT NULL,
    stage_action_count INT NOT NULL,

    -- Cross-stage metrics
    overall_progress_velocity FLOAT NOT NULL,
    error_accumulation_rate FLOAT NOT NULL,
    context_staleness_score FLOAT NOT NULL,
    intervention_effectiveness FLOAT,

    -- Baseline comparison
    baseline_delta JSONB DEFAULT '{}'::jsonb,

    recorded_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_metrics_task (task_id),
    INDEX idx_metrics_agent (agent_id),
    INDEX idx_metrics_stage (stage),
    INDEX idx_metrics_recorded (recorded_at DESC)
);

-- Intervention records table (COMPASS ACE-2)
CREATE TABLE intervention_records (
    intervention_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,

    -- Trigger information
    trigger_type VARCHAR(50) NOT NULL,
    trigger_signals JSONB NOT NULL,
    trigger_metrics_id UUID REFERENCES performance_metrics(metric_id),

    -- Decision information
    intervention_type VARCHAR(50) NOT NULL,
    intervention_rationale TEXT NOT NULL,
    decision_confidence FLOAT NOT NULL,

    -- Execution information
    executed_at TIMESTAMP NOT NULL,
    execution_duration_ms INT NOT NULL,
    execution_status VARCHAR(20) NOT NULL,

    -- Outcome tracking
    pre_metrics_id UUID REFERENCES performance_metrics(metric_id),
    post_metrics_id UUID REFERENCES performance_metrics(metric_id),
    effectiveness_delta FLOAT,

    INDEX idx_intervention_task (task_id),
    INDEX idx_intervention_agent (agent_id),
    INDEX idx_intervention_type (intervention_type),
    INDEX idx_intervention_executed (executed_at DESC)
);

-- Context queries table (COMPASS ACE-3)
CREATE TABLE context_queries (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_type VARCHAR(50) NOT NULL,
    task_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    current_metrics_id UUID REFERENCES performance_metrics(metric_id),
    focus_areas JSONB NOT NULL,
    max_context_tokens INT NOT NULL,

    -- Response information
    response_json JSONB,  -- Serialized StrategicContext
    retrieval_duration_ms INT,
    token_count INT,

    queried_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_query_task (task_id),
    INDEX idx_query_agent (agent_id),
    INDEX idx_query_type (query_type),
    INDEX idx_query_queried (queried_at DESC)
);

-- Capability fitness table (COMPASS ACE-4)
CREATE TABLE capability_fitness (
    fitness_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    capability_id VARCHAR(255) NOT NULL,
    capability_name VARCHAR(255) NOT NULL,

    -- Fitness metrics
    success_rate FLOAT NOT NULL,
    error_correlation FLOAT NOT NULL,
    usage_frequency INT NOT NULL,
    fitness_score FLOAT NOT NULL,

    -- Context
    task_type VARCHAR(100),
    sample_size INT NOT NULL,

    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (agent_id, capability_id, task_type),
    INDEX idx_fitness_agent (agent_id),
    INDEX idx_fitness_capability (capability_id),
    INDEX idx_fitness_score (fitness_score DESC),
    INDEX idx_fitness_updated (updated_at DESC)
);

-- Capability recommendations table (COMPASS ACE-4)
CREATE TABLE capability_recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    task_id UUID NOT NULL,

    -- Current state
    current_capabilities JSONB NOT NULL,
    underperforming_capabilities JSONB NOT NULL,

    -- Recommendations
    capabilities_to_add JSONB NOT NULL,
    capabilities_to_remove JSONB NOT NULL,
    rationale TEXT NOT NULL,
    confidence FLOAT NOT NULL,

    -- Alternatives
    alternatives_evaluated JSONB,

    generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP,

    INDEX idx_recommendation_agent (agent_id),
    INDEX idx_recommendation_task (task_id),
    INDEX idx_recommendation_generated (generated_at DESC),
    INDEX idx_recommendation_applied (applied, applied_at DESC)
);

-- Performance baselines table (COMPASS ACE-1)
CREATE TABLE performance_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(100),
    stage VARCHAR(50) NOT NULL,

    -- Baseline metrics
    baseline_success_rate FLOAT NOT NULL,
    baseline_error_rate FLOAT NOT NULL,
    baseline_velocity FLOAT NOT NULL,
    baseline_staleness FLOAT NOT NULL,

    -- Statistical properties
    sample_size INT NOT NULL,
    confidence_interval JSONB,  -- {lower: float, upper: float}

    -- Version tracking
    version INT NOT NULL DEFAULT 1,
    established_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (agent_id, task_type, stage, version),
    INDEX idx_baseline_agent (agent_id),
    INDEX idx_baseline_stage (stage),
    INDEX idx_baseline_version (version DESC),
    INDEX idx_baseline_updated (updated_at DESC)
);
```

### 5.2 Enhanced Original Tables

```sql
-- Enhanced context playbooks table
ALTER TABLE context_playbooks
ADD COLUMN intervention_history JSONB DEFAULT '[]'::jsonb,
ADD COLUMN mem_task_context_id UUID,
ADD COLUMN last_performance_check TIMESTAMP;

-- Enhanced context deltas table
ALTER TABLE context_deltas
ADD COLUMN intervention_triggered BOOLEAN DEFAULT FALSE,
ADD COLUMN intervention_id UUID REFERENCES intervention_records(intervention_id);

-- Enhanced execution traces table
ALTER TABLE execution_traces
ADD COLUMN stage VARCHAR(50),
ADD COLUMN performance_metrics_id UUID REFERENCES performance_metrics(metric_id);

-- Enhanced evolution status table
ALTER TABLE evolution_status
ADD COLUMN intervention_count INT DEFAULT 0,
ADD COLUMN last_intervention_at TIMESTAMP,
ADD COLUMN average_intervention_effectiveness FLOAT DEFAULT 0.0;

-- New indexes for COMPASS queries
CREATE INDEX idx_playbooks_mem_context ON context_playbooks(mem_task_context_id)
    WHERE mem_task_context_id IS NOT NULL;
CREATE INDEX idx_deltas_intervention ON context_deltas(intervention_id)
    WHERE intervention_id IS NOT NULL;
CREATE INDEX idx_traces_stage ON execution_traces(stage)
    WHERE stage IS NOT NULL;
```

---

## 6. API Specifications

### 6.1 Performance Monitor API (COMPASS ACE-1) 🆕

```python
class PerformanceMonitor:
    """Monitor agent performance with stage awareness"""

    async def record_metrics(
        self,
        task_id: UUID,
        agent_id: str,
        stage: str,
        metrics: PerformanceMetrics
    ) -> None:
        """Record performance metrics for a stage"""
        pass

    async def get_current_metrics(
        self,
        task_id: UUID,
        agent_id: str
    ) -> PerformanceMetrics:
        """Get latest metrics for task"""
        pass

    async def get_baseline(
        self,
        agent_id: str,
        task_type: str | None,
        stage: str
    ) -> PerformanceBaseline:
        """Get performance baseline for comparison"""
        pass

    async def compute_baseline_delta(
        self,
        current_metrics: PerformanceMetrics,
        baseline: PerformanceBaseline
    ) -> dict[str, float]:
        """Compute deviation from baseline"""
        pass

    async def update_baseline(
        self,
        agent_id: str,
        stage: str,
        metrics_history: list[PerformanceMetrics]
    ) -> PerformanceBaseline:
        """Update rolling baseline from recent metrics"""
        pass
```

### 6.2 Intervention Engine API (COMPASS ACE-2) 🆕

```python
class InterventionEngine:
    """Strategic intervention decision and execution"""

    async def detect_intervention_signals(
        self,
        task_id: UUID,
        agent_id: str,
        current_metrics: PerformanceMetrics
    ) -> list[str]:
        """Detect if intervention signals present"""
        pass

    async def decide_intervention(
        self,
        task_id: UUID,
        agent_id: str,
        signals: list[str],
        strategic_context: StrategicContext
    ) -> InterventionRecord:
        """Decide on intervention type and execute"""
        pass

    async def execute_intervention(
        self,
        intervention: InterventionRecord
    ) -> InterventionRecord:
        """Execute the intervention via Agent Runtime"""
        pass

    async def track_intervention_outcome(
        self,
        intervention_id: UUID,
        post_metrics: PerformanceMetrics
    ) -> float:
        """Track effectiveness of intervention"""
        pass

    async def get_intervention_history(
        self,
        agent_id: str,
        limit: int = 10
    ) -> list[InterventionRecord]:
        """Get recent intervention history"""
        pass
```

### 6.3 ACE-MEM Integration API (COMPASS ACE-3) 🆕

```python
class ACEMemoryInterface:
    """Interface for ACE to query MEM"""

    async def query_strategic_context(
        self,
        query: ContextQuery
    ) -> StrategicContext:
        """Query MEM for strategic context"""
        pass

    async def record_intervention_outcome(
        self,
        intervention_id: UUID,
        success: bool,
        effectiveness_delta: float
    ) -> None:
        """Record intervention outcome in MEM"""
        pass

    async def get_task_context_health(
        self,
        task_id: UUID
    ) -> float:
        """Get MEM's assessment of context health (0-1)"""
        pass

    async def request_context_refresh(
        self,
        task_id: UUID,
        agent_id: str
    ) -> str:
        """Request fresh context compilation from MEM"""
        pass
```

### 6.4 Capability Evaluator API (COMPASS ACE-4) 🆕

```python
class CapabilityEvaluator:
    """Evaluate and recommend agent capabilities"""

    async def evaluate_capability_fitness(
        self,
        agent_id: str,
        task_id: UUID,
        current_capabilities: list[str]
    ) -> dict[str, CapabilityFitness]:
        """Evaluate fitness of current capabilities"""
        pass

    async def recommend_capability_changes(
        self,
        agent_id: str,
        task_id: UUID,
        fitness_scores: dict[str, CapabilityFitness]
    ) -> CapabilityRecommendation:
        """Recommend capability additions/removals"""
        pass

    async def execute_capability_switch(
        self,
        agent_id: str,
        recommendation: CapabilityRecommendation
    ) -> bool:
        """Execute capability switch via Agent Manager"""
        pass

    async def track_switch_effectiveness(
        self,
        agent_id: str,
        recommendation_id: UUID,
        post_metrics: PerformanceMetrics
    ) -> float:
        """Track effectiveness of capability switch"""
        pass
```

### 6.5 Enhanced Context Manager API

```python
class ContextManager:
    """Enhanced with COMPASS Meta-Thinker capabilities"""

    # Original methods
    async def create_playbook(...) -> ContextPlaybook:
        pass
    async def get_playbook(...) -> ContextPlaybook:
        pass
    async def get_execution_context(...) -> str:
        pass
    async def record_execution(...) -> None:
        pass
    async def trigger_evolution(...) -> None:
        pass
    async def get_evolution_status(...) -> EvolutionStatus:
        pass
    async def delete_playbook(...) -> None:
        pass

    # COMPASS-enhanced methods 🆕
    async def get_performance_metrics(
        self,
        agent_id: str,
        task_id: UUID | None = None
    ) -> list[PerformanceMetrics]:
        """Get performance metrics for agent"""
        pass

    async def get_intervention_history(
        self,
        agent_id: str,
        limit: int = 10
    ) -> list[InterventionRecord]:
        """Get intervention history"""
        pass

    async def trigger_intervention_check(
        self,
        task_id: UUID,
        agent_id: str
    ) -> InterventionRecord | None:
        """Manually trigger intervention check"""
        pass
```

---

## 7. JSON-RPC Methods (COMPASS-Enhanced) 🆕

### 7.1 Performance Monitoring Methods

```python
# Get current performance metrics
ace.get_metrics(agent_id: str, task_id: str) -> PerformanceMetrics

# Get performance baseline
ace.get_baseline(agent_id: str, task_type: str, stage: str) -> PerformanceBaseline

# Get metric history
ace.get_metric_history(
    agent_id: str,
    task_id: str,
    start_time: datetime,
    end_time: datetime
) -> list[PerformanceMetrics]
```

### 7.2 Intervention Methods

```python
# Trigger manual intervention check
ace.check_intervention(task_id: str, agent_id: str) -> InterventionRecord | None

# Get intervention history
ace.get_interventions(agent_id: str, limit: int = 10) -> list[InterventionRecord]

# Get intervention effectiveness stats
ace.get_intervention_stats(agent_id: str) -> dict[str, float]
```

### 7.3 Capability Evaluation Methods

```python
# Evaluate current capabilities
ace.evaluate_capabilities(
    agent_id: str,
    task_id: str
) -> dict[str, CapabilityFitness]

# Get capability recommendations
ace.recommend_capability_switch(
    agent_id: str,
    task_id: str
) -> CapabilityRecommendation

# Execute capability switch
ace.switch_capabilities(
    agent_id: str,
    recommendation_id: str
) -> bool
```

### 7.4 ACE-MEM Integration Methods

```python
# Query strategic context from MEM
ace.query_mem_context(
    task_id: str,
    query_type: str,
    focus_areas: list[str]
) -> StrategicContext

# Get context health assessment
ace.get_context_health(task_id: str) -> float

# Request context refresh
ace.refresh_context(task_id: str, agent_id: str) -> str
```

---

## 8. Configuration (COMPASS-Enhanced)

```toml
[ace]
# Feature flags
enabled = false  # Disabled by default
phase = 2  # COMPASS-enhanced phase
compass_enhanced = true  # Enable Meta-Thinker features

# Evolution configuration (original)
evolution_frequency = 10
evolution_mode = "async"
min_confidence_threshold = 0.7
max_context_sections = 20

# Performance monitoring (COMPASS ACE-1) 🆕
[ace.performance_monitoring]
enabled = true
metric_collection_interval_ms = 1000
baseline_sample_size = 50  # Number of executions for baseline
baseline_confidence_level = 0.95
baseline_update_frequency = 50  # Update every N executions

# Intervention engine (COMPASS ACE-2) 🆕
[ace.interventions]
enabled = true
check_frequency_ms = 5000  # Check every 5 seconds

# Trigger thresholds
[ace.interventions.thresholds]
performance_degradation_pct = 0.5  # 50% below baseline
error_rate_multiplier = 2.0  # 2x baseline error rate
error_accumulation_count = 3  # 3+ errors in stage
context_staleness_threshold = 0.7  # 0-1 score
capability_mismatch_threshold = 0.5  # 50%+ actions failing

# Intervention decision
intervention_confidence_threshold = 0.7
intervention_cooldown_seconds = 60  # Min time between interventions
max_interventions_per_task = 5

# ACE-MEM integration (COMPASS ACE-3) 🆕
[ace.mem_integration]
enabled = true
query_timeout_ms = 150
max_context_tokens = 2000
cache_strategic_context = true
cache_ttl_seconds = 300

# Capability evaluation (COMPASS ACE-4) 🆕
[ace.capability_evaluation]
enabled = true
evaluation_frequency = 20  # Evaluate every N executions
fitness_threshold = 0.5  # Below this, recommend switch
min_sample_size = 10  # Min executions before evaluation
capability_switch_cooldown_seconds = 300  # 5 min between switches

# LLM configuration
[ace.llm]
delta_generation_model = "gpt-4o-mini"
intervention_decision_model = "gpt-4o-mini"  # 🆕
capability_evaluation_model = "gpt-4o-mini"  # 🆕
generation_timeout_seconds = 5
max_retries = 3

# Cost controls (enhanced)
[ace.cost_control]
max_monthly_budget_usd = 150  # Increased for Meta-Thinker
alert_threshold_usd = 112  # 75% of budget
intervention_budget_usd = 50  # Separate budget for interventions 🆕
cost_tracking_enabled = true

# Performance tuning
[ace.performance]
batch_evolution = true
batch_size = 10
worker_concurrency = 5
cache_playbooks = true
cache_ttl_seconds = 300
metrics_batch_size = 100  # 🆕 Batch metric writes
intervention_queue_size = 50  # 🆕 Max queued interventions
```

---

## 9. Testing Requirements (COMPASS-Enhanced)

### 9.1 Unit Tests (Enhanced)

**Original Tests:**
- Context playbook CRUD operations
- Delta generation with mocked LLM
- Curator logic (filtering, application, pruning)
- Execution trace capture
- Configuration loading

**COMPASS-Enhanced Tests:** 🆕
- Performance metric computation
- Baseline calculation and comparison
- Intervention trigger detection
- Intervention decision logic
- ACE-MEM query construction
- Capability fitness scoring
- Capability recommendation generation

### 9.2 Integration Tests (Enhanced)

**Original Tests:**
- End-to-end evolution workflow
- Agent lifecycle with ACE enabled
- Task completion with trace capture
- Database transactions and rollback
- Cost tracking and budget enforcement

**COMPASS-Enhanced Tests:** 🆕
- End-to-end intervention workflow (trigger → decision → execution → outcome)
- ACE-MEM coordination (query → response → decision)
- Capability switch execution (evaluate → recommend → switch → validate)
- Multi-stage task with performance tracking
- Intervention effectiveness validation

### 9.3 Performance Tests (Enhanced)

**Original Tests:**
- Playbook retrieval latency
- Delta generation throughput
- Evolution worker scalability
- Database query performance
- Memory usage under load

**COMPASS-Enhanced Tests:** 🆕
- Metric computation latency (<100ms target)
- Intervention detection latency (<50ms target)
- ACE-MEM query latency (<150ms target)
- Capability evaluation latency (<500ms target)
- Concurrent intervention handling
- Database write throughput for metrics

### 9.4 COMPASS Validation Tests 🆕

```python
@pytest.mark.asyncio
async def test_compass_intervention_effectiveness():
    """
    Validate COMPASS target: Interventions improve performance.

    Test scenario:
    - Create task with declining performance
    - Verify intervention triggered
    - Validate post-intervention performance improves
    - Target: 70%+ interventions show improvement
    """
    pass

@pytest.mark.asyncio
async def test_compass_error_recall():
    """
    Validate COMPASS target: 90%+ critical error recall.

    Test scenario:
    - Inject critical errors into task execution
    - Verify ACE detects and tracks errors
    - Validate error patterns identified
    - Target: 90%+ critical errors flagged
    """
    pass

@pytest.mark.asyncio
async def test_compass_ace_mem_coordination():
    """
    Validate COMPASS target: ACE-MEM coordination <200ms.

    Test scenario:
    - Trigger 100 strategic context queries
    - Measure ACE → MEM → ACE latency
    - Target: p95 < 200ms
    """
    pass
```

---

## 10. Success Criteria (COMPASS-Enhanced)

### 10.1 Phase 1 Success (Original + COMPASS)

**Minimum Success:**
- ✅ No production incidents caused by ACE
- ✅ System overhead <8% (increased for Meta-Thinker)
- ✅ Token costs <$150/month for 100 agents
- ✅ At least 1 user reports improved agent performance
- ✅ **Intervention trigger detection working (<50ms latency)** 🆕
- ✅ **ACE-MEM coordination functional (<200ms latency)** 🆕

**Target Success:**
- ✅ **+20% performance improvement on long-horizon tasks (COMPASS target)** 🆕
- ✅ **90%+ critical error recall** 🆕
- ✅ **85%+ intervention precision** 🆕
- ✅ 30% reduction in context-related support tickets
- ✅ Positive user feedback from 3+ customers
- ✅ Context playbooks prevent degradation over 100+ executions

**Exceptional Success:**
- ✅ **+25% performance improvement (exceeds COMPASS)** 🆕
- ✅ **95%+ intervention precision** 🆕
- ✅ **Capability recommendations accepted by users** 🆕
- ✅ Users actively request Phase 2 features
- ✅ Competitive differentiation recognized in market
- ✅ Low maintenance burden (minimal bug reports)

---

## 11. References

**Research Documents:**
- `docs/research/ace-integration-analysis.md` - Original ACE analysis
- `.docs/research/compass-enhancement-analysis.md` - **COMPASS paper analysis** 🆕

**Papers:**
- **COMPASS Paper:** <https://arxiv.org/abs/2510.08790> - **Meta-Thinker architecture** 🆕
- **ACE Paper:** <https://arxiv.org/abs/2510.04618> - Original context engineering

**AgentCore Specs:**
- `docs/agentcore-architecture-and-development-plan.md` - System architecture
- `docs/specs/agent-runtime/spec.md` - Agent Runtime
- `docs/specs/a2a-protocol/spec.md` - A2A Protocol
- **`docs/specs/memory-system/spec.md` - MEM (Context Manager) specification** 🆕

---

**Document Status:** Ready for implementation planning (COMPASS-Enhanced)
**Next Steps:** Run `/sage.plan ace-integration` to generate COMPASS-aware implementation plan
