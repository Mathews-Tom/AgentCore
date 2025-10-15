# Specification: ACE (Agentic Context Engineering) Integration - Phase 1

**Component:** ACE Integration Layer
**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Owner:** Architecture Team
**Created:** 2025-10-12
**Based On:** Research document `docs/research/ace-integration-analysis.md`

---

## 1. Overview

### 1.1 Purpose

Implement Phase 1 of Agentic Context Engineering (ACE) to provide AgentCore with evolving context management capabilities that prevent context degradation in long-running agents and enable systematic self-improvement through execution feedback.

### 1.2 Scope

**In Scope (Phase 1):**

- Context Playbook data structures and storage
- Basic delta generation from execution traces
- Simple curation with confidence-threshold filtering
- Integration with existing Agent Lifecycle Manager
- Execution trace capture in Task Manager
- Background evolution worker
- PostgreSQL storage for contexts and deltas
- Configuration and feature flags

**Out of Scope (Deferred to Phase 2+):**

- A2A context sharing protocol extensions
- Full reflection loop (using confidence filtering instead)
- Real-time context streaming
- DSPy + ACE dual optimization
- Advanced analytics dashboards
- Multi-tenant context isolation
- Cross-agent learning networks

### 1.3 Goals & Success Metrics

**Primary Goals:**

1. Prevent context degradation in long-running agents
2. Enable self-supervised context improvement from execution feedback
3. Reduce manual prompt engineering overhead
4. Provide foundation for advanced ACE features (Phase 2+)

**Success Metrics:**

- Performance improvement: +5-7% on long-running agents (100+ executions)
- Reliability: 30% reduction in context-related failures
- System overhead: <5% additional latency and compute
- Cost: <$100/month token costs for 100 agents
- Quality: Playbooks prevent degradation over 100+ executions

### 1.4 Dependencies

**Required:**

- PostgreSQL database (existing)
- LLM provider via Portkey (existing)
- Agent Runtime Layer (existing)
- Task Manager (existing)
- Python asyncio (existing)

**Optional:**

- DSPy integration (for future dual optimization)
- A2A protocol extensions (for future context sharing)

---

## 2. Functional Requirements

### 2.1 Context Playbook Management

**FR-ACE-001: Context Playbook Data Structure**

The system SHALL maintain structured context playbooks for each agent containing:

- Agent metadata (ID, philosophy, version)
- Context sections (strategies, patterns, failures, learnings)
- Evolution history (list of applied delta IDs)
- Performance baseline metrics
- Timestamps (created, last updated)

**Acceptance Criteria:**

- Playbook can be serialized to/from JSON
- Playbook supports CRUD operations
- Playbook version increments on each evolution
- Playbook can be compiled into execution context string

**FR-ACE-002: Context Section Structure**

Each context section SHALL contain:

- Section ID (unique identifier)
- Category (strategies, patterns, failures, learnings)
- Content (text description)
- Confidence score (0.0-1.0)
- Usage count (tracking frequency of use)
- Timestamps (created, last updated)

**Acceptance Criteria:**

- Sections can be sorted by confidence
- Sections can be filtered by category
- Sections track usage statistics
- Sections support confidence updates

**FR-ACE-003: Playbook Initialization**

The system SHALL automatically create initial playbook when agent is created with:

- Empty sections list
- Philosophy-specific default context (optional)
- Baseline performance metrics initialized to zero
- Version set to 1

**Acceptance Criteria:**

- Playbook created synchronously during agent creation
- Playbook persisted to database
- Playbook accessible via agent ID
- Initialization failure does not block agent creation

### 2.2 Delta Generation

**FR-ACE-004: Execution Trace Capture**

The system SHALL capture execution traces containing:

- Agent ID and task ID
- Input parameters and output results
- Execution duration
- Success/failure status
- Timestamp

**Acceptance Criteria:**

- Traces captured for all task completions
- Traces persisted to database
- Traces retained for 30 days (configurable)
- Trace capture does not impact task execution performance

**FR-ACE-005: Basic Delta Generation**

The system SHALL generate context deltas from execution traces using LLM with:

- Current playbook context as input
- Execution trace summary as feedback
- Performance metrics as indicators
- 1-3 delta suggestions per generation

**Acceptance Criteria:**

- Delta generation completes within 5 seconds
- Deltas include confidence scores
- Deltas include rationale for suggestions
- Delta generation failures are logged and do not crash system

**FR-ACE-006: Delta Structure**

Each context delta SHALL contain:

- Delta ID (unique identifier)
- Agent ID reference
- Delta type (add, update, remove)
- Target section (category name)
- Content (text update)
- Rationale (explanation)
- Confidence score (0.0-1.0)
- Applied status (boolean)
- Timestamp

**Acceptance Criteria:**

- Deltas can be persisted to database
- Deltas can be retrieved by agent ID
- Deltas support filtering by confidence
- Deltas track application status

### 2.3 Context Curation

**FR-ACE-007: Simple Curation Logic**

The system SHALL apply deltas to playbooks using confidence-threshold filtering:

- Minimum confidence threshold (default 0.7, configurable)
- Automatic application of high-confidence deltas
- Rejection of low-confidence deltas
- Version increment on successful application

**Acceptance Criteria:**

- Only deltas above threshold are applied
- Playbook version increments correctly
- Evolution history records applied deltas
- Application failures roll back cleanly

**FR-ACE-008: Delta Application Operations**

The curator SHALL support delta operations:

- **Add**: Append new section to playbook
- **Update**: Modify existing section content
- **Remove**: Delete section from playbook (if confidence drops)

**Acceptance Criteria:**

- Add operation creates new section
- Update operation preserves section ID
- Remove operation soft-deletes section
- All operations are atomic

**FR-ACE-009: Context Limits**

The system SHALL enforce context limits:

- Maximum sections per playbook (default 20)
- Maximum section content length (default 1000 characters)
- Automatic pruning of low-confidence sections when limit reached

**Acceptance Criteria:**

- Limits are configurable per agent
- Pruning preserves high-confidence sections
- Pruning logs removed sections
- Limits prevent unbounded growth

### 2.4 Evolution Workflow

**FR-ACE-010: Periodic Evolution Trigger**

The system SHALL trigger context evolution:

- Every N executions (default 10, configurable)
- Via manual API call
- Via background scheduled job (optional)

**Acceptance Criteria:**

- Execution count tracked per agent
- Evolution triggers asynchronously
- Multiple triggers queued without blocking
- Evolution can be disabled per agent

**FR-ACE-011: Evolution Worker**

The system SHALL provide background evolution worker that:

- Processes evolution queue asynchronously
- Retrieves recent execution traces
- Generates deltas via LLM
- Applies approved deltas to playbook
- Updates performance metrics
- Logs evolution results

**Acceptance Criteria:**

- Worker processes agents sequentially
- Worker handles errors gracefully
- Worker respects cost budgets
- Worker can be paused/resumed

**FR-ACE-012: Evolution Status**

The system SHALL track evolution status:

- Last evolution timestamp
- Evolution success/failure count
- Total deltas generated/applied
- Average confidence scores

**Acceptance Criteria:**

- Status accessible via API
- Status persisted to database
- Status used for monitoring
- Status includes error details

### 2.5 Integration Points

**FR-ACE-013: Agent Lifecycle Integration**

The Agent Lifecycle Manager SHALL:

- Initialize playbook on agent creation
- Provide playbook context during execution
- Trigger evolution based on execution count
- Cleanup playbook on agent termination

**Acceptance Criteria:**

- Integration does not break existing functionality
- ACE can be disabled via feature flag
- Integration adds <100ms latency to agent creation
- Integration is backwards compatible

**FR-ACE-014: Task Manager Integration**

The Task Manager SHALL:

- Capture execution traces on task completion
- Include trace data in task results
- Trigger evolution worker notifications
- Maintain trace retention policy

**Acceptance Criteria:**

- Trace capture adds <10ms overhead
- Trace capture failures do not fail tasks
- Traces linked to task IDs
- Integration is opt-in via configuration

**FR-ACE-015: Execution Context Injection**

The system SHALL inject playbook context into agent execution:

- Compile playbook sections into formatted string
- Append to agent system prompt or context
- Include section categories as headers
- Sort by confidence (high to low)

**Acceptance Criteria:**

- Context injection adds <50ms latency
- Context format is readable by LLM
- Context size is monitored and limited
- Injection can be customized per philosophy

---

## 3. Non-Functional Requirements

### 3.1 Performance

**NFR-ACE-001: Latency Limits**

- Playbook retrieval: <50ms (p95)
- Delta generation: <5s (p95)
- Delta application: <100ms (p95)
- Context injection: <50ms (p95)

**NFR-ACE-002: Throughput**

- Support 100+ concurrent agents
- Handle 1000+ executions per hour
- Process 100+ evolution cycles per hour
- Store 10K+ execution traces

**NFR-ACE-003: Resource Usage**

- CPU overhead: <5% of baseline
- Memory overhead: <100MB per 100 agents
- Database storage: <10MB per agent
- LLM token usage: <$100/month for 100 agents

### 3.2 Reliability

**NFR-ACE-004: Availability**

- ACE failures do not impact core agent execution
- Graceful degradation when LLM unavailable
- Automatic retry with exponential backoff
- Circuit breaker for cost protection

**NFR-ACE-005: Data Integrity**

- Playbook updates are atomic
- Delta application is transactional
- Trace capture does not lose data
- Database migrations are reversible

**NFR-ACE-006: Fault Tolerance**

- Evolution worker restarts on crash
- Queued evolutions persist across restarts
- Partial updates roll back cleanly
- Error states are recoverable

### 3.3 Security

**NFR-ACE-007: Data Isolation**

- Playbooks isolated per agent
- Traces contain no sensitive data (or are encrypted)
- Delta content sanitized for injection attacks
- Database access follows least-privilege

**NFR-ACE-008: Cost Control**

- Monthly token budget enforced
- Alert when 75% budget consumed
- Hard cap prevents budget overrun
- Cost tracking per agent

### 3.4 Maintainability

**NFR-ACE-009: Code Quality**

- 90%+ test coverage for ACE components
- Comprehensive integration tests
- Clear separation of concerns
- Minimal dependencies

**NFR-ACE-010: Observability**

- Evolution metrics exported to monitoring
- Delta quality tracked and logged
- Performance metrics per agent
- Cost metrics per agent

**NFR-ACE-011: Configuration**

- All thresholds configurable
- Feature flags for granular control
- Configuration hot-reloadable
- Defaults for common scenarios

---

## 4. Data Models

### 4.1 Context Playbook

```python
class ContextSection(BaseModel):
    section_id: str
    category: str  # "strategies", "patterns", "failures", "learnings"
    content: str
    confidence: float  # 0.0-1.0
    created_at: datetime
    last_updated: datetime
    usage_count: int = 0

class ContextPlaybook(BaseModel):
    playbook_id: str
    agent_id: str
    philosophy: str  # "react", "cot", "autonomous", "multi_agent"
    version: int
    sections: List[ContextSection]
    evolution_history: List[str]  # Delta IDs
    performance_baseline: Dict[str, float]
    created_at: datetime
    updated_at: datetime
```

### 4.2 Context Delta

```python
class ContextDelta(BaseModel):
    delta_id: str
    playbook_id: str
    agent_id: str
    delta_type: str  # "add", "update", "remove"
    target_section: str
    content: str
    rationale: str
    confidence_score: float  # 0.0-1.0
    applied: bool = False
    created_at: datetime
```

### 4.3 Execution Trace

```python
class ExecutionTrace(BaseModel):
    trace_id: str
    agent_id: str
    task_id: Optional[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: int
    success: bool
    feedback_signals: List[str] = []
    recorded_at: datetime
```

### 4.4 Evolution Status

```python
class EvolutionStatus(BaseModel):
    agent_id: str
    last_evolution_at: Optional[datetime]
    evolution_count: int
    success_count: int
    failure_count: int
    total_deltas_generated: int
    total_deltas_applied: int
    average_confidence: float
    last_error: Optional[str]
```

---

## 5. Database Schema

### 5.1 Tables

```sql
-- Context playbooks table
CREATE TABLE context_playbooks (
    playbook_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    philosophy VARCHAR(50) NOT NULL,
    version INT NOT NULL DEFAULT 1,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    evolution_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    performance_baseline JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_agent FOREIGN KEY (agent_id)
        REFERENCES agent_records(agent_id) ON DELETE CASCADE
);

-- Context deltas table
CREATE TABLE context_deltas (
    delta_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    playbook_id UUID NOT NULL REFERENCES context_playbooks(playbook_id) ON DELETE CASCADE,
    agent_id VARCHAR(255) NOT NULL,
    delta_type VARCHAR(20) NOT NULL CHECK (delta_type IN ('add', 'update', 'remove')),
    target_section VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    rationale TEXT,
    confidence_score FLOAT NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Execution traces table
CREATE TABLE execution_traces (
    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    task_id UUID,
    inputs JSONB,
    outputs JSONB,
    duration_ms INT,
    success BOOLEAN,
    feedback_signals JSONB DEFAULT '[]'::jsonb,
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Evolution status tracking table
CREATE TABLE evolution_status (
    agent_id VARCHAR(255) PRIMARY KEY,
    last_evolution_at TIMESTAMP,
    evolution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    total_deltas_generated INT DEFAULT 0,
    total_deltas_applied INT DEFAULT 0,
    average_confidence FLOAT DEFAULT 0.0,
    last_error TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### 5.2 Indexes

```sql
CREATE INDEX idx_playbooks_agent ON context_playbooks(agent_id);
CREATE INDEX idx_playbooks_updated ON context_playbooks(updated_at DESC);

CREATE INDEX idx_deltas_playbook ON context_deltas(playbook_id);
CREATE INDEX idx_deltas_agent ON context_deltas(agent_id);
CREATE INDEX idx_deltas_applied ON context_deltas(applied, created_at DESC);

CREATE INDEX idx_traces_agent ON execution_traces(agent_id);
CREATE INDEX idx_traces_recorded ON execution_traces(recorded_at DESC);
CREATE INDEX idx_traces_task ON execution_traces(task_id) WHERE task_id IS NOT NULL;

CREATE INDEX idx_evolution_status_updated ON evolution_status(updated_at DESC);
```

---

## 6. API Specifications

### 6.1 Context Manager API

```python
class ContextManager:
    """Core ACE Phase 1 service"""

    async def create_playbook(
        self,
        agent_id: str,
        philosophy: str,
        initial_sections: List[ContextSection] = []
    ) -> ContextPlaybook:
        """Initialize new context playbook"""
        pass

    async def get_playbook(self, agent_id: str) -> ContextPlaybook:
        """Retrieve existing playbook"""
        pass

    async def get_execution_context(self, agent_id: str) -> str:
        """Compile playbook into execution context string"""
        pass

    async def record_execution(self, trace: ExecutionTrace) -> None:
        """Capture execution trace for evolution"""
        pass

    async def trigger_evolution(self, agent_id: str) -> None:
        """Queue agent for context evolution"""
        pass

    async def get_evolution_status(self, agent_id: str) -> EvolutionStatus:
        """Get evolution statistics"""
        pass

    async def delete_playbook(self, agent_id: str) -> None:
        """Cleanup playbook on agent termination"""
        pass
```

### 6.2 Delta Generator API

```python
class DeltaGenerator:
    """LLM-based delta generation"""

    async def generate_deltas(
        self,
        playbook: ContextPlaybook,
        traces: List[ExecutionTrace],
        performance_metrics: Dict[str, float]
    ) -> List[ContextDelta]:
        """Generate context improvement deltas"""
        pass

    def _build_generation_prompt(
        self,
        context: str,
        traces: List[ExecutionTrace],
        metrics: Dict[str, float]
    ) -> str:
        """Construct prompt for LLM"""
        pass

    def _parse_deltas(
        self,
        llm_response: str,
        agent_id: str
    ) -> List[ContextDelta]:
        """Parse LLM response into structured deltas"""
        pass
```

### 6.3 Curator API

```python
class SimpleCurator:
    """Confidence-based delta curation"""

    async def apply_deltas(
        self,
        playbook: ContextPlaybook,
        deltas: List[ContextDelta]
    ) -> ContextPlaybook:
        """Apply high-confidence deltas to playbook"""
        pass

    def _filter_by_confidence(
        self,
        deltas: List[ContextDelta],
        threshold: float
    ) -> List[ContextDelta]:
        """Filter deltas by confidence threshold"""
        pass

    def _apply_delta(
        self,
        playbook: ContextPlaybook,
        delta: ContextDelta
    ) -> None:
        """Apply single delta operation"""
        pass

    def _prune_low_confidence_sections(
        self,
        playbook: ContextPlaybook,
        max_sections: int
    ) -> None:
        """Remove low-confidence sections when limit reached"""
        pass
```

---

## 7. Configuration

### 7.1 Configuration Structure

```toml
[ace]
# Feature flag
enabled = false  # Disabled by default, opt-in per agent
phase = 1  # Current implementation phase

# Evolution frequency
evolution_frequency = 10  # Evolve every N executions
evolution_mode = "async"  # "async" or "scheduled"

# Curation thresholds
min_confidence_threshold = 0.7
max_context_sections = 20

# LLM configuration
delta_generation_model = "gpt-4o-mini"  # Cost-effective model
generation_timeout_seconds = 5
max_retries = 3

# Cost controls
max_monthly_budget_usd = 100
alert_threshold_usd = 75
cost_tracking_enabled = true

# Storage configuration
[ace.storage]
playbook_table = "context_playbooks"
delta_table = "context_deltas"
trace_table = "execution_traces"
status_table = "evolution_status"

# Retention policies
trace_retention_days = 30
delta_retention_days = 90
failed_evolution_retention_days = 7

# Performance tuning
[ace.performance]
batch_evolution = true  # Process in batches
batch_size = 10
worker_concurrency = 5
cache_playbooks = true
cache_ttl_seconds = 300
```

---

## 8. Testing Requirements

### 8.1 Unit Tests

- Context playbook CRUD operations
- Delta generation with mocked LLM
- Curator logic (filtering, application, pruning)
- Execution trace capture
- Configuration loading

### 8.2 Integration Tests

- End-to-end evolution workflow
- Agent lifecycle with ACE enabled
- Task completion with trace capture
- Database transactions and rollback
- Cost tracking and budget enforcement

### 8.3 Performance Tests

- Playbook retrieval latency
- Delta generation throughput
- Evolution worker scalability
- Database query performance
- Memory usage under load

### 8.4 Failure Tests

- LLM unavailability
- Database connection failures
- Invalid delta application
- Budget exhaustion
- Worker crash recovery

---

## 9. Deployment Requirements

### 9.1 Database Migration

- Alembic migration script for new tables
- Migration reversibility (downgrade)
- Data seeding for testing
- Index creation for performance

### 9.2 Configuration Deployment

- Add ACE section to config.toml
- Environment variable support
- Feature flag management
- Cost monitoring alerts

### 9.3 Monitoring & Observability

- Evolution success/failure metrics
- Delta generation latency
- Token cost per agent
- Playbook version tracking
- Error rate alerts

---

## 10. Success Criteria

### 10.1 Phase 1 Minimum Success

- ✅ No production incidents caused by ACE
- ✅ System overhead <5% (latency + compute)
- ✅ Token costs <$100/month for 100 agents
- ✅ At least 1 user reports improved agent performance

### 10.2 Phase 1 Target Success

- ✅ +5-7% performance improvement (validated A/B test)
- ✅ 30% reduction in context-related support tickets
- ✅ Positive user feedback from 3+ customers
- ✅ Context playbooks prevent degradation over 100+ executions

### 10.3 Phase 1 Exceptional Success

- ✅ +8-10% performance improvement
- ✅ Users actively request Phase 2 features
- ✅ Competitive differentiation recognized in market
- ✅ Low maintenance burden (minimal bug reports)

---

## 11. Future Considerations (Phase 2+)

**Not in Phase 1, but designed to support:**

- Full reflection loop (Generator → Reflector → Curator)
- A2A context sharing protocol
- Cross-agent learning networks
- DSPy + ACE dual optimization
- Real-time context streaming
- Advanced analytics and visualization
- Multi-tenant context isolation

**Design Principles for Future Expansion:**

- Modular architecture allows incremental enhancement
- Database schema supports versioning and evolution
- API designed for backward compatibility
- Configuration supports feature flags for new capabilities

---

## 12. References

- **Research Document:** `docs/research/ace-integration-analysis.md`
- **ACE Paper:** <https://arxiv.org/abs/2510.04618>
- **AgentCore Architecture:** `docs/agentcore-architecture-and-development-plan.md`
- **Agent Runtime Spec:** `docs/specs/agent-runtime/spec.md`
- **A2A Protocol Spec:** `docs/specs/a2a-protocol/spec.md`

---

**Document Status:** Ready for implementation planning
**Next Steps:** Run `/sage.plan` to generate implementation plan
