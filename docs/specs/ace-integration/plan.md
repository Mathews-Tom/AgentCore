# Implementation Plan: ACE (Agentic Context Engineering) Integration - Phase 1

**Source:** `docs/specs/ace-integration/spec.md`
**Date:** 2025-10-12
**Version:** 1.0
**Status:** Ready for Implementation

---

## 1. Executive Summary

### Business Alignment

ACE Phase 1 integration addresses critical production challenges in long-running agent deployments where context degradation leads to performance decay and increased manual maintenance overhead. By implementing self-supervised context evolution, AgentCore will provide the first A2A-compliant framework with native context engineering capabilities, offering significant competitive differentiation.

### Technical Approach

Implement a modular, opt-in context evolution system consisting of four core components:

1. **ContextManager** - Central orchestrator for playbook lifecycle
2. **DeltaGenerator** - LLM-based improvement suggestion engine
3. **SimpleCurator** - Confidence-threshold based delta approval
4. **Database Layer** - PostgreSQL storage with 4 new tables

The implementation leverages existing AgentCore infrastructure (PostgreSQL, Portkey, async workers) to minimize new dependencies and integration complexity.

### Key Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Performance Improvement** | +5-7% | A/B testing on long-running agents (100+ executions) |
| **System Overhead** | <5% | Latency and compute monitoring |
| **Cost Control** | <$100/month for 100 agents | Token usage tracking via Portkey |
| **Reliability** | 30% reduction in context failures | Support ticket analysis |
| **Quality** | Zero production incidents | Incident tracking |

---

## 2. Technology Stack

### Recommended Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Runtime** | Python | 3.12+ | Existing AgentCore language, excellent async support |
| **Data Models** | Pydantic | 2.x | Type safety, validation, JSON serialization (already in use) |
| **Database** | PostgreSQL | 14+ | JSONB support for flexible schemas, existing infrastructure |
| **Migrations** | Alembic | Latest | Already integrated, supports reversible migrations |
| **LLM Gateway** | Portkey | Latest | Cost control, caching, multi-provider (already in use) |
| **LLM Model** | GPT-4o-mini | Latest | Cost-effective ($0.15/1M tokens), sufficient quality for delta generation |
| **Async Workers** | Python asyncio | Built-in | No new dependencies, integrates with existing FastAPI |
| **Testing** | pytest + pytest-asyncio | Latest | Existing test framework |

**Total New Dependencies:** 0 (all technologies already in AgentCore stack)

### Alternatives Considered

**Option 2: Vector Database for Context Search**

- **Pros:** Semantic search, similarity matching, scalable
- **Cons:** New infrastructure dependency, added complexity, not needed for Phase 1
- **Decision:** Defer to Phase 2+ if semantic search becomes requirement

**Option 3: Dedicated Context Service (Microservice)**

- **Pros:** Independent scaling, service isolation
- **Cons:** Increased operational overhead, network latency, premature optimization
- **Decision:** Start monolithic, extract to service if proven bottleneck

**Option 4: GPT-4 for Delta Generation**

- **Pros:** Higher quality deltas, better reasoning
- **Cons:** 10x higher cost ($3/1M tokens vs $0.15/1M), slower latency
- **Decision:** Use GPT-4o-mini for Phase 1, upgrade to GPT-4 if quality insufficient

---

## 3. Architecture

### System Design

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                    AgentCore + ACE Integration                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  Agent Lifecycle     │         │   Task Manager       │
│  Manager             │         │   (A2A Protocol)     │
│                      │         │                      │
│  - create_agent()    │         │  - complete_task()   │
│  - monitor_agent()   │         │  - capture traces    │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           │ Initialize/Trigger             │ Record Execution
           ▼                                ▼
    ┌─────────────────────────────────────────────┐
    │         Context Manager (NEW)               │
    │  ┌──────────────────────────────────────┐   │
    │  │  create_playbook()                   │   │
    │  │  get_execution_context()             │   │
    │  │  record_execution()                  │   │
    │  │  trigger_evolution()                 │   │
    │  │  get_evolution_status()              │   │
    │  └──────────────────────────────────────┘   │
    └──────┬──────────────────────┬───────────────┘
           │                      │
           │ Queue                │ Retrieve/Update
           ▼                      ▼
    ┌─────────────┐        ┌──────────────┐
    │  Evolution  │        │  PostgreSQL  │
    │  Worker     │◄───────┤  Database    │
    │  (Async)    │        │              │
    └──────┬──────┘        │  - playbooks │
           │               │  - deltas    │
           │               │  - traces    │
           ▼               │  - status    │
    ┌─────────────────────┐               │
    │  Delta Generator    │               │
    │  ┌───────────────┐  │               │
    │  │ LLM Provider  │  │               │
    │  │ (via Portkey) │  │               │
    │  └───────────────┘  │               │
    │  - generate_deltas()│               │
    └──────┬──────────────┘               │
           │ Generated Deltas             │
           ▼                              │
    ┌─────────────────────┐               │
    │  Simple Curator     │               │
    │  ┌───────────────┐  │               │
    │  │ Confidence    │  │               │
    │  │ Filtering     │  │               │
    │  └───────────────┘  │               │
    │  - apply_deltas()   │               │
    │  - prune_sections() │               │
    └──────┬──────────────┘               │
           │ Approved Deltas              │
           └──────────────────────────────┘
                        Update Playbook

┌──────────────────────────────────────────────────────────┐
│  Monitoring & Observability                              │
│  - Evolution success/failure rates                       │
│  - Delta quality metrics (confidence scores)             │
│  - Token costs per agent                                 │
│  - Performance impact (latency, throughput)              │
└──────────────────────────────────────────────────────────┘
```

### Architecture Decisions

**Pattern: Modular Monolith**

- **Rationale:** ACE is tightly coupled with agent lifecycle and task execution; microservice overhead not justified for Phase 1
- **Benefits:** Simpler deployment, lower latency, shared database transactions
- **Future:** Can extract to service if scaling requirements emerge

**Integration: Event-Driven + Direct Calls**

- **Rationale:** Mix of synchronous calls (playbook retrieval) and asynchronous events (evolution triggers)
- **Benefits:** Predictable performance for hot path, non-blocking for background tasks
- **Implementation:** asyncio.Queue for evolution triggers, direct async calls for CRUD

**Data Flow: Write-Through Cache**

- **Rationale:** Playbooks cached in memory after retrieval, invalidated on updates
- **Benefits:** <50ms playbook retrieval for execution context injection
- **Limitation:** Cache per-process; use Redis for multi-process deployments (Phase 2)

### Key Components

#### 1. ContextManager Service

**Purpose:** Central orchestrator for all ACE operations

**Responsibilities:**

- Playbook lifecycle management (CRUD)
- Execution trace recording
- Evolution trigger queue management
- Status tracking and reporting
- Cost budget enforcement

**Technology:** Python async service, integrated into existing `services/` layer

**Integration Points:**

- Called by AgentLifecycleManager (playbook initialization, context retrieval)
- Called by TaskManager (execution trace capture)
- Background worker (evolution processing)

#### 2. DeltaGenerator

**Purpose:** Generate context improvement suggestions using LLM

**Responsibilities:**

- Build generation prompts from playbook + traces
- Call LLM via Portkey gateway
- Parse structured deltas from LLM response
- Assign confidence scores
- Handle LLM failures gracefully

**Technology:** Python class with LLM client integration

**LLM Prompt Strategy:**

```python
prompt = f"""
Analyze this agent's execution and suggest context improvements.

Current Context:
{playbook.get_context_for_execution()}

Recent Executions (last 10):
{trace_summary}

Performance Metrics:
- Success Rate: {metrics['success_rate']}%
- Avg Duration: {metrics['avg_duration_ms']}ms

Suggest 1-3 specific improvements focusing on:
1. Strategies that worked well (add to context)
2. Common patterns observed (document as learnings)
3. Failure modes to avoid (add to failures section)

Format each suggestion as:
- Category: [strategies|patterns|failures|learnings]
- Content: [Specific, actionable insight]
- Confidence: [0.0-1.0 based on evidence strength]
- Rationale: [Why this improvement matters]
"""
```

#### 3. SimpleCurator

**Purpose:** Filter and apply high-confidence deltas to playbooks

**Responsibilities:**

- Filter deltas by confidence threshold (default 0.7)
- Apply CRUD operations to playbook sections
- Prune low-confidence sections when limit reached
- Increment playbook version
- Record evolution history

**Technology:** Python class with transactional database updates

**Curation Logic:**

```python
async def apply_deltas(playbook, deltas):
    # Filter by confidence
    approved = [d for d in deltas if d.confidence >= 0.7]

    # Apply in transaction
    async with db.transaction():
        for delta in approved:
            if delta.type == "add":
                playbook.sections.append(create_section(delta))
            elif delta.type == "update":
                update_section(playbook, delta)
            elif delta.type == "remove":
                remove_section(playbook, delta)

        # Prune if over limit
        if len(playbook.sections) > max_sections:
            prune_low_confidence(playbook)

        playbook.version += 1
        await save_playbook(playbook)
```

#### 4. Evolution Worker

**Purpose:** Background processor for async evolution cycles

**Responsibilities:**

- Consume evolution queue (agent IDs)
- Retrieve recent execution traces
- Generate deltas via DeltaGenerator
- Apply deltas via SimpleCurator
- Update evolution status
- Respect cost budgets

**Technology:** Python async worker, single instance per deployment

**Worker Loop:**

```python
async def evolution_worker():
    while True:
        agent_id = await evolution_queue.get()

        try:
            # Check budget
            if monthly_cost >= budget_cap:
                logger.warning("Budget exhausted, skipping")
                continue

            # Load playbook + traces
            playbook = await get_playbook(agent_id)
            traces = await get_recent_traces(agent_id, limit=10)

            # Generate deltas
            deltas = await generator.generate_deltas(playbook, traces)

            # Apply deltas
            updated = await curator.apply_deltas(playbook, deltas)

            # Update status
            await update_evolution_status(agent_id, success=True)

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            await update_evolution_status(agent_id, success=False, error=str(e))
```

---

## 4. Technical Specification

### Data Model

#### Entity Relationship Diagram

```plaintext
┌─────────────────────┐         ┌─────────────────────┐
│  agent_records      │◄────────│  context_playbooks  │
│  (existing)         │1       1│  (NEW)              │
│                     │         │                     │
│  - agent_id (PK)    │         │  - playbook_id (PK) │
│  - philosophy       │         │  - agent_id (FK)    │
│  - status           │         │  - philosophy       │
└─────────────────────┘         │  - version          │
                                │  - sections (JSONB) │
                                │  - evolution_history│
                                └──────────┬──────────┘
                                          1│
                                           │
                                          *│
                                ┌──────────▼──────────┐
                                │  context_deltas     │
                                │  (NEW)              │
                                │                     │
                                │  - delta_id (PK)    │
                                │  - playbook_id (FK) │
                                │  - agent_id         │
                                │  - delta_type       │
                                │  - confidence_score │
                                │  - applied          │
                                └─────────────────────┘

┌─────────────────────┐
│  task_records       │
│  (existing)         │         ┌─────────────────────┐
│                     │         │  execution_traces   │
│  - task_id (PK)     │1       *│  (NEW)              │
│  - agent_id         │◄────────│                     │
│  - status           │         │  - trace_id (PK)    │
└─────────────────────┘         │  - agent_id         │
                                │  - task_id (FK)     │
                                │  - inputs (JSONB)   │
                                │  - outputs (JSONB)  │
                                │  - duration_ms      │
                                │  - success          │
                                └─────────────────────┘

┌──────────────────────┐
│  evolution_status    │
│  (NEW)               │
│                      │
│  - agent_id (PK)     │
│  - last_evolution_at │
│  - evolution_count   │
│  - success_count     │
│  - total_deltas_*    │
│  - average_confidence│
└──────────────────────┘
```

#### Validation Rules

**ContextPlaybook:**

- `agent_id` must reference existing agent
- `philosophy` must match agent's philosophy
- `version` must increment monotonically
- `sections` array max length: 20 (configurable)
- `evolution_history` append-only

**ContextSection:**

- `confidence` must be 0.0-1.0
- `content` max length: 1000 characters
- `category` must be one of: strategies, patterns, failures, learnings
- `usage_count` incremented on each agent execution

**ContextDelta:**

- `confidence_score` must be 0.0-1.0
- `delta_type` must be: add, update, remove
- `applied` immutable after set to true
- `target_section` must exist for update/remove operations

**ExecutionTrace:**

- `agent_id` must reference existing agent
- `task_id` optional (can be NULL for non-task executions)
- `duration_ms` must be positive
- `recorded_at` auto-set, cannot be future

#### Indexing Strategy

**Performance-Critical Indexes:**

```sql
-- Hot path: playbook retrieval by agent_id
CREATE INDEX idx_playbooks_agent ON context_playbooks(agent_id);

-- Evolution worker: find recent traces
CREATE INDEX idx_traces_agent_recorded ON execution_traces(agent_id, recorded_at DESC);

-- Curator: find unapplied deltas
CREATE INDEX idx_deltas_unapplied ON context_deltas(applied, playbook_id) WHERE applied = FALSE;

-- Monitoring: evolution status queries
CREATE INDEX idx_evolution_status_updated ON evolution_status(updated_at DESC);
```

**JSONB GIN Indexes (Optional, Phase 2):**

```sql
-- Full-text search on playbook sections
CREATE INDEX idx_playbooks_sections_gin ON context_playbooks USING GIN (sections jsonb_path_ops);
```

### API Design

#### Top 6 Critical Endpoints

**1. Create Playbook**

```python
POST /internal/ace/playbooks
{
  "agent_id": "agent-123",
  "philosophy": "react",
  "initial_sections": []  # Optional
}

Response 201:
{
  "playbook_id": "pb-uuid",
  "agent_id": "agent-123",
  "version": 1,
  "created_at": "2025-10-12T10:00:00Z"
}

Errors:
- 400: Agent ID invalid or missing
- 409: Playbook already exists for agent
- 500: Database failure
```

**2. Get Execution Context**

```python
GET /internal/ace/playbooks/{agent_id}/context

Response 200:
{
  "agent_id": "agent-123",
  "context": "## Strategies\n...\n## Patterns\n...",
  "version": 5,
  "section_count": 12
}

Errors:
- 404: Playbook not found
- 500: Compilation failure
```

**3. Record Execution Trace**

```python
POST /internal/ace/traces
{
  "agent_id": "agent-123",
  "task_id": "task-uuid",  # Optional
  "inputs": {...},
  "outputs": {...},
  "duration_ms": 1234,
  "success": true
}

Response 201:
{
  "trace_id": "trace-uuid",
  "recorded_at": "2025-10-12T10:05:00Z"
}

Errors:
- 400: Invalid trace data
- 500: Database failure
```

**4. Trigger Evolution**

```python
POST /internal/ace/playbooks/{agent_id}/evolve

Response 202:
{
  "agent_id": "agent-123",
  "queued_at": "2025-10-12T10:10:00Z",
  "queue_position": 3
}

Errors:
- 404: Playbook not found
- 429: Evolution already in progress
- 503: Worker unavailable
```

**5. Get Evolution Status**

```python
GET /internal/ace/playbooks/{agent_id}/status

Response 200:
{
  "agent_id": "agent-123",
  "last_evolution_at": "2025-10-12T09:00:00Z",
  "evolution_count": 15,
  "success_count": 14,
  "failure_count": 1,
  "total_deltas_generated": 42,
  "total_deltas_applied": 35,
  "average_confidence": 0.78,
  "last_error": null
}

Errors:
- 404: Status not found
```

**6. Delete Playbook**

```python
DELETE /internal/ace/playbooks/{agent_id}

Response 204: (No Content)

Errors:
- 404: Playbook not found
- 500: Deletion failed (check foreign keys)
```

**Note:** All endpoints are internal (`/internal/` prefix) and require service-to-service authentication. Not exposed in public API.

### Security

#### Authentication & Authorization

**Service-to-Service Auth:**

- All ACE endpoints require JWT token with `service:internal` scope
- Tokens issued by existing AgentCore auth system
- No user-facing API endpoints (internal use only)

**Data Access Control:**

- Playbooks scoped to agent_id (no cross-agent access)
- Execution traces contain no PII (sanitized at capture)
- LLM prompts sanitized to remove sensitive data

#### Secrets Management

**LLM API Keys:**

- Stored in environment variables (`LLM_API_KEY`)
- Accessed via Portkey gateway (keys not in application code)
- Rotated quarterly via configuration management

**Database Credentials:**

- Managed by existing PostgreSQL connection pool
- No changes to current secrets management

#### Data Encryption

**In Transit:**

- All HTTP communication over TLS 1.3
- LLM API calls via HTTPS (Portkey enforces)

**At Rest:**

- PostgreSQL encryption at rest (existing configuration)
- JSONB fields contain no PII or sensitive data
- Execution traces sanitized before storage

#### Compliance Considerations

**Data Retention:**

- Execution traces: 30 days (configurable)
- Context deltas: 90 days (configurable)
- Playbooks: Agent lifetime (deleted on agent termination)

**Audit Trail:**

- All evolution events logged with timestamps
- Delta application tracked in `evolution_history`
- Status changes recorded in `evolution_status` table

**GDPR Compliance:**

- No personal data in playbooks or traces
- Right to deletion: CASCADE delete on agent removal
- Data portability: Playbooks exportable as JSON

### Performance

#### Caching Strategy

**Playbook Cache:**

```python
# In-memory LRU cache (per process)
playbook_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes

async def get_playbook_cached(agent_id):
    if agent_id in playbook_cache:
        return playbook_cache[agent_id]

    playbook = await db.fetch_playbook(agent_id)
    playbook_cache[agent_id] = playbook
    return playbook

# Invalidate on update
async def update_playbook(playbook):
    await db.save_playbook(playbook)
    del playbook_cache[playbook.agent_id]
```

**LLM Response Cache (via Portkey):**

- Semantic caching enabled (85%+ hit rate for similar prompts)
- 24-hour TTL for delta generation responses
- Cost savings: ~50% reduction in LLM API calls

#### Database Optimization

**Connection Pooling:**

- Reuse existing async connection pool (sqlalchemy.ext.asyncio)
- Min connections: 5, Max connections: 20

**Query Optimization:**

```sql
-- Prepared statement for hot path
PREPARE get_playbook_by_agent (text) AS
  SELECT * FROM context_playbooks WHERE agent_id = $1;

-- Batch insert for traces
INSERT INTO execution_traces (agent_id, task_id, ...)
  VALUES ($1, $2, ...), ($3, $4, ...), ...
  ON CONFLICT DO NOTHING;
```

**JSONB Performance:**

- Use `jsonb_set()` for incremental updates (avoid full replacement)
- GIN indexes on frequently queried JSONB paths (Phase 2)

#### Scaling Approach

**Horizontal Scaling:**

- ContextManager stateless (can run multiple instances)
- Evolution worker: Single instance per deployment (Phase 1)
  - Phase 2: Multiple workers with distributed queue (Redis)
- Database: Read replicas for trace queries (Phase 2)

**Vertical Scaling:**

- Evolution worker CPU-bound (LLM API calls are I/O)
- Increase worker concurrency before adding instances

**Load Targets:**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Playbook Retrieval | <50ms p95 | Load testing with 1000 RPS |
| Delta Generation | <5s p95 | End-to-end timer with LLM |
| Evolution Throughput | 100+ agents/hour | Worker monitoring |
| Concurrent Agents | 100+ | Resource utilization <80% |

---

## 5. Development Setup

### Required Tools & Versions

```bash
# Core tooling (already installed)
- Python 3.12+
- PostgreSQL 14+
- Redis 7+ (optional for Phase 2)
- uv (package manager)

# Development tools
- pytest 8.x
- pytest-asyncio 0.23+
- pytest-cov 5.x
- Alembic 1.13+
- black (code formatting)
- ruff (linting)
- mypy (type checking)
```

### Local Environment Setup

**1. Database Initialization:**

```bash
# Create ACE tables
uv run alembic upgrade head

# Seed test data (optional)
uv run python scripts/seed_ace_data.py
```

**2. Environment Variables:**

```bash
# .env file
ACE_ENABLED=true
ACE_DELTA_MODEL=gpt-4o-mini
ACE_MIN_CONFIDENCE=0.7
ACE_MAX_SECTIONS=20
ACE_EVOLUTION_FREQUENCY=10
ACE_MONTHLY_BUDGET_USD=100

# LLM credentials (via Portkey)
PORTKEY_API_KEY=pk-...
```

**3. Docker Compose (Local Development):**

```yaml
# docker-compose.dev.yml (updated)
services:
  agentcore:
    environment:
      - ACE_ENABLED=true
      - ACE_DELTA_MODEL=gpt-4o-mini
    depends_on:
      - postgres
      - portkey

  postgres:
    # Existing configuration
    volumes:
      - ./alembic/versions:/migrations

  portkey:
    image: portkey/gateway:latest
    ports:
      - "8787:8787"
    environment:
      - PORTKEY_API_KEY=${PORTKEY_API_KEY}
```

### CI/CD Pipeline Requirements

**GitHub Actions Workflow:**

```yaml
# .github/workflows/ace-tests.yml
name: ACE Integration Tests

on: [push, pull_request]

jobs:
  test-ace:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s

    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync

      - name: Run ACE tests
        run: |
          uv run pytest tests/ace/ -v --cov=src/agentcore/ace --cov-report=xml

      - name: Check coverage
        run: |
          uv run coverage report --fail-under=90
```

### Testing Framework & Coverage Targets

**Coverage Requirements:**

- Overall: 90% minimum
- Core services (ContextManager, DeltaGenerator, Curator): 95%
- Integration tests: End-to-end evolution workflow
- Performance tests: Latency and throughput validation

**Test Organization:**

```plaintext
tests/ace/
├── unit/
│   ├── test_context_manager.py
│   ├── test_delta_generator.py
│   ├── test_simple_curator.py
│   └── test_models.py
├── integration/
│   ├── test_evolution_workflow.py
│   ├── test_agent_lifecycle_integration.py
│   └── test_task_manager_integration.py
├── performance/
│   ├── test_playbook_retrieval.py
│   └── test_evolution_throughput.py
└── fixtures/
    ├── sample_playbooks.py
    ├── sample_traces.py
    └── mock_llm_responses.py
```

---

## 6. Risk Management

| Risk | Impact | Likelihood | Mitigation Strategy |
|------|--------|------------|---------------------|
| **LLM Cost Escalation** | High | Medium | Monthly budget caps ($100), alert at 75%, circuit breaker, use gpt-4o-mini model |
| **Delta Quality Issues** | Medium | Medium | Confidence threshold (0.7), manual review dashboard, A/B testing validation |
| **Integration Breaking Changes** | High | Low | Feature flags, opt-in per agent, comprehensive integration tests, rollback plan |
| **Performance Degradation** | Medium | Low | Async processing, caching, performance tests in CI, load testing before deploy |
| **Database Migration Failures** | High | Low | Reversible migrations, staging validation, backup before production, zero-downtime |
| **Evolution Worker Crashes** | Medium | Medium | Auto-restart, persistent queue, error logging, dead letter queue for failed evolutions |
| **Context Collapse Despite ACE** | Medium | Low | Monitor playbook versions, track delta quality over time, provide manual override |
| **Budget Exhaustion Mid-Month** | Low | Medium | Alert at 75%, graceful degradation (disable new evolutions), prioritize high-value agents |

**Critical Path Risks:**

1. **LLM Provider Outage:**
   - **Mitigation:** Retry with exponential backoff, fallback to cached responses, graceful degradation (skip evolution)
   - **Contingency:** Multi-provider support via Portkey (OpenAI, Anthropic, Azure)

2. **Database Connection Saturation:**
   - **Mitigation:** Connection pooling (max 20), read replicas for traces (Phase 2), query optimization
   - **Contingency:** Increase connection pool, add database proxy (PgBouncer)

3. **Evolution Worker Blocking Agent Execution:**
   - **Mitigation:** Async queue, worker runs in separate thread/process, timeout enforcement (5s per delta generation)
   - **Contingency:** Circuit breaker disables worker if execution latency exceeds threshold

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Sprint 1: Database & Data Models (Week 1)**

- **Day 1-2:** Database schema design and Alembic migration
  - Create 4 new tables (playbooks, deltas, traces, status)
  - Add indexes for performance
  - Write migration tests (up/down)
  - **Deliverable:** `alembic/versions/xxx_add_ace_tables.py`

- **Day 3-4:** Pydantic data models
  - ContextPlaybook, ContextSection, ContextDelta, ExecutionTrace models
  - Validation rules and constraints
  - JSON serialization/deserialization
  - **Deliverable:** `src/agentcore/ace/models/`

- **Day 5:** ContextManager skeleton
  - Service interface definition
  - Database repository layer
  - Basic CRUD operations (no business logic)
  - **Deliverable:** `src/agentcore/ace/services/context_manager.py`

**Sprint 2: Core Services (Week 2)**

- **Day 1-2:** DeltaGenerator implementation
  - LLM client integration (Portkey)
  - Prompt template design
  - Response parsing logic
  - Confidence scoring
  - **Deliverable:** `src/agentcore/ace/services/delta_generator.py`

- **Day 3-4:** SimpleCurator implementation
  - Confidence filtering
  - Delta application (add/update/remove)
  - Section pruning logic
  - Version management
  - **Deliverable:** `src/agentcore/ace/services/simple_curator.py`

- **Day 5:** Unit tests for core services
  - ContextManager tests (95% coverage)
  - DeltaGenerator tests with mocked LLM
  - SimpleCurator tests (all delta types)
  - **Deliverable:** `tests/ace/unit/` (30+ tests)

### Phase 2: Integration (Week 3-4)

**Sprint 3: Agent Lifecycle Integration (Week 3)**

- **Day 1-2:** AgentLifecycleManager modifications
  - Add `_context_manager` dependency injection
  - Hook `create_agent()` to initialize playbook
  - Hook `_monitor_agent()` to trigger evolution
  - Hook `terminate_agent()` to cleanup playbook
  - **Deliverable:** Updated `services/agent_lifecycle.py`

- **Day 3-4:** Execution context injection
  - `get_execution_context()` implementation
  - Compile playbook into formatted string
  - Inject into agent system prompt
  - Cache compiled contexts (5 min TTL)
  - **Deliverable:** Context injection logic

- **Day 5:** Integration tests for lifecycle
  - End-to-end agent creation with ACE
  - Context injection validation
  - Playbook cleanup on termination
  - **Deliverable:** `tests/ace/integration/test_lifecycle.py`

**Sprint 4: Task Manager & Evolution Worker (Week 4)**

- **Day 1-2:** TaskManager trace capture
  - Hook `complete_task()` to record execution
  - ExecutionTrace construction
  - Asynchronous recording (no blocking)
  - **Deliverable:** Updated `services/task_manager.py`

- **Day 3-4:** Evolution worker implementation
  - Async queue management
  - Worker loop (consume, generate, curate, update)
  - Error handling and retry logic
  - Cost tracking integration
  - **Deliverable:** `src/agentcore/ace/workers/evolution_worker.py`

- **Day 5:** Integration tests for evolution
  - End-to-end evolution workflow
  - Trace capture → delta generation → playbook update
  - Worker error recovery
  - **Deliverable:** `tests/ace/integration/test_evolution.py`

### Phase 3: Hardening (Week 5-6)

**Sprint 5: Configuration & Observability (Week 5)**

- **Day 1:** Configuration management
  - Add ACE section to `config.toml`
  - Feature flags implementation
  - Budget enforcement logic
  - **Deliverable:** `src/agentcore/ace/config.py`

- **Day 2:** Monitoring & metrics
  - Prometheus metrics export
  - Evolution success/failure rates
  - Delta quality metrics
  - Token cost tracking
  - **Deliverable:** `src/agentcore/ace/monitoring.py`

- **Day 3:** Logging & alerting
  - Structured logging for all ACE operations
  - Alert rules (budget, errors, performance)
  - Dashboard definitions (Grafana)
  - **Deliverable:** Monitoring dashboards

- **Day 4-5:** Performance testing
  - Load tests (1000 RPS playbook retrieval)
  - Evolution throughput tests (100+ agents/hour)
  - Database query performance validation
  - **Deliverable:** `tests/ace/performance/`

**Sprint 6: Documentation & Deployment Prep (Week 6)**

- **Day 1-2:** API documentation
  - OpenAPI specs for internal endpoints
  - Usage examples and code snippets
  - Configuration reference
  - **Deliverable:** `docs/ace-api.md`

- **Day 3:** Deployment guide
  - Production deployment checklist
  - Database migration runbook
  - Rollback procedures
  - Monitoring setup guide
  - **Deliverable:** `docs/ace-deployment.md`

- **Day 4:** User guide
  - How to enable ACE for an agent
  - Interpreting evolution metrics
  - Troubleshooting common issues
  - **Deliverable:** `docs/ace-user-guide.md`

- **Day 5:** Final validation
  - Full integration test suite (100+ tests)
  - Performance benchmarks documented
  - Security review checklist
  - **Deliverable:** Release candidate

### Phase 4: Launch & Validation (Week 7-8)

**Sprint 7: Staging Deployment (Week 7)**

- **Day 1:** Deploy to staging environment
  - Run database migrations
  - Enable ACE for 5 test agents
  - Monitor for 24 hours
  - **Deliverable:** Staging deployment report

- **Day 2-3:** A/B testing setup
  - Create control group (10 agents without ACE)
  - Create test group (10 agents with ACE)
  - Define success metrics tracking
  - **Deliverable:** A/B test configuration

- **Day 4-5:** Validation testing
  - Run both groups for 100+ executions each
  - Collect performance metrics
  - Validate cost controls (<$100/month)
  - **Deliverable:** Validation report

**Sprint 8: Production Launch (Week 8)**

- **Day 1:** Production deployment (canary)
  - Deploy to 5% of production agents
  - Monitor for 48 hours
  - Validate no incidents
  - **Deliverable:** Canary deployment report

- **Day 2-3:** Gradual rollout
  - 10% of agents (Day 2)
  - 25% of agents (Day 3)
  - Monitor metrics and costs
  - **Deliverable:** Rollout progress tracking

- **Day 4:** Full rollout
  - 100% of opted-in agents
  - Final metrics collection
  - Document lessons learned
  - **Deliverable:** Launch report

- **Day 5:** Post-launch support
  - Monitor for issues
  - Respond to user feedback
  - Prepare Phase 2 planning
  - **Deliverable:** Post-launch summary

---

## 8. Quality Assurance

### Testing Strategy

**Unit Tests (90% coverage minimum):**

- All data models (Pydantic validation)
- ContextManager CRUD operations
- DeltaGenerator prompt building and parsing
- SimpleCurator filtering and application logic
- Configuration loading and validation

**Integration Tests (E2E workflows):**

- Agent creation → playbook initialization → context injection
- Task completion → trace capture → evolution trigger
- Delta generation → curation → playbook update
- Evolution worker error handling and recovery
- Feature flag enable/disable behavior

**Performance Tests (SLO validation):**

- Playbook retrieval latency (<50ms p95)
- Delta generation throughput (<5s p95)
- Evolution worker scalability (100+ agents/hour)
- Database query performance (index effectiveness)
- Memory usage under load (<100MB per 100 agents)

**Failure Tests (Resilience validation):**

- LLM provider unavailability (retry logic)
- Database connection failures (graceful degradation)
- Invalid delta application (transaction rollback)
- Budget exhaustion (circuit breaker activation)
- Worker crash recovery (queue persistence)

### Code Quality Gates

**Pre-Commit Checks:**

```bash
# Linting (ruff)
uv run ruff check src/agentcore/ace/ tests/ace/

# Type checking (mypy)
uv run mypy src/agentcore/ace/ --strict

# Code formatting (black)
uv run black --check src/agentcore/ace/ tests/ace/

# Test coverage
uv run pytest tests/ace/ --cov=src/agentcore/ace --cov-fail-under=90
```

**CI Pipeline Gates:**

- All tests pass (unit, integration, performance)
- Code coverage ≥90%
- No type errors (mypy strict mode)
- No linting violations (ruff)
- Documentation builds successfully
- Database migrations reversible (up/down tested)

### Deployment Verification Checklist

**Pre-Deployment:**

- [ ] All tests passing in CI
- [ ] Code review approved (2+ reviewers)
- [ ] Security review completed
- [ ] Performance benchmarks meet SLOs
- [ ] Database backup verified
- [ ] Rollback plan documented

**During Deployment:**

- [ ] Database migrations applied successfully
- [ ] Configuration deployed to all environments
- [ ] Feature flags set correctly (disabled by default)
- [ ] Monitoring dashboards accessible
- [ ] Alerts configured and tested

**Post-Deployment:**

- [ ] Smoke tests passing (API health checks)
- [ ] No error spikes in logs
- [ ] Metrics collecting correctly
- [ ] Cost tracking functioning
- [ ] User feedback channels monitored

### Monitoring & Alerting Setup

**Key Metrics to Track:**

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| Evolution success rate | <90% | Investigate LLM or database issues |
| Delta confidence score | <0.6 avg | Review LLM prompt quality |
| Token cost per agent | >$1/agent/month | Optimize generation frequency or prompt length |
| Playbook retrieval latency | >100ms p95 | Check cache hit rate, database query performance |
| Worker queue depth | >100 | Scale worker or increase concurrency |
| Monthly budget usage | >75% | Alert to adjust limits or pause evolutions |

**Dashboards:**

1. **ACE Overview:** Evolution metrics, cost tracking, system health
2. **Agent Performance:** Per-agent playbook versions, delta quality, execution success rates
3. **System Performance:** Latency, throughput, resource utilization
4. **Cost Analysis:** Token usage by agent, model, and time period

---

## 9. References

### Supporting Documents

- **Research Analysis:** `docs/research/ace-integration-analysis.md`
- **Specification:** `docs/specs/ace-integration/spec.md`
- **ACE Paper:** <https://arxiv.org/abs/2510.04618>
- **AgentCore Architecture:** `docs/agentcore-architecture-and-development-plan.md`
- **Agent Runtime Spec:** `docs/specs/agent-runtime/spec.md`
- **A2A Protocol Spec:** `docs/specs/a2a-protocol/spec.md`

### Research Sources

**LLM Cost Optimization:**

- OpenAI Pricing: <https://openai.com/pricing>
- GPT-4o-mini benchmarks: <https://artificialanalysis.ai/models/gpt-4o-mini>
- Portkey semantic caching: <https://portkey.ai/features/semantic-cache>

**Context Management Best Practices:**

- Long-context LLM evaluation: <https://arxiv.org/abs/2404.07143>
- Prompt engineering guide: <https://platform.openai.com/docs/guides/prompt-engineering>

**PostgreSQL JSONB Performance:**

- JSONB indexing: <https://www.postgresql.org/docs/14/datatype-json.html>
- GIN indexes: <https://www.postgresql.org/docs/14/gin-intro.html>

### Related Specifications

**Cross-Component Dependencies:**

- **Agent Runtime (ART):** Lifecycle manager integration, execution state management
- **A2A Protocol:** Task manager integration, execution trace capture
- **DSPy Optimization (future):** Potential dual optimization architecture (Phase 2+)

**Integration Points:**

- `src/agentcore/agent_runtime/services/agent_lifecycle.py`
- `src/agentcore/a2a_protocol/services/task_manager.py`
- `src/agentcore/a2a_protocol/database/models.py` (agent_records table)

---

## Appendix A: Detailed Architecture Diagrams

### Component Interaction Sequence

```plaintext
Agent Creation Flow:
┌────────┐         ┌──────────────┐         ┌────────────────┐         ┌──────────┐
│ User   │         │  Lifecycle   │         │  Context       │         │ Database │
│        │         │  Manager     │         │  Manager       │         │          │
└───┬────┘         └──────┬───────┘         └────────┬───────┘         └─────┬────┘
    │                     │                          │                       │
    │ create_agent()      │                          │                       │
    ├────────────────────>│                          │                       │
    │                     │                          │                       │
    │                     │ create_playbook()        │                       │
    │                     ├─────────────────────────>│                       │
    │                     │                          │                       │
    │                     │                          │ INSERT playbook       │
    │                     │                          ├──────────────────────>│
    │                     │                          │                       │
    │                     │                          │<──────────────────────┤
    │                     │                          │ playbook_id           │
    │                     │                          │                       │
    │                     │<─────────────────────────┤                       │
    │                     │ playbook_id              │                       │
    │                     │                          │                       │
    │<────────────────────┤                          │                       │
    │ agent_id            │                          │                       │
    │                     │                          │                       │


Task Execution Flow with Trace Capture:
┌────────┐    ┌──────────┐    ┌────────────────┐    ┌────────────────┐    ┌──────────┐
│ Agent  │    │  Task    │    │  Context       │    │  Execution     │    │ Evolution│
│        │    │  Manager │    │  Manager       │    │  Trace         │    │  Queue   │
└───┬────┘    └────┬─────┘    └────────┬───────┘    └────────┬───────┘    └─────┬────┘
    │              │                   │                     │                  │
    │ execute()    │                   │                     │                  │
    ├─────────────>│                   │                     │                  │
    │              │                   │                     │                  │
    │              │ get_context()     │                     │                  │
    │              ├──────────────────>│                     │                  │
    │              │                   │                     │                  │
    │              │<──────────────────┤                     │                  │
    │              │ context_string    │                     │                  │
    │              │                   │                     │                  │
    │<─────────────┤                   │                     │                  │
    │ task+context │                   │                     │                  │
    │              │                   │                     │                  │
    │ [execution]  │                   │                     │                  │
    │──────────────│                   │                     │                  │
    │              │                   │                     │                  │
    │ result       │                   │                     │                  │
    ├─────────────>│                   │                     │                  │
    │              │                   │                     │                  │
    │              │ record_trace()    │                     │                  │
    │              ├────────────────────────────────────────>│                  │
    │              │                   │                     │                  │
    │              │                   │                     │ INSERT trace     │
    │              │                   │                     ├──────────────────┤
    │              │                   │                     │                  │
    │              │ trigger_evolution()                     │                  │
    │              ├──────────────────>│                     │                  │
    │              │                   │                     │                  │
    │              │                   │ queue(agent_id)     │                  │
    │              │                   ├───────────────────────────────────────>│
    │              │                   │                     │                  │


Evolution Cycle Flow:
┌────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Evolution     │    │  Delta       │    │  Simple      │    │ Database │
│  Worker        │    │  Generator   │    │  Curator     │    │          │
└────────┬───────┘    └──────┬───────┘    └──────┬───────┘    └─────┬────┘
         │                   │                   │                  │
         │ dequeue(agent_id) │                   │                  │
         │───────────────────│                   │                  │
         │                   │                   │                  │
         │ get_playbook()    │                   │                  │
         ├─────────────────────────────────────────────────────────>│
         │                   │                   │                  │
         │<─────────────────────────────────────────────────────────┤
         │ playbook          │                   │                  │
         │                   │                   │                  │
         │ get_traces()      │                   │                  │
         ├─────────────────────────────────────────────────────────>│
         │                   │                   │                  │
         │<─────────────────────────────────────────────────────────┤
         │ traces[10]        │                   │                  │
         │                   │                   │                  │
         │ generate_deltas() │                   │                  │
         ├──────────────────>│                   │                  │
         │                   │                   │                  │
         │                   │ [LLM API call]    │                  │
         │                   │───────────────────│                  │
         │                   │                   │                  │
         │<──────────────────┤                   │                  │
         │ deltas[3]         │                   │                  │
         │                   │                   │                  │
         │ apply_deltas()    │                   │                  │
         ├──────────────────────────────────────>│                  │
         │                   │                   │                  │
         │                   │                   │ BEGIN TRANSACTION│
         │                   │                   ├─────────────────>│
         │                   │                   │                  │
         │                   │                   │ UPDATE playbook  │
         │                   │                   ├─────────────────>│
         │                   │                   │                  │
         │                   │                   │ INSERT deltas    │
         │                   │                   ├─────────────────>│
         │                   │                   │                  │
         │                   │                   │ COMMIT           │
         │                   │                   ├─────────────────>│
         │                   │                   │                  │
         │<──────────────────────────────────────┤                  │
         │ updated_playbook  │                   │                  │
         │                   │                   │                  │
         │ update_status()   │                   │                  │
         ├─────────────────────────────────────────────────────────>│
         │                   │                   │                  │
```

---

**Plan Status:** ✅ Complete and Ready for Task Breakdown
**Next Step:** Run `/sage.tasks` to generate detailed task breakdown with story points
