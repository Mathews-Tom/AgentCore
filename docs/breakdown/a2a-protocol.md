# A2A Protocol Layer - Technical Breakdown

**Created:** 2025-10-01
**Updated:** 2025-10-01 (Session Management Phase 5 added)
**Sources:** [spec](../specs/a2a-protocol/spec.md) | [plan](../specs/a2a-protocol/plan.md) | [tasks](../specs/a2a-protocol/tasks.md)

---

## Quick Reference

```yaml
complexity: High
risk_level: Medium
team_size: 2-3 engineers (1 senior, 1-2 mid-level, 0.5 ML)
duration: 10 weeks (5 sprints)
story_points: 96 SP
dependencies:
  - PostgreSQL 14+ with pgvector extension
  - Redis Cluster for caching and streams
  - sentence-transformers (all-MiniLM-L6-v2)
  - FastAPI for HTTP/WebSocket endpoints
```

## Component Overview

**Purpose:** Foundational communication infrastructure implementing Google's Agent2Agent (A2A) protocol v0.2 for standardized, secure, and interoperable agent-to-agent communication with semantic capability matching and session management.

**Key Capabilities:**
- JSON-RPC 2.0 compliant agent communication
- Semantic capability matching via vector embeddings (pgvector)
- Cost-biased intelligent agent routing
- Real-time messaging (WebSocket/SSE)
- Task lifecycle management with artifacts
- Long-running workflow session persistence and resumption
- Agent discovery and registration

**Success Metrics:**
- Protocol Compliance: 99.9% A2A v0.2 compliance
- Performance: <50ms agent discovery (p95), <10ms message routing (p95), <1s session resume (p95)
- Scalability: 1000+ concurrent agent connections per instance
- Reliability: 99.9% uptime SLA
- Semantic Matching: >90% recall vs exact string matching
- Cost Optimization: 20-30% reduction through intelligent routing

## System Context

```mermaid
graph LR
    A[External Agents] -->|HTTPS/WSS| B[A2A Protocol Layer]
    B -->|JSON-RPC| C[Agent Runtime]
    B -->|Events| D[Orchestration Engine]
    E[Gateway Layer] -->|HTTP| B
    B -->|Query| F[(PostgreSQL + pgvector)]
    B -->|Cache| G[(Redis Cluster)]
    B -->|Embeddings| H[sentence-transformers]
```

**Integration Points:**

1. **Upstream (Consumers):**
   - External agents via HTTPS/WebSocket (JSON-RPC 2.0)
   - Gateway Layer for HTTP REST API proxy
   - CLI Layer for developer commands
   - Orchestration Engine for workflow coordination

2. **Downstream (Dependencies):**
   - PostgreSQL: Agent registry, tasks, artifacts, sessions
   - Redis: Agent cache, rate limiting, WebSocket tracking
   - pgvector: Semantic capability search
   - sentence-transformers: Capability embedding generation

3. **Async (Event-Driven):**
   - Task status updates via Redis Streams
   - Agent health events
   - Session lifecycle events

## Architecture Design

### Component Structure

```mermaid
graph TD
    subgraph "A2A Protocol Layer"
        API[API Layer<br/>FastAPI + WebSocket]
        JSONRPC[JSON-RPC Handler<br/>Method Registry]

        subgraph "Service Layer"
            AM[Agent Manager]
            TM[Task Manager]
            SM[Session Manager]
            MR[Message Router]
            SEC[Security Service]
            EM[Event Manager]
        end

        subgraph "Data Layer"
            REPO[Repositories<br/>Agent/Task/Session]
            CACHE[Redis Cache]
        end
    end

    API --> JSONRPC
    JSONRPC --> AM
    JSONRPC --> TM
    JSONRPC --> SM
    JSONRPC --> MR
    API --> SEC
    AM --> REPO
    TM --> REPO
    SM --> REPO
    AM --> CACHE
    REPO --> DB[(PostgreSQL)]
```

### Key Modules

**Module 1: services/jsonrpc_handler.py**
- **Responsibility:** Core JSON-RPC 2.0 request/response processing, method registry, batch requests
- **Dependencies:** Pydantic (validation), internal services (agent, task, session managers)
- **Complexity:** High

**Module 2: services/agent_manager.py**
- **Responsibility:** Agent registration, discovery (exact + semantic), health monitoring, capability indexing
- **Dependencies:** AgentRepository, Redis cache, sentence-transformers
- **Complexity:** High

**Module 3: services/task_manager.py**
- **Responsibility:** Task creation, execution tracking, artifact storage, status streaming
- **Dependencies:** TaskRepository, Redis Streams, message_router
- **Complexity:** High

**Module 4: services/session_manager.py** (NEW - Phase 5)
- **Responsibility:** Session save/resume, state serialization, event history management
- **Dependencies:** SessionRepository, task_manager, agent_manager
- **Complexity:** Medium

**Module 5: services/message_router.py**
- **Responsibility:** Intelligent agent routing with cost-biased optimization, multi-objective scoring
- **Dependencies:** agent_manager (capability matching), cost/latency/quality metadata
- **Complexity:** High

**Module 6: services/security_service.py**
- **Responsibility:** JWT authentication, RBAC authorization, rate limiting, audit logging
- **Dependencies:** JWT library, Redis (rate limiting)
- **Complexity:** Medium

**Module 7: services/event_manager.py**
- **Responsibility:** Real-time event streaming via SSE/WebSocket, event publishing to Redis Streams
- **Dependencies:** Redis Streams, WebSocket connections
- **Complexity:** Medium

**Module 8: database/repositories.py**
- **Responsibility:** Async data access patterns for agents, tasks, sessions
- **Dependencies:** SQLAlchemy (async), PostgreSQL
- **Complexity:** Medium

## Interface Contracts

### JSON-RPC API

**Agent Registration**
```yaml
Method: agent.register
Request:
  - agent_id: string (required)
  - agent_card: AgentCard (required)
    - schema_version: "0.2"
    - agent_name: string
    - capabilities: string[] (indexed for semantic search)
    - supported_interactions: string[]
    - authentication: object
    - endpoints: Endpoint[]
Response:
  - agent_id: string
  - status: "registered"
  - discovery_url: string
Errors:
  - -32602: Invalid AgentCard schema
  - -32000: Agent already registered
  - -32001: Capability validation failed
```

**Agent Discovery (Semantic + Exact)**
```yaml
Method: agent.discover
Request:
  - capabilities: string[] (required)
  - semantic_search: boolean (default: true)
  - similarity_threshold: float (default: 0.75, range: 0-1)
  - filters: object (optional)
    - latency_max_ms: number
    - cost_max: number
    - quality_min: number
Response:
  - agents: Agent[]
    - agent_id: string
    - agent_name: string
    - capabilities: string[]
    - match_score: float (0-1)
    - similarity_score: float (semantic match)
    - cost_score: float
    - latency_p95_ms: number
    - endpoints: Endpoint[]
  - search_metadata:
    - total_matches: number
    - semantic_used: boolean
    - search_time_ms: number
Errors:
  - -32602: Invalid capability format
  - -32003: No matching agents found
```

**Task Creation**
```yaml
Method: task.create
Request:
  - task_type: string (required)
  - target_agent: string (required, agent_id or auto-route)
  - input_data: object (required)
  - execution_options: object
    - timeout: number (seconds, default: 300)
    - priority: "low"|"normal"|"high" (default: "normal")
Response:
  - task_id: string (UUID)
  - status: "queued"
  - assigned_agent: string
  - estimated_cost: number
  - created_at: ISO8601
Errors:
  - -32602: Invalid input_data
  - -32004: Target agent unavailable
  - -32005: Routing failed
```

**Session Save** (NEW)
```yaml
Method: session.save
Request:
  - name: string (required)
  - description: string (optional)
  - tags: string[] (optional)
  - metadata: object (optional)
Response:
  - session_id: UUID
  - state: "PAUSED"
  - snapshot_size_bytes: number
  - tasks_captured: number
  - agents_captured: number
  - created_at: ISO8601
Errors:
  - -32006: No active workflow to save
  - -32007: Serialization failed
Performance: <500ms p95
```

**Session Resume** (NEW)
```yaml
Method: session.resume
Request:
  - session_id: UUID (required)
  - replay_events: boolean (default: false)
Response:
  - session_id: UUID
  - state: "RESUMED"
  - tasks_restored: number
  - agents_reconnected: number
  - resumed_at: ISO8601
Errors:
  - -32008: Session not found
  - -32009: Deserialization failed
  - -32010: Agent reconnection failed
Performance: <1s p95
```

### WebSocket Endpoint

```yaml
Path: /ws/agent/{agent_id}
Protocol: WebSocket + JSON-RPC 2.0
Authentication: JWT in query param or header

Message Format:
  - Bidirectional JSON-RPC messages
  - Heartbeat every 30s
  - Auto-reconnect on disconnect

Connection Limits:
  - Max concurrent: 1000 per instance
  - Rate limit: 100 messages/second per agent
```

### Server-Sent Events (SSE)

```yaml
Path: /sse/task/{task_id}
Protocol: SSE (text/event-stream)
Authentication: JWT in Authorization header

Event Types:
  - task.status: Status updates
  - task.progress: Progress percentage
  - task.artifact: Artifact created
  - task.completed: Task finished
  - task.failed: Task error
```

### A2A Discovery Endpoint

```yaml
Path: /.well-known/agent.json
Method: GET
Response: AgentCard (JSON)
Cache: Public, max-age=3600
```

## Data Models

### PostgreSQL Schema

**agents table**
```sql
CREATE TABLE agents (
    agent_id VARCHAR(255) PRIMARY KEY,
    agent_name VARCHAR(255) NOT NULL,
    agent_card JSONB NOT NULL,
    capabilities TEXT[] NOT NULL,
    capability_embedding VECTOR(768), -- pgvector for semantic search
    endpoints JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    health_status VARCHAR(50) DEFAULT 'healthy',
    last_health_check TIMESTAMP,
    cost_per_request DECIMAL(10,4),
    latency_p95_ms INTEGER,
    quality_score DECIMAL(3,2),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agents_status ON agents(status) WHERE status = 'active';
CREATE INDEX idx_agents_capabilities ON agents USING GIN(capabilities);
CREATE INDEX idx_agents_embedding ON agents USING ivfflat(capability_embedding vector_cosine_ops)
    WITH (lists = 100); -- HNSW alternative: USING hnsw(capability_embedding vector_cosine_ops)
```

**tasks table**
```sql
CREATE TABLE tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type VARCHAR(255) NOT NULL,
    target_agent_id VARCHAR(255) NOT NULL REFERENCES agents(agent_id),
    input_data JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    progress DECIMAL(5,2) DEFAULT 0.0,
    execution_options JSONB,
    assigned_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent ON tasks(target_agent_id);
CREATE INDEX idx_tasks_created_at ON tasks(created_at DESC);
```

**task_artifacts table**
```sql
CREATE TABLE task_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    artifact_type VARCHAR(100) NOT NULL,
    content JSONB NOT NULL,
    content_summary TEXT,
    context_lineage JSONB, -- For context engineering
    size_bytes INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_artifacts_task ON task_artifacts(task_id);
```

**session_snapshots table** (NEW)
```sql
CREATE TABLE session_snapshots (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tags TEXT[],
    state VARCHAR(50) NOT NULL,
    context JSONB NOT NULL,
    tasks JSONB NOT NULL,
    agents JSONB NOT NULL,
    event_history JSONB,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    resumed_at TIMESTAMP,
    completed_at TIMESTAMP,
    CONSTRAINT valid_state CHECK (state IN ('ACTIVE', 'PAUSED', 'RESUMED', 'COMPLETED'))
);

CREATE INDEX idx_session_state ON session_snapshots(state);
CREATE INDEX idx_session_created_at ON session_snapshots(created_at DESC);
CREATE INDEX idx_session_tags ON session_snapshots USING GIN(tags);
```

### Pydantic Models

**AgentCard (A2A v0.2 Spec)**
```python
class AgentCard(BaseModel):
    schema_version: Literal["0.2"]
    agent_name: str
    capabilities: List[str]
    supported_interactions: List[str]
    authentication: Dict[str, Any]
    endpoints: List[Endpoint]

    # Enhanced metadata for cost-biased routing
    cost_per_request: Optional[float] = None
    latency_p95_ms: Optional[int] = None
    quality_score: Optional[float] = None

    # Context engineering support
    system_context: Optional[str] = None
    interaction_examples: Optional[List[Dict[str, Any]]] = None
```

**SessionSnapshot** (NEW)
```python
class SessionSnapshot(BaseModel):
    session_id: UUID
    name: str
    description: str = ""
    tags: List[str] = []
    state: Literal["ACTIVE", "PAUSED", "RESUMED", "COMPLETED"]
    context: Dict[str, Any]
    tasks: List[TaskSnapshot]
    agents: List[AgentState]
    event_history: List[Event] = Field(max_items=1000)
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    resumed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class TaskSnapshot(BaseModel):
    task_id: str
    status: TaskStatus
    execution_state: Dict[str, Any]
    assigned_agent_id: Optional[str]
    progress: float
    checkpoints: List[Dict[str, Any]]

class AgentState(BaseModel):
    agent_id: str
    connection_state: Literal["CONNECTED", "DISCONNECTED", "RECONNECTING"]
    assigned_tasks: List[str]
    context: Dict[str, Any]
```

## Implementation Details

### Technology Stack

**Runtime:** Python 3.12+
**Framework:** FastAPI 0.100+ (async)
**Database:** PostgreSQL 14+ with pgvector 0.5+
**Cache:** Redis 7+ (cluster mode for production)
**Vector Search:** pgvector with HNSW or IVFFlat indexing
**Embeddings:** sentence-transformers (all-MiniLM-L6-v2, 768 dimensions)
**ORM:** SQLAlchemy 2.0+ (async)
**WebSocket:** FastAPI WebSocket + Starlette
**Validation:** Pydantic 2.0+

**Rationale:**
- Python 3.12 for latest async improvements and type hints
- FastAPI for native async, OpenAPI docs, and WebSocket support
- PostgreSQL + pgvector for unified relational + vector storage
- Redis Cluster for high-availability caching and streams
- sentence-transformers for efficient CPU-based embeddings (no GPU required)
- HNSW index for best query performance (<20ms for 1000+ agents)

### Design Patterns

1. **Repository Pattern** - Data access abstraction with async SQLAlchemy sessions
2. **Dependency Injection** - FastAPI dependencies for services, auth, database sessions
3. **Circuit Breaker** - External agent call resilience (using tenacity library)
4. **CQRS (Light)** - Read-optimized queries via Redis cache, write through to PostgreSQL
5. **Event Sourcing (Partial)** - Task status changes published to Redis Streams
6. **Strategy Pattern** - Routing algorithms (semantic, exact, cost-biased)

### Configuration

```yaml
# Environment variables
DATABASE_URL: postgresql+asyncpg://user:pass@host:5432/agentcore
REDIS_URL: redis://redis:6379/0  # Single instance
REDIS_CLUSTER_URLS: redis://node1:6379,redis://node2:6379  # Cluster mode

# A2A Protocol
A2A_PROTOCOL_VERSION: "0.2"
MAX_CONCURRENT_CONNECTIONS: 1000
MESSAGE_TIMEOUT_SECONDS: 30
ENABLE_SEMANTIC_SEARCH: true
SIMILARITY_THRESHOLD: 0.75

# Semantic Search
EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: 768
VECTOR_INDEX_TYPE: "hnsw"  # or "ivfflat"

# Session Management
SESSION_RETENTION_DAYS: 30
MAX_SESSION_SIZE_MB: 50
ENABLE_EVENT_REPLAY: true

# Security
JWT_SECRET_KEY: ${SECRET_FROM_ENV}
JWT_ALGORITHM: "RS256"
JWT_EXPIRATION_HOURS: 24
ENABLE_RATE_LIMITING: true
RATE_LIMIT_PER_AGENT: 100  # requests per second

# Monitoring
ENABLE_METRICS: true
LOG_LEVEL: "INFO"
SENTRY_DSN: ${SENTRY_DSN}
```

## Testing Strategy

### Unit Tests (Target: 95% coverage)

**Critical Paths:**
1. **JSON-RPC Parser** (95% coverage)
   - Valid request/response parsing
   - Batch request handling
   - Error response formatting
   - A2A context extraction

2. **Agent Capability Matching** (95% coverage)
   - Exact string matching
   - Semantic similarity computation
   - Multi-objective scoring (similarity, cost, latency, quality)
   - Threshold filtering

3. **Session Serialization** (95% coverage)
   - TaskSnapshot serialization
   - AgentState serialization
   - Context compression
   - Circular reference handling

4. **Cost-Biased Routing** (90% coverage)
   - Scoring algorithm correctness
   - Hard constraint enforcement
   - Fallback strategies

**Tools:** pytest, pytest-asyncio, pytest-cov

### Integration Tests

**Scenarios:**

1. **Agent Lifecycle** (testcontainers: PostgreSQL + Redis)
   - Register agent → Discover → Health check → Deregister

2. **Task Execution Flow**
   - Create task → Monitor status via SSE → Retrieve artifact

3. **Session Save/Resume** (<1s target)
   - Create workflow with 10 tasks → Save session → Resume
   - Verify resume time < 1.0s p95

4. **Vector Similarity Search** (pgvector)
   - Register 100 agents → Test recall rate vs exact matching
   - Verify semantic recall > 90%

**Tools:** pytest-asyncio, testcontainers-python, httpx (async client)

### End-to-End Tests

**User Flows:**

1. **Multi-Agent Communication**
   - Agent A registers → Agent B registers → Agent A discovers Agent B
   - Agent A sends task to Agent B → Agent B processes and returns artifact

2. **Long-Running Workflow with Session Resume**
   - Start complex workflow (20+ tasks)
   - Save session after 10 tasks complete
   - Simulate server restart
   - Resume session
   - Verify remaining 10 tasks execute correctly

3. **Semantic Routing Under Load**
   - Register 1000 agents with diverse capabilities
   - Send 100 concurrent discovery requests
   - Verify all requests complete <100ms p95
   - Verify semantic matching accuracy >90%

**Tools:** Playwright (for CLI testing), pytest-xdist (parallel execution)

### Performance Tests

**Load Scenarios:**

1. **Message Routing Latency**
   - Concurrent requests: 1000/second
   - Target: <10ms p95 latency
   - Duration: 10 minutes

2. **Agent Discovery Performance**
   - Agent registry size: 1000+ agents
   - Semantic search enabled
   - Target: <50ms p95 latency
   - Concurrent requests: 100/second

3. **Session Resume Speed**
   - Session sizes: 1MB, 10MB, 50MB
   - Target: <1s p95 for all sizes
   - Test with/without event replay

4. **WebSocket Connection Stability**
   - Concurrent connections: 1000
   - Message rate: 100/second per connection
   - Duration: 1 hour
   - Verify: No dropped messages, <1% connection drops

**Tools:** k6, locust

### Security Tests

**OWASP Top 10:**
- SQL Injection: Parameterized queries (SQLAlchemy)
- XSS: JSON-only responses, no HTML rendering
- Broken Authentication: JWT validation, expiration checks
- Sensitive Data Exposure: TLS 1.3, encrypted artifacts

**Penetration Testing:**
- Rate limiting bypass attempts
- JWT token tampering
- WebSocket hijacking
- Agent impersonation

**Tools:** Bandit (static analysis), Safety (dependency scanning), OWASP ZAP

## Operational Concerns

### Infrastructure

```yaml
Compute:
  - Container: 4 vCPU, 8GB RAM (recommendation)
  - Auto-scaling: 2-10 instances based on CPU/connection count
  - Health checks: /health endpoint (200 OK)
  - Readiness: /ready endpoint (checks DB + Redis)

Storage:
  Database:
    - PostgreSQL: 100GB initial, auto-expand
    - Backup: Daily, 30-day retention
    - Replication: 1 read replica

  Cache:
    - Redis: 16GB memory (cluster mode)
    - Persistence: RDB + AOF
    - Eviction: LRU for cache keys

  Embeddings:
    - Additional storage: 1-2GB per 1000 agents (768D vectors)

  Logs:
    - Retention: 30 days
    - Format: JSON structured logging
    - Destination: CloudWatch/Stackdriver

Network:
  - Load balancer: Application LB with sticky sessions (for WebSocket)
  - TLS termination: At load balancer
  - Internal: Private VPC
```

### Monitoring

**Metrics (Prometheus):**

1. **RED Metrics**
   ```promql
   # Request Rate
   rate(jsonrpc_requests_total[5m])

   # Error Rate
   rate(jsonrpc_errors_total[5m]) / rate(jsonrpc_requests_total[5m])

   # Duration (p95)
   histogram_quantile(0.95, rate(jsonrpc_request_duration_seconds_bucket[5m]))
   ```

2. **Agent Metrics**
   ```promql
   # Active agents
   agent_registry_active_count

   # Registration rate
   rate(agent_registrations_total[5m])

   # Discovery latency (p95)
   histogram_quantile(0.95, rate(agent_discovery_duration_seconds_bucket[5m]))
   ```

3. **Task Metrics**
   ```promql
   # Task creation rate
   rate(task_created_total[5m])

   # Task completion rate
   rate(task_completed_total[5m])

   # Task failure rate
   rate(task_failed_total[5m]) / rate(task_created_total[5m])
   ```

4. **Session Metrics** (NEW)
   ```promql
   # Session save latency
   histogram_quantile(0.95, rate(session_save_duration_seconds_bucket[5m]))

   # Session resume latency (target: <1s p95)
   histogram_quantile(0.95, rate(session_resume_duration_seconds_bucket[5m]))

   # Session size distribution
   histogram_quantile(0.95, session_size_bytes)
   ```

5. **Infrastructure Metrics**
   ```promql
   # WebSocket connections
   websocket_connections_active

   # PostgreSQL connection pool
   database_connections_in_use / database_connections_max

   # Redis memory usage
   redis_memory_used_bytes / redis_memory_max_bytes

   # Vector search latency
   histogram_quantile(0.95, rate(pgvector_search_duration_seconds_bucket[5m]))
   ```

**Alerts (PagerDuty/Opsgenie):**

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High error rate | >1% for 5min | Critical | Page on-call |
| High latency | p95 >500ms for 5min | Warning | Investigate |
| Agent discovery slow | >100ms p95 for 5min | Warning | Check pgvector index |
| Session resume slow | >1s p95 for 3min | Warning | Check serialization |
| DB connections high | >80% for 5min | Critical | Scale up or investigate leak |
| Redis memory high | >90% for 10min | Warning | Increase memory or evict |
| WebSocket drops | >5% disconnects/min | Warning | Check network/LB |

**Dashboards (Grafana):**

1. **Service Health Overview**
   - Request rate, error rate, latency (RED)
   - Active agents, tasks, sessions
   - Infrastructure health (CPU, memory, connections)

2. **Agent Discovery Performance**
   - Discovery latency histogram
   - Semantic vs exact search ratio
   - Match score distribution
   - Vector search performance

3. **Task Execution**
   - Task creation/completion rates
   - Task duration distribution
   - Artifact size distribution
   - Error breakdown by type

4. **Session Management** (NEW)
   - Session save/resume latency
   - Session size distribution
   - Active sessions by age
   - Resume success rate

### Security

**Authentication:**
- JWT tokens with RS256 signing (asymmetric keys)
- Token expiration: 24 hours default
- Refresh token support (optional)
- Agent-to-agent: Mutual TLS or JWT exchange

**Authorization:**
- RBAC with roles: `agent_admin`, `agent_user`, `agent_readonly`
- Permissions: `agent:register`, `agent:discover`, `task:create`, `session:manage`
- Scope-based access control per agent namespace

**Data Protection:**
- Encryption at rest: PostgreSQL transparent encryption
- Encryption in transit: TLS 1.3 minimum
- Sensitive artifacts: AES-256-GCM encryption
- PII handling: Logged data scrubbing, GDPR compliance

**Secrets Management:**
- JWT keys: AWS Secrets Manager / HashiCorp Vault
- Database credentials: Secrets rotation every 90 days
- API keys: Encrypted in PostgreSQL with application-level encryption

**Audit Logging:**
```python
# All operations logged with:
{
  "timestamp": "2025-10-01T12:00:00Z",
  "user_id": "agent-001",
  "action": "agent.register",
  "resource": "agent-002",
  "status": "success",
  "ip_address": "10.0.1.5",
  "trace_id": "abc123"
}
```

### Scaling Strategy

**Horizontal Scaling:**
- Stateless application instances (2-10 auto-scaling)
- WebSocket: Sticky sessions at load balancer
- Redis Pub/Sub for cross-instance communication
- Load balancer: Round-robin with health checks

**Vertical Scaling:**
- PostgreSQL: Read replicas for agent discovery queries
- pgvector index optimization: HNSW for best performance
- Connection pooling: 20 connections per instance
- Query optimization: Index on (status, created_at) for task queries

**Caching Strategy:**
- Agent registry: Redis cache with 5-minute TTL
- Capability embeddings: Persistent cache (regenerate on update)
- Task status: Redis Streams for real-time updates
- Session snapshots: No caching (direct PostgreSQL)

**Database Optimization:**
```sql
-- Partitioning for tasks table (by created_at)
CREATE TABLE tasks_2025_10 PARTITION OF tasks
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

-- Materialized view for agent discovery
CREATE MATERIALIZED VIEW agent_search_cache AS
SELECT agent_id, agent_name, capabilities, capability_embedding
FROM agents WHERE status = 'active';

-- Refresh every 5 minutes
REFRESH MATERIALIZED VIEW CONCURRENTLY agent_search_cache;
```

## Risk Analysis

### Technical Risks

| Risk | Impact | Likelihood | Mitigation | Contingency |
|------|--------|------------|------------|-------------|
| pgvector search slow at scale (>10k agents) | High | Medium | HNSW index, query optimization, result caching | Fallback to exact matching only |
| Session serialization >10MB causes timeout | High | Medium | Incremental checkpoints, compression (gzip), context pruning | Warn user, fail save gracefully |
| WebSocket connections unstable under load | Medium | High | Connection pooling, heartbeat monitoring, auto-reconnect | SSE fallback for status updates |
| Event ordering issues in distributed setup | High | Low | Redis Streams with consumer groups, sequence numbers | Single Redis master for ordering |
| Cost-biased routing inaccurate | Medium | Medium | Configurable weights, A/B testing, feedback loop | Allow manual agent selection |
| Embedding generation latency >100ms | Medium | Medium | Batch processing, GPU acceleration (optional), caching | Pre-compute embeddings on registration |
| PostgreSQL connection pool exhausted | High | Medium | Connection pooling (20/instance), leak detection | Scale up instances, restart |

### Dependency Risks

| Dependency | Risk | Impact | Mitigation |
|------------|------|--------|------------|
| PostgreSQL pgvector extension | Extension not available on managed service | High | Validate on startup, provide fallback |
| sentence-transformers model | Model download failure or size (>400MB) | Medium | Bundle model in container, cache locally |
| Redis Cluster | Node failure during high load | High | Redis Sentinel for auto-failover, graceful degradation |
| FastAPI | Breaking changes in minor versions | Low | Pin versions, test upgrades in staging |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory leak in WebSocket connections | High | Connection limits, monitoring, auto-restart |
| Database migration failure | Critical | Blue-green deployment, rollback plan, backups |
| TLS certificate expiration | Critical | Automated renewal (Let's Encrypt), alerts |
| Runaway queries (no LIMIT) | Medium | Query timeout (30s), connection pooling |

## Development Workflow

### Local Setup

```bash
# Clone repository
git clone https://github.com/agentcore/agentcore.git
cd agentcore

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
uv add fastapi sqlalchemy asyncpg redis pydantic sentence-transformers pgvector-python

# Start dependencies (PostgreSQL + Redis)
docker compose -f docker-compose.dev.yml up -d

# Wait for PostgreSQL to be ready
until pg_isready -h localhost -p 5432; do sleep 1; done

# Run migrations
uv run alembic upgrade head

# Seed test data (optional)
uv run python scripts/seed_test_agents.py

# Start development server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# API docs available at http://localhost:8001/docs (DEBUG mode only)
```

### Code Quality

**Linting:**
```bash
# Ruff (linter + formatter)
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

**Type Safety:**
```bash
# Mypy strict mode
uv run mypy src/ --strict
```

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-redis, types-requests]
```

**CI Checks:**
- Lint: Ruff
- Type check: Mypy strict
- Test: pytest with 95% coverage
- Security: Bandit, Safety
- Build: Docker image

### Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy A2A Protocol

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: ankane/pgvector:latest
        env:
          POSTGRES_PASSWORD: test
      redis:
        image: redis:7-alpine
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv pip install -r requirements.txt
      - run: uv run pytest --cov --cov-report=xml
      - run: uv run mypy src/
      - run: uv run ruff check src/

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install bandit safety
      - run: bandit -r src/
      - run: safety check

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: agentcore/a2a-protocol:${{ github.sha }}

  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
      - run: kubectl set image deployment/a2a-protocol a2a-protocol=agentcore/a2a-protocol:${{ github.sha }}
      - run: kubectl rollout status deployment/a2a-protocol

  e2e-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - run: uv run pytest tests/e2e/ --env=staging

  deploy-production:
    needs: e2e-tests
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - run: kubectl set image deployment/a2a-protocol a2a-protocol=agentcore/a2a-protocol:${{ github.sha }} -n production
      - run: kubectl rollout status deployment/a2a-protocol -n production
```

## Implementation Checklist

### Phase 1: Foundation (Sprint 1, 13 SP)
- [x] FastAPI project scaffolding
- [x] PostgreSQL database schema (agents, tasks)
- [x] JSON-RPC 2.0 core processor
- [x] Agent registration endpoint
- [x] Basic agent discovery (exact matching)
- [x] Unit tests (90%+ coverage)

### Phase 2: Core Features (Sprint 2, 21 SP)
- [x] Task lifecycle management (create, status, artifacts)
- [x] Message routing with capability matching
- [x] JWT authentication and RBAC
- [x] WebSocket endpoints for real-time communication
- [x] Redis caching layer
- [x] Integration tests

### Phase 3: Production (Sprint 3, 31 SP)
- [x] Event system (SSE + Redis Streams)
- [x] Agent health monitoring
- [x] PostgreSQL optimization and indexing
- [x] Performance testing (load, stress)
- [x] Docker containerization
- [x] Kubernetes deployment manifests

### Phase 4: Semantic Enhancement (Sprint 4, 18 SP)
- [ ] pgvector extension setup
- [ ] Embedding generation (sentence-transformers)
- [ ] Semantic capability matching
- [ ] Cost-biased agent selection
- [ ] Context engineering patterns
- [ ] Vector search performance optimization

### Phase 5: Session Management (Sprint 5, 13 SP) - NEW
- [ ] SessionSnapshot Pydantic model
- [ ] session_snapshots PostgreSQL table
- [ ] Session save operation (<500ms)
- [ ] Session resume operation (<1s)
- [ ] Session lifecycle API (list, info, delete)
- [ ] Session integration tests

### Phase 6: Operations
- [ ] Prometheus metrics collection
- [ ] Grafana dashboards (4 dashboards)
- [ ] PagerDuty/Opsgenie alerting
- [ ] Runbooks for common incidents
- [ ] Blue-green deployment setup
- [ ] Automated backup testing

## References

**Internal:**
- [Specification](../specs/a2a-protocol/spec.md)
- [Implementation Plan](../specs/a2a-protocol/plan.md)
- [Task Breakdown](../specs/a2a-protocol/tasks.md)
- [CLI Layer Integration](../specs/cli-layer/spec.md)

**External:**
- [A2A Protocol Official Spec v0.2](https://a2a-protocol.org/v0.2.6/specification/)
- [Google A2A Tutorial](https://medium.com/google-cloud/getting-started-with-google-a2a-a-hands-on-tutorial-for-the-agent2agent-protocol-3d3b5e055127)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [pgvector Performance Optimization](https://cloud.google.com/blog/products/databases/faster-similarity-search-performance-with-pgvector-indexes)
- [FastAPI WebSocket Security](https://github.com/permitio/fastapi_websocket_rpc)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

**Research Papers:**
- "Efficient Vector Similarity Search with pgvector HNSW" (Google Cloud, 2024)
- "Agent-to-Agent Protocol Design Principles" (Google, 2024)

**Similar Implementations:**
- [Google Agent Development Kit (ADK)](https://developers.googleblog.com/en/agents-adk-agent-engine-a2a-enhancements-google-io/)
- [Microsoft Semantic Kernel A2A Integration](https://github.com/microsoft/semantic-kernel)
