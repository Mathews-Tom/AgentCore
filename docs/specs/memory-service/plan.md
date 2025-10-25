# Memory Service Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-25
**Specification:** `docs/specs/memory-service/spec.md` + `docs/specs/memory-system/spec.md`
**Research:** `docs/research/evolving-memory-system.md`
**Component ID:** MEM
**Priority:** P0 (HIGH - Phase 2)

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → COMPASS Enhancement → This Plan**

1. **Research Foundation:** `docs/research/evolving-memory-system.md`
   - Established four-layer memory architecture concept
   - Defined memory operations: encode, store, retrieve, update, prune
   - Identified vector database requirements and compression strategies
   - Expected benefits: 25-30% improvement on multi-turn tasks, 5-10x context reduction

2. **Formal Specification:** `docs/specs/memory-service/spec.md`
   - Applied research to Mem0 integration with four layers
   - Defined functional requirements (FR-1 to FR-7)
   - Specified technology stack: Qdrant, Redis, PostgreSQL, Mem0
   - Target: 100% persistence, 90%+ retrieval accuracy, <100ms search latency

3. **COMPASS Enhancement:** `docs/specs/memory-system/spec.md`
   - Added stage-aware context organization (planning, execution, reflection, verification)
   - Progressive compression: 10:1 stage, 5:1 task ratios
   - Test-time scaling with mini models for 70-80% cost reduction
   - Error tracking and ACE integration for meta-cognitive monitoring
   - Enhanced targets: 95%+ retrieval precision, 80%+ context reduction, 20% accuracy improvement

### Related Documentation

**System Context:**

- Architecture: `docs/agentcore-architecture-and-development-plan.md`
  - 6-layer architecture: Intelligence, Experience, Enterprise Ops, Core, Infrastructure, Runtime
  - Memory Service integrates with Core Layer (Task Management, Communication Hub)
  - Existing infrastructure: PostgreSQL with PGVector, Redis Cluster, FastAPI

**Dependencies:**

- A2A-009 (PostgreSQL Integration): Required for database schema
- A2A-004 (Task Management): Required for task_id references
- A2A-002 (JSON-RPC Core): Required for method registration
- A2A-019 to A2A-021 (Session Management): Integrates with session context

**Future Integrations:**

- ART-014 (Agent State Persistence): Will query memory for context
- ART-005 (Multi-Agent Coordination): Uses procedural memory patterns
- ART-013 (Error Handling): Integrates with error tracking

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Implement a comprehensive memory service combining four-layer memory architecture with COMPASS stage-aware context management to enable agents that learn, remember, and improve over long-horizon tasks.

**Value Proposition:**

- **Context Efficiency:** 80%+ reduction in context tokens through intelligent compression
- **Performance Improvement:** 20% increase in long-horizon task success rates
- **Cost Reduction:** 70-80% reduction in LLM API costs via test-time scaling
- **Knowledge Accumulation:** Agents build domain expertise over time
- **Error Prevention:** Explicit error tracking prevents compounding mistakes

**Target Users:**

- **Agents:** Four-layer memory with automatic transitions (working → episodic → semantic)
- **Sessions:** Coherent context across multi-turn conversations
- **ACE (Meta-Thinker):** Strategic context for intervention decisions
- **Developers:** Memory-aware routing and context optimization
- **Operators:** Agent behavior analysis and memory pattern insights

### Technical Approach

**Architecture Pattern:** Layered Service Architecture

- Orchestration layer (MemoryManager) coordinates memory operations
- Storage layers: Redis (working memory), PostgreSQL/PGVector (long-term memory)
- Service layers: Encoding, Retrieval, Compression, Stage Management, Error Tracking
- Integration layer: JSON-RPC methods for A2A protocol compliance

**Technology Stack:**

- **Database:** PostgreSQL 14+ with PGVector extension (existing infrastructure)
- **Cache:** Redis 6+ (existing infrastructure)
- **Memory Management:** Mem0 ^0.1.0 for extraction and fact recognition
- **Embeddings:** Hybrid (SentenceTransformers local + OpenAI for critical memories)
- **Compression:** gpt-4.1-mini for test-time scaling (per CLAUDE.md governance)
- **Framework:** Python 3.12+, FastAPI, Pydantic 2.5+, SQLAlchemy async

**Implementation Strategy:** Phased approach over 9-10 weeks

1. **Phase 1 (Weeks 1-2):** Foundation - Database, models, basic operations
2. **Phase 2 (Week 3):** Core operations - Vector search, retrieval, JSON-RPC
3. **Phase 3 (Weeks 4-5):** COMPASS stage management and compression
4. **Phase 4 (Week 6):** Error tracking and pattern detection
5. **Phase 5 (Week 7):** Enhanced retrieval with criticality scoring
6. **Phase 6 (Weeks 8-9):** Integration, optimization, validation

### Key Success Metrics

**Service Level Objectives (SLOs):**

- **Availability:** 99.9% uptime for memory operations
- **Response Time:** p95 <100ms for retrieval, <50ms for context formatting
- **Throughput:** 100+ concurrent operations, 1000+ memory stores per hour
- **Error Rate:** <1% failed operations

**Key Performance Indicators (KPIs):**

- **Context Efficiency:** 80%+ reduction in context tokens for long sessions
- **Retrieval Precision:** 95%+ relevant results (improved from 90% baseline)
- **Task Performance:** 20%+ improvement in multi-turn task success rates
- **Cost Reduction:** 70-80% lower LLM API costs via test-time scaling
- **Memory Coherence:** <5% contradictory retrievals
- **Storage Scalability:** 1M+ memories per agent without degradation
- **Critical Fact Coverage:** 100% of must-remember facts retained
- **Error Capture Rate:** 100% of errors recorded and tracked

---

## 💻 Code Examples & Patterns

### Repository Patterns (Will Establish New Patterns)

Since `.sage/agent/examples/` is empty (new project), this implementation will **establish** reusable patterns:

**1. Async Repository Pattern:** `.sage/agent/examples/python/data-access/async-repository.md`

- **Application:** MemoryRepository with async SQLAlchemy 2.0
- **Usage Example:**

  ```python
  from agentcore.database import get_session
  from agentcore.database.repositories import MemoryRepository

  async with get_session() as session:
      repo = MemoryRepository(session)
      memory = await repo.get_by_id(memory_id)
      await repo.add(new_memory)
      await session.commit()
  ```

- **Reusability:** Pattern for all AgentCore data access services

**2. Vector Search Pattern:** `.sage/agent/examples/python/vector-search/pgvector-similarity.md`

- **Application:** Hybrid search combining PGVector similarity + metadata filtering
- **Usage Example:**

  ```python
  from pgvector.sqlalchemy import Vector

  # Vector similarity search with metadata filters
  results = await session.execute(
      select(MemoryModel)
      .filter(MemoryModel.agent_id == agent_id)
      .order_by(MemoryModel.embedding.cosine_distance(query_embedding))
      .limit(k)
  )
  ```

- **Reusability:** Any service needing semantic search

**3. Multi-Layer Caching Pattern:** `.sage/agent/examples/python/caching/redis-multi-layer.md`

- **Application:** Redis cache with TTL-based eviction for working memory
- **Usage Example:**

  ```python
  import redis.asyncio as redis

  class WorkingMemoryService:
      async def set(self, key: str, value: dict, ttl: int = 3600):
          await self.redis.setex(
              key, ttl, json.dumps(value)
          )

      async def get(self, key: str) -> dict | None:
          data = await self.redis.get(key)
          return json.loads(data) if data else None
  ```

- **Reusability:** Fast temporary storage for any service

**4. Progressive Compression Pattern:** `.sage/agent/examples/python/context-engineering/progressive-compression.md`

- **Application:** Hierarchical summarization (raw → stage → task) with test-time scaling
- **Usage Example:**

  ```python
  class ContextCompressor:
      async def compress_stage(
          self,
          memories: list[MemoryRecord],
          model: str = "gpt-4.1-mini"
      ) -> StageMemory:
          # 10:1 compression using mini model
          summary = await self.llm.generate(
              prompt=self._build_compression_prompt(memories),
              model=model,
              max_tokens=len(memories) * 100 // 10
          )
          return StageMemory(summary=summary, compression_ratio=10.0)
  ```

- **Reusability:** Context compression for any long-running agent workflow

**5. JSON-RPC Service Pattern:** `.sage/agent/examples/python/jsonrpc/service-registration.md`

- **Application:** Register memory service methods using decorator pattern
- **Usage Example:**

  ```python
  from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

  @register_jsonrpc_method("memory.store")
  async def handle_memory_store(request: JsonRpcRequest) -> dict[str, Any]:
      content = request.params["content"]
      memory_id = await memory_manager.add_memory(content)
      return {"memory_id": memory_id}
  ```

- **Reusability:** All A2A protocol services

### Existing Patterns to Follow (from AgentCore codebase)

**From `src/agentcore/a2a_protocol/`:**

- Pydantic models for all data structures (follow `models/agent.py` pattern)
- Async/await for all I/O operations (follow `services/agent_manager.py`)
- Repository pattern for data access (follow `database/repositories.py`)
- JSON-RPC method registration (follow `services/agent_jsonrpc.py`)
- Configuration via Pydantic Settings (follow `config.py`)

---

## 🔧 Technology Stack

### Recommended Stack (from Research & Architecture Alignment)

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | Existing AgentCore standard, built-in generics per CLAUDE.md |
| Framework | FastAPI | 0.104.0+ | Existing infrastructure, async support |
| Database | PostgreSQL | 14+ | Existing (A2A-009), ACID guarantees, vector extension |
| Vector DB | PGVector | 0.4.1+ | **Already in dependencies**, simpler than Qdrant, sufficient for 1M vectors |
| Cache | Redis | 6+ | Existing cluster with streams, TTL support |
| Memory Library | Mem0 | ^0.1.0 | **Need to add**, extraction and fact recognition |
| Embeddings (Local) | SentenceTransformers | 5.1.1+ | **Already in dependencies**, free, 384-768 dims |
| Embeddings (Cloud) | OpenAI text-embedding-3-small | Latest | Optional for critical memories, 1536 dims, $0.02/1M tokens |
| Compression Model | gpt-4.1-mini | Latest | Per CLAUDE.md governance, $0.15/1M tokens, test-time scaling |
| ORM | SQLAlchemy | 2.0+ | Existing async support |
| Models | Pydantic | 2.5+ | Existing validation framework |
| Testing | pytest + testcontainers | Latest | Existing test infrastructure |

### Key Technology Decisions

**Decision 1: PGVector instead of Qdrant (Modified from Spec)**

- **Rationale from Research:** Research recommended Qdrant, Pinecone, Weaviate, or PGVector
- **Architectural Alignment:** PGVector already in AgentCore dependencies (pgvector>=0.4.1)
- **Operational Simplicity:** No separate service to deploy, manage, or scale
- **Performance Validation:** PGVector proven at 1M vectors with <100ms retrieval using IVFFlat index
- **Tradeoff:** Qdrant has better performance at 10M+ scale, but AgentCore targets 1M memories per agent
- **Fallback Plan:** Can migrate to Qdrant later if needed (compatible semantic search interface)

**Decision 2: Hybrid Embeddings (Cost Optimization)**

- **Rationale from Research:** Research emphasized cost efficiency (5-10x reduction)
- **COMPASS Enhancement:** Test-time scaling principle applies to embeddings too
- **Strategy:**
  - Default: Local SentenceTransformers (free, sufficient for most memories)
  - Critical memories only: OpenAI embeddings (higher quality, $0.02/1M tokens)
  - Automatic detection: Mark memories as critical based on importance
- **Expected Savings:** 70-80% embedding cost reduction while maintaining quality

**Decision 3: Test-Time Scaling for Compression (COMPASS)**

- **Rationale from Research:** COMPASS paper validated 70-80% cost reduction
- **Model Governance:** Per CLAUDE.md, use approved mini models (gpt-4.1-mini, gpt-5-mini)
- **Implementation:**
  - Compression: gpt-4.1-mini ($0.15/1M tokens)
  - Reasoning: gpt-4.1 or gpt-5 (agent runtime)
  - Validation: Track compression quality (≥95% fact retention)

**Decision 4: Mem0 for Memory Extraction**

- **Rationale:** Research identified need for entity extraction, fact recognition
- **Library Choice:** Mem0 provides these capabilities out-of-box
- **Risk Mitigation:** Abstract behind MemoryExtractor protocol for easy replacement if library breaks
- **Version Pinning:** Use `mem0>=0.1.0,<0.2.0` to prevent breaking changes

### Research Citations

**Four-Layer Memory Architecture:**

- Source: `docs/research/evolving-memory-system.md`
- Validation: 25-30% improvement on multi-turn tasks, 5-10x context reduction

**COMPASS Enhancements:**

- Source: `docs/specs/memory-system/spec.md` (based on COMPASS paper)
- Validation: 20% accuracy improvement, 60-80% context reduction, 70-80% cost savings

**Technology Recommendations:**

- Vector DB options: Research section "Resource Requirements"
- Embedding models: Research section "Embedding Model"
- Compression strategies: Research section "Context Compression"

### Alignment with Existing System

**From `docs/agentcore-architecture-and-development-plan.md`:**

**Consistent With:**

- PostgreSQL database layer (already deployed in A2A-009)
- Redis cache layer (already deployed as cluster)
- FastAPI gateway (existing A2A protocol infrastructure)
- SQLAlchemy async ORM (existing pattern in repositories)
- Pydantic models (existing pattern throughout codebase)

**New Additions:**

- PGVector extension for PostgreSQL (already in dependencies, just need to enable)
- Mem0 library (need to add `mem0>=0.1.0` to pyproject.toml)
- SentenceTransformers (already in dependencies: sentence-transformers>=5.1.1)

**Migration Considerations:**

- None - New service, no existing code to migrate
- Database migration: Add memory tables to existing PostgreSQL instance
- Redis namespace: Use separate DB or key prefix for memory cache

---

## 🏗️ Architecture Design

### System Context

**From `docs/agentcore-architecture-and-development-plan.md`:**

AgentCore employs a 6-layer architecture:

1. **External Integrations:** LLM providers, APIs, monitoring
2. **Intelligence Layer:** DSPy optimization, self-evolution, resource prediction
3. **Experience Layer:** Developer APIs, hot-reload, debugging
4. **Enterprise Operations:** Decision lineage, multi-tenancy, security
5. **Core Layer:** Registry, A2A Protocol, Task Management, Communication Hub, **Orchestration**, Advanced Cache
6. **Infrastructure:** PostgreSQL, Redis, FastAPI, UV project management

**Memory Service Integration:**

Memory Service belongs to the **Core Layer** and integrates with:

- **Task Management** (A2A-004): Store task artifacts, link memories to tasks
- **Session Management** (A2A-019 to A2A-021): Provide session context
- **Communication Hub** (A2A-005): Memory-aware message routing
- **Agent Registry** (A2A-003): Store agent preferences and learnings
- **ACE Integration** (Future): Provide strategic context for meta-cognitive monitoring

### Component Architecture

**Architecture Pattern:** Layered Service Architecture with Repository Pattern

**Rationale:**

- Aligns with existing AgentCore service structure (`src/agentcore/a2a_protocol/services/`)
- Separates concerns: orchestration, storage, retrieval, encoding, compression
- Enables independent testing and evolution of each layer
- Fits JSON-RPC method registration pattern

**System Design:**

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                     JSON-RPC Interface Layer                    │
│  (memory.store, memory.retrieve, memory.get_context, etc.)     │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    Memory Manager (Orchestration)               │
│  - Coordinates encode/store/retrieve/update/prune               │
│  - Manages layer transitions (working → episodic → semantic)    │
│  - Triggers compression and error tracking                      │
└─┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
  │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌──────────┐
│Work │  │Epis │  │Sema │  │Proc │  │ Stage    │
│Mem  │  │odic │  │ntic │  │edur │  │ Manager  │
│(Red │  │(PG  │  │(PG  │  │al   │  │(COMPASS) │
│is)  │  │Vect)│  │Vect)│  │(PG  │  │          │
└─────┘  └─────┘  └─────┘  │Vect)│  └──────────┘
                            └─────┘
  │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼
┌───────────────────────────────────────────────────┐
│         Shared Services Layer                     │
│  - EncodingService (entity/fact extraction)       │
│  - RetrievalService (hybrid search + scoring)     │
│  - CompressionService (progressive compression)   │
│  - ErrorTracker (pattern detection)               │
└───────────────────┬───────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────┐
│         Storage & External Services               │
│  - PostgreSQL/PGVector (metadata + vectors)       │
│  - Redis (working memory cache)                   │
│  - Mem0 (memory extraction)                       │
│  - OpenAI/Local (embeddings)                      │
└───────────────────────────────────────────────────┘
```

### Architecture Decisions

**Decision 1: Four-Layer Memory Organization (From Research)**

- **Choice:** Working (Redis) + Episodic (PGVector) + Semantic (PGVector) + Procedural (PGVector)
- **Rationale:** Research validated this structure for bounded context with unbounded memory
- **Implementation:**
  - Working memory: Redis cache with 1-hour TTL (2-4K tokens)
  - Long-term memory: PGVector with memory_type field to distinguish layers
  - Automatic transitions based on importance, time, access patterns
- **Trade-offs:** Complexity vs flexibility; chosen for proven effectiveness

**Decision 2: COMPASS Stage-Aware Context Management (Enhancement)**

- **Choice:** Hierarchical organization (raw → stage → task) with progressive compression
- **Rationale:** COMPASS paper validated 20% accuracy improvement, 60-80% context reduction
- **Implementation:**
  - Stage detection: Analyze agent action patterns (planning/execution/reflection/verification)
  - Stage completion: Compress raw memories → stage summary (10:1 ratio)
  - Task compression: Compress stage summaries → task progress (5:1 ratio)
  - Quality monitoring: Track fact retention (≥95% target)
- **Trade-offs:** Added complexity vs significant performance/cost benefits

**Decision 3: Hybrid Search with Multi-Factor Importance Scoring**

- **Choice:** Combine vector similarity + metadata filtering + multi-factor scoring
- **Rationale:** Research showed pure vector search misses critical evidence
- **Implementation:**
  - Vector similarity (PGVector cosine distance): 35% weight
  - Recency (exponential decay): 15% weight
  - Frequency (log access count): 10% weight
  - Stage relevance (current stage match): 20% weight
  - Criticality boost (2x multiplier): 10% weight
  - Error correction (recent error relevance): 10% weight
- **Trade-offs:** Query complexity vs retrieval quality (95% target)

**Decision 4: Test-Time Scaling for Cost Optimization (COMPASS)**

- **Choice:** Use mini models for compression, full models for reasoning
- **Rationale:** COMPASS validated 70-80% cost reduction with no quality loss
- **Implementation:**
  - Compression: gpt-4.1-mini ($0.15/1M tokens)
  - Reasoning: gpt-4.1 or gpt-5 (agent runtime)
  - Monitoring: Track compression quality, fallback if degradation
- **Trade-offs:** Slight latency increase vs massive cost savings

### Component Breakdown

**Core Components:**

**1. MemoryManager (Orchestration)**

- **Purpose:** Coordinate all memory operations, manage layer transitions
- **Technology:** Python 3.12, async/await
- **Pattern:** Facade pattern over memory services
- **Interfaces:**
  - `add_interaction(interaction: Interaction) -> str` - Store new memory
  - `get_relevant_context(query: str, task_id: str, max_tokens: int) -> str` - Retrieve formatted context
  - `retrieve_memories(query: str, memory_layers: list[str], k: int) -> list[MemoryRecord]` - Search memories
  - `complete_stage(stage_id: str) -> StageMemory` - Trigger stage compression
- **Dependencies:** All memory services, stage manager, error tracker

**2. WorkingMemoryService (Redis Layer)**

- **Purpose:** Fast access to immediate context (2-4K tokens, 1-hour TTL)
- **Technology:** Redis async client (redis.asyncio)
- **Pattern:** Cache-aside pattern
- **Interfaces:**
  - `set_working_memory(task_id: str, context: dict, ttl: int = 3600)`
  - `get_working_memory(task_id: str) -> dict | None`
  - `clear_working_memory(task_id: str)`
- **Dependencies:** Redis cluster

**3. VectorMemoryService (PGVector Layer)**

- **Purpose:** Long-term storage for episodic, semantic, procedural memories
- **Technology:** PGVector extension, SQLAlchemy async
- **Pattern:** Repository pattern with vector search
- **Interfaces:**
  - `store_memory(record: MemoryRecord) -> str`
  - `search_similar(embedding: list[float], filters: dict, k: int) -> list[MemoryRecord]`
  - `get_by_id(memory_id: str) -> MemoryRecord`
  - `update_access_count(memory_id: str)`
- **Dependencies:** PostgreSQL with PGVector, MemoryRepository

**4. EncodingService (Entity/Fact Extraction)**

- **Purpose:** Extract structured information from interactions
- **Technology:** Mem0 library, local LLM for extraction
- **Pattern:** Strategy pattern (multiple extraction strategies)
- **Interfaces:**
  - `encode_memory(interaction: Interaction) -> MemoryRecord`
  - `extract_entities(content: str) -> list[str]`
  - `extract_facts(content: str) -> list[str]`
  - `generate_embedding(content: str, critical: bool = False) -> list[float]`
- **Dependencies:** Mem0, SentenceTransformers, optionally OpenAI

**5. RetrievalService (Hybrid Search + Scoring)**

- **Purpose:** Retrieve relevant memories with multi-factor importance scoring
- **Technology:** PGVector search + custom scoring algorithm
- **Pattern:** Template method pattern (scoring strategy)
- **Interfaces:**
  - `retrieve(query: str, filters: dict, k: int, stage_type: str | None) -> list[ScoredMemory]`
  - `compute_importance_score(memory: MemoryRecord, query: str, context: dict) -> float`
  - `validate_retrieval_quality(results: list[MemoryRecord]) -> QualityMetrics`
- **Dependencies:** VectorMemoryService, scoring algorithms

**6. CompressionService (Progressive Compression) - COMPASS**

- **Purpose:** Compress memories hierarchically (raw → stage → task)
- **Technology:** LLM API (gpt-4.1-mini for test-time scaling)
- **Pattern:** Chain of responsibility (compression levels)
- **Interfaces:**
  - `compress_stage(memories: list[MemoryRecord], model: str) -> StageMemory`
  - `compress_task(stages: list[StageMemory], model: str) -> TaskContext`
  - `validate_compression_quality(original: list, compressed: str) -> CompressionQuality`
  - `extract_critical_facts(compressed: str) -> list[str]`
- **Dependencies:** LLM service, fact extraction

**7. StageManager (Stage Detection & Transitions) - COMPASS**

- **Purpose:** Detect reasoning stages, manage transitions, trigger compression
- **Technology:** Pattern matching on agent actions
- **Pattern:** State machine pattern
- **Interfaces:**
  - `detect_current_stage(actions: list[str]) -> StageType`
  - `check_stage_transition(stage_id: str, actions: list[str]) -> bool`
  - `complete_stage(stage_id: str) -> StageMemory`
  - `get_stage_context(stage_id: str, max_tokens: int) -> str`
- **Dependencies:** CompressionService, VectorMemoryService

**8. ErrorTracker (Error Recording & Patterns) - COMPASS**

- **Purpose:** Track errors explicitly, detect patterns, enable reflection
- **Technology:** PostgreSQL storage + LLM pattern extraction
- **Pattern:** Observer pattern (error events)
- **Interfaces:**
  - `record_error(error: ErrorRecord)`
  - `detect_patterns(task_id: str, lookback: int) -> list[ErrorPattern]`
  - `get_error_history(filters: dict, limit: int) -> list[ErrorRecord]`
  - `get_error_aware_context(query: str, recent_errors: list[ErrorRecord]) -> str`
- **Dependencies:** ErrorRepository, LLM service

### Data Flow & Boundaries

**Request Flow: Store Memory**

1. Agent interaction occurs → `memory.store` JSON-RPC call
2. MemoryManager.add_interaction() receives request
3. EncodingService.encode_memory() extracts entities/facts, generates embedding
4. VectorMemoryService.store_memory() persists to PostgreSQL/PGVector
5. WorkingMemoryService.set_working_memory() updates Redis cache
6. StageManager.check_stage_transition() checks if stage should complete
7. If stage completes → CompressionService.compress_stage() creates summary
8. ErrorTracker.record_error() if error occurred
9. Return memory_id to caller

**Request Flow: Retrieve Context**

1. Agent needs context → `memory.get_context` JSON-RPC call
2. MemoryManager.get_relevant_context() receives query
3. WorkingMemoryService.get_working_memory() fetches immediate context
4. RetrievalService.retrieve() searches relevant episodic/semantic memories
5. RetrievalService.compute_importance_score() ranks memories
6. CompressionService.format_context() creates formatted context string
7. Validate token budget, truncate if needed (preserve critical facts)
8. Return formatted context

**Request Flow: Stage Completion (COMPASS)**

1. Stage detection signals completion → `memory.complete_stage` call
2. StageManager.complete_stage() retrieves all raw memories from stage
3. CompressionService.compress_stage() sends to mini model
4. CompressionService.validate_compression_quality() checks fact retention
5. VectorMemoryService.store_memory() persists StageMemory
6. Update TaskContext with completed stage
7. Return StageMemory summary

**Component Boundaries:**

**Public Interface (JSON-RPC Methods):**

- `memory.store`, `memory.retrieve`, `memory.get_context`
- `memory.get_stage_context`, `memory.complete_stage` (COMPASS)
- `memory.record_error`, `memory.detect_error_patterns` (COMPASS)
- `memory.compress_stage`, `memory.get_compression_metrics` (COMPASS)
- `memory.get_strategic_context` (ACE integration)

**Internal Implementation (Service Layer):**

- MemoryManager orchestration logic
- EncodingService extraction algorithms
- RetrievalService scoring algorithms
- CompressionService compression logic
- StageManager stage detection
- ErrorTracker pattern detection

**Cross-Component Contracts:**

- TaskManager: Provides task_id, receives artifact storage
- SessionManager: Provides session_id, receives session context
- MessageRouter: Receives memory-aware routing context
- ACE: Provides metrics, receives strategic context

---

## 🔧 Technical Specification

### Data Model

**MemoryRecord (Base Model)**

```python
class MemoryLayer(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryRecord(BaseModel):
    memory_id: str
    memory_layer: MemoryLayer
    content: str
    summary: str
    embedding: list[float]

    # Scope
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    task_id: str | None = None

    # Metadata
    timestamp: datetime
    entities: list[str] = []
    facts: list[str] = []
    keywords: list[str] = []

    # Relationships
    related_memory_ids: list[str] = []
    parent_memory_id: str | None = None

    # Tracking
    relevance_score: float = 1.0
    access_count: int = 0
    last_accessed: datetime | None = None

    # COMPASS enhancements
    stage_id: str | None = None
    is_critical: bool = False
    criticality_reason: str | None = None

    # Procedural
    actions: list[str] = []
    outcome: str | None = None
    success: bool | None = None
```

**StageMemory (COMPASS)**

```python
class StageType(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    VERIFICATION = "verification"

class StageMemory(BaseModel):
    stage_id: str
    task_id: str
    agent_id: str
    stage_type: StageType
    stage_summary: str
    stage_insights: list[str]
    raw_memory_refs: list[str]  # IDs of raw memories
    relevance_score: float = 1.0
    compression_ratio: float
    compression_model: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
```

**TaskContext (COMPASS)**

```python
class TaskContext(BaseModel):
    task_id: str
    agent_id: str
    task_goal: str
    current_stage_id: str | None
    task_progress_summary: str
    critical_constraints: list[str]
    performance_metrics: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**ErrorRecord (COMPASS)**

```python
class ErrorType(str, Enum):
    HALLUCINATION = "hallucination"
    MISSING_INFO = "missing_info"
    INCORRECT_ACTION = "incorrect_action"
    CONTEXT_DEGRADATION = "context_degradation"

class ErrorRecord(BaseModel):
    error_id: str
    task_id: str
    stage_id: str | None
    agent_id: str
    error_type: ErrorType
    error_description: str
    context_when_occurred: str
    recovery_action: str | None
    error_severity: float  # 0-1 scale
    recorded_at: datetime
```

**Database Schema (PostgreSQL):**

```sql
-- Memories table (all layers except working)
CREATE TABLE memories (
    memory_id UUID PRIMARY KEY,
    memory_layer VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding VECTOR(768),  -- Using local embeddings (768 dims)

    -- Scope
    agent_id UUID,
    session_id UUID,
    user_id UUID,
    task_id UUID,

    -- Metadata
    timestamp TIMESTAMP NOT NULL,
    entities TEXT[],
    facts TEXT[],
    keywords TEXT[],

    -- Relationships
    related_memory_ids UUID[],
    parent_memory_id UUID,

    -- Tracking
    relevance_score FLOAT DEFAULT 1.0,
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMP,

    -- COMPASS
    stage_id UUID,
    is_critical BOOLEAN DEFAULT FALSE,
    criticality_reason TEXT,

    -- Procedural
    actions TEXT[],
    outcome TEXT,
    success BOOLEAN,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_memories_layer ON memories(memory_layer);
CREATE INDEX idx_memories_agent ON memories(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_memories_session ON memories(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_memories_task ON memories(task_id) WHERE task_id IS NOT NULL;
CREATE INDEX idx_memories_stage ON memories(stage_id) WHERE stage_id IS NOT NULL;
CREATE INDEX idx_memories_critical ON memories(is_critical) WHERE is_critical = TRUE;
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Stage memories table
CREATE TABLE stage_memories (
    stage_id UUID PRIMARY KEY,
    task_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    stage_type VARCHAR(50) NOT NULL,
    stage_summary TEXT NOT NULL,
    stage_insights TEXT[],
    raw_memory_refs UUID[] NOT NULL,
    relevance_score FLOAT DEFAULT 1.0,
    compression_ratio FLOAT,
    compression_model VARCHAR(100),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE INDEX idx_stage_memories_task ON stage_memories(task_id);
CREATE INDEX idx_stage_memories_type ON stage_memories(stage_type);
CREATE INDEX idx_stage_memories_agent ON stage_memories(agent_id);

-- Task contexts table
CREATE TABLE task_contexts (
    task_id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    task_goal TEXT NOT NULL,
    current_stage_id UUID,
    task_progress_summary TEXT,
    critical_constraints TEXT[],
    performance_metrics JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_task_contexts_agent ON task_contexts(agent_id);

-- Error records table
CREATE TABLE error_records (
    error_id UUID PRIMARY KEY,
    task_id UUID NOT NULL,
    stage_id UUID,
    agent_id UUID NOT NULL,
    error_type VARCHAR(50) NOT NULL,
    error_description TEXT NOT NULL,
    context_when_occurred TEXT,
    recovery_action TEXT,
    error_severity FLOAT CHECK (error_severity BETWEEN 0 AND 1),
    recorded_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_error_records_task ON error_records(task_id);
CREATE INDEX idx_error_records_stage ON error_records(stage_id);
CREATE INDEX idx_error_records_type ON error_records(error_type);

-- Compression metrics table
CREATE TABLE compression_metrics (
    metric_id UUID PRIMARY KEY,
    stage_id UUID,
    task_id UUID,
    compression_type VARCHAR(50),
    input_tokens INT,
    output_tokens INT,
    compression_ratio FLOAT,
    critical_fact_retention_rate FLOAT,
    coherence_score FLOAT,
    cost_usd DECIMAL(10,4),
    model_used VARCHAR(100),
    recorded_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_compression_metrics_task ON compression_metrics(task_id);
```

**Validation Rules:**

- At least one scope field (agent_id, session_id, user_id) must be set
- Embedding dimensions must match configured model (768 for local, 1536 for OpenAI)
- Compression ratio must be between 1.0 and 20.0
- Error severity must be between 0.0 and 1.0
- Memory layer transitions: working → episodic → semantic (procedural parallel)

**Indexing Strategy:**

- IVFFlat index on embeddings for fast similarity search (lists = sqrt(total_vectors))
- B-tree indexes on agent_id, session_id, task_id for filtering
- Partial indexes on is_critical (only index true values)
- Composite index on (task_id, timestamp) for timeline queries

**Migration Approach:**

- Alembic migration to add all tables
- Enable PGVector extension in PostgreSQL
- Create indexes after initial data load (for performance)
- Backup existing task/session data before migration

### API Design

**Top 6 Critical Endpoints (JSON-RPC Methods):**

**1. memory.store - Store New Memory**

```json
{
  "jsonrpc": "2.0",
  "method": "memory.store",
  "params": {
    "content": "User prefers detailed technical explanations",
    "agent_id": "agent-123",
    "session_id": "session-456",
    "metadata": {
      "memory_layer": "semantic",
      "importance": 0.9
    }
  },
  "id": 1
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "memory_id": "mem-789",
    "summary": "User preference: detailed technical explanations",
    "entities": ["user", "preference"],
    "facts": ["prefers detailed explanations", "likes technical content"]
  },
  "id": 1
}
```

**Error Handling:**

- Invalid scope: JsonRpcErrorCode.INVALID_PARAMS
- Encoding failure: JsonRpcErrorCode.INTERNAL_ERROR
- Database error: Retry with exponential backoff

**2. memory.retrieve - Semantic Search**

```json
{
  "jsonrpc": "2.0",
  "method": "memory.retrieve",
  "params": {
    "query": "What does the user prefer for explanations?",
    "agent_id": "agent-123",
    "limit": 5,
    "memory_layers": ["semantic", "episodic"],
    "scoring_weights": {
      "relevance": 0.4,
      "recency": 0.2,
      "frequency": 0.2,
      "criticality": 0.2
    }
  },
  "id": 2
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "memories": [
      {
        "memory_id": "mem-789",
        "summary": "User preference: detailed technical explanations",
        "importance_score": 0.92,
        "timestamp": "2025-10-20T14:30:00Z",
        "is_critical": true
      }
    ],
    "total_found": 12,
    "query_time_ms": 45
  },
  "id": 2
}
```

**Error Handling:**

- Invalid filters: JsonRpcErrorCode.INVALID_PARAMS
- Search timeout: Return partial results with warning

**3. memory.get_context - Get Formatted Context**

```json
{
  "jsonrpc": "2.0",
  "method": "memory.get_context",
  "params": {
    "query": "Explain the authentication system",
    "task_id": "task-101",
    "max_tokens": 2000,
    "include_stage_context": true
  },
  "id": 3
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "context": "## Task Progress\\n[Task summary]\\n\\n## Current Stage Context\\n[Stage summary]\\n\\n## Relevant Past Work\\n[Episodic memories]\\n\\n## Critical Facts\\n[Must-remember constraints]",
    "tokens_used": 1850,
    "critical_facts_included": 5,
    "compression_ratio": 8.5
  },
  "id": 3
}
```

**Error Handling:**

- Task not found: Return empty context
- Token budget exceeded: Truncate with priority (critical facts first)

**4. memory.complete_stage - Trigger Stage Compression (COMPASS)**

```json
{
  "jsonrpc": "2.0",
  "method": "memory.complete_stage",
  "params": {
    "stage_id": "stage-555",
    "compression_model": "gpt-4.1-mini"
  },
  "id": 4
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "stage_id": "stage-555",
    "stage_summary": "Planned authentication implementation using JWT...",
    "compression_ratio": 10.2,
    "critical_facts": ["Use JWT for auth", "Store tokens in Redis"],
    "quality_metrics": {
      "fact_retention_rate": 0.97,
      "coherence_score": 1.0,
      "cost_usd": 0.0042
    }
  },
  "id": 4
}
```

**Error Handling:**

- Compression quality too low: Retry with less aggressive compression
- LLM timeout: Fallback to simple concatenation

**5. memory.record_error - Track Error (COMPASS)**

```json
{
  "jsonrpc": "2.0",
  "method": "memory.record_error",
  "params": {
    "task_id": "task-101",
    "stage_id": "stage-555",
    "error_type": "incorrect_action",
    "error_description": "Used wrong endpoint for token refresh",
    "error_severity": 0.6,
    "recovery_action": "Corrected endpoint to /auth/refresh"
  },
  "id": 5
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "error_id": "err-999",
    "patterns_detected": [
      {
        "pattern_type": "endpoint_confusion",
        "frequency": 3,
        "prevention_strategy": "Always check API documentation first"
      }
    ]
  },
  "id": 5
}
```

**Error Handling:**

- Invalid error type: JsonRpcErrorCode.INVALID_PARAMS

**6. memory.get_strategic_context - ACE Integration (COMPASS)**

```json
{
  "jsonrpc": "2.0",
  "method": "memory.get_strategic_context",
  "params": {
    "task_id": "task-101",
    "decision_type": "intervention_needed",
    "performance_metrics": {
      "error_rate": 0.35,
      "progress_rate": 0.15
    }
  },
  "id": 6
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "strategic_context": "High error rate detected. Recent errors show endpoint confusion pattern. Recommend reviewing API documentation and past successful authentication implementations.",
    "critical_memories": [
      {"memory_id": "mem-success-auth", "summary": "Successful JWT implementation pattern"}
    ],
    "error_patterns": [
      {"pattern": "endpoint_confusion", "frequency": 3}
    ],
    "recommended_action": "reflect_on_errors"
  },
  "id": 6
}
```

### Security

**Authentication/Authorization:**

- **Approach:** JWT authentication via A2A protocol (existing)
- **Implementation:** Memory operations require valid JWT with agent_id claim
- **Scope Validation:** Agent can only access memories scoped to their agent_id
- **Standards:** Follow AgentCore security layer (A2A-006)

**Secrets Management:**

- **Strategy:** Environment variables for API keys
- **OpenAI API Key:** `OPENAI_API_KEY` (for embeddings and compression)
- **Redis Connection:** `REDIS_URL` (existing)
- **PostgreSQL:** `DATABASE_URL` (existing)
- **Rotation:** Use secret management service (AWS Secrets Manager, HashiCorp Vault)

**Data Protection:**

- **Encryption in Transit:** TLS 1.3 for all external API calls (OpenAI, Mem0)
- **Encryption at Rest:** PostgreSQL database-level encryption
- **PII Handling:**
  - Never store PII in memory content (extract and hash)
  - Anonymize user_id in logs
  - Support GDPR right-to-be-forgotten (memory deletion)

**Security Testing:**

- **Approach:** SAST via Bandit (already in dependencies)
- **Tools:** bandit>=1.8.6 for vulnerability scanning
- **Injection Prevention:** Pydantic validation prevents SQL injection
- **API Key Protection:** Never log API keys, mask in error messages

**Compliance:**

- **GDPR:** Support memory deletion by user_id
- **Data Minimization:** Only store necessary information
- **Audit Logging:** Log all memory operations with agent_id and timestamp

### Performance

**Performance Targets (from COMPASS spec):**

- **Response Time:**
  - p50 <50ms for retrieval
  - p95 <100ms for retrieval
  - p99 <200ms for retrieval
  - p95 <50ms for context formatting
  - p95 <5s for stage compression
- **Throughput:**
  - 100+ concurrent retrieval requests
  - 1000+ memory storage operations per hour
  - 100+ stage compressions per hour
- **Resource Usage:**
  - Memory: <2GB per 100K memories
  - CPU: <50% during normal load
  - Database connections: <20 concurrent

**Caching Strategy:**

- **Approach:** Multi-level caching (Redis + in-memory)
- **Pattern:** Cache-aside for working memory, write-through for frequently accessed
- **Implementation:**
  - **L1 Cache (In-Memory):** Recent retrieval results (LRU, 100 entries, 5-minute TTL)
  - **L2 Cache (Redis):** Working memory (1-hour TTL), frequent queries (30-minute TTL)
  - **L3 (Database):** Long-term memories with vector index
- **TTL Strategy:**
  - Working memory: 1 hour (task-based)
  - Frequent queries: 30 minutes
  - Retrieval results: 5 minutes
- **Invalidation:**
  - On memory update: Invalidate specific memory_id
  - On stage completion: Invalidate stage_id cache
  - Time-based: TTL expiration

**Database Optimization:**

- **Indexing:**
  - IVFFlat index on embeddings (lists = sqrt(total_vectors), ~316 for 100K)
  - B-tree on agent_id, session_id, task_id, timestamp
  - Partial index on is_critical = true
- **Query Patterns:**
  - Use LIMIT with vector search to prevent full scans
  - Batch memory inserts (10 at a time)
  - Lazy loading for embeddings (only fetch when needed)
- **Connection Pooling:**
  - SQLAlchemy async pool: min=5, max=20 connections
  - Connection recycling: 3600 seconds
- **Partitioning:**
  - Consider time-based partitioning if >10M memories
  - Partition by agent_id if multi-tenant at scale

**Scaling Strategy:**

- **Horizontal:**
  - Read replicas for PostgreSQL (search queries)
  - Redis cluster already deployed
  - Stateless memory service (multiple instances behind load balancer)
- **Vertical:**
  - PostgreSQL: 4-8GB RAM for 1M vectors
  - Redis: 2-4GB RAM for working memory
- **Auto-scaling:**
  - Kubernetes HPA based on CPU (target: 70%)
  - Scale read replicas based on query latency
- **Performance Monitoring:**
  - Prometheus metrics: query latency, cache hit rate, compression cost
  - Grafana dashboards: Memory service health
  - Alerts: p95 latency >100ms, cache hit rate <70%, compression quality <90%

---

## 📋 Development Setup

### Required Tools and Versions

**Runtime:**

- Python 3.12+ (existing AgentCore standard)
- UV package manager (existing)

**Databases:**

- PostgreSQL 14+ with PGVector extension
- Redis 6+

**Development Tools:**

- pytest 7.4+ (existing)
- pytest-asyncio 0.23+ (existing)
- testcontainers[redis] 3.7+ (existing)
- ruff 0.1+ (existing)
- mypy 1.7+ (existing)

### Local Environment

**Docker Compose (docker-compose.dev.yml):**

```yaml
services:
  # Existing services: postgres, redis

  # Enable PGVector extension
  postgres:
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_DB: agentcore
      POSTGRES_USER: agentcore
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-pgvector.sql:/docker-entrypoint-initdb.d/init.sql

  # Existing Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
```

**PGVector Initialization Script (scripts/init-pgvector.sql):**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

**Environment Variables (.env):**

```bash
# Existing AgentCore variables
DATABASE_URL=postgresql+asyncpg://agentcore:dev_password@localhost:5432/agentcore
REDIS_URL=redis://localhost:6379/0

# Memory Service additions
REDIS_MEMORY_DB=1
OPENAI_API_KEY=sk-xxx  # For embeddings and compression
MEMORY_EMBEDDING_MODEL=sentence-transformers  # or "openai"
MEMORY_COMPRESSION_MODEL=gpt-4.1-mini
MEMORY_MAX_TOKENS_BUDGET=1000000  # Monthly token budget

# Working Memory Configuration
WORKING_MEMORY_TTL=3600
WORKING_MEMORY_MAX_TOKENS=4000

# COMPASS Configuration
STAGE_COMPRESSION_ENABLED=true
STAGE_COMPRESSION_RATIO_TARGET=10.0
ERROR_TRACKING_ENABLED=true
```

**Setup Commands:**

```bash
# Install dependencies (adds Mem0)
uv add "mem0>=0.1.0"

# Start infrastructure
docker compose -f docker-compose.dev.yml up -d

# Run database migrations
uv run alembic upgrade head

# Verify PGVector
uv run python scripts/verify_pgvector.py

# Run tests
uv run pytest tests/unit/test_memory_service.py
uv run pytest tests/integration/test_memory_integration.py

# Start development server
uv run uvicorn agentcore.a2a_protocol.main:app --reload --port 8001
```

### CI/CD Pipeline Requirements

**GitHub Actions Workflow (.github/workflows/memory-service.yml):**

```yaml
name: Memory Service CI

on:
  push:
    paths:
      - 'src/agentcore/a2a_protocol/models/memory.py'
      - 'src/agentcore/a2a_protocol/services/memory/**'
      - 'tests/**/test_memory*.py'

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg14
        env:
          POSTGRES_PASSWORD: test_password
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
      redis:
        image: redis:7-alpine

    steps:
      - uses: actions/checkout@v4
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync
      - name: Run linting
        run: uv run ruff check src/agentcore/a2a_protocol/services/memory/
      - name: Run type checking
        run: uv run mypy src/agentcore/a2a_protocol/services/memory/
      - name: Run unit tests
        run: uv run pytest tests/unit/test_memory*.py --cov=src/agentcore/a2a_protocol/services/memory --cov-report=xml
      - name: Run integration tests
        run: uv run pytest tests/integration/test_memory*.py
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Quality Gates:**

- Unit test coverage: ≥90% (enforced by pytest-cov)
- Type checking: 100% (mypy strict mode)
- Linting: 0 errors (ruff)
- Integration tests: 100% pass rate
- Performance benchmarks: p95 <100ms retrieval

### Testing Framework and Coverage Targets

**Testing Strategy:**

**Unit Tests (90%+ coverage):**

- Location: `tests/unit/test_memory_*.py`
- Mock external dependencies (OpenAI, Mem0, Redis, PostgreSQL)
- Test encoding logic, scoring algorithms, compression quality
- Fast execution (<1 second per test)

**Integration Tests:**

- Location: `tests/integration/test_memory_integration.py`
- Real PostgreSQL with PGVector (testcontainers)
- Real Redis (testcontainers)
- Test end-to-end flows, cross-component integration
- Moderate execution (<30 seconds total)

**Performance Tests:**

- Location: `tests/performance/test_memory_performance.py`
- Benchmark retrieval latency with varying dataset sizes (10K, 100K, 1M)
- Benchmark compression throughput
- Validate caching effectiveness

**Load Tests:**

- Location: `tests/load/test_memory_load.py`
- Simulate concurrent operations (100+ queries)
- Test memory service under sustained load
- Validate no degradation over time

**Example Test Structure:**

```python
# tests/unit/test_memory_retrieval.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from agentcore.a2a_protocol.services.memory.retrieval import RetrievalService

@pytest.mark.asyncio
async def test_compute_importance_score():
    """Test multi-factor importance scoring."""
    retrieval = RetrievalService()
    memory = create_test_memory(
        relevance_score=0.9,
        access_count=5,
        timestamp=datetime.now() - timedelta(hours=2)
    )

    score = retrieval.compute_importance_score(
        memory=memory,
        query="test query",
        context={"current_stage": "execution"}
    )

    assert 0.0 <= score <= 1.0
    assert score > 0.5  # High relevance + recent + accessed

# tests/integration/test_memory_integration.py
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

@pytest.mark.asyncio
async def test_memory_persistence_across_restart():
    """Test memories persist across service restart."""
    async with PostgresContainer("pgvector/pgvector:pg14") as postgres:
        # Store memory
        memory_id = await memory_service.add_memory(
            content="Test memory",
            agent_id="test-agent"
        )

        # Simulate restart
        await memory_service.shutdown()
        await memory_service.initialize()

        # Retrieve memory
        retrieved = await memory_service.get_memory(memory_id)
        assert retrieved.content == "Test memory"
```

---

## ⚠️ Risk Management

### Critical Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation | Validation |
|------|--------|------------|------------|------------|
| **PGVector Performance at Scale** | HIGH - If cannot handle 1M vectors with <100ms retrieval | MEDIUM - PGVector proven but requires tuning | - Use IVFFlat index with proper list count<br>- Query optimization (LIMIT, early termination)<br>- Fallback: Migrate to Qdrant if needed | Benchmark with 100K vectors in Phase 1 |
| **Mem0 Library Stability** | MEDIUM - v0.1.0 is early stage, API changes possible | MEDIUM - Early library | - Pin exact version `mem0==0.1.x`<br>- Abstract behind MemoryExtractor protocol<br>- Fallback: Custom LLM extraction | Comprehensive integration tests |
| **Cost Overrun from Embeddings** | HIGH - OpenAI embeddings could exceed budget | LOW - Mitigated by hybrid approach | - Default to local SentenceTransformers<br>- OpenAI only for critical memories<br>- Track costs, alert at 75% budget | Cost tracking dashboard |
| **Compression Quality Degradation** | HIGH - Poor compression loses critical info | MEDIUM - LLM summarization can miss details | - Validate compression (≥95% fact retention)<br>- Preserve originals as backup<br>- Adaptive compression | Quality metrics after each compression |
| **Dependency on A2A-009** | CRITICAL - Cannot deploy without PostgreSQL | LOW - Likely complete | - Verify A2A-009 status before starting<br>- Request priority if blocked<br>- Develop with local PostgreSQL in parallel | Check ticket status |
| **ACE Integration Complexity** | MEDIUM - ACE component may not be ready | MEDIUM - Future integration | - Design ACE interface early<br>- Mock ACE for testing<br>- Make ACE integration optional (Phase 6) | Stub ACE interface in Phase 1 |
| **Stage Detection Accuracy** | MEDIUM - Wrong stage = wrong context | MEDIUM - Heuristic-based detection | - Multiple detection strategies (action patterns, explicit signals, time-based)<br>- Allow manual stage markers<br>- Validate accuracy ≥90% | A/B test stage detection algorithms |
| **Token Budget Exhaustion** | HIGH - Exceeding budget stops service | LOW - Monitoring prevents | - Hard cap at monthly budget<br>- Alert at 75% consumption<br>- Fallback: Disable compression, use local embeddings | Monthly cost reports |

### Monitoring and Alerting

**Metrics to Track:**

- Retrieval latency (p50, p95, p99)
- Compression quality (fact retention rate, coherence score)
- Cost metrics (tokens, dollars per operation)
- Error rates by type
- Cache hit rates
- Database query performance
- Memory service availability

**Alert Thresholds:**

- **Critical:** p95 latency >200ms, error rate >5%, availability <99%, compression quality <90%
- **Warning:** p95 latency >100ms, cache hit rate <70%, cost budget >75%
- **Info:** Unusual access patterns, new error types

---

## 📅 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) - P0

**Week 1: Database & Models**

- Database schema migration (memories, stage_memories, task_contexts, error_records)
- Enable PGVector extension in PostgreSQL
- Create Pydantic models (MemoryRecord, StageMemory, TaskContext, ErrorRecord)
- MemoryRepository with basic CRUD operations
- Unit tests for models and repository

**Week 2: Basic Memory Operations**

- MemoryManager orchestration layer
- WorkingMemoryService (Redis cache integration)
- EncodingService (local embeddings with SentenceTransformers)
- Basic VectorMemoryService (PGVector storage and retrieval)
- Integration tests with testcontainers

**Deliverables:**

- ✅ Database schema deployed
- ✅ All Pydantic models defined
- ✅ Basic memory storage and retrieval working
- ✅ Unit test coverage ≥90%

**Success Criteria:**

- Can store and retrieve memories
- PGVector similarity search functional
- Redis working memory cache operational
- All tests passing

---

### Phase 2: Core Memory Operations (Week 3) - P0

**Core Functionality:**

- RetrievalService (hybrid search: vector + metadata filtering)
- Basic importance scoring (relevance + recency + frequency)
- JSON-RPC methods (memory.store, memory.retrieve, memory.get_context)
- Context formatting service (token budget management)
- Integration with TaskManager and SessionManager

**Deliverables:**

- ✅ Semantic search with importance scoring
- ✅ JSON-RPC API complete
- ✅ Context formatting with token limits
- ✅ Cross-component integration

**Success Criteria:**

- Retrieval precision ≥85% (baseline, will improve to 95% in Phase 5)
- p95 latency <100ms
- Token budget respected (never exceed max_tokens)
- Integration tests with Task/Session managers passing

---

### Phase 3: COMPASS Stage Management (Weeks 4-5) - P1

**Week 4: Stage Detection & Transitions**

- StageManager (detect reasoning stages from action patterns)
- Stage transition logic (planning → execution → reflection → verification)
- Stage completion triggers
- Stage-aware memory organization
- Unit tests for stage detection

**Week 5: Progressive Compression**

- CompressionService (progressive compression pipeline)
- Test-time scaling integration (gpt-4.1-mini for compression)
- Compression quality monitoring (fact retention validation)
- Cost tracking per compression operation
- Adaptive compression (adjust based on quality metrics)

**Deliverables:**

- ✅ Stage detection with ≥90% accuracy
- ✅ Progressive compression (10:1 stage, 5:1 task)
- ✅ Compression quality metrics tracked
- ✅ Cost tracking dashboard

**Success Criteria:**

- Stage detection accuracy ≥90%
- Compression ratio 10:1 for stages, 5:1 for tasks
- Fact retention rate ≥95%
- Cost per compression <$0.01

---

### Phase 4: COMPASS Error Tracking (Week 6) - P1

**Error System:**

- ErrorTracker (record errors with context)
- Error pattern detection (frequency analysis, sequence detection)
- Error-aware retrieval (boost error-correction memories)
- ACE integration signals (alert on high error rates)
- Error history queries

**Deliverables:**

- ✅ Error recording system operational
- ✅ Pattern detection with ≥80% accuracy
- ✅ Error-aware retrieval integrated
- ✅ ACE interface defined

**Success Criteria:**

- 100% error capture rate
- Pattern detection accuracy ≥80%
- Error-aware retrieval improves precision by ≥5%
- ACE receives signals for error rates >30%

---

### Phase 5: Enhanced Retrieval (Week 7) - P1

**Advanced Retrieval:**

- Multi-factor importance scoring (all 6 factors)
- Critical memory identification (automatic + manual)
- Stage-aware retrieval optimization
- Retrieval quality validation
- A/B testing enhanced vs baseline

**Deliverables:**

- ✅ Full multi-factor scoring implemented
- ✅ Critical memory marking automatic
- ✅ Stage-aware retrieval operational
- ✅ Retrieval precision ≥95%

**Success Criteria:**

- Retrieval precision improved from 85% to 95%
- Critical evidence never missed (100% coverage)
- Contradiction rate <5%
- Stage-aware retrieval outperforms flat by ≥10%

---

### Phase 6: Integration & Optimization (Weeks 8-9)

**Week 8: Cross-Component Integration**

- MessageRouter memory-aware routing integration
- ACE strategic context query implementation
- Intervention outcome recording
- End-to-end COMPASS workflow testing
- Documentation updates

**Week 9: Performance Optimization & Validation**

- Database query optimization (index tuning)
- Caching strategy refinement
- Comprehensive benchmarking (10K, 100K, 1M vectors)
- Load testing (100+ concurrent operations)
- A/B testing vs baseline
- Deployment preparation

**Deliverables:**

- ✅ All integrations complete
- ✅ Performance targets validated
- ✅ Documentation complete
- ✅ Production-ready deployment

**Success Criteria:**

- All 8 acceptance criteria passing (from memory-system spec)
- Context efficiency: 80%+ reduction
- Long-horizon task improvement: 20%+
- Cost reduction: 70-80%
- Production deployment successful

---

### Timeline Summary

**Total Implementation:** 9 weeks (realistic with COMPASS enhancements)

**Dependencies:**

- A2A-009 (PostgreSQL Integration) must complete before Phase 1
- A2A-004 (Task Management) and A2A-002 (JSON-RPC Core) must complete before Phase 2
- ACE component optional for Phase 4 (can stub for testing)

**Parallel Work:**

- Unit tests developed alongside each phase
- Documentation updated continuously
- Performance benchmarks run at end of each phase

---

## ✅ Quality Assurance

### Testing Strategy

**Unit Testing (90%+ coverage required):**

- **Scope:** All service classes, scoring algorithms, compression logic
- **Mocking:** Mock external dependencies (OpenAI, Mem0, Redis, PostgreSQL)
- **Test Cases:**
  - Memory encoding and entity extraction
  - Importance scoring with all 6 factors
  - Stage detection from action patterns
  - Compression quality validation
  - Error pattern detection algorithms
  - Cache invalidation logic
- **Tools:** pytest, pytest-asyncio, pytest-mock
- **Execution:** <1 second per test, run on every commit

**Integration Testing:**

- **Scope:** End-to-end flows, cross-component integration
- **Infrastructure:** Real PostgreSQL with PGVector (testcontainers), Redis (testcontainers)
- **Test Cases:**
  - Memory persistence across service restart
  - Semantic search accuracy with real embeddings
  - Stage transitions and compression pipeline
  - Error tracking and pattern detection
  - Task/Session manager integration
  - ACE strategic context queries
- **Tools:** pytest, testcontainers
- **Execution:** <30 seconds total, run on pull requests

**Performance Testing:**

- **Scope:** Retrieval latency, compression throughput, cache effectiveness
- **Dataset Sizes:** 10K, 100K, 1M vectors
- **Metrics:**
  - Retrieval latency (p50, p95, p99)
  - Compression throughput (stages/hour)
  - Cache hit rates
  - Database query performance
- **Tools:** pytest-benchmark, locust
- **Execution:** Weekly on staging environment

**Load Testing:**

- **Scope:** Concurrent operations, sustained load
- **Scenarios:**
  - 100 concurrent retrieval requests
  - 1000 memory stores per hour
  - 10 agents with 50-turn conversations each
- **Metrics:**
  - Throughput (ops/sec)
  - Error rate under load
  - Resource usage (CPU, memory, connections)
  - Latency degradation
- **Tools:** Locust (existing in dependencies)
- **Execution:** Before each release

**Functional Testing:**

- **Scope:** User stories and acceptance criteria
- **Test Cases:**
  - Cross-session memory continuity
  - Agent preference learning
  - Context-aware routing
  - Memory cleanup and TTL
  - GDPR compliance (right to be forgotten)
- **Tools:** pytest, manual testing
- **Execution:** Before release

### Code Quality Gates

**Pre-Commit Hooks:**

- Ruff linting (0 errors required)
- Mypy type checking (strict mode, 100% coverage)
- pytest unit tests (90%+ coverage)
- Format check (ruff format)

**Pull Request Requirements:**

- All tests passing (unit + integration)
- Code coverage ≥90%
- No type errors (mypy strict)
- No linting errors (ruff)
- Peer review approval
- Documentation updated

**Release Gates:**

- All acceptance criteria passing
- Performance benchmarks validated
- Load tests passing
- Security scan clean (Bandit)
- Documentation complete
- Deployment runbook reviewed

### Deployment Verification Checklist

**Pre-Deployment:**

- [ ] All tests passing on staging
- [ ] Performance benchmarks validated
- [ ] Database migrations tested
- [ ] PGVector extension enabled
- [ ] Environment variables configured
- [ ] Secrets rotated
- [ ] Monitoring dashboards created
- [ ] Alerts configured
- [ ] Rollback plan documented

**Deployment:**

- [ ] Database migrations applied
- [ ] Memory service deployed
- [ ] Health checks passing
- [ ] Smoke tests passing
- [ ] Integration tests passing on production

**Post-Deployment:**

- [ ] Metrics being collected
- [ ] No error spikes
- [ ] Latency within targets
- [ ] Cost tracking operational
- [ ] Documentation published
- [ ] Team notified

### Monitoring and Alerting Setup

**Prometheus Metrics:**

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
memory_stores_total = Counter('memory_stores_total', 'Total memory stores', ['layer', 'agent_id'])
memory_retrievals_total = Counter('memory_retrievals_total', 'Total memory retrievals')
compression_operations_total = Counter('compression_operations_total', 'Total compressions', ['type'])
errors_total = Counter('errors_total', 'Total errors', ['error_type'])

# Histograms
retrieval_latency_seconds = Histogram('retrieval_latency_seconds', 'Retrieval latency')
compression_latency_seconds = Histogram('compression_latency_seconds', 'Compression latency')
compression_cost_usd = Histogram('compression_cost_usd', 'Compression cost')

# Gauges
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate')
compression_quality_score = Gauge('compression_quality_score', 'Compression quality', ['type'])
monthly_token_budget_used_pct = Gauge('monthly_token_budget_used_pct', 'Token budget usage %')
```

**Grafana Dashboards:**

- Memory Service Health (latency, throughput, errors)
- Compression Performance (quality, cost, throughput)
- Cost Tracking (embeddings, compression, total)
- Cache Effectiveness (hit rates, evictions)

**Alerts:**

- **Critical:** Availability <99%, p95 latency >200ms, error rate >5%, compression quality <90%
- **Warning:** p95 latency >100ms, cache hit rate <70%, cost budget >75%
- **Info:** Unusual access patterns, new error types detected

---

## 📚 References & Traceability

### Source Documentation

**Research Foundation:**

- `docs/research/evolving-memory-system.md`
  - Four-layer memory architecture concept
  - Memory operations: encode, store, retrieve, update, prune
  - Expected benefits: 25-30% multi-turn improvement, 5-10x context reduction
  - Vector database and compression strategy recommendations

**Formal Specifications:**

- `docs/specs/memory-service/spec.md`
  - Functional requirements (FR-1 to FR-7)
  - Non-functional requirements (performance, scalability, security)
  - Technology stack: Qdrant (modified to PGVector), Redis, PostgreSQL, Mem0
  - Acceptance criteria and validation approach

**COMPASS Enhancements:**

- `docs/specs/memory-system/spec.md`
  - Stage-aware context organization (FR-1: MEM-1)
  - Progressive compression with test-time scaling (FR-2: MEM-2)
  - Error tracking and pattern detection (FR-3: MEM-3)
  - Enhanced retrieval with criticality scoring (FR-4: MEM-4)
  - Enhanced targets: 95% precision, 80% context reduction, 20% accuracy improvement

### System Context

**Architecture Documentation:**

- `docs/agentcore-architecture-and-development-plan.md`
  - 6-layer architecture overview
  - Core Layer integration (Task Management, Communication Hub, Orchestration)
  - Existing infrastructure: PostgreSQL, Redis, FastAPI
  - Technology stack: Python 3.12+, SQLAlchemy async, Pydantic

**Technology Stack:**

- `pyproject.toml` - Current dependencies
  - PostgreSQL with pgvector>=0.4.1 ✅
  - Redis ✅
  - SentenceTransformers ✅
  - Need to add: mem0>=0.1.0

### Related Components

**Dependencies:**

- **A2A-009 (PostgreSQL Integration):** Required for database schema - `docs/specs/a2a-protocol/` (assumed complete)
- **A2A-004 (Task Management):** Required for task_id references - Integration with task artifacts
- **A2A-002 (JSON-RPC 2.0 Core):** Required for method registration - `@register_jsonrpc_method` pattern
- **A2A-019 to A2A-021 (Session Management):** Integration with session context retrieval

**Integrations:**

- **MessageRouter (A2A-005):** Memory-aware routing will use `memory.search()` to inform agent selection
- **ACE Integration (Future):** Meta-cognitive monitoring will query `memory.get_strategic_context()`
- **Agent Runtime (ART-014):** Agent state persistence will leverage memory service for context

### Architecture Decision Records

**ADR-001: PGVector instead of Qdrant**

- **Decision:** Use PGVector (existing infrastructure) instead of Qdrant for vector storage
- **Rationale:** Reduces operational complexity, sufficient performance for 1M vectors, already in dependencies
- **Trade-off:** Qdrant has better performance at 10M+ scale, but AgentCore targets 1M per agent
- **Fallback:** Can migrate to Qdrant if performance insufficient (compatible interface)

**ADR-002: Hybrid Embeddings (Local + Cloud)**

- **Decision:** Use local SentenceTransformers by default, OpenAI for critical memories only
- **Rationale:** Cost optimization (70-80% savings) while maintaining quality for important content
- **Trade-off:** Slight quality reduction for non-critical memories (768 dims vs 1536 dims)
- **Validation:** A/B test retrieval quality with hybrid vs full OpenAI

**ADR-003: Test-Time Scaling for Compression**

- **Decision:** Use gpt-4.1-mini for compression, gpt-4.1/gpt-5 for agent reasoning
- **Rationale:** COMPASS paper validated 70-80% cost reduction with maintained quality
- **Trade-off:** Slight latency increase for compression operations
- **Validation:** Track compression quality (≥95% fact retention target)

**ADR-004: Four-Layer Memory Architecture**

- **Decision:** Working (Redis) + Episodic (PGVector) + Semantic (PGVector) + Procedural (PGVector)
- **Rationale:** Research validated this structure for bounded context with unbounded memory
- **Trade-off:** Complexity vs effectiveness
- **Validation:** Measure context reduction (target: 80%+) and task performance (target: +20%)

---

## 🎯 Success Metrics Summary

**Context Engineering Standards - PRP Format:**

✅ **Comprehensive Context Assembly:** All six priority levels loaded

1. Specifications: memory-service/spec.md, memory-system/spec.md ✅
2. Research outputs: evolving-memory-system.md ✅
3. Feature requests: None (research-first approach) ✅
4. Code examples: Will establish new patterns (project baseline) ✅
5. System documentation: architecture-and-development-plan.md ✅
6. Other documentation: pyproject.toml (tech stack) ✅

✅ **Traceability Chain Complete:** Research → Spec → Enhancement → Plan documented
✅ **Research Integration:** All technology choices cite research (four-layer architecture, COMPASS enhancements)
✅ **Pattern Reuse:** Existing AgentCore patterns identified, new patterns will be established
✅ **System Alignment:** Integration with Core Layer (Task, Session, Router, ACE) documented
✅ **Concrete Metrics:** SLOs and KPIs defined with targets (95% precision, 80% reduction, 20% improvement)
✅ **Error Handling:** Edge cases from research addressed (compression quality, cost overrun, PGVector scale)
✅ **Dependencies Mapped:** Blocking (A2A-009), parallel (A2A-005), future (ART-014) dependencies clear
✅ **Risk Mitigation:** 8 critical risks identified with mitigation strategies
✅ **Timeline Realistic:** 9 weeks based on COMPASS enhancements (vs 3-4 week unrealistic spec)
✅ **Implementation Ready:** Sufficient detail for /sage.tasks breakdown
✅ **Epic Tickets Ready:** Architecture, tech stack, dependencies prepared for ticket updates

---

**Plan Generated:** 2025-10-25
**Next Steps:** Run `/sage.tasks memory-service` to generate SMART task breakdown from this plan
**Validation:** All PRP quality criteria met ✅
