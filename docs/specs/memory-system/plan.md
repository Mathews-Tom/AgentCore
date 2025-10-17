# Evolving Memory System Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-15
**Specification:** `docs/specs/memory-system/spec.md`
**Research:** `docs/research/evolving-memory-system.md`

---

## 📖 Context & Documentation

### Traceability Chain

**Research → Specification → This Plan**

1. **Research & Technical Analysis:** docs/research/evolving-memory-system.md
   - Four-layer memory architecture (Working, Episodic, Semantic, Procedural)
   - Memory operations (encode, store, retrieve, update, prune)
   - Vector embedding strategy with similarity search
   - Performance targets: 80% context reduction, 25-30% task improvement
   - Technology evaluation: PGVector, OpenAI embeddings, Redis caching

2. **Formal Specification:** docs/specs/memory-system/spec.md
   - 6 functional requirements (FR-1 through FR-6)
   - 6 non-functional requirements (NFR-1 through NFR-6)
   - 8 feature breakdowns with priorities
   - 8 acceptance criteria with quantifiable targets

### Related Documentation

**System Context:**

- Architecture: docs/agentcore-architecture-and-development-plan.md
- Project Guide: CLAUDE.md (AgentCore patterns and conventions)
- Tech Stack: Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy async

**Existing Code Patterns:**

- src/agentcore/a2a_protocol/ - Core protocol implementation
- src/agentcore/a2a_protocol/services/ - Service managers and JSON-RPC handlers
- src/agentcore/a2a_protocol/database/ - SQLAlchemy models and repositories
- src/agentcore/a2a_protocol/services/embedding_service.py - Existing embedding integration

**Cross-Component Dependencies:**

- Task Management: src/agentcore/a2a_protocol/services/task_manager.py (task_id references)
- Agent Manager: src/agentcore/a2a_protocol/services/agent_manager.py (agent_id scoping)
- JSON-RPC Handler: src/agentcore/a2a_protocol/services/jsonrpc_handler.py (method registration)

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable agents to maintain contextual awareness across multi-turn interactions by intelligently managing memory storage, retrieval, and compression.

**Value Proposition:**

- **80% Context Efficiency**: Reduce context tokens from 25K to 5K in 50-turn sessions
- **25-30% Performance Gain**: Improved success rate on multi-turn tasks through better context
- **Cost Reduction**: 5-10x reduction in LLM API costs through selective retrieval vs full history
- **Knowledge Accumulation**: Agents learn from past interactions and improve over time
- **Unbounded Conversations**: No practical limit on interaction depth or session length

**Target Users:**

- **Agent Developers**: Building complex multi-step workflows with persistent context
- **AgentCore Operators**: Managing agent systems with long-running sessions
- **End Users**: Benefiting from agents that remember past context without re-stating

### Technical Approach

**Architecture Pattern:** Service-oriented with layered memory architecture

- Four memory layers (Working, Episodic, Semantic, Procedural) with different retention policies
- Hybrid retrieval combining vector similarity, temporal recency, and access frequency
- Automatic compression and summarization to maintain bounded context windows

**Technology Stack:**

- **Vector Storage**: PostgreSQL with PGVector extension (no external vector DB)
- **Embedding Generation**: Existing embedding_service.py (OpenAI, Cohere, or local)
- **Caching**: Redis for working memory with TTL-based expiration
- **Archival**: S3-compatible storage for pruned memories (optional, Phase 4)

**Implementation Strategy:**

- Phase 1 (Weeks 1-2): Core memory system (models, manager, database)
- Phase 2 (Week 3): JSON-RPC integration
- Phase 3 (Week 4): Agent integration and caching
- Phase 4 (Weeks 5-6): Advanced features (compression, pruning, optimization)

### Key Success Metrics

**Service Level Objectives (SLOs):**

- Availability: 99.9% (memory operations non-blocking)
- Response Time: <100ms (p95 retrieval latency)
- Throughput: 100+ concurrent retrievals without degradation
- Error Rate: <0.1% (excluding embedding service failures)

**Key Performance Indicators (KPIs):**

- Context Efficiency: 80%+ reduction (retrieved_tokens / full_history_tokens)
- Retrieval Precision: 90%+ relevant memories in top-5
- Task Performance: +20% success rate on multi-turn benchmarks
- Memory Coherence: <5% contradictory retrievals
- Storage Efficiency: <50GB for 1M memories per agent (with compression)

---

## 💻 Code Examples & Patterns

### Repository Patterns (from AgentCore codebase)

**1. Service Manager Pattern:** `src/agentcore/a2a_protocol/services/task_manager.py`

**Application:** MemoryManager follows the same pattern as TaskManager for consistency.

**Pattern Structure:**

```python
from agentcore.a2a_protocol.database import get_session
from agentcore.memory.database.repositories import MemoryRepository

class MemoryManager:
    """Manages all memory operations (encode, store, retrieve, update, prune)."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        redis_client: Redis,
        capacity_limits: dict[str, int]
    ):
        self.embedding_service = embedding_service
        self.redis = redis_client
        self.capacity_limits = capacity_limits

    async def add_interaction(
        self,
        interaction: Interaction,
        agent_id: str
    ) -> None:
        """Process and store a new interaction as memory."""
        # Encode interaction into memory records
        episodic_record = await self._encode_episodic(interaction, agent_id)
        semantic_records = await self._extract_semantic_facts(interaction, agent_id)

        # Store in database
        async with get_session() as session:
            repo = MemoryRepository(session)
            await repo.create(episodic_record)
            for record in semantic_records:
                await repo.create(record)

        # Update working memory cache
        if interaction.task_id:
            await self._update_working_memory(interaction.task_id, episodic_record)

        # Prune if capacity exceeded
        await self._check_and_prune(agent_id)
```

**Key Takeaways:**

- Manager class handles business logic
- Uses get_session() context manager for database access
- Async-first design with async/await
- Clear separation of concerns (encoding, storage, caching, pruning)

**2. JSON-RPC Handler Pattern:** `src/agentcore/a2a_protocol/services/task_jsonrpc.py`

**Application:** Memory JSON-RPC handlers use decorator-based registration.

**Pattern Structure:**

```python
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest, JsonRpcErrorCode
from agentcore.memory.manager import MemoryManager
from agentcore.memory.models import Interaction

# Global instance (initialized in main.py)
memory_manager: MemoryManager | None = None

@register_jsonrpc_method("memory.store")
async def store_memory(request: JsonRpcRequest) -> dict[str, Any]:
    """Store an interaction as memory."""
    if not memory_manager:
        raise JsonRpcError(JsonRpcErrorCode.INTERNAL_ERROR, "MemoryManager not initialized")

    # Validate params with Pydantic
    interaction = Interaction(**request.params["interaction"])
    agent_id = request.a2a_context.source_agent if request.a2a_context else None

    if not agent_id:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "agent_id required in A2A context")

    # Delegate to manager
    await memory_manager.add_interaction(interaction, agent_id)

    return {"success": True}
```

**Key Takeaways:**

- Use `@register_jsonrpc_method` decorator
- Validate inputs with Pydantic models
- Extract agent_id from A2A context
- Return simple dict (auto-wrapped in JSON-RPC response)

**3. Database Model Pattern:** `src/agentcore/a2a_protocol/database/models.py`

**Application:** MemoryModel uses SQLAlchemy async ORM with PGVector.

**Pattern Structure:**

```python
from sqlalchemy import Column, String, Text, Integer, Float, TIMESTAMP, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from agentcore.a2a_protocol.database import Base

class MemoryModel(Base):
    """SQLAlchemy model for memory records."""

    __tablename__ = "memories"

    # Primary key
    memory_id = Column(UUID(as_uuid=True), primary_key=True)

    # Foreign keys and metadata
    agent_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    memory_type = Column(String(50), nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    interaction_id = Column(UUID(as_uuid=True), nullable=True)

    # Content
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # PGVector type

    # Timestamps
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default="NOW()")
    updated_at = Column(TIMESTAMP(timezone=True), server_default="NOW()", onupdate="NOW()")

    # Extracted entities and facts
    entities = Column(ARRAY(Text), default=[])
    facts = Column(ARRAY(Text), default=[])
    keywords = Column(ARRAY(Text), default=[])

    # Relevance tracking
    relevance_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    last_accessed = Column(TIMESTAMP(timezone=True), nullable=True)

    # Additional metadata
    metadata = Column(JSON, default={})
```

**Key Takeaways:**

- Use pgvector.sqlalchemy.Vector for embeddings
- Index frequently queried columns (agent_id, memory_type, task_id, timestamp)
- Use TIMESTAMP(timezone=True) for proper UTC handling
- ARRAY and JSON types for structured data

**4. Repository Pattern:** `src/agentcore/a2a_protocol/database/repositories.py`

**Application:** MemoryRepository implements data access layer.

**Pattern Structure:**

```python
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from agentcore.memory.database.models import MemoryModel

class MemoryRepository:
    """Data access layer for memories."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, memory: MemoryModel) -> MemoryModel:
        """Create a new memory record."""
        self.session.add(memory)
        await self.session.commit()
        await self.session.refresh(memory)
        return memory

    async def search_by_embedding(
        self,
        agent_id: str,
        query_embedding: list[float],
        memory_types: list[str],
        k: int = 5
    ) -> list[MemoryModel]:
        """Search memories by vector similarity."""
        stmt = (
            select(MemoryModel)
            .where(
                and_(
                    MemoryModel.agent_id == agent_id,
                    MemoryModel.memory_type.in_(memory_types)
                )
            )
            .order_by(MemoryModel.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
```

**Key Takeaways:**

- Repository encapsulates database queries
- Use SQLAlchemy select() for async queries
- PGVector provides `.cosine_distance()` method for similarity search
- Always filter by agent_id for security

### Implementation Reference Examples

**From Research (evolving-memory-system.md):**

**Importance Scoring Algorithm:**

```python
import math
from datetime import datetime

def calculate_importance_score(
    memory: MemoryRecord,
    query_embedding: list[float],
    current_time: datetime
) -> float:
    """Calculate importance score combining recency, frequency, and relevance."""

    # Recency: Exponential decay with 24-hour half-life
    age_hours = (current_time - memory.timestamp).total_seconds() / 3600
    recency_score = math.exp(-age_hours / 24)

    # Frequency: Normalized access count (cap at 10)
    frequency_score = min(memory.access_count / 10, 1.0)

    # Relevance: Cosine similarity from embedding
    relevance_score = 1 - cosine_distance(query_embedding, memory.embedding)

    # Weighted combination
    importance = (
        0.4 * relevance_score +
        0.3 * recency_score +
        0.3 * frequency_score
    )

    return importance
```

**Context Compression with Token Limits:**

```python
async def format_context(
    self,
    working: list[MemoryRecord],
    episodic: list[MemoryRecord],
    semantic: list[MemoryRecord],
    max_tokens: int
) -> str:
    """Format memories into context string with token budget."""
    sections = []

    # Current task context (highest priority)
    if working:
        sections.append("## Current Task Context")
        for record in working:
            sections.append(f"- {record.summary}")

    # Relevant past interactions
    if episodic:
        sections.append("\n## Relevant Past Interactions")
        for record in episodic:
            sections.append(f"- [{record.timestamp.isoformat()}] {record.summary}")

    # Relevant facts and knowledge
    if semantic:
        sections.append("\n## Relevant Knowledge")
        for record in semantic:
            sections.append(f"- {record.summary}")

    context = "\n".join(sections)

    # Truncate if exceeds limit
    tokens = count_tokens(context)  # Use tiktoken
    if tokens > max_tokens:
        context = await self._compress_with_llm(context, max_tokens)

    return context
```

### Anti-Patterns to Avoid

**From AgentCore Conventions (CLAUDE.md):**

- ❌ Do not use `typing.Any` without justification (use proper types)
- ❌ Do not use deprecated typing imports (`List`, `Dict`, `Optional`)
- ❌ Do not create synchronous database operations (must be async)
- ❌ Do not bypass Pydantic validation for API inputs
- ❌ Do not hardcode configuration (use config.py with env vars)
- ❌ Do not skip error handling for external services (embedding API)

---

## 🔧 Technology Stack

### Recommended Stack (Aligned with AgentCore)

**Based on research from:** `docs/research/evolving-memory-system.md` and existing AgentCore stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | AgentCore standard, modern typing (PEP 695) |
| Framework | FastAPI | Latest | Existing AgentCore framework |
| Database | PostgreSQL | 14+ | Existing AgentCore database |
| Vector Extension | PGVector | 0.5.0+ | Simpler than external vector DB, sufficient for scale |
| Embeddings | OpenAI API | text-embedding-3-small | Via existing embedding_service.py |
| Cache | Redis | 6+ | Existing AgentCore cache, TTL support |
| ORM | SQLAlchemy | 2.0+ (async) | Existing AgentCore pattern |
| Migrations | Alembic | Latest | Existing AgentCore migration tool |
| Validation | Pydantic | 2.0+ | Existing AgentCore pattern |
| Testing | pytest-asyncio | Latest | AgentCore standard, 90%+ coverage |
| Archival | boto3 (S3) | Latest | Industry standard for object storage |

**Key Technology Decisions:**

1. **PGVector over External Vector DB**
   - **Rationale:** Simpler deployment (single database), sufficient performance (<100ms at 1M scale), already using PostgreSQL
   - **Trade-off:** Slightly lower throughput vs Pinecone/Qdrant, but eliminates external dependency
   - **Research Support:** "Options: Pinecone, Weaviate, Qdrant, **PGVector**" (evolving-memory-system.md:516)

2. **Reuse Existing embedding_service.py**
   - **Rationale:** Already integrated with OpenAI/Cohere, tested, no duplication
   - **Trade-off:** None (pure win)
   - **Code Location:** src/agentcore/a2a_protocol/services/embedding_service.py

3. **Redis for Working Memory Cache**
   - **Rationale:** Already configured, sub-10ms latency, TTL support
   - **Pattern:** Key format `memory:working:{task_id}`, 1-hour TTL
   - **Research Support:** "Redis for working memory cache" (evolving-memory-system.md:528)

4. **S3 for Archival (Optional Phase 4)**
   - **Rationale:** Industry standard, 90% cost reduction vs database storage
   - **Trade-off:** Additional complexity, only needed if pruning frequently
   - **Research Support:** "S3 for memory archives" (evolving-memory-system.md:529)

### Alternatives Considered

**Option 2: Pinecone Vector Database**

- **Pros:** Higher throughput (1000+ QPS), managed service, purpose-built for vectors
- **Cons:** External dependency, additional cost, deployment complexity
- **Why Not Chosen:** PGVector sufficient for AgentCore scale (<1M vectors/agent), prefer single database

**Option 3: Local SentenceTransformers**

- **Pros:** No API costs, no rate limits, data privacy
- **Cons:** CPU/GPU resource usage, slower than API, model management complexity
- **Why Not Chosen:** AgentCore already integrated with OpenAI, prefer consistency

### Alignment with Existing System

**From AgentCore Stack (CLAUDE.md, src/):**

- ✅ **Consistent With:** PostgreSQL, Redis, FastAPI, Pydantic, SQLAlchemy async, Alembic
- ✅ **Extends:** embedding_service.py (already exists)
- ➕ **New Additions:** PGVector extension (PostgreSQL plugin), boto3 (S3 client, optional)
- 🔄 **Migration Considerations:** Alembic migration to add PGVector extension and memories table

---

## 🏗️ Architecture Design

### System Context (from AgentCore Architecture)

**Existing AgentCore Architecture:**

- **Layered Architecture**: 6 layers (Infrastructure, Runtime, Core, Operations, Experience, Intelligence)
- **Core Protocol**: JSON-RPC 2.0 with A2A context for distributed tracing
- **Data Flow**: FastAPI → JSON-RPC Handler → Service Managers → Repositories → PostgreSQL/Redis
- **Agent Runtime**: ReAct, CoT, Multi-Agent, Autonomous engines

**Integration Points:**

- **Task Manager**: Memory system retrieves context by task_id, stores interactions linked to tasks
- **Agent Manager**: Agent-scoped memories (agent_id filtering), agent discovery with memory capabilities
- **Embedding Service**: Generate embeddings for memory encoding (reuse existing service)
- **JSON-RPC Handler**: Register memory.* methods (store, retrieve, get_context, update, prune)

**New Architectural Patterns Introduced:**

- **Multi-Layer Memory**: Four distinct memory layers with different retention and retrieval policies
- **Hybrid Retrieval**: Combining vector similarity, temporal recency, and access frequency
- **Automatic Compression**: LLM-based summarization to fit token budgets
- **Importance Weighting**: Dynamic relevance scoring based on multiple factors

### Component Architecture

**Architecture Pattern:** Modular service within monolithic AgentCore deployment

- **Rationale:** Aligns with existing AgentCore modular monolith, shared database/cache, simpler deployment
- **Alignment:** Follows pattern of agent_manager, task_manager, event_manager

**System Design:**

```plaintext
┌──────────────────────────────────────────────────────────────────┐
│                        AgentCore System                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐         ┌───────────────────┐                   │
│  │   Agent     │ task_id │   Memory System   │                   │
│  │   Runtime   ├────────>│                   │                   │
│  │             │         │  ┌─────────────┐  │                   │
│  └─────────────┘         │  │  Memory     │  │                   │
│         │                │  │  Manager    │  │                   │
│         │ query          │  └──────┬──────┘  │                   │
│         v                │         │         │                   │
│  ┌─────────────┐         │  ┌──────v──────┐  │                   │
│  │   Task      │         │  │  Encoding   │  │                   │
│  │   Manager   │         │  │  Service    │  │                   │
│  └──────┬──────┘         │  └──────┬──────┘  │                   │
│         │                │         │         │                   │
│         │ store          │  ┌──────v──────┐  │                   │
│         └───────────────>│  │  Retrieval  │  │                   │
│                          │  │  Service    │  │                   │
│  ┌─────────────┐         │  └──────┬──────┘  │                   │
│  │  Embedding  │<────────│         │         │                   │
│  │  Service    │ embed   │  ┌──────v──────┐  │                   │
│  └─────────────┘         │  │ Compression │  │                   │
│                          │  │  Service    │  │                   │
│  ┌─────────────┐         │  └─────────────┘  │                   │
│  │  JSON-RPC   │         │                   │                   │
│  │  Handler    │<────────┤  JSON-RPC Methods │                   │
│  └──────┬──────┘         │  memory.*         │                   │
│         │                └───────────────────┘                   │
│         v                                                        │
│  ┌─────────────┐         ┌───────────────────┐                   │
│  │ PostgreSQL  │         │      Redis        │                   │
│  │ + PGVector  │         │  Working Memory   │                   │
│  └─────────────┘         └───────────────────┘                   │
│         │                                                        │
│         │ (optional)                                             │
│         v                                                        │
│  ┌─────────────┐                                                 │
│  │  S3 Archive │                                                 │
│  └─────────────┘                                                 │
└──────────────────────────────────────────────────────────────────┘
```

### Architecture Decisions

**Decision 1: JSON-RPC Integration Pattern**

- **Choice:** REST-like JSON-RPC methods (memory.store, memory.retrieve, memory.get_context)
- **Rationale:** Consistent with AgentCore's A2A protocol, supports distributed tracing via A2A context
- **Implementation:** Register methods with @register_jsonrpc_method decorator, import in main.py
- **Trade-offs:** More boilerplate vs raw REST, but gains A2A compliance and agent interoperability

**Decision 2: Hybrid Data Flow (Sync vs Async)**

- **Choice:** Async-first for all I/O (database, Redis, embedding API)
- **Rationale:** AgentCore convention, non-blocking, supports high concurrency
- **Implementation:** async/await throughout, AsyncSession for database, aioredis for cache
- **Trade-offs:** More complex error handling, but required for AgentCore integration

**Decision 3: Memory Layer State Management**

- **Choice:** Stateless MemoryManager with state in PostgreSQL + Redis
- **Rationale:** Horizontal scalability, crash recovery, aligns with AgentCore stateless services
- **Implementation:** Working memory in Redis (ephemeral), episodic/semantic/procedural in PostgreSQL (durable)
- **Trade-offs:** Network latency for retrieval, but gains durability and scalability

**Decision 4: Agent-Scoped Memory Isolation**

- **Choice:** Filter all queries by agent_id from A2A context
- **Rationale:** Security (agents can't access others' memories), multi-tenancy support
- **Implementation:** Repository methods require agent_id, extract from request.a2a_context.source_agent
- **Trade-offs:** No cross-agent knowledge sharing (acceptable for MVP, can add later)

### Component Breakdown

**Core Components:**

**1. MemoryManager (src/agentcore/memory/manager.py)**

- **Purpose:** Orchestrate all memory operations (encode, store, retrieve, update, prune)
- **Technology:** Python async class, uses EmbeddingService and MemoryRepository
- **Pattern:** Similar to TaskManager (src/agentcore/a2a_protocol/services/task_manager.py)
- **Interfaces:**
  - `add_interaction(interaction, agent_id)` → Store new interaction
  - `retrieve_memories(query, agent_id, k)` → Hybrid search
  - `get_relevant_context(query, task_id, agent_id, max_tokens)` → Formatted context
  - `update_memory(memory_id, updates, agent_id)` → Update existing
  - `prune_memories(agent_id, strategy)` → Capacity management
- **Dependencies:** EmbeddingService, Redis, MemoryRepository

**2. EncodingService (src/agentcore/memory/encoding.py)**

- **Purpose:** Encode interactions into structured memory records
- **Technology:** Async functions, uses LLM for summarization/entity extraction
- **Responsibilities:**
  - Generate summaries (condensed content)
  - Extract entities (named entities, key concepts)
  - Extract facts (semantic assertions)
  - Create embeddings (via EmbeddingService)
- **Interfaces:**
  - `encode_episodic(interaction, agent_id)` → MemoryRecord
  - `extract_semantic_facts(interaction, agent_id)` → list[MemoryRecord]
- **Dependencies:** EmbeddingService, LLM client (for summarization)

**3. RetrievalService (src/agentcore/memory/retrieval.py)**

- **Purpose:** Implement hybrid retrieval algorithms
- **Technology:** Async functions, uses MemoryRepository for vector search
- **Responsibilities:**
  - Vector similarity search (PGVector)
  - Importance scoring (recency + frequency + relevance)
  - Access tracking (update access_count, last_accessed)
- **Interfaces:**
  - `search_memories(query_embedding, agent_id, memory_types, k)` → list[MemoryRecord]
  - `calculate_importance(memory, query_embedding, current_time)` → float
- **Dependencies:** MemoryRepository

**4. CompressionService (src/agentcore/memory/compression.py)**

- **Purpose:** Compress memory sequences to fit token budgets
- **Technology:** Async functions, uses LLM for hierarchical summarization
- **Responsibilities:**
  - Token counting (tiktoken)
  - Hierarchical summarization
  - Importance-based pruning
- **Interfaces:**
  - `compress_context(memories, max_tokens)` → str
  - `count_tokens(text)` → int
- **Dependencies:** LLM client (for summarization), tiktoken

**5. MemoryRepository (src/agentcore/memory/database/repositories.py)**

- **Purpose:** Data access layer for PostgreSQL + PGVector
- **Technology:** SQLAlchemy async, uses MemoryModel
- **Pattern:** Similar to AgentRepository, TaskRepository
- **Interfaces:**
  - `create(memory)` → MemoryModel
  - `get_by_id(memory_id, agent_id)` → MemoryModel | None
  - `search_by_embedding(agent_id, embedding, memory_types, k)` → list[MemoryModel]
  - `update(memory_id, updates, agent_id)` → MemoryModel
  - `delete(memory_id, agent_id)` → None
  - `count_by_type(agent_id, memory_type)` → int
- **Dependencies:** AsyncSession, MemoryModel

**6. JSON-RPC Handler (src/agentcore/memory/jsonrpc.py)**

- **Purpose:** Expose memory operations via JSON-RPC protocol
- **Technology:** FastAPI route handlers, Pydantic validation
- **Pattern:** Similar to task_jsonrpc.py, agent_jsonrpc.py
- **Methods:**
  - `memory.store` - Store interaction as memory
  - `memory.retrieve` - Retrieve top-k memories
  - `memory.get_context` - Get formatted context string
  - `memory.update` - Update existing memory
  - `memory.prune` - Manually trigger pruning
- **Dependencies:** MemoryManager

### Data Flow & Boundaries

**Request Flow (memory.get_context):**

1. **Entry**: Client → POST /api/v1/jsonrpc with `{"method": "memory.get_context", "params": {"query": "...", "task_id": "...", "max_tokens": 2000}}`
2. **JSON-RPC Handler**: Validate request, extract agent_id from A2A context
3. **MemoryManager**: Call get_relevant_context()
   - Get working memory from Redis (by task_id)
   - Generate query embedding via EmbeddingService
   - Search episodic memories (RetrievalService)
   - Search semantic memories (RetrievalService)
   - Calculate importance scores
   - Format context (CompressionService if exceeds max_tokens)
4. **Response**: Return `{"result": {"context": "...", "token_count": 1800}}`

**Storage Flow (memory.store):**

1. **Entry**: Client → POST /api/v1/jsonrpc with `{"method": "memory.store", "params": {"interaction": {...}}}`
2. **JSON-RPC Handler**: Validate interaction, extract agent_id
3. **MemoryManager**: Call add_interaction()
   - Encode episodic memory (EncodingService):
     - Summarize interaction (LLM)
     - Extract entities/facts (LLM)
     - Generate embedding (EmbeddingService)
   - Store episodic in PostgreSQL (MemoryRepository)
   - Extract and store semantic facts (if applicable)
   - Update working memory in Redis (if task_id present)
   - Check capacity and prune if needed
4. **Response**: Return `{"result": {"success": true, "memory_id": "..."}}`

**Component Boundaries:**

**Public Interface (JSON-RPC):**

- memory.store, memory.retrieve, memory.get_context, memory.update, memory.prune
- Exposed via POST /api/v1/jsonrpc
- Validated with Pydantic models
- Requires JWT authentication (via A2A context)

**Internal Implementation:**

- MemoryManager (orchestration)
- EncodingService, RetrievalService, CompressionService (domain logic)
- MemoryRepository (data access)
- Redis client (caching)
- EmbeddingService (external API wrapper)

**Cross-Component Contracts:**

- **Task Manager** → Memory System: Provides task_id for working memory scoping
- **Agent Manager** → Memory System: Provides agent_id for memory isolation
- **Embedding Service** → Memory System: Generates embeddings for memory encoding

---

## 🔧 Technical Specification

### Data Model

**Pydantic Models (src/agentcore/memory/models.py):**

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class MemoryRecord(BaseModel):
    """Pydantic model for memory records (API contract)."""

    memory_id: UUID
    agent_id: UUID
    memory_type: str  # "working", "episodic", "semantic", "procedural"

    # Content
    content: str
    summary: str
    embedding: list[float]

    # Metadata
    timestamp: datetime
    interaction_id: UUID | None = None
    task_id: UUID | None = None

    # Extracted information
    entities: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    # Relevance tracking
    relevance_score: float = 1.0
    access_count: int = 0
    last_accessed: datetime | None = None

    # Procedural memory (action-outcome pairs)
    actions: list[str] = Field(default_factory=list)
    outcome: str | None = None
    success: bool | None = None

class Interaction(BaseModel):
    """Pydantic model for agent interactions (input to memory.store)."""

    id: UUID
    task_id: UUID | None = None
    query: str
    response: str | None = None
    actions: list[str] = Field(default_factory=list)
    outcome: str | None = None
    success: bool | None = None
    timestamp: datetime
```

**SQLAlchemy Model (src/agentcore/memory/database/models.py):**

```python
from sqlalchemy import Column, String, Text, Integer, Float, TIMESTAMP, ARRAY, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from agentcore.a2a_protocol.database import Base
import uuid

class MemoryModel(Base):
    """SQLAlchemy model for memories table."""

    __tablename__ = "memories"

    # Primary key
    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys and scoping
    agent_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    memory_type = Column(String(50), nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    interaction_id = Column(UUID(as_uuid=True), nullable=True)

    # Content
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # PGVector type for embeddings

    # Timestamps
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default="NOW()")
    updated_at = Column(TIMESTAMP(timezone=True), server_default="NOW()", onupdate="NOW()")

    # Extracted information
    entities = Column(ARRAY(Text), default=list)
    facts = Column(ARRAY(Text), default=list)
    keywords = Column(ARRAY(Text), default=list)

    # Relevance tracking
    relevance_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    last_accessed = Column(TIMESTAMP(timezone=True), nullable=True)

    # Additional metadata
    metadata = Column(JSON, default=dict)

    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_agent_type_timestamp', 'agent_id', 'memory_type', 'timestamp'),
        Index('idx_agent_task', 'agent_id', 'task_id'),
        Index('idx_embedding', 'embedding', postgresql_using='ivfflat', postgresql_with={'lists': 100}),
    )
```

**Validation Rules:**

- memory_type ∈ {"working", "episodic", "semantic", "procedural"}
- embedding dimension = 1536 (OpenAI text-embedding-3-small)
- agent_id must exist in agents table (foreign key constraint)
- task_id must exist in tasks table if not NULL (foreign key constraint)
- content and summary required (non-empty strings)
- relevance_score ∈ [0.0, 1.0]
- access_count ≥ 0

**Indexing Strategy:**

- Primary index on memory_id (UUID, clustered)
- Composite index on (agent_id, memory_type, timestamp) for filtered retrieval
- Composite index on (agent_id, task_id) for task-scoped queries
- IVFFlat index on embedding for vector similarity search (lists=100 for ~10K-100K vectors)
- Individual indexes on agent_id, memory_type, task_id, timestamp for flexibility

**Migration Approach:**

1. Create Alembic migration: `uv run alembic revision --autogenerate -m "add memories table with pgvector"`
2. In migration up():
   - `CREATE EXTENSION IF NOT EXISTS vector;`
   - Create memories table with PGVector column
   - Create indexes (including IVFFlat for embeddings)
   - Add foreign key constraints (agent_id → agents.id, task_id → tasks.id)
3. In migration down():
   - Drop memories table
   - Drop indexes
   - Optionally drop vector extension (if not used elsewhere)

### API Design

**Top 6 Critical JSON-RPC Methods:**

**1. memory.store**

- **Purpose:** Store an interaction as memory (creates episodic + semantic records)
- **Request Schema:**

  ```json
  {
    "method": "memory.store",
    "params": {
      "interaction": {
        "id": "uuid",
        "task_id": "uuid",
        "query": "user query text",
        "response": "agent response text",
        "actions": ["action1", "action2"],
        "outcome": "success description",
        "success": true,
        "timestamp": "2025-10-15T12:00:00Z"
      }
    },
    "a2a_context": {
      "source_agent": "agent-uuid",
      "trace_id": "trace-uuid"
    }
  }
  ```

- **Response Schema:**

  ```json
  {
    "result": {
      "success": true,
      "memory_id": "uuid",
      "semantic_memory_ids": ["uuid1", "uuid2"]
    }
  }
  ```

- **Error Codes:**
  - INVALID_PARAMS (-32602): Invalid interaction format, missing agent_id
  - INTERNAL_ERROR (-32603): Embedding service failure, database error

**2. memory.retrieve**

- **Purpose:** Retrieve top-k relevant memories using hybrid search
- **Request Schema:**

  ```json
  {
    "method": "memory.retrieve",
    "params": {
      "query": "search query text",
      "k": 5,
      "memory_types": ["episodic", "semantic"]
    },
    "a2a_context": {
      "source_agent": "agent-uuid"
    }
  }
  ```

- **Response Schema:**

  ```json
  {
    "result": {
      "memories": [
        {
          "memory_id": "uuid",
          "summary": "condensed content",
          "timestamp": "2025-10-15T12:00:00Z",
          "relevance_score": 0.95,
          "importance_score": 0.87
        }
      ]
    }
  }
  ```

- **Error Codes:**
  - INVALID_PARAMS (-32602): Invalid k value, unknown memory_type
  - INTERNAL_ERROR (-32603): Embedding service failure, database error

**3. memory.get_context**

- **Purpose:** Get formatted context string for LLM prompt augmentation
- **Request Schema:**

  ```json
  {
    "method": "memory.get_context",
    "params": {
      "query": "current user query",
      "task_id": "uuid",
      "max_tokens": 2000
    },
    "a2a_context": {
      "source_agent": "agent-uuid"
    }
  }
  ```

- **Response Schema:**

  ```json
  {
    "result": {
      "context": "## Current Task Context\n- summary1\n\n## Relevant Past Interactions\n- [2025-10-15] summary2",
      "token_count": 1850,
      "memory_count": 7
    }
  }
  ```

- **Error Codes:**
  - INVALID_PARAMS (-32602): Invalid max_tokens, missing agent_id
  - INTERNAL_ERROR (-32603): Embedding/compression failure

**4. memory.update**

- **Purpose:** Update existing memory record (corrections, additions)
- **Request Schema:**

  ```json
  {
    "method": "memory.update",
    "params": {
      "memory_id": "uuid",
      "updates": {
        "content": "corrected content",
        "summary": "updated summary",
        "relevance_score": 0.8
      }
    },
    "a2a_context": {
      "source_agent": "agent-uuid"
    }
  }
  ```

- **Response Schema:**

  ```json
  {
    "result": {
      "success": true,
      "updated_fields": ["content", "summary", "relevance_score"]
    }
  }
  ```

- **Error Codes:**
  - INVALID_PARAMS (-32602): Invalid memory_id, unauthorized agent_id
  - NOT_FOUND (-32001): Memory not found
  - INTERNAL_ERROR (-32603): Database error

**5. memory.prune**

- **Purpose:** Manually trigger pruning for capacity management
- **Request Schema:**

  ```json
  {
    "method": "memory.prune",
    "params": {
      "memory_type": "episodic",
      "strategy": "least_relevant"
    },
    "a2a_context": {
      "source_agent": "agent-uuid"
    }
  }
  ```

- **Response Schema:**

  ```json
  {
    "result": {
      "pruned_count": 15,
      "archived_count": 15,
      "remaining_count": 35
    }
  }
  ```

- **Error Codes:**
  - INVALID_PARAMS (-32602): Unknown memory_type or strategy
  - INTERNAL_ERROR (-32603): Database/S3 error

**6. memory.stats**

- **Purpose:** Get memory statistics for monitoring (bonus method)
- **Request Schema:**

  ```json
  {
    "method": "memory.stats",
    "params": {},
    "a2a_context": {
      "source_agent": "agent-uuid"
    }
  }
  ```

- **Response Schema:**

  ```json
  {
    "result": {
      "total_memories": 150,
      "by_type": {
        "working": 5,
        "episodic": 50,
        "semantic": 80,
        "procedural": 15
      },
      "avg_relevance": 0.85,
      "total_size_mb": 12.5
    }
  }
  ```

### Security (Aligned with AgentCore)

**Authentication/Authorization:**

- **Approach:** JWT authentication via A2A context (existing AgentCore pattern)
- **Implementation:**
  - Extract `agent_id` from `request.a2a_context.source_agent`
  - Validate JWT signature using existing security_service.py
  - Filter all queries by agent_id (row-level security)
- **Standards:** OAuth 2.0 bearer tokens, RS256 signing

**Agent Isolation:**

- **Strategy:** Every database query filters by agent_id
- **Pattern:**

  ```python
  async def get_memories(agent_id: UUID, ...) -> list[MemoryModel]:
      stmt = select(MemoryModel).where(MemoryModel.agent_id == agent_id)
      # Always filter by agent_id - no global queries
  ```

- **Enforcement:** Repository methods require agent_id, no default value

**Secrets Management:**

- **Strategy:** Environment variables for API keys, database credentials
- **Pattern:** Pydantic Settings in config.py (existing AgentCore pattern)
- **Configuration:**

  ```python
  # .env
  OPENAI_API_KEY=sk-...
  MEMORY_ARCHIVAL_S3_BUCKET=agentcore-memory-archive
  MEMORY_ARCHIVAL_S3_ACCESS_KEY=...
  MEMORY_ARCHIVAL_S3_SECRET_KEY=...
  ```

- **Rotation:** Use AWS IAM roles for S3 (no static keys), rotate OpenAI API keys quarterly

**Data Protection:**

- **Encryption in Transit:** TLS 1.3 for all HTTP/WebSocket connections (AgentCore standard)
- **Encryption at Rest:** PostgreSQL database-level encryption (transparent data encryption)
- **PII Handling:**
  - No automatic PII detection (responsibility of agent developers)
  - Provide opt-in PII redaction via configuration flag
  - Document GDPR compliance requirements in API docs

**Security Testing:**

- **Approach:** Integration tests for authorization, SQL injection, XSS
- **Tools:**
  - pytest-asyncio for auth tests
  - sqlmap for SQL injection testing (CI/CD)
  - bandit for SAST (Python security linter)
- **Test Coverage:**
  - Unauthorized access attempts (wrong agent_id)
  - Malicious input (SQL injection in query strings)
  - Token expiration handling

**Compliance:**

- **GDPR:** Right to erasure (memory.delete method), data portability (memory export)
- **SOC 2:** Audit logging (who accessed which memories), access controls (agent-scoped)

### Performance (from Research)

**Performance Targets (from Research: evolving-memory-system.md):**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Response Time | <100ms (p95) | Retrieval latency from memory.retrieve |
| Throughput | >100 req/s | Concurrent memory.get_context calls |
| Context Efficiency | 80%+ reduction | retrieved_tokens / full_history_tokens |
| Retrieval Precision | 90%+ | Relevant memories in top-5 (ground truth) |
| Storage Growth | <50GB/1M memories | Database size monitoring |

**Caching Strategy:**

**1. Working Memory Cache (Redis):**

- **Approach:** TTL-based cache for current task context
- **Pattern:**

  ```python
  # Key format: memory:working:{task_id}
  # Value: JSON-serialized list[MemoryRecord]
  # TTL: 3600 seconds (1 hour)

  async def get_working_memory(task_id: UUID) -> list[MemoryRecord]:
      key = f"memory:working:{task_id}"
      cached = await redis.get(key)
      if cached:
          return [MemoryRecord(**m) for m in json.loads(cached)]
      return []

  async def update_working_memory(task_id: UUID, memory: MemoryRecord) -> None:
      key = f"memory:working:{task_id}"
      memories = await get_working_memory(task_id)
      memories.append(memory)
      await redis.setex(key, 3600, json.dumps([m.model_dump() for m in memories]))
  ```

- **Invalidation:** Automatic expiration after 1 hour, manual clear on task completion

**2. Embedding Cache (Redis):**

- **Approach:** Cache query embeddings to avoid redundant API calls
- **Pattern:**

  ```python
  # Key format: embedding:sha256:{query_hash}
  # Value: JSON-serialized list[float]
  # TTL: 86400 seconds (24 hours)

  async def get_or_create_embedding(query: str) -> list[float]:
      query_hash = hashlib.sha256(query.encode()).hexdigest()
      key = f"embedding:sha256:{query_hash}"
      cached = await redis.get(key)
      if cached:
          return json.loads(cached)

      embedding = await embedding_service.embed(query)
      await redis.setex(key, 86400, json.dumps(embedding))
      return embedding
  ```

- **Invalidation:** 24-hour TTL, LRU eviction if Redis memory full

**Database Optimization:**

**1. Indexing Strategy:**

- **Primary Lookups:** Composite index on (agent_id, memory_type, timestamp)
- **Vector Search:** IVFFlat index on embedding column

  ```sql
  CREATE INDEX idx_embedding ON memories
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
  ```

- **IVFFlat Tuning:** lists=100 for 10K-100K vectors (adjust to sqrt(row_count) for larger datasets)

**2. Query Patterns:**

- **Prevent N+1:** Use JOIN for related data (agents, tasks)
- **Batching:** Batch insert for multiple semantic facts
- **Pagination:** Limit results with LIMIT/OFFSET, use cursor-based for large result sets

**3. Connection Pooling:**

- **Configuration:** SQLAlchemy async pool with 10-20 connections
- **Settings:**

  ```python
  engine = create_async_engine(
      DATABASE_URL,
      pool_size=10,
      max_overflow=20,
      pool_pre_ping=True
  )
  ```

**4. Partitioning (Future):**

- **Strategy:** Range partitioning by timestamp for archival queries
- **Implementation:** Create monthly partitions, archive old partitions to S3

**Scaling Strategy:**

**1. Horizontal Scaling:**

- **Load Balancing:** Multiple AgentCore instances behind ALB/NGINX
- **Stateless Design:** MemoryManager has no local state (all in PostgreSQL + Redis)
- **Shared Infrastructure:** All instances share same PostgreSQL + Redis cluster

**2. Vertical Scaling:**

- **Resource Limits:** PostgreSQL: 16GB RAM, 4 vCPU; Redis: 8GB RAM
- **Auto-scaling:** Kubernetes HPA based on CPU (target 70%) and memory (target 80%)

**3. Database Read Replicas:**

- **Strategy:** Route read queries (memory.retrieve, memory.get_context) to replicas
- **Replication:** PostgreSQL streaming replication with <100ms lag

**4. Performance Monitoring:**

- **Tools:** Prometheus + Grafana for metrics, Jaeger for distributed tracing
- **Metrics:**
  - `memory_retrieval_latency_seconds` (histogram, p50/p95/p99)
  - `memory_cache_hit_rate` (gauge, Redis hit ratio)
  - `memory_database_query_duration_seconds` (histogram, by query type)
  - `memory_embedding_api_latency_seconds` (histogram, external API)
- **Alerts:**
  - p95 latency > 100ms for 5 minutes
  - Cache hit rate < 50% for 10 minutes
  - Embedding API error rate > 5% for 5 minutes

---

## 5. Development Setup

### Required Tools and Versions

**Core Tools:**

- Python 3.12+ (modern typing support)
- uv package manager (AgentCore standard)
- PostgreSQL 14+ with PGVector extension
- Redis 6+ (AgentCore standard)
- Docker + Docker Compose (local development)

**Python Packages (add to pyproject.toml):**

```toml
[project.dependencies]
# Existing AgentCore dependencies
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.5.0"
sqlalchemy = "^2.0.23"
asyncpg = "^0.29.0"
alembic = "^1.12.1"
redis = "^5.0.1"

# New dependencies for memory system
pgvector = "^0.2.3"  # PGVector Python bindings
tiktoken = "^0.5.1"  # Token counting for compression
boto3 = "^1.29.0"    # S3 client for archival (optional)
```

**Development Tools:**

```bash
uv add --dev pytest-asyncio ruff mypy bandit
```

### Local Environment Setup

**1. Docker Compose (docker-compose.dev.yml):**

Extend existing AgentCore docker-compose.dev.yml:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: agentcore
      POSTGRES_PASSWORD: dev_password
      POSTGRES_DB: agentcore_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_pgvector.sql:/docker-entrypoint-initdb.d/01_init_pgvector.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
```

**2. PGVector Initialization (scripts/init_pgvector.sql):**

```sql
-- Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

**3. Environment Variables (.env.local):**

```bash
# Database
DATABASE_URL=postgresql+asyncpg://agentcore:dev_password@localhost:5432/agentcore_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# Embedding Service
OPENAI_API_KEY=sk-your-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Memory System Configuration
MEMORY_WORKING_TTL=3600
MEMORY_EPISODIC_CAPACITY=50
MEMORY_SEMANTIC_CAPACITY=1000
MEMORY_PRUNING_STRATEGY=least_relevant
MEMORY_PRUNING_ENABLED=true

# Archival (optional)
MEMORY_ARCHIVAL_ENABLED=false
MEMORY_ARCHIVAL_S3_BUCKET=agentcore-memory-archive
MEMORY_ARCHIVAL_S3_REGION=us-east-1
```

**4. Database Migration:**

```bash
# Start PostgreSQL + Redis
docker compose -f docker-compose.dev.yml up -d postgres redis

# Create migration
uv run alembic revision --autogenerate -m "add memories table with pgvector"

# Apply migration
uv run alembic upgrade head

# Verify PGVector extension
uv run python -c "
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://agentcore:dev_password@localhost:5432/agentcore_dev')
with engine.connect() as conn:
    result = conn.execute(text('SELECT * FROM pg_extension WHERE extname = \\\'vector\\\';'))
    print(list(result))
"
```

**5. Run Development Server:**

```bash
# Install dependencies
uv sync

# Start AgentCore with memory system
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# Test memory endpoints
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "memory.stats",
    "params": {},
    "id": 1,
    "a2a_context": {"source_agent": "test-agent"}
  }'
```

### CI/CD Pipeline Requirements

**GitHub Actions Workflow (.github/workflows/memory-system.yml):**

```yaml
name: Memory System CI

on:
  pull_request:
    paths:
      - 'src/agentcore/memory/**'
      - 'tests/memory/**'
      - 'alembic/versions/*_add_memories*'

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: agentcore
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: agentcore_test
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v1

      - name: Install dependencies
        run: uv sync

      - name: Run migrations
        run: uv run alembic upgrade head
        env:
          DATABASE_URL: postgresql+asyncpg://agentcore:test_password@localhost:5432/agentcore_test

      - name: Run tests
        run: uv run pytest tests/memory/ --cov=src/agentcore/memory --cov-report=xml --cov-fail-under=90
        env:
          DATABASE_URL: postgresql+asyncpg://agentcore:test_password@localhost:5432/agentcore_test
          REDIS_URL: redis://localhost:6379/0
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

### Testing Framework and Coverage Targets

**Test Structure:**

```
tests/
├── memory/
│   ├── __init__.py
│   ├── conftest.py               # Fixtures (memory_manager, test_db, test_redis)
│   ├── unit/
│   │   ├── test_encoding.py      # EncodingService unit tests
│   │   ├── test_retrieval.py     # RetrievalService unit tests
│   │   ├── test_compression.py   # CompressionService unit tests
│   │   └── test_repository.py    # MemoryRepository unit tests
│   ├── integration/
│   │   ├── test_manager.py       # MemoryManager integration tests
│   │   ├── test_jsonrpc.py       # JSON-RPC method integration tests
│   │   └── test_workflow.py      # End-to-end workflow tests
│   └── load/
│       ├── test_performance.py   # Load tests (100 concurrent users)
│       └── test_scaling.py       # Scaling tests (1M memories)
```

**Coverage Targets (AgentCore Standard):**

- Overall: 90%+ line coverage
- Critical paths: 100% (retrieval, storage, agent isolation)
- Unit tests: 95%+ (pure functions, business logic)
- Integration tests: 85%+ (database, Redis, external APIs)

**Test Fixtures (tests/memory/conftest.py):**

```python
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from agentcore.memory.manager import MemoryManager
from agentcore.memory.database.repositories import MemoryRepository
from agentcore.a2a_protocol.services.embedding_service import EmbeddingService
import redis.asyncio as aioredis

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """Create test database with PGVector."""
    engine = create_async_engine(
        "postgresql+asyncpg://agentcore:test_password@localhost:5432/agentcore_test",
        echo=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def test_redis():
    """Create test Redis client."""
    redis = await aioredis.from_url("redis://localhost:6379/1")  # Use DB 1 for tests
    await redis.flushdb()  # Clear test database
    yield redis
    await redis.flushdb()
    await redis.close()

@pytest.fixture
async def memory_manager(test_db, test_redis):
    """Create MemoryManager with test dependencies."""
    embedding_service = EmbeddingService()  # Mock or use test API key
    capacity_limits = {
        "working": 10,
        "episodic": 50,
        "semantic": 100,
        "procedural": 20
    }
    manager = MemoryManager(embedding_service, test_redis, capacity_limits)
    yield manager
```

**Example Unit Test (tests/memory/unit/test_retrieval.py):**

```python
import pytest
from datetime import datetime, timedelta
from agentcore.memory.retrieval import calculate_importance_score
from agentcore.memory.models import MemoryRecord
from uuid import uuid4

@pytest.mark.asyncio
async def test_importance_score_recency():
    """Test that recent memories score higher."""
    now = datetime.now(UTC)
    query_embedding = [0.1] * 1536

    # Recent memory (1 hour ago)
    recent_memory = MemoryRecord(
        memory_id=uuid4(),
        agent_id=uuid4(),
        memory_type="episodic",
        content="test",
        summary="test",
        embedding=[0.15] * 1536,
        timestamp=now - timedelta(hours=1),
        relevance_score=0.9,
        access_count=5
    )

    # Old memory (48 hours ago)
    old_memory = MemoryRecord(
        memory_id=uuid4(),
        agent_id=uuid4(),
        memory_type="episodic",
        content="test",
        summary="test",
        embedding=[0.15] * 1536,
        timestamp=now - timedelta(hours=48),
        relevance_score=0.9,
        access_count=5
    )

    recent_score = calculate_importance_score(recent_memory, query_embedding, now)
    old_score = calculate_importance_score(old_memory, query_embedding, now)

    assert recent_score > old_score, "Recent memory should score higher"
```

**Example Integration Test (tests/memory/integration/test_manager.py):**

```python
import pytest
from uuid import uuid4
from datetime import datetime
from agentcore.memory.models import Interaction

@pytest.mark.asyncio
async def test_add_and_retrieve_interaction(memory_manager):
    """Test storing and retrieving an interaction."""
    agent_id = uuid4()
    task_id = uuid4()

    # Create interaction
    interaction = Interaction(
        id=uuid4(),
        task_id=task_id,
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        actions=["search", "answer"],
        outcome="Answered correctly",
        success=True,
        timestamp=datetime.now(UTC)
    )

    # Store interaction
    await memory_manager.add_interaction(interaction, agent_id)

    # Retrieve memories
    memories = await memory_manager.retrieve_memories(
        query="capital of France",
        agent_id=agent_id,
        memory_types=["episodic"],
        k=5
    )

    # Assertions
    assert len(memories) >= 1, "Should retrieve at least one memory"
    assert any("Paris" in m.summary for m in memories), "Should mention Paris"
    assert memories[0].agent_id == agent_id, "Should be scoped to agent"
```

---

## 6. Risk Management

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| **Memory Coherence**: Contradictory memories retrieved together | HIGH | MEDIUM | Implement conflict detection algorithm, timestamp-based recency bias, confidence scoring. Test with synthetic contradiction scenarios. | Retrieval Team |
| **Retrieval Accuracy**: Relevant memories not found (low precision) | HIGH | MEDIUM | Hybrid search (embedding + metadata + temporal), tunable k parameter, A/B test different algorithms. Ground truth evaluation dataset. | Retrieval Team |
| **Storage Scaling**: Database grows unbounded, queries slow down | MEDIUM | HIGH | Automatic pruning with configurable capacity limits, archival to S3, PGVector IVFFlat indexes. Load test at 100K, 1M scale. | Database Team |
| **PGVector Performance**: Vector search latency exceeds 100ms at scale | MEDIUM | MEDIUM | Tune IVFFlat index (lists parameter), use read replicas for queries, implement connection pooling. Benchmark p95 latency. | Database Team |
| **Embedding API Failures**: OpenAI/Cohere API unavailable or rate-limited | MEDIUM | LOW | Exponential backoff retry (3 attempts), circuit breaker pattern, graceful degradation (skip encoding, use keyword search). Chaos engineering tests. | Integration Team |
| **Migration Complexity**: PGVector extension installation fails in production | LOW | LOW | Test migration on staging environment first, document manual installation steps, create rollback plan. Include in runbook. | DevOps Team |
| **Memory Leaks**: Working memory cache grows unbounded in Redis | MEDIUM | MEDIUM | Enforce TTL on all Redis keys (default 1 hour), monitor Redis memory usage, set maxmemory policy to LRU eviction. | Cache Team |
| **Agent Isolation Breach**: Agent accesses another agent's memories | HIGH | LOW | Repository methods always filter by agent_id, integration tests for unauthorized access, SQL injection tests with sqlmap. Security audit. | Security Team |
| **Token Budget Overrun**: Compression fails, context exceeds max_tokens | LOW | MEDIUM | Implement hard truncation fallback, warn users when compression needed, add compression quality metrics. Test with large histories. | Compression Team |
| **Dependency Version Conflicts**: PGVector, tiktoken, boto3 conflict with existing packages | LOW | LOW | Pin versions in pyproject.toml, test in isolated venv, use uv's dependency resolver. CI tests catch conflicts early. | DevOps Team |

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Week 1: Core Infrastructure**

**Goals:**

- Database schema with PGVector extension
- Basic Pydantic and SQLAlchemy models
- Repository layer with vector search
- Development environment setup

**Tasks:**

1. Create Alembic migration for memories table
   - Add PGVector extension
   - Define table schema with vector column
   - Create indexes (composite, IVFFlat)
   - Foreign key constraints (agent_id, task_id)
2. Implement Pydantic models (MemoryRecord, Interaction)
3. Implement SQLAlchemy model (MemoryModel)
4. Implement MemoryRepository with CRUD operations
   - create(), get_by_id(), search_by_embedding()
   - update(), delete(), count_by_type()
5. Set up Docker Compose with PGVector
6. Write unit tests for repository (90%+ coverage)

**Deliverables:**

- `src/agentcore/memory/models.py` (Pydantic models)
- `src/agentcore/memory/database/models.py` (SQLAlchemy model)
- `src/agentcore/memory/database/repositories.py` (Repository)
- `alembic/versions/XXX_add_memories_table.py` (Migration)
- `tests/memory/unit/test_repository.py` (Unit tests)
- `docker-compose.dev.yml` (Updated with PGVector)

**Week 2: Core Services**

**Goals:**

- MemoryManager orchestration
- EncodingService for interaction encoding
- RetrievalService with hybrid search
- Integration with existing EmbeddingService

**Tasks:**

1. Implement MemoryManager class
   - add_interaction(), retrieve_memories()
   - get_relevant_context(), update_memory(), prune_memories()
2. Implement EncodingService
   - encode_episodic(), extract_semantic_facts()
   - Integration with LLM for summarization/entity extraction
3. Implement RetrievalService
   - Hybrid search algorithm (vector + metadata + temporal)
   - calculate_importance_score()
   - Access tracking (update access_count, last_accessed)
4. Write integration tests for MemoryManager
5. Write unit tests for encoding and retrieval services

**Deliverables:**

- `src/agentcore/memory/manager.py` (MemoryManager)
- `src/agentcore/memory/encoding.py` (EncodingService)
- `src/agentcore/memory/retrieval.py` (RetrievalService)
- `tests/memory/integration/test_manager.py` (Integration tests)
- `tests/memory/unit/test_encoding.py` (Unit tests)
- `tests/memory/unit/test_retrieval.py` (Unit tests)

### Phase 2: JSON-RPC Integration (Week 3)

**Goals:**

- JSON-RPC method handlers for all memory operations
- Integration with existing JsonRpcProcessor
- A2A context handling (agent_id extraction, JWT validation)
- End-to-end workflow tests

**Tasks:**

1. Implement JSON-RPC handlers (memory/jsonrpc.py)
   - memory.store, memory.retrieve, memory.get_context
   - memory.update, memory.prune, memory.stats
2. Register methods with @register_jsonrpc_method decorator
3. Import memory.jsonrpc in main.py for auto-registration
4. Implement agent_id extraction from A2A context
5. Add JWT authentication middleware (reuse existing security_service)
6. Write integration tests for JSON-RPC methods
7. Write end-to-end workflow tests (store → retrieve → get_context)

**Deliverables:**

- `src/agentcore/memory/jsonrpc.py` (JSON-RPC handlers)
- `src/agentcore/a2a_protocol/main.py` (Updated with import)
- `tests/memory/integration/test_jsonrpc.py` (Integration tests)
- `tests/memory/integration/test_workflow.py` (E2E tests)

### Phase 3: Agent Integration & Caching (Week 4)

**Goals:**

- Working memory cache in Redis
- MemoryEnabledAgent base class
- Integration with existing AgentRuntime
- Performance optimization

**Tasks:**

1. Implement Redis working memory cache
   - Key format: `memory:working:{task_id}`
   - TTL: 3600 seconds
   - JSON serialization of MemoryRecord list
2. Implement embedding cache in Redis
   - Key format: `embedding:sha256:{query_hash}`
   - TTL: 86400 seconds
3. Create MemoryEnabledAgent base class (optional)
   - Automatic context retrieval in process_query()
   - Automatic interaction storage after completion
4. Add configuration management (config.py)
   - MEMORY_WORKING_TTL, MEMORY_EPISODIC_CAPACITY, etc.
5. Write cache integration tests
6. Performance benchmarking (latency, throughput)

**Deliverables:**

- `src/agentcore/memory/cache.py` (Redis cache logic)
- `src/agentcore/memory/agent.py` (MemoryEnabledAgent base class)
- `src/agentcore/a2a_protocol/config.py` (Updated with memory settings)
- `tests/memory/integration/test_cache.py` (Cache tests)
- `tests/memory/load/test_performance.py` (Load tests)

### Phase 4: Advanced Features (Weeks 5-6)

**Week 5: Compression & Pruning**

**Goals:**

- Context compression with token budgets
- Automatic pruning with multiple strategies
- S3 archival for pruned memories

**Tasks:**

1. Implement CompressionService
   - compress_context() with LLM summarization
   - count_tokens() using tiktoken
   - Importance-based memory selection
2. Implement pruning strategies
   - least_relevant, oldest_first, frequency_based
   - Automatic pruning when capacity exceeded
3. Implement S3 archival (optional)
   - Upload pruned memories to S3
   - Metadata tracking in archive_metadata table
   - Download from archive for historical queries
4. Write compression unit tests
5. Write pruning integration tests
6. Write S3 integration tests (mocked)

**Deliverables:**

- `src/agentcore/memory/compression.py` (CompressionService)
- `src/agentcore/memory/pruning.py` (Pruning strategies)
- `src/agentcore/memory/archival.py` (S3 archival logic)
- `tests/memory/unit/test_compression.py` (Unit tests)
- `tests/memory/integration/test_pruning.py` (Integration tests)
- `tests/memory/integration/test_archival.py` (S3 tests)

**Week 6: Monitoring & Production Readiness**

**Goals:**

- Prometheus metrics for monitoring
- Performance optimization (indexing, query tuning)
- Load testing at scale (1M memories)
- Documentation and runbook

**Tasks:**

1. Add Prometheus metrics
   - memory_retrieval_latency_seconds (histogram)
   - memory_cache_hit_rate (gauge)
   - memory_database_query_duration_seconds (histogram)
   - memory_embedding_api_latency_seconds (histogram)
2. Tune PGVector IVFFlat index (lists parameter)
3. Optimize slow queries (EXPLAIN ANALYZE)
4. Load test with 1M memories (Locust)
   - Validate p95 latency <100ms
   - Validate throughput >100 req/s
5. Write operational runbook
   - Deployment checklist
   - Monitoring and alerting setup
   - Troubleshooting guide
6. API documentation (OpenAPI/Swagger)

**Deliverables:**

- `src/agentcore/memory/metrics.py` (Prometheus metrics)
- `tests/memory/load/test_scaling.py` (1M record load test)
- `docs/memory-system-runbook.md` (Operational guide)
- `docs/memory-system-api.md` (API documentation)
- Performance tuning report

### Timeline Summary

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| Phase 1 | Weeks 1-2 | Foundation | Database schema, models, repository, core services |
| Phase 2 | Week 3 | JSON-RPC | API methods, A2A integration, E2E tests |
| Phase 3 | Week 4 | Caching | Redis cache, agent integration, performance tuning |
| Phase 4 | Weeks 5-6 | Advanced | Compression, pruning, monitoring, production readiness |

**Total Duration:** 6 weeks (42 days)

**Critical Path:**
Phase 1 (database) → Phase 2 (API) → Phase 3 (caching) → Phase 4 (optimization)

**Parallel Work:**

- Documentation can start in Week 3
- Load testing can start in Week 4 (with smaller datasets)
- Monitoring setup can start in Week 5

---

## 8. Quality Assurance

### Testing Strategy

**Unit Tests (95%+ coverage):**

- **Scope:** Pure functions, business logic, algorithms
- **Tools:** pytest, pytest-asyncio, pytest-cov
- **Targets:**
  - encoding.py: Summarization, entity extraction (mocked LLM)
  - retrieval.py: Importance scoring, hybrid search logic
  - compression.py: Token counting, compression algorithms
  - models.py: Pydantic validation rules

**Integration Tests (85%+ coverage):**

- **Scope:** Database interactions, Redis cache, external APIs
- **Tools:** pytest-asyncio, testcontainers, Docker Compose
- **Targets:**
  - MemoryRepository: CRUD operations, vector search
  - MemoryManager: End-to-end workflows (store → retrieve)
  - JSON-RPC handlers: API contract validation
  - Cache: Redis working memory, embedding cache

**End-to-End Tests:**

- **Scope:** Full user workflows across multiple components
- **Tools:** pytest-asyncio, httpx (async HTTP client)
- **Scenarios:**
  - Agent stores interaction → retrieves context → makes decision
  - Multi-turn conversation with working memory
  - Pruning triggers archival → retrieval from archive
  - Agent isolation (cross-agent access denied)

**Load Tests:**

- **Scope:** Performance under sustained load
- **Tools:** Locust (HTTP load testing), pytest-benchmark
- **Targets:**
  - 100 concurrent users calling memory.get_context
  - 1M memories in database (retrieval latency)
  - 1000 req/s throughput test

**Security Tests:**

- **Scope:** Authorization, injection attacks, data leakage
- **Tools:** pytest, sqlmap, bandit (SAST)
- **Scenarios:**
  - Unauthorized agent_id access (expect 403)
  - SQL injection in query strings (expect sanitization)
  - XSS in memory content (expect escaping)
  - JWT expiration handling

### Code Quality Gates

**Pre-commit Checks (Git Hooks):**

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        additional_dependencies: [types-redis, sqlalchemy[mypy]]
```

**CI/CD Quality Gates:**

1. **Linting:** Ruff (no errors)
2. **Type Checking:** mypy --strict (no errors)
3. **Test Coverage:** pytest --cov-fail-under=90
4. **Security Scan:** bandit -r src/agentcore/memory (no high-severity issues)
5. **Integration Tests:** All pass
6. **Load Tests:** p95 latency <100ms

**Code Review Checklist:**

- [ ] All tests passing (unit + integration + E2E)
- [ ] Coverage ≥90% for new code
- [ ] Type hints for all public functions
- [ ] Pydantic validation for all API inputs
- [ ] Agent isolation enforced (agent_id filtering)
- [ ] Error handling for external services (embedding API)
- [ ] Logging with structured context (agent_id, task_id, trace_id)
- [ ] Documentation updated (docstrings, API docs)

### Deployment Verification Checklist

**Pre-Deployment:**

- [ ] Alembic migration tested on staging database
- [ ] PGVector extension installed and verified
- [ ] Configuration secrets deployed (OPENAI_API_KEY, DATABASE_URL)
- [ ] Load balancer health checks configured
- [ ] Monitoring dashboards created (Grafana)
- [ ] Alerting rules deployed (Prometheus)

**Deployment:**

- [ ] Run database migrations (alembic upgrade head)
- [ ] Deploy AgentCore with memory module
- [ ] Verify /api/v1/jsonrpc endpoint responds
- [ ] Smoke test: memory.stats returns valid response
- [ ] Check Prometheus metrics export
- [ ] Verify no errors in application logs

**Post-Deployment:**

- [ ] Run integration test suite against production
- [ ] Monitor p95 latency for 1 hour (expect <100ms)
- [ ] Monitor error rate for 1 hour (expect <0.1%)
- [ ] Monitor cache hit rate (expect >50% after warm-up)
- [ ] Verify agent isolation (test cross-agent access denial)
- [ ] Performance regression test (compare with baseline)

### Monitoring and Alerting Setup

**Prometheus Metrics (src/agentcore/memory/metrics.py):**

```python
from prometheus_client import Histogram, Gauge, Counter

# Latency metrics
memory_retrieval_latency = Histogram(
    'memory_retrieval_latency_seconds',
    'Time to retrieve memories',
    ['agent_id', 'memory_type']
)

memory_encoding_latency = Histogram(
    'memory_encoding_latency_seconds',
    'Time to encode interaction',
    ['agent_id']
)

# Cache metrics
memory_cache_hit_rate = Gauge(
    'memory_cache_hit_rate',
    'Redis cache hit rate',
    ['cache_type']  # working, embedding
)

# Database metrics
memory_database_query_duration = Histogram(
    'memory_database_query_duration_seconds',
    'PostgreSQL query duration',
    ['query_type']  # search, insert, update, delete
)

# External API metrics
memory_embedding_api_latency = Histogram(
    'memory_embedding_api_latency_seconds',
    'Embedding service API latency'
)

memory_embedding_api_errors = Counter(
    'memory_embedding_api_errors_total',
    'Embedding service API errors',
    ['error_type']
)
```

**Grafana Dashboard:**

- Panel 1: p95 retrieval latency (line graph, target <100ms)
- Panel 2: Throughput (req/s, line graph)
- Panel 3: Cache hit rate (gauge, target >50%)
- Panel 4: Database query duration by type (stacked area)
- Panel 5: Embedding API latency and error rate
- Panel 6: Memory count by type (stacked bar)

**Alerting Rules (Prometheus AlertManager):**

```yaml
groups:
  - name: memory_system
    rules:
      - alert: HighRetrievalLatency
        expr: histogram_quantile(0.95, memory_retrieval_latency_seconds) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Memory retrieval p95 latency above 100ms"

      - alert: LowCacheHitRate
        expr: memory_cache_hit_rate{cache_type="working"} < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Working memory cache hit rate below 50%"

      - alert: EmbeddingAPIErrors
        expr: rate(memory_embedding_api_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Embedding API error rate above 5%"
```

---

## ⚠️ Error Handling & Edge Cases

### Error Scenarios (from Research)

**Critical Error Paths:**

**1. Embedding Service Failure**

- **Cause:** OpenAI API unavailable, rate limit exceeded, network timeout
- **Impact:** Cannot encode new memories (HIGH)
- **Handling:**
  - Retry with exponential backoff (3 attempts, 1s/2s/4s delays)
  - Circuit breaker pattern (open after 5 consecutive failures, 60s timeout)
  - Graceful degradation: Store memory without embedding, queue for retry
- **Recovery:**
  - Background job processes failed encodings when service recovers
  - Alert on-call engineer if circuit breaker opens
- **User Experience:**
  - Return 202 Accepted with "Memory stored, encoding pending"
  - Retry embedding on next retrieval attempt

**2. Database Connection Failure**

- **Cause:** PostgreSQL unavailable, connection pool exhausted, network partition
- **Impact:** Cannot store/retrieve memories (HIGH)
- **Handling:**
  - SQLAlchemy retry logic (3 attempts with pool_pre_ping=True)
  - Read from replica if primary unavailable (fallback)
  - Fail fast with clear error message
- **Recovery:**
  - Automatic reconnection when database recovers
  - Log error with trace_id for debugging
- **User Experience:**
  - Return 503 Service Unavailable with "Memory service temporarily unavailable"
  - Suggest retry after 30 seconds

**3. Redis Cache Unavailable**

- **Cause:** Redis crash, network partition, memory eviction
- **Impact:** Working memory cache miss, higher latency (MEDIUM)
- **Handling:**
  - Degrade gracefully to database-only operation
  - Log warning but don't fail request
  - Automatic cache rebuild on Redis recovery
- **Recovery:**
  - Background job warms up cache after Redis restart
  - Monitor cache hit rate, alert if drops below 20%
- **User Experience:**
  - Transparent to user (slightly higher latency)
  - No error message, request succeeds

**4. Token Budget Exceeded**

- **Cause:** Too many memories retrieved, compression fails, large content
- **Impact:** Cannot fit context in max_tokens (MEDIUM)
- **Handling:**
  - Hard truncation fallback (truncate to max_tokens, preserve recent memories)
  - Log warning with agent_id and query for analysis
  - Return partial context with warning flag
- **Recovery:**
  - Suggest user increase max_tokens or reduce query scope
  - Trigger manual compression review
- **User Experience:**
  - Return context with `"truncated": true` flag
  - Include warning: "Context truncated to fit token budget"

### Edge Cases (from Feature Request & Research)

**Identified Edge Cases:**

| Edge Case | Detection | Handling | Testing Approach |
|-----------|-----------|----------|------------------|
| **Empty Query String** | `len(query.strip()) == 0` | Return empty context with warning | Unit test with `query=""` |
| **Very Long Content (>100K chars)** | `len(content) > 100000` | Truncate with warning, summarize | Integration test with long text |
| **Duplicate Memories** | Hash content, check duplicates | Skip insert, update access_count | Unit test with same content |
| **Agent with No Memories** | `count_by_type(agent_id) == 0` | Return empty list, no error | Integration test with new agent |
| **Task ID Not Found** | Check task_id in tasks table | Log warning, proceed without task scope | Integration test with invalid task_id |
| **Contradictory Memories** | Detect semantic contradictions | Flag both, prefer recent | Unit test with synthetic contradictions |
| **Expired Working Memory (TTL)** | Redis key missing | Fetch from database, rebuild cache | Integration test with expired TTL |
| **Embedding Dimension Mismatch** | `len(embedding) != 1536` | Reject with validation error | Unit test with wrong dimension |
| **Malicious SQL in Query** | Pydantic validation, parameterized queries | Sanitize, log security event | Security test with sqlmap |
| **Cross-Agent Access Attempt** | `agent_id != request.a2a_context.source_agent` | Return 403 Forbidden | Integration test with wrong agent_id |

### Input Validation

**Validation Rules (Pydantic):**

```python
from pydantic import BaseModel, Field, field_validator

class MemoryStoreParams(BaseModel):
    """Parameters for memory.store method."""

    interaction: Interaction

    @field_validator('interaction')
    @classmethod
    def validate_interaction(cls, v: Interaction) -> Interaction:
        # Validate non-empty query
        if not v.query.strip():
            raise ValueError("Interaction query cannot be empty")

        # Validate content length
        if len(v.query) > 100000:
            raise ValueError("Interaction query exceeds maximum length (100K chars)")

        # Validate timestamp not in future
        if v.timestamp > datetime.now(UTC):
            raise ValueError("Interaction timestamp cannot be in future")

        return v

class MemoryRetrieveParams(BaseModel):
    """Parameters for memory.retrieve method."""

    query: str = Field(min_length=1, max_length=10000)
    k: int = Field(ge=1, le=50, default=5)
    memory_types: list[str] = Field(default=["episodic", "semantic"])

    @field_validator('memory_types')
    @classmethod
    def validate_memory_types(cls, v: list[str]) -> list[str]:
        valid_types = {"working", "episodic", "semantic", "procedural"}
        invalid = set(v) - valid_types
        if invalid:
            raise ValueError(f"Invalid memory types: {invalid}")
        return v
```

**Sanitization:**

- **XSS Prevention:** HTML escape all user content before display
- **SQL Injection Prevention:** Use parameterized queries (SQLAlchemy prevents by default)
- **Input Normalization:** Strip whitespace, lowercase for case-insensitive search

### Graceful Degradation

**Fallback Strategies:**

**1. Embedding Service Unavailable:**

- **Fallback:** Keyword-based search (PostgreSQL full-text search)
- **Degraded Mode:** 60% precision vs 90% with embeddings
- **Implementation:**

  ```python
  async def retrieve_memories_fallback(query: str, agent_id: UUID, k: int):
      # Use PostgreSQL tsvector for keyword search
      stmt = select(MemoryModel).where(
          and_(
              MemoryModel.agent_id == agent_id,
              MemoryModel.summary.match(query)  # Full-text search
          )
      ).order_by(MemoryModel.timestamp.desc()).limit(k)
      return await session.execute(stmt)
  ```

**2. Compression Service Fails:**

- **Fallback:** Hard truncation to max_tokens
- **Degraded Mode:** Context may lose less important information
- **Implementation:**

  ```python
  def truncate_context(context: str, max_tokens: int) -> str:
      tokens = tokenizer.encode(context)
      if len(tokens) > max_tokens:
          truncated = tokenizer.decode(tokens[:max_tokens])
          return truncated + "\n[Context truncated due to compression failure]"
      return context
  ```

**3. Database Read Replica Unavailable:**

- **Fallback:** Query primary database
- **Degraded Mode:** Higher load on primary
- **Implementation:**

  ```python
  async def get_memories(agent_id: UUID):
      try:
          # Try read replica first
          async with get_session(replica=True) as session:
              return await repository.search(agent_id)
      except ConnectionError:
          # Fallback to primary
          async with get_session(replica=False) as session:
              return await repository.search(agent_id)
  ```

### Monitoring & Alerting

**Error Tracking (Sentry):**

```python
import sentry_sdk

try:
    embedding = await embedding_service.embed(query)
except EmbeddingServiceError as e:
    sentry_sdk.capture_exception(e)
    sentry_sdk.set_context("memory", {
        "agent_id": str(agent_id),
        "query": query[:100],  # First 100 chars
        "error_type": type(e).__name__
    })
    # Fallback to keyword search
    return await retrieve_memories_fallback(query, agent_id, k)
```

**Error Thresholds:**

- Embedding API errors: Alert if >5% error rate in 5 minutes
- Database connection errors: Alert immediately (critical)
- Cache miss rate: Alert if <20% hit rate in 10 minutes
- Retrieval latency: Alert if p95 >200ms in 5 minutes

---

## 📚 References & Traceability

### Source Documentation

**Research & Intelligence:**

- **docs/research/evolving-memory-system.md**
  - Four-layer memory architecture (Working, Episodic, Semantic, Procedural)
  - Memory operations (encode, store, retrieve, update, prune)
  - Code examples for importance scoring, compression, retrieval
  - Performance targets: 80% context reduction, 25-30% task improvement
  - Technology recommendations: PGVector, OpenAI embeddings, Redis caching
  - Success metrics: Context efficiency, retrieval precision, coherence, latency

**Specification:**

- **docs/specs/memory-system/spec.md**
  - 6 functional requirements (FR-1 through FR-6)
  - 6 non-functional requirements (NFR-1 through NFR-6)
  - 8 feature breakdowns with priorities (P0, P1, P2)
  - 8 acceptance criteria with quantifiable targets
  - Dependencies: PostgreSQL, Redis, embedding service, task management

### System Context

**Architecture & Patterns:**

- **CLAUDE.md** - AgentCore project guide
  - Stack: Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy async
  - Patterns: Service managers, JSON-RPC handlers, repositories
  - Development commands: uv, pytest, alembic, docker compose
  - Architecture overview: Core components, key design patterns

- **docs/agentcore-architecture-and-development-plan.md** - System architecture
  - 6-layer architecture (Infrastructure, Runtime, Core, Operations, Experience, Intelligence)
  - A2A protocol implementation (JSON-RPC 2.0, distributed tracing)
  - Integration points: Task Manager, Agent Manager, Embedding Service

**Code Examples:**

- **src/agentcore/a2a_protocol/services/task_manager.py** - Service manager pattern
- **src/agentcore/a2a_protocol/services/task_jsonrpc.py** - JSON-RPC handler pattern
- **src/agentcore/a2a_protocol/database/models.py** - SQLAlchemy model pattern
- **src/agentcore/a2a_protocol/database/repositories.py** - Repository pattern
- **src/agentcore/a2a_protocol/services/embedding_service.py** - Existing embedding integration

### Technology Evaluation

**Vector Database Selection:**

- **PGVector Documentation**: <https://github.com/pgvector/pgvector>
  - Extension for PostgreSQL providing vector similarity search
  - Supports cosine distance, L2 distance, inner product
  - IVFFlat indexing for approximate nearest neighbor search
  - Performance: Sub-100ms for 1M vectors with proper indexing

**Embedding Models:**

- **OpenAI text-embedding-3-small**: <https://platform.openai.com/docs/guides/embeddings>
  - Dimensions: 1536
  - Cost: $0.00002 per 1K tokens
  - Performance: Industry-leading semantic understanding

**Performance Benchmarks:**

- **PGVector Benchmarks**: <https://github.com/pgvector/pgvector#benchmark>
  - 1M vectors: ~50ms p95 latency with IVFFlat (lists=100)
  - 10M vectors: ~150ms p95 latency (requires tuning)

### Related Components

**Dependencies:**

- **Task Management**: docs/specs/task-system/spec.md (if exists)
  - Provides task_id for memory scoping
  - Task lifecycle events trigger memory operations
- **Agent Manager**: src/agentcore/a2a_protocol/services/agent_manager.py
  - Provides agent_id for memory isolation
  - Agent discovery includes memory capabilities
- **Embedding Service**: src/agentcore/a2a_protocol/services/embedding_service.py
  - Generates embeddings for memory encoding
  - Supports OpenAI, Cohere, local models

**Dependents:**

- **Bounded Context Reasoning** (BCR-001, UNPROCESSED): docs/specs/bounded-context-reasoning/plan.md
  - Will use memory system for context-aware reasoning
  - Memory provides domain-specific knowledge
- **Multi-tool Integration** (future): docs/research/multi-tool-integration.md
  - Will use procedural memory for tool selection heuristics
- **Flow-based Optimization** (future): docs/research/flow-based-optimization.md
  - Will use memory to track successful workflow patterns

---

**Plan Status:** ✅ Complete and Ready for Implementation

**Next Steps:**

1. Review plan with engineering team
2. Create epic ticket MEM-001 with dependencies
3. Run `/sage.tasks MEM-001` to generate story tickets
4. Begin Phase 1 implementation

**Estimated Effort:** 6 weeks (1 senior engineer full-time)

**Risk Level:** LOW (technology proven, dependencies exist, clear acceptance criteria)
