# Evolving Memory System - Task Breakdown

**Component:** memory-system
**Epic:** MEM-001
**Total Effort:** 60 Story Points (6 weeks)
**Team Size:** 2-3 Engineers
**Start Date:** TBD
**Target Completion:** Week 6

## Summary

This document provides a detailed task breakdown for implementing the Evolving Memory System as specified in [spec.md](./spec.md) and [plan.md](./plan.md). The implementation is divided into 4 phases over 6 weeks, with clear dependencies and acceptance criteria for each task.

**Key Metrics:**

- **Total Tasks:** 25 stories
- **Total Effort:** 60 story points
- **Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 (sequential dependencies)
- **Risk Level:** LOW (proven technology stack, clear requirements)

**Phase Distribution:**

- Phase 1 (Foundation): 20 SP, 8 stories, Weeks 1-2
- Phase 2 (JSON-RPC Integration): 10 SP, 7 stories, Week 3
- Phase 3 (Agent Integration): 10 SP, 6 stories, Week 4
- Phase 4 (Advanced Features): 20 SP, 4 stories, Weeks 5-6

## Phase 1: Core Memory System (Weeks 1-2, 20 SP)

### MEM-002: Database Schema and Migrations

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** None
**Owner:** Backend Engineer

**Description:**
Create PostgreSQL database schema with PGVector extension support for storing memory records with vector embeddings.

**Acceptance Criteria:**

- [ ] Alembic migration creates `memories` table with all required columns
- [ ] PGVector extension installed and configured
- [ ] IVFFlat index created on embedding column (lists=100)
- [ ] Agent isolation enforced via agent_id column with index
- [ ] Foreign key constraints to tasks table (task_id)
- [ ] Default values set for relevance_score (1.0), access_count (0)
- [ ] Migration reversible (downgrade script works)
- [ ] Database initialization documented in README

**Technical Details:**

```sql
CREATE TABLE memories (
    memory_id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding VECTOR(1536),
    timestamp TIMESTAMP NOT NULL,
    interaction_id UUID,
    task_id UUID,
    entities TEXT[],
    facts TEXT[],
    keywords TEXT[],
    related_memory_ids UUID[],
    parent_memory_id UUID,
    relevance_score FLOAT DEFAULT 1.0,
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMP,
    actions TEXT[],
    outcome TEXT,
    success BOOLEAN,
    metadata JSONB,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE INDEX idx_memories_agent_id ON memories(agent_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);
```

**Files:**

- `alembic/versions/XXXX_add_memory_tables.py`
- `docs/database/schema.md` (update)

---

### MEM-003: Pydantic Models for Memory System

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** None
**Owner:** Backend Engineer

**Description:**
Define Pydantic models for memory records, interactions, and memory operations following AgentCore validation patterns.

**Acceptance Criteria:**

- [ ] `MemoryRecord` model with all fields from spec
- [ ] `Interaction` model for encoding inputs
- [ ] `MemoryQuery` model for retrieval requests
- [ ] `MemoryStats` model for analytics
- [ ] Field validation (timestamp formats, UUID validation, enum constraints)
- [ ] JSON schema export working
- [ ] Example instances in docstrings
- [ ] 100% test coverage for model validation

**Technical Details:**

```python
class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryRecord(BaseModel):
    memory_id: str
    agent_id: str
    memory_type: MemoryType
    content: str
    summary: str
    embedding: list[float]
    timestamp: datetime
    interaction_id: str | None = None
    task_id: str | None = None
    entities: list[str] = []
    facts: list[str] = []
    keywords: list[str] = []
    related_memory_ids: list[str] = []
    parent_memory_id: str | None = None
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: datetime | None = None
    actions: list[str] = []
    outcome: str | None = None
    success: bool | None = None
    metadata: dict[str, Any] = {}
```

**Files:**

- `src/agentcore/memory/models.py`
- `tests/unit/memory/test_models.py`

---

### MEM-004: SQLAlchemy ORM Models

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** MEM-002
**Owner:** Backend Engineer

**Description:**
Create SQLAlchemy async ORM models for memory storage with PGVector column types.

**Acceptance Criteria:**

- [ ] `MemoryModel` class with all database columns
- [ ] PGVector column type correctly configured
- [ ] Relationships to TaskRecord and AgentRecord (if applicable)
- [ ] to_dict() method for serialization
- [ ] from_pydantic() class method for conversion
- [ ] Indexes defined in model metadata
- [ ] Compatible with async SQLAlchemy sessions
- [ ] Unit tests for model operations (create, read, update)

**Technical Details:**

```python
from pgvector.sqlalchemy import Vector

class MemoryModel(Base):
    __tablename__ = "memories"

    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    memory_type = Column(String(50), nullable=False, index=True)
    embedding = Column(Vector(1536))
    # ... other columns
```

**Files:**

- `src/agentcore/memory/database/models.py`
- `tests/unit/memory/test_database_models.py`

---

### MEM-005: Memory Repository with Vector Search

**Priority:** P0
**Type:** Story
**Effort:** 5 SP
**Dependencies:** MEM-003, MEM-004
**Owner:** Backend Engineer

**Description:**
Implement MemoryRepository for CRUD operations and vector similarity search using PGVector.

**Acceptance Criteria:**

- [ ] CRUD methods: create(), get_by_id(), update(), delete()
- [ ] vector_search() method with cosine similarity
- [ ] search_by_metadata() for filtering by agent_id, task_id, memory_type
- [ ] hybrid_search() combining vector similarity + metadata filters
- [ ] Batch operations: create_many(), get_many()
- [ ] Agent isolation enforced (all queries filtered by agent_id)
- [ ] Proper error handling (NotFoundError, DatabaseError)
- [ ] 90%+ test coverage with integration tests using testcontainers

**Technical Details:**

```python
class MemoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def vector_search(
        self,
        agent_id: str,
        embedding: list[float],
        memory_types: list[str] | None = None,
        k: int = 5,
        threshold: float = 0.7
    ) -> list[MemoryModel]:
        """Search using PGVector cosine distance."""
        query = select(MemoryModel).where(
            MemoryModel.agent_id == agent_id
        )
        if memory_types:
            query = query.where(MemoryModel.memory_type.in_(memory_types))

        query = query.order_by(
            MemoryModel.embedding.cosine_distance(embedding)
        ).limit(k)

        result = await self.session.execute(query)
        return result.scalars().all()
```

**Files:**

- `src/agentcore/memory/database/repositories.py`
- `tests/integration/memory/test_repository.py`

---

### MEM-006: Encoding Service

**Priority:** P0
**Type:** Story
**Effort:** 3 SP
**Dependencies:** MEM-003
**Owner:** Backend Engineer

**Description:**
Implement EncodingService for converting interactions into memory records with summarization and entity extraction.

**Acceptance Criteria:**

- [ ] encode_episodic() creates episodic memory from interaction
- [ ] encode_semantic() extracts semantic facts from content
- [ ] encode_procedural() creates action-outcome pairs
- [ ] Uses existing embedding_service.py for vector generation
- [ ] LLM-based summarization (configurable model)
- [ ] Entity extraction using NER or LLM
- [ ] Keyword extraction (TF-IDF or LLM)
- [ ] Graceful degradation if embedding service fails
- [ ] 85%+ test coverage with mocked LLM calls

**Technical Details:**

```python
class EncodingService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_client: AsyncOpenAI
    ):
        self.embedding_service = embedding_service
        self.llm = llm_client

    async def encode_episodic(
        self,
        interaction: Interaction
    ) -> MemoryRecord:
        summary = await self._summarize(interaction.content)
        embedding = await self.embedding_service.embed(summary)
        entities = await self._extract_entities(interaction.content)
        facts = await self._extract_facts(interaction.content)

        return MemoryRecord(
            memory_id=str(uuid.uuid4()),
            memory_type=MemoryType.EPISODIC,
            content=interaction.content,
            summary=summary,
            embedding=embedding,
            timestamp=interaction.timestamp,
            entities=entities,
            facts=facts,
            # ... other fields
        )
```

**Files:**

- `src/agentcore/memory/encoding.py`
- `tests/unit/memory/test_encoding.py`

---

### MEM-007: Retrieval Service with Hybrid Search

**Priority:** P0
**Type:** Story
**Effort:** 3 SP
**Dependencies:** MEM-005
**Owner:** Backend Engineer

**Description:**
Implement RetrievalService with hybrid search combining vector similarity, temporal recency, and access frequency.

**Acceptance Criteria:**

- [ ] retrieve() method with configurable scoring weights
- [ ] Vector similarity using repository.vector_search()
- [ ] Temporal recency scoring (exponential decay, 24h half-life)
- [ ] Access frequency scoring (normalized access_count)
- [ ] Combined importance score calculation
- [ ] Top-k selection with score threshold
- [ ] Access tracking (increment access_count, update last_accessed)
- [ ] Configurable weights via Settings
- [ ] 90%+ test coverage with synthetic test data

**Technical Details:**

```python
class RetrievalService:
    def __init__(
        self,
        repository: MemoryRepository,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.3
    ):
        self.repository = repository
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight

    async def retrieve(
        self,
        agent_id: str,
        query_embedding: list[float],
        memory_types: list[str],
        k: int = 5
    ) -> list[MemoryRecord]:
        # Get candidates from vector search
        candidates = await self.repository.vector_search(
            agent_id=agent_id,
            embedding=query_embedding,
            memory_types=memory_types,
            k=k * 2  # Over-fetch for reranking
        )

        # Calculate importance scores
        scored = [
            (memory, self._calculate_importance(memory))
            for memory in candidates
        ]

        # Sort by importance and take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m for m, _ in scored[:k]]

        # Update access tracking
        await self._update_access_tracking(top_memories)

        return top_memories
```

**Files:**

- `src/agentcore/memory/retrieval.py`
- `tests/unit/memory/test_retrieval.py`

---

### MEM-008: Memory Manager Orchestration

**Priority:** P0
**Type:** Story
**Effort:** 5 SP
**Dependencies:** MEM-006, MEM-007
**Owner:** Backend Engineer

**Description:**
Implement MemoryManager as central orchestrator coordinating encoding, storage, and retrieval operations.

**Acceptance Criteria:**

- [ ] add_interaction() orchestrates encode → store → prune workflow
- [ ] get_relevant_context() retrieves and formats memories for LLM
- [ ] update_memory() allows corrections to existing memories
- [ ] prune_memories() removes low-value memories when capacity exceeded
- [ ] Agent isolation enforced across all operations
- [ ] Transaction management for multi-step operations
- [ ] Error handling with rollback on failures
- [ ] Metrics instrumentation (latency, memory count, cache hit rate)
- [ ] 85%+ test coverage with integration tests

**Technical Details:**

```python
class MemoryManager:
    def __init__(
        self,
        repository: MemoryRepository,
        encoding_service: EncodingService,
        retrieval_service: RetrievalService,
        capacity_limits: dict[str, int]
    ):
        self.repository = repository
        self.encoding = encoding_service
        self.retrieval = retrieval_service
        self.capacity_limits = capacity_limits

    async def add_interaction(
        self,
        agent_id: str,
        interaction: Interaction
    ) -> str:
        # Encode
        episodic = await self.encoding.encode_episodic(interaction)
        episodic.agent_id = agent_id

        # Store
        memory_id = await self.repository.create(episodic)

        # Check capacity and prune if needed
        await self._check_and_prune(agent_id, MemoryType.EPISODIC)

        return memory_id

    async def get_relevant_context(
        self,
        agent_id: str,
        query: str,
        task_id: str | None = None,
        max_tokens: int = 2000
    ) -> str:
        # Generate query embedding
        embedding = await self.encoding.embedding_service.embed(query)

        # Retrieve memories
        memories = await self.retrieval.retrieve(
            agent_id=agent_id,
            query_embedding=embedding,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            k=10
        )

        # Format and truncate
        context = self._format_context(memories, max_tokens)
        return context
```

**Files:**

- `src/agentcore/memory/manager.py`
- `tests/integration/memory/test_manager.py`

---

### MEM-009: Compression Service

**Priority:** P1
**Type:** Story
**Effort:** 3 SP
**Dependencies:** MEM-008
**Owner:** Backend Engineer

**Description:**
Implement CompressionService for token counting and LLM-based summarization to maintain bounded context.

**Acceptance Criteria:**

- [ ] count_tokens() using tiktoken for accurate token counting
- [ ] truncate_to_tokens() truncates text to target token count
- [ ] compress_memories() summarizes memory sequence with LLM
- [ ] hierarchical_summarization() for large memory sets
- [ ] Configurable target compression ratio (default 50%)
- [ ] Preserves key entities and facts during compression
- [ ] Fallback to simple truncation if LLM fails
- [ ] 80%+ test coverage with token count assertions

**Technical Details:**

```python
import tiktoken

class CompressionService:
    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def compress_memories(
        self,
        memories: list[MemoryRecord],
        target_tokens: int
    ) -> str:
        if not memories:
            return ""

        current_text = "\n".join(m.summary for m in memories)
        current_tokens = self.count_tokens(current_text)

        if current_tokens <= target_tokens:
            return current_text

        # LLM compression
        prompt = f"""Compress the following memories to ~{target_tokens} tokens:

{current_text}

Focus on key facts and patterns. Preserve entity names."""

        compressed = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=target_tokens
        )

        return compressed.choices[0].message.content
```

**Files:**

- `src/agentcore/memory/compression.py`
- `tests/unit/memory/test_compression.py`

---

## Phase 2: JSON-RPC Integration (Week 3, 10 SP)

### MEM-010: JSON-RPC Method Registration

**Priority:** P0
**Type:** Story
**Effort:** 1 SP
**Dependencies:** MEM-008
**Owner:** Backend Engineer

**Description:**
Create memory_jsonrpc.py module and register all memory.* JSON-RPC methods with jsonrpc_processor.

**Acceptance Criteria:**

- [ ] memory_jsonrpc.py follows task_jsonrpc.py pattern
- [ ] All 7 methods registered: memory.store, memory.retrieve, memory.get_context, memory.update, memory.prune, memory.stats, memory.search
- [ ] Import in main.py for auto-registration
- [ ] Method names follow namespace.action convention
- [ ] Docstrings with parameter and return value descriptions
- [ ] Methods show up in rpc.methods response

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py`
- `src/agentcore/a2a_protocol/main.py` (update imports)

---

### MEM-011: memory.store Handler

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** MEM-010
**Owner:** Backend Engineer

**Description:**
Implement memory.store JSON-RPC method for storing new interaction memories.

**Acceptance Criteria:**

- [ ] Accepts Interaction object in params
- [ ] Extracts agent_id from a2a_context.source_agent
- [ ] Calls memory_manager.add_interaction()
- [ ] Returns memory_id on success
- [ ] Error handling: INVALID_PARAMS, INTERNAL_ERROR
- [ ] Input validation with Pydantic
- [ ] Integration test with full request/response cycle
- [ ] API documented in docs/api/memory.md

**Technical Details:**

```python
@register_jsonrpc_method("memory.store")
async def handle_memory_store(request: JsonRpcRequest) -> dict[str, Any]:
    """Store a new memory from an interaction."""
    if not request.a2a_context or not request.a2a_context.source_agent:
        raise JsonRpcError(
            code=JsonRpcErrorCode.INVALID_PARAMS,
            message="source_agent required in a2a_context"
        )

    interaction = Interaction(**request.params["interaction"])
    agent_id = request.a2a_context.source_agent

    memory_id = await memory_manager.add_interaction(agent_id, interaction)

    return {"memory_id": memory_id}
```

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py` (update)
- `tests/integration/memory/test_memory_jsonrpc.py`
- `docs/api/memory.md`

---

### MEM-012: memory.retrieve Handler

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** MEM-010
**Owner:** Backend Engineer

**Description:**
Implement memory.retrieve JSON-RPC method for searching memories by semantic similarity.

**Acceptance Criteria:**

- [ ] Accepts query, memory_types, k in params
- [ ] Extracts agent_id from a2a_context
- [ ] Calls memory_manager.retrieval.retrieve()
- [ ] Returns list of MemoryRecord summaries with scores
- [ ] Default k=5, memory_types=["episodic", "semantic"]
- [ ] Error handling for missing agent context
- [ ] Integration test with sample memories
- [ ] Performance test: <100ms for 10K memories

**Technical Details:**

```python
@register_jsonrpc_method("memory.retrieve")
async def handle_memory_retrieve(request: JsonRpcRequest) -> dict[str, Any]:
    """Retrieve relevant memories by semantic search."""
    agent_id = request.a2a_context.source_agent
    query = request.params["query"]
    memory_types = request.params.get("memory_types", ["episodic", "semantic"])
    k = request.params.get("k", 5)

    # Generate embedding
    embedding = await memory_manager.encoding.embedding_service.embed(query)

    # Retrieve
    memories = await memory_manager.retrieval.retrieve(
        agent_id=agent_id,
        query_embedding=embedding,
        memory_types=memory_types,
        k=k
    )

    return {
        "memories": [
            {
                "memory_id": m.memory_id,
                "summary": m.summary,
                "timestamp": m.timestamp.isoformat(),
                "relevance_score": m.relevance_score,
                "memory_type": m.memory_type
            }
            for m in memories
        ]
    }
```

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py` (update)
- `tests/integration/memory/test_memory_jsonrpc.py` (update)

---

### MEM-013: memory.get_context Handler

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** MEM-010
**Owner:** Backend Engineer

**Description:**
Implement memory.get_context JSON-RPC method for retrieving formatted context suitable for LLM prompts.

**Acceptance Criteria:**

- [ ] Accepts query, task_id, max_tokens in params
- [ ] Calls memory_manager.get_relevant_context()
- [ ] Returns formatted markdown string
- [ ] Default max_tokens=2000
- [ ] Context includes sections: Current Task, Relevant Interactions, Relevant Knowledge
- [ ] Token counting accurate (within 5% of actual)
- [ ] Integration test validates format and token limit
- [ ] End-to-end test with real LLM call

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py` (update)
- `tests/integration/memory/test_memory_jsonrpc.py` (update)

---

### MEM-014: memory.update Handler

**Priority:** P1
**Type:** Story
**Effort:** 1 SP
**Dependencies:** MEM-010
**Owner:** Backend Engineer

**Description:**
Implement memory.update JSON-RPC method for correcting or enhancing existing memories.

**Acceptance Criteria:**

- [ ] Accepts memory_id and updates dict in params
- [ ] Validates agent ownership before update
- [ ] Supports updating: summary, facts, entities, relevance_score
- [ ] Regenerates embedding if summary changed
- [ ] Returns success status
- [ ] Error handling: NOT_FOUND, PERMISSION_DENIED
- [ ] Integration test with update scenarios

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py` (update)
- `tests/integration/memory/test_memory_jsonrpc.py` (update)

---

### MEM-015: memory.prune Handler

**Priority:** P1
**Type:** Story
**Effort:** 1 SP
**Dependencies:** MEM-010
**Owner:** Backend Engineer

**Description:**
Implement memory.prune JSON-RPC method for manual memory pruning by agents.

**Acceptance Criteria:**

- [ ] Accepts memory_type, strategy, count in params
- [ ] Strategies: "least_relevant", "oldest_first", "least_accessed"
- [ ] Calls memory_manager.prune_memories()
- [ ] Returns count of pruned memories
- [ ] Respects agent isolation (only prune own memories)
- [ ] Integration test validates correct memories pruned

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py` (update)
- `tests/integration/memory/test_memory_jsonrpc.py` (update)

---

### MEM-016: memory.stats Handler

**Priority:** P2
**Type:** Story
**Effort:** 1 SP
**Dependencies:** MEM-010
**Owner:** Backend Engineer

**Description:**
Implement memory.stats JSON-RPC method for retrieving memory usage statistics.

**Acceptance Criteria:**

- [ ] Returns stats per memory_type: count, total_tokens, avg_relevance
- [ ] Includes overall stats: total_memories, storage_bytes
- [ ] Retrieval performance metrics: avg_latency, cache_hit_rate
- [ ] Agent-specific stats only (respects isolation)
- [ ] Integration test validates accuracy

**Files:**

- `src/agentcore/a2a_protocol/services/memory_jsonrpc.py` (update)
- `tests/integration/memory/test_memory_jsonrpc.py` (update)

---

## Phase 3: Agent Integration & Caching (Week 4, 10 SP)

### MEM-017: Redis Cache for Working Memory

**Priority:** P0
**Type:** Story
**Effort:** 3 SP
**Dependencies:** MEM-008
**Owner:** Backend Engineer

**Description:**
Implement Redis-based caching layer for working memory with TTL-based expiration.

**Acceptance Criteria:**

- [ ] MemoryCacheService using existing Redis connection
- [ ] set_working_memory() stores task-scoped context
- [ ] get_working_memory() retrieves cached context
- [ ] Default TTL: 1 hour (configurable)
- [ ] clear_working_memory() for task completion
- [ ] Key format: "memory:working:{agent_id}:{task_id}"
- [ ] JSON serialization of MemoryRecord
- [ ] 90%+ test coverage with redis-mock
- [ ] Integration test with real Redis (testcontainers)

**Technical Details:**

```python
class MemoryCacheService:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    async def set_working_memory(
        self,
        agent_id: str,
        task_id: str,
        memories: list[MemoryRecord]
    ) -> None:
        key = f"memory:working:{agent_id}:{task_id}"
        value = json.dumps([m.model_dump() for m in memories])
        await self.redis.setex(key, self.ttl, value)

    async def get_working_memory(
        self,
        agent_id: str,
        task_id: str
    ) -> list[MemoryRecord]:
        key = f"memory:working:{agent_id}:{task_id}"
        value = await self.redis.get(key)
        if not value:
            return []
        data = json.loads(value)
        return [MemoryRecord(**d) for d in data]
```

**Files:**

- `src/agentcore/memory/cache.py`
- `tests/unit/memory/test_cache.py`
- `tests/integration/memory/test_cache_integration.py`

---

### MEM-018: Embedding Cache Layer

**Priority:** P1
**Type:** Story
**Effort:** 2 SP
**Dependencies:** MEM-017
**Owner:** Backend Engineer

**Description:**
Add Redis caching for embeddings to reduce API calls to embedding service.

**Acceptance Criteria:**

- [ ] Cache key: SHA-256 hash of input text
- [ ] TTL: 24 hours (configurable)
- [ ] Cache hit rate metric instrumented
- [ ] Falls back to embedding service on cache miss
- [ ] Atomic cache-aside pattern (check → fetch → store)
- [ ] Integration test validates cache behavior
- [ ] Performance test: 95%+ cache hit rate on repeated queries

**Files:**

- `src/agentcore/memory/encoding.py` (update)
- `tests/integration/memory/test_embedding_cache.py`

---

### MEM-019: MemoryEnabledAgent Base Class

**Priority:** P1
**Type:** Story
**Effort:** 3 SP
**Dependencies:** MEM-017
**Owner:** Backend Engineer

**Description:**
Create MemoryEnabledAgent base class that agents can extend to gain memory capabilities.

**Acceptance Criteria:**

- [ ] Mixin class with memory_manager dependency
- [ ] _store_interaction() helper method
- [ ] _retrieve_context() helper method
- [ ] _format_prompt_with_context() helper
- [ ] Working memory automatically managed per task
- [ ] Example agent implementation in documentation
- [ ] Integration test with sample agent workflow
- [ ] Backward compatible with existing Agent interface

**Technical Details:**

```python
class MemoryEnabledAgent:
    """Mixin for agents with memory capabilities."""

    def __init__(self, memory_manager: MemoryManager, **kwargs):
        self.memory = memory_manager
        super().__init__(**kwargs)

    async def _store_interaction(
        self,
        query: str,
        response: str,
        task_id: str | None = None,
        success: bool = True
    ) -> None:
        interaction = Interaction(
            id=str(uuid.uuid4()),
            task_id=task_id,
            query=query,
            response=response,
            timestamp=datetime.utcnow(),
            success=success
        )
        await self.memory.add_interaction(self.agent_id, interaction)

    async def _retrieve_context(
        self,
        query: str,
        task_id: str | None = None,
        max_tokens: int = 2000
    ) -> str:
        return await self.memory.get_relevant_context(
            agent_id=self.agent_id,
            query=query,
            task_id=task_id,
            max_tokens=max_tokens
        )
```

**Files:**

- `src/agentcore/memory/agent_mixin.py`
- `examples/memory_enabled_agent.py`
- `tests/integration/memory/test_agent_mixin.py`

---

### MEM-020: Configuration Management

**Priority:** P0
**Type:** Story
**Effort:** 1 SP
**Dependencies:** None
**Owner:** Backend Engineer

**Description:**
Add memory system configuration to Settings with environment variable support.

**Acceptance Criteria:**

- [ ] MemorySettings class in config.py
- [ ] Environment variables: MEMORY_CAPACITY_*, MEMORY_CACHE_TTL, MEMORY_RETRIEVAL_*
- [ ] Default values for all settings
- [ ] Validation: positive integers, float ranges 0-1
- [ ] Documentation in .env.example
- [ ] Settings loaded in memory module **init**.py
- [ ] Integration test validates config loading

**Technical Details:**

```python
class MemorySettings(BaseSettings):
    # Capacity limits per memory type
    memory_capacity_working: int = Field(default=10, ge=1)
    memory_capacity_episodic: int = Field(default=50, ge=10)
    memory_capacity_semantic: int = Field(default=1000, ge=100)
    memory_capacity_procedural: int = Field(default=500, ge=50)

    # Cache settings
    memory_cache_ttl_seconds: int = Field(default=3600, ge=60)
    memory_embedding_cache_ttl_seconds: int = Field(default=86400, ge=3600)

    # Retrieval settings
    memory_retrieval_k: int = Field(default=5, ge=1, le=50)
    memory_retrieval_relevance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    memory_retrieval_recency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    memory_retrieval_frequency_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Compression settings
    memory_compression_target_ratio: float = Field(default=0.5, ge=0.1, le=0.9)
```

**Files:**

- `src/agentcore/a2a_protocol/config.py` (update)
- `.env.example` (update)
- `docs/configuration.md` (update)

---

### MEM-021: Memory Module Initialization

**Priority:** P0
**Type:** Story
**Effort:** 1 SP
**Dependencies:** MEM-008, MEM-017, MEM-020
**Owner:** Backend Engineer

**Description:**
Set up memory module initialization with dependency injection and global instance management.

**Acceptance Criteria:**

- [ ] init_memory() function creates MemoryManager singleton
- [ ] Dependency injection for database session, Redis, embedding service
- [ ] Called from main.py startup event
- [ ] close_memory() for cleanup on shutdown
- [ ] Thread-safe initialization (use asyncio.Lock)
- [ ] Health check endpoint includes memory system status
- [ ] Integration test validates initialization

**Files:**

- `src/agentcore/memory/__init__.py`
- `src/agentcore/a2a_protocol/main.py` (update)
- `tests/integration/memory/test_initialization.py`

---

### MEM-022: End-to-End Agent Workflow Test

**Priority:** P0
**Type:** Story
**Effort:** 2 SP
**Dependencies:** MEM-019, MEM-021
**Owner:** QA Engineer / Backend Engineer

**Description:**
Create comprehensive integration test demonstrating full agent workflow with memory.

**Acceptance Criteria:**

- [ ] Test scenario: Multi-turn conversation with context accumulation
- [ ] Agent stores interactions after each turn
- [ ] Agent retrieves context before responding
- [ ] Working memory cached in Redis
- [ ] Episodic and semantic memories persisted in PostgreSQL
- [ ] Validates context efficiency (token reduction)
- [ ] Validates retrieval relevance (manual inspection)
- [ ] Performance assertions: <100ms retrieval, <500ms end-to-end

**Test Scenario:**

```python
async def test_multi_turn_workflow():
    # Turn 1: Initial query
    response1 = await agent.process_query(
        "What is the capital of France?",
        task_id="task-123"
    )
    # Validate response1, check memory stored

    # Turn 2: Related query (should use context)
    response2 = await agent.process_query(
        "What is its population?",
        task_id="task-123"
    )
    # Validate response2 uses context from Turn 1

    # Turn 3: Unrelated query
    response3 = await agent.process_query(
        "What is the capital of Spain?",
        task_id="task-123"
    )
    # Validate correct memories retrieved
```

**Files:**

- `tests/e2e/memory/test_agent_workflow.py`

---

## Phase 4: Advanced Features & Production Readiness (Weeks 5-6, 20 SP)

### MEM-023: Memory Pruning and Archival

**Priority:** P1
**Type:** Story
**Effort:** 5 SP
**Dependencies:** MEM-008
**Owner:** Backend Engineer

**Description:**
Implement automatic memory pruning with configurable strategies and optional S3 archival.

**Acceptance Criteria:**

- [ ] Automatic pruning when capacity exceeded (background task)
- [ ] Pruning strategies: least_relevant, oldest_first, least_accessed
- [ ] archive_memory() exports to S3 before deletion (optional)
- [ ] restore_memory() imports from S3 (optional)
- [ ] Scheduled pruning job (hourly via APScheduler)
- [ ] Metrics: pruned_count, archived_count, storage_bytes_saved
- [ ] Configuration: pruning_enabled, archival_enabled, s3_bucket
- [ ] Integration test with mock S3 (moto)
- [ ] Load test: prune 10K memories in <5 seconds

**Technical Details:**

```python
class PruningService:
    def __init__(
        self,
        repository: MemoryRepository,
        capacity_limits: dict[str, int],
        archival_service: ArchivalService | None = None
    ):
        self.repository = repository
        self.capacity_limits = capacity_limits
        self.archival = archival_service

    async def prune_if_needed(
        self,
        agent_id: str,
        memory_type: str,
        strategy: str = "least_relevant"
    ) -> int:
        current_count = await self.repository.count(agent_id, memory_type)
        capacity = self.capacity_limits[memory_type]

        if current_count <= capacity:
            return 0

        to_prune = current_count - capacity
        memories = await self._select_for_pruning(
            agent_id, memory_type, strategy, to_prune
        )

        # Archive if enabled
        if self.archival:
            await self.archival.archive_many(memories)

        # Delete
        await self.repository.delete_many([m.memory_id for m in memories])

        return len(memories)
```

**Files:**

- `src/agentcore/memory/pruning.py`
- `src/agentcore/memory/archival.py`
- `tests/unit/memory/test_pruning.py`
- `tests/integration/memory/test_archival.py`

---

### MEM-024: Performance Optimization and Monitoring

**Priority:** P0
**Type:** Story
**Effort:** 8 SP
**Dependencies:** MEM-021, MEM-023
**Owner:** Backend Engineer + DevOps

**Description:**
Optimize memory system performance and add comprehensive monitoring with Prometheus metrics.

**Acceptance Criteria:**

- [ ] PGVector IVFFlat index tuned (benchmark lists parameter)
- [ ] Database connection pooling optimized
- [ ] Query performance analysis (EXPLAIN ANALYZE)
- [ ] Read replica support for retrieval queries
- [ ] Prometheus metrics exported: memory_retrieval_latency_seconds, memory_cache_hit_rate, memory_storage_bytes
- [ ] Grafana dashboard with key metrics
- [ ] Load test: 100 RPS with <100ms p95 latency
- [ ] Scalability test: 1M memories per agent without degradation
- [ ] Performance regression tests in CI

**Metrics to Track:**

- Retrieval latency (p50, p95, p99)
- Cache hit rates (working memory, embeddings)
- Memory counts per type per agent
- Storage size (bytes)
- Pruning/archival rates
- API error rates

**Load Test Scenarios:**

1. High read: 90% retrieval, 10% store (100 RPS)
2. High write: 50% retrieval, 50% store (50 RPS)
3. Burst: 500 RPS for 30 seconds
4. Scale: 1000 agents × 1000 memories each

**Files:**

- `src/agentcore/memory/metrics.py`
- `load_tests/memory_load_test.py`
- `k8s/grafana/dashboards/memory.json`
- `docs/performance/memory_benchmarks.md`

---

### MEM-025: Documentation and Examples

**Priority:** P0
**Type:** Story
**Effort:** 3 SP
**Dependencies:** MEM-022, MEM-024
**Owner:** Technical Writer / Backend Engineer

**Description:**
Create comprehensive documentation and example implementations for memory system.

**Acceptance Criteria:**

- [ ] API reference: All JSON-RPC methods documented
- [ ] Architecture guide: Component diagram, data flow, design decisions
- [ ] Integration guide: Step-by-step agent integration
- [ ] Configuration guide: All settings explained with examples
- [ ] Performance guide: Optimization tips, capacity planning
- [ ] Example implementations: 3 agent types (conversational, task-oriented, learning)
- [ ] Troubleshooting guide: Common issues and solutions
- [ ] Migration guide: Adding memory to existing agents
- [ ] All code examples tested and working

**Documentation Structure:**

```
docs/memory/
├── README.md                    # Overview and quick start
├── architecture.md              # System design and patterns
├── api/
│   ├── jsonrpc_methods.md      # All memory.* methods
│   └── python_api.md           # MemoryManager, services
├── guides/
│   ├── integration.md          # Agent integration guide
│   ├── configuration.md        # Settings and tuning
│   └── performance.md          # Optimization guide
├── examples/
│   ├── conversational_agent.py
│   ├── task_agent.py
│   └── learning_agent.py
└── troubleshooting.md
```

**Files:**

- `docs/memory/*` (new directory)
- `examples/memory/*` (new directory)
- `README.md` (update with memory system section)

---

### MEM-026: Production Deployment and Migration

**Priority:** P0
**Type:** Story
**Effort:** 5 SP
**Dependencies:** MEM-024, MEM-025
**Owner:** DevOps + Backend Engineer

**Description:**
Prepare production deployment artifacts and create migration plan for existing agents.

**Acceptance Criteria:**

- [ ] Kubernetes manifests for memory system components
- [ ] PGVector extension installation automation (Helm chart)
- [ ] Database migration scripts tested on staging
- [ ] Redis cluster configuration for production scale
- [ ] Rollback plan documented and tested
- [ ] Feature flag for gradual rollout (memory_enabled)
- [ ] Migration script for existing agent data (if applicable)
- [ ] Runbook for common operational tasks
- [ ] Staging deployment validated (1 week soak test)
- [ ] Production deployment checklist

**Deployment Checklist:**

- [ ] PGVector extension installed on PostgreSQL
- [ ] Database migrations applied (memories table)
- [ ] Redis cluster healthy with memory capacity
- [ ] Memory service deployed and health checks passing
- [ ] Monitoring dashboards configured
- [ ] Alerts configured (high latency, low cache hit rate, storage capacity)
- [ ] Feature flag enabled for pilot agents
- [ ] Smoke tests passing in production
- [ ] Rollback plan validated
- [ ] On-call team briefed

**Files:**

- `k8s/memory/` (new directory with manifests)
- `helm/agentcore/templates/memory/` (Helm chart updates)
- `scripts/migrate_to_memory_system.py`
- `docs/deployment/memory_system_deployment.md`
- `docs/operations/memory_runbook.md`

---

## Critical Path Analysis

**Sequential Dependencies:**

1. **Phase 1 Foundation** (Weeks 1-2): All Phase 2-4 tasks depend on Phase 1 completion
   - Critical: MEM-002 → MEM-004 → MEM-005 → MEM-007 → MEM-008
   - Parallel: MEM-003, MEM-006 can start immediately
   - Parallel: MEM-009 can start after MEM-008

2. **Phase 2 JSON-RPC** (Week 3): All handlers depend on MEM-010 registration
   - MEM-010 must complete first
   - MEM-011 through MEM-016 can be done in parallel after MEM-010

3. **Phase 3 Integration** (Week 4): Depends on Phase 1 + Phase 2
   - MEM-017 (cache) can start after MEM-008
   - MEM-018 depends on MEM-017
   - MEM-019 (agent mixin) depends on MEM-017
   - MEM-020 (config) independent, can start anytime
   - MEM-021 (init) depends on MEM-008, MEM-017, MEM-020
   - MEM-022 (E2E test) depends on MEM-019, MEM-021

4. **Phase 4 Advanced** (Weeks 5-6): Depends on Phase 3 completion
   - MEM-023 (pruning) depends on MEM-008
   - MEM-024 (optimization) depends on MEM-021, MEM-023
   - MEM-025 (docs) depends on MEM-022, MEM-024
   - MEM-026 (deployment) depends on MEM-024, MEM-025

**Parallelization Opportunities:**

- Phase 1: MEM-003 + MEM-006 parallel to MEM-002 → MEM-004 sequence
- Phase 2: MEM-011 through MEM-016 all parallel after MEM-010
- Phase 3: MEM-017 + MEM-020 can start in parallel
- Phase 4: MEM-023 + MEM-025 can partially overlap with MEM-024

**Critical Path Duration:** 6 weeks (assumes 2-3 engineers with good parallelization)

---

## Team Allocation

**Week 1-2 (Phase 1):**

- Engineer 1: MEM-002, MEM-004, MEM-005 (database layer)
- Engineer 2: MEM-003, MEM-006, MEM-007 (models + services)
- Engineer 3: MEM-008, MEM-009 (manager + compression)

**Week 3 (Phase 2):**

- Engineer 1: MEM-010, MEM-011, MEM-012 (registration + core handlers)
- Engineer 2: MEM-013, MEM-014, MEM-015 (secondary handlers)
- Engineer 3: MEM-016 (stats) + integration test support

**Week 4 (Phase 3):**

- Engineer 1: MEM-017, MEM-018 (caching)
- Engineer 2: MEM-019, MEM-021 (agent integration)
- Engineer 3: MEM-020, MEM-022 (config + E2E tests)

**Week 5-6 (Phase 4):**

- Engineer 1: MEM-023 (pruning/archival)
- Engineer 2: MEM-024 (performance optimization)
- Engineer 3: MEM-025, MEM-026 (docs + deployment)
- DevOps: MEM-026 (deployment automation)

---

## Risk Mitigation

**Technical Risks:**

1. **PGVector Performance** (MEM-024)
   - Mitigation: Benchmark early, tune IVFFlat parameters, have read replica fallback
   - Contingency: Switch to external vector DB (Qdrant) if needed (3-day spike)

2. **Memory Coherence** (MEM-007)
   - Mitigation: Implement synthetic contradiction tests
   - Contingency: Add conflict resolution logic with user feedback loop

3. **Embedding API Rate Limits** (MEM-006)
   - Mitigation: Aggressive caching (MEM-018), rate limiting, circuit breaker
   - Contingency: Fall back to local SentenceTransformers model

**Process Risks:**

1. **Scope Creep**
   - Mitigation: Strict adherence to acceptance criteria, P0/P1/P2 prioritization
   - Phase 4 features (MEM-023 archival, advanced monitoring) are nice-to-have

2. **Integration Delays**
   - Mitigation: Early integration tests (MEM-022), weekly sync with agent teams
   - Buffer: 1 week of slack before hard deadline

---

## Success Criteria

**Functional Completeness:**

- [ ] All 25 story tickets completed and accepted
- [ ] All acceptance criteria met (100%)
- [ ] 90%+ test coverage maintained
- [ ] Zero P0/P1 bugs in production

**Performance Targets:**

- [ ] 80%+ reduction in context tokens (long sessions)
- [ ] 90%+ retrieval precision (manual evaluation on 100 test cases)
- [ ] <100ms p95 retrieval latency
- [ ] 1M+ memories per agent supported without degradation

**Production Readiness:**

- [ ] Deployed to staging for 1 week soak test
- [ ] Load tests passing at 2x expected traffic
- [ ] Monitoring dashboards operational
- [ ] Runbooks and documentation complete
- [ ] On-call team trained

**Adoption:**

- [ ] 3+ example agent implementations
- [ ] 5+ pilot agents using memory system in production
- [ ] Positive feedback from agent developers

---

## Appendix: Story Point Reference

**Story Point Scale (Fibonacci):**

- **1 SP**: 1-2 hours, simple config/registration
- **2 SP**: 2-4 hours, straightforward implementation (models, basic handlers)
- **3 SP**: 1 day, moderate complexity (repository, service integration)
- **5 SP**: 1-2 days, complex logic (orchestration, hybrid algorithms)
- **8 SP**: 2-3 days, significant integration (performance tuning, monitoring)
- **13 SP**: 3-5 days, high complexity/uncertainty (load testing, optimization)
- **21 SP**: 1+ weeks, should be broken down further

**Estimation Factors:**

- Technical complexity
- Integration points
- Testing requirements
- Documentation needs
- Uncertainty/unknowns
