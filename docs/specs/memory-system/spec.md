# Evolving Memory System Specification

## 1. Overview

### Purpose and Business Value

The Evolving Memory System provides agents with the ability to maintain, update, and utilize contextual information across multiple turns and interactions. Unlike static context windows that concatenate past messages, this system intelligently manages what information to retain, how to represent it compactly, and when to retrieve relevant memories to inform current decisions.

**Business Value:**

- **80% Context Reduction**: 5-10x reduction in context tokens for long sessions vs full history concatenation
- **25-30% Performance Improvement**: Enhanced success rate on multi-turn tasks through better context awareness
- **Cost Efficiency**: Reduced LLM API costs through selective context retrieval
- **Knowledge Accumulation**: Agents learn from past interactions and improve performance over time
- **Unbounded Interactions**: Support persistent agents with no practical limit on conversation depth

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Context Efficiency | 80%+ reduction in context tokens | `retrieved_tokens / full_history_tokens` |
| Retrieval Precision | 90%+ relevant memories | Manual annotation or LLM evaluation |
| Task Performance | +20% success rate on multi-turn tasks | `success_rate_with_memory / success_rate_without` |
| Memory Coherence | <5% contradictory retrievals | Contradiction detection rate |
| Retrieval Latency | <100ms (p95) | p95 retrieval latency |

### Target Users

- **Agent Developers**: Building memory-enabled agents for complex workflows
- **AgentCore Operators**: Managing agent systems with persistent context
- **End Users**: Benefiting from agents that remember past interactions and provide contextually relevant responses

**Source:** [docs/research/evolving-memory-system.md](../../research/evolving-memory-system.md)

---

## 2. Functional Requirements

### FR-1: Memory Storage and Encoding

The system shall provide capabilities to store and encode interaction data into structured memory records.

**FR-1.1** The system shall encode interactions into four memory types:

- Working Memory: Short-term memory for current task execution (2K-4K tokens capacity)
- Episodic Memory: Medium-term memory for recent interaction history (10-50 episodes)
- Semantic Memory: Long-term memory for accumulated knowledge (unbounded with pruning)
- Procedural Memory: Long-term memory for learned strategies and action patterns

**FR-1.2** The system shall create MemoryRecord objects containing:

- Unique memory_id, memory_type, timestamp
- Original content, condensed summary, vector embedding
- Extracted entities, facts, keywords
- Relationship metadata (related_memory_ids, parent_memory_id)
- Relevance tracking (relevance_score, access_count, last_accessed)
- Action-outcome pairs for procedural memory

**FR-1.3** The system shall generate vector embeddings for semantic search using configurable embedding models (OpenAI, Cohere, or local).

**User Story:** As an agent developer, I want interactions to be automatically encoded into structured memory records so that agents can retrieve relevant context efficiently.

### FR-2: Memory Retrieval

The system shall provide intelligent memory retrieval based on query relevance.

**FR-2.1** The system shall retrieve memories using hybrid search combining:

- Embedding similarity (cosine distance)
- Temporal relevance (recency decay)
- Access frequency (interaction count)
- Explicit relationships (memory links)

**FR-2.2** The system shall support filtered retrieval by:

- Memory type (working, episodic, semantic, procedural)
- Task ID (for task-scoped memories)
- Time range (timestamp filtering)
- Entity/keyword matching

**FR-2.3** The system shall return top-k memories ranked by importance score (configurable k, default 5).

**FR-2.4** The system shall track memory access for importance weighting (update access_count, last_accessed).

**User Story:** As an agent, I want to retrieve the most relevant memories for my current query so that I can provide contextually informed responses.

### FR-3: Context Formatting

The system shall format retrieved memories into LLM-ready context strings.

**FR-3.1** The system shall generate formatted context with sections:

- Current Task Context (working memory for active task)
- Relevant Past Interactions (episodic memories with timestamps)
- Relevant Knowledge (semantic facts and insights)

**FR-3.2** The system shall truncate context to specified token limits using hierarchical summarization.

**FR-3.3** The system shall compress memory sequences when full content exceeds token budget.

**User Story:** As an agent runtime, I want formatted context strings that fit within token limits so that I can augment LLM prompts with relevant memory.

### FR-4: Memory Updates

The system shall support updating existing memory records.

**FR-4.1** The system shall allow updating memory fields:

- Content corrections (fixing incorrect information)
- Context additions (adding new related information)
- Relevance score adjustments
- Relationship modifications

**FR-4.2** The system shall maintain update history for audit trails.

**User Story:** As an agent developer, I want to update memories when new information corrects or supplements existing records so that agents maintain accurate context.

### FR-5: Memory Pruning

The system shall maintain bounded memory capacity through intelligent pruning.

**FR-5.1** The system shall enforce capacity limits per memory layer (configurable).

**FR-5.2** The system shall support pruning strategies:

- Least Relevant: Remove memories with lowest importance scores
- Oldest First: Remove oldest memories by timestamp
- Frequency Based: Remove rarely accessed memories

**FR-5.3** The system shall archive pruned memories to cold storage (S3 or equivalent) before deletion.

**User Story:** As an AgentCore operator, I want automatic memory pruning so that the system maintains performance without unbounded storage growth.

### FR-6: JSON-RPC Methods

The system shall expose memory operations via JSON-RPC 2.0 protocol.

**FR-6.1** `memory.store`: Store an interaction as memory

- Params: `interaction` (Interaction object)
- Returns: `{success: bool, memory_id: str}`

**FR-6.2** `memory.retrieve`: Retrieve relevant memories

- Params: `query` (str), `k` (int, default 5), `memory_types` (list[str])
- Returns: `{memories: list[MemorySummary]}`

**FR-6.3** `memory.get_context`: Get formatted context for query

- Params: `query` (str), `task_id` (str | null), `max_tokens` (int, default 2000)
- Returns: `{context: str, token_count: int}`

**FR-6.4** `memory.update`: Update existing memory

- Params: `memory_id` (str), `updates` (dict)
- Returns: `{success: bool}`

**FR-6.5** `memory.prune`: Manually trigger pruning

- Params: `memory_layer` (str), `strategy` (str, default "least_relevant")
- Returns: `{pruned_count: int}`

**User Story:** As an agent developer, I want to interact with the memory system via JSON-RPC so that I can integrate memory capabilities into agents using standard protocols.

---

## 3. Non-Functional Requirements

### NFR-1: Performance

**NFR-1.1** Memory retrieval shall complete within 100ms at p95 latency.

**NFR-1.2** Context formatting shall complete within 50ms for 2K token contexts.

**NFR-1.3** Memory encoding shall complete within 200ms per interaction (including embedding generation).

**NFR-1.4** The system shall support 100+ concurrent retrieval requests without degradation.

### NFR-2: Scalability

**NFR-2.1** The system shall scale to 1M+ memory records per agent without performance degradation.

**NFR-2.2** Vector search shall maintain sub-100ms latency at 1M+ vector scale using approximate nearest neighbor (ANN) indexes.

**NFR-2.3** The system shall support horizontal scaling through:

- Database read replicas for retrieval operations
- Redis caching for working memory
- Distributed vector store (sharding/replication)

### NFR-3: Accuracy

**NFR-3.1** Memory retrieval precision shall be 90%+ (relevant memories in top-k).

**NFR-3.2** Memory coherence violations shall be <5% (contradictory memories retrieved together).

**NFR-3.3** Embedding consistency shall be maintained (same content → same embedding ±1% variance).

### NFR-4: Reliability

**NFR-4.1** Memory write operations shall be durable (persist to PostgreSQL).

**NFR-4.2** The system shall gracefully handle embedding service failures with retries and exponential backoff.

**NFR-4.3** Memory retrieval failures shall not block agent operations (fallback to empty context).

### NFR-5: Storage Efficiency

**NFR-5.1** Memory records shall be compressed (summaries vs full content) to reduce storage by 60%+.

**NFR-5.2** Working memory shall use in-memory cache (Redis) with TTL for fast access.

**NFR-5.3** Archived memories shall use cold storage (S3) with 90% cost reduction vs database storage.

### NFR-6: Security

**NFR-6.1** Memory access shall be scoped by agent_id (agents cannot access other agents' memories).

**NFR-6.2** Memory operations shall require JWT authentication via A2A protocol.

**NFR-6.3** Sensitive data in memories shall be encrypted at rest (database-level encryption).

---

## 4. Features & Flows

### Feature 1: Core Memory Manager (P0)

**Description:** Central MemoryManager class implementing encode/store/retrieve/update/prune operations.

**Components:**

- `agentcore/memory/manager.py`: MemoryManager class
- `agentcore/memory/models.py`: MemoryRecord, Interaction Pydantic models
- `agentcore/memory/encoding.py`: Memory encoding logic (summarization, entity extraction)
- `agentcore/memory/retrieval.py`: Retrieval algorithms (hybrid search, importance scoring)
- `agentcore/memory/storage.py`: Storage backend adapters (PostgreSQL, Redis, Vector DB)
- `agentcore/memory/compression.py`: Context compression utilities

**User Flow:**

1. Agent interaction occurs (query + actions + outcome)
2. MemoryManager.add_interaction() encodes interaction
3. System generates summary, extracts entities/facts, creates embedding
4. MemoryRecord stored in appropriate layer(s) (episodic + semantic)
5. Working memory updated if task_id present
6. Automatic pruning triggered if capacity exceeded

### Feature 2: Database Schema and Persistence (P0)

**Description:** PostgreSQL schema with PGVector extension for vector storage and similarity search.

**Schema:**

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
    relevance_score FLOAT DEFAULT 1.0,
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_memories_agent_type ON memories(agent_id, memory_type);
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_memories_task ON memories(task_id) WHERE task_id IS NOT NULL;
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**User Flow:**

1. Alembic migration creates schema with PGVector extension
2. MemoryRepository provides async CRUD operations
3. Vector similarity queries use PGVector's `<=>` operator
4. Indexes ensure fast retrieval by agent_id, type, timestamp

### Feature 3: JSON-RPC Integration (P0)

**Description:** JSON-RPC 2.0 methods for memory operations exposed via `/api/v1/jsonrpc` endpoint.

**Components:**

- `agentcore/memory/jsonrpc.py`: JSON-RPC method handlers
- Methods: `memory.store`, `memory.retrieve`, `memory.get_context`, `memory.update`, `memory.prune`

**User Flow:**

1. Agent sends JSON-RPC request: `{"method": "memory.retrieve", "params": {"query": "...", "k": 5}}`
2. JsonRpcProcessor routes to memory.retrieve handler
3. MemoryManager.retrieve_memories() performs hybrid search
4. Response returns: `{"result": {"memories": [...]}}`

**Input/Output:**

- Input: JSON-RPC request with method name + params (validated by Pydantic)
- Output: JSON-RPC response with result or error (JsonRpcErrorCode on failure)

### Feature 4: Vector Embeddings and Search (P0)

**Description:** Integration with embedding models and vector similarity search.

**Components:**

- `agentcore/memory/embeddings.py`: Embedding model interface (OpenAI, Cohere, local)
- `agentcore/memory/vector_store.py`: Vector store interface (PGVector, Pinecone, Qdrant, Weaviate)

**Configuration:**

```python
# config.py
EMBEDDING_MODEL = "openai"  # openai | cohere | local
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
VECTOR_STORE = "pgvector"  # pgvector | pinecone | qdrant | weaviate
```

**User Flow:**

1. MemoryManager encodes interaction summary
2. EmbeddingModel.embed(text) generates vector
3. Vector stored in memories.embedding column
4. Retrieval performs cosine similarity search: `ORDER BY embedding <=> query_embedding LIMIT k`

### Feature 5: Context Compression (P1)

**Description:** Intelligent compression of memory sequences to fit token budgets.

**Strategies:**

- Hierarchical summarization: LLM-generated summary of memory sequence
- Importance weighting: Prioritize high-importance memories, summarize low-importance
- Sliding window: Recent memories in full, older memories summarized

**User Flow:**

1. MemoryManager.get_relevant_context() retrieves memories
2. Total token count exceeds max_tokens limit
3. compress_memory_sequence() generates hierarchical summary
4. Formatted context fits within token budget

### Feature 6: Working Memory Cache (P1)

**Description:** Redis-based cache for fast access to current task working memory.

**Components:**

- Redis key format: `memory:working:{task_id}`
- TTL: 1 hour (cleared when task completes or times out)
- Value: JSON-serialized MemoryRecord(s) for task

**User Flow:**

1. Agent starts task with task_id
2. Working memory stored in Redis: `SET memory:working:{task_id} {...} EX 3600`
3. Fast retrieval: `GET memory:working:{task_id}` (sub-10ms)
4. Task completion clears cache: `DEL memory:working:{task_id}`

### Feature 7: Memory Pruning and Archival (P2)

**Description:** Automatic and manual pruning with archival to cold storage.

**Components:**

- `agentcore/memory/pruning.py`: Pruning strategies and execution
- `agentcore/memory/archival.py`: S3 archival for pruned memories

**User Flow:**

1. Memory layer reaches capacity limit (e.g., episodic > 50 records)
2. Pruning triggered with "least_relevant" strategy
3. Calculate importance scores for all memories
4. Archive bottom 20% to S3: `s3://agentcore-memory-archive/{agent_id}/{memory_id}.json`
5. Delete from PostgreSQL
6. Return pruned_count

### Feature 8: Agent Integration (P1)

**Description:** MemoryEnabledAgent base class with automatic memory management.

**Components:**

- `agentcore/agents/memory_enabled_agent.py`: Base class for agents with memory

**User Flow:**

1. Agent subclasses MemoryEnabledAgent
2. Override process_query() method
3. Automatic context retrieval via self.memory.get_relevant_context()
4. Automatic interaction storage after query completion
5. Memory-augmented prompt generation

**Example:**

```python
class MyAgent(MemoryEnabledAgent):
    async def process_query(self, query: str, task_id: str) -> str:
        context = await self.memory.get_relevant_context(query, task_id)
        response = await self.llm.generate(f"Context:\n{context}\n\nQuery: {query}")
        await self._store_interaction(query, response, task_id)
        return response
```

---

## 5. Acceptance Criteria

### AC-1: Context Efficiency

- **Given** an agent with 50-turn conversation history (25K tokens total)
- **When** memory system retrieves relevant context
- **Then** context size shall be ≤5K tokens (80% reduction)
- **And** retrieved context contains all information needed for current query

### AC-2: Retrieval Precision

- **Given** a test dataset of 100 queries with labeled relevant memories
- **When** memory.retrieve is called with k=5
- **Then** precision@5 shall be ≥90% (ground truth evaluation)
- **And** at least one highly relevant memory in top-3 for 95% of queries

### AC-3: Task Performance Improvement

- **Given** a benchmark of 50 multi-turn tasks (3+ steps)
- **When** agents use memory system vs baseline (no memory)
- **Then** success rate improvement shall be ≥20%
- **And** average turns-to-completion reduced by ≥15%

### AC-4: Memory Coherence

- **Given** 1000 memory retrieval operations
- **When** checking for contradictory memories in results
- **Then** contradiction rate shall be <5%
- **And** no critical contradictions (high-importance conflicting facts)

### AC-5: Retrieval Latency

- **Given** memory database with 100K records per agent
- **When** executing memory.retrieve with k=5
- **Then** p50 latency shall be <50ms
- **And** p95 latency shall be <100ms
- **And** p99 latency shall be <200ms

### AC-6: Storage Scalability

- **Given** 10 agents with 100K memories each (1M total)
- **When** system is under normal load (10 queries/sec)
- **Then** retrieval latency remains <100ms p95
- **And** storage usage is ≤50GB (with compression)
- **And** no degradation in retrieval accuracy

### AC-7: JSON-RPC Integration

- **Given** an agent client using JSON-RPC protocol
- **When** calling memory.store, memory.retrieve, memory.get_context
- **Then** all methods return valid JSON-RPC 2.0 responses
- **And** error cases return appropriate JsonRpcErrorCode
- **And** A2A context (trace_id, agent_id) is preserved

### AC-8: Memory Persistence

- **Given** memories stored via memory.store
- **When** AgentCore service restarts
- **Then** all memories remain accessible (durability)
- **And** working memory is restored from Redis (if TTL not expired)
- **And** no data loss occurs

---

## 6. Dependencies

### Technical Dependencies

**Required Infrastructure:**

- PostgreSQL 14+ with PGVector extension (vector similarity search)
- Redis 6+ (working memory cache, TTL support)
- Python 3.12+ (built-in generics, modern typing)

**External Services:**

- Embedding Model API: OpenAI (text-embedding-3-small) or Cohere or local SentenceTransformers
- Optional: External vector database (Pinecone, Qdrant, Weaviate) if not using PGVector
- Optional: S3-compatible storage for memory archival

**Python Packages:**

- `pgvector`: PostgreSQL vector extension bindings
- `openai` or `cohere`: Embedding generation
- `sentence-transformers`: Local embedding model (optional)
- `redis`: Redis client for caching
- `boto3`: S3 client for archival (optional)

### AgentCore Component Dependencies

**Required Components:**

- Database Layer (`agentcore.database`): SessionLocal, repositories
- JSON-RPC Handler (`agentcore.a2a_protocol.services.jsonrpc_handler`): Method registration
- Task Management (`agentcore.a2a_protocol.services.task_manager`): task_id references
- Configuration (`agentcore.a2a_protocol.config`): Settings management

**Optional Components:**

- Agent Manager (`agentcore.a2a_protocol.services.agent_manager`): For agent-scoped memories
- Event Manager (`agentcore.a2a_protocol.services.event_manager`): For memory update events

### External Integrations

**Embedding Services:**

- OpenAI API: `https://api.openai.com/v1/embeddings` (requires API key)
- Cohere API: `https://api.cohere.ai/v1/embed` (requires API key)
- Local: SentenceTransformers models (no external dependency)

**Vector Stores (if not using PGVector):**

- Pinecone: Cloud-hosted vector database
- Qdrant: Self-hosted or cloud vector database
- Weaviate: Self-hosted or cloud vector database

### Deployment Dependencies

**Database Migration:**

- Alembic migration to create `memories` table with PGVector extension
- PGVector extension must be installed in PostgreSQL: `CREATE EXTENSION vector;`

**Configuration:**

```bash
# .env additions
EMBEDDING_MODEL=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
OPENAI_API_KEY=sk-...
VECTOR_STORE=pgvector
MEMORY_WORKING_TTL=3600
MEMORY_EPISODIC_CAPACITY=50
MEMORY_PRUNING_STRATEGY=least_relevant
MEMORY_ARCHIVAL_ENABLED=true
MEMORY_ARCHIVAL_S3_BUCKET=agentcore-memory-archive
```

### Related Components

**Depends On:**

- Task Management System (for task_id references and task lifecycle)
- Agent Registration System (for agent_id scoping and authentication)

**Depended On By:**

- Future: Multi-tool Integration (agents use memory for tool selection)
- Future: Bounded Context Reasoning (memory provides domain context)
- Future: Flow-based Optimization (memory tracks successful flows)

---

## Implementation Phases

### Phase 1: Core Memory System (Weeks 1-2)

- Implement MemoryManager, models, encoding, retrieval, storage
- Database schema with PGVector
- Repository layer and async CRUD
- Unit tests with 90%+ coverage

### Phase 2: JSON-RPC Integration (Week 3)

- JSON-RPC method handlers
- Integration with JsonRpcProcessor
- A2A context handling
- Integration tests for all methods

### Phase 3: Agent Integration (Week 4)

- MemoryEnabledAgent base class
- Automatic context retrieval and storage
- Working memory cache (Redis)
- End-to-end tests with sample agents

### Phase 4: Advanced Features (Weeks 5-6)

- Context compression and summarization
- Memory pruning and archival
- Performance optimization (indexing, caching)
- Load testing and benchmarking

---

**Generated from:** [docs/research/evolving-memory-system.md](../../research/evolving-memory-system.md)
