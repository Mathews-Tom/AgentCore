# Memory System Enhancement Analysis
## Cognee Integration Concepts for AgentCore

**Date:** 2025-11-06
**Branch:** feature/quick-wins-completion
**Source:** https://github.com/topoteretes/cognee (7.9k stars)

---

## Executive Summary

Cognee is an advanced AI memory system that combines **vector search** with **graph databases** to create persistent, dynamic memory for AI agents. Their ECL (Extract, Cognify, Load) pipeline architecture and hybrid storage approach offer valuable patterns for AgentCore's memory-system component (36 tickets, 0% complete).

**Key Value Proposition:** Replace traditional RAG with interconnected knowledge graphs that maintain semantic relationships and context across agent interactions.

---

## 1. Core Architecture Concepts

### 1.1 ECL Pipeline Pattern

```
Extract → Cognify → Load
   ↓         ↓        ↓
Ingest   Generate   Store
Data     Knowledge  with
         Graphs     Search
```

**Recommendation for AgentCore:**
- Adopt similar 3-phase pipeline architecture
- Phase 1 (Extract): Multi-source data ingestion
- Phase 2 (Process): Entity extraction + relationship mapping
- Phase 3 (Load): Hybrid storage (vector + graph)

### 1.2 Hybrid Storage Architecture

**Cognee's Approach:**
- **Vector DB**: Semantic/meaning-based search
- **Graph DB**: Entity relationships and context
- **Memify Operation**: Memory optimization algorithms

**AgentCore Implementation:**
```python
# Proposed architecture
memory_system/
├── storage/
│   ├── vector/          # Semantic search layer
│   │   ├── chromadb/
│   │   ├── lancedb/
│   │   └── pgvector/
│   ├── graph/           # Relationship layer
│   │   ├── neo4j/
│   │   └── neptune/
│   └── hybrid/          # Combined queries
├── pipeline/
│   ├── extract.py       # Data ingestion
│   ├── process.py       # Knowledge graph generation
│   └── load.py          # Storage operations
└── api/
    ├── query.py         # Search interface
    ├── update.py        # Incremental updates
    └── memify.py        # Memory optimization
```

---

## 2. Database Integration Strategy

### 2.1 Vector Database Support

**Cognee Supports:**
1. **ChromaDB** - Simple, embeddable vector store
2. **LanceDB** - Serverless vector DB for ML
3. **PGVector** - PostgreSQL extension (already in AgentCore!)

**Recommendation:**
- **Start with PGVector** (AgentCore already uses PostgreSQL)
- Add ChromaDB for in-memory/embedded scenarios
- LanceDB for high-performance production use

**Benefits:**
- ✅ Reuse existing PostgreSQL infrastructure
- ✅ Lower operational complexity
- ✅ ACID guarantees with PGVector

### 2.2 Graph Database Integration

**Cognee Uses:**
- Neo4j for relationship mapping
- AWS Neptune Analytics for hybrid queries

**Recommendation for AgentCore:**
```python
# Priority order
1. Neo4j (mature, feature-rich)
   - Cypher query language
   - Native graph storage
   - Excellent tooling

2. AWS Neptune (cloud-native)
   - Managed service
   - Gremlin/SPARQL support
   - Auto-scaling

3. PostgreSQL with Apache AGE (cost-effective)
   - Reuse existing PostgreSQL
   - Graph extension
   - Lower infrastructure cost
```

### 2.3 Database Abstraction Layer

**Pattern from Cognee:**
```python
# Abstract interface for multiple backends
class VectorDBInterface:
    async def create_index(...)
    async def insert(...)
    async def search(...)
    async def delete(...)

# Factory pattern for engine creation
def create_vector_engine(provider: str) -> VectorDBInterface:
    if provider == "pgvector":
        return PGVectorEngine()
    elif provider == "chromadb":
        return ChromaDBEngine()
    # ...
```

**AgentCore Implementation:**
```python
# memory_system/storage/base.py
from abc import ABC, abstractmethod
from typing import Any

class MemoryStore(ABC):
    """Abstract base for all memory storage backends."""

    @abstractmethod
    async def store(self, data: dict[str, Any]) -> str:
        """Store memory entry, return ID."""
        pass

    @abstractmethod
    async def query(self, query: str, filters: dict) -> list[dict]:
        """Search memory."""
        pass

    @abstractmethod
    async def update(self, id: str, data: dict) -> bool:
        """Update existing memory."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Remove memory entry."""
        pass

# memory_system/storage/factory.py
from .vector.pgvector import PGVectorStore
from .graph.neo4j import Neo4jStore
from .hybrid.combined import HybridStore

def create_memory_store(
    storage_type: str,
    **config
) -> MemoryStore:
    """Factory for memory storage backends."""
    stores = {
        "vector": PGVectorStore,
        "graph": Neo4jStore,
        "hybrid": HybridStore,
    }
    return stores[storage_type](**config)
```

---

## 3. Memory Management Patterns

### 3.1 Memory Lifecycle

**Cognee's Operations:**
- `add()` - Ingest new data
- `cognify()` - Generate knowledge graph
- `memify()` - Optimize memory structure
- `search()` - Query memory
- `update()` - Incremental updates

**AgentCore Adaptation:**
```python
# memory_system/service.py
class MemoryService:
    """Agent memory management service."""

    async def store_interaction(
        self,
        agent_id: str,
        interaction: dict,
        context: dict | None = None
    ) -> str:
        """Store agent interaction in memory.

        1. Extract entities and relationships
        2. Generate embeddings
        3. Store in vector + graph DB
        4. Return memory ID
        """
        # Extract phase
        entities = await self._extract_entities(interaction)
        relationships = await self._extract_relationships(interaction)

        # Process phase (cognify)
        embedding = await self._generate_embedding(interaction)
        knowledge_graph = await self._build_graph(entities, relationships)

        # Load phase
        vector_id = await self.vector_store.insert(embedding, metadata)
        graph_id = await self.graph_store.add_nodes(knowledge_graph)

        return f"{vector_id}:{graph_id}"

    async def query_memory(
        self,
        agent_id: str,
        query: str,
        k: int = 5
    ) -> list[dict]:
        """Hybrid search combining semantic + graph."""
        # Vector search for semantic similarity
        vector_results = await self.vector_store.search(query, k=k*2)

        # Graph traversal for contextual relationships
        entity_ids = [r["entity_id"] for r in vector_results]
        graph_context = await self.graph_store.get_context(entity_ids)

        # Merge and rank results
        return self._merge_results(vector_results, graph_context, k)

    async def optimize_memory(self, agent_id: str) -> None:
        """Memify operation - optimize agent memory.

        1. Identify frequently accessed patterns
        2. Pre-compute common queries
        3. Prune low-value memories
        4. Update graph relationships
        """
        # Memory optimization algorithms
        await self._identify_memory_patterns(agent_id)
        await self._consolidate_similar_memories(agent_id)
        await self._prune_old_memories(agent_id)
```

### 3.2 Incremental Updates

**Key Pattern from Cognee:**
- Don't rebuild entire knowledge graph
- Update only affected nodes/relationships
- Maintain temporal consistency

**Implementation:**
```python
async def update_memory(
    self,
    memory_id: str,
    updates: dict
) -> bool:
    """Incremental memory update."""
    vector_id, graph_id = memory_id.split(":")

    # Update vector embedding if content changed
    if "content" in updates:
        new_embedding = await self._generate_embedding(updates["content"])
        await self.vector_store.update(vector_id, new_embedding)

    # Update graph relationships if entities changed
    if "entities" in updates or "relationships" in updates:
        await self.graph_store.update_node(graph_id, updates)

    return True
```

---

## 4. Modular Pipeline Architecture

### 4.1 Task-Based Processing

**Cognee's Pattern:**
```python
# Extensible task system
tasks/
├── extraction/
│   ├── entity_extraction.py
│   ├── relationship_extraction.py
│   └── metadata_extraction.py
├── chunking/
│   ├── text_splitter.py
│   ├── semantic_chunker.py
│   └── recursive_chunker.py
└── embeddings/
    ├── openai_embeddings.py
    ├── local_embeddings.py
    └── custom_embeddings.py
```

**AgentCore Adaptation:**
```python
# memory_system/tasks/
from abc import ABC, abstractmethod

class MemoryTask(ABC):
    """Base class for memory processing tasks."""

    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        """Execute task and return results."""
        pass

# Example task
class EntityExtractionTask(MemoryTask):
    """Extract entities from text using LLM."""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def execute(self, input_data: dict) -> dict:
        text = input_data["text"]

        # Use AgentCore's existing LLM service
        response = await self.llm.complete(
            prompt=f"Extract entities from: {text}",
            model="gpt-4"
        )

        return {
            "entities": self._parse_entities(response),
            "original_text": text
        }

# Pipeline composition
class MemoryPipeline:
    """Compose tasks into processing pipeline."""

    def __init__(self):
        self.tasks: list[MemoryTask] = []

    def add_task(self, task: MemoryTask) -> "MemoryPipeline":
        self.tasks.append(task)
        return self

    async def execute(self, input_data: dict) -> dict:
        """Execute all tasks in sequence."""
        result = input_data
        for task in self.tasks:
            result = await task.execute(result)
        return result

# Usage
pipeline = (
    MemoryPipeline()
    .add_task(EntityExtractionTask(llm_service))
    .add_task(RelationshipExtractionTask(llm_service))
    .add_task(GraphGenerationTask(graph_store))
    .add_task(EmbeddingGenerationTask(embedding_service))
    .add_task(StorageTask(vector_store, graph_store))
)

result = await pipeline.execute({"text": "User interaction..."})
```

---

## 5. Integration with AgentCore Services

### 5.1 LLM Service Integration

**Cognee has:** LLMGateway abstraction
**AgentCore has:** llm_gateway (17 files, 100% complete)

**Integration Point:**
```python
# Reuse existing LLM gateway
from agentcore.llm_gateway import LLMService

class MemoryLLMService:
    """Memory-specific LLM operations using AgentCore's gateway."""

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def extract_entities(self, text: str) -> list[dict]:
        """Extract entities using structured output."""
        response = await self.llm.complete(
            messages=[{
                "role": "system",
                "content": "You are an entity extraction expert."
            }, {
                "role": "user",
                "content": f"Extract entities: {text}"
            }],
            model="gpt-4",
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
```

### 5.2 A2A Protocol Integration

**Memory as A2A JSON-RPC Methods:**
```python
# memory_system/jsonrpc/memory_jsonrpc.py
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

@register_jsonrpc_method("memory.store")
async def store_memory(request: JsonRpcRequest) -> dict:
    """Store memory via JSON-RPC."""
    agent_id = request.params["agent_id"]
    data = request.params["data"]

    memory_id = await memory_service.store_interaction(agent_id, data)

    return {
        "memory_id": memory_id,
        "stored_at": datetime.now(UTC).isoformat()
    }

@register_jsonrpc_method("memory.query")
async def query_memory(request: JsonRpcRequest) -> dict:
    """Query memory via JSON-RPC."""
    agent_id = request.params["agent_id"]
    query = request.params["query"]
    k = request.params.get("top_k", 5)

    results = await memory_service.query_memory(agent_id, query, k)

    return {
        "results": results,
        "query": query,
        "count": len(results)
    }

@register_jsonrpc_method("memory.optimize")
async def optimize_memory(request: JsonRpcRequest) -> dict:
    """Optimize memory (memify operation)."""
    agent_id = request.params["agent_id"]

    await memory_service.optimize_memory(agent_id)

    return {
        "status": "optimized",
        "agent_id": agent_id
    }
```

### 5.3 Coordination Service Integration

**Use existing coordination for memory routing:**
```python
# Route memory queries to optimal storage backend
from agentcore.a2a_protocol.services.coordination_service import CoordinationService

class MemoryRouter:
    """Route memory operations based on load/capacity."""

    def __init__(self, coord_service: CoordinationService):
        self.coordinator = coord_service

    async def route_query(self, query: dict) -> str:
        """Select optimal memory backend."""
        # Get coordination state for memory backends
        states = await self.coordinator.get_coordination_states(
            capability="memory.storage"
        )

        # Use multi-objective optimization
        best_backend = await self.coordinator.select_agent(
            states,
            criteria=["low_load", "high_quality"]
        )

        return best_backend.agent_id
```

---

## 6. Recommended Implementation Phases

### Phase 1: Foundation (Tickets MEM-001 to MEM-010)
**Goal:** Basic memory storage and retrieval

1. **MEM-001:** Database schema design
   - PGVector extension setup
   - Memory tables (interactions, embeddings, metadata)

2. **MEM-002:** Vector storage implementation
   - PGVector adapter
   - Embedding generation (reuse llm_gateway)
   - Basic CRUD operations

3. **MEM-003:** Graph storage foundation
   - Choose graph DB (Neo4j or PostgreSQL + AGE)
   - Entity and relationship models
   - Graph adapter interface

4. **MEM-004:** Memory service core
   - MemoryService class
   - store_interaction()
   - query_memory()

5. **MEM-005:** JSON-RPC integration
   - memory.store endpoint
   - memory.query endpoint
   - A2A protocol compliance

6. **MEM-006:** Configuration management
   - Database connection settings
   - Vector dimension config
   - Graph connection config

7. **MEM-007:** Error handling
   - Memory-specific exceptions
   - Retry logic for storage failures
   - Validation

8. **MEM-008:** Logging and observability
   - Prometheus metrics (like coordination service)
   - Structured logging
   - Query performance tracking

9. **MEM-009:** Unit tests
   - Storage adapter tests
   - Service layer tests
   - Mock backends

10. **MEM-010:** Integration tests
    - End-to-end storage → retrieval
    - Multi-backend scenarios
    - Performance validation

### Phase 2: ECL Pipeline (Tickets MEM-011 to MEM-020)
**Goal:** Knowledge graph generation

11. **MEM-011:** Entity extraction task
12. **MEM-012:** Relationship extraction task
13. **MEM-013:** Pipeline orchestration
14. **MEM-014:** Chunking strategies
15. **MEM-015:** Graph generation
16. **MEM-016:** Incremental updates
17. **MEM-017:** Pipeline configuration
18. **MEM-018:** Task registry
19. **MEM-019:** Pipeline testing
20. **MEM-020:** Performance optimization

### Phase 3: Advanced Features (Tickets MEM-021 to MEM-030)
**Goal:** Memory optimization and intelligence

21. **MEM-021:** Memify operation (memory optimization)
22. **MEM-022:** Temporal memory management
23. **MEM-023:** Memory consolidation
24. **MEM-024:** Context window management
25. **MEM-025:** Multi-agent shared memory
26. **MEM-026:** Memory versioning
27. **MEM-027:** Hybrid search optimization
28. **MEM-028:** Memory analytics
29. **MEM-029:** Export/import functionality
30. **MEM-030:** Memory visualization

### Phase 4: Enhancements (Tickets MEM-031 to MEM-036)
**Goal:** Production readiness

31. **MEM-031:** Performance benchmarks
32. **MEM-032:** Security and access control
33. **MEM-033:** Distributed memory (optional)
34. **MEM-034:** Documentation and examples
35. **MEM-035:** Migration tools
36. **MEM-036:** Production monitoring

---

## 7. Technology Stack Recommendations

### Core Dependencies
```toml
[dependencies]
# Vector database
pgvector = "^0.3.2"  # PostgreSQL vector extension
chromadb = "^0.5.0"  # Embedded vector DB (optional)

# Graph database
neo4j = "^5.24.0"  # Graph database driver
py2neo = "^2021.2.4"  # Alternative Neo4j client

# Embeddings
sentence-transformers = "^3.1.0"  # Local embeddings
openai = "^1.54.0"  # Already in AgentCore

# Graph processing
networkx = "^3.3"  # Graph algorithms
igraph = "^0.11.6"  # Fast graph processing

# Utilities
asyncpg = "^0.29.0"  # Already in AgentCore for PostgreSQL
pydantic = "^2.9.0"  # Already in AgentCore
```

### Database Setup
```sql
-- Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memory storage table
CREATE TABLE agent_memories (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI ada-002 dimension
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

-- Vector similarity index
CREATE INDEX ON agent_memories
USING ivfflat (embedding vector_cosine_ops);

-- Metadata index
CREATE INDEX ON agent_memories USING GIN (metadata);

-- Entity table
CREATE TABLE memory_entities (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID REFERENCES agent_memories(memory_id),
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    metadata JSONB
);

-- Relationships table
CREATE TABLE memory_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID REFERENCES memory_entities(entity_id),
    target_entity_id UUID REFERENCES memory_entities(entity_id),
    relationship_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata JSONB
);
```

---

## 8. Key Learnings from Cognee

### What to Adopt ✅

1. **Hybrid Storage Pattern**
   - Vector for semantic search
   - Graph for relationships
   - Combined queries for context

2. **ECL Pipeline Architecture**
   - Modular, extensible design
   - Task-based processing
   - Clean separation of concerns

3. **Memify Operation**
   - Continuous memory optimization
   - Pattern detection
   - Memory consolidation

4. **Abstraction Layers**
   - Multiple backend support
   - Factory pattern for engines
   - Interface-driven design

5. **Async-First**
   - All operations are async
   - Better scalability
   - Matches AgentCore patterns

### What to Avoid/Adapt ❌

1. **Don't Create Separate CLI**
   - Use existing A2A JSON-RPC
   - Integrate with agentcore_cli

2. **Don't Duplicate LLM Gateway**
   - Reuse AgentCore's llm_gateway
   - Avoid vendor lock-in

3. **Start Simpler than Cognee**
   - They support many vector DBs
   - Start with PGVector only
   - Add others based on need

4. **Leverage Existing Infrastructure**
   - Use PostgreSQL (already deployed)
   - Reuse coordination service
   - Integrate with observability

---

## 9. Success Metrics

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Vector search latency | <50ms | p95 query time |
| Graph traversal latency | <100ms | p95 relationship queries |
| Memory storage throughput | >100 ops/sec | Writes per second |
| Memory retrieval accuracy | >90% | Relevance@5 |
| Storage efficiency | <10MB/agent/day | Average storage growth |

### Quality Targets

| Metric | Target |
|--------|--------|
| Test coverage | >90% |
| Entity extraction accuracy | >85% |
| Relationship accuracy | >80% |
| Uptime | >99.9% |

---

## 10. Next Steps

### Immediate Actions

1. **Review and Approve Architecture** (This Document)
   - Stakeholder review
   - Technical validation
   - Budget approval (if cloud DBs)

2. **Create MEM Tickets** (36 tickets)
   - Use this document as spec
   - Break down into phases
   - Assign story points

3. **Setup Development Environment**
   - Install PGVector extension
   - Setup Neo4j (local/Docker)
   - Configure test databases

4. **Start Phase 1 Implementation**
   - Begin with MEM-001 (schema)
   - Use `/sage.implement` workflow
   - Target 2-3 tickets per week

### Timeline Estimate

- **Phase 1 (Foundation):** 3-4 weeks (10 tickets)
- **Phase 2 (Pipeline):** 3-4 weeks (10 tickets)
- **Phase 3 (Advanced):** 2-3 weeks (10 tickets)
- **Phase 4 (Production):** 1-2 weeks (6 tickets)

**Total:** ~10-13 weeks for complete memory-system component

---

## 11. References

- **Cognee Repository:** https://github.com/topoteretes/cognee
- **AgentCore Repository:** https://github.com/Mathews-Tom/AgentCore
- **PGVector:** https://github.com/pgvector/pgvector
- **Neo4j Python Driver:** https://neo4j.com/docs/python-manual/current/
- **Cognee Architecture:** Based on reverse engineering (no formal docs available)

---

## Appendix A: Cognee Module Structure

```
cognee/
├── api/                    # FastAPI endpoints
├── modules/
│   ├── memify/            # Memory optimization
│   ├── ontology/          # Knowledge representation
│   ├── graph/             # Graph operations
│   ├── retrieval/         # Query engine
│   ├── search/            # Search functionality
│   ├── cognify/           # Embedding/vectorization
│   ├── chunking/          # Text segmentation
│   ├── ingestion/         # Data input
│   ├── pipelines/         # Workflow orchestration
│   ├── data/              # Core data structures
│   ├── engine/            # Execution engine
│   ├── observability/     # Monitoring
│   ├── settings/          # Configuration
│   └── users/             # User management
└── infrastructure/
    ├── databases/
    │   ├── vector/
    │   │   ├── chromadb/
    │   │   ├── lancedb/
    │   │   └── pgvector/
    │   ├── graph/
    │   │   └── neo4j/
    │   ├── relational/
    │   ├── hybrid/
    │   └── cache/
    ├── llm/
    │   └── LLMGateway.py
    └── files/
```

---

**End of Analysis**
