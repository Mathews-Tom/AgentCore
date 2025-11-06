# Memory System Architecture Comparison
## Existing Spec vs. Cognee Enhancement

**Date:** 2025-11-06
**Status:** Architecture Decision Required

---

## Executive Summary

AgentCore has **two potential memory system architectures** to choose from:

1. **Existing Spec** (docs/specs/memory-service/spec.md): Mem0 + COMPASS + Qdrant
2. **Cognee-Inspired** (docs/architecture/MEMORY_SYSTEM_ENHANCEMENT_ANALYSIS.md): ECL Pipeline + PGVector + Neo4j

**Recommendation:** **Hybrid Approach** - Merge the best concepts from both architectures.

---

## Architecture Comparison

### 1. Core Philosophy

| Aspect | Existing Spec (Mem0 + COMPASS) | Cognee-Inspired |
|--------|--------------------------------|-----------------|
| **Memory Model** | Cognitive psychology-based (4 layers) | Knowledge graph-based (ECL pipeline) |
| **Inspiration** | Human memory system | Document knowledge bases |
| **Primary Focus** | Agent reasoning stages | Semantic relationships |
| **Organization** | Hierarchical (Working → Episodic → Semantic → Procedural) | Graph-based (entities + relationships) |

### 2. Storage Technologies

| Component | Existing Spec | Cognee-Inspired | Winner |
|-----------|---------------|-----------------|--------|
| **Vector DB** | Qdrant | PGVector | **Qdrant** (more features) |
| **Graph DB** | Not included | Neo4j | **Neo4j** (add to spec) |
| **Fast Cache** | Redis | Redis | **Tie** |
| **Framework** | Mem0 | Custom ECL | **Hybrid** |

**Analysis:**
- **Qdrant** is more feature-rich than PGVector (collections, filtering, hybrid search)
- **Neo4j** adds valuable relationship modeling missing from existing spec
- **Redis** is needed for working memory in both approaches
- **Mem0** provides good abstractions but may limit flexibility

### 3. Memory Layers/Stages

#### Existing Spec (4 Layers)

```
┌─────────────────────────────────────────┐
│   Working Memory (Redis)                │
│   - 2-4K tokens, 1-hour TTL            │
│   - Current conversation context        │
└─────────────────┬───────────────────────┘
                  ↓ (automatic promotion)
┌─────────────────────────────────────────┐
│   Episodic Memory (Qdrant)             │
│   - 50 recent episodes                  │
│   - Temporal context, stage-aware       │
└─────────────────┬───────────────────────┘
                  ↓ (fact extraction)
┌─────────────────────────────────────────┐
│   Semantic Memory (Qdrant)             │
│   - 1000+ facts, preferences            │
│   - Importance-scored, deduplicated     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│   Procedural Memory (Qdrant)           │
│   - Action-outcome patterns             │
│   - Success rate tracking               │
└─────────────────────────────────────────┘
```

#### Cognee-Inspired (ECL Pipeline)

```
┌─────────────────────────────────────────┐
│   Extract Phase                         │
│   - Multi-source data ingestion         │
│   - Chunking strategies                 │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   Cognify Phase                         │
│   - Entity extraction                   │
│   - Relationship detection              │
│   - Knowledge graph generation          │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   Load Phase                            │
│   - Vector storage (semantic search)    │
│   - Graph storage (relationships)       │
│   - Memify optimization                 │
└─────────────────────────────────────────┘
```

### 4. Key Features Comparison

| Feature | Existing Spec | Cognee-Inspired | Recommendation |
|---------|---------------|-----------------|----------------|
| **Stage Awareness (COMPASS)** | ✅ Yes (Planning, Execution, Reflection, Verification) | ❌ No | **Keep from existing** |
| **Hierarchical Compression** | ✅ Yes (10:1 → 5:1 ratios) | ❌ No | **Keep from existing** |
| **Test-Time Scaling** | ✅ Yes (gpt-4o-mini for compression) | ❌ No | **Keep from existing** |
| **Error Tracking** | ✅ Yes (Pattern detection) | ❌ No | **Keep from existing** |
| **Knowledge Graphs** | ❌ No | ✅ Yes (Neo4j) | **Add from Cognee** |
| **Entity Relationships** | ⚠️ Limited | ✅ Yes (Explicit modeling) | **Add from Cognee** |
| **Memify Optimization** | ❌ No | ✅ Yes | **Add from Cognee** |
| **Modular Task Pipeline** | ⚠️ Partial | ✅ Yes | **Add from Cognee** |
| **Multi-Backend Support** | ❌ No (Qdrant only) | ✅ Yes | **Add from Cognee** |

### 5. Storage Architecture

#### Existing Spec Storage

```python
# Working Memory
Redis:
  - Key: f"working:{agent_id}:{session_id}"
  - Value: JSON (messages, context)
  - TTL: 1 hour

# Long-term Memory (Episodic, Semantic, Procedural)
Qdrant:
  Collections:
    - episodic_memory
    - semantic_memory
    - procedural_memory

  Embeddings: text-embedding-3-small (OpenAI)

  Metadata:
    - agent_id
    - session_id
    - timestamp
    - importance_score
    - stage (planning/execution/reflection/verification)
```

#### Cognee-Inspired Storage

```python
# Vector Storage
PGVector (or Qdrant):
  Table: agent_memories
  Columns:
    - memory_id UUID
    - agent_id TEXT
    - content TEXT
    - embedding vector(1536)
    - metadata JSONB
    - created_at, accessed_at

# Graph Storage
Neo4j:
  Nodes:
    - Memory (id, content, timestamp)
    - Entity (id, type, value)
    - Concept (id, name)

  Relationships:
    - MENTIONS (Memory → Entity)
    - RELATES_TO (Entity → Entity)
    - PART_OF (Memory → Concept)
    - FOLLOWS (Memory → Memory) [temporal]
```

---

## Recommended Hybrid Architecture

### Merged Storage Strategy

**Layer 1: Working Memory (Redis)**
- Fast cache for immediate context
- TTL-based eviction
- Stage-aware organization (from existing spec)

**Layer 2: Vector Memory (Qdrant)**
- Episodic, Semantic, Procedural collections (from existing spec)
- COMPASS hierarchical compression (from existing spec)
- Embeddings for semantic search
- Multi-factor importance scoring

**Layer 3: Graph Memory (Neo4j)** ⭐ **NEW from Cognee**
- Entity and relationship modeling
- Knowledge graph structure
- Contextual traversal
- Cross-session connections

**Layer 4: Task Pipeline (ECL-inspired)** ⭐ **NEW from Cognee**
- Extract: Multi-source ingestion
- Cognify: Entity/relationship extraction
- Load: Multi-backend storage
- Memify: Graph optimization

### Hybrid Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│              Agent Interaction                           │
└─────────────────────┬────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  Working Memory (Redis) - COMPASS Stage-Aware          │
│  - Planning, Execution, Reflection, Verification        │
│  - 2-4K tokens, 1-hour TTL                             │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  ECL Pipeline (Task-Based Processing) ⭐ NEW             │
│                                                          │
│  Extract → Cognify → Load → Memify                      │
│    ↓        ↓        ↓        ↓                          │
│  Ingest  Entities  Store  Optimize                      │
│  Data    Relations Vector   Graph                       │
└─────────────────────┬───────────────────────────────────┘
                      ↓
        ┌─────────────┴──────────────┐
        ↓                            ↓
┌──────────────────────┐    ┌───────────────────────┐
│  Vector Storage      │    │  Graph Storage ⭐ NEW  │
│  (Qdrant)           │    │  (Neo4j)              │
│                      │    │                       │
│  Collections:        │    │  Nodes:               │
│  - Episodic         │◄───┤  - Memory             │
│  - Semantic         │───►│  - Entity             │
│  - Procedural       │    │  - Concept            │
│                      │    │                       │
│  COMPASS Compressed  │    │  Relationships:       │
│  10:1 → 5:1 ratios  │    │  - MENTIONS           │
│                      │    │  - RELATES_TO         │
│  Mem0 Embeddings    │    │  - PART_OF            │
└──────────────────────┘    └───────────────────────┘
```

### Hybrid API Design

```python
# Memory Service with Hybrid Storage
class HybridMemoryService:
    """Combines Mem0+COMPASS (existing) with Cognee concepts."""

    def __init__(
        self,
        redis_client: Redis,
        qdrant_client: QdrantClient,
        neo4j_driver: Neo4jDriver,
        llm_service: LLMService
    ):
        # Layer 1: Working Memory (Redis)
        self.working_memory = WorkingMemoryStore(redis_client)

        # Layer 2: Vector Memory (Qdrant with Mem0)
        self.vector_memory = VectorMemoryStore(qdrant_client)

        # Layer 3: Graph Memory (Neo4j) - NEW
        self.graph_memory = GraphMemoryStore(neo4j_driver)

        # Layer 4: ECL Pipeline - NEW
        self.pipeline = MemoryPipeline(llm_service)

    async def store_interaction(
        self,
        agent_id: str,
        interaction: dict,
        stage: str  # COMPASS stage
    ) -> str:
        """Store interaction with hybrid approach."""

        # 1. Store in working memory (Redis) with stage
        await self.working_memory.add(
            agent_id,
            interaction,
            stage=stage  # COMPASS stage-aware
        )

        # 2. Run ECL pipeline for processing (NEW)
        result = await self.pipeline.execute({
            "text": interaction["content"],
            "metadata": {
                "agent_id": agent_id,
                "stage": stage
            }
        })

        # 3. Store in vector memory (Qdrant)
        vector_id = await self.vector_memory.add_to_collection(
            collection=self._get_memory_type(stage),  # episodic/semantic/procedural
            embedding=result["embedding"],
            payload={
                "agent_id": agent_id,
                "stage": stage,
                "importance": result["importance_score"]
            }
        )

        # 4. Store in graph memory (Neo4j) - NEW
        graph_id = await self.graph_memory.add_nodes_and_relationships(
            entities=result["entities"],
            relationships=result["relationships"],
            memory_id=vector_id
        )

        return f"{vector_id}:{graph_id}"

    async def query_memory(
        self,
        agent_id: str,
        query: str,
        stage: str,  # COMPASS stage
        k: int = 5
    ) -> list[dict]:
        """Hybrid search combining vector + graph."""

        # 1. Vector search in Qdrant (semantic similarity)
        vector_results = await self.vector_memory.search(
            collection=self._get_memory_type(stage),
            query_vector=await self._embed_query(query),
            filter={"agent_id": agent_id},
            limit=k * 2
        )

        # 2. Graph traversal in Neo4j (contextual relationships) - NEW
        entity_ids = [r.payload["entity_ids"] for r in vector_results]
        graph_context = await self.graph_memory.get_related_context(
            entity_ids,
            max_depth=2
        )

        # 3. Merge results with COMPASS importance scoring
        merged = self._merge_and_rank(
            vector_results,
            graph_context,
            stage=stage  # Stage-specific relevance
        )

        return merged[:k]

    async def compress_stage(
        self,
        agent_id: str,
        stage: str
    ) -> dict:
        """COMPASS compression with graph optimization."""

        # 1. Get all memories for stage from vector DB
        stage_memories = await self.vector_memory.get_by_stage(
            agent_id,
            stage
        )

        # 2. COMPASS compression (10:1 ratio)
        compressed = await self._compress_with_llm(
            stage_memories,
            target_ratio=0.1,  # 10:1
            model="gpt-4o-mini"  # Test-time scaling
        )

        # 3. Extract critical entities for graph (NEW)
        entities = await self._extract_critical_entities(compressed)

        # 4. Update graph with compressed relationships (NEW)
        await self.graph_memory.consolidate_stage(
            agent_id,
            stage,
            entities
        )

        # 5. Store compressed summary
        return await self.vector_memory.store_stage_summary(
            agent_id,
            stage,
            compressed
        )

    async def memify(self, agent_id: str) -> None:
        """Memory optimization (from Cognee) + COMPASS pruning."""

        # 1. COMPASS: Identify low-importance memories for pruning
        low_importance = await self.vector_memory.get_low_importance(
            agent_id,
            threshold=0.3
        )

        # 2. Cognee: Optimize graph structure
        await self.graph_memory.optimize_relationships(agent_id)

        # 3. Cognee: Consolidate similar nodes
        await self.graph_memory.consolidate_similar_entities(
            agent_id,
            similarity_threshold=0.9
        )

        # 4. COMPASS: Prune low-value memories
        await self.vector_memory.delete_batch(low_importance)
```

---

## Technology Stack (Hybrid)

### Required Dependencies

```toml
[dependencies]
# Existing Spec Dependencies
redis = "^5.1.0"              # Working memory
qdrant-client = "^1.11.0"     # Vector storage
mem0 = "^0.1.0"               # Memory framework (if keeping)

# Cognee-Inspired Additions
neo4j = "^5.24.0"             # Graph database
sentence-transformers = "^3.1.0"  # Local embeddings (optional)

# Already in AgentCore
openai = "^1.54.0"            # Embeddings + LLM
asyncpg = "^0.29.0"           # PostgreSQL
pydantic = "^2.9.0"           # Models
```

### Database Setup

```bash
# Redis (Working Memory)
docker run -d -p 6379:6379 redis:7-alpine

# Qdrant (Vector Memory)
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Neo4j (Graph Memory) - NEW
docker run -d \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community
```

---

## Implementation Phases (Merged)

### Phase 1: Foundation (Weeks 1-4)
**Keep from Existing Spec:**
- MEM-001: Four-layer architecture setup
- MEM-002: Database migrations (Qdrant collections)
- MEM-003: COMPASS Pydantic models
- MEM-004: SQLAlchemy ORM models
- MEM-005: Repository layer
- MEM-010: Integrate with existing EmbeddingService

**Add from Cognee:**
- MEM-006-NEW: Neo4j setup and graph schema
- MEM-007-NEW: Graph adapter interface
- MEM-008-NEW: Entity extraction task
- MEM-009-NEW: Relationship extraction task

### Phase 2: ECL Pipeline (Weeks 5-8)
**Add from Cognee:**
- MEM-012-NEW: Pipeline orchestration (Extract → Cognify → Load)
- MEM-013-NEW: Task registry and composition
- MEM-014-NEW: Chunking strategies
- MEM-015-NEW: Graph generation from extractions
- MEM-016: Incremental updates (adapt existing)

**Keep from Existing Spec:**
- MEM-006: StageManager core (COMPASS)
- MEM-007: Stage detection logic
- MEM-011: JSON-RPC handlers

### Phase 3: COMPASS + Optimization (Weeks 9-11)
**Keep from Existing Spec:**
- MEM-012: ContextCompressor core
- MEM-013: tiktoken integration
- MEM-014: Compression quality validation
- MEM-015: Cost tracking
- MEM-017: Compression caching in Redis
- MEM-018: Optimize compression prompts

**Add from Cognee:**
- MEM-021-NEW: Memify operation (graph optimization)
- MEM-022-NEW: Graph-aware compression

### Phase 4: Advanced Features (Weeks 12-13)
**Keep from Existing Spec:**
- MEM-020: ErrorTracker core
- MEM-021: Error pattern detection
- MEM-023: Enhanced importance scoring
- MEM-024: Stage relevance calculation
- MEM-028: ACE integration methods

**Merge Concepts:**
- MEM-030: JSON-RPC handlers (vector + graph endpoints)

### Phase 5: Production (Weeks 14-15)
**Keep from Existing Spec:**
- MEM-031: Tune Qdrant indexes
- MEM-032: Optimize queries
- MEM-033: Stress testing
- MEM-034: COMPASS validation
- MEM-035: Monitoring and alerting
- MEM-036: Documentation

**Add from Cognee:**
- MEM-031-NEW: Neo4j index optimization
- MEM-037-NEW: Graph query performance testing

---

## Decision Matrix

| Criteria | Existing Spec Only | Cognee Only | **Hybrid (Recommended)** |
|----------|-------------------|-------------|-------------------------|
| **Stage Awareness** | ✅ COMPASS | ❌ No | ✅ **COMPASS** |
| **Compression** | ✅ 10:1 → 5:1 | ❌ No | ✅ **10:1 → 5:1** |
| **Error Tracking** | ✅ Yes | ❌ No | ✅ **Yes** |
| **Knowledge Graphs** | ❌ No | ✅ Neo4j | ✅ **Neo4j** |
| **Entity Relations** | ⚠️ Limited | ✅ Explicit | ✅ **Explicit** |
| **Memify Optimization** | ❌ No | ✅ Yes | ✅ **Yes** |
| **Vector DB** | ✅ Qdrant | ⚠️ PGVector | ✅ **Qdrant** |
| **Memory Layers** | ✅ 4 layers | ❌ No | ✅ **4 layers** |
| **Pipeline Modularity** | ⚠️ Partial | ✅ Full | ✅ **Full** |
| **Complexity** | Medium | Medium | **High** |
| **Implementation Time** | 4-5 weeks | 10-13 weeks | **6-8 weeks** |
| **Leverage Existing Work** | ✅ Full | ❌ None | ✅ **Most** |

---

## Recommendation

### ✅ Adopt Hybrid Architecture

**Rationale:**
1. **Leverage Existing Work** - Don't throw away the detailed Mem0+COMPASS spec
2. **Add Missing Capabilities** - Graph relationships are valuable for agent memory
3. **Best of Both Worlds** - COMPASS compression + Cognee knowledge graphs
4. **Realistic Timeline** - 6-8 weeks vs 10-13 weeks for full Cognee approach

### Implementation Strategy

1. **Start with Existing Spec** (Weeks 1-4)
   - Implement 4-layer architecture
   - Setup Qdrant and Redis
   - COMPASS stage awareness
   - Working → Episodic → Semantic transitions

2. **Add Graph Layer** (Weeks 5-6)
   - Setup Neo4j
   - Entity extraction using existing LLM gateway
   - Relationship detection
   - Dual storage (Qdrant + Neo4j)

3. **Enhance with ECL Concepts** (Weeks 7-8)
   - Modular task pipeline
   - Graph-aware retrieval
   - Memify optimization
   - Hybrid search (vector + graph)

### What to Keep from Each

**From Existing Spec (Mem0 + COMPASS):**
- ✅ Four-layer memory model (Working, Episodic, Semantic, Procedural)
- ✅ COMPASS stage awareness (Planning, Execution, Reflection, Verification)
- ✅ Hierarchical compression (10:1 → 5:1 ratios)
- ✅ Test-time scaling with gpt-4o-mini
- ✅ Error tracking and pattern detection
- ✅ Qdrant for vector storage
- ✅ Redis for working memory
- ✅ Mem0 integration (if it provides value)

**From Cognee Analysis:**
- ✅ Neo4j graph database for relationships
- ✅ ECL pipeline modularity (Extract → Cognify → Load)
- ✅ Task-based processing architecture
- ✅ Memify optimization operation
- ✅ Graph-aware retrieval patterns
- ✅ Entity and relationship modeling
- ✅ Knowledge graph generation

**What to Discard:**
- ❌ PGVector (Qdrant is better for this use case)
- ❌ Complete rewrite of existing spec
- ❌ ChromaDB/LanceDB (unnecessary complexity)

---

## Action Items

1. **Update Existing Spec** (docs/specs/memory-service/spec.md)
   - Add Neo4j graph storage section
   - Add ECL pipeline concepts
   - Add Memify operation
   - Update architecture diagrams

2. **Merge Ticket Plans**
   - Review existing MEM-001 to MEM-036 tickets
   - Add graph-related tickets (MEM-037 to MEM-040)
   - Update task descriptions to include graph operations

3. **Update MEMORY_SYSTEM_ENHANCEMENT_ANALYSIS.md**
   - Mark as "Hybrid Approach Adopted"
   - Reference this comparison document
   - Keep Cognee patterns as reference

4. **Create Implementation Plan**
   - 6-8 week timeline
   - Hybrid architecture
   - Phased rollout

---

## References

- **Existing Spec:** docs/specs/memory-service/spec.md
- **Cognee Analysis:** docs/architecture/MEMORY_SYSTEM_ENHANCEMENT_ANALYSIS.md
- **Research Doc:** docs/research/evolving-memory-system.md
- **Mem0:** https://mem0.ai/
- **Cognee:** https://github.com/topoteretes/cognee
- **COMPASS:** Research paper on hierarchical memory compression
- **Qdrant:** https://qdrant.tech/
- **Neo4j:** https://neo4j.com/

---

**Decision Required:** Should we proceed with the Hybrid Architecture?
- [ ] Yes - Merge existing spec with Cognee concepts (Recommended)
- [ ] No - Use existing spec as-is (Mem0 + COMPASS only)
- [ ] No - Use Cognee approach only (discard existing work)

**Next Step After Decision:**
- Update docs/specs/memory-service/spec.md with hybrid architecture
- Create updated ticket breakdown (MEM-001 to MEM-040)
- Begin Phase 1 implementation
