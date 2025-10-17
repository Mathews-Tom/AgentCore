# Evolving Memory System

## Overview

An evolving memory system provides agents with the ability to maintain, update, and utilize contextual information across multiple turns and interactions. Unlike static context windows that simply concatenate past messages, an evolving memory system intelligently manages what information to retain, how to represent it compactly, and when to retrieve relevant memories to inform current decisions.

This capability is critical for enabling agents to handle complex, multi-step workflows where context from earlier steps influences later actions, and where the full interaction history would exceed available context limits.

## Technical Description

### Memory Architecture

The evolving memory system consists of multiple memory layers, each serving different purposes:

**1. Working Memory**

- Short-term memory for current task execution
- Stores immediate context (current query, recent actions, temporary state)
- Fast access, limited capacity (typically 2K-4K tokens)
- Cleared or archived when task completes

**2. Episodic Memory**

- Medium-term memory for recent interaction history
- Stores structured records of past turns (query, actions, results, outcomes)
- Semantic indexing for relevance-based retrieval
- Capacity: 10-50 recent episodes

**3. Semantic Memory**

- Long-term memory for accumulated knowledge
- Stores facts, insights, patterns learned from past interactions
- Vector embeddings for similarity search
- Unbounded capacity with relevance-based pruning

**4. Procedural Memory**

- Long-term memory for learned strategies and patterns
- Stores successful action sequences and heuristics
- Indexed by task type and context
- Updated through reinforcement signals

### Memory Operations

**1. Encode**

```python
async def encode_memory(
    interaction: Interaction,
    memory_type: str
) -> MemoryRecord:
    """
    Encode an interaction into a memory record.

    Creates structured representation with:
    - Semantic summary
    - Key entities and facts
    - Action-outcome pairs
    - Relevance metadata
    """
    pass
```

**2. Store**

```python
async def store_memory(
    record: MemoryRecord,
    memory_layer: str
) -> str:
    """
    Store memory record in appropriate layer.

    Returns memory_id for future retrieval.
    """
    pass
```

**3. Retrieve**

```python
async def retrieve_memories(
    query: str,
    memory_layers: list[str],
    k: int = 5
) -> list[MemoryRecord]:
    """
    Retrieve relevant memories using semantic search.

    Combines:
    - Embedding similarity
    - Temporal relevance
    - Interaction frequency
    - Explicit relationships
    """
    pass
```

**4. Update**

```python
async def update_memory(
    memory_id: str,
    updates: dict[str, Any]
) -> None:
    """
    Update existing memory record.

    Used for:
    - Correcting incorrect information
    - Adding new context
    - Updating relevance scores
    """
    pass
```

**5. Prune**

```python
async def prune_memories(
    memory_layer: str,
    strategy: str = "least_relevant"
) -> int:
    """
    Remove low-value memories to maintain capacity.

    Strategies:
    - least_relevant: Remove lowest relevance scores
    - oldest_first: Remove oldest memories
    - frequency_based: Remove rarely accessed
    """
    pass
```

### Memory Record Structure

```python
from pydantic import BaseModel
from datetime import datetime

class MemoryRecord(BaseModel):
    """Structured memory record."""

    memory_id: str
    memory_type: str  # "working", "episodic", "semantic", "procedural"

    # Content
    content: str  # Original content
    summary: str  # Condensed summary
    embedding: list[float]  # Vector representation

    # Metadata
    timestamp: datetime
    interaction_id: str | None
    task_id: str | None

    # Entities and facts
    entities: list[str] = []
    facts: list[str] = []
    keywords: list[str] = []

    # Relationships
    related_memory_ids: list[str] = []
    parent_memory_id: str | None = None

    # Relevance tracking
    relevance_score: float = 1.0
    access_count: int = 0
    last_accessed: datetime | None = None

    # Outcome tracking (for procedural memory)
    actions: list[str] = []
    outcome: str | None = None
    success: bool | None = None
```

### Memory Manager

```python
class MemoryManager:
    """Manages all memory operations."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        capacity_limits: dict[str, int]
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.capacity_limits = capacity_limits

        # In-memory cache for working memory
        self.working_memory: dict[str, MemoryRecord] = {}

    async def add_interaction(
        self,
        interaction: Interaction
    ) -> None:
        """
        Process and store a new interaction.

        Workflow:
        1. Encode interaction into memory record
        2. Store in appropriate layer(s)
        3. Update related memories
        4. Prune if capacity exceeded
        """
        # Encode
        episodic_record = await self._encode_episodic(interaction)
        semantic_records = await self._extract_semantic_facts(interaction)

        # Store
        await self.store_memory(episodic_record, "episodic")
        for record in semantic_records:
            await self.store_memory(record, "semantic")

        # Update working memory
        if interaction.task_id:
            self._update_working_memory(interaction.task_id, episodic_record)

        # Prune if needed
        await self._check_and_prune()

    async def get_relevant_context(
        self,
        query: str,
        task_id: str | None = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Retrieve and format relevant memories as context.

        Returns formatted string suitable for LLM context.
        """
        # Get working memory for current task
        working = []
        if task_id and task_id in self.working_memory:
            working = [self.working_memory[task_id]]

        # Retrieve relevant episodic memories
        episodic = await self.retrieve_memories(
            query=query,
            memory_layers=["episodic"],
            k=5
        )

        # Retrieve relevant semantic facts
        semantic = await self.retrieve_memories(
            query=query,
            memory_layers=["semantic"],
            k=10
        )

        # Format and truncate to token limit
        context = self._format_context(
            working=working,
            episodic=episodic,
            semantic=semantic,
            max_tokens=max_tokens
        )

        return context

    def _format_context(
        self,
        working: list[MemoryRecord],
        episodic: list[MemoryRecord],
        semantic: list[MemoryRecord],
        max_tokens: int
    ) -> str:
        """Format memories into context string."""
        sections = []

        # Current task context
        if working:
            sections.append("## Current Task Context")
            for record in working:
                sections.append(f"- {record.summary}")

        # Relevant past interactions
        if episodic:
            sections.append("\n## Relevant Past Interactions")
            for record in episodic:
                sections.append(
                    f"- [{record.timestamp.isoformat()}] {record.summary}"
                )

        # Relevant facts and knowledge
        if semantic:
            sections.append("\n## Relevant Knowledge")
            for record in semantic:
                sections.append(f"- {record.summary}")

        context = "\n".join(sections)

        # Truncate if exceeds limit
        tokens = self._count_tokens(context)
        if tokens > max_tokens:
            context = self._truncate_to_tokens(context, max_tokens)

        return context

    async def _encode_episodic(
        self,
        interaction: Interaction
    ) -> MemoryRecord:
        """Encode interaction as episodic memory."""
        # Generate summary
        summary = await self._summarize_interaction(interaction)

        # Generate embedding
        embedding = await self.embedding_model.embed(summary)

        # Extract entities and facts
        entities = await self._extract_entities(interaction)
        facts = await self._extract_facts(interaction)

        return MemoryRecord(
            memory_id=str(uuid.uuid4()),
            memory_type="episodic",
            content=interaction.to_string(),
            summary=summary,
            embedding=embedding,
            timestamp=interaction.timestamp,
            interaction_id=interaction.id,
            task_id=interaction.task_id,
            entities=entities,
            facts=facts,
            actions=interaction.actions,
            outcome=interaction.outcome,
            success=interaction.success
        )

    async def retrieve_memories(
        self,
        query: str,
        memory_layers: list[str],
        k: int = 5
    ) -> list[MemoryRecord]:
        """Retrieve top-k relevant memories."""
        # Generate query embedding
        query_embedding = await self.embedding_model.embed(query)

        # Search vector store
        results = await self.vector_store.search(
            embedding=query_embedding,
            filters={"memory_type": {"$in": memory_layers}},
            k=k
        )

        # Convert to MemoryRecords
        memories = [
            MemoryRecord(**result.metadata)
            for result in results
        ]

        # Update access tracking
        for memory in memories:
            await self._update_access_tracking(memory.memory_id)

        return memories
```

### Context Compression

To maintain bounded context, implement compression strategies:

**1. Summarization**

```python
async def compress_memory_sequence(
    memories: list[MemoryRecord],
    target_tokens: int
) -> str:
    """Compress sequence of memories into target token count."""
    if len(memories) == 0:
        return ""

    # Generate hierarchical summary
    summary = await self.llm.generate(
        prompt=f"""Summarize the following sequence of memories concisely:

{chr(10).join(m.summary for m in memories)}

Target length: ~{target_tokens} tokens
Focus on: Key facts, patterns, and outcomes
""",
        max_tokens=target_tokens
    )

    return summary.strip()
```

**2. Importance Weighting**

```python
def calculate_importance_score(
    memory: MemoryRecord,
    query: str,
    current_time: datetime
) -> float:
    """Calculate importance score for memory."""
    # Recency: Exponential decay
    age_hours = (current_time - memory.timestamp).total_seconds() / 3600
    recency_score = math.exp(-age_hours / 24)  # Half-life of 24 hours

    # Frequency: Access count
    frequency_score = min(memory.access_count / 10, 1.0)

    # Relevance: Embedding similarity
    relevance_score = memory.relevance_score

    # Combined score
    importance = (
        0.4 * relevance_score +
        0.3 * recency_score +
        0.3 * frequency_score
    )

    return importance
```

## Value Analysis

### Performance Benefits

**1. Enhanced Context Awareness**

- Agents maintain understanding across long interactions
- Can reference past information without re-stating
- Expected improvement: +25-30% on multi-turn tasks

**2. Reduced Context Size**

- Selective retrieval vs full history concatenation
- 5-10x reduction in context tokens for long sessions
- Lower latency and cost per query

**3. Knowledge Accumulation**

- Agents learn from past interactions
- Build up domain knowledge over time
- Improve performance with usage

**4. Better Error Recovery**

- Can reference what was tried before
- Avoid repeating failed approaches
- Expected improvement: +15-20% recovery rate

### Scalability Benefits

**1. Unbounded Interaction Length**

- No practical limit on conversation depth
- Working memory stays bounded
- Enables persistent agents

**2. Multi-Session Support**

- Memory persists across sessions
- Resume conversations seamlessly
- Share knowledge across tasks

**3. Efficient Resource Utilization**

```
Traditional (Full History):
- Session with 50 turns
- Average 500 tokens/turn
- Total context: 25K tokens
- Cost: High, growing linearly

Evolving Memory:
- Session with 50 turns
- Working memory: 2K tokens
- Retrieved context: 2K tokens
- Total context: 4K tokens
- Cost: 84% reduction
```

## Implementation Considerations

### Technical Challenges

**1. Memory Coherence**

- Challenge: Ensuring retrieved memories are coherent together
- Solution: Implement memory linking, temporal ordering
- Metrics: Measure context coherence scores

**2. Retrieval Accuracy**

- Challenge: Finding truly relevant memories
- Solution: Hybrid search (embedding + metadata + graph)
- Metrics: Precision@K, Recall@K

**3. Storage Scaling**

- Challenge: Growing memory database
- Solution: Tiered storage, archival policies
- Metrics: Storage cost, retrieval latency

**4. Consistency**

- Challenge: Contradictory memories
- Solution: Confidence tracking, conflict resolution
- Metrics: Consistency violations per session

### Resource Requirements

**1. Vector Database**

- Embedding storage and similarity search
- Options: Pinecone, Weaviate, Qdrant, PGVector
- Scale: 1M+ vectors for production

**2. Embedding Model**

- Generate embeddings for memory encoding
- Options: OpenAI, Cohere, local (SentenceTransformers)
- Throughput: 100+ embeddings/sec

**3. Storage**

- PostgreSQL for structured memory records
- Redis for working memory cache
- S3 for memory archives

## Integration Strategy

### Phase 1: Core Memory System (Weeks 1-2)

**Implement Memory Manager**

```python
# agentcore/memory/
├── __init__.py
├── manager.py          # MemoryManager
├── models.py           # MemoryRecord, Interaction
├── encoding.py         # Memory encoding logic
├── retrieval.py        # Memory retrieval
├── storage.py          # Storage backends
└── compression.py      # Context compression
```

**Database Schema**

```sql
CREATE TABLE memories (
    memory_id UUID PRIMARY KEY,
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
    metadata JSONB
);

CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);
```

### Phase 2: JSON-RPC Integration (Week 3)

**Memory Methods**

```python
@register_jsonrpc_method("memory.store")
async def store_memory(request: JsonRpcRequest) -> dict[str, Any]:
    """Store a memory."""
    interaction = Interaction(**request.params["interaction"])
    await memory_manager.add_interaction(interaction)
    return {"success": True}

@register_jsonrpc_method("memory.retrieve")
async def retrieve_memories(request: JsonRpcRequest) -> dict[str, Any]:
    """Retrieve relevant memories."""
    query = request.params["query"]
    k = request.params.get("k", 5)

    memories = await memory_manager.retrieve_memories(
        query=query,
        memory_layers=["episodic", "semantic"],
        k=k
    )

    return {
        "memories": [
            {
                "memory_id": m.memory_id,
                "summary": m.summary,
                "timestamp": m.timestamp.isoformat(),
                "relevance_score": m.relevance_score
            }
            for m in memories
        ]
    }

@register_jsonrpc_method("memory.get_context")
async def get_context(request: JsonRpcRequest) -> dict[str, Any]:
    """Get formatted context for current query."""
    query = request.params["query"]
    task_id = request.params.get("task_id")
    max_tokens = request.params.get("max_tokens", 2000)

    context = await memory_manager.get_relevant_context(
        query=query,
        task_id=task_id,
        max_tokens=max_tokens
    )

    return {"context": context}
```

### Phase 3: Agent Integration (Week 4)

**Memory-Enabled Agents**

```python
class MemoryEnabledAgent:
    """Agent with evolving memory."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    async def process_query(self, query: str, task_id: str) -> str:
        # Retrieve relevant context
        context = await self.memory.get_relevant_context(
            query=query,
            task_id=task_id
        )

        # Generate response with context
        response = await self.llm.generate(
            prompt=f"""Context from memory:
{context}

Current query: {query}

Response:""",
            max_tokens=500
        )

        # Store interaction
        await self.memory.add_interaction(
            Interaction(
                id=str(uuid.uuid4()),
                task_id=task_id,
                query=query,
                response=response,
                timestamp=datetime.now(UTC),
                success=True
            )
        )

        return response
```

### Phase 4: Advanced Features (Weeks 5-6)

**1. Memory Visualization**

- Web UI for browsing memories
- Temporal timeline view
- Entity relationship graphs

**2. Memory Export/Import**

- Export memories for analysis
- Import knowledge bases
- Cross-agent memory sharing

**3. Adaptive Retrieval**

- Learn retrieval strategies from feedback
- Optimize k value per query type
- Personalized memory importance

## Success Metrics

1. **Context Efficiency**
   - Target: 80%+ reduction in context tokens for long sessions
   - Measure: retrieved_tokens / full_history_tokens

2. **Retrieval Precision**
   - Target: 90%+ of retrieved memories are relevant
   - Measure: manual annotation or LLM-based evaluation

3. **Task Performance**
   - Target: +20% success rate on multi-turn tasks
   - Measure: success_rate_with_memory / success_rate_without

4. **Memory Coherence**
   - Target: <5% contradictory memory retrievals
   - Measure: contradiction detection rate

5. **Latency**
   - Target: <100ms for memory retrieval
   - Measure: p95 retrieval latency

## Conclusion

An evolving memory system transforms agents from stateless responders to contextually aware assistants that learn and improve over time. By intelligently managing what to remember, how to represent it compactly, and when to retrieve it, AgentCore can support complex, long-running workflows while maintaining bounded resource usage. This foundation enables building persistent agents that accumulate knowledge and provide increasingly valuable assistance.
