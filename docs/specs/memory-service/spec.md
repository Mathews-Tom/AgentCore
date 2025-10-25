# Memory Service Specification (Mem0 + COMPASS Enhanced)

**Component ID:** MEM
**Status:** Ready for Implementation
**Priority:** P0 (HIGH - Phase 2)
**Effort:** 4-5 weeks
**Owner:** Backend Team
**Source:** Merged from memory-service (Mem0) + memory-system (COMPASS)

---

## 1. Overview

### Purpose and Business Value

Implement a comprehensive memory service combining:

1. **Four-Layer Memory Architecture** (Mem0 Integration)
2. **COMPASS Hierarchical Organization** (Stage-Aware Context Management)
3. **Test-Time Scaling** (Cost-Optimized Compression)
4. **Error-Aware Memory** (Pattern Detection & Prevention)

**Four-Layer Memory Architecture:**

- **Working Memory**: Immediate context (Redis, 2-4K tokens, 1-hour TTL)
- **Episodic Memory**: Recent conversation episodes (Mem0/Qdrant, 50 episodes)
- **Semantic Memory**: Long-term facts and knowledge (Mem0/Qdrant, 1000+ entries)
- **Procedural Memory**: Action-outcome patterns (Mem0/Qdrant, workflows)

**COMPASS Enhancements:**

- **Hierarchical Organization**: Raw memories → Stage summaries (10:1) → Task context (5:1)
- **Stage-Aware Context**: Planning, execution, reflection, verification stages
- **Test-Time Scaling**: gpt-4o-mini for compression (70-80% cost reduction)
- **Error Tracking**: Explicit error memory with pattern detection
- **Enhanced Retrieval**: Multi-factor importance scoring with criticality boost

**Mem0 Integration Benefits:**

- Semantic search with vector embeddings
- Cross-session memory continuity
- Intelligent context retrieval
- Task artifact storage
- +26% accuracy improvement (per Mem0 benchmarks)

**COMPASS Validation:**

- 60-80% context reduction (validated on long-horizon tasks)
- 20% accuracy improvement on GAIA benchmark
- 70-80% cost reduction via test-time scaling
- 95%+ information retention in compression

### Success Metrics

**Context Efficiency (COMPASS):**

- 60-80% reduction in context tokens for long sessions
- 10:1 compression ratio for stage summaries
- 5:1 compression ratio for task summaries
- 95%+ critical fact retention after compression

**Performance:**

- <100ms for memory search queries (p95)
- <5s for stage compression (p95)
- 100% of agent/session interactions stored
- 90%+ relevance in semantic search results

**Task Performance:**

- +20% improvement in multi-turn task success rates (COMPASS target)
- <5% contradictory retrievals
- 100% error capture rate
- 80%+ error pattern detection accuracy

**Cost Efficiency:**

- 70-80% reduction in LLM API costs (test-time scaling)
- <$0.01 per stage compression
- Monthly token budget enforcement

**Storage Scalability:**

- 1M+ memories per agent without degradation
- <100ms retrieval at 1M+ scale

### Target Users

- **Agents**: Four-layer memory with automatic transitions + COMPASS stage awareness
- **Sessions**: Maintain coherent context across multi-turn conversations with compression
- **ACE (Meta-Thinker)**: Strategic context for intervention decisions
- **Operators**: Analyze agent behavior, memory patterns, and error trends
- **Developers**: Leverage memory-aware routing and context optimization

---

## 2. Functional Requirements

### FR-1: Four-Layer Memory Architecture (Mem0 Integration)

**FR-1.1 Working Memory** (Immediate Context)

- System SHALL maintain working memory in Redis cache (2-4K tokens)
- Working memory SHALL have 1-hour TTL with automatic eviction
- System SHALL provide fast access (<10ms) to current conversation context
- Working memory SHALL include recent messages, active tasks, and immediate goals

**FR-1.2 Episodic Memory** (Recent Episodes)

- System SHALL store recent conversation episodes in Mem0/Qdrant (50 episodes default)
- Each episode SHALL include temporal context (start/end timestamps)
- System SHALL support episodic retrieval by time range or similarity
- Episodes SHALL transition from working memory automatically
- Episodes SHALL be organized by COMPASS stages

**FR-1.3 Semantic Memory** (Long-term Knowledge)

- System SHALL extract and store facts, preferences, and learnings in Mem0/Qdrant
- Semantic memory SHALL support capacity limits (1000+ entries) with automatic pruning
- System SHALL deduplicate and consolidate related semantic entries
- Facts SHALL be scored by importance and access frequency
- Critical facts SHALL be marked and preserved during compression

**FR-1.4 Procedural Memory** (Action Patterns)

- System SHALL record action-outcome pairs for learned behaviors
- Procedural memory SHALL enable pattern matching for similar situations
- System SHALL track success rates for different action strategies
- Procedures SHALL be retrieved based on context similarity

**FR-1.5 Memory Transitions**

- System SHALL automatically promote working memory to episodic memory
- System SHALL extract semantic facts from episodic memories using Mem0
- System SHALL identify procedural patterns from repeated actions
- Transitions SHALL occur based on time, importance, and access patterns
- Transitions SHALL respect COMPASS stage boundaries

### FR-2: COMPASS Hierarchical Organization (Stage-Aware)

**FR-2.1 Stage-Aware Memory Layers**

The system SHALL organize memories hierarchically across three levels:

- **Level 1: Raw Memories** - Individual interactions across all four layers
- **Level 2: Stage Memories** - Compressed summaries per reasoning stage (10:1 ratio)
- **Level 3: Task Context** - Progressive task summary across stages (5:1 ratio)

**FR-2.2 Reasoning Stage Types**

The system SHALL support four reasoning stage types:

- `planning`: Initial task analysis and strategy formulation
- `execution`: Action execution and outcome observation
- `reflection`: Error analysis and learning
- `verification`: Quality checks and validation

**FR-2.3 Stage Detection and Transitions**

The system SHALL automatically detect reasoning stage transitions based on:

- Agent action patterns (tool usage, reasoning type)
- Explicit stage markers from agent
- Time-based heuristics (default stage duration)
- ACE intervention signals

**FR-2.4 Progressive Summarization**

The system SHALL compress memories progressively:

- When stage completes: Compress raw memories → stage summary (10:1)
- Periodically: Compress completed stages → task progress summary (5:1)
- On demand: Provide hierarchical context at any level
- Quality validation: 95%+ critical fact retention required

### FR-3: Test-Time Scaling Compression (COMPASS Cost Optimization)

**FR-3.1 Compression Pipeline**

The system SHALL implement multi-level compression:

- **Stage Compression**: Raw memories → stage summary (10:1 ratio)
- **Task Compression**: Stage summaries → task progress brief (5:1 ratio)
- **Critical Fact Extraction**: Identify must-remember information
- **Error Pattern Detection**: Track recurring errors across stages

**FR-3.2 Model Selection for Cost Optimization**

The system SHALL use different models for compression vs reasoning:

- **Compression Model**: gpt-4.1-mini ($0.15/1M tokens) for all summarization
- **Extraction Model**: gpt-4.1-mini for critical fact extraction
- **Reasoning Model**: gpt-4.1 or gpt-5 for agent reasoning only
- **Target**: 70-80% cost reduction vs using full model for all operations

**FR-3.3 Compression Quality Monitoring**

The system SHALL track compression quality metrics:

- Critical fact retention rate (target: ≥95%)
- Compression ratio achieved (target: 10:1 for stages, 5:1 for tasks)
- Token cost savings (target: 70-80% reduction)
- Coherence score (no contradictions introduced)

**FR-3.4 Adaptive Compression**

The system SHALL adjust compression aggressiveness based on:

- Task complexity (higher complexity → less aggressive)
- Error rate (high errors → preserve more context)
- Token budget (approaching limit → more aggressive)
- Quality metrics (degrading → fallback to less compression)

### FR-4: Error Memory and Pattern Tracking (COMPASS Learning)

**FR-4.1 Error Record Tracking**

The system SHALL capture error records containing:

- Error ID, task ID, stage ID
- Error type (`hallucination`, `missing_info`, `incorrect_action`, `context_degradation`)
- Error description and context when occurred
- Recovery action taken (if any)
- Error severity (0-1 scale)
- Timestamp

**FR-4.2 Error Pattern Detection**

The system SHALL detect error patterns using:

- Frequency analysis (recurring error types)
- Sequence detection (errors that follow each other)
- Context correlation (errors in similar contexts)
- LLM-based pattern extraction with Mem0

**FR-4.3 Error History Queries**

The system SHALL support error history queries:

- Last N errors for a task
- Errors in specific stage types
- Errors with severity above threshold
- Error patterns for a task or agent

**FR-4.4 Error-Aware Retrieval**

The system SHALL use error history to improve retrieval:

- Boost memories that corrected previous errors
- Retrieve error-prevention knowledge when similar context detected
- Provide error context to ACE for intervention decisions
- Mark error-related memories as critical

### FR-5: Enhanced Retrieval with Criticality Scoring (COMPASS)

**FR-5.1 Multi-Factor Importance Scoring**

The system SHALL compute importance scores using:

- **Embedding similarity**: Cosine distance to query (weight: 0.35)
- **Recency**: Exponential decay from timestamp (weight: 0.15)
- **Frequency**: Logarithmic access count (weight: 0.10)
- **Stage relevance**: Relevance to current stage (weight: 0.20)
- **Criticality boost**: 2x multiplier for critical memories (weight: 0.10)
- **Error correction**: Relevance to recent errors (weight: 0.10)

**FR-5.2 Critical Memory Identification**

The system SHALL automatically mark memories as critical if they:

- Contain explicit constraints or requirements (keyword detection)
- Represent successful error recovery (linked to error records)
- Are frequently accessed in successful task completions (usage pattern)
- Contradict common mistakes (error prevention knowledge)

**FR-5.3 Stage-Aware Retrieval**

The system SHALL adjust retrieval based on reasoning stage:

- **Planning stage**: Prioritize high-level strategies, past plans
- **Execution stage**: Prioritize action memories, tool usage patterns
- **Reflection stage**: Prioritize error history, lessons learned
- **Verification stage**: Prioritize success criteria, quality checks

**FR-5.4 Retrieval Validation**

The system SHALL validate retrieval quality:

- Precision@k measurement (target: ≥95%)
- Critical evidence coverage (all must-remember facts retrieved)
- Contradiction detection (no conflicting memories returned)
- User feedback integration (manual quality ratings)

### FR-6: Multi-Scope Memory Hierarchy (Mem0 Integration)

**FR-6.1** The system SHALL support user-level memory (global across all agents/sessions)
**FR-6.2** The system SHALL support agent-level memory (per-agent preferences and learnings)
**FR-6.3** The system SHALL support session-level memory (conversation context)
**FR-6.4** The system SHALL support task-level memory (task-specific context and stages)
**FR-6.5** The system SHALL link memories across scope levels using Mem0

### FR-7: Memory Storage and Retrieval (Mem0 + Qdrant)

**FR-7.1 Storage Backend**

- The system SHALL use Redis for working memory (fast access, TTL-based)
- The system SHALL use Qdrant for episodic, semantic, and procedural memory (vector storage)
- The system SHALL use PostgreSQL for memory metadata, stage summaries, and relationships
- The system SHALL ensure eventual consistency across storage layers

**FR-7.2 Mem0 Integration**

- The system SHALL use Mem0 for memory extraction and fact recognition
- The system SHALL use Mem0 for semantic search across stored memories
- The system SHALL use Mem0 embeddings (text-embedding-3-small) for vector search
- The system SHALL leverage Mem0's cross-session memory continuity

**FR-7.3 Memory Operations**

- The system SHALL store memory entries with content, metadata, timestamps, and layer designation
- The system SHALL support adding memories with agent_id, session_id, task_id, or user_id scope
- The system SHALL retrieve memories across all layers for a given scope
- The system SHALL support memory deletion and bulk clearing operations per layer

**FR-7.4 Task Artifact Storage**

- The system SHALL store task artifacts as episodic memory metadata
- The system SHALL link task artifacts to sessions and stages
- The system SHALL support artifact retrieval by session_id and stage_id
- The system SHALL track participant agents in session memories

### FR-8: ACE Integration (COMPASS Meta-Thinker Interface)

**FR-8.1 Strategic Context Retrieval**

- The system SHALL provide strategic context for ACE decision-making
- The system SHALL include task progress, error patterns, and critical constraints
- The system SHALL format context optimized for intervention decisions

**FR-8.2 Intervention Outcome Tracking**

- The system SHALL record ACE intervention outcomes
- The system SHALL learn from intervention effectiveness
- The system SHALL update criticality scores based on intervention results

**FR-8.3 Performance Metrics Integration**

- The system SHALL track performance metrics in TaskContext
- The system SHALL signal ACE when error rates exceed thresholds
- The system SHALL provide context degradation warnings

### User Stories

**US-1**: As an agent, I want to access immediate context from working memory so that I can respond quickly to current conversation.

**US-2**: As an agent, I want Mem0 to extract and store semantic facts automatically so that I build long-term knowledge.

**US-3**: As an agent, I want stage-aware context retrieval so that I receive planning context during planning, execution context during execution, etc.

**US-4**: As an agent, I want compressed stage summaries so that I can maintain context efficiency in long tasks without losing critical information.

**US-5**: As an agent, I want error memory tracking so that I can learn from mistakes and avoid repeating them.

**US-6**: As a session manager, I want automatic memory layer transitions so that important information is preserved while working memory stays fresh.

**US-7**: As ACE (Meta-Thinker), I want strategic context queries so that I can make informed intervention decisions.

**US-8**: As a developer, I want cost-optimized compression using mini models so that long-running agents remain economically viable.

### Business Rules

- **BR-1**: Memories must be scoped to at least one of: user_id, agent_id, session_id, task_id
- **BR-2**: Memory search queries must specify scope and may filter by layer and stage
- **BR-3**: Memory deletion must be explicit (no cascading deletes across layers or stages)
- **BR-4**: All memory operations must be async
- **BR-5**: Approved models only: gpt-4.1-mini for compression, gpt-4.1/gpt-5 for reasoning (per CLAUDE.md)
- **BR-6**: Working memory limited to 2-4K tokens with 1-hour TTL
- **BR-7**: Stage compression occurs automatically on stage completion
- **BR-8**: Stage boundaries must not be crossed during compression
- **BR-9**: Critical fact retention must be ≥95% after compression
- **BR-10**: Compression operations must use gpt-4.1-mini only (cost optimization)

---

## 3. Non-Functional Requirements

### Performance

- **NFR-P1**: Memory addition SHALL complete in <50ms (p95)
- **NFR-P2**: Memory search SHALL complete in <100ms (p95)
- **NFR-P3**: Memory retrieval SHALL complete in <200ms for 100 entries (p95)
- **NFR-P4**: Stage compression SHALL complete in <5s (p95)
- **NFR-P5**: Cache hit rate SHALL exceed 80% for recent memories
- **NFR-P6**: Working memory access SHALL complete in <10ms (p95)

### Scalability

- **NFR-S1**: SHALL support 10,000+ memories per agent without degradation
- **NFR-S2**: SHALL support 1,000 concurrent memory operations
- **NFR-S3**: Qdrant SHALL scale horizontally for vector storage
- **NFR-S4**: Redis cache SHALL support LRU eviction under memory pressure
- **NFR-S5**: SHALL support 10+ concurrent long-horizon tasks (50+ turns each)

### Cost Efficiency (COMPASS)

- **NFR-CE1**: SHALL achieve 70-80% cost reduction through test-time scaling
- **NFR-CE2**: Compression operations SHALL use gpt-4.1-mini only ($0.15/1M tokens)
- **NFR-CE3**: Monthly token budget SHALL be enforced with alerts at 75%
- **NFR-CE4**: Cost tracking SHALL be available per agent, task, and operation type

### Compression Quality (COMPASS)

- **NFR-CQ1**: Stage compression SHALL achieve 10:1 ratio (±20%)
- **NFR-CQ2**: Task compression SHALL achieve 5:1 ratio (±20%)
- **NFR-CQ3**: Critical fact retention SHALL be ≥95%
- **NFR-CQ4**: No contradictions SHALL be introduced during compression

### Security

- **NFR-SEC1**: Memories SHALL be isolated by user/agent/session/task scope
- **NFR-SEC2**: Sensitive data SHALL NOT be stored in memory content
- **NFR-SEC3**: Memory access SHALL validate caller authorization
- **NFR-SEC4**: Vector embeddings SHALL use secure API keys
- **NFR-SEC5**: Mem0 and Qdrant connections SHALL use TLS encryption

### Reliability

- **NFR-R1**: SHALL achieve 99.9% uptime for memory operations
- **NFR-R2**: SHALL handle Qdrant/Mem0 unavailability gracefully (cache fallback)
- **NFR-R3**: SHALL implement retry logic for transient failures
- **NFR-R4**: SHALL support backup and recovery of memory data
- **NFR-R5**: Compression failures SHALL fallback to less aggressive compression

### Data Retention

- **NFR-DR1**: Memories SHALL persist indefinitely unless explicitly deleted
- **NFR-DR2**: SHALL support configurable memory TTL per scope
- **NFR-DR3**: SHALL implement automated cleanup of expired memories
- **NFR-DR4**: SHALL provide memory export functionality
- **NFR-DR5**: Stage summaries SHALL be backed up before compression

---

## 4. Features & Flows

### Feature 1: Memory Addition with Mem0 Extraction (Priority: P0)

**Description**: Store new memory entries with automatic Mem0 extraction and stage assignment.

**Key Flow**:

1. Client calls `add_memory(content, agent_id, session_id, task_id, metadata)`
2. MemoryManager validates input and scope
3. StageManager determines current stage for task_id
4. Mem0 extracts memory and generates embeddings
5. Memory stored in appropriate layer (episodic/semantic/procedural) in Qdrant
6. Memory linked to current stage
7. PostgreSQL metadata updated
8. Redis cache updated for working memory
9. Memory ID returned to caller

**Input**: `content` (str), `agent_id`, `session_id`, `task_id`, `metadata` (dict)
**Output**: `memory_id` (str)

### Feature 2: Stage-Aware Semantic Search (Priority: P0)

**Description**: Search memories with Mem0 semantic search + COMPASS stage awareness.

**Key Flow**:

1. Client submits `MemorySearchQuery(query, agent_id, current_stage, limit)`
2. MemoryManager validates query and scope
3. Mem0 generates query embedding
4. Qdrant performs vector similarity search across specified layers
5. Results enhanced with multi-factor importance scoring:
   - Embedding similarity (Mem0)
   - Recency decay
   - Access frequency
   - Stage relevance boost (COMPASS)
   - Criticality boost
   - Error correction relevance
6. Top-K results returned as `MemorySearchResult[]`

**Input**: `MemorySearchQuery(query, limit, agent_id, session_id, current_stage, layers, scoring_weights)`
**Output**: `list[MemorySearchResult]` with importance scores and metadata

### Feature 3: Stage Completion with Compression (Priority: P0 - COMPASS)

**Description**: Complete reasoning stage and compress raw memories into summary.

**Key Flow**:

1. Client requests `complete_stage(task_id, stage_id)` or automatic stage transition detected
2. StageManager retrieves all raw memories for stage_id
3. ContextCompressor compresses memories using gpt-4.1-mini:
   - Target: 10:1 compression ratio
   - Extract critical facts
   - Generate stage insights
4. Validate compression quality (≥95% fact retention)
5. Store StageMemory in PostgreSQL with references to raw memories
6. Update TaskContext with completed stage
7. Track compression metrics (cost, ratio, quality)
8. Return compressed summary

**Input**: `task_id` (str), `stage_id` (str)
**Output**: `StageMemory(summary, insights, compression_ratio, quality_metrics, cost)`

### Feature 4: Task Progress Summary (Priority: P0 - COMPASS)

**Description**: Generate progressive task summary from stage summaries.

**Key Flow**:

1. Client requests `get_task_progress(task_id)`
2. Retrieve all completed StageMemory records for task
3. ContextCompressor compresses stage summaries using gpt-4.1-mini:
   - Target: 5:1 compression ratio
   - Preserve critical constraints
   - Track performance metrics
4. Update TaskContext with progressive summary
5. Return task progress brief

**Input**: `task_id` (str)
**Output**: `TaskContext(progress_summary, current_stage, completed_stages, critical_constraints, performance_metrics)`

### Feature 5: Error Tracking and Pattern Detection (Priority: P1 - COMPASS)

**Description**: Record errors explicitly and detect recurring patterns.

**Key Flow**:

1. Agent or ACE records error via `record_error(error_record)`
2. ErrorTracker stores ErrorRecord with full context
3. ErrorTracker analyzes recent errors for patterns:
   - Group by error_type
   - Extract common context using Mem0
   - Calculate pattern severity
4. Mark error-related memories as critical
5. Generate error recovery recommendations
6. Return error patterns

**Input**: `ErrorRecord(task_id, stage_id, error_type, description, context, severity)`
**Output**: `list[ErrorPattern]` with recommendations

### Feature 6: ACE Strategic Context Query (Priority: P1 - COMPASS)

**Description**: Provide strategic context for ACE Meta-Thinker intervention decisions.

**Key Flow**:

1. ACE queries `get_strategic_context(task_id, decision_type, performance_metrics)`
2. MemoryManager analyzes metrics to determine context needs:
   - High error rate → return error history + prevention knowledge
   - Slow progress → return successful patterns + strategies
   - Low coherence → return critical facts + constraints
3. Retrieve relevant memories with enhanced importance scoring
4. Format context optimized for ACE decision
5. Return strategic context brief

**Input**: `task_id`, `decision_type`, `performance_metrics`
**Output**: `StrategicContext(relevant_memories, error_patterns, critical_facts, recommended_action)`

### Feature 7: Memory-Aware Message Routing (Priority: P1)

**Description**: Enhance message routing with memory context using Mem0 search.

**Integration with MessageRouter**:

1. Router receives message envelope with session_id
2. Router queries `search_memory(content, session_id, limit=5)` via Mem0
3. Relevant memories inform routing decision
4. Agent selection considers memory context (preferences, capabilities)

**Input**: Message envelope + session context
**Output**: Enhanced routing decision

---

## 5. Acceptance Criteria

### Definition of Done

**Infrastructure:**

- [ ] Mem0 integrated with AgentCore services
- [ ] Qdrant vector database deployed and operational
- [ ] PostgreSQL schema with stage_memories, task_contexts, error_records tables
- [ ] Redis working memory cache operational

**Core Features:**

- [ ] Four-layer memory hierarchy (Working/Episodic/Semantic/Procedural) implemented
- [ ] COMPASS hierarchical organization (raw → stage → task) operational
- [ ] Test-time scaling compression with gpt-4.1-mini functional
- [ ] Error tracking and pattern detection working
- [ ] Enhanced retrieval with multi-factor scoring operational

**Integration:**

- [ ] SessionManager memory integration complete
- [ ] MessageRouter context-aware routing functional
- [ ] TaskManager artifact storage working
- [ ] ACE strategic context interface implemented

**Quality:**

- [ ] Unit tests achieving 90%+ coverage
- [ ] Integration tests with Qdrant and Mem0 passing
- [ ] COMPASS validation tests passing (60-80% context reduction, 70-80% cost reduction, 95%+ fact retention)
- [ ] Performance targets validated (p95 <100ms search, <5s compression)

**Operations:**

- [ ] JSON-RPC methods registered (memory.*, all COMPASS methods)
- [ ] CLI memory commands operational
- [ ] Prometheus metrics instrumented
- [ ] Documentation complete with usage examples
- [ ] Model governance enforced (gpt-4.1-mini for compression only)

### Validation Approach

**Unit Testing**:

- Mock Mem0 client for fast tests
- Test memory scoping logic
- Test multi-factor scoring algorithms
- Test compression quality validation
- Test error pattern detection
- Test cache behavior

**Integration Testing**:

- Real Qdrant instance with testcontainers
- Real Mem0 integration
- Test memory persistence across restarts
- Test semantic search accuracy
- Test stage compression pipeline
- Test error tracking workflow
- Test multi-scope hierarchy

**Performance Testing**:

- Benchmark memory addition latency
- Benchmark Mem0 search latency with 10K+ memories
- Benchmark compression throughput and cost
- Test cache hit rates
- Load test 1000 concurrent operations

**COMPASS Validation Testing**:

- Context efficiency: Validate 60-80% reduction
- Cost reduction: Validate 70-80% savings
- Compression quality: Validate 95%+ fact retention
- Long-horizon accuracy: Validate 20% improvement
- Stage compression ratio: Validate 10:1 (±20%)
- Task compression ratio: Validate 5:1 (±20%)

**Functional Testing**:

- Test cross-session memory continuity (Mem0)
- Test agent preference learning
- Test context-aware routing
- Test memory cleanup
- Test ACE strategic context queries
- Test error-aware retrieval

---

## 6. Dependencies

### Technical Stack

- **Core**: Python 3.12+, FastAPI, Pydantic, asyncio
- **Memory Layer**: mem0 ^0.1.0
- **Vector DB**: Qdrant (via qdrant-client)
- **Cache**: Redis (existing AgentCore infrastructure)
- **Database**: PostgreSQL (existing AgentCore infrastructure)
- **Embeddings**: OpenAI text-embedding-3-small (via Mem0)
- **Compression**: OpenAI gpt-4.1-mini (test-time scaling)
- **Reasoning**: OpenAI gpt-4.1 or gpt-5 (agent tasks only)

### External Integrations

**Qdrant**: Vector database for memory embeddings

- Connection: `QDRANT_URL` (<http://localhost:6333> for dev)
- Auth: `QDRANT_API_KEY` (optional for local, required for cloud)
- Collection: `agentcore_memories`

**Mem0**: Memory extraction and management

- LLM: Uses approved model from `LLM_DEFAULT_MODEL`
- Embedder: OpenAI text-embedding-3-small
- Config: Integrated with AgentCore settings

**OpenAI**: Embeddings and compression

- API Key: `OPENAI_API_KEY`
- Embedding Model: text-embedding-3-small ($0.02/1M tokens)
- Compression Model: gpt-4.1-mini ($0.15/1M tokens)

### Infrastructure Requirements

**Qdrant Deployment**: Docker container or K8s deployment

- Storage: Persistent volume (20GB minimum)
- Resources: 2GB RAM, 1 CPU core
- Ports: 6333 (HTTP), 6334 (gRPC)

**Redis**: Existing AgentCore Redis instance

- Additional memory: ~500MB for working memory cache
- TTL: 1 hour for working memory
- Separate DB: `REDIS_MEMORY_DB=1`

**PostgreSQL**: Existing AgentCore PostgreSQL instance

- Additional storage: ~10GB for stage summaries and error records
- New tables: stage_memories, task_contexts, error_records, compression_metrics

### Related Components

- **SessionManager**: Integrates memory for session context
- **AgentManager**: Stores agent preferences and learnings
- **MessageRouter**: Uses memory for context-aware routing
- **TaskManager**: Stores task artifacts in session memory, provides task_id
- **LLMService**: Required for Mem0 extraction and compression (approved models)
- **ACE (Future)**: Consumes strategic context for meta-cognitive monitoring

### Technical Assumptions

- Qdrant deployment available and accessible
- Mem0 library stable at v0.1.0
- OpenAI API key for embeddings and compression
- Sufficient storage for long-term memory retention
- Network latency to Qdrant <10ms (same datacenter)

---

## 7. Implementation Notes

### Component Structure

```
src/agentcore/a2a_protocol/
├── models/
│   └── memory.py                  # MemoryLayer enum, MemoryRecord, StageMemory,
│                                  # TaskContext, ErrorRecord, MemorySearchQuery
├── services/
│   ├── memory/
│   │   ├── manager.py             # MemoryManager (orchestration, Mem0 integration)
│   │   ├── stage_manager.py       # StageManager (COMPASS hierarchical org)
│   │   ├── compression.py         # ContextCompressor (test-time scaling)
│   │   ├── retrieval.py           # EnhancedRetrieval (multi-factor scoring)
│   │   ├── error_tracker.py       # ErrorTracker (pattern detection)
│   │   ├── working_memory.py      # WorkingMemoryService (Redis)
│   │   ├── mem0_client.py         # Mem0Client (wrapper for mem0 library)
│   │   └── cost_tracker.py        # CostTracker (compression cost monitoring)
│   └── memory_jsonrpc.py          # JSON-RPC methods (memory.*)
├── cache/
│   └── memory_cache.py            # Redis cache adapter
└── database/
    ├── models.py                  # SQLAlchemy models (StageMemoryModel, TaskContextModel, ErrorModel)
    └── repositories.py            # MemoryRepository, StageMemoryRepository, ErrorRepository
```

### Configuration

```python
# config.py additions

# Qdrant (Episodic, Semantic, Procedural layers)
QDRANT_URL: str = "http://localhost:6333"
QDRANT_API_KEY: str | None = None
QDRANT_COLLECTION_NAME: str = "agentcore_memories"

# Redis (Working Memory layer)
REDIS_MEMORY_DB: int = 1  # Separate DB for memory cache
WORKING_MEMORY_TTL: int = 3600  # 1 hour
WORKING_MEMORY_MAX_TOKENS: int = 4000  # 2-4K tokens

# Mem0 Configuration
MEM0_EMBEDDING_MODEL: str = "text-embedding-3-small"
MEM0_LLM_MODEL: str = "gpt-4.1-mini"  # For extraction only
MEMORY_MAX_CONTEXT_LENGTH: int = 10000

# Memory Layer Configuration
EPISODIC_MEMORY_MAX_EPISODES: int = 50  # Recent episodes to retain
SEMANTIC_MEMORY_CAPACITY: int = 1000  # Max facts before pruning
PROCEDURAL_MEMORY_MIN_OCCURRENCES: int = 3  # Min pattern repetitions

# COMPASS: Test-Time Scaling
COMPRESSION_MODEL: str = "gpt-4.1-mini"  # $0.15/1M tokens
REASONING_MODEL: str = "gpt-4.1"  # Reserve for agent tasks
STAGE_COMPRESSION_RATIO_TARGET: float = 10.0  # 10:1
TASK_COMPRESSION_RATIO_TARGET: float = 5.0  # 5:1
COMPRESSION_QUALITY_THRESHOLD: float = 0.95  # 95% fact retention

# COMPASS: Memory Scoring Weights
MEMORY_RELEVANCE_WEIGHT: float = 0.35  # Embedding similarity
MEMORY_RECENCY_WEIGHT: float = 0.15  # Temporal decay
MEMORY_FREQUENCY_WEIGHT: float = 0.10  # Access count
MEMORY_STAGE_RELEVANCE_WEIGHT: float = 0.20  # Current stage
MEMORY_CRITICALITY_WEIGHT: float = 0.10  # Critical boost
MEMORY_ERROR_CORRECTION_WEIGHT: float = 0.10  # Error relevance

# COMPASS: Stage Management
STAGE_AUTO_TRANSITION: bool = True
STAGE_COMPLETION_AUTO_COMPRESS: bool = True
ERROR_TRACKING_ENABLED: bool = True
ERROR_PATTERN_DETECTION_THRESHOLD: int = 2  # Min occurrences

# Cost Control
MONTHLY_TOKEN_BUDGET_USD: float = 1000.0
ALERT_THRESHOLD_PCT: float = 75.0
```

### Infrastructure Setup

**Docker Compose (dev)**:

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_data:/qdrant/storage
```

**Kubernetes (prod)**:

- StatefulSet with persistent volume
- Service exposing ports 6333, 6334
- PVC with 20GB storage
- Resource limits: 4GB RAM, 2 CPU

### Timeline

**Week 1-2: Infrastructure & Core (P0)**

- Qdrant deployment and Mem0 integration
- Database schema (stage_memories, task_contexts, error_records)
- Basic memory models and repositories
- Redis working memory cache implementation
- MemoryManager orchestration layer

**Week 3: Stage Management (P0 - COMPASS MEM-1)**

- StageManager implementation
- Stage detection and transitions
- Stage-to-memory linking
- Basic stage compression preparation

**Week 4: Compression Pipeline (P0 - COMPASS MEM-2)**

- ContextCompressor with gpt-4.1-mini
- Stage compression (10:1 ratio)
- Task compression (5:1 ratio)
- Compression quality validation
- Cost tracking

**Week 5: Enhanced Retrieval (P0 - COMPASS MEM-4)**

- Multi-factor importance scoring
- Stage-aware retrieval
- Critical memory identification
- Retrieval quality validation
- Integration with Mem0 search

**Week 6: Error Tracking (P1 - COMPASS MEM-3)**

- ErrorTracker implementation
- Pattern detection algorithms
- Error-aware retrieval
- ACE integration interface

**Week 7-8: Integration & Testing**

- Service integrations (SessionManager, MessageRouter, TaskManager)
- JSON-RPC methods and CLI commands
- Integration testing with Mem0 and Qdrant
- Performance validation and optimization
- COMPASS validation (context reduction, cost savings, quality)
- Documentation and deployment

---

## 8. References

**Source Documentation:**

- `docs/specs/memory-service/spec.md` - Original Mem0 integration spec
- `docs/specs/memory-system/spec.md` - COMPASS enhancement spec
- `docs/research/evolving-memory-system.md` - Four-layer architecture research
- `.docs/research/compass-enhancement-analysis.md` - COMPASS paper analysis

**External Documentation:**

- Mem0 Documentation: <https://docs.mem0.ai/>
- Qdrant Documentation: <https://qdrant.tech/documentation/>
- COMPASS Paper: <https://arxiv.org/abs/2510.08790>
- OpenAI Embeddings: <https://platform.openai.com/docs/guides/embeddings>
- OpenAI Models (Pricing): <https://openai.com/api/pricing/>
- Redis Documentation: <https://redis.io/docs/>

**Related AgentCore Documentation:**

- AgentCore Architecture: `docs/agentcore-architecture-and-development-plan.md`
- Embedding Service: `src/agentcore/a2a_protocol/services/embedding_service.py`
- Task Management: `src/agentcore/a2a_protocol/services/task_manager.py`
- JSON-RPC Handler: `src/agentcore/a2a_protocol/services/jsonrpc_handler.py`
