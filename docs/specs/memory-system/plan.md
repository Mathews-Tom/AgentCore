# Memory System Implementation Blueprint (PRP) - COMPASS Enhanced

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-10-23
**Version:** 2.0 (COMPASS-Enhanced)
**Specification:** `docs/specs/memory-system/spec.md` v2.0
**Research:**
- `docs/research/evolving-memory-system.md`
- `.docs/research/compass-enhancement-analysis.md`

---

## 📖 Context & Documentation

### Traceability Chain

**Feature Request → Research → COMPASS Analysis → Enhanced Specification → This Plan**

1. **Original Research & Technical Analysis:** docs/research/evolving-memory-system.md
   - Four-layer memory architecture (Working, Episodic, Semantic, Procedural)
   - Memory operations (encode, store, retrieve, update, prune)
   - Vector embedding strategy with similarity search
   - Performance targets: 80% context reduction, 25-30% task improvement
   - Technology evaluation: PGVector, OpenAI embeddings, Redis caching

2. **COMPASS Enhancement Analysis:** .docs/research/compass-enhancement-analysis.md
   - **Key Finding:** 20% accuracy improvement through context management
   - **Core Insight:** Hierarchical, evolving context with stage-aware compression
   - **MEM Enhancements:**
     - MEM-1: Hierarchical Memory Organization (3 levels: raw → stage → task)
     - MEM-2: Evolving Context Compression with Test-Time Scaling
     - MEM-3: Error Memory and Pattern Tracking
     - MEM-4: Enhanced Retrieval with Criticality Scoring
   - **Validation:** 60-80% context reduction, 70-80% cost reduction via test-time scaling

3. **Enhanced Specification:** docs/specs/memory-system/spec.md v2.0
   - COMPASS-enhanced functional requirements (FR-1 through FR-7)
   - Stage-aware memory organization with 4 reasoning stages
   - Test-time scaling: gpt-4o-mini for compression, gpt-4.1 for reasoning
   - Error tracking with pattern detection
   - Enhanced retrieval with multi-factor importance scoring
   - 8 acceptance criteria with COMPASS validation targets

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
- **ACE Integration (Future):** ACE Meta-Thinker will query MEM Context Manager for strategic decisions

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable agents to maintain contextual awareness across multi-turn interactions through **hierarchical, stage-aware memory management** inspired by COMPASS architecture.

**Value Proposition (COMPASS-Enhanced):**

- **60-80% Context Efficiency**: COMPASS-validated context reduction through hierarchical compression
- **20% Performance Gain**: COMPASS-demonstrated improvement on long-horizon tasks (GAIA benchmark)
- **70-80% Cost Reduction**: Test-time scaling with gpt-4o-mini ($0.15/1M tokens) for compression
- **Error Prevention**: Explicit error tracking prevents compounding mistakes
- **Stage-Aware Retrieval**: Context organized by reasoning stage (planning, execution, reflection, verification)
- **Unbounded Conversations**: No practical limit on interaction depth with progressive summarization

**Target Users:**

- **Agent Developers**: Building complex multi-step workflows with long-horizon tasks
- **AgentCore Operators**: Managing agent systems with error-aware context management
- **End Users**: Benefiting from agents that learn from errors and maintain critical context

### Technical Approach

**Architecture Pattern:** COMPASS Context Manager with 3-level hierarchical memory

- **Level 1: Raw Memories** - Individual interactions (episodic, semantic, procedural)
- **Level 2: Stage Memories** - Compressed stage summaries (10:1 compression ratio)
- **Level 3: Task Context** - Progressive task summary (5:1 compression from stages)

**COMPASS-Inspired Innovations:**

1. **Hierarchical Memory Organization (MEM-1):**
   - Stage-aware memory with 4 reasoning stages: planning, execution, reflection, verification
   - Progressive summarization: raw → stage (10:1) → task (5:1)
   - Cross-stage references for coherence

2. **Test-Time Scaling (MEM-2):**
   - Use gpt-4o-mini ($0.15/1M tokens) for compression and summarization
   - Reserve gpt-4.1 for agent reasoning tasks
   - Achieve 70-80% cost reduction while maintaining quality

3. **Error-Aware Memory (MEM-3):**
   - Explicit error tracking with severity scoring
   - Pattern detection to prevent compounding mistakes
   - Error context preserved for recovery actions

4. **Enhanced Retrieval (MEM-4):**
   - Multi-factor importance: embedding similarity + recency + frequency + stage relevance + criticality
   - Error-aware retrieval: boost recent error memories
   - Stage-specific context formatting

**Technology Stack (COMPASS-Enhanced):**

- **Vector Storage**: PostgreSQL with PGVector extension
- **Embedding Generation**: Existing embedding_service.py (OpenAI text-embedding-3-small)
- **Compression Model**: gpt-4o-mini (cost-optimized for summarization)
- **Reasoning Model**: gpt-4.1 (agent tasks only)
- **Caching**: Redis for working memory with TTL-based expiration
- **Archival**: S3-compatible storage for pruned memories (optional, Phase 6)

**Implementation Strategy:**

- Phase 1 (Weeks 1-3): Foundation + Hierarchical Organization (MEM-1)
- Phase 2 (Weeks 4-5): Test-Time Scaling Compression (MEM-2)
- Phase 3 (Week 6): Error Tracking (MEM-3)
- Phase 4 (Week 7): Enhanced Retrieval (MEM-4)
- Phase 5 (Week 8): ACE Integration Layer
- Phase 6 (Weeks 9-10): Optimization + COMPASS Validation

### Key Success Metrics (COMPASS-Validated)

**Service Level Objectives (SLOs):**

- Availability: 99.9% (memory operations non-blocking)
- Response Time: <100ms (p95 retrieval latency)
- Throughput: 100+ concurrent retrievals without degradation
- Error Rate: <0.1% (excluding embedding service failures)

**Key Performance Indicators (KPIs) - COMPASS Targets:**

| Metric | Target | COMPASS Validation |
|--------|--------|-------------------|
| Context Efficiency | 60-80% reduction | ✅ COMPASS achieved 60-80% |
| Long-Horizon Accuracy | +20% improvement | ✅ COMPASS validated on GAIA |
| Cost Reduction | 70-80% reduction | ✅ Test-time scaling |
| Retrieval Precision | 95%+ relevant | ⬆️ Enhanced from 90% baseline |
| Compression Quality | 10:1 stage, 5:1 task | ✅ COMPASS methodology |
| Error Recall | 90%+ critical errors | 🆕 COMPASS error tracking |

---

## 💻 Code Examples & Patterns

### COMPASS-Enhanced Data Models

**1. Hierarchical Memory Models (MEM-1):**

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class StageMemory(BaseModel):
    """Stage-level memory with progressive summarization (MEM-1)."""

    stage_id: UUID
    task_id: UUID
    agent_id: UUID
    stage_type: str  # "planning", "execution", "reflection", "verification"

    # Compressed stage summary (10:1 ratio)
    stage_summary: str  # Generated by gpt-4o-mini
    stage_insights: list[str] = Field(default_factory=list)

    # References to raw memories
    raw_memory_refs: list[UUID]

    # Metadata
    relevance_score: float = 1.0
    compression_ratio: float  # Actual achieved ratio
    compression_model: str = "gpt-4o-mini"

    # Timestamps
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

class TaskContext(BaseModel):
    """Task-level context with progressive summarization (MEM-1)."""

    task_id: UUID
    agent_id: UUID
    task_goal: str

    # Current stage
    current_stage_id: UUID | None = None

    # Progressive task summary (5:1 compression from stages)
    task_progress_summary: str  # Generated by gpt-4o-mini

    # Critical constraints
    critical_constraints: list[str] = Field(default_factory=list)

    # Performance metrics (from ACE integration)
    performance_metrics: dict[str, float] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime
    updated_at: datetime

class ErrorRecord(BaseModel):
    """Error memory for pattern tracking (MEM-3)."""

    error_id: UUID
    task_id: UUID
    stage_id: UUID | None
    agent_id: UUID

    # Error details
    error_type: str  # "logic", "execution", "validation", "constraint"
    error_description: str
    context_when_occurred: str

    # Recovery
    recovery_action: str | None = None

    # Severity
    error_severity: float = Field(ge=0.0, le=1.0)

    # Timestamp
    recorded_at: datetime
```

**2. Test-Time Scaling Configuration (MEM-2):**

```python
from pydantic import BaseModel

class CompressionConfig(BaseModel):
    """Configuration for test-time scaling compression."""

    # Model selection
    compression_model: str = "gpt-4o-mini"  # $0.15/1M tokens
    reasoning_model: str = "gpt-4.1"  # Reserve for agent tasks

    # Compression targets
    stage_compression_ratio: float = 10.0  # 10:1 for stage summaries
    task_compression_ratio: float = 5.0   # 5:1 for task summaries

    # Quality thresholds
    min_information_retention: float = 0.95  # 95% information preserved
    max_tokens_per_stage: int = 500
    max_tokens_per_task: int = 2000

    # Cost optimization
    use_caching: bool = True
    cache_ttl_hours: int = 24

# Example usage
compression_config = CompressionConfig(
    compression_model="gpt-4o-mini",
    stage_compression_ratio=10.0,
    task_compression_ratio=5.0
)
```

**3. Enhanced Retrieval with Multi-Factor Scoring (MEM-4):**

```python
import math
from datetime import datetime

def calculate_enhanced_importance_score(
    memory: MemoryRecord,
    query_embedding: list[float],
    current_time: datetime,
    current_stage: str,
    recent_errors: list[ErrorRecord]
) -> float:
    """
    COMPASS-enhanced importance scoring with stage awareness and error criticality.

    Factors:
    1. Embedding similarity (40% weight)
    2. Recency (20% weight)
    3. Frequency (15% weight)
    4. Stage relevance (15% weight)
    5. Criticality boost (10% weight)
    """

    # 1. Relevance: Cosine similarity from embedding
    relevance_score = 1 - cosine_distance(query_embedding, memory.embedding)

    # 2. Recency: Exponential decay with 24-hour half-life
    age_hours = (current_time - memory.timestamp).total_seconds() / 3600
    recency_score = math.exp(-age_hours / 24)

    # 3. Frequency: Normalized access count (cap at 10)
    frequency_score = min(memory.access_count / 10, 1.0)

    # 4. Stage Relevance: Boost if memory from current stage (COMPASS MEM-1)
    stage_relevance = 1.0
    if memory.stage_id and memory.stage_relevance_map:
        stage_relevance = memory.stage_relevance_map.get(current_stage, 0.5)

    # 5. Criticality Boost: Boost if memory is critical or error-related (COMPASS MEM-3)
    criticality_boost = 1.0
    if memory.is_critical:
        criticality_boost = 1.5  # 50% boost for critical memories

    # Check if related to recent errors
    if memory.memory_id in [err.memory_id for err in recent_errors]:
        criticality_boost = max(criticality_boost, 1.3)  # 30% boost for error context

    # Weighted combination
    base_importance = (
        0.40 * relevance_score +
        0.20 * recency_score +
        0.15 * frequency_score +
        0.15 * stage_relevance +
        0.10 * criticality_boost
    )

    return base_importance
```

**4. Stage-Aware Context Compression (MEM-2):**

```python
async def compress_stage_memories(
    self,
    stage_memories: list[MemoryRecord],
    stage_type: str,
    target_ratio: float = 10.0,
    compression_model: str = "gpt-4o-mini"
) -> StageMemory:
    """
    Compress raw memories into stage summary using test-time scaling.

    Uses gpt-4o-mini for cost-effective compression (70-80% cost reduction).
    """
    # Concatenate raw memory content
    raw_content = "\n\n".join([
        f"[{mem.timestamp.isoformat()}] {mem.summary}"
        for mem in stage_memories
    ])

    # Count original tokens
    original_tokens = self.count_tokens(raw_content)
    target_tokens = int(original_tokens / target_ratio)

    # Compression prompt for gpt-4o-mini
    compression_prompt = f"""
    Compress the following {stage_type} stage memories into a concise summary.

    Original content ({original_tokens} tokens):
    {raw_content}

    Requirements:
    1. Target length: {target_tokens} tokens (10:1 compression)
    2. Preserve critical information: key decisions, actions, outcomes
    3. Extract stage insights: patterns, learnings, constraints discovered
    4. Maintain temporal coherence
    5. Flag any errors or issues encountered

    Output format:
    {{
        "stage_summary": "compressed summary here",
        "stage_insights": ["insight 1", "insight 2"],
        "critical_items": ["critical item 1"]
    }}
    """

    # Call gpt-4o-mini for compression (cost-optimized)
    response = await self.llm_client.generate(
        model=compression_model,
        prompt=compression_prompt,
        max_tokens=target_tokens + 100,  # Buffer for JSON structure
        temperature=0.3  # Low temperature for factual compression
    )

    # Parse response
    compressed_data = json.loads(response)

    # Calculate actual compression ratio
    compressed_tokens = self.count_tokens(compressed_data["stage_summary"])
    actual_ratio = original_tokens / compressed_tokens

    # Create StageMemory
    stage_memory = StageMemory(
        stage_id=uuid.uuid4(),
        task_id=stage_memories[0].task_id,
        agent_id=stage_memories[0].agent_id,
        stage_type=stage_type,
        stage_summary=compressed_data["stage_summary"],
        stage_insights=compressed_data["stage_insights"],
        raw_memory_refs=[mem.memory_id for mem in stage_memories],
        compression_ratio=actual_ratio,
        compression_model=compression_model,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )

    return stage_memory
```

**5. Error Pattern Detection (MEM-3):**

```python
async def detect_error_patterns(
    self,
    task_id: UUID,
    lookback_stages: int = 3
) -> list[dict[str, Any]]:
    """
    Detect recurring error patterns to prevent compounding mistakes.

    COMPASS insight: Agents fail when they don't learn from past errors.
    """
    # Retrieve recent errors
    async with get_session() as session:
        repo = ErrorRepository(session)
        recent_errors = await repo.get_by_task(
            task_id=task_id,
            limit=lookback_stages * 10  # ~10 errors per stage
        )

    # Group by error_type
    error_groups: dict[str, list[ErrorRecord]] = {}
    for error in recent_errors:
        error_groups.setdefault(error.error_type, []).append(error)

    # Detect patterns
    patterns = []
    for error_type, errors in error_groups.items():
        if len(errors) >= 2:  # Pattern threshold
            # Extract common context
            contexts = [err.context_when_occurred for err in errors]
            common_context = self._extract_common_phrases(contexts)

            # Calculate pattern severity
            avg_severity = sum(err.error_severity for err in errors) / len(errors)

            patterns.append({
                "error_type": error_type,
                "occurrences": len(errors),
                "avg_severity": avg_severity,
                "common_context": common_context,
                "recommendation": self._generate_error_recommendation(error_type, errors)
            })

    return patterns
```

### Repository Patterns (Enhanced)

**6. Stage Memory Repository (MEM-1):**

```python
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

class StageMemoryRepository:
    """Data access layer for stage memories."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, stage_memory: StageMemoryModel) -> StageMemoryModel:
        """Create a new stage memory."""
        self.session.add(stage_memory)
        await self.session.commit()
        await self.session.refresh(stage_memory)
        return stage_memory

    async def get_by_task_and_stage(
        self,
        task_id: UUID,
        stage_type: str,
        agent_id: UUID
    ) -> StageMemoryModel | None:
        """Get stage memory for specific task and stage."""
        stmt = select(StageMemoryModel).where(
            and_(
                StageMemoryModel.task_id == task_id,
                StageMemoryModel.stage_type == stage_type,
                StageMemoryModel.agent_id == agent_id
            )
        ).order_by(desc(StageMemoryModel.created_at))

        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_all_stages_for_task(
        self,
        task_id: UUID,
        agent_id: UUID
    ) -> list[StageMemoryModel]:
        """Get all stage memories for a task (for task-level compression)."""
        stmt = select(StageMemoryModel).where(
            and_(
                StageMemoryModel.task_id == task_id,
                StageMemoryModel.agent_id == agent_id,
                StageMemoryModel.completed_at.isnot(None)
            )
        ).order_by(StageMemoryModel.created_at)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())
```

### Anti-Patterns to Avoid

**From AgentCore Conventions + COMPASS Learnings:**

- ❌ Do not use expensive models (gpt-4.1) for compression tasks
- ❌ Do not ignore error context when retrieving memories
- ❌ Do not flatten hierarchical structure (loses stage coherence)
- ❌ Do not compress across stage boundaries (violates COMPASS principle)
- ❌ Do not use uniform importance scoring (stage-awareness critical)
- ❌ Do not skip validation of compression quality (information loss risk)

---

## 🔧 Technology Stack

### Recommended Stack (COMPASS-Enhanced)

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.12+ | AgentCore standard, modern typing |
| Framework | FastAPI | Latest | Existing AgentCore framework |
| Database | PostgreSQL | 14+ | Existing AgentCore database |
| Vector Extension | PGVector | 0.5.0+ | Vector similarity search |
| Embeddings | OpenAI API | text-embedding-3-small | Via existing embedding_service.py |
| **Compression Model** | **gpt-4o-mini** | **Latest** | **$0.15/1M tokens, 70-80% cost savings** |
| **Reasoning Model** | **gpt-4.1** | **Latest** | **Reserve for agent tasks only** |
| Cache | Redis | 6+ | Existing AgentCore cache |
| ORM | SQLAlchemy | 2.0+ (async) | Existing AgentCore pattern |
| Migrations | Alembic | Latest | Existing AgentCore migration tool |
| Validation | Pydantic | 2.0+ | Existing AgentCore pattern |
| Testing | pytest-asyncio | Latest | AgentCore standard, 90%+ coverage |
| Token Counting | tiktoken | Latest | OpenAI token counter for compression |
| Archival | boto3 (S3) | Latest | Industry standard for object storage |

### Key Technology Decisions (COMPASS-Informed)

**1. Test-Time Scaling: gpt-4o-mini for Compression**

- **Rationale:** COMPASS demonstrates 70-80% cost reduction by using smaller models for compression
- **Cost Analysis:**
  - gpt-4.1: ~$3.00/1M tokens (reasoning tasks)
  - gpt-4o-mini: $0.15/1M tokens (compression tasks)
  - **Savings: 95% per compression operation**
- **Quality:** COMPASS validates compression quality remains high (95%+ information retention)
- **Use Cases:**
  - Stage summarization (10:1 compression)
  - Task summarization (5:1 compression)
  - Entity extraction
  - Insight generation

**2. Hierarchical Storage: 3-Level Architecture**

- **Rationale:** COMPASS demonstrates benefits of structured, stage-aware context
- **Implementation:**
  - Level 1: Raw memories (memories table)
  - Level 2: Stage memories (stage_memories table)
  - Level 3: Task context (task_contexts table)
- **Trade-off:** Additional storage overhead vs dramatic context efficiency gains
- **COMPASS Validation:** 60-80% context reduction while maintaining coherence

**3. Error-Aware Retrieval: Explicit Error Tracking**

- **Rationale:** COMPASS shows agents fail when errors are forgotten/overlooked
- **Implementation:** Dedicated error_records table with pattern detection
- **Integration:** Error memories get criticality boost in retrieval scoring
- **Trade-off:** Additional storage/complexity vs prevention of compounding mistakes

**4. PGVector over External Vector DB**

- **Rationale:** Simpler deployment, sufficient performance (<100ms at 1M scale)
- **Trade-off:** Slightly lower throughput vs eliminated external dependency
- **Alignment:** Consistent with original research decision

---

## 🏗️ Architecture Design

### COMPASS Context Manager Architecture

**System Design (COMPASS-Enhanced):**

```plaintext
┌─────────────────────────────────────────────────────────────────────┐
│                    AgentCore System (COMPASS-Enhanced)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐         ┌──────────────────────────────┐          │
│  │   Agent     │ query   │   Memory System              │          │
│  │   Runtime   ├────────>│   (COMPASS Context Manager)  │          │
│  │             │         │                              │          │
│  └─────────────┘         │  ┌────────────────────────┐  │          │
│         │                │  │   MemoryManager        │  │          │
│         │ store          │  │   (Orchestration)      │  │          │
│         v                │  └──────────┬─────────────┘  │          │
│  ┌─────────────┐         │             │                │          │
│  │   Task      │         │  ┌──────────v─────────────┐  │          │
│  │   Manager   │         │  │  StageManager          │  │          │
│  └─────────────┘         │  │  (MEM-1: Hierarchical) │  │          │
│                          │  └──────────┬─────────────┘  │          │
│  ┌─────────────┐         │             │                │          │
│  │  ACE        │<────────┤  ┌──────────v─────────────┐  │          │
│  │  (Future)   │ context │  │  ContextCompressor     │  │          │
│  │  Meta-Think │         │  │  (MEM-2: Test-Time     │  │          │
│  └─────────────┘         │  │   Scaling w/ 4o-mini)  │  │          │
│                          │  └──────────┬─────────────┘  │          │
│  ┌─────────────┐         │             │                │          │
│  │  Embedding  │<────────┤  ┌──────────v─────────────┐  │          │
│  │  Service    │ embed   │  │  EnhancedRetrieval     │  │          │
│  └─────────────┘         │  │  (MEM-4: Multi-Factor  │  │          │
│                          │  │   + Stage-Aware)       │  │          │
│  ┌─────────────┐         │  └──────────┬─────────────┘  │          │
│  │  JSON-RPC   │         │             │                │          │
│  │  Handler    │<────────┤  ┌──────────v─────────────┐  │          │
│  └──────┬──────┘         │  │  ErrorTracker          │  │          │
│         │                │  │  (MEM-3: Pattern       │  │          │
│         v                │  │   Detection)           │  │          │
│  ┌─────────────┐         │  └────────────────────────┘  │          │
│  │ PostgreSQL  │         └──────────────────────────────┘          │
│  │ + PGVector  │                                                   │
│  │             │         ┌──────────────────────────────┐          │
│  │ Tables:     │         │      Redis Cache             │          │
│  │ - memories  │         │  - Working memory (TTL)      │          │
│  │ - stage_mem │         │  - Embedding cache           │          │
│  │ - task_ctx  │         └──────────────────────────────┘          │
│  │ - errors    │                                                   │
│  └─────────────┘                                                   │
│                                                                     │
│  Hierarchical Memory Flow (COMPASS MEM-1):                         │
│  Raw Memories → Stage Summary (10:1) → Task Context (5:1)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown (COMPASS-Enhanced)

**Core Components:**

**1. MemoryManager (src/agentcore/memory/manager.py)**

- **Purpose:** Orchestrate all memory operations with COMPASS enhancements
- **New Methods:**
  - `complete_stage(task_id, stage_id)` → Trigger stage compression (MEM-1)
  - `get_stage_context(task_id, current_stage)` → Stage-aware retrieval (MEM-1)
  - `record_error(error_record)` → Track errors (MEM-3)
  - `detect_error_patterns(task_id)` → Pattern detection (MEM-3)
  - `get_strategic_context(task_id, decision_type)` → ACE integration (MEM-5)
- **Existing Methods (Enhanced):**
  - `add_interaction()` → Now assigns to current stage
  - `retrieve_memories()` → Now uses enhanced multi-factor scoring
  - `get_relevant_context()` → Now stage-aware formatting

**2. StageManager (src/agentcore/memory/stage_manager.py) - NEW**

- **Purpose:** Manage stage lifecycle and progressive compression (MEM-1)
- **Responsibilities:**
  - Track current stage for each task
  - Trigger stage compression on completion
  - Generate stage insights
  - Maintain stage-to-raw-memory mappings
- **Interfaces:**
  - `create_stage(task_id, stage_type)` → Initialize new stage
  - `add_to_stage(stage_id, memory_id)` → Link memory to stage
  - `complete_stage(stage_id)` → Compress and finalize stage
  - `get_stage_summary(stage_id)` → Retrieve compressed summary
- **Dependencies:** ContextCompressor, MemoryRepository, StageMemoryRepository

**3. ContextCompressor (src/agentcore/memory/compression.py) - ENHANCED**

- **Purpose:** Test-time scaling compression with gpt-4o-mini (MEM-2)
- **New Responsibilities:**
  - Stage-level compression (10:1 ratio)
  - Task-level compression (5:1 ratio)
  - Compression quality validation (95%+ information retention)
  - Cost tracking (tokens used per compression)
- **Interfaces:**
  - `compress_stage(stage_memories, stage_type)` → StageMemory
  - `compress_task(stage_summaries, task_goal)` → TaskContext
  - `validate_compression_quality(original, compressed)` → float (retention score)
  - `estimate_compression_cost(token_count)` → float (USD)
- **Dependencies:** LLM client (gpt-4o-mini), tiktoken

**4. EnhancedRetrievalService (src/agentcore/memory/retrieval.py) - ENHANCED**

- **Purpose:** Multi-factor importance scoring with stage awareness (MEM-4)
- **New Responsibilities:**
  - Stage relevance scoring
  - Criticality boost for important memories
  - Error-aware retrieval (boost recent error context)
  - Stage-specific context formatting
- **Interfaces:**
  - `calculate_enhanced_importance(memory, query_embedding, current_stage, recent_errors)` → float
  - `retrieve_stage_aware(query, current_stage, k)` → list[MemoryRecord]
  - `format_stage_context(memories, stage_type)` → str
- **Dependencies:** MemoryRepository, ErrorRepository

**5. ErrorTracker (src/agentcore/memory/error_tracker.py) - NEW**

- **Purpose:** Error recording and pattern detection (MEM-3)
- **Responsibilities:**
  - Record errors with full context
  - Detect recurring error patterns
  - Generate error recovery recommendations
  - Severity scoring
- **Interfaces:**
  - `record_error(error_record)` → None
  - `detect_patterns(task_id, lookback_stages)` → list[ErrorPattern]
  - `get_error_history(task_id)` → list[ErrorRecord]
  - `calculate_error_severity(error_type, context)` → float
- **Dependencies:** ErrorRepository, MemoryRepository

**6. StageMemoryRepository (src/agentcore/memory/database/repositories.py) - NEW**

- **Purpose:** Data access layer for stage memories
- **Interfaces:**
  - `create(stage_memory)` → StageMemoryModel
  - `get_by_task_and_stage(task_id, stage_type, agent_id)` → StageMemoryModel
  - `get_all_stages_for_task(task_id, agent_id)` → list[StageMemoryModel]
  - `update(stage_id, updates, agent_id)` → StageMemoryModel

**7. ErrorRepository (src/agentcore/memory/database/repositories.py) - NEW**

- **Purpose:** Data access layer for error records
- **Interfaces:**
  - `create(error_record)` → ErrorModel
  - `get_by_task(task_id, limit)` → list[ErrorModel]
  - `get_by_stage(stage_id)` → list[ErrorModel]
  - `get_patterns(task_id)` → list[dict]

---

## 🔧 Technical Specification

### Data Model (COMPASS-Enhanced)

**SQLAlchemy Models (Complete Schema):**

```sql
-- Stage memories table (COMPASS MEM-1)
CREATE TABLE stage_memories (
    stage_id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(task_id),
    agent_id UUID NOT NULL,
    stage_type VARCHAR(50) NOT NULL, -- planning, execution, reflection, verification

    -- Compressed content (10:1 ratio)
    stage_summary TEXT NOT NULL,
    stage_insights TEXT[],

    -- Raw memory references
    raw_memory_refs UUID[] NOT NULL,

    -- Metadata
    relevance_score FLOAT DEFAULT 1.0,
    compression_ratio FLOAT,
    compression_model VARCHAR(100) DEFAULT 'gpt-4o-mini',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Indexes
    INDEX idx_stage_task_agent (task_id, agent_id),
    INDEX idx_stage_type (stage_type),
    INDEX idx_stage_completed (completed_at)
);

-- Task contexts table (COMPASS MEM-1)
CREATE TABLE task_contexts (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id),
    agent_id UUID NOT NULL,
    task_goal TEXT NOT NULL,

    -- Current stage
    current_stage_id UUID REFERENCES stage_memories(stage_id),

    -- Progressive summary (5:1 compression from stages)
    task_progress_summary TEXT,

    -- Critical constraints
    critical_constraints TEXT[],

    -- Performance metrics (from ACE integration)
    performance_metrics JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Indexes
    INDEX idx_task_agent (agent_id),
    INDEX idx_task_current_stage (current_stage_id)
);

-- Error records table (COMPASS MEM-3)
CREATE TABLE error_records (
    error_id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(task_id),
    stage_id UUID REFERENCES stage_memories(stage_id),
    agent_id UUID NOT NULL,

    -- Error details
    error_type VARCHAR(50) NOT NULL, -- logic, execution, validation, constraint
    error_description TEXT NOT NULL,
    context_when_occurred TEXT,

    -- Recovery
    recovery_action TEXT,

    -- Severity
    error_severity FLOAT CHECK (error_severity BETWEEN 0 AND 1),

    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Indexes
    INDEX idx_error_task (task_id),
    INDEX idx_error_stage (stage_id),
    INDEX idx_error_type (error_type),
    INDEX idx_error_severity (error_severity DESC)
);

-- Compression metrics table (for monitoring)
CREATE TABLE compression_metrics (
    metric_id UUID PRIMARY KEY,
    stage_id UUID REFERENCES stage_memories(stage_id),

    -- Compression details
    original_tokens INT NOT NULL,
    compressed_tokens INT NOT NULL,
    compression_ratio FLOAT NOT NULL,

    -- Quality metrics
    information_retention_score FLOAT, -- 0-1, target >0.95

    -- Cost tracking
    model_used VARCHAR(100) NOT NULL,
    cost_usd FLOAT NOT NULL,

    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Indexes
    INDEX idx_compression_stage (stage_id),
    INDEX idx_compression_model (model_used),
    INDEX idx_compression_recorded (recorded_at)
);

-- Enhanced memories table (add COMPASS columns)
ALTER TABLE memories ADD COLUMN stage_id UUID REFERENCES stage_memories(stage_id);
ALTER TABLE memories ADD COLUMN is_critical BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN criticality_reason TEXT;
ALTER TABLE memories ADD COLUMN stage_relevance_map JSONB; -- {"planning": 0.9, "execution": 0.5, ...}

CREATE INDEX idx_memory_stage ON memories(stage_id);
CREATE INDEX idx_memory_critical ON memories(is_critical) WHERE is_critical = TRUE;
```

### API Design (COMPASS-Enhanced Methods)

**New JSON-RPC Methods (COMPASS MEM-1 through MEM-4):**

**7. memory.get_stage_context** (MEM-1)

```json
{
  "method": "memory.get_stage_context",
  "params": {
    "task_id": "uuid",
    "current_stage": "execution"
  },
  "a2a_context": {
    "source_agent": "agent-uuid"
  }
}
```

Response:
```json
{
  "result": {
    "stage_summary": "Current execution stage summary...",
    "stage_insights": ["insight 1", "insight 2"],
    "relevant_memories": [
      {
        "memory_id": "uuid",
        "summary": "...",
        "importance_score": 0.92
      }
    ],
    "recent_errors": [
      {
        "error_type": "validation",
        "description": "...",
        "severity": 0.7
      }
    ]
  }
}
```

**8. memory.complete_stage** (MEM-1)

```json
{
  "method": "memory.complete_stage",
  "params": {
    "task_id": "uuid",
    "stage_id": "uuid"
  },
  "a2a_context": {
    "source_agent": "agent-uuid"
  }
}
```

Response:
```json
{
  "result": {
    "stage_summary": "Compressed stage summary (10:1 ratio)...",
    "stage_insights": ["insight 1"],
    "compression_ratio": 9.8,
    "original_tokens": 5000,
    "compressed_tokens": 510,
    "information_retention": 0.96,
    "cost_usd": 0.00075
  }
}
```

**9. memory.get_task_progress** (MEM-1)

```json
{
  "method": "memory.get_task_progress",
  "params": {
    "task_id": "uuid"
  },
  "a2a_context": {
    "source_agent": "agent-uuid"
  }
}
```

Response:
```json
{
  "result": {
    "task_goal": "Original task goal",
    "task_progress_summary": "Progressive summary across all stages (5:1 ratio)...",
    "current_stage": "reflection",
    "completed_stages": ["planning", "execution"],
    "critical_constraints": ["constraint 1", "constraint 2"],
    "performance_metrics": {
      "accuracy": 0.85,
      "efficiency": 0.78
    }
  }
}
```

**10. memory.record_error** (MEM-3)

```json
{
  "method": "memory.record_error",
  "params": {
    "error": {
      "task_id": "uuid",
      "stage_id": "uuid",
      "error_type": "validation",
      "error_description": "Output validation failed for...",
      "context_when_occurred": "During execution stage, after action X",
      "error_severity": 0.7
    }
  },
  "a2a_context": {
    "source_agent": "agent-uuid"
  }
}
```

**11. memory.detect_error_patterns** (MEM-3)

```json
{
  "method": "memory.detect_error_patterns",
  "params": {
    "task_id": "uuid",
    "lookback_stages": 3
  },
  "a2a_context": {
    "source_agent": "agent-uuid"
  }
}
```

Response:
```json
{
  "result": {
    "patterns": [
      {
        "error_type": "validation",
        "occurrences": 3,
        "avg_severity": 0.65,
        "common_context": "occurs after action X",
        "recommendation": "Add pre-validation before action X"
      }
    ]
  }
}
```

**12. memory.get_error_history** (MEM-3)

```json
{
  "method": "memory.get_error_history",
  "params": {
    "task_id": "uuid",
    "lookback_stages": 3
  },
  "a2a_context": {
    "source_agent": "agent-uuid"
  }
}
```

---

## 7. Implementation Roadmap (COMPASS-Enhanced)

### Phase 1: Foundation + Hierarchical Organization (Weeks 1-3)

**Week 1: Core Infrastructure**

**Goals:**
- Database schema with PGVector + COMPASS tables
- Basic models (Pydantic + SQLAlchemy)
- Repository layer
- Development environment setup

**Tasks:**
1. Create Alembic migration for COMPASS-enhanced schema
   - Add PGVector extension
   - Create memories, stage_memories, task_contexts, error_records tables
   - Create compression_metrics table
   - Create indexes (composite, IVFFlat, stage-specific)
2. Implement Pydantic models (MemoryRecord, StageMemory, TaskContext, ErrorRecord)
3. Implement SQLAlchemy models with COMPASS columns
4. Implement base repositories (Memory, StageMem, TaskContext, Error)
5. Set up Docker Compose with PGVector
6. Write unit tests for repositories (90%+ coverage)

**Deliverables:**
- `src/agentcore/memory/models.py` (COMPASS-enhanced Pydantic models)
- `src/agentcore/memory/database/models.py` (SQLAlchemy models)
- `src/agentcore/memory/database/repositories.py` (All repositories)
- `alembic/versions/XXX_add_compass_memory_tables.py` (Migration)
- `tests/memory/unit/test_repositories.py` (Unit tests)

**Week 2: Stage Management (MEM-1)**

**Goals:**
- StageManager for hierarchical organization
- Stage lifecycle management
- Raw-to-stage memory linking
- Stage completion workflow

**Tasks:**
1. Implement StageManager class
   - `create_stage()`, `add_to_stage()`, `complete_stage()`
   - Stage type validation (planning, execution, reflection, verification)
2. Implement stage-to-memory linking logic
3. Implement stage completion workflow (prepare for compression)
4. Update MemoryManager with stage awareness
5. Write integration tests for stage management
6. Write unit tests for stage operations

**Deliverables:**
- `src/agentcore/memory/stage_manager.py` (StageManager)
- `tests/memory/integration/test_stage_manager.py` (Integration tests)
- `tests/memory/unit/test_stage_operations.py` (Unit tests)

**Week 3: Basic Retrieval + MemoryManager**

**Goals:**
- MemoryManager orchestration with stage awareness
- Basic retrieval with stage filtering
- Integration with existing EmbeddingService

**Tasks:**
1. Implement MemoryManager with stage-aware methods
   - `add_interaction()` with stage assignment
   - `get_stage_context()`
   - `complete_stage()`
2. Implement basic stage-aware retrieval
3. Integration with EmbeddingService
4. Write integration tests for MemoryManager
5. Write end-to-end workflow tests

**Deliverables:**
- `src/agentcore/memory/manager.py` (MemoryManager)
- `src/agentcore/memory/retrieval.py` (Basic retrieval)
- `tests/memory/integration/test_manager.py` (Integration tests)

### Phase 2: Test-Time Scaling Compression (Weeks 4-5)

**Week 4: Compression Infrastructure (MEM-2)**

**Goals:**
- ContextCompressor with gpt-4o-mini integration
- Stage-level compression (10:1 ratio)
- Token counting and cost tracking
- Compression quality validation

**Tasks:**
1. Implement ContextCompressor class
   - `compress_stage()` with gpt-4o-mini
   - `validate_compression_quality()`
   - `estimate_compression_cost()`
2. Integrate tiktoken for token counting
3. Implement compression quality metrics (information retention)
4. Add cost tracking to compression_metrics table
5. Write unit tests for compression logic
6. Write integration tests with gpt-4o-mini

**Deliverables:**
- `src/agentcore/memory/compression.py` (ContextCompressor)
- `src/agentcore/memory/cost_tracker.py` (Cost tracking)
- `tests/memory/unit/test_compression.py` (Unit tests)
- `tests/memory/integration/test_compression_quality.py` (Integration tests)

**Week 5: Task-Level Compression + Optimization**

**Goals:**
- Task-level progressive summarization (5:1 ratio)
- Compression quality optimization
- Caching for repeated compressions
- Performance benchmarking

**Tasks:**
1. Implement task-level compression (`compress_task()`)
2. Add compression caching in Redis (24h TTL)
3. Optimize compression prompts for quality
4. Performance benchmarking (latency, cost, quality)
5. Write integration tests for task compression
6. Load testing for compression service

**Deliverables:**
- Enhanced `src/agentcore/memory/compression.py` (Task compression)
- `tests/memory/integration/test_task_compression.py` (Integration tests)
- `tests/memory/load/test_compression_performance.py` (Load tests)
- Compression performance report

### Phase 3: Error Tracking (Week 6)

**Week 6: Error Memory + Pattern Detection (MEM-3)**

**Goals:**
- ErrorTracker implementation
- Error recording with full context
- Pattern detection algorithms
- Error-aware retrieval

**Tasks:**
1. Implement ErrorTracker class
   - `record_error()`, `detect_patterns()`, `get_error_history()`
2. Implement pattern detection algorithms
   - Group by error_type
   - Extract common context
   - Calculate pattern severity
3. Integrate error tracking with MemoryManager
4. Write unit tests for pattern detection
5. Write integration tests for error workflows

**Deliverables:**
- `src/agentcore/memory/error_tracker.py` (ErrorTracker)
- `src/agentcore/memory/pattern_detector.py` (Pattern algorithms)
- `tests/memory/unit/test_error_tracker.py` (Unit tests)
- `tests/memory/integration/test_error_workflows.py` (Integration tests)

### Phase 4: Enhanced Retrieval (Week 7)

**Week 7: Multi-Factor Scoring + Stage Awareness (MEM-4)**

**Goals:**
- Enhanced importance scoring with 5 factors
- Stage relevance scoring
- Criticality boost implementation
- Error-aware retrieval

**Tasks:**
1. Implement enhanced importance scoring
   - Embedding similarity (40%)
   - Recency (20%)
   - Frequency (15%)
   - Stage relevance (15%)
   - Criticality boost (10%)
2. Implement stage relevance calculation
3. Integrate error context into retrieval
4. Implement stage-specific context formatting
5. Write unit tests for scoring algorithms
6. Write integration tests for enhanced retrieval
7. A/B testing vs baseline retrieval

**Deliverables:**
- Enhanced `src/agentcore/memory/retrieval.py` (Multi-factor scoring)
- `tests/memory/unit/test_enhanced_scoring.py` (Unit tests)
- `tests/memory/integration/test_enhanced_retrieval.py` (Integration tests)
- Retrieval quality comparison report

### Phase 5: ACE Integration Layer (Week 8)

**Week 8: Meta-Thinker Interface (MEM-5)**

**Goals:**
- ACE integration methods
- Strategic context retrieval
- Intervention outcome tracking
- Performance metrics integration

**Tasks:**
1. Implement ACE integration methods
   - `get_strategic_context()`
   - `record_intervention_outcome()`
2. Design ACE-MEM communication protocol
3. Implement performance metrics tracking
4. Write JSON-RPC handlers for ACE methods
5. Write integration tests for ACE interface
6. Documentation for ACE developers

**Deliverables:**
- `src/agentcore/memory/ace_integration.py` (ACE interface)
- Enhanced JSON-RPC handlers
- `tests/memory/integration/test_ace_integration.py` (Integration tests)
- `docs/memory-ace-integration.md` (Integration guide)

### Phase 6: Optimization + COMPASS Validation (Weeks 9-10)

**Week 9: Production Optimization**

**Goals:**
- Performance tuning (target <100ms p95)
- PGVector index optimization
- Redis caching optimization
- Load testing at scale

**Tasks:**
1. Tune PGVector IVFFlat indexes
2. Optimize slow queries (EXPLAIN ANALYZE)
3. Implement connection pooling best practices
4. Load test with 1M memories
5. Stress test compression service
6. Memory leak detection and fixes

**Deliverables:**
- Performance tuning report
- `tests/memory/load/test_1m_scale.py` (Scale tests)
- Optimized database indexes
- Resource utilization report

**Week 10: COMPASS Validation + Documentation**

**Goals:**
- Validate against COMPASS benchmarks
- Comprehensive documentation
- Monitoring and alerting setup
- Production readiness checklist

**Tasks:**
1. COMPASS validation testing
   - Context efficiency: Target 60-80% reduction
   - Accuracy: Target +20% on long-horizon tasks
   - Cost reduction: Target 70-80% via test-time scaling
2. Set up Prometheus metrics and Grafana dashboards
3. Configure alerting rules
4. Write operational runbook
5. Write API documentation
6. Final production readiness review

**Deliverables:**
- COMPASS validation report
- `docs/memory-system-runbook.md` (Operations guide)
- `docs/memory-system-api.md` (API documentation)
- Prometheus metrics + Grafana dashboards
- Production deployment checklist

### Timeline Summary (COMPASS-Enhanced)

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| Phase 1 | Weeks 1-3 | Foundation + Hierarchical Org (MEM-1) | Database, models, repositories, StageManager |
| Phase 2 | Weeks 4-5 | Test-Time Scaling (MEM-2) | ContextCompressor with gpt-4o-mini, cost tracking |
| Phase 3 | Week 6 | Error Tracking (MEM-3) | ErrorTracker, pattern detection |
| Phase 4 | Week 7 | Enhanced Retrieval (MEM-4) | Multi-factor scoring, stage awareness |
| Phase 5 | Week 8 | ACE Integration (MEM-5) | Meta-Thinker interface, strategic context |
| Phase 6 | Weeks 9-10 | Optimization + Validation | COMPASS validation, monitoring, production readiness |

**Total Duration:** 10 weeks (70 days)

**Critical Path:**
Phase 1 (foundation + hierarchical) → Phase 2 (compression) → Phase 3 (errors) → Phase 4 (retrieval) → Phase 5 (ACE) → Phase 6 (validation)

---

## 8. Quality Assurance

### Testing Strategy (COMPASS-Enhanced)

**COMPASS-Specific Test Scenarios:**

**1. Hierarchical Memory Tests (MEM-1):**
- Stage compression maintains 10:1 ratio
- Task compression maintains 5:1 ratio
- Stage boundaries preserved (no cross-stage compression)
- Raw-to-stage-to-task traceability maintained

**2. Test-Time Scaling Tests (MEM-2):**
- gpt-4o-mini used for all compression operations
- gpt-4.1 never called for compression tasks
- Cost reduction validated (70-80% target)
- Compression quality maintained (95%+ information retention)

**3. Error Tracking Tests (MEM-3):**
- Error patterns detected correctly (2+ occurrences)
- Error context preserved for recovery
- Pattern severity calculated accurately
- Error-aware retrieval boosts recent errors

**4. Enhanced Retrieval Tests (MEM-4):**
- Multi-factor scoring produces expected rankings
- Stage relevance boosts memories from current stage
- Criticality boost applied to critical memories
- Error memories prioritized appropriately

**COMPASS Validation Tests:**

```python
@pytest.mark.asyncio
async def test_compass_context_efficiency(memory_manager):
    """
    Validate COMPASS target: 60-80% context reduction.

    Test scenario:
    - 50-turn conversation (baseline: 25K tokens)
    - With hierarchical compression: <10K tokens
    - Target: 60%+ reduction
    """
    agent_id = uuid.uuid4()
    task_id = uuid.uuid4()

    # Simulate 50-turn conversation
    interactions = generate_test_interactions(count=50, task_id=task_id)
    for interaction in interactions:
        await memory_manager.add_interaction(interaction, agent_id)

    # Measure baseline (full history)
    full_history = "\n".join([i.query + "\n" + i.response for i in interactions])
    baseline_tokens = count_tokens(full_history)

    # Measure compressed context
    compressed_context = await memory_manager.get_relevant_context(
        query=interactions[-1].query,
        task_id=task_id,
        agent_id=agent_id,
        max_tokens=10000
    )
    compressed_tokens = count_tokens(compressed_context)

    # Validate COMPASS target
    reduction_pct = (1 - compressed_tokens / baseline_tokens) * 100
    assert reduction_pct >= 60, f"Context reduction {reduction_pct}% below COMPASS target (60%)"
    assert reduction_pct <= 85, f"Context reduction {reduction_pct}% suspiciously high (>85%)"

    print(f"✅ COMPASS Context Efficiency: {reduction_pct:.1f}% reduction")

@pytest.mark.asyncio
async def test_compass_cost_reduction(memory_manager):
    """
    Validate COMPASS target: 70-80% cost reduction via test-time scaling.

    Test scenario:
    - Compress 10 stages with gpt-4o-mini vs gpt-4.1
    - Measure cost difference
    - Target: 70%+ savings
    """
    # Simulate 10 stage compressions
    stage_memories = [generate_test_stage_memories() for _ in range(10)]

    # Cost with gpt-4o-mini (actual)
    actual_cost = 0.0
    for stage in stage_memories:
        compressed = await memory_manager.compress_stage(stage, model="gpt-4o-mini")
        actual_cost += compressed.cost_usd

    # Cost with gpt-4.1 (baseline)
    baseline_cost = actual_cost * (3.00 / 0.15)  # Price ratio

    # Validate cost reduction
    cost_reduction_pct = (1 - actual_cost / baseline_cost) * 100
    assert cost_reduction_pct >= 70, f"Cost reduction {cost_reduction_pct}% below COMPASS target (70%)"

    print(f"✅ COMPASS Cost Reduction: {cost_reduction_pct:.1f}% savings (${actual_cost:.4f} vs ${baseline_cost:.4f})")

@pytest.mark.asyncio
async def test_compass_compression_quality(memory_manager):
    """
    Validate COMPASS target: 95%+ information retention in compression.

    Test scenario:
    - Compress stage with known facts
    - Validate facts preserved in summary
    - Target: 95%+ retention
    """
    # Create stage with known facts
    facts = [
        "User wants to analyze sales data",
        "Dataset has 10,000 rows",
        "Found 3 anomalies in Q2",
        "Recommended removing outliers",
        "User approved outlier removal"
    ]

    stage_memories = create_stage_with_facts(facts)
    compressed = await memory_manager.compress_stage(stage_memories, stage_type="execution")

    # Check fact retention
    retained_facts = sum(1 for fact in facts if fact.lower() in compressed.stage_summary.lower())
    retention_rate = retained_facts / len(facts)

    assert retention_rate >= 0.95, f"Information retention {retention_rate:.1%} below COMPASS target (95%)"

    print(f"✅ COMPASS Compression Quality: {retention_rate:.1%} retention ({retained_facts}/{len(facts)} facts)")
```

### Code Quality Gates (COMPASS-Enhanced)

**Additional Quality Gates:**

1. **Compression Model Enforcement:**
   - Linter check: No gpt-4.1 calls in compression.py
   - Test: All compression operations use gpt-4o-mini
   - CI fails if compression model misconfigured

2. **COMPASS Target Validation:**
   - Context efficiency: 60-80% reduction (fail if <60%)
   - Cost reduction: 70-80% savings (fail if <70%)
   - Compression quality: 95%+ retention (fail if <95%)

3. **Hierarchical Integrity:**
   - Test: Stage compression ratio 9-11x (10:1 target)
   - Test: Task compression ratio 4-6x (5:1 target)
   - Test: No cross-stage boundary compression

---

## 📚 References & Traceability

### Source Documentation (COMPASS-Enhanced)

**Research & Intelligence:**

1. **docs/research/evolving-memory-system.md**
   - Original four-layer memory architecture
   - Memory operations baseline
   - Performance targets (pre-COMPASS)

2. **.docs/research/compass-enhancement-analysis.md** ⭐ NEW
   - COMPASS paper analysis (https://arxiv.org/abs/2510.08790)
   - **Key Finding:** 20% accuracy improvement through context management
   - **Core Insight:** Hierarchical, evolving context with stage-aware compression
   - **Validation:** 60-80% context reduction, 70-80% cost reduction
   - **Enhancement Recommendations:** MEM-1 through MEM-4

**Specification:**

3. **docs/specs/memory-system/spec.md v2.0** (COMPASS-Enhanced)
   - 7 functional requirements (FR-1 through FR-7, including COMPASS enhancements)
   - Stage-aware memory organization
   - Test-time scaling configuration
   - Error tracking schema
   - Enhanced retrieval algorithms
   - 8 acceptance criteria with COMPASS validation targets

### Technology Evaluation (COMPASS-Enhanced)

**COMPASS Paper:**
- **Title:** "COMPASS: Efficient Long-Horizon Planning via Context Management"
- **Source:** https://arxiv.org/abs/2510.08790
- **Key Results:**
  - 20% accuracy improvement on GAIA benchmark
  - 60-80% context reduction through hierarchical compression
  - 70-80% cost reduction via test-time scaling
  - Three-component architecture: Main Agent, Meta-Thinker, Context Manager

**Test-Time Scaling:**
- **gpt-4o-mini Pricing:** $0.15/1M input tokens, $0.60/1M output tokens
- **gpt-4.1 Pricing:** ~$3.00/1M input tokens (20x more expensive)
- **Cost Savings:** 95% per compression operation
- **Quality:** COMPASS validates 95%+ information retention

---

**Plan Status:** ✅ Complete and Ready for Implementation (COMPASS-Enhanced)

**Next Steps:**

1. Review COMPASS-enhanced plan with engineering team
2. Validate test-time scaling cost projections
3. Create epic ticket MEM-001 with COMPASS enhancements
4. Run `/sage.tasks memory-system` to generate COMPASS-aware story tickets
5. Begin Phase 1 implementation (Foundation + Hierarchical Organization)

**Estimated Effort:** 10 weeks (1 senior engineer full-time)

**Risk Level:** LOW-MEDIUM
- Technology proven (COMPASS paper validation)
- Dependencies exist (embedding service, database)
- Clear acceptance criteria with COMPASS targets
- New complexity: hierarchical compression, error tracking (manageable)

**COMPASS Validation Commitment:**
- All implementations will be validated against COMPASS benchmarks
- Context efficiency, cost reduction, and compression quality targets mandatory
- Test suite includes COMPASS-specific validation scenarios
