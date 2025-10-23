# Evolving Memory System Specification (Enhanced with COMPASS)

**Version:** 2.0 (COMPASS-Enhanced)
**Status:** Ready for Implementation
**Priority:** P0
**Research Basis:** COMPASS Paper Analysis (`.docs/research/compass-enhancement-analysis.md`)

---

## 1. Overview

### Purpose and Business Value

The Evolving Memory System serves as the **Context Manager** (COMPASS terminology) for AgentCore, providing hierarchical, stage-aware context organization that prevents information overload and context degradation in long-horizon tasks. Unlike static context windows, this system intelligently organizes memory across reasoning stages, compresses context progressively, and maintains critical information throughout task execution.

**Business Value:**

- **80% Context Reduction**: 5-10x reduction in context tokens through stage-aware compression
- **20% Performance Improvement**: Enhanced success rate on long-horizon tasks (validated by COMPASS benchmarks)
- **70-80% Cost Reduction**: Test-time scaling using smaller models for compression
- **Unbounded Task Horizons**: Support for arbitrarily long conversations without degradation
- **Error Prevention**: Explicit error tracking prevents compounding mistakes
- **Strategic Oversight**: Provides high-level progress view for ACE monitoring

**COMPASS Alignment:**

This component implements the **Context Manager** role from COMPASS architecture:
- Maintains concise, relevant progress briefs for different reasoning stages
- Organizes context hierarchically (raw → stage → task)
- Prevents agents from overlooking critical evidence
- Enables strategic interventions by ACE (Meta-Thinker)

### Success Metrics

| Metric | Target | Measurement | COMPASS Validation |
|--------|--------|-------------|--------------------|
| Context Efficiency | 80%+ reduction | `retrieved_tokens / full_history_tokens` | ✅ 60-80% achieved |
| Long-Horizon Accuracy | +20% improvement | Success rate on GAIA-style tasks | ✅ 20% validated |
| Cost Reduction | 70-80% reduction | Monthly LLM API costs | ✅ Test-time scaling |
| Retrieval Precision | 95%+ relevant | Manual annotation or LLM eval | ⬆️ From 90% baseline |
| Memory Coherence | <5% contradictions | Contradiction detection | New metric |
| Retrieval Latency | <100ms (p95) | p95 retrieval latency | Maintained |

### Target Users

- **Agent Developers**: Building memory-enabled agents for complex workflows
- **ACE Integration**: Providing context for strategic decision-making
- **AgentCore Operators**: Managing agent systems with persistent context
- **End Users**: Benefiting from agents that remember and learn

---

## 2. Functional Requirements

### FR-1: Hierarchical Memory Organization (COMPASS MEM-1)

**Priority:** P0 (Critical for COMPASS benefits)

The system SHALL organize memory hierarchically across three levels: raw memories, stage memories, and task contexts.

**FR-1.1 Stage-Aware Memory Layers**

The system SHALL maintain stage memories with:

- Stage ID, type (`planning`, `execution`, `reflection`, `verification`)
- Stage summary (distilled from raw memories, 10:1 compression)
- Stage insights (key learnings)
- References to raw memory IDs
- Relevance score (how relevant to current work)
- Timestamps (created, updated, completed)

**FR-1.2 Task Context Management**

The system SHALL maintain task-level contexts containing:

- Task ID, goal, current stage, completed stages
- Task progress summary (compressed from stage summaries, 5:1 ratio)
- Critical constraints (must-remember facts)
- Error log (tracked errors and patterns)
- Performance metrics (from ACE integration)

**FR-1.3 Stage Detection and Transitions**

The system SHALL automatically detect reasoning stage transitions based on:

- Agent action patterns (tool usage, reasoning type)
- Explicit stage markers from agent
- Time-based heuristics (default stage duration)
- ACE intervention signals

**FR-1.4 Progressive Summarization**

The system SHALL compress memories progressively:

- When stage completes: Compress raw memories → stage summary
- Periodically: Compress completed stages → task progress summary
- On demand: Provide hierarchical context at any level

**Acceptance Criteria:**

- ✅ Stage detection accuracy ≥90% (validated on test tasks)
- ✅ Compression maintains critical information (≥95% fact retention)
- ✅ Context size reduced by 60-80% vs full history
- ✅ Hierarchical retrieval works at all levels (raw, stage, task)

**User Story:** As the Context Manager (COMPASS role), I want to organize memories by reasoning stage so that agents receive stage-appropriate context without information overload.

---

### FR-2: Evolving Context Compression (COMPASS MEM-2)

**Priority:** P0 (Critical for long-horizon tasks)

The system SHALL provide adaptive context compression using test-time scaling with smaller models.

**FR-2.1 Compression Pipeline**

The system SHALL implement compression stages:

- **Stage Compression**: Raw memories → stage summary (10:1 ratio)
- **Task Compression**: Stage summaries → task progress brief (5:1 ratio)
- **Critical Fact Extraction**: Identify must-remember information
- **Error Pattern Detection**: Track recurring errors across stages

**FR-2.2 Test-Time Scaling**

The system SHALL use different models for compression vs reasoning:

- **Compression Model**: `gpt-4o-mini` or `gpt-4.1-mini` ($0.15/1M tokens)
- **Extraction Model**: `gpt-4.1-mini` for critical fact extraction
- **Reasoning Model**: `gpt-4.1` or `gpt-5` for agent reasoning

**FR-2.3 Compression Quality Monitoring**

The system SHALL track compression quality metrics:

- Critical fact retention rate (target: ≥95%)
- Compression ratio achieved (target: 10:1 for stages, 5:1 for tasks)
- Token cost savings (target: 70-80% reduction)
- Coherence score (no contradictions introduced)

**FR-2.4 Adaptive Compression**

The system SHALL adjust compression aggressiveness based on:

- Task complexity (higher complexity → less aggressive)
- Error rate (high errors → preserve more context)
- Token budget (approaching limit → more aggressive)
- Quality metrics (degrading → fallback to less compression)

**Acceptance Criteria:**

- ✅ 70-80% context token reduction achieved
- ✅ Critical information maintained (≥95% fact retention validated)
- ✅ Compression cost ≤$100/month for 100 agents
- ✅ No degradation in task success rate from compression

**User Story:** As an agent running long-horizon tasks, I want context compressed intelligently so that I maintain essential information while reducing token costs and avoiding context limits.

---

### FR-3: Error Memory and Pattern Tracking (COMPASS MEM-3)

**Priority:** P1 (High value for learning)

The system SHALL explicitly track errors and detect patterns to enable reflection and prevent compounding mistakes.

**FR-3.1 Error Record Tracking**

The system SHALL capture error records containing:

- Error ID, task ID, stage ID
- Error type (`hallucination`, `missing_info`, `incorrect_action`, `context_degradation`)
- Error description and context when occurred
- Recovery action taken (if any)
- Error severity (0-1 scale)
- Timestamp

**FR-3.2 Error Pattern Detection**

The system SHALL detect error patterns using:

- Frequency analysis (recurring error types)
- Sequence detection (errors that follow each other)
- Context correlation (errors in similar contexts)
- LLM-based pattern extraction

**FR-3.3 Error History Queries**

The system SHALL support error history queries:

- Last N errors for a task
- Errors in specific stage types
- Errors with severity above threshold
- Error patterns for a task or agent

**FR-3.4 Error-Aware Retrieval**

The system SHALL use error history to improve retrieval:

- Boost memories that corrected previous errors
- Retrieve error-prevention knowledge when similar context detected
- Provide error context to ACE for intervention decisions

**Acceptance Criteria:**

- ✅ All errors captured with <10ms overhead
- ✅ Pattern detection identifies ≥80% of recurring patterns
- ✅ Error-aware retrieval improves precision by ≥5%
- ✅ Error tracking enables ACE reflection interventions

**User Story:** As the system, I want to track errors explicitly so that I can detect patterns, prevent recurrence, and signal ACE when intervention is needed.

---

### FR-4: Enhanced Retrieval with Criticality Scoring (COMPASS MEM-4)

**Priority:** P1 (Improves retrieval quality)

The system SHALL enhance retrieval relevance scoring to prevent overlooking critical evidence.

**FR-4.1 Multi-Factor Importance Scoring**

The system SHALL compute importance scores using:

- **Embedding similarity**: Cosine distance to query (weight: 0.35)
- **Recency**: Exponential decay from timestamp (weight: 0.15)
- **Frequency**: Logarithmic access count (weight: 0.10)
- **Stage relevance**: Relevance to current stage (weight: 0.20)
- **Criticality boost**: 2x multiplier for critical memories (weight: 0.10)
- **Error correction**: Relevance to recent errors (weight: 0.10)

**FR-4.2 Critical Memory Identification**

The system SHALL automatically mark memories as critical if they:

- Contain explicit constraints or requirements (keyword detection)
- Represent successful error recovery (linked to error records)
- Are frequently accessed in successful task completions (usage pattern)
- Contradict common mistakes (error prevention knowledge)

**FR-4.3 Stage-Aware Retrieval**

The system SHALL adjust retrieval based on reasoning stage:

- **Planning stage**: Prioritize high-level strategies, past plans
- **Execution stage**: Prioritize action memories, tool usage patterns
- **Reflection stage**: Prioritize error history, lessons learned
- **Verification stage**: Prioritize success criteria, quality checks

**FR-4.4 Retrieval Validation**

The system SHALL validate retrieval quality:

- Precision@k measurement (target: ≥95%)
- Critical evidence coverage (all must-remember facts retrieved)
- Contradiction detection (no conflicting memories returned)
- User feedback integration (manual quality ratings)

**Acceptance Criteria:**

- ✅ Retrieval precision improved from 90% to 95%+
- ✅ Critical evidence never overlooked (100% coverage)
- ✅ Contradiction rate <5%
- ✅ Stage-aware retrieval outperforms flat retrieval by ≥10%

**User Story:** As an agent, I want retrieval to prioritize critical information and stage-relevant context so that I never overlook essential evidence while making decisions.

---

### FR-5: Memory Storage and Encoding

**Priority:** P0 (Foundation)

The system SHALL encode interactions into structured memory records with vector embeddings.

**FR-5.1 Memory Record Structure**

The system SHALL create MemoryRecord objects containing:

- Unique memory_id, memory_type, timestamp
- Original content, condensed summary, vector embedding
- Extracted entities, facts, keywords
- Relationship metadata (related_memory_ids, parent_memory_id)
- Relevance tracking (relevance_score, access_count, last_accessed)
- Criticality flags (is_critical, criticality_reason)
- Stage association (stage_id, stage_type)

**FR-5.2 Vector Embeddings**

The system SHALL generate embeddings using:

- Configurable models (OpenAI, Cohere, local SentenceTransformers)
- Default: `text-embedding-3-small` (1536 dimensions)
- Batch embedding for efficiency
- Embedding consistency validation (same content → same embedding ±1%)

**FR-5.3 Entity and Fact Extraction**

The system SHALL extract structured information:

- Named entities (people, places, organizations)
- Key facts (statements that can be validated)
- Action-outcome pairs (for procedural memory)
- Constraints and requirements (for critical memory marking)

**Acceptance Criteria:**

- ✅ Memory encoding completes within 200ms (p95)
- ✅ Embeddings generated with <5% error rate
- ✅ Entity extraction accuracy ≥85%
- ✅ All memories persisted durably to PostgreSQL

**User Story:** As the system, I want to encode interactions into rich, structured memory records so that I can support advanced retrieval and analysis.

---

### FR-6: Context Formatting and Injection

**Priority:** P0 (Core capability)

The system SHALL format retrieved memories into LLM-ready context strings.

**FR-6.1 Hierarchical Context Formatting**

The system SHALL generate formatted context with sections:

- **Task Progress**: High-level task summary and current stage
- **Current Stage Context**: Stage summary and active working memory
- **Relevant Past Stages**: Compressed summaries of completed stages
- **Critical Facts**: Must-remember constraints and requirements
- **Error Context**: Recent errors and prevention strategies (if relevant)

**FR-6.2 Token Budget Management**

The system SHALL enforce token limits:

- Specified token budget (default: 2000, configurable)
- Hierarchical truncation (remove least critical content first)
- Critical fact preservation (never truncate must-remember facts)
- Compression fallback (use more aggressive compression if needed)

**FR-6.3 Context Quality Metrics**

The system SHALL track context quality:

- Token count and budget utilization
- Critical fact coverage (100% required)
- Compression ratio achieved
- Coherence score (no contradictions)

**Acceptance Criteria:**

- ✅ Context formatting completes within 50ms (p95)
- ✅ Token limits never exceeded
- ✅ Critical facts always included (100% coverage)
- ✅ Formatted context readable and coherent

**User Story:** As an agent runtime, I want formatted context that fits token limits while preserving critical information so that I can augment LLM prompts effectively.

---

### FR-7: JSON-RPC Methods

**Priority:** P0 (Integration requirement)

The system SHALL expose memory operations via JSON-RPC 2.0 protocol.

**FR-7.1 Core Memory Operations**

- `memory.store`: Store interaction as memory
- `memory.retrieve`: Retrieve relevant memories (enhanced with criticality scoring)
- `memory.get_context`: Get formatted context for query
- `memory.update`: Update existing memory
- `memory.prune`: Manually trigger pruning

**FR-7.2 Stage-Aware Operations (COMPASS Enhancement)**

- `memory.get_stage_context`: Get stage-aware context
- `memory.complete_stage`: Mark stage complete and trigger summarization
- `memory.get_task_progress`: Get task-level progress brief
- `memory.detect_stage_transition`: Check if stage should transition

**FR-7.3 Error Tracking Operations (COMPASS Enhancement)**

- `memory.record_error`: Record error with context
- `memory.detect_error_patterns`: Detect error patterns
- `memory.get_error_history`: Retrieve error history
- `memory.get_error_aware_context`: Get context with error prevention knowledge

**FR-7.4 Compression Operations (COMPASS Enhancement)**

- `memory.compress_stage`: Manually trigger stage compression
- `memory.compress_task`: Generate task progress summary
- `memory.get_compression_metrics`: Get compression quality metrics
- `memory.adjust_compression`: Adjust compression aggressiveness

**FR-7.5 ACE Integration Operations (COMPASS Enhancement)**

- `memory.get_strategic_context`: Get context for ACE decision-making
- `memory.record_intervention_outcome`: Record ACE intervention results
- `memory.get_critical_memories`: Retrieve only critical memories
- `memory.update_criticality`: Update memory criticality flags

**Acceptance Criteria:**

- ✅ All methods return valid JSON-RPC 2.0 responses
- ✅ Error cases return appropriate JsonRpcErrorCode
- ✅ A2A context (trace_id, agent_id) preserved
- ✅ All methods complete within latency targets

**User Story:** As an agent developer, I want comprehensive JSON-RPC methods so that I can integrate memory capabilities using standard protocols.

---

## 3. Non-Functional Requirements

### NFR-1: Performance

**NFR-1.1 Latency Targets**

- Memory retrieval: <100ms (p95)
- Context formatting: <50ms (p95)
- Memory encoding: <200ms (p95)
- Stage compression: <5s (p95)
- Error pattern detection: <500ms (p95)

**NFR-1.2 Throughput**

- Support 100+ concurrent retrieval requests
- Handle 1000+ memory storage operations per hour
- Process 100+ stage compressions per hour
- Support 10+ concurrent long-horizon tasks (50+ turns each)

**NFR-1.3 Scalability**

- Scale to 1M+ memory records per agent
- Maintain sub-100ms retrieval at 1M+ scale (using ANN indexes)
- Support 100+ agents concurrently
- Horizontal scaling via read replicas and sharding

### NFR-2: Cost Efficiency (COMPASS Enhancement)

**NFR-2.1 Test-Time Scaling**

- Use `gpt-4o-mini` for compression (≤$0.15/1M tokens)
- Use `gpt-4.1-mini` for extraction (≤$0.15/1M tokens)
- Reserve `gpt-4.1`/`gpt-5` for agent reasoning only
- Target: 70-80% cost reduction vs baseline

**NFR-2.2 Cost Tracking**

- Track token usage per operation type
- Monitor compression cost vs reasoning cost
- Alert when approaching monthly budget
- Provide cost breakdown per agent and task

**NFR-2.3 Cost Optimization**

- Batch compression operations
- Cache frequent queries (Redis)
- Reuse embeddings for similar content
- Adaptive compression based on budget

### NFR-3: Accuracy

**NFR-3.1 Retrieval Quality**

- Precision@5: ≥95% (improved from 90% baseline)
- Critical fact coverage: 100%
- Contradiction rate: <5%
- Stage relevance: ≥90% for stage-aware queries

**NFR-3.2 Compression Quality**

- Critical fact retention: ≥95%
- Compression ratio: 10:1 (stages), 5:1 (tasks)
- Coherence: No contradictions introduced
- Reversibility: Can reconstruct key information from compressed form

**NFR-3.3 Error Detection**

- Error capture rate: 100% (all errors recorded)
- Pattern detection accuracy: ≥80%
- False positive rate: <10%
- Error-aware retrieval improvement: ≥5% precision gain

### NFR-4: Reliability

**NFR-4.1 Data Durability**

- Memory writes persist to PostgreSQL (ACID guarantees)
- No data loss on service restart
- Working memory restored from Redis (if TTL not expired)
- Stage summaries backed up before compression

**NFR-4.2 Fault Tolerance**

- Graceful handling of embedding service failures (retry with backoff)
- Retrieval failures don't block agent operations (fallback to empty context)
- Compression failures fall back to less aggressive compression
- Corruption detection and recovery for compressed data

**NFR-4.3 Consistency**

- Stage transitions are atomic
- Compression operations are transactional
- No partial updates visible to retrieval
- Error records always linked to memories

### NFR-5: Security

**NFR-5.1 Data Isolation**

- Memories scoped by agent_id (agents cannot access others' memories)
- Memory operations require JWT authentication via A2A protocol
- Sensitive data encrypted at rest (database-level encryption)
- Audit logging for all memory operations

**NFR-5.2 Cost Control**

- Monthly token budget enforced (default: $1000/month)
- Alert at 75% budget consumption
- Hard cap prevents budget overrun
- Cost tracking per agent and task

### NFR-6: Observability

**NFR-6.1 Metrics**

- Retrieval latency (p50, p95, p99)
- Compression quality scores
- Cost metrics (tokens, dollars per operation)
- Error rates (by type)
- Cache hit rates

**NFR-6.2 Logging**

- Structured logs for all operations
- Performance traces for slow queries
- Error logs with context
- Audit trails for sensitive operations

**NFR-6.3 Monitoring**

- Alerting on latency spikes
- Alerting on quality degradation
- Alerting on cost anomalies
- Dashboard for memory system health

---

## 4. Features & Flows

### Feature 1: Core Memory Manager (P0)

**Description:** Central MemoryManager implementing encode/store/retrieve/update/prune with COMPASS enhancements.

**Components:**

- `agentcore/memory/manager.py`: MemoryManager class
- `agentcore/memory/models.py`: MemoryRecord, StageMemory, TaskContext models
- `agentcore/memory/encoding.py`: Memory encoding with entity/fact extraction
- `agentcore/memory/retrieval.py`: Enhanced retrieval with criticality scoring
- `agentcore/memory/storage.py`: PostgreSQL + Redis storage backends
- `agentcore/memory/compression.py`: Stage and task compression
- `agentcore/memory/stage_manager.py`: Stage detection and management
- `agentcore/memory/error_tracker.py`: Error recording and pattern detection

**User Flow:**

1. Agent interaction occurs (query + actions + outcome)
2. MemoryManager.add_interaction() encodes interaction
3. System generates summary, extracts entities/facts, creates embedding
4. MemoryRecord stored in appropriate layer (episodic + semantic)
5. Working memory updated if task_id present
6. Stage detection checks for stage transition
7. If stage completes: compress raw memories → stage summary
8. If errors occurred: record in error log
9. If capacity exceeded: trigger pruning

### Feature 2: Hierarchical Memory with Stage Organization (P0 - COMPASS MEM-1)

**Description:** Stage-aware memory organization with progressive summarization.

**Components:**

- `agentcore/memory/stage_manager.py`: Stage detection, transitions, summarization
- `agentcore/memory/models.py`: StageMemory, TaskContext models
- Database tables: `stage_memories`, `task_contexts`

**User Flow:**

1. Agent executes task step
2. Stage manager detects current stage (planning/execution/reflection/verification)
3. Memories tagged with stage_id and stage_type
4. When stage completes (detected via action patterns or explicit signal):
   - Retrieve all raw memories from this stage
   - Compress using ContextCompressor (10:1 ratio)
   - Store StageMemory with summary and insights
   - Update TaskContext with completed stage
5. Periodically (or on demand):
   - Compress completed stages → task progress summary (5:1 ratio)
   - Update TaskContext with high-level progress
6. Retrieval queries appropriate level (raw/stage/task) based on query type

**Input/Output:**

- **Input**: Agent interactions, stage transition signals
- **Output**: StageMemory objects, TaskContext objects, compressed context strings

### Feature 3: Evolving Context Compression (P0 - COMPASS MEM-2)

**Description:** Adaptive compression pipeline using test-time scaling with smaller models.

**Components:**

- `agentcore/memory/compression.py`: ContextCompressor class
- `agentcore/memory/quality.py`: Compression quality monitoring
- Configuration: Compression models, ratios, thresholds

**User Flow:**

1. Stage completion triggers compression
2. ContextCompressor.compress_stage():
   - Retrieves all raw memories from stage
   - Sends to compression model (gpt-4o-mini)
   - Receives stage summary (10:1 compression)
   - Extracts critical facts (must-remember information)
   - Computes quality metrics (fact retention, coherence)
3. If quality acceptable: store StageMemory
4. If quality poor: adjust compression (less aggressive) and retry
5. Task compression follows similar flow (stages → task summary, 5:1 ratio)
6. Cost tracking updates with token usage

**Input/Output:**

- **Input**: Raw memories (MemoryRecord list), compression model config
- **Output**: StageMemory with compressed summary, quality metrics, cost data

### Feature 4: Error Memory and Pattern Detection (P1 - COMPASS MEM-3)

**Description:** Explicit error tracking and pattern detection for reflection and prevention.

**Components:**

- `agentcore/memory/error_tracker.py`: Error recording and pattern detection
- `agentcore/memory/models.py`: ErrorRecord, ErrorPattern models
- Database table: `error_records`

**User Flow:**

1. Agent or ACE detects error (hallucination, missing info, incorrect action)
2. memory.record_error() creates ErrorRecord:
   - Captures error details, context, severity
   - Links to current stage and task
   - Records recovery action if taken
3. Periodically or on demand:
   - Pattern detection analyzes error history
   - Identifies recurring error types, sequences, context patterns
   - Generates ErrorPattern objects with prevention strategies
4. During retrieval:
   - If recent errors detected, boost error-correction memories
   - Include error prevention knowledge in context
   - Signal ACE if error rate exceeds threshold

**Input/Output:**

- **Input**: Error events from agent/ACE
- **Output**: ErrorRecord storage, ErrorPattern detection, error-aware context

### Feature 5: Enhanced Retrieval with Criticality (P1 - COMPASS MEM-4)

**Description:** Multi-factor importance scoring with critical memory identification.

**Components:**

- `agentcore/memory/retrieval.py`: Enhanced retrieval algorithms
- `agentcore/memory/criticality.py`: Critical memory identification
- `agentcore/memory/scoring.py`: Importance score computation

**User Flow:**

1. Retrieval request (query, k, memory_types, stage)
2. Generate query embedding
3. For each candidate memory:
   - Compute embedding similarity
   - Compute recency score
   - Compute frequency score
   - Compute stage relevance score
   - Apply criticality boost (2x if critical)
   - Compute error correction score
   - Weighted combination → importance score
4. Rank memories by importance
5. Select top-k memories
6. Validate: Check critical fact coverage, contradiction detection
7. Return memories with confidence scores

**Input/Output:**

- **Input**: Query string, retrieval parameters, current stage
- **Output**: Ranked MemoryRecord list with importance scores

### Feature 6: Database Schema and Persistence (P0)

**Description:** PostgreSQL schema with PGVector extension, supporting COMPASS enhancements.

**Schema Additions (COMPASS):**

```sql
-- Stage memories table
CREATE TABLE stage_memories (
    stage_id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(task_id),
    agent_id UUID NOT NULL,
    stage_type VARCHAR(50) NOT NULL, -- planning, execution, reflection, verification
    stage_summary TEXT NOT NULL,
    stage_insights TEXT[],
    raw_memory_refs UUID[] NOT NULL, -- References to memories table
    relevance_score FLOAT DEFAULT 1.0,
    compression_ratio FLOAT, -- Actual compression achieved
    compression_model VARCHAR(100), -- Model used for compression
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE INDEX idx_stage_memories_task ON stage_memories(task_id);
CREATE INDEX idx_stage_memories_type ON stage_memories(stage_type);
CREATE INDEX idx_stage_memories_agent ON stage_memories(agent_id);

-- Task contexts table
CREATE TABLE task_contexts (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id),
    agent_id UUID NOT NULL,
    task_goal TEXT NOT NULL,
    current_stage_id UUID REFERENCES stage_memories(stage_id),
    task_progress_summary TEXT,
    critical_constraints TEXT[],
    performance_metrics JSONB, -- From ACE integration
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_task_contexts_agent ON task_contexts(agent_id);

-- Error records table
CREATE TABLE error_records (
    error_id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(task_id),
    stage_id UUID REFERENCES stage_memories(stage_id),
    agent_id UUID NOT NULL,
    error_type VARCHAR(50) NOT NULL, -- hallucination, missing_info, incorrect_action, context_degradation
    error_description TEXT NOT NULL,
    context_when_occurred TEXT,
    recovery_action TEXT,
    error_severity FLOAT CHECK (error_severity BETWEEN 0 AND 1),
    recorded_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_error_records_task ON error_records(task_id);
CREATE INDEX idx_error_records_stage ON error_records(stage_id);
CREATE INDEX idx_error_records_type ON error_records(error_type);

-- Enhance existing memories table with COMPASS fields
ALTER TABLE memories ADD COLUMN stage_id UUID REFERENCES stage_memories(stage_id);
ALTER TABLE memories ADD COLUMN is_critical BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN criticality_reason TEXT;
ALTER TABLE memories ADD COLUMN stage_relevance_map JSONB; -- {stage_type: relevance_score}

CREATE INDEX idx_memories_stage ON memories(stage_id) WHERE stage_id IS NOT NULL;
CREATE INDEX idx_memories_critical ON memories(is_critical) WHERE is_critical = TRUE;

-- Compression metrics table
CREATE TABLE compression_metrics (
    metric_id UUID PRIMARY KEY,
    stage_id UUID REFERENCES stage_memories(stage_id),
    task_id UUID REFERENCES tasks(task_id),
    compression_type VARCHAR(50), -- stage, task
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

### Feature 7: JSON-RPC Integration (P0)

**Description:** Comprehensive JSON-RPC 2.0 methods including COMPASS enhancements.

**Method Categories:**

**Core Operations:**
- `memory.store`, `memory.retrieve`, `memory.get_context`, `memory.update`, `memory.prune`

**Stage-Aware Operations (COMPASS):**
- `memory.get_stage_context`, `memory.complete_stage`, `memory.get_task_progress`, `memory.detect_stage_transition`

**Error Tracking Operations (COMPASS):**
- `memory.record_error`, `memory.detect_error_patterns`, `memory.get_error_history`, `memory.get_error_aware_context`

**Compression Operations (COMPASS):**
- `memory.compress_stage`, `memory.compress_task`, `memory.get_compression_metrics`, `memory.adjust_compression`

**ACE Integration Operations (COMPASS):**
- `memory.get_strategic_context`, `memory.record_intervention_outcome`, `memory.get_critical_memories`, `memory.update_criticality`

**User Flow:**

1. Agent sends JSON-RPC request with COMPASS-enhanced methods
2. JsonRpcProcessor routes to memory handler
3. Handler validates request, extracts A2A context
4. MemoryManager executes operation
5. Response includes results + metrics (compression ratio, cost, quality)

### Feature 8: ACE Integration Interface (P0 - COMPASS Enhancement)

**Description:** Tight coupling between MEM (Context Manager) and ACE (Meta-Thinker).

**Components:**

- `agentcore/memory/ace_interface.py`: ACE-MEM integration
- Shared data models: TaskContext, StrategicContext, ContextQuery

**User Flow:**

1. ACE monitors agent performance, detects issue (e.g., high error rate)
2. ACE queries MEM via `memory.get_strategic_context`:
   - Sends ContextQuery with decision type, current metrics, focus areas
   - MEM analyzes metrics to determine relevant context
   - High errors → return error history + prevention knowledge
   - Slow progress → return successful patterns + strategies
   - Low coherence → return critical facts + constraints
3. MEM returns StrategicContext optimized for ACE decision
4. ACE makes intervention decision using this context
5. ACE records intervention outcome via `memory.record_intervention_outcome`
6. MEM learns from effectiveness, updates criticality and relevance scores

**Input/Output:**

- **Input**: ContextQuery from ACE with performance metrics
- **Output**: StrategicContext with relevant memories and insights

---

## 5. Acceptance Criteria

### AC-1: Context Efficiency (COMPASS Validation)

- **Given** an agent with 50-turn conversation history (25K tokens total)
- **When** memory system retrieves stage-aware context
- **Then** context size shall be ≤5K tokens (80% reduction)
- **And** context contains all critical facts (100% coverage)
- **And** compression quality ≥95% (fact retention validated)

### AC-2: Long-Horizon Task Performance (COMPASS Validation)

- **Given** a benchmark of 50 multi-turn tasks (10-30 turns each)
- **When** agents use COMPASS-enhanced memory vs baseline
- **Then** success rate improvement shall be ≥20%
- **And** average turns-to-completion reduced by ≥15%
- **And** cost per task reduced by ≥70%

### AC-3: Retrieval Precision (COMPASS Enhancement)

- **Given** a test dataset of 100 queries with labeled relevant memories
- **When** memory.retrieve is called with k=5
- **Then** precision@5 shall be ≥95% (improved from 90% baseline)
- **And** critical memories never missed (100% coverage)
- **And** contradiction rate <5%

### AC-4: Error Detection and Prevention (COMPASS)

- **Given** 100 tasks with injected errors
- **When** error tracking and pattern detection active
- **Then** all errors captured (100% capture rate)
- **And** patterns detected with ≥80% accuracy
- **And** error-aware retrieval reduces recurrence by ≥40%
- **And** ACE receives intervention signals for high error rates

### AC-5: Compression Quality (COMPASS)

- **Given** 50 completed stages with known critical facts
- **When** stage compression applied
- **Then** compression ratio ≥10:1 achieved
- **And** critical fact retention ≥95%
- **And** no contradictions introduced (coherence score 1.0)
- **And** cost ≤$0.15/1M tokens (using mini model)

### AC-6: ACE Integration (COMPASS)

- **Given** ACE monitoring detects context degradation
- **When** ACE queries MEM for strategic context
- **Then** MEM returns relevant context within 100ms
- **And** context optimized for intervention decision
- **And** intervention outcome recorded and learned from
- **And** feedback loop improves future context provision

### AC-7: Retrieval Latency (Maintained)

- **Given** memory database with 100K records per agent
- **When** executing memory.retrieve with k=5
- **Then** p50 latency shall be <50ms
- **And** p95 latency shall be <100ms
- **And** p99 latency shall be <200ms

### AC-8: Storage Scalability (Maintained)

- **Given** 10 agents with 100K memories each (1M total)
- **When** system is under normal load (10 queries/sec)
- **Then** retrieval latency remains <100ms p95
- **And** storage usage is ≤50GB (with compression)
- **And** no degradation in retrieval accuracy

---

## 6. Dependencies

### Technical Dependencies

**Required Infrastructure:**

- PostgreSQL 14+ with PGVector extension
- Redis 6+ (working memory cache)
- Python 3.12+ (built-in generics)

**External Services:**

- **Compression Models**: OpenAI API (gpt-4o-mini, gpt-4.1-mini) or Anthropic (claude-3-haiku)
- **Reasoning Models**: OpenAI (gpt-4.1, gpt-5) or Anthropic (claude-3.5-sonnet)
- **Embedding Models**: OpenAI (text-embedding-3-small) or Cohere or local
- **Optional**: S3-compatible storage for memory archival

**Python Packages:**

- `pgvector`: PostgreSQL vector extension
- `openai` or `anthropic`: LLM APIs
- `sentence-transformers`: Local embeddings (optional)
- `redis`: Redis client
- `boto3`: S3 client (optional)

### AgentCore Component Dependencies

**Required Components:**

- Database Layer (`agentcore.database`): Sessions, repositories
- JSON-RPC Handler (`agentcore.a2a_protocol.services.jsonrpc_handler`): Method registration
- Task Management (`agentcore.a2a_protocol.services.task_manager`): task_id references
- Configuration (`agentcore.a2a_protocol.config`): Settings management

**Integration Components (COMPASS):**

- ACE Integration (`agentcore.ace`): For meta-cognitive monitoring and interventions
- Agent Manager (`agentcore.a2a_protocol.services.agent_manager`): Agent-scoped memories

### Configuration

```toml
[memory]
# Embedding configuration
embedding_model = "openai"
embedding_model_name = "text-embedding-3-small"
embedding_dimensions = 1536

# Vector store
vector_store = "pgvector"

# Memory layers (COMPASS)
working_memory_ttl = 3600  # 1 hour
episodic_memory_capacity = 50
semantic_memory_capacity = 1000

# Compression (COMPASS)
compression_model = "gpt-4o-mini"
extraction_model = "gpt-4.1-mini"
stage_compression_ratio_target = 10.0
task_compression_ratio_target = 5.0
compression_quality_threshold = 0.95  # Critical fact retention

# Retrieval (COMPASS)
retrieval_precision_target = 0.95
criticality_boost_multiplier = 2.0
stage_relevance_weight = 0.20
error_correction_weight = 0.10

# Cost control (COMPASS)
monthly_token_budget_usd = 1000
alert_threshold_pct = 75
test_time_scaling_enabled = true

# Error tracking (COMPASS)
error_pattern_detection_enabled = true
error_history_lookback_stages = 3
error_severity_threshold = 0.7  # For ACE signaling

# Stage management (COMPASS)
stage_detection_enabled = true
stage_auto_transition = true
stage_completion_auto_compress = true
```

---

## Implementation Phases

### Phase 1: Foundation + Stage Organization (Weeks 1-3) - MEM-1

- Implement StageMemory, TaskContext data models
- Add stage detection and transition logic
- Implement progressive summarization (raw → stage → task)
- Database schema with stage_memories, task_contexts tables
- Basic compression without test-time scaling
- Unit tests for stage management
- Integration tests for hierarchical retrieval

### Phase 2: Test-Time Scaling Compression (Weeks 4-5) - MEM-2

- Implement ContextCompressor with model selection
- Add compression quality monitoring
- Implement adaptive compression
- Cost tracking and budgeting
- Performance optimization for compression
- Load testing with multiple concurrent compressions

### Phase 3: Error Tracking and Patterns (Week 6) - MEM-3

- Implement ErrorRecord tracking
- Add error pattern detection algorithms
- Implement error-aware retrieval
- Database schema for error_records
- Integration with ACE for intervention signaling
- Tests for error detection and pattern accuracy

### Phase 4: Enhanced Retrieval (Week 7) - MEM-4

- Implement multi-factor importance scoring
- Add critical memory identification
- Implement stage-aware retrieval
- Optimize importance score computation
- A/B testing for retrieval quality
- Benchmarking against baseline

### Phase 5: ACE Integration (Week 8)

- Implement ACE-MEM interface
- Add strategic context query methods
- Implement intervention outcome recording
- Feedback loop for learning
- Integration tests with ACE component
- End-to-end tests for COMPASS workflow

### Phase 6: Optimization and Validation (Weeks 9-10)

- Performance optimization (database queries, caching)
- Cost optimization (batch operations, caching)
- Comprehensive benchmarking
- A/B testing enhanced vs baseline
- Documentation and runbook

---

**Total Implementation Timeline:** 10 weeks
**Success Target:** 20% accuracy improvement, 70-80% cost reduction (COMPASS validation)
**Research Basis:** `.docs/research/compass-enhancement-analysis.md`

---

**Generated from:** COMPASS paper analysis + existing memory-system specification
**Next Steps:** Run `/sage.plan memory-system` to generate implementation plan
