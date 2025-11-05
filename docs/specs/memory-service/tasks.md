# Tasks: Memory System (Hybrid: Mem0 + COMPASS + Graph)

**⚠️ NEEDS REGENERATION:** This task breakdown is based on the old PGVector-only architecture. It must be regenerated to reflect the hybrid architecture (Qdrant + Neo4j + ECL Pipeline + Memify) using `/sage.tasks memory-service`.

**From:** `spec.md` v3.0 (Hybrid Architecture) + `plan.md` v3.0
**Timeline:** 8 weeks (updated from 10 weeks)
**Team:** 1 senior backend engineer (full-time)
**Created:** 2025-10-23
**Last Updated:** 2025-11-06 (Marked for regeneration)

---

## Summary

- **Total tasks:** 35 story tasks (MEM-002 through MEM-036)
- **Estimated effort:** 165 story points (~10 weeks)
- **Critical path duration:** 10 weeks (sequential phases)
- **Key risks:**
  1. PGVector performance at scale (mitigation: early load testing)
  2. Compression quality validation (mitigation: COMPASS benchmarks)
  3. ACE integration timing (mitigation: mock interface first)

**COMPASS Targets:**

- 60-80% context efficiency
- 20% performance improvement on long-horizon tasks
- 70-80% cost reduction via test-time scaling
- 95%+ compression information retention

---

## Phase Breakdown

### Phase 1: Foundation + Hierarchical Organization (Sprint 1-2, 58 SP)

**Goal:** Establish COMPASS-enhanced database schema, models, and hierarchical stage management
**Deliverable:** Working stage-aware memory storage with 3-level hierarchy

#### Week 1: Core Infrastructure (26 SP)

**MEM-002: Create COMPASS-Enhanced Database Migration**

- **Description:** Implement Alembic migration for stage_memories, task_contexts, error_records, compression_metrics tables with PGVector extension
- **Acceptance:**
  - [ ] PGVector extension added to PostgreSQL
  - [ ] All 5 COMPASS tables created (memories, stage_memories, task_contexts, error_records, compression_metrics)
  - [ ] Composite indexes created for stage + agent + task queries
  - [ ] IVFFlat vector indexes configured for embedding similarity
  - [ ] Migration reversible (downgrade tested)
  - [ ] Seed data loads successfully
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-001 (epic)
- **Priority:** P0 (Blocker for all phases)
- **Files:**
  - `alembic/versions/XXX_add_compass_memory_tables.py`
  - `docker-compose.dev.yml` (add PGVector to PostgreSQL)

**MEM-003: Implement COMPASS Pydantic Models**

- **Description:** Create Pydantic models for MemoryRecord, StageMemory, TaskContext, ErrorRecord, CompressionConfig with COMPASS enhancements
- **Acceptance:**
  - [ ] StageMemory model with stage_type enum validation
  - [ ] TaskContext model with performance_metrics dict
  - [ ] ErrorRecord model with severity scoring (0-1)
  - [ ] CompressionConfig model with test-time scaling parameters
  - [ ] All models support JSON serialization
  - [ ] Modern typing (use `list[]`, `dict[]`, `|` unions)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-002
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/models.py`

**MEM-004: Implement SQLAlchemy ORM Models**

- **Description:** Create SQLAlchemy models matching database schema with async support and COMPASS columns
- **Acceptance:**
  - [ ] StageMemoryModel with JSONB for raw_memory_refs
  - [ ] TaskContextModel with foreign key to stage_memories
  - [ ] ErrorModel with error_type constraint validation
  - [ ] CompressionMetricsModel for cost tracking
  - [ ] All models use AsyncSession
  - [ ] Relationships configured (task → stages → memories)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-003
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/database/models.py`

**MEM-005: Implement Repository Layer**

- **Description:** Create async repositories for Memory, StageMemory, TaskContext, Error with COMPASS query patterns
- **Acceptance:**
  - [ ] MemoryRepository with stage_id filtering
  - [ ] StageMemoryRepository with get_by_task_and_stage()
  - [ ] TaskContextRepository with current_stage tracking
  - [ ] ErrorRepository with pattern detection queries
  - [ ] All methods use async/await
  - [ ] Unit tests for each repository (90%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-004
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/database/repositories.py`
  - `tests/memory/unit/test_repositories.py`

#### Week 2: Stage Management (MEM-1) (18 SP)

**MEM-006: Implement StageManager Core**

- **Description:** Create StageManager class for stage lifecycle (create, add, complete) with stage type validation
- **Acceptance:**
  - [ ] create_stage(task_id, stage_type) validates stage enum
  - [ ] add_to_stage(stage_id, memory_id) links memory to stage
  - [ ] complete_stage(stage_id) triggers compression preparation
  - [ ] Stage transitions tracked in TaskContext.current_stage_id
  - [ ] Stage type enum: planning, execution, reflection, verification
  - [ ] Unit tests for stage lifecycle (95%+ coverage)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-005
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/stage_manager.py`
  - `tests/memory/unit/test_stage_operations.py`

**MEM-007: Implement Stage Detection Logic**

- **Description:** Add automatic stage transition detection based on agent actions and time heuristics
- **Acceptance:**
  - [ ] Action pattern analysis (tool usage classification)
  - [ ] Time-based default stage duration (configurable)
  - [ ] Explicit stage markers from agent honored
  - [ ] Stage transitions logged with rationale
  - [ ] Integration with agent action stream
  - [ ] Integration tests for stage detection
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-006
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/stage_detector.py`
  - `tests/memory/integration/test_stage_detection.py`

**MEM-008: Integrate StageManager with MemoryManager**

- **Description:** Update MemoryManager to be stage-aware and orchestrate stage operations
- **Acceptance:**
  - [ ] add_interaction() assigns memories to current stage
  - [ ] get_stage_context() retrieves stage-filtered memories
  - [ ] complete_stage() delegates to StageManager
  - [ ] Stage transitions trigger appropriate hooks
  - [ ] Backwards compatible with non-stage-aware agents
  - [ ] Integration tests for stage-aware workflows
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-007
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/manager.py`
  - `tests/memory/integration/test_stage_manager.py`

#### Week 3: Basic Retrieval + EmbeddingService Integration (14 SP)

**MEM-009: Implement Basic Stage-Aware Retrieval**

- **Description:** Create retrieval service with stage filtering and embedding similarity
- **Acceptance:**
  - [ ] retrieve_stage_aware(query, current_stage, k) returns top-k memories
  - [ ] Stage filtering applied before similarity search
  - [ ] PGVector cosine similarity used for ranking
  - [ ] Query embeddings generated via EmbeddingService
  - [ ] Retrieval latency <100ms (p95) for 10K memories
  - [ ] Unit tests for retrieval logic
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-008
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/retrieval.py`
  - `tests/memory/unit/test_basic_retrieval.py`

**MEM-010: Integrate with Existing EmbeddingService**

- **Description:** Connect memory storage to existing embedding_service.py for vector generation
- **Acceptance:**
  - [ ] Memory creation triggers embedding generation
  - [ ] Embeddings stored in memories.embedding column
  - [ ] Batch embedding generation for multiple memories
  - [ ] Embedding failures handled gracefully (retry logic)
  - [ ] Embedding cost tracked per agent
  - [ ] Integration tests with mocked OpenAI API
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-009
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/embedding_integration.py`
  - `tests/memory/integration/test_embedding_service.py`

**MEM-011: Implement JSON-RPC Handlers for Phase 1**

- **Description:** Register JSON-RPC methods for memory.add, memory.retrieve, memory.get_stage_context
- **Acceptance:**
  - [ ] memory.add(interaction) creates memory with embedding
  - [ ] memory.retrieve(query, k) returns relevant memories
  - [ ] memory.get_stage_context(task_id, stage) returns stage summary
  - [ ] memory.complete_stage(task_id, stage_id) triggers completion
  - [ ] A2A context (agent_id, task_id) handled correctly
  - [ ] JSON-RPC error handling for invalid params
- **Effort:** 4 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-010
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/jsonrpc_handlers.py`
  - `tests/memory/integration/test_jsonrpc.py`

---

### Phase 2: Test-Time Scaling Compression (Sprint 2-3, 32 SP)

**Goal:** Implement COMPASS-validated compression with gpt-4o-mini (70-80% cost reduction)
**Deliverable:** Stage and task compression with quality validation

#### Week 4: Compression Infrastructure (MEM-2) (18 SP)

**MEM-012: Implement ContextCompressor Core**

- **Description:** Create ContextCompressor with gpt-4o-mini integration for stage compression (10:1 target)
- **Acceptance:**
  - [ ] compress_stage(stage_memories, stage_type) returns StageMemory
  - [ ] Uses gpt-4o-mini (NOT gpt-4.1) for compression
  - [ ] Target 10:1 compression ratio (9-11x acceptable)
  - [ ] Compression prompt preserves critical information
  - [ ] Stage insights extracted (key learnings)
  - [ ] Unit tests for compression logic
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-011
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/compression.py`
  - `tests/memory/unit/test_compression.py`

**MEM-013: Integrate tiktoken for Token Counting**

- **Description:** Add token counting for compression ratio validation and cost tracking
- **Acceptance:**
  - [ ] count_tokens(text) uses tiktoken
  - [ ] Token counts recorded in CompressionMetrics
  - [ ] Compression ratio computed: original_tokens / compressed_tokens
  - [ ] Target ratio validation (warn if <8x or >12x)
  - [ ] Cost estimation per compression operation
  - [ ] Unit tests for token counting
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-012
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/token_counter.py`
  - `tests/memory/unit/test_token_counter.py`

**MEM-014: Implement Compression Quality Validation**

- **Description:** Add information retention scoring to validate 95%+ quality target
- **Acceptance:**
  - [ ] validate_compression_quality(original, compressed) returns score
  - [ ] Fact extraction and preservation checking
  - [ ] Score >0.95 required (COMPASS target)
  - [ ] Quality metrics logged to compression_metrics table
  - [ ] Alert on quality degradation (<0.90)
  - [ ] Unit tests for validation algorithms
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-013
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/quality_validator.py`
  - `tests/memory/unit/test_quality_validation.py`

**MEM-015: Implement Cost Tracking**

- **Description:** Track compression costs per agent and enforce budget limits
- **Acceptance:**
  - [ ] Cost computed: tokens * model_price_per_1M
  - [ ] Cost recorded in compression_metrics.cost_usd
  - [ ] Monthly budget tracking per agent
  - [ ] Alert at 75% budget threshold
  - [ ] Hard cap prevents budget overrun
  - [ ] Cost reports exportable for analysis
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-014
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/cost_tracker.py`
  - `tests/memory/unit/test_cost_tracker.py`

#### Week 5: Task-Level Compression + Optimization (14 SP)

**MEM-016: Implement Task-Level Compression**

- **Description:** Add compress_task() for progressive task summarization (5:1 target from stage summaries)
- **Acceptance:**
  - [ ] compress_task(stage_summaries, task_goal) returns TaskContext
  - [ ] Uses gpt-4o-mini for task compression
  - [ ] Target 5:1 compression ratio (4-6x acceptable)
  - [ ] Task progress summary includes all completed stages
  - [ ] Critical constraints preserved
  - [ ] Integration tests for task compression
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-015
- **Priority:** P0
- **Files:**
  - Enhanced `src/agentcore/memory/compression.py`
  - `tests/memory/integration/test_task_compression.py`

**MEM-017: Add Compression Caching in Redis**

- **Description:** Cache compressed summaries in Redis (24h TTL) to reduce repeated compressions
- **Acceptance:**
  - [ ] Stage summaries cached by stage_id
  - [ ] Task summaries cached by task_id
  - [ ] Cache TTL configurable (default 24h)
  - [ ] Cache invalidation on stage/task updates
  - [ ] Cache hit rate tracked (target >60%)
  - [ ] Integration tests with Redis
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-016
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/compression_cache.py`
  - `tests/memory/integration/test_compression_cache.py`

**MEM-018: Optimize Compression Prompts**

- **Description:** Refine prompts for gpt-4o-mini to improve compression quality and consistency
- **Acceptance:**
  - [ ] Stage-specific prompts (planning vs execution vs reflection)
  - [ ] Prompt includes compression ratio target
  - [ ] Prompt emphasizes critical information preservation
  - [ ] JSON output format for structured data
  - [ ] A/B testing shows >95% quality retention
  - [ ] Prompt templates versioned
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-017
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/prompts.py`
  - `tests/memory/integration/test_prompt_quality.py`

**MEM-019: Performance Benchmarking**

- **Description:** Benchmark compression latency, cost, and quality across 1000 compressions
- **Acceptance:**
  - [ ] Latency: <5s (p95) for stage compression
  - [ ] Cost: <$0.01 per stage compression (gpt-4o-mini pricing)
  - [ ] Quality: >95% information retention (COMPASS target)
  - [ ] Compression ratio: 9-11x for stages, 4-6x for tasks
  - [ ] Load tests with 100 concurrent compressions
  - [ ] Performance report generated
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-018
- **Priority:** P1
- **Files:**
  - `tests/memory/load/test_compression_performance.py`
  - `docs/memory-compression-performance-report.md`

---

### Phase 3: Error Tracking (Sprint 3, 14 SP)

**Goal:** Implement error memory and pattern detection (MEM-3)
**Deliverable:** Error tracking preventing compounding mistakes

#### Week 6: Error Memory + Pattern Detection (14 SP)

**MEM-020: Implement ErrorTracker Core**

- **Description:** Create ErrorTracker for recording errors with full context and severity
- **Acceptance:**
  - [ ] record_error(error_record) stores error with context
  - [ ] Error severity computed (0-1 scale)
  - [ ] Error linked to task_id and stage_id
  - [ ] Recovery actions recorded
  - [ ] get_error_history(task_id) retrieves recent errors
  - [ ] Unit tests for error recording
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-019
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/error_tracker.py`
  - `tests/memory/unit/test_error_tracker.py`

**MEM-021: Implement Error Pattern Detection**

- **Description:** Add pattern detection to identify recurring error types (2+ occurrences)
- **Acceptance:**
  - [ ] detect_patterns(task_id, lookback_stages) groups errors
  - [ ] Common context extraction from error descriptions
  - [ ] Pattern severity: avg(error_severity) * occurrences
  - [ ] Recommendations generated for pattern mitigation
  - [ ] Compounding error detection (related errors in sequence)
  - [ ] Unit tests for pattern detection algorithms
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-020
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/pattern_detector.py`
  - `tests/memory/unit/test_pattern_detector.py`

**MEM-022: Integrate Error Tracking with MemoryManager**

- **Description:** Connect error tracking to memory retrieval for error-aware context
- **Acceptance:**
  - [ ] Errors trigger memory tagging (is_critical flag)
  - [ ] Error-related memories boosted in retrieval (30% boost)
  - [ ] Error patterns surfaced in get_stage_context()
  - [ ] Error recovery context provided to agents
  - [ ] Integration tests for error workflows
- **Effort:** 4 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-021
- **Priority:** P0
- **Files:**
  - Enhanced `src/agentcore/memory/manager.py`
  - `tests/memory/integration/test_error_workflows.py`

---

### Phase 4: Enhanced Retrieval (Sprint 4, 24 SP)

**Goal:** Implement multi-factor importance scoring (MEM-4)
**Deliverable:** Stage-aware retrieval with criticality boosting

#### Week 7: Multi-Factor Scoring + Stage Awareness (24 SP)

**MEM-023: Implement Enhanced Importance Scoring**

- **Description:** Add 5-factor importance scoring (embedding, recency, frequency, stage, criticality)
- **Acceptance:**
  - [ ] Embedding similarity (40% weight)
  - [ ] Recency (20% weight, exponential decay 24h half-life)
  - [ ] Frequency (15% weight, access count normalized)
  - [ ] Stage relevance (15% weight, boost current stage)
  - [ ] Criticality boost (10% weight, is_critical flag)
  - [ ] Unit tests for each factor
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-022
- **Priority:** P0
- **Files:**
  - Enhanced `src/agentcore/memory/retrieval.py`
  - `tests/memory/unit/test_enhanced_scoring.py`

**MEM-024: Implement Stage Relevance Calculation**

- **Description:** Add stage_relevance_map to memories for cross-stage importance
- **Acceptance:**
  - [ ] Stage relevance computed per memory
  - [ ] Memories tagged with relevant stages (JSONB map)
  - [ ] Current stage memories boosted (1.5x multiplier)
  - [ ] Cross-stage references preserved
  - [ ] Stage transitions update relevance scores
  - [ ] Unit tests for relevance calculation
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-023
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/stage_relevance.py`
  - `tests/memory/unit/test_stage_relevance.py`

**MEM-025: Implement Criticality Boosting**

- **Description:** Add is_critical flag and criticality_reason for important memories
- **Acceptance:**
  - [ ] Critical memories flagged during storage
  - [ ] Criticality boost: 1.5x importance score
  - [ ] Error-related memories auto-flagged as critical
  - [ ] Agent-specified critical constraints preserved
  - [ ] Critical memory ratio tracked (target <10%)
  - [ ] Integration tests for criticality
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-024
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/criticality.py`
  - `tests/memory/integration/test_criticality.py`

**MEM-026: Implement Stage-Specific Context Formatting**

- **Description:** Format retrieved context based on current reasoning stage
- **Acceptance:**
  - [ ] Planning stage: goals, constraints, strategies
  - [ ] Execution stage: actions, tools, immediate context
  - [ ] Reflection stage: outcomes, learnings, insights
  - [ ] Verification stage: results, validations, comparisons
  - [ ] Format includes stage headers and structure
  - [ ] Unit tests for formatting logic
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-025
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/context_formatter.py`
  - `tests/memory/unit/test_context_formatting.py`

**MEM-027: A/B Testing vs Baseline Retrieval**

- **Description:** Compare enhanced retrieval (5-factor) against baseline (embedding-only)
- **Acceptance:**
  - [ ] Test dataset: 1000 queries with ground truth
  - [ ] Metrics: precision@5, recall@10, NDCG@10
  - [ ] Enhanced retrieval shows >10% improvement
  - [ ] Latency overhead <20ms per query
  - [ ] A/B test report with statistical significance
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-026
- **Priority:** P1
- **Files:**
  - `tests/memory/integration/test_enhanced_retrieval.py`
  - `docs/memory-retrieval-comparison-report.md`

---

### Phase 5: ACE Integration Layer (Sprint 4, 16 SP)

**Goal:** Implement Meta-Thinker interface for ACE coordination
**Deliverable:** Strategic context queries and intervention tracking

#### Week 8: Meta-Thinker Interface (MEM-5) (16 SP)

**MEM-028: Implement ACE Integration Methods**

- **Description:** Create ACEMemoryInterface for strategic context queries
- **Acceptance:**
  - [ ] get_strategic_context(query) returns StrategicContext
  - [ ] Query types: strategic_decision, error_analysis, capability_evaluation, context_refresh
  - [ ] Strategic context includes: stage summaries, critical facts, error patterns, successful patterns
  - [ ] Context health score computed (0-1)
  - [ ] Query latency <150ms (p95)
  - [ ] Unit tests for ACE methods
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-027
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/ace_integration.py`
  - `tests/memory/unit/test_ace_integration.py`

**MEM-029: Implement Intervention Outcome Tracking**

- **Description:** Track ACE intervention effectiveness in memory system
- **Acceptance:**
  - [ ] record_intervention_outcome(intervention_id, success, delta)
  - [ ] Intervention outcomes linked to memories
  - [ ] Pre/post metrics tracked for effectiveness
  - [ ] Learning updates stage relevance weights
  - [ ] Intervention history queryable
  - [ ] Integration tests with mocked ACE
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-028
- **Priority:** P1
- **Files:**
  - Enhanced `src/agentcore/memory/ace_integration.py`
  - `tests/memory/integration/test_ace_interventions.py`

**MEM-030: Add JSON-RPC Handlers for ACE Methods**

- **Description:** Register JSON-RPC methods for ACE-MEM coordination
- **Acceptance:**
  - [ ] memory.get_strategic_context(query) exposed
  - [ ] memory.record_intervention_outcome(intervention) exposed
  - [ ] memory.get_context_health(task_id) exposed
  - [ ] memory.request_context_refresh(task_id) exposed
  - [ ] A2A context handled correctly
  - [ ] Integration tests for JSON-RPC ACE methods
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-029
- **Priority:** P0
- **Files:**
  - Enhanced `src/agentcore/memory/jsonrpc_handlers.py`
  - `tests/memory/integration/test_ace_jsonrpc.py`

---

### Phase 6: Optimization + COMPASS Validation (Sprint 5, 21 SP)

**Goal:** Production-ready system with COMPASS validation
**Deliverable:** Validated, monitored, documented memory system

#### Week 9: Production Optimization (12 SP)

**MEM-031: Tune PGVector Indexes**

- **Description:** Optimize IVFFlat indexes for retrieval performance at 1M scale
- **Acceptance:**
  - [ ] IVFFlat list count tuned (target sqrt(N) lists)
  - [ ] Index build time <10 minutes for 1M vectors
  - [ ] Query latency <100ms (p95) at 1M scale
  - [ ] Index size <30% of vector data size
  - [ ] EXPLAIN ANALYZE validates index usage
  - [ ] Performance report with benchmarks
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-030
- **Priority:** P1
- **Files:**
  - Database index tuning scripts
  - `tests/memory/load/test_1m_scale.py`

**MEM-032: Optimize Database Queries**

- **Description:** Identify and optimize slow queries using EXPLAIN ANALYZE
- **Acceptance:**
  - [ ] All critical path queries <50ms
  - [ ] No table scans on large tables
  - [ ] Composite indexes for common filters
  - [ ] Connection pooling configured (min 10, max 50)
  - [ ] Query plan cache enabled
  - [ ] Slow query log analyzed
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-031
- **Priority:** P1
- **Files:**
  - Database configuration updates
  - `docs/memory-query-optimization.md`

**MEM-033: Stress Test Compression Service**

- **Description:** Load test compression with 100 concurrent requests
- **Acceptance:**
  - [ ] 100 concurrent compressions without errors
  - [ ] Queue processing time <10s per batch
  - [ ] Memory usage <500MB under load
  - [ ] CPU usage <80% under load
  - [ ] Rate limiting prevents gpt-4o-mini quota exhaustion
  - [ ] Load test report generated
- **Effort:** 4 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-032
- **Priority:** P1
- **Files:**
  - `tests/memory/load/test_compression_stress.py`
  - `docs/memory-load-test-report.md`

#### Week 10: COMPASS Validation + Documentation (9 SP)

**MEM-034: COMPASS Validation Testing**

- **Description:** Validate against COMPASS benchmarks (context efficiency, accuracy, cost)
- **Acceptance:**
  - [ ] Context efficiency: 60-80% reduction achieved
  - [ ] Long-horizon accuracy: +20% improvement on GAIA-style tasks
  - [ ] Cost reduction: 70-80% via test-time scaling validated
  - [ ] Compression quality: 95%+ information retention
  - [ ] All COMPASS targets met or exceeded
  - [ ] Validation report with statistical analysis
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-033
- **Priority:** P0
- **Files:**
  - `tests/memory/validation/test_compass_benchmarks.py`
  - `docs/memory-compass-validation-report.md`

**MEM-035: Set Up Monitoring and Alerting**

- **Description:** Configure Prometheus metrics and Grafana dashboards
- **Acceptance:**
  - [ ] Metrics: retrieval latency, compression cost, memory count, error rate
  - [ ] Grafana dashboard for memory system health
  - [ ] Alerts: high latency, budget exhaustion, quality degradation
  - [ ] Metric retention: 30 days
  - [ ] Alert routing to on-call engineer
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer + DevOps
- **Dependencies:** MEM-034
- **Priority:** P1
- **Files:**
  - `prometheus/memory_metrics.yml`
  - `grafana/memory_dashboard.json`

**MEM-036: Write Operational Documentation**

- **Description:** Create runbook and API documentation for operations and developers
- **Acceptance:**
  - [ ] Runbook: deployment, troubleshooting, common issues
  - [ ] API docs: JSON-RPC methods, examples, error codes
  - [ ] Architecture diagram: component interactions
  - [ ] Configuration guide: all settings explained
  - [ ] COMPASS validation results included
  - [ ] Production readiness checklist completed
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-035
- **Priority:** P0
- **Files:**
  - `docs/memory-system-runbook.md`
  - `docs/memory-system-api.md`
  - `docs/memory-architecture.md`

---

## Critical Path

```plaintext
Foundation (Week 1-3):
MEM-002 → MEM-003 → MEM-004 → MEM-005 → MEM-006 → MEM-007 → MEM-008 → MEM-009 → MEM-010 → MEM-011
  (3d)     (2d)       (2d)       (3d)       (2d)       (2d)       (3d)       (2d)       (2d)       (1d)
                                        [22 days]

Compression (Week 4-5):
MEM-012 → MEM-013 → MEM-014 → MEM-015 → MEM-016 → MEM-017 → MEM-018 → MEM-019
  (3d)      (1d)      (2d)       (1d)      (2d)       (1d)       (1d)       (1d)
                                        [12 days]

Error + Retrieval (Week 6-7):
MEM-020 → MEM-021 → MEM-022 → MEM-023 → MEM-024 → MEM-025 → MEM-026 → MEM-027
  (2d)      (2d)      (1d)       (3d)       (2d)       (1d)       (1d)       (2d)
                                        [14 days]

ACE + Production (Week 8-10):
MEM-028 → MEM-029 → MEM-030 → MEM-031 → MEM-032 → MEM-033 → MEM-034 → MEM-035 → MEM-036
  (3d)      (2d)      (1d)       (2d)       (1d)       (1d)       (2d)       (1d)       (1d)
                                        [14 days]

Total Critical Path: 62 days (~10 weeks)
```

**Bottlenecks:**

- **MEM-008 (StageManager Integration)**: Complex orchestration (8 SP, highest risk)
- **MEM-023 (Enhanced Scoring)**: Multiple algorithms (8 SP, complexity risk)
- **MEM-034 (COMPASS Validation)**: External benchmark dependency (5 SP)

**Parallel Tracks:**

- **Testing**: Unit tests can be written concurrently with implementation
- **Documentation**: API docs can be written as features complete
- **Monitoring**: Metrics setup can happen alongside Week 9 optimization

---

## Quick Wins (Week 1-2)

1. **MEM-002 (Database Migration)** - Unblocks all development
2. **MEM-003/004 (Models)** - Enables parallel repository work
3. **MEM-005 (Repositories)** - Core data access ready early

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| MEM-008 | Stage detection complexity | Early integration testing, mock data | Simplify to manual stage markers initially |
| MEM-012 | gpt-4o-mini quality issues | COMPASS benchmarks validate approach | Fall back to gpt-4.1 with budget increase |
| MEM-023 | Enhanced scoring performance | Profile and optimize early | Use simpler 3-factor scoring (embed, recency, stage) |
| MEM-031 | PGVector scaling issues | Load test at 100K before 1M | Consider external vector DB (Pinecone, Weaviate) |
| MEM-034 | COMPASS validation fails | Iterative tuning with benchmark feedback | Document deviations, target 80% of goals |

---

## Testing Strategy

### Automated Testing Tasks

- **MEM-005, MEM-006, MEM-009, etc.** - Unit tests embedded in each story (target 95% coverage)
- **MEM-008, MEM-010, MEM-022** - Integration tests for component interactions
- **MEM-019, MEM-033** - Load tests for performance validation
- **MEM-034** - COMPASS validation tests (benchmark suite)

### Quality Gates

- **90%+ test coverage** for all memory components
- **All critical paths have integration tests** (stage workflows, compression, retrieval)
- **Performance tests validate SLOs** (<100ms retrieval, 10:1 compression ratio)
- **COMPASS benchmarks met** (60-80% context reduction, +20% accuracy, 70-80% cost reduction)

---

## Team Allocation

**Backend Engineer (1 FTE):**

- Database and models (Week 1)
- Stage management (Week 2)
- Retrieval and embedding (Week 3)
- Compression infrastructure (Week 4-5)
- Error tracking (Week 6)
- Enhanced retrieval (Week 7)
- ACE integration (Week 8)
- Optimization and validation (Week 9-10)

**DevOps Support (0.2 FTE):**

- PGVector setup (Week 1)
- Redis configuration (Week 5)
- Monitoring setup (Week 10)

---

## Sprint Planning

**2-week sprints, 32-35 SP velocity (1 engineer)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 (Week 1-2) | Foundation + Stage Mgmt | 32 SP | Database, models, StageManager |
| Sprint 2 (Week 3-4) | Retrieval + Compression Start | 32 SP | Basic retrieval, compression core |
| Sprint 3 (Week 5-6) | Compression + Error Tracking | 33 SP | Task compression, error patterns |
| Sprint 4 (Week 7-8) | Enhanced Retrieval + ACE | 35 SP | Multi-factor scoring, ACE interface |
| Sprint 5 (Week 9-10) | Optimization + Validation | 33 SP | Performance tuning, COMPASS validation |

**Total: 165 SP over 10 weeks**

---

## Appendix

**Estimation Method:** Fibonacci story points based on complexity and effort
**Story Point Scale:** 1 (trivial), 2 (simple), 3 (moderate), 5 (complex), 8 (very complex), 13 (epic-size)

**Definition of Done:**

- Code implemented and reviewed
- Unit tests written (90%+ coverage for task)
- Integration tests passing
- Documentation updated (inline + API docs)
- Deployed to staging environment
- Performance validated (if applicable)
- COMPASS targets met (if applicable)

**COMPASS Validation Commitment:**

All tasks contributing to COMPASS targets (MEM-012 through MEM-034) will be validated against benchmarks:

- Context efficiency measured on 50-turn conversations
- Cost reduction validated via actual gpt-4o-mini vs gpt-4.1 comparison
- Compression quality validated via fact retention tests
- Long-horizon accuracy validated on GAIA-style evaluation dataset

---

**Document Status:** ✅ Ready for Ticket Generation
**Next Steps:** Generate story tickets (MEM-002 through MEM-036) in `.sage/tickets/`
