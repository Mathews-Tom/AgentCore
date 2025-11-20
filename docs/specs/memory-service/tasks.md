# Tasks: Memory Service (Hybrid: Mem0 + COMPASS + Graph)

**From:** `spec.md` v3.0 (Hybrid Architecture) + `plan.md` v3.0
**Timeline:** 8 weeks, 4 sprints (2-week sprints)
**Team:** 1 senior backend engineer (full-time)
**Created:** 2025-10-23
**Last Updated:** 2025-11-06 (Regenerated for hybrid architecture)

---

## Summary

- **Total tasks:** 27 story tasks (MEM-002 through MEM-028)
- **Estimated effort:** 175 story points (~8 weeks)
- **Critical path duration:** 8 weeks (60 SP critical tasks)
- **Key risks:**
  1. Neo4j integration complexity (mitigation: early deployment in Sprint 1)
  2. Hybrid search performance (mitigation: benchmark in Sprint 3)
  3. Memify algorithm effectiveness (mitigation: quality metrics validation)

**Hybrid Architecture Targets:**

- 60-80% context efficiency (COMPASS compression)
- 20% performance improvement on long-horizon tasks
- 70-80% cost reduction via test-time scaling
- 95%+ compression information retention
- <200ms graph traversal (p95, 2-hop)
- <300ms hybrid search (p95)
- 80%+ entity extraction accuracy
- 75%+ relationship detection accuracy
- 90%+ Memify consolidation accuracy

---

## Phase Breakdown

### Phase 1: Infrastructure & Core (Sprint 1, Weeks 1-2, 39 SP)

**Goal:** Deploy hybrid storage infrastructure (Qdrant + Neo4j) and establish foundational models
**Deliverable:** All databases operational, models defined, repositories functional

#### Week 1: Infrastructure Deployment (18 SP)

**MEM-002: Deploy Qdrant Vector Database**

- **Description:** Deploy Qdrant for vector similarity search (episodic, semantic, procedural memories)
- **Acceptance:**
  - [ ] Qdrant deployed via Docker Compose (dev) and K8s (prod)
  - [ ] Collections created for memory layers
  - [ ] Vector similarity search operational
  - [ ] Testcontainers integration for tests
  - [ ] <100ms vector search latency (p95)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** None
- **Priority:** P0 (Blocker for vector memory)
- **Files:**
  - `docker-compose.dev.yml` (Qdrant service)
  - `k8s/qdrant-deployment.yaml`
  - `tests/fixtures/qdrant_container.py`

**MEM-003: Deploy Neo4j Graph Database**

- **Description:** Deploy Neo4j for entity relationships and knowledge graphs with APOC + Graph Data Science plugins
- **Acceptance:**
  - [ ] Neo4j deployed via Docker Compose (dev) and K8s (prod)
  - [ ] APOC and Graph Data Science plugins installed
  - [ ] Graph schema defined (Memory, Entity, Concept nodes)
  - [ ] Testcontainers integration for tests
  - [ ] Connection pooling configured (neo4j-driver async)
  - [ ] <200ms graph traversal (p95, 2-hop queries)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** None
- **Priority:** P0 (Blocker for graph integration, HIGH RISK - front-loaded)
- **Files:**
  - `docker-compose.dev.yml` (Neo4j service)
  - `k8s/neo4j-statefulset.yaml`
  - `src/agentcore/memory/graph/schema.cypher`
  - `tests/fixtures/neo4j_container.py`

**MEM-004: Create Hybrid Database Migration**

- **Description:** Implement Alembic migration for stage_memories, task_contexts, error_records, compression_metrics tables
- **Acceptance:**
  - [ ] All 4 tables created (memories, stage_memories, task_contexts, error_records)
  - [ ] Compression_metrics table for cost tracking
  - [ ] Composite indexes for stage + agent + task queries
  - [ ] Vector indexes configured for Qdrant coordination
  - [ ] Migration reversible (downgrade tested)
  - [ ] Seed data loads successfully
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** None
- **Priority:** P0 (Blocker for all data access)
- **Files:**
  - `alembic/versions/XXX_add_hybrid_memory_tables.py`
  - `scripts/seed_memory_data.py`

#### Week 2: Models & Repositories (21 SP)

**MEM-005: Implement Hybrid Pydantic Models**

- **Description:** Create Pydantic models for MemoryRecord, StageMemory, TaskContext, ErrorRecord, EntityNode, RelationshipEdge
- **Acceptance:**
  - [ ] MemoryRecord with layer, stage, criticality fields
  - [ ] StageMemory with compression metrics
  - [ ] TaskContext with performance tracking
  - [ ] ErrorRecord with severity scoring (0-1)
  - [ ] EntityNode model for graph entities (NEW)
  - [ ] RelationshipEdge model for graph relationships (NEW)
  - [ ] All models support JSON serialization
  - [ ] Modern typing (`list[]`, `dict[]`, `|` unions)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-004
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/models.py`
  - `src/agentcore/memory/graph/models.py` (NEW)

**MEM-006: Implement SQLAlchemy ORM Models**

- **Description:** Create SQLAlchemy models matching database schema with async support
- **Acceptance:**
  - [ ] StageMemoryModel with JSONB for raw_memory_refs
  - [ ] TaskContextModel with foreign key to stage_memories
  - [ ] ErrorModel with error_type constraint validation
  - [ ] CompressionMetricsModel for cost tracking
  - [ ] All models use AsyncSession
  - [ ] Relationships configured (task → stages → memories)
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-005
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/database/models.py`

**MEM-007: Implement Repository Layer with Graph Support**

- **Description:** Create async repositories for Memory, StageMemory, TaskContext, Error, and Graph (Neo4j)
- **Acceptance:**
  - [ ] MemoryRepository with stage_id filtering
  - [ ] StageMemoryRepository with get_by_task_and_stage()
  - [ ] TaskContextRepository with current_stage tracking
  - [ ] ErrorRepository with pattern detection queries
  - [ ] GraphRepository for Neo4j operations (NEW)
  - [ ] All methods use async/await
  - [ ] Unit tests for each repository (90%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-006
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/database/repositories.py`
  - `src/agentcore/memory/graph/repository.py` (NEW)
  - `tests/unit/test_repositories.py`

---

### Phase 2: Stage Management & ECL Pipeline (Sprint 2, Weeks 3-4, 50 SP)

**Goal:** Implement COMPASS stage-aware context management and ECL pipeline foundation
**Deliverable:** Stage detection operational, ECL pipeline processing memories, entity extraction working

#### Week 3: Stage & ECL Foundation (21 SP)

**MEM-008: Implement StageManager Core**

- **Description:** Implement StageManager for COMPASS hierarchical organization with stage detection
- **Acceptance:**
  - [ ] Stage creation and lifecycle management
  - [ ] Stage type enum (planning, execution, reflection, verification)
  - [ ] Link memories to stages
  - [ ] Trigger compression on stage completion
  - [ ] Stage context retrieval
  - [ ] Unit tests (90%+ coverage)
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-007
- **Priority:** P0 (COMPASS core feature)
- **Files:**
  - `src/agentcore/memory/stage_manager.py`
  - `tests/unit/test_stage_manager.py`

**MEM-009: Implement Stage Detection Logic**

- **Description:** Implement automatic stage transition detection based on agent action patterns
- **Acceptance:**
  - [ ] Detect stage from tool usage patterns
  - [ ] Support explicit stage markers from agent
  - [ ] Time-based heuristics for default transitions
  - [ ] ACE intervention signal handling
  - [ ] 90%+ detection accuracy
  - [ ] Integration tests with sample workflows
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-008
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/stage_detection.py`
  - `tests/integration/test_stage_detection.py`

**MEM-010: Implement ECL Pipeline Base Classes**

- **Description:** Implement Extract, Cognify, Load pipeline base classes with task registry (Cognee-inspired)
- **Acceptance:**
  - [ ] TaskBase abstract class with execute() method
  - [ ] Pipeline class for task composition
  - [ ] Extract phase supports multiple data sources
  - [ ] Cognify phase interface for knowledge extraction
  - [ ] Load phase coordinates multi-backend storage
  - [ ] Async task execution support
  - [ ] Unit tests for pipeline framework
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-007
- **Priority:** P0 (NEW architecture pattern)
- **Files:**
  - `src/agentcore/memory/ecl/pipeline.py` (NEW)
  - `src/agentcore/memory/ecl/task_base.py` (NEW)
  - `tests/unit/test_ecl_pipeline.py`

**MEM-011: Implement Task Registry and Composition**

- **Description:** Implement task registry for discovering and composing ECL tasks
- **Acceptance:**
  - [ ] Task registry with registration decorator
  - [ ] Discover available tasks dynamically
  - [ ] Compose tasks into pipelines
  - [ ] Parallel execution where dependencies allow
  - [ ] Task-level error handling and retry logic
  - [ ] Unit tests for registry
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-010
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/ecl/registry.py` (NEW)
  - `tests/unit/test_task_registry.py`

#### Week 4: Compression & Entity Extraction (29 SP)

**MEM-012: Implement ContextCompressor with Test-Time Scaling** ✅ COMPLETED

- **Description:** Implement progressive compression using gpt-4.1-mini for cost optimization
- **Acceptance:**
  - [x] Stage compression (10:1 ratio target)
  - [x] Task compression (5:1 ratio target)
  - [x] Use gpt-4.1-mini for all compression (test-time scaling)
  - [x] Critical fact extraction
  - [x] Compression prompt optimization
  - [x] <5s compression latency (p95)
  - [x] Unit tests with mocked LLM
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-008
- **Priority:** P0 (COMPASS core feature)
- **Status:** COMPLETED (2025-11-21)
- **Files:**
  - `src/agentcore/a2a_protocol/services/memory/context_compressor.py` (616 lines)
  - `src/agentcore/a2a_protocol/services/memory/prompts/compression.py` (170 lines)
  - `tests/unit/services/memory/test_context_compressor.py` (531 lines)
- **Test Results:** 18/18 tests passing, 90%+ coverage
- **Implementation Notes:**
  - Implements CompressionTrigger protocol for StageManager integration
  - Achieves 10:1 stage compression and 5:1 task compression
  - Uses gpt-4.1-mini exclusively (test-time scaling)
  - Compression quality validation with 95%+ fact retention target
  - Cost tracking integration via CostTracker
  - Optimized prompts for different stage types (planning, execution, reflection, verification)
  - Error handling with fallback to truncated content on LLM failures
  - Heuristic quality validation fallback if LLM parsing fails

**MEM-013: Implement Compression Quality Validation**

- **Description:** Implement quality metrics for compression validation (fact retention, coherence)
- **Acceptance:**
  - [ ] Critical fact retention tracking (target: ≥95%)
  - [ ] Compression ratio validation (10:1, 5:1)
  - [ ] Coherence score (no contradictions)
  - [ ] Quality degradation alerts
  - [ ] Fallback to less aggressive compression if quality drops
  - [ ] Integration tests with real compression
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-012
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/quality_metrics.py`
  - `tests/integration/test_compression_quality.py`

**MEM-014: Implement Cost Tracking for Compression**

- **Description:** Implement token counting and cost tracking for compression operations
- **Acceptance:**
  - [ ] Track tokens per compression operation
  - [ ] Calculate cost using gpt-4.1-mini pricing
  - [ ] Monthly budget tracking
  - [ ] Alert at 75% budget consumption
  - [ ] Cost metrics stored in compression_metrics table
  - [ ] Dashboard query support
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-012
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/cost_tracker.py`
  - `tests/unit/test_cost_tracking.py`

**MEM-015: Implement Entity Extraction Task**

- **Description:** Implement ECL Cognify task for extracting entities (people, concepts, tools, constraints)
- **Acceptance:**
  - [ ] Extract entities from memory content using gpt-4.1-mini
  - [ ] Entity classification (person, concept, tool, constraint)
  - [ ] Entity normalization and deduplication
  - [ ] 80%+ extraction accuracy (target)
  - [ ] Integration with ECL pipeline
  - [ ] Unit tests with sample memories
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-010, MEM-011
- **Priority:** P0 (NEW feature for graph)
- **Files:**
  - `src/agentcore/memory/ecl/tasks/entity_extractor.py` (NEW)
  - `src/agentcore/memory/prompts/entity_extraction.py` (NEW)
  - `tests/unit/test_entity_extraction.py`

**MEM-016: Implement Entity Classification**

- **Description:** Implement entity type classification and confidence scoring
- **Acceptance:**
  - [ ] Classify entities by type (person, concept, tool, constraint, other)
  - [ ] Confidence scores for classifications
  - [ ] Handle ambiguous entities
  - [ ] Support custom entity types
  - [ ] Validation tests with diverse entity types
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-015
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/ecl/tasks/entity_classifier.py` (NEW)
  - `tests/unit/test_entity_classification.py`

---

### Phase 3: Graph Storage & Hybrid Retrieval (Sprint 3, Weeks 5-6, 47 SP)

**Goal:** Integrate Neo4j graph storage, implement relationship detection, and create hybrid search
**Deliverable:** Graph memory operational, hybrid search combining vector + graph

#### Week 5: Graph Storage & Relationships (26 SP)

**MEM-017: Implement GraphMemoryService (Neo4j Integration)**

- **Description:** Implement Neo4j integration for storing entities and relationships in knowledge graph
- **Acceptance:**
  - [ ] Store Memory, Entity, Concept nodes
  - [ ] Create MENTIONS, RELATES_TO, PART_OF relationships
  - [ ] Support temporal relationships (FOLLOWS, PRECEDES)
  - [ ] Index entities by type and properties
  - [ ] Traverse graph with depth 1-3 multi-hop queries
  - [ ] <200ms graph traversal (p95, 2-hop)
  - [ ] Async Neo4j driver integration
  - [ ] Unit tests with mocked Neo4j
  - [ ] Integration tests with real Neo4j (testcontainers)
- **Effort:** 13 story points (5-6 days) [HIGH COMPLEXITY]
- **Owner:** Backend Engineer
- **Dependencies:** MEM-003, MEM-007, MEM-010
- **Priority:** P0 (NEW critical feature, highest complexity)
- **Files:**
  - `src/agentcore/memory/graph/service.py` (NEW)
  - `src/agentcore/memory/graph/queries.cypher` (NEW)
  - `tests/unit/test_graph_service.py`
  - `tests/integration/test_neo4j_integration.py`

**MEM-018: Implement Relationship Detection Task**

- **Description:** Implement ECL Cognify task for detecting connections between entities
- **Acceptance:**
  - [ ] Detect relationships using LLM analysis
  - [ ] Pattern matching for common relationships
  - [ ] Relationship type classification (MENTIONS, RELATES_TO, PART_OF, etc.)
  - [ ] Relationship strength scoring
  - [ ] 75%+ detection accuracy (target)
  - [ ] Integration with ECL pipeline
  - [ ] Unit tests with sample entity pairs
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-015, MEM-017
- **Priority:** P0 (NEW feature for graph)
- **Files:**
  - `src/agentcore/memory/ecl/tasks/relationship_detector.py` (NEW)
  - `src/agentcore/memory/prompts/relationship_detection.py` (NEW)
  - `tests/unit/test_relationship_detection.py`

**MEM-019: Implement Graph Query Patterns**

- **Description:** Implement common Cypher query patterns for graph traversal and analysis
- **Acceptance:**
  - [ ] 1-hop neighbor queries
  - [ ] 2-hop relationship queries
  - [ ] 3-hop path finding
  - [ ] Entity similarity queries
  - [ ] Relationship strength aggregation
  - [ ] Query performance optimization (indexes)
  - [ ] Unit tests for each query pattern
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-017
- **Priority:** P1
- **Files:**
  - `src/agentcore/memory/graph/queries.py` (NEW)
  - `tests/unit/test_graph_queries.py`

#### Week 6: Hybrid Retrieval (21 SP)

**MEM-020: Implement Enhanced Retrieval Service**

- **Description:** Implement multi-factor importance scoring with stage relevance
- **Acceptance:**
  - [ ] Embedding similarity scoring (35% weight)
  - [ ] Recency decay scoring (15% weight)
  - [ ] Frequency scoring (10% weight)
  - [ ] Stage relevance scoring (20% weight)
  - [ ] Criticality boost (10% weight)
  - [ ] Error correction relevance (10% weight)
  - [ ] Configurable scoring weights
  - [ ] Unit tests for scoring algorithms
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-007
- **Priority:** P0 (COMPASS core feature)
- **Files:**
  - `src/agentcore/memory/retrieval.py`
  - `tests/unit/test_retrieval_scoring.py`

**MEM-021: Implement Hybrid Search (Vector + Graph)**

- **Description:** Implement hybrid search combining Qdrant vector search with Neo4j graph traversal
- **Acceptance:**
  - [ ] Vector search in Qdrant for semantic similarity
  - [ ] Graph traversal in Neo4j for contextual relationships
  - [ ] Merge and rank results from both databases
  - [ ] Relationship-based relevance boosting
  - [ ] Graph proximity scoring
  - [ ] <300ms hybrid search latency (p95)
  - [ ] 90%+ retrieval precision (target)
  - [ ] Integration tests with both databases
- **Effort:** 13 story points (5-6 days) [HIGH COMPLEXITY, novel approach]
- **Owner:** Backend Engineer
- **Dependencies:** MEM-002, MEM-017, MEM-020
- **Priority:** P0 (NEW critical feature)
- **Files:**
  - `src/agentcore/memory/hybrid_search.py` (NEW)
  - `tests/integration/test_hybrid_search.py`

**MEM-022: Implement Graph-Aware Context Expansion**

- **Description:** Implement context expansion using graph relationships for retrieval results
- **Acceptance:**
  - [ ] For each vector result, traverse Neo4j for related entities
  - [ ] Include 1-hop neighbors (direct connections)
  - [ ] Include 2-hop neighbors for critical memories
  - [ ] Preserve graph structure in context
  - [ ] Entity → relationship → entity format
  - [ ] Integration tests with hybrid search
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-021
- **Priority:** P1 (NEW feature)
- **Files:**
  - `src/agentcore/memory/context_expansion.py` (NEW)
  - `tests/unit/test_context_expansion.py`

---

### Phase 4: Optimization & Integration (Sprint 4, Weeks 7-8, 39 SP)

**Goal:** Implement Memify graph optimization, error tracking, and complete system integration
**Deliverable:** Production-ready memory service with all acceptance criteria met

#### Week 7: Memify & Error Tracking (18 SP)

**MEM-023: Implement Memify Graph Optimizer**

- **Description:** Implement Memify operation for graph optimization (consolidation, pruning, pattern detection)
- **Acceptance:**
  - [ ] Entity consolidation: Merge similar entities with >90% similarity
  - [ ] Relationship pruning: Remove low-value edges (access count < 2)
  - [ ] Pattern detection: Identify frequently traversed paths
  - [ ] Index optimization: Update Neo4j indexes based on query patterns
  - [ ] Quality metrics: Track connectivity, relationship density
  - [ ] <5s optimization per 1000 entities
  - [ ] 90%+ consolidation accuracy
  - [ ] <5% duplicate entities after optimization
  - [ ] Scheduled execution support (cron)
  - [ ] Unit tests for algorithms
- **Effort:** 13 story points (5-6 days) [HIGH COMPLEXITY, graph algorithms]
- **Owner:** Backend Engineer
- **Dependencies:** MEM-017, MEM-019
- **Priority:** P0 (NEW critical feature)
- **Files:**
  - `src/agentcore/memory/memify_optimizer.py` (NEW)
  - `src/agentcore/memory/memify/consolidation.py` (NEW)
  - `src/agentcore/memory/memify/pruning.py` (NEW)
  - `src/agentcore/memory/memify/patterns.py` (NEW)
  - `tests/unit/test_memify.py`

**MEM-024: Implement ErrorTracker**

- **Description:** Implement error tracking and pattern detection for COMPASS learning
- **Acceptance:**
  - [ ] Record errors with full context
  - [ ] Error type classification (hallucination, missing_info, incorrect_action, context_degradation)
  - [ ] Severity scoring (0-1)
  - [ ] Pattern detection: frequency, sequence, context correlation
  - [ ] Error history queries
  - [ ] Error-aware retrieval integration
  - [ ] ACE integration signals (error rate >30%)
  - [ ] 100% error capture rate
  - [ ] 80%+ pattern detection accuracy
  - [ ] Unit tests for pattern algorithms
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-007
- **Priority:** P0 (COMPASS core feature)
- **Files:**
  - `src/agentcore/memory/error_tracker.py`
  - `tests/unit/test_error_tracker.py`

#### Week 8: Integration & Testing (21 SP)

**MEM-025: Implement Service Integrations**

- **Description:** Integrate memory service with SessionManager, MessageRouter, TaskManager
- **Acceptance:**
  - [ ] SessionManager memory integration (session context)
  - [ ] MessageRouter memory-aware routing
  - [ ] TaskManager artifact storage
  - [ ] ACE strategic context interface
  - [ ] All cross-component contracts satisfied
  - [ ] Integration tests with each service
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-020, MEM-021, MEM-024
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/integrations/session.py`
  - `src/agentcore/memory/integrations/router.py`
  - `src/agentcore/memory/integrations/task.py`
  - `src/agentcore/memory/integrations/ace.py`
  - `tests/integration/test_service_integrations.py`

**MEM-026: Implement JSON-RPC Methods**

- **Description:** Implement JSON-RPC methods for memory service API
- **Acceptance:**
  - [ ] memory.store - Store new memory
  - [ ] memory.retrieve - Semantic search
  - [ ] memory.get_context - Formatted context retrieval
  - [ ] memory.complete_stage - Trigger compression
  - [ ] memory.record_error - Track error
  - [ ] memory.get_strategic_context - ACE interface
  - [ ] memory.run_memify - Trigger optimization (NEW)
  - [ ] All methods registered with @register_jsonrpc_method
  - [ ] Request/response validation with Pydantic
  - [ ] Unit tests for each method
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-025
- **Priority:** P0
- **Files:**
  - `src/agentcore/memory/jsonrpc.py`
  - `tests/unit/test_memory_jsonrpc.py`

**MEM-027: Implement Integration Tests**

- **Description:** Implement comprehensive integration tests for hybrid architecture
- **Acceptance:**
  - [ ] Real Qdrant instance (testcontainers)
  - [ ] Real Neo4j instance (testcontainers)
  - [ ] Test vector + graph coordination
  - [ ] Test ECL pipeline end-to-end
  - [ ] Test Memify operations
  - [ ] Test hybrid search accuracy
  - [ ] Test memory persistence across restarts
  - [ ] Test stage compression pipeline
  - [ ] Test error tracking workflow
  - [ ] 90%+ code coverage
  - [ ] All acceptance criteria validated
- **Effort:** 8 story points (3-4 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-026
- **Priority:** P0
- **Files:**
  - `tests/integration/test_hybrid_memory_service.py`
  - `tests/integration/test_ecl_pipeline_e2e.py`
  - `tests/integration/test_graph_vector_coordination.py`

**MEM-028: Implement Performance Validation**

- **Description:** Validate all performance targets and conduct benchmarking
- **Acceptance:**
  - [ ] Vector search: <100ms (p95) with 1M vectors
  - [ ] Graph traversal: <200ms (p95, 2-hop) with 100K nodes
  - [ ] Hybrid search: <300ms (p95) combined
  - [ ] Stage compression: <5s (p95)
  - [ ] Memify optimization: <5s per 1000 entities
  - [ ] Context efficiency: 60-80% reduction validated
  - [ ] Cost reduction: 70-80% validated
  - [ ] Entity extraction: 80%+ accuracy
  - [ ] Relationship detection: 75%+ accuracy
  - [ ] Memify consolidation: 90%+ accuracy
  - [ ] Load testing: 100+ concurrent operations
  - [ ] Performance benchmarks documented
- **Effort:** 5 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** MEM-027
- **Priority:** P0
- **Files:**
  - `tests/performance/test_memory_benchmarks.py`
  - `tests/load/test_memory_load.py`
  - `docs/performance/memory-service-benchmarks.md`

---

## Critical Path

```plaintext
MEM-004 → MEM-007 → MEM-010 → MEM-017 → MEM-021 → MEM-027
  (8d)      (8d)      (5d)       (13d)      (13d)      (8d)
              [55 days total ~8 weeks with parallelization]
```

**Bottlenecks:**

- MEM-017: GraphMemoryService (13 SP, highest complexity, NEW technology)
- MEM-021: Hybrid search (13 SP, novel approach, performance critical)
- MEM-023: Memify optimizer (13 SP, complex graph algorithms)

**Parallel Tracks:**

- Vector memory (MEM-002, MEM-012, MEM-020) can develop in parallel with graph (MEM-003, MEM-017)
- Entity extraction (MEM-015) parallel with compression (MEM-012)
- Error tracking (MEM-024) parallel with Memify (MEM-023)

---

## Quick Wins (Weeks 1-2)

1. **MEM-002: Qdrant deployment** (5 SP, Week 1)
   - Unblocks vector memory development
   - Well-documented, low risk
   - Early validation of vector search

2. **MEM-005: Pydantic models** (5 SP, Week 1)
   - Unblocks all service development
   - Demonstrates hybrid architecture (EntityNode, RelationshipEdge)

3. **MEM-003: Neo4j deployment** (8 SP, Week 1)
   - Front-loads highest risk
   - Early derisking of graph integration
   - Validates APOC plugin setup

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| MEM-003 | Neo4j deployment complexity | Front-load to Week 1, early validation | Fall back to simpler graph DB if issues |
| MEM-017 | GraphMemoryService complexity | Week 3 spike for Cypher patterns, testcontainers | Simplify to 1-hop only if needed |
| MEM-021 | Hybrid search performance | Benchmark in Sprint 3, optimize indexes | Fallback to vector-only with graph enrichment |
| MEM-023 | Memify algorithm effectiveness | Validate with quality metrics, iterate | Manual consolidation if algorithms fail |

---

## Testing Strategy

### Automated Testing Tasks

- **Unit Tests** (ongoing, embedded in each task) - 90%+ coverage required
- **MEM-027: Integration Tests** (8 SP) - Sprint 4, Week 8
- **MEM-028: Performance Validation** (5 SP) - Sprint 4, Week 8

### Quality Gates

- 90%+ code coverage (unit + integration)
- All critical paths have integration tests
- Performance tests validate all SLOs
- Graph + vector coordination validated
- Hybrid search accuracy ≥90%

### Testing Infrastructure

- **Testcontainers**: Qdrant, Neo4j, PostgreSQL, Redis
- **pytest-asyncio**: All async test support
- **pytest-benchmark**: Performance testing
- **Locust**: Load testing (100+ concurrent operations)

---

## Sprint Planning

**2-week sprints, 175 SP total (~43.75 SP/sprint)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 (Weeks 1-2) | Infrastructure | 39 SP | Qdrant + Neo4j deployed, models defined, repositories functional |
| Sprint 2 (Weeks 3-4) | Stage & ECL | 50 SP | StageManager operational, ECL pipeline processing, entity extraction working |
| Sprint 3 (Weeks 5-6) | Graph & Hybrid | 47 SP | Graph storage operational, hybrid search functional, performance validated |
| Sprint 4 (Weeks 7-8) | Optimization | 39 SP | Memify operational, error tracking complete, production-ready |

**Velocity assumption**: 1 senior backend engineer, 40-45 SP per 2-week sprint

---

## Team Allocation

**Backend (1 senior engineer)**

- All implementation tasks
- Architecture decisions (graph schema, hybrid search)
- Performance optimization

**QA (shared, 0.25 engineer)**

- Test review (MEM-027, MEM-028)
- Performance validation
- Load testing coordination

**DevOps (shared, 0.1 engineer)**

- Infrastructure deployment (MEM-002, MEM-003)
- K8s manifests
- CI/CD pipeline updates

---

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Dependencies,Sprint
MEM-002,Deploy Qdrant,Deploy Qdrant vector database,5,P0,,1
MEM-003,Deploy Neo4j,Deploy Neo4j graph database with plugins,8,P0,,1
MEM-004,Database Migration,Create hybrid database schema,8,P0,,1
MEM-005,Pydantic Models,Implement hybrid Pydantic models,5,P0,MEM-004,1
MEM-006,SQLAlchemy Models,Implement ORM models,5,P0,MEM-005,1
MEM-007,Repository Layer,Implement repositories with graph support,8,P0,MEM-006,1
MEM-008,StageManager,Implement COMPASS stage management,8,P0,MEM-007,2
MEM-009,Stage Detection,Implement stage transition detection,5,P0,MEM-008,2
MEM-010,ECL Pipeline Base,Implement ECL pipeline framework,5,P0,MEM-007,2
MEM-011,Task Registry,Implement task registry and composition,3,P0,MEM-010,2
MEM-012,ContextCompressor,Implement compression with test-time scaling,8,P0,MEM-008,2
MEM-013,Compression Validation,Implement quality metrics,5,P0,MEM-012,2
MEM-014,Cost Tracking,Implement cost tracking,3,P1,MEM-012,2
MEM-015,Entity Extraction,Implement entity extraction task,8,P0,"MEM-010,MEM-011",2
MEM-016,Entity Classification,Implement entity classification,5,P1,MEM-015,2
MEM-017,GraphMemoryService,Implement Neo4j integration,13,P0,"MEM-003,MEM-007,MEM-010",3
MEM-018,Relationship Detection,Implement relationship detection task,8,P0,"MEM-015,MEM-017",3
MEM-019,Graph Query Patterns,Implement Cypher query patterns,5,P1,MEM-017,3
MEM-020,Enhanced Retrieval,Implement multi-factor scoring,5,P0,MEM-007,3
MEM-021,Hybrid Search,Implement vector + graph hybrid search,13,P0,"MEM-002,MEM-017,MEM-020",3
MEM-022,Graph Context Expansion,Implement context expansion with graph,3,P1,MEM-021,3
MEM-023,Memify Optimizer,Implement graph optimization,13,P0,"MEM-017,MEM-019",4
MEM-024,ErrorTracker,Implement error tracking,5,P0,MEM-007,4
MEM-025,Service Integrations,Integrate with other services,5,P0,"MEM-020,MEM-021,MEM-024",4
MEM-026,JSON-RPC Methods,Implement API methods,3,P0,MEM-025,4
MEM-027,Integration Tests,Comprehensive integration testing,8,P0,MEM-026,4
MEM-028,Performance Validation,Validate all performance targets,5,P0,MEM-027,4
```

---

## Appendix

**Estimation Method:** Planning Poker with sequential thinking analysis
**Story Point Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)
**Definition of Done:**

- Code reviewed and approved
- Unit tests written and passing (90%+ coverage)
- Integration tests passing where applicable
- Performance targets validated
- Documentation updated (docstrings, architecture docs)
- Deployed to staging and smoke tested

**NEW Components (Hybrid Architecture)**:

- Neo4j graph database integration
- ECL Pipeline (Extract, Cognify, Load)
- Entity extraction and classification
- Relationship detection
- GraphMemoryService
- Hybrid search (vector + graph)
- Memify graph optimizer
- Graph-aware context expansion

**Technology Stack**:

- Vector DB: Qdrant
- Graph DB: Neo4j 5.15+ with APOC + Graph Data Science
- Cache: Redis
- Database: PostgreSQL
- Memory Framework: Mem0 ^0.1.0
- Embeddings: OpenAI text-embedding-3-small
- Compression: gpt-4.1-mini (test-time scaling)
- ORM: SQLAlchemy 2.0+ (async)
- Models: Pydantic 2.5+
- Testing: pytest + testcontainers
