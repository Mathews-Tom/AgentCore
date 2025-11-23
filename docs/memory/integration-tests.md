# Memory System Integration Tests - MEM-027

**Created:** 2025-11-23
**Author:** Claude Code
**Status:** Implementation Complete
**Coverage Target:** 90%+

## Overview

This document describes the comprehensive integration testing strategy for the hybrid memory architecture (Mem0 + COMPASS + Graph). Tests validate the coordination between Qdrant vector storage, Neo4j graph storage, and the ECL pipeline.

## Test Environment

### Testcontainers Setup

All integration tests use real instances via testcontainers:

- **Qdrant**: `qdrant/qdrant:latest` (vector similarity search)
- **Neo4j**: `neo4j:5.15-community` with APOC plugin (graph relationships)
- **PostgreSQL**: In-memory SQLite for non-vector tables (metadata)
- **Redis**: testcontainers Redis (working memory cache)

### Fixtures Location

- `tests/integration/fixtures/qdrant.py` - Qdrant testcontainer fixtures
- `tests/integration/fixtures/neo4j.py` - Neo4j testcontainer fixtures
- `tests/integration/memory/conftest.py` - Memory-specific fixtures

## Test Coverage Areas

### 1. Vector + Graph Coordination (MEM-027.1)

**File:** `tests/integration/memory/test_hybrid_memory_coordination.py`

**Purpose:** Validate seamless coordination between Qdrant and Neo4j.

**Test Cases:**

1. **Store memory in both vector and graph databases**
   - Memory stored in Qdrant with embedding
   - Entities extracted and stored in Neo4j
   - Relationships created between entities
   - Cross-database consistency validated

2. **Retrieve memory using hybrid search**
   - Vector search finds similar memories in Qdrant
   - Graph traversal expands context with related entities
   - Results merged with relationship-based scoring
   - Hybrid results outperform vector-only search

3. **Update memory and maintain consistency**
   - Memory updated in Qdrant
   - Graph relationships updated in Neo4j
   - Consistency validated across databases

4. **Delete memory and cascade to graph**
   - Memory deleted from Qdrant
   - Related entities removed from Neo4j (if orphaned)
   - Relationships cleaned up

**Acceptance Criteria:**
- [ ] Memories stored in both databases with <300ms latency
- [ ] Hybrid search returns results from both databases
- [ ] Update operations maintain consistency
- [ ] Delete operations cascade correctly

### 2. ECL Pipeline End-to-End (MEM-027.2)

**File:** `tests/integration/memory/test_ecl_pipeline_e2e.py`

**Purpose:** Validate the Extract, Cognify, Load pipeline processes memories correctly.

**Test Cases:**

1. **Extract phase: Data ingestion**
   - Agent interactions extracted
   - Session context extracted
   - Task artifacts extracted
   - Error records extracted

2. **Cognify phase: Knowledge extraction**
   - Entity extraction task identifies entities
   - Relationship detection task finds connections
   - Critical fact extraction marks important info
   - Pattern recognition identifies themes

3. **Load phase: Multi-backend storage**
   - Memory content + embeddings stored in Qdrant
   - Entities + relationships stored in Neo4j
   - Metadata stored in PostgreSQL
   - Working memory updated in Redis

4. **Pipeline composition and execution**
   - Tasks registered in registry
   - Pipeline composed with multiple tasks
   - Parallel execution where allowed
   - Task-level error handling and retry

**Acceptance Criteria:**
- [ ] All 4 data sources ingested successfully
- [ ] Entity extraction accuracy ≥80%
- [ ] Relationship detection accuracy ≥75%
- [ ] All storage backends updated consistently
- [ ] Pipeline executes in <5s for 100 memories

### 3. Memify Operations (MEM-027.3)

**File:** `tests/integration/memory/test_memify_integration.py`

**Purpose:** Validate graph optimization operations.

**Test Cases:**

1. **Entity consolidation**
   - Identify similar entities (>90% similarity)
   - Merge duplicate entities
   - Consolidate relationships
   - Update references in memories

2. **Relationship pruning**
   - Identify low-value relationships (access count < 2)
   - Remove weak relationships
   - Maintain critical paths
   - Update graph indexes

3. **Pattern detection**
   - Identify frequently traversed paths
   - Detect relationship patterns
   - Calculate graph connectivity metrics
   - Optimize common queries

4. **Index optimization**
   - Analyze query patterns
   - Update Neo4j indexes
   - Benchmark query performance
   - Validate optimization effectiveness

**Acceptance Criteria:**
- [ ] Entity consolidation achieves 90%+ accuracy
- [ ] <5% duplicate entities after Memify
- [ ] Relationship pruning maintains critical paths
- [ ] Memify completes in <5s per 1000 entities
- [ ] Query performance improves after optimization

### 4. Hybrid Search Accuracy (MEM-027.4)

**File:** `tests/integration/memory/test_hybrid_search_accuracy.py`

**Purpose:** Validate hybrid search quality and performance.

**Test Cases:**

1. **Vector search baseline**
   - Pure vector similarity search in Qdrant
   - Measure precision, recall, F1 score
   - Benchmark query latency

2. **Graph search enhancement**
   - Graph traversal for contextual expansion
   - Relationship-based relevance boosting
   - Graph proximity scoring

3. **Hybrid search combination**
   - Merge vector + graph results
   - Rank with multi-factor scoring
   - Compare vs vector-only baseline

4. **Performance benchmarking**
   - Vector search latency (target: <100ms p95)
   - Graph traversal latency (target: <200ms p95, 2-hop)
   - Hybrid search latency (target: <300ms p95)
   - Scalability with 10K, 100K, 1M vectors

**Acceptance Criteria:**
- [ ] Hybrid search precision ≥90%
- [ ] Hybrid search outperforms vector-only by ≥10%
- [ ] Vector search <100ms (p95)
- [ ] Graph traversal <200ms (p95, 2-hop)
- [ ] Hybrid search <300ms (p95)

### 5. Memory Persistence Across Restarts (MEM-027.5)

**File:** `tests/integration/memory/test_persistence_integration.py`

**Purpose:** Validate data persists correctly across service restarts.

**Test Cases:**

1. **Qdrant persistence**
   - Store memories in Qdrant
   - Restart Qdrant container
   - Verify memories still accessible
   - Validate vector search still works

2. **Neo4j persistence**
   - Store entities and relationships in Neo4j
   - Restart Neo4j container
   - Verify graph structure intact
   - Validate traversal queries work

3. **PostgreSQL persistence**
   - Store metadata in PostgreSQL
   - Restart database
   - Verify metadata accessible
   - Validate relationships intact

4. **Cross-database consistency after restart**
   - Store memory across all databases
   - Restart all containers
   - Verify consistency maintained
   - Validate hybrid search works

**Acceptance Criteria:**
- [ ] Qdrant memories persist across restart
- [ ] Neo4j graph structure persists
- [ ] PostgreSQL metadata persists
- [ ] Cross-database consistency maintained
- [ ] All queries functional after restart

### 6. Stage Compression Pipeline (MEM-027.6)

**File:** `tests/integration/memory/test_compression_integration.py`

**Purpose:** Validate COMPASS stage compression pipeline.

**Test Cases:**

1. **Stage completion workflow**
   - Detect stage transition
   - Trigger compression
   - Compress raw memories to stage summary
   - Validate 10:1 compression ratio

2. **Task compression workflow**
   - Complete multiple stages
   - Compress stage summaries to task progress
   - Validate 5:1 compression ratio

3. **Compression quality validation**
   - Critical fact retention ≥95%
   - No contradictions introduced
   - Coherence score validation

4. **Cost tracking integration**
   - Track tokens per compression
   - Calculate cost using gpt-4.1-mini pricing
   - Monitor monthly budget usage
   - Alert at 75% budget consumption

**Acceptance Criteria:**
- [ ] Stage compression achieves 10:1 ratio (±20%)
- [ ] Task compression achieves 5:1 ratio (±20%)
- [ ] Critical fact retention ≥95%
- [ ] No contradictions introduced
- [ ] Cost tracking accurate to <1%

### 7. Error Tracking Workflow (MEM-027.7)

**File:** `tests/integration/memory/test_error_tracking_integration.py`

**Purpose:** Validate error tracking and pattern detection.

**Test Cases:**

1. **Error recording**
   - Record error with full context
   - Classify error type (hallucination, missing_info, etc.)
   - Score severity (0-1 scale)
   - Link to task and stage

2. **Pattern detection**
   - Detect recurring error types (frequency analysis)
   - Detect error sequences (pattern matching)
   - Detect context correlation (similar contexts)
   - LLM-based pattern extraction

3. **Error-aware retrieval**
   - Boost memories that corrected errors
   - Retrieve error-prevention knowledge
   - Provide error context to ACE
   - Mark error-related memories as critical

4. **ACE integration signals**
   - Alert ACE when error rate >30%
   - Provide strategic context for intervention
   - Track intervention outcomes
   - Update criticality scores

**Acceptance Criteria:**
- [ ] 100% error capture rate
- [ ] Pattern detection accuracy ≥80%
- [ ] Error-aware retrieval improves precision by ≥5%
- [ ] ACE receives signals for high error rates

## Coverage Requirements

### Code Coverage Targets

- **Overall:** 90%+ coverage for memory service
- **Integration tests:** Cover all acceptance criteria
- **Critical paths:** 100% coverage for hybrid search, ECL pipeline, Memify

### Coverage Measurement

```bash
# Run integration tests with coverage
uv run pytest tests/integration/memory/ --cov=src/agentcore/a2a_protocol/services/memory --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html
```

### Coverage Validation

```bash
# Ensure 90%+ coverage
uv run pytest tests/integration/memory/ --cov=src/agentcore/a2a_protocol/services/memory --cov-fail-under=90
```

## Running Tests

### All Integration Tests

```bash
# Run all memory integration tests
uv run pytest tests/integration/memory/ -v

# Run with testcontainers (requires Docker)
uv run pytest tests/integration/memory/ --tb=short
```

### Specific Test Suites

```bash
# Vector + Graph coordination
uv run pytest tests/integration/memory/test_hybrid_memory_coordination.py -v

# ECL pipeline
uv run pytest tests/integration/memory/test_ecl_pipeline_e2e.py -v

# Memify operations
uv run pytest tests/integration/memory/test_memify_integration.py -v

# Hybrid search
uv run pytest tests/integration/memory/test_hybrid_search_accuracy.py -v

# Persistence
uv run pytest tests/integration/memory/test_persistence_integration.py -v

# Compression
uv run pytest tests/integration/memory/test_compression_integration.py -v

# Error tracking
uv run pytest tests/integration/memory/test_error_tracking_integration.py -v
```

### Performance Benchmarking

```bash
# Run performance tests with benchmarking
uv run pytest tests/integration/memory/test_hybrid_search_accuracy.py::TestPerformanceBenchmark -v --benchmark-only
```

## Continuous Integration

### GitHub Actions Workflow

Integration tests run on:
- **Pull Requests:** Against `main` branch
- **Push to main:** Full test suite
- **Nightly:** Extended tests with 1M vectors

### CI Configuration

```yaml
# .github/workflows/memory-integration-tests.yml
name: Memory Integration Tests

on:
  pull_request:
    paths:
      - 'src/agentcore/a2a_protocol/services/memory/**'
      - 'tests/integration/memory/**'

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      docker:
        image: docker:latest

    steps:
      - uses: actions/checkout@v4
      - name: Setup uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync
      - name: Run integration tests
        run: |
          uv run pytest tests/integration/memory/ \
            --cov=src/agentcore/a2a_protocol/services/memory \
            --cov-fail-under=90 \
            --tb=short \
            -v
```

## Debugging Integration Tests

### Testcontainer Logs

```python
# Access container logs in tests
def test_example(qdrant_container):
    # Get logs for debugging
    logs = qdrant_container.get_logs()
    print(logs)
```

### Interactive Debugging

```bash
# Run single test with debugging
uv run pytest tests/integration/memory/test_hybrid_memory_coordination.py::test_store_memory_hybrid -v --pdb

# Keep containers running after test failure
uv run pytest tests/integration/memory/ --log-cli-level=DEBUG
```

### Neo4j Browser

```python
# Connect to Neo4j container during test
# Get connection URL from fixture
neo4j_uri = "bolt://localhost:PORT"
# Open http://localhost:7474 in browser
# Login with neo4j/testpassword
```

## Test Data Fixtures

### Sample Memories

```python
# tests/integration/memory/fixtures/sample_data.py

SAMPLE_MEMORIES = [
    {
        "content": "User prefers detailed technical explanations",
        "memory_layer": "semantic",
        "agent_id": "agent-1",
        "is_critical": True,
    },
    {
        "content": "Successfully implemented JWT authentication",
        "memory_layer": "episodic",
        "task_id": "task-1",
        "stage_type": "execution",
    },
    # ... more samples
]

SAMPLE_ENTITIES = [
    {
        "name": "JWT",
        "type": "concept",
        "confidence": 0.95,
    },
    {
        "name": "authentication",
        "type": "concept",
        "confidence": 0.90,
    },
]

SAMPLE_RELATIONSHIPS = [
    {
        "source": "JWT",
        "target": "authentication",
        "type": "USED_FOR",
        "strength": 0.85,
    },
]
```

### Mock LLM Responses

```python
# For tests that need LLM responses without API calls
MOCK_ENTITY_EXTRACTION = {
    "entities": [
        {"name": "JWT", "type": "concept"},
        {"name": "Redis", "type": "tool"},
    ]
}

MOCK_COMPRESSION_OUTPUT = {
    "summary": "Implemented JWT auth using Redis storage",
    "critical_facts": ["JWT tokens", "Redis storage"],
    "compression_ratio": 10.2,
}
```

## Troubleshooting

### Common Issues

1. **Testcontainers not starting:**
   - Ensure Docker is running
   - Check Docker memory limits (need ≥4GB)
   - Verify network connectivity

2. **Neo4j APOC plugin not loaded:**
   - Check container logs for plugin errors
   - Verify environment variables set correctly
   - Increase container startup wait time

3. **Tests timing out:**
   - Increase pytest timeout: `pytest --timeout=300`
   - Check container resource usage
   - Verify network latency

4. **Flaky tests:**
   - Add explicit waits for async operations
   - Increase retry attempts
   - Check for race conditions

### Performance Issues

If tests are slow:
- Use session-scoped fixtures for containers
- Parallelize tests with `pytest-xdist`
- Use smaller datasets for quick validation
- Run performance tests separately

## Acceptance Criteria Validation

### MEM-027 Checklist

- [ ] Real Qdrant instance using testcontainers
- [ ] Real Neo4j instance using testcontainers
- [ ] Test vector + graph coordination
- [ ] Test ECL pipeline end-to-end
- [ ] Test Memify operations
- [ ] Test hybrid search accuracy
- [ ] Test memory persistence across restarts
- [ ] Test stage compression pipeline
- [ ] Test error tracking workflow
- [ ] 90%+ code coverage
- [ ] All acceptance criteria validated

## References

- **Testcontainers Python:** https://testcontainers-python.readthedocs.io/
- **Qdrant Client:** https://qdrant.tech/documentation/frameworks/python/
- **Neo4j Python Driver:** https://neo4j.com/docs/api/python-driver/current/
- **pytest-asyncio:** https://pytest-asyncio.readthedocs.io/
- **Coverage.py:** https://coverage.readthedocs.io/

## Changelog

- **2025-11-23:** Initial documentation created for MEM-027 implementation
