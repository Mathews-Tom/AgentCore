# Semantic Capability Matching and Context Engineering Enhancement Specification

**Component:** A2A Protocol Layer - Phase 4 Enhancements
**Version:** 1.0
**Status:** Draft
**Created:** 2025-09-30
**Compliance:** A2A Protocol v0.2

---

## 1. Executive Summary

This specification details the implementation of semantic capability matching and context engineering enhancements to AgentCore's A2A Protocol Layer. These enhancements maintain full compliance with A2A Protocol v0.2 while providing significant competitive advantages through intelligent agent discovery and cost-optimized routing.

**Key Objectives:**

- Enable semantic capability matching using vector embeddings for flexible agent discovery
- Implement cost-biased optimization for intelligent agent selection
- Provide context engineering patterns for multi-stage workflows
- Maintain 100% backward compatibility with exact string matching
- Achieve >90% recall improvement vs exact matching, 20-30% cost reduction

**Business Impact:**

- 9-12 month competitive moat through unique combination of features
- 20-30% cost reduction in agent routing operations
- Enhanced developer experience with flexible capability descriptions
- Foundation for future federated agent architectures

---

## 2. Semantic Capability Matching

### 2.1 Overview

Semantic capability matching extends the A2A protocol's agent discovery mechanism with vector similarity search, allowing agents to be matched based on semantic meaning rather than exact string matching.

**Example:**

```text
Query: "analyze financial reports"
Exact Match: No results
Semantic Match: Finds agents with capabilities:
  - "financial document analysis" (similarity: 0.89)
  - "report processing and insights" (similarity: 0.82)
  - "accounting data extraction" (similarity: 0.78)
```

### 2.2 Technical Architecture

**Components:**

1. **Embedding Service**: Generates 768-dimensional vector embeddings from capability descriptions
2. **Vector Storage**: PostgreSQL with pgvector extension for efficient similarity search
3. **Discovery Service**: Enhanced to perform parallel exact + semantic search
4. **Message Router**: Consumes semantic match scores for intelligent routing

**Data Flow:**

```text
Agent Registration
    ↓
Capability Description → Embedding Service → 768-dim Vector
    ↓
PostgreSQL (pgvector) with HNSW Index
    ↓
Discovery Query → Parallel Search (Exact + Semantic)
    ↓
Ranked Results (Similarity Score + Metadata)
```

### 2.3 Implementation Details

#### 2.3.1 Database Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column to agent_capabilities table
ALTER TABLE agent_capabilities
ADD COLUMN embedding vector(768);

-- Create HNSW index for fast similarity search
CREATE INDEX ON agent_capabilities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Migration: Generate embeddings for existing capabilities
-- (handled by migration script)
```

#### 2.3.2 Embedding Service

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingService:
    """
    Generates semantic embeddings for capability descriptions.

    Uses sentence-transformers/all-MiniLM-L6-v2 for lightweight,
    CPU-based inference with <50ms latency target.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 768

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for single capability description.

        Args:
            text: Capability description string

        Returns:
            768-dimensional numpy array

        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Capability description cannot be empty")

        # Generate embedding with normalization
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple descriptions efficiently.

        Args:
            texts: List of capability descriptions

        Returns:
            List of 768-dimensional numpy arrays
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )

        return embeddings
```

#### 2.3.3 Enhanced Discovery Service

```python
from typing import List, Dict, Optional, Tuple
import asyncpg
from dataclasses import dataclass

@dataclass
class SemanticMatch:
    """Semantic search result with metadata."""
    agent_id: str
    capability_name: str
    similarity_score: float
    match_type: str  # "exact" or "semantic"

class SemanticDiscoveryService:
    """
    Enhanced agent discovery with semantic capability matching.

    Performs parallel exact string matching and semantic similarity
    search, combining results with configurable weights.
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.75
    ):
        self.db_pool = db_pool
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    async def discover_agents(
        self,
        capability_query: str,
        enable_semantic: bool = True
    ) -> List[SemanticMatch]:
        """
        Discover agents matching capability query.

        Args:
            capability_query: Natural language capability description
            enable_semantic: Enable semantic search (default: True)

        Returns:
            List of matching agents with similarity scores
        """
        # Parallel search: exact + semantic
        exact_task = self._exact_match_search(capability_query)

        if enable_semantic:
            semantic_task = self._semantic_match_search(capability_query)
            exact_results, semantic_results = await asyncio.gather(
                exact_task, semantic_task
            )

            # Combine and deduplicate results
            return self._merge_results(exact_results, semantic_results)
        else:
            # Backward compatibility: exact matching only
            return await exact_task

    async def _exact_match_search(
        self,
        capability_query: str
    ) -> List[SemanticMatch]:
        """Perform exact string matching search."""
        query = """
            SELECT agent_id, capability_name
            FROM agent_capabilities
            WHERE capability_name ILIKE $1
            AND agent_status = 'active'
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, f"%{capability_query}%")

        return [
            SemanticMatch(
                agent_id=row["agent_id"],
                capability_name=row["capability_name"],
                similarity_score=1.0,
                match_type="exact"
            )
            for row in rows
        ]

    async def _semantic_match_search(
        self,
        capability_query: str
    ) -> List[SemanticMatch]:
        """Perform semantic similarity search using pgvector."""
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(
            capability_query
        )

        query = """
            SELECT
                agent_id,
                capability_name,
                1 - (embedding <=> $1) AS similarity
            FROM agent_capabilities
            WHERE
                agent_status = 'active'
                AND 1 - (embedding <=> $1) >= $2
            ORDER BY similarity DESC
            LIMIT 50
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                query_embedding.tolist(),
                self.similarity_threshold
            )

        return [
            SemanticMatch(
                agent_id=row["agent_id"],
                capability_name=row["capability_name"],
                similarity_score=float(row["similarity"]),
                match_type="semantic"
            )
            for row in rows
        ]

    def _merge_results(
        self,
        exact_results: List[SemanticMatch],
        semantic_results: List[SemanticMatch]
    ) -> List[SemanticMatch]:
        """
        Merge and deduplicate exact + semantic results.

        Exact matches are prioritized over semantic matches.
        """
        # Create map by (agent_id, capability_name)
        results_map: Dict[Tuple[str, str], SemanticMatch] = {}

        # Add exact matches first (higher priority)
        for match in exact_results:
            key = (match.agent_id, match.capability_name)
            results_map[key] = match

        # Add semantic matches (only if not already in map)
        for match in semantic_results:
            key = (match.agent_id, match.capability_name)
            if key not in results_map:
                results_map[key] = match

        # Sort by similarity score descending
        return sorted(
            results_map.values(),
            key=lambda m: m.similarity_score,
            reverse=True
        )
```

### 2.4 Performance Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| Embedding generation | <50ms | Per capability description |
| Vector search latency | <20ms | For 1000+ agent registry |
| Discovery end-to-end | <100ms p95 | Including parallel search |
| Recall improvement | >90% | vs exact matching baseline |
| Index build time | <5min | For 10,000 capabilities |

### 2.5 Configuration

```yaml
# Environment variables
ENABLE_SEMANTIC_SEARCH: true
EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION: 768
SIMILARITY_THRESHOLD: 0.75
SEMANTIC_SEARCH_LIMIT: 50

# PostgreSQL pgvector configuration
PGVECTOR_INDEX_TYPE: hnsw
PGVECTOR_INDEX_M: 16          # Max connections per node
PGVECTOR_INDEX_EF_CONSTRUCTION: 64  # Build-time accuracy
```

---

## 3. Cost-Biased Agent Selection

### 3.1 Overview

Cost-biased optimization extends the message router with multi-objective scoring, enabling intelligent agent selection that balances similarity, latency, cost, and quality metrics.

**Objective Function:**

```
score = (w_sim × similarity) + (w_lat × latency_norm) + (w_cost × cost_norm) + (w_qual × quality)

Where:
  w_sim = 0.40 (similarity weight)
  w_lat = 0.30 (latency weight)
  w_cost = 0.20 (cost weight)
  w_qual = 0.10 (quality weight)
```

### 3.2 Enhanced Agent Metadata

```python
from pydantic import BaseModel, Field
from typing import Optional

class AgentCapabilityMetadata(BaseModel):
    """
    Enhanced agent capability with cost/performance metadata.

    Extends A2A protocol AgentCapability with optional fields
    for intelligent routing optimization.
    """

    # A2A v0.2 required fields
    capability_id: str
    capability_name: str
    description: str

    # Enhanced metadata (optional, backward compatible)
    cost_per_request: Optional[float] = Field(
        default=None,
        description="Average cost per request in USD"
    )
    avg_latency_ms: Optional[float] = Field(
        default=None,
        description="Average response latency in milliseconds"
    )
    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score from 0.0 to 1.0"
    )

    # Context engineering (optional)
    system_context: Optional[str] = Field(
        default=None,
        description="System prompt context for this capability"
    )
    interaction_examples: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Example interactions for context engineering"
    )
```

### 3.3 Intelligent Routing Algorithm

```python
from typing import List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class RoutingCandidate:
    """Agent candidate for routing with scoring metadata."""
    agent_id: str
    similarity_score: float
    cost_per_request: Optional[float]
    avg_latency_ms: Optional[float]
    quality_score: Optional[float]
    total_score: float = 0.0

class CostBiasedRouter:
    """
    Intelligent message router with cost-biased optimization.

    Implements multi-objective scoring to balance capability match,
    latency, cost, and quality for optimal agent selection.
    """

    def __init__(
        self,
        weight_similarity: float = 0.40,
        weight_latency: float = 0.30,
        weight_cost: float = 0.20,
        weight_quality: float = 0.10,
        max_latency_ms: Optional[float] = None,
        max_cost: Optional[float] = None
    ):
        self.w_sim = weight_similarity
        self.w_lat = weight_latency
        self.w_cost = weight_cost
        self.w_qual = weight_quality

        # Hard constraints
        self.max_latency_ms = max_latency_ms
        self.max_cost = max_cost

    def select_optimal_agent(
        self,
        candidates: List[RoutingCandidate]
    ) -> Optional[RoutingCandidate]:
        """
        Select optimal agent using multi-objective scoring.

        Args:
            candidates: List of agent candidates with metadata

        Returns:
            Optimal agent or None if no candidates pass constraints
        """
        if not candidates:
            return None

        # Apply hard constraints
        filtered = self._apply_constraints(candidates)
        if not filtered:
            return None

        # Calculate multi-objective scores
        scored = self._calculate_scores(filtered)

        # Return highest scoring candidate
        return max(scored, key=lambda c: c.total_score)

    def _apply_constraints(
        self,
        candidates: List[RoutingCandidate]
    ) -> List[RoutingCandidate]:
        """Apply hard constraints for latency and cost."""
        filtered = []

        for candidate in candidates:
            # Check latency constraint
            if self.max_latency_ms is not None:
                if (candidate.avg_latency_ms is not None and
                    candidate.avg_latency_ms > self.max_latency_ms):
                    continue

            # Check cost constraint
            if self.max_cost is not None:
                if (candidate.cost_per_request is not None and
                    candidate.cost_per_request > self.max_cost):
                    continue

            filtered.append(candidate)

        return filtered

    def _calculate_scores(
        self,
        candidates: List[RoutingCandidate]
    ) -> List[RoutingCandidate]:
        """
        Calculate multi-objective scores for all candidates.

        Normalizes each dimension and applies weighted sum.
        """
        # Extract metric arrays
        similarities = np.array([c.similarity_score for c in candidates])

        latencies = np.array([
            c.avg_latency_ms if c.avg_latency_ms is not None else 100.0
            for c in candidates
        ])

        costs = np.array([
            c.cost_per_request if c.cost_per_request is not None else 0.01
            for c in candidates
        ])

        qualities = np.array([
            c.quality_score if c.quality_score is not None else 0.75
            for c in candidates
        ])

        # Normalize metrics (similarity and quality: higher is better)
        sim_norm = similarities / similarities.max() if similarities.max() > 0 else similarities
        qual_norm = qualities / qualities.max() if qualities.max() > 0 else qualities

        # Normalize latency and cost (lower is better, so invert)
        lat_norm = 1.0 - (latencies / latencies.max()) if latencies.max() > 0 else np.ones_like(latencies)
        cost_norm = 1.0 - (costs / costs.max()) if costs.max() > 0 else np.ones_like(costs)

        # Calculate weighted scores
        for i, candidate in enumerate(candidates):
            candidate.total_score = (
                self.w_sim * sim_norm[i] +
                self.w_lat * lat_norm[i] +
                self.w_cost * cost_norm[i] +
                self.w_qual * qual_norm[i]
            )

        return candidates
```

### 3.4 Routing Metrics

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class RoutingMetrics:
    """Metrics for cost-biased routing optimization."""

    total_routings: int = 0
    cost_optimized_routings: int = 0
    total_cost_saved: float = 0.0
    avg_similarity_score: float = 0.0
    avg_latency_ms: float = 0.0

    def record_routing(
        self,
        selected_agent: RoutingCandidate,
        baseline_cost: Optional[float] = None
    ):
        """Record routing decision for metrics tracking."""
        self.total_routings += 1

        # Track cost savings
        if baseline_cost and selected_agent.cost_per_request:
            cost_saved = baseline_cost - selected_agent.cost_per_request
            if cost_saved > 0:
                self.cost_optimized_routings += 1
                self.total_cost_saved += cost_saved

        # Update running averages
        self.avg_similarity_score = (
            (self.avg_similarity_score * (self.total_routings - 1) +
             selected_agent.similarity_score) / self.total_routings
        )

        if selected_agent.avg_latency_ms:
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.total_routings - 1) +
                 selected_agent.avg_latency_ms) / self.total_routings
            )

    def get_cost_optimization_percentage(self) -> float:
        """Calculate percentage of routings with cost optimization."""
        if self.total_routings == 0:
            return 0.0
        return (self.cost_optimized_routings / self.total_routings) * 100
```

---

## 4. Context Engineering Patterns

### 4.1 Overview

Context engineering provides utilities and patterns for optimizing context propagation in multi-stage agent workflows, enabling efficient context accumulation and tracking.

### 4.2 Enhanced Task Artifact

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class TaskArtifact(BaseModel):
    """
    Enhanced A2A task artifact with context engineering support.

    Extends A2A v0.2 TaskArtifact with optional context metadata
    for multi-stage workflow tracking and debugging.
    """

    # A2A v0.2 required fields
    artifact_id: str
    task_id: str
    artifact_type: str
    content: Dict
    created_at: str

    # Context engineering fields (optional)
    context_summary: Optional[str] = Field(
        default=None,
        description="Summary of context used to generate this artifact"
    )
    context_lineage: Optional[List[str]] = Field(
        default=None,
        description="Chain of task IDs representing context flow"
    )
    metadata: Optional[Dict[str, any]] = Field(
        default=None,
        description="Structured metadata for downstream context propagation"
    )
```

### 4.3 ContextChain Utility

```python
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ContextNode:
    """Single node in context chain."""
    task_id: str
    agent_id: str
    context_summary: str
    artifact_id: Optional[str] = None

class ContextChain:
    """
    Utility for building and tracking context across multi-stage workflows.

    Enables progressive context enrichment where each workflow stage
    accumulates context from previous stages.
    """

    def __init__(self):
        self.nodes: List[ContextNode] = []

    def add_stage(
        self,
        task_id: str,
        agent_id: str,
        context_summary: str,
        artifact_id: Optional[str] = None
    ):
        """Add a new stage to the context chain."""
        node = ContextNode(
            task_id=task_id,
            agent_id=agent_id,
            context_summary=context_summary,
            artifact_id=artifact_id
        )
        self.nodes.append(node)

    def get_accumulated_context(self) -> str:
        """
        Get accumulated context summary from all stages.

        Returns:
            Concatenated context summaries from all workflow stages
        """
        return "\n\n".join([
            f"[Stage {i+1} - {node.agent_id}]\n{node.context_summary}"
            for i, node in enumerate(self.nodes)
        ])

    def get_lineage(self) -> List[str]:
        """Get list of task IDs representing context flow."""
        return [node.task_id for node in self.nodes]

    def to_metadata(self) -> Dict:
        """Export context chain as structured metadata."""
        return {
            "chain_length": len(self.nodes),
            "lineage": self.get_lineage(),
            "stages": [
                {
                    "task_id": node.task_id,
                    "agent_id": node.agent_id,
                    "summary": node.context_summary,
                    "artifact_id": node.artifact_id
                }
                for node in self.nodes
            ]
        }
```

### 4.4 Context Engineering Patterns

**Pattern 1: Progressive Enrichment**

```python
async def calendar_analysis_workflow(user_request: str):
    """
    Multi-stage workflow with progressive context enrichment.

    Stage 1: Calendar events extraction
    Stage 2: Analysis and synthesis
    Stage 3: Response generation with full context
    """
    context_chain = ContextChain()

    # Stage 1: Extract calendar events
    calendar_agent = await discover_agent("calendar access")
    events = await calendar_agent.execute({
        "query": user_request
    })

    context_chain.add_stage(
        task_id=events.task_id,
        agent_id=calendar_agent.id,
        context_summary=f"Retrieved {len(events.data)} calendar events",
        artifact_id=events.artifact_id
    )

    # Stage 2: Analyze events
    analysis_agent = await discover_agent("data analysis")
    analysis = await analysis_agent.execute({
        "events": events.data,
        "context": context_chain.get_accumulated_context()
    })

    context_chain.add_stage(
        task_id=analysis.task_id,
        agent_id=analysis_agent.id,
        context_summary="Analyzed calendar patterns and conflicts",
        artifact_id=analysis.artifact_id
    )

    # Stage 3: Generate response
    response_agent = await discover_agent("natural language generation")
    response = await response_agent.execute({
        "analysis": analysis.data,
        "context": context_chain.get_accumulated_context(),
        "metadata": context_chain.to_metadata()
    })

    return response
```

**Pattern 2: Context Isolation**

```python
async def isolated_task_execution(task_request: str):
    """
    Execute task with isolated context to prevent leakage.

    Useful for security-sensitive operations or when previous
    context should not influence current task.
    """
    agent = await discover_agent("secure computation")

    # Execute with minimal context
    result = await agent.execute({
        "task": task_request,
        "context": None,  # Explicit isolation
        "metadata": {
            "context_policy": "isolated",
            "no_lineage": True
        }
    })

    return result
```

---

## 5. Migration Strategy

### 5.1 Database Migration

```python
"""
Alembic migration: Add semantic search support

Revision ID: a2a_semantic_001
Create Date: 2025-09-30
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add embedding column
    op.add_column(
        'agent_capabilities',
        sa.Column('embedding', sa.String(), nullable=True)
    )

    # Add metadata columns for cost-biased routing
    op.add_column(
        'agent_capabilities',
        sa.Column('cost_per_request', sa.Float(), nullable=True)
    )
    op.add_column(
        'agent_capabilities',
        sa.Column('avg_latency_ms', sa.Float(), nullable=True)
    )
    op.add_column(
        'agent_capabilities',
        sa.Column('quality_score', sa.Float(), nullable=True)
    )

    # Add context engineering columns
    op.add_column(
        'agent_capabilities',
        sa.Column('system_context', sa.Text(), nullable=True)
    )
    op.add_column(
        'agent_capabilities',
        sa.Column('interaction_examples', sa.JSON(), nullable=True)
    )

    # Add context fields to task_artifacts
    op.add_column(
        'task_artifacts',
        sa.Column('context_summary', sa.Text(), nullable=True)
    )
    op.add_column(
        'task_artifacts',
        sa.Column('context_lineage', sa.JSON(), nullable=True)
    )

    print("✓ Columns added. Run embedding generation script next.")

def downgrade():
    # Remove columns in reverse order
    op.drop_column('task_artifacts', 'context_lineage')
    op.drop_column('task_artifacts', 'context_summary')

    op.drop_column('agent_capabilities', 'interaction_examples')
    op.drop_column('agent_capabilities', 'system_context')
    op.drop_column('agent_capabilities', 'quality_score')
    op.drop_column('agent_capabilities', 'avg_latency_ms')
    op.drop_column('agent_capabilities', 'cost_per_request')
    op.drop_column('agent_capabilities', 'embedding')
```

### 5.2 Embedding Generation Script

```python
"""
Script to generate embeddings for existing agent capabilities.

Usage:
    uv run python scripts/generate_embeddings.py --batch-size 100
"""

import asyncio
import asyncpg
from agentcore.a2a_protocol.services.embedding_service import EmbeddingService

async def generate_embeddings_for_existing_agents():
    """Generate embeddings for all existing agent capabilities."""

    # Initialize services
    embedding_service = EmbeddingService()
    db_pool = await asyncpg.create_pool(DATABASE_URL)

    try:
        # Fetch all capabilities without embeddings
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT capability_id, capability_name, description
                FROM agent_capabilities
                WHERE embedding IS NULL
            """)

        print(f"Generating embeddings for {len(rows)} capabilities...")

        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]

            # Create text descriptions
            texts = [
                f"{row['capability_name']} - {row['description']}"
                for row in batch
            ]

            # Generate embeddings
            embeddings = embedding_service.generate_embeddings_batch(texts)

            # Update database
            async with db_pool.acquire() as conn:
                for row, embedding in zip(batch, embeddings):
                    await conn.execute("""
                        UPDATE agent_capabilities
                        SET embedding = $1
                        WHERE capability_id = $2
                    """, embedding.tolist(), row['capability_id'])

            print(f"✓ Processed {min(i + batch_size, len(rows))}/{len(rows)}")

        # Create HNSW index
        print("Creating HNSW index...")
        async with db_pool.acquire() as conn:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_capabilities_embedding_idx
                ON agent_capabilities
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

        print("✓ Embedding generation complete!")

    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(generate_embeddings_for_existing_agents())
```

### 5.3 Backward Compatibility

All enhancements are designed to be backward compatible with A2A Protocol v0.2:

1. **Optional Fields**: All new metadata fields are optional
2. **Parallel Search**: Exact string matching remains as fallback
3. **Feature Flags**: Semantic search can be disabled via configuration
4. **Graceful Degradation**: Missing metadata uses default values

```python
# Example: Agent registration without enhanced metadata (still works)
agent_card = {
    "schema_version": "0.2",
    "agent_name": "legacy-agent",
    "capabilities": ["basic-task"],  # No enhanced metadata
    # ... other required A2A fields
}
# ✓ Agent registers successfully, uses exact matching only

# Example: Agent registration with enhanced metadata
agent_card = {
    "schema_version": "0.2",
    "agent_name": "enhanced-agent",
    "capabilities": [
        {
            "capability_name": "advanced-analysis",
            "description": "Perform advanced data analysis",
            "cost_per_request": 0.05,
            "avg_latency_ms": 250.0,
            "quality_score": 0.92
        }
    ],
    # ... other required A2A fields
}
# ✓ Agent registers with semantic search + cost-biased routing
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
import pytest
from agentcore.a2a_protocol.services.semantic_search import EmbeddingService

class TestEmbeddingService:
    """Unit tests for embedding generation."""

    def test_generate_embedding_success(self):
        service = EmbeddingService()
        embedding = service.generate_embedding("analyze financial data")

        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

    def test_generate_embedding_empty_text(self):
        service = EmbeddingService()

        with pytest.raises(ValueError):
            service.generate_embedding("")

    def test_embeddings_are_normalized(self):
        service = EmbeddingService()
        embedding = service.generate_embedding("test capability")

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Normalized to unit length
```

### 6.2 Integration Tests

```python
import pytest
from agentcore.a2a_protocol.services.discovery import SemanticDiscoveryService

@pytest.mark.asyncio
class TestSemanticDiscovery:
    """Integration tests for semantic agent discovery."""

    async def test_semantic_match_improves_recall(self, db_pool, embedding_service):
        """Verify semantic matching finds more agents than exact matching."""
        service = SemanticDiscoveryService(db_pool, embedding_service)

        # Exact match query
        exact_results = await service.discover_agents(
            "financial report analysis",
            enable_semantic=False
        )

        # Semantic match query
        semantic_results = await service.discover_agents(
            "analyze financial reports",
            enable_semantic=True
        )

        assert len(semantic_results) >= len(exact_results)
        assert len(semantic_results) > 0

    async def test_similarity_threshold_filtering(self, db_pool, embedding_service):
        """Verify similarity threshold filters low-quality matches."""
        service = SemanticDiscoveryService(
            db_pool,
            embedding_service,
            similarity_threshold=0.85  # High threshold
        )

        results = await service.discover_agents("test capability")

        for match in results:
            assert match.similarity_score >= 0.85
```

### 6.3 Performance Tests

```python
import pytest
import time
from agentcore.a2a_protocol.services.semantic_search import EmbeddingService

class TestPerformance:
    """Performance benchmarks for semantic search."""

    def test_embedding_generation_latency(self):
        """Verify embedding generation meets <50ms target."""
        service = EmbeddingService()

        start = time.perf_counter()
        service.generate_embedding("test capability description")
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 50.0

    @pytest.mark.asyncio
    async def test_vector_search_latency(self, db_pool, embedding_service):
        """Verify vector search meets <20ms target."""
        service = SemanticDiscoveryService(db_pool, embedding_service)

        start = time.perf_counter()
        await service._semantic_match_search("test query")
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 20.0

    @pytest.mark.asyncio
    async def test_discovery_end_to_end_latency(self, db_pool, embedding_service):
        """Verify end-to-end discovery meets <100ms p95 target."""
        service = SemanticDiscoveryService(db_pool, embedding_service)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await service.discover_agents("test capability")
            duration_ms = (time.perf_counter() - start) * 1000
            latencies.append(duration_ms)

        p95 = np.percentile(latencies, 95)
        assert p95 < 100.0
```

---

## 7. Monitoring and Metrics

### 7.1 Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Semantic search metrics
semantic_search_requests = Counter(
    'a2a_semantic_search_requests_total',
    'Total semantic search requests',
    ['match_type']  # exact, semantic, hybrid
)

embedding_generation_duration = Histogram(
    'a2a_embedding_generation_duration_seconds',
    'Duration of embedding generation',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

vector_search_duration = Histogram(
    'a2a_vector_search_duration_seconds',
    'Duration of vector similarity search',
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1]
)

semantic_match_recall = Gauge(
    'a2a_semantic_match_recall',
    'Recall rate of semantic matching vs exact matching'
)

# Cost optimization metrics
cost_optimization_rate = Gauge(
    'a2a_cost_optimization_rate',
    'Percentage of routings with cost optimization'
)

total_cost_saved = Counter(
    'a2a_total_cost_saved_usd',
    'Total cost saved through intelligent routing'
)

routing_score = Histogram(
    'a2a_routing_score',
    'Multi-objective routing scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
```

### 7.2 Grafana Dashboard

```yaml
Dashboard: A2A Semantic Search & Cost Optimization

Panels:
  1. Semantic Search Performance:
     - Embedding generation latency (p50, p95, p99)
     - Vector search latency (p50, p95, p99)
     - Discovery end-to-end latency

  2. Semantic Match Quality:
     - Recall improvement over exact matching
     - Similarity score distribution
     - Match type breakdown (exact/semantic/hybrid)

  3. Cost Optimization:
     - Cost optimization rate over time
     - Total cost saved (cumulative)
     - Routing score distribution
     - Average similarity vs average cost trade-off

  4. Infrastructure:
     - pgvector index size
     - PostgreSQL query performance
     - Embedding model inference time
```

---

## 8. Success Criteria

### 8.1 Functional Requirements

- [ ] Semantic capability matching with >90% recall vs exact matching
- [ ] Cost-biased routing achieves 20-30% cost reduction in benchmarks
- [ ] Context engineering utilities enable multi-stage workflows
- [ ] 100% backward compatibility with A2A Protocol v0.2
- [ ] Migration completes successfully for existing agents

### 8.2 Performance Requirements

- [ ] Embedding generation: <50ms per capability
- [ ] Vector search: <20ms for 1000+ agent registry
- [ ] Discovery end-to-end: <100ms p95
- [ ] No regression in exact string matching performance

### 8.3 Quality Requirements

- [ ] 95%+ unit test coverage
- [ ] Integration tests pass with pgvector
- [ ] Performance benchmarks meet all targets
- [ ] Documentation complete with examples
- [ ] Security review passed

---

## 9. References

### 9.1 Internal Documentation

- [A2A Protocol Specification](docs/specs/a2a-protocol/spec.md)
- [Implementation Plan](docs/specs/a2a-protocol/plan.md)
- [Task Breakdown](docs/specs/a2a-protocol/tasks.md)
- [Federation Research Analysis](docs/research/federation-context-engineering-analysis.md)

### 9.2 External Resources

- [A2A Protocol v0.2 Specification](https://a2aprotocol.ai/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [sentence-transformers](https://www.sbert.net/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [Federation of Agents Research (arxiv:2509.20175)](https://arxiv.org/abs/2509.20175)

---

**Document Status:** Draft
**Approval Required:** Engineering Lead, Product Manager
**Implementation Target:** Week 7-8 (Phase 4)
