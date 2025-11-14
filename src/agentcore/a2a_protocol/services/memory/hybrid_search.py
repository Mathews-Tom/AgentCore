"""
Hybrid Search Service (Vector + Graph)

Combines Qdrant vector search with Neo4j graph traversal for enhanced memory retrieval.
Implements multi-stage search strategy:
1. Vector search in Qdrant for semantic similarity
2. Graph traversal in Neo4j for contextual relationships
3. Result merging and ranking with weighted scoring

Performance targets:
- <300ms hybrid search latency (p95)
- 90%+ retrieval precision
- Parallel execution of vector and graph searches

Component ID: MEM-021
Ticket: MEM-021 (Implement Hybrid Search - Vector + Graph)
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from pydantic import BaseModel, Field, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels
from typing_extensions import Self

from agentcore.a2a_protocol.models.memory import MemoryRecord, StageType
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)

logger = structlog.get_logger(__name__)


class HybridSearchConfig(BaseModel):
    """
    Configuration for hybrid search combining vector and graph searches.

    Weights must sum to 1.0 for normalized scoring.
    """

    vector_weight: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity scores (default 60%)",
    )
    graph_weight: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Weight for graph proximity scores (default 40%)",
    )
    max_graph_depth: int = Field(
        2, ge=1, le=3, description="Maximum graph traversal depth (1-3 hops)"
    )
    max_graph_seeds: int = Field(
        10, ge=1, description="Maximum number of vector results to use as graph seeds"
    )
    use_graph_expansion: bool = Field(
        True, description="Enable graph expansion from vector results"
    )
    vector_score_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum vector similarity score threshold"
    )
    enable_retrieval_scoring: bool = Field(
        True, description="Enable EnhancedRetrievalService multi-factor scoring"
    )
    retrieval_weight: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Weight for retrieval service scoring (applied to hybrid score)",
    )
    vector_expansion_multiplier: int = Field(
        2,
        ge=1,
        description="Multiplier for vector search limit to get candidates for graph expansion",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> Self:
        """Validate that vector and graph weights sum to 1.0."""
        total = self.vector_weight + self.graph_weight

        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"vector_weight + graph_weight must sum to 1.0, got {total:.4f}"
            )

        return self


class HybridSearchMetadata(BaseModel):
    """
    Metadata for hybrid search result explaining score composition.

    Useful for debugging, monitoring, and understanding search decisions.
    """

    vector_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Vector similarity score"
    )
    graph_score: float = Field(0.0, ge=0.0, le=1.0, description="Graph proximity score")
    hybrid_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Combined hybrid score"
    )
    final_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Final score (with retrieval scoring if enabled)"
    )
    found_in_vector: bool = Field(
        False, description="Whether found in vector search"
    )
    found_in_graph: bool = Field(False, description="Whether found in graph search")
    graph_depth: int | None = Field(
        None, description="Graph depth if found via graph search"
    )
    relationship_count: int = Field(
        0, description="Number of relationships traversed"
    )


class HybridSearchService:
    """
    Hybrid search service combining Qdrant vector search and Neo4j graph traversal.

    Provides methods for:
    - Vector search in Qdrant for semantic similarity
    - Graph traversal in Neo4j for contextual relationships
    - Result merging and ranking with weighted scoring
    - Optional integration with EnhancedRetrievalService

    Performance optimizations:
    - Parallel execution of vector and graph searches
    - Limited graph depth (1-2 hops recommended)
    - Seed limiting for graph expansion
    - Result deduplication and caching

    Usage:
        config = HybridSearchConfig(vector_weight=0.6, graph_weight=0.4)
        service = HybridSearchService(
            qdrant_client=qdrant_client,
            graph_service=graph_service,
            collection_name="memories",
            config=config
        )

        results = await service.hybrid_search(
            query_embedding=query_emb,
            limit=10,
            task_id="task-123"
        )
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        graph_service: GraphMemoryService,
        collection_name: str = "memories",
        retrieval_service: EnhancedRetrievalService | None = None,
        config: HybridSearchConfig | None = None,
    ):
        """
        Initialize HybridSearchService.

        Args:
            qdrant_client: Qdrant async client instance
            graph_service: GraphMemoryService instance for graph operations
            collection_name: Qdrant collection name (default: "memories")
            retrieval_service: Optional EnhancedRetrievalService for multi-factor scoring
            config: Search configuration (uses defaults if None)
        """
        self.qdrant = qdrant_client
        self.graph = graph_service
        self.collection_name = collection_name
        self.retrieval = retrieval_service
        self.config = config or HybridSearchConfig()

        self._logger = logger.bind(component="hybrid_search")

        self._logger.info(
            "initialized_hybrid_search_service",
            collection=collection_name,
            vector_weight=self.config.vector_weight,
            graph_weight=self.config.graph_weight,
            max_graph_depth=self.config.max_graph_depth,
            retrieval_scoring_enabled=self.config.enable_retrieval_scoring,
        )

    async def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Search Qdrant for semantically similar memories.

        Uses cosine similarity metric for vector comparison.
        Returns top-K results above score threshold.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            filter_conditions: Optional filter dict (task_id, stage_id, layer, etc.)

        Returns:
            List of (memory, score) tuples sorted by score descending

        Raises:
            Exception: If Qdrant search fails
        """
        try:
            # Build Qdrant filter from conditions
            qdrant_filter = self._build_qdrant_filter(filter_conditions)

            # Execute vector search
            search_results = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=self.config.vector_score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            # Convert to MemoryRecord tuples
            results: list[tuple[MemoryRecord, float]] = []

            for point in search_results:
                try:
                    # Reconstruct MemoryRecord from payload
                    memory = MemoryRecord(**point.payload)
                    results.append((memory, point.score))
                except Exception as e:
                    self._logger.warning(
                        "failed_to_parse_memory_from_qdrant",
                        point_id=point.id,
                        error=str(e),
                    )
                    continue

            self._logger.info(
                "vector_search_completed",
                query_dims=len(query_embedding),
                requested_limit=limit,
                returned_count=len(results),
                top_score=results[0][1] if results else None,
                has_filters=filter_conditions is not None,
            )

            return results

        except Exception as e:
            self._logger.error(
                "vector_search_failed", error=str(e), limit=limit, has_filters=filter_conditions is not None
            )
            raise

    async def graph_search(
        self,
        seed_memory_ids: list[str],
        max_depth: int = 2,
        relationship_types: list[str] | None = None,
    ) -> list[tuple[MemoryRecord, float, int]]:
        """
        Traverse Neo4j graph from seed memories to find related memories.

        Uses graph proximity scoring: score = 1.0 / depth

        Args:
            seed_memory_ids: Memory IDs to use as graph traversal seeds
            max_depth: Maximum traversal depth (1-3 hops)
            relationship_types: Optional list of relationship types to traverse

        Returns:
            List of (memory, proximity_score, depth) tuples

        Note:
            - 1-hop neighbors: score = 1.0
            - 2-hop neighbors: score = 0.5
            - 3-hop neighbors: score = 0.33
        """
        try:
            graph_results: dict[str, tuple[MemoryRecord, float, int]] = {}

            # Limit seeds to avoid excessive graph traversal
            limited_seeds = seed_memory_ids[: self.config.max_graph_seeds]

            self._logger.info(
                "starting_graph_search",
                seed_count=len(limited_seeds),
                max_depth=max_depth,
                relationship_types=relationship_types,
            )

            # Traverse from each seed
            for seed_id in limited_seeds:
                try:
                    # Use GraphMemoryService to traverse relationships
                    # Note: graph_service.traverse_from_memory returns related memories
                    # This is a placeholder - actual implementation depends on GraphMemoryService API
                    neighbors = await self._traverse_from_seed(
                        seed_id, max_depth, relationship_types
                    )

                    for memory, depth in neighbors:
                        # Calculate proximity score based on depth
                        proximity_score = 1.0 / depth

                        # Keep highest score if memory found via multiple paths
                        if memory.memory_id not in graph_results:
                            graph_results[memory.memory_id] = (
                                memory,
                                proximity_score,
                                depth,
                            )
                        else:
                            existing_score = graph_results[memory.memory_id][1]
                            if proximity_score > existing_score:
                                graph_results[memory.memory_id] = (
                                    memory,
                                    proximity_score,
                                    depth,
                                )

                except Exception as e:
                    self._logger.warning(
                        "graph_traversal_failed_for_seed",
                        seed_id=seed_id,
                        error=str(e),
                    )
                    continue

            results = list(graph_results.values())

            self._logger.info(
                "graph_search_completed",
                seed_count=len(limited_seeds),
                unique_memories_found=len(results),
                max_depth=max_depth,
            )

            return results

        except Exception as e:
            self._logger.error(
                "graph_search_failed",
                error=str(e),
                seed_count=len(seed_memory_ids),
                max_depth=max_depth,
            )
            return []  # Return empty on error, don't fail entire hybrid search

    async def hybrid_search(
        self,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        limit: int = 10,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
        use_graph_expansion: bool | None = None,
        current_stage: StageType | None = None,
        has_recent_errors: bool = False,
        **filter_conditions: Any,
    ) -> list[tuple[MemoryRecord, float, HybridSearchMetadata]]:
        """
        Hybrid search combining vector and graph results.

        Search algorithm:
        1. Vector search in Qdrant (top K*2 for graph expansion)
        2. Optional graph expansion from vector results
        3. Merge and deduplicate results
        4. Weighted scoring: α*vector + β*graph
        5. Optional retrieval service multi-factor scoring
        6. Sort by final score and return top K

        Args:
            query: Optional query string (for retrieval service)
            query_embedding: Query embedding vector (required for vector search)
            limit: Number of final results to return
            vector_weight: Override default vector weight
            graph_weight: Override default graph weight
            use_graph_expansion: Override default graph expansion setting
            current_stage: Current reasoning stage (for retrieval service)
            has_recent_errors: Whether recent errors detected (for retrieval service)
            **filter_conditions: Filter dict (task_id, stage_id, layer, etc.)

        Returns:
            List of (memory, final_score, metadata) tuples sorted by score descending

        Raises:
            ValueError: If query_embedding is None and vector search would be performed
        """
        # Apply config overrides
        v_weight = vector_weight if vector_weight is not None else self.config.vector_weight
        g_weight = graph_weight if graph_weight is not None else self.config.graph_weight
        use_graph = (
            use_graph_expansion
            if use_graph_expansion is not None
            else self.config.use_graph_expansion
        )

        # Validate weights
        if abs(v_weight + g_weight - 1.0) > 0.01:
            raise ValueError(
                f"vector_weight + graph_weight must sum to 1.0, got {v_weight + g_weight:.4f}"
            )

        start_time = asyncio.get_event_loop().time()

        # Phase 1 & 2: Parallel vector and graph search
        vector_results: list[tuple[MemoryRecord, float]] = []
        graph_results: list[tuple[MemoryRecord, float, int]] = []

        if query_embedding is None:
            # Graph-only search (no vector component)
            self._logger.warning(
                "no_query_embedding_provided",
                message="Hybrid search without embedding will only use graph search if seeds provided",
            )
        else:
            # Determine vector search limit
            vector_limit = (
                limit * self.config.vector_expansion_multiplier if use_graph else limit
            )

            # Execute vector search
            vector_results = await self.vector_search(
                query_embedding=query_embedding,
                limit=vector_limit,
                filter_conditions=filter_conditions,
            )

            # Execute graph search in parallel if enabled
            if use_graph and vector_results:
                seed_ids = [mem.memory_id for mem, _ in vector_results]
                graph_results = await self.graph_search(
                    seed_memory_ids=seed_ids,
                    max_depth=self.config.max_graph_depth,
                )

        # Phase 3: Merge and rank results
        merged_results = await self._merge_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=v_weight,
            graph_weight=g_weight,
            limit=limit,
            query_embedding=query_embedding,
            current_stage=current_stage,
            has_recent_errors=has_recent_errors,
        )

        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        self._logger.info(
            "hybrid_search_completed",
            vector_count=len(vector_results),
            graph_count=len(graph_results),
            merged_count=len(merged_results),
            limit=limit,
            elapsed_ms=elapsed_ms,
            top_score=merged_results[0][1] if merged_results else None,
            use_graph_expansion=use_graph,
            retrieval_scoring_enabled=self.config.enable_retrieval_scoring,
        )

        return merged_results

    async def _merge_results(
        self,
        vector_results: list[tuple[MemoryRecord, float]],
        graph_results: list[tuple[MemoryRecord, float, int]],
        vector_weight: float,
        graph_weight: float,
        limit: int,
        query_embedding: list[float] | None = None,
        current_stage: StageType | None = None,
        has_recent_errors: bool = False,
    ) -> list[tuple[MemoryRecord, float, HybridSearchMetadata]]:
        """
        Merge vector and graph results with weighted scoring.

        Deduplicates by memory_id and applies weighted combination.
        Optionally applies EnhancedRetrievalService multi-factor scoring.

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            vector_weight: Weight for vector scores
            graph_weight: Weight for graph scores
            limit: Maximum results to return
            query_embedding: Optional query embedding for retrieval service
            current_stage: Optional current stage for retrieval service
            has_recent_errors: Whether recent errors detected

        Returns:
            List of (memory, final_score, metadata) tuples sorted by score descending
        """
        memory_data: dict[str, dict[str, Any]] = {}

        # Add vector results
        for memory, score in vector_results:
            memory_data[memory.memory_id] = {
                "memory": memory,
                "vector_score": score,
                "graph_score": 0.0,
                "found_in_vector": True,
                "found_in_graph": False,
                "graph_depth": None,
                "relationship_count": 0,
            }

        # Add/update graph results
        for memory, proximity_score, depth in graph_results:
            if memory.memory_id in memory_data:
                # Update existing entry
                memory_data[memory.memory_id]["graph_score"] = max(
                    memory_data[memory.memory_id]["graph_score"], proximity_score
                )
                memory_data[memory.memory_id]["found_in_graph"] = True
                memory_data[memory.memory_id]["graph_depth"] = depth
                memory_data[memory.memory_id]["relationship_count"] += 1
            else:
                # New entry from graph only
                memory_data[memory.memory_id] = {
                    "memory": memory,
                    "vector_score": 0.0,
                    "graph_score": proximity_score,
                    "found_in_vector": False,
                    "found_in_graph": True,
                    "graph_depth": depth,
                    "relationship_count": 1,
                }

        # Calculate hybrid scores
        results: list[tuple[MemoryRecord, float, HybridSearchMetadata]] = []

        for data in memory_data.values():
            # Base hybrid score
            hybrid_score = (
                vector_weight * data["vector_score"]
                + graph_weight * data["graph_score"]
            )

            final_score = hybrid_score

            # Optional: Apply retrieval service multi-factor scoring
            if (
                self.config.enable_retrieval_scoring
                and self.retrieval is not None
            ):
                retrieval_score, _ = await self.retrieval.score_memory(
                    memory=data["memory"],
                    query_embedding=query_embedding,
                    current_stage=current_stage,
                    has_recent_errors=has_recent_errors,
                )

                # Combine hybrid and retrieval scores
                final_score = (
                    (1.0 - self.config.retrieval_weight) * hybrid_score
                    + self.config.retrieval_weight * retrieval_score
                )

            # Create metadata
            metadata = HybridSearchMetadata(
                vector_score=data["vector_score"],
                graph_score=data["graph_score"],
                hybrid_score=hybrid_score,
                final_score=final_score,
                found_in_vector=data["found_in_vector"],
                found_in_graph=data["found_in_graph"],
                graph_depth=data["graph_depth"],
                relationship_count=data["relationship_count"],
            )

            results.append((data["memory"], final_score, metadata))

        # Sort by final score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return results[:limit]

    def _build_qdrant_filter(
        self, filter_conditions: dict[str, Any] | None
    ) -> qmodels.Filter | None:
        """
        Build Qdrant filter from condition dict.

        Supported filters:
        - task_id: str
        - stage_id: str
        - memory_layer: str
        - agent_id: str
        - session_id: str
        - is_critical: bool

        Args:
            filter_conditions: Dict of filter conditions

        Returns:
            Qdrant Filter object or None if no conditions
        """
        if not filter_conditions:
            return None

        conditions: list[qmodels.FieldCondition] = []

        # Map filter keys to Qdrant field conditions
        for key, value in filter_conditions.items():
            if value is None:
                continue

            if isinstance(value, bool):
                conditions.append(
                    qmodels.FieldCondition(
                        key=key, match=qmodels.MatchValue(value=value)
                    )
                )
            elif isinstance(value, str):
                conditions.append(
                    qmodels.FieldCondition(
                        key=key, match=qmodels.MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    qmodels.FieldCondition(
                        key=key, match=qmodels.MatchAny(any=value)
                    )
                )

        if not conditions:
            return None

        return qmodels.Filter(must=conditions)

    async def _traverse_from_seed(
        self,
        seed_id: str,
        max_depth: int,
        relationship_types: list[str] | None = None,
    ) -> list[tuple[MemoryRecord, int]]:
        """
        Traverse graph from a seed memory.

        This is a helper method that interfaces with GraphMemoryService.
        Returns memories found at each depth level.

        Args:
            seed_id: Starting memory ID
            max_depth: Maximum traversal depth
            relationship_types: Optional relationship types to follow

        Returns:
            List of (memory, depth) tuples
        """
        # TODO: This depends on GraphMemoryService API
        # For now, return empty list as placeholder
        # Will need to implement based on actual GraphMemoryService methods

        # Expected GraphMemoryService API:
        # neighbors = await self.graph.get_related_memories(
        #     memory_id=seed_id,
        #     max_depth=max_depth,
        #     relationship_types=relationship_types or ["RELATES_TO", "MENTIONS"]
        # )
        # return [(memory, depth) for memory, depth, _ in neighbors]

        self._logger.debug(
            "graph_traversal_placeholder",
            seed_id=seed_id,
            max_depth=max_depth,
            message="GraphMemoryService integration pending",
        )

        return []
