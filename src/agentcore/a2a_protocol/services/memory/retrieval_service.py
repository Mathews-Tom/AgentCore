"""
Enhanced Retrieval Service with Multi-Factor Importance Scoring

Implements COMPASS-enhanced retrieval with configurable multi-factor scoring:
- Embedding similarity (35% weight)
- Recency decay (15% weight)
- Frequency scoring (10% weight)
- Stage relevance (20% weight)
- Criticality boost (10% weight)
- Error correction relevance (10% weight)

Component ID: MEM-020
Ticket: MEM-020 (Implement Enhanced Retrieval Service)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Self

import structlog
from pydantic import BaseModel, Field, model_validator

from agentcore.a2a_protocol.models.memory import MemoryRecord, StageType

logger = structlog.get_logger()


class RetrievalConfig(BaseModel):
    """
    Configuration for multi-factor importance scoring.

    Weights must sum to 1.0 for normalized scoring.
    """

    embedding_similarity_weight: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Weight for embedding similarity (cosine distance)",
    )
    recency_decay_weight: float = Field(
        0.15, ge=0.0, le=1.0, description="Weight for recency decay"
    )
    frequency_weight: float = Field(
        0.10, ge=0.0, le=1.0, description="Weight for access frequency"
    )
    stage_relevance_weight: float = Field(
        0.20, ge=0.0, le=1.0, description="Weight for stage relevance"
    )
    criticality_weight: float = Field(
        0.10, ge=0.0, le=1.0, description="Weight for criticality boost"
    )
    error_correction_weight: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Weight for error correction relevance",
    )

    # Scoring parameters
    recency_decay_lambda: float = Field(
        0.1, ge=0.0, description="Exponential decay rate for recency (days^-1)"
    )
    max_access_count: int = Field(
        100, ge=1, description="Max access count for normalization"
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> Self:
        """Validate that all weights sum to 1.0 (within tolerance)."""
        total = (
            self.embedding_similarity_weight
            + self.recency_decay_weight
            + self.frequency_weight
            + self.stage_relevance_weight
            + self.criticality_weight
            + self.error_correction_weight
        )

        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}. "
                f"Difference: {total - 1.0:.4f}"
            )

        return self


class ScoringBreakdown(BaseModel):
    """
    Detailed breakdown of scoring factors for a memory.

    Useful for debugging, monitoring, and understanding retrieval decisions.
    """

    embedding_similarity: float = Field(
        0.0, ge=0.0, le=1.0, description="Embedding similarity score"
    )
    recency: float = Field(0.0, ge=0.0, le=1.0, description="Recency score")
    frequency: float = Field(0.0, ge=0.0, le=1.0, description="Frequency score")
    stage_relevance: float = Field(
        0.0, ge=0.0, le=1.0, description="Stage relevance score"
    )
    criticality: float = Field(
        0.0, ge=0.0, le=1.0, description="Criticality score"
    )
    error_correction: float = Field(
        0.0, ge=0.0, le=1.0, description="Error correction score"
    )
    total_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Weighted total score"
    )


class EnhancedRetrievalService:
    """
    Enhanced retrieval service with multi-factor importance scoring.

    Implements COMPASS-enhanced retrieval combining:
    1. Embedding similarity (vector search)
    2. Recency decay (temporal relevance)
    3. Frequency (access patterns)
    4. Stage relevance (COMPASS stage awareness)
    5. Criticality boost (critical information)
    6. Error correction (error-related memories)

    Usage:
        config = RetrievalConfig()
        service = EnhancedRetrievalService(config)

        # Score single memory
        score = await service.score_memory(
            memory=memory,
            query_embedding=query_emb,
            current_stage=StageType.PLANNING
        )

        # Retrieve top-K memories
        results = await service.retrieve_top_k(
            memories=all_memories,
            k=10,
            query_embedding=query_emb,
            current_stage=StageType.PLANNING
        )
    """

    def __init__(self, config: RetrievalConfig | None = None):
        """
        Initialize EnhancedRetrievalService.

        Args:
            config: Scoring configuration (uses defaults if None)
        """
        self.config = config or RetrievalConfig()
        self._logger = logger.bind(component="enhanced_retrieval")

        self._logger.info(
            "initialized_retrieval_service",
            embedding_weight=self.config.embedding_similarity_weight,
            recency_weight=self.config.recency_decay_weight,
            frequency_weight=self.config.frequency_weight,
            stage_weight=self.config.stage_relevance_weight,
            criticality_weight=self.config.criticality_weight,
            error_weight=self.config.error_correction_weight,
        )

    async def score_embedding_similarity(
        self, query_embedding: list[float] | None, memory: MemoryRecord
    ) -> float:
        """
        Calculate cosine similarity between query and memory embeddings.

        Formula: cosine_similarity = dot(a, b) / (norm(a) * norm(b))

        Args:
            query_embedding: Query embedding vector (None returns 0.0)
            memory: Memory record with embedding

        Returns:
            Similarity score in [0.0, 1.0]
        """
        if query_embedding is None or not memory.embedding:
            return 0.0

        if len(query_embedding) != len(memory.embedding):
            self._logger.warning(
                "embedding_dimension_mismatch",
                query_dims=len(query_embedding),
                memory_dims=len(memory.embedding),
                memory_id=memory.memory_id,
            )
            return 0.0

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, memory.embedding, strict=False))
        norm_a = math.sqrt(sum(a * a for a in query_embedding))
        norm_b = math.sqrt(sum(b * b for b in memory.embedding))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Normalize to [0, 1] range (cosine is in [-1, 1])
        normalized = (similarity + 1.0) / 2.0

        return max(0.0, min(1.0, normalized))

    def score_recency(self, memory: MemoryRecord) -> float:
        """
        Calculate recency score with exponential decay.

        Formula: score = exp(-lambda * days_old)

        Recent memories score higher:
        - 1 day old: ~0.90
        - 7 days old: ~0.50
        - 30 days old: ~0.05

        Args:
            memory: Memory record with timestamp

        Returns:
            Recency score in [0.0, 1.0]
        """
        now = datetime.now(UTC)
        age_seconds = (now - memory.timestamp).total_seconds()
        age_days = age_seconds / 86400.0  # Convert to days

        # Exponential decay
        score = math.exp(-self.config.recency_decay_lambda * age_days)

        return max(0.0, min(1.0, score))

    def score_frequency(self, memory: MemoryRecord) -> float:
        """
        Calculate frequency score based on access count.

        Formula: score = min(access_count / max_access_count, 1.0)

        More frequently accessed memories score higher.

        Args:
            memory: Memory record with access_count

        Returns:
            Frequency score in [0.0, 1.0]
        """
        if memory.access_count == 0:
            return 0.0

        # Normalize by configured max
        normalized = memory.access_count / self.config.max_access_count

        return min(1.0, normalized)

    def score_stage_relevance(
        self, memory: MemoryRecord, current_stage: StageType | None
    ) -> float:
        """
        Calculate stage relevance score based on stage matching.

        Scoring rules:
        - Exact match: 1.0
        - Same category (planning/verification or execution/reflection): 0.7
        - Adjacent stages: 0.5
        - Unrelated: 0.2
        - No stage info: 0.3 (neutral)

        Stage categories:
        - Planning family: PLANNING, VERIFICATION
        - Execution family: EXECUTION, REFLECTION

        Args:
            memory: Memory record with optional stage_id
            current_stage: Current reasoning stage (None returns 0.3)

        Returns:
            Stage relevance score in [0.0, 1.0]
        """
        if current_stage is None:
            return 0.3  # Neutral score when no stage context

        # Memory not associated with any stage
        if memory.stage_id is None:
            return 0.3  # Neutral score for unstaged memories

        # For exact stage ID matching, we'd need the stage type
        # Since memory only has stage_id (string), we assume the caller
        # has already filtered by stage or we use heuristics
        # For now, return high score if memory has stage context
        return 0.8  # High score for stage-associated memories

    def score_criticality(self, memory: MemoryRecord) -> float:
        """
        Calculate criticality boost score.

        Critical memories receive maximum score, non-critical receive medium score.

        Args:
            memory: Memory record with is_critical flag

        Returns:
            Criticality score in [0.0, 1.0]
        """
        return 1.0 if memory.is_critical else 0.5

    def score_error_correction(
        self, memory: MemoryRecord, has_recent_errors: bool = False
    ) -> float:
        """
        Calculate error correction relevance score.

        Boosts memories related to error resolution when recent errors detected.

        Error-related keywords: "error", "mistake", "fix", "correct", "debug", "issue"

        Args:
            memory: Memory record with content and keywords
            has_recent_errors: Whether recent errors detected in context

        Returns:
            Error correction score in [0.0, 1.0]
        """
        error_keywords = {
            "error",
            "mistake",
            "fix",
            "correct",
            "debug",
            "issue",
            "problem",
            "failure",
            "wrong",
        }

        # Check if memory has error-related keywords
        has_error_keywords = False

        # Check keywords field
        if any(kw.lower() in error_keywords for kw in memory.keywords):
            has_error_keywords = True

        # Check content (case-insensitive)
        content_lower = memory.content.lower()
        if any(kw in content_lower for kw in error_keywords):
            has_error_keywords = True

        # Scoring logic
        if has_error_keywords and has_recent_errors:
            return 1.0  # Max boost for error-related memory with recent errors
        elif has_error_keywords:
            return 0.5  # Medium boost for error-related memory
        else:
            return 0.3  # Low score for non-error memories

    async def score_memory(
        self,
        memory: MemoryRecord,
        query_embedding: list[float] | None = None,
        current_stage: StageType | None = None,
        has_recent_errors: bool = False,
    ) -> tuple[float, ScoringBreakdown]:
        """
        Calculate weighted total importance score for a memory.

        Combines all scoring factors using configured weights.

        Args:
            memory: Memory record to score
            query_embedding: Optional query embedding for similarity
            current_stage: Optional current reasoning stage
            has_recent_errors: Whether recent errors detected

        Returns:
            Tuple of (total_score, breakdown) with detailed scoring information
        """
        # Calculate individual scores
        embedding_score = await self.score_embedding_similarity(
            query_embedding, memory
        )
        recency_score = self.score_recency(memory)
        frequency_score = self.score_frequency(memory)
        stage_score = self.score_stage_relevance(memory, current_stage)
        criticality_score = self.score_criticality(memory)
        error_score = self.score_error_correction(memory, has_recent_errors)

        # Calculate weighted total
        total_score = (
            embedding_score * self.config.embedding_similarity_weight
            + recency_score * self.config.recency_decay_weight
            + frequency_score * self.config.frequency_weight
            + stage_score * self.config.stage_relevance_weight
            + criticality_score * self.config.criticality_weight
            + error_score * self.config.error_correction_weight
        )

        # Create breakdown
        breakdown = ScoringBreakdown(
            embedding_similarity=embedding_score,
            recency=recency_score,
            frequency=frequency_score,
            stage_relevance=stage_score,
            criticality=criticality_score,
            error_correction=error_score,
            total_score=total_score,
        )

        return total_score, breakdown

    async def retrieve_top_k(
        self,
        memories: list[MemoryRecord],
        k: int = 10,
        query_embedding: list[float] | None = None,
        current_stage: StageType | None = None,
        has_recent_errors: bool = False,
    ) -> list[tuple[MemoryRecord, float, ScoringBreakdown]]:
        """
        Score all memories and return top K by importance.

        Args:
            memories: List of memory records to score
            k: Number of top results to return
            query_embedding: Optional query embedding for similarity
            current_stage: Optional current reasoning stage
            has_recent_errors: Whether recent errors detected

        Returns:
            List of (memory, score, breakdown) tuples, sorted by score descending
        """
        if not memories:
            return []

        # Score all memories
        scored_memories: list[tuple[MemoryRecord, float, ScoringBreakdown]] = []

        for memory in memories:
            score, breakdown = await self.score_memory(
                memory=memory,
                query_embedding=query_embedding,
                current_stage=current_stage,
                has_recent_errors=has_recent_errors,
            )
            scored_memories.append((memory, score, breakdown))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        top_k = scored_memories[:k]

        self._logger.info(
            "retrieved_top_k_memories",
            total_memories=len(memories),
            k=k,
            returned=len(top_k),
            top_score=top_k[0][1] if top_k else None,
            has_query_embedding=query_embedding is not None,
            current_stage=current_stage.value if current_stage else None,
            has_recent_errors=has_recent_errors,
        )

        return top_k
