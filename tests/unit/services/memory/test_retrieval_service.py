"""
Unit tests for Enhanced Retrieval Service with Multi-Factor Scoring.

Tests all scoring algorithms, weight configuration, and top-K retrieval.

Component ID: MEM-020
Ticket: MEM-020 (Implement Enhanced Retrieval Service)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord, StageType
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
    RetrievalConfig,
    ScoringBreakdown,
)


class TestRetrievalConfig:
    """Test RetrievalConfig weight validation."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = RetrievalConfig()

        total = (
            config.embedding_similarity_weight
            + config.recency_decay_weight
            + config.frequency_weight
            + config.stage_relevance_weight
            + config.criticality_weight
            + config.error_correction_weight
        )

        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_custom_weights_valid(self):
        """Test custom weights that sum to 1.0."""
        config = RetrievalConfig(
            embedding_similarity_weight=0.4,
            recency_decay_weight=0.2,
            frequency_weight=0.1,
            stage_relevance_weight=0.15,
            criticality_weight=0.1,
            error_correction_weight=0.05,
        )

        assert config.embedding_similarity_weight == 0.4

    def test_weights_validation_fails_when_not_sum_to_one(self):
        """Test that weights validation fails if sum != 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            RetrievalConfig(
                embedding_similarity_weight=0.5,
                recency_decay_weight=0.2,
                frequency_weight=0.1,
                stage_relevance_weight=0.1,
                criticality_weight=0.1,
                error_correction_weight=0.1,  # Total = 1.1 (invalid)
            )

    def test_weight_bounds_validation(self):
        """Test that weights must be in [0.0, 1.0]."""
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            RetrievalConfig(
                embedding_similarity_weight=-0.1,  # Negative (invalid)
                recency_decay_weight=0.3,
                frequency_weight=0.3,
                stage_relevance_weight=0.3,
                criticality_weight=0.15,
                error_correction_weight=0.05,
            )


class TestEmbeddingSimilarity:
    """Test embedding similarity scoring."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    @pytest.fixture
    def memory_with_embedding(self):
        # Create 768-dimensional unit vector along first axis
        embedding = [1.0] + [0.0] * 767
        return MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Test memory",
            summary="Test",
            embedding=embedding,
        )

    @pytest.mark.asyncio
    async def test_identical_embeddings_score_high(self, service, memory_with_embedding):
        """Test that identical embeddings score ~1.0."""
        query_embedding = [1.0] + [0.0] * 767  # Same as memory
        score = await service.score_embedding_similarity(
            query_embedding, memory_with_embedding
        )

        # Cosine similarity of identical vectors = 1.0
        # Normalized: (1.0 + 1.0) / 2.0 = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_orthogonal_embeddings_score_medium(
        self, service, memory_with_embedding
    ):
        """Test that orthogonal embeddings score ~0.5."""
        query_embedding = [0.0, 1.0] + [0.0] * 766  # Perpendicular
        score = await service.score_embedding_similarity(
            query_embedding, memory_with_embedding
        )

        # Cosine similarity of orthogonal vectors = 0.0
        # Normalized: (0.0 + 1.0) / 2.0 = 0.5
        assert score == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_opposite_embeddings_score_low(self, service, memory_with_embedding):
        """Test that opposite embeddings score ~0.0."""
        query_embedding = [-1.0] + [0.0] * 767  # Opposite direction
        score = await service.score_embedding_similarity(
            query_embedding, memory_with_embedding
        )

        # Cosine similarity of opposite vectors = -1.0
        # Normalized: (-1.0 + 1.0) / 2.0 = 0.0
        assert score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_none_query_embedding_returns_zero(
        self, service, memory_with_embedding
    ):
        """Test that None query embedding returns 0.0."""
        score = await service.score_embedding_similarity(None, memory_with_embedding)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_memory_embedding_returns_zero(self, service):
        """Test that empty memory embedding returns 0.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Test",
            summary="Test",
            embedding=[],  # Empty
        )
        score = await service.score_embedding_similarity([1.0, 0.0, 0.0], memory)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_dimension_mismatch_returns_zero(self, service):
        """Test that dimension mismatch returns 0.0."""
        # Create 768-dim memory but 1536-dim query (valid dims but mismatched)
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Test",
            summary="Test",
            embedding=[1.0] + [0.0] * 767,  # 768D
        )
        query_embedding = [1.0] + [0.0] * 1535  # 1536D
        score = await service.score_embedding_similarity(query_embedding, memory)
        assert score == 0.0


class TestRecencyScoring:
    """Test recency decay scoring."""

    @pytest.fixture
    def service(self):
        config = RetrievalConfig(recency_decay_lambda=0.1)
        return EnhancedRetrievalService(config)

    def test_recent_memory_scores_high(self, service):
        """Test that 1-day old memory scores ~0.90."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Recent memory",
            summary="Recent",
            timestamp=datetime.now(UTC) - timedelta(days=1),
        )

        score = service.score_recency(memory)

        # Expected: exp(-0.1 * 1) = exp(-0.1) ≈ 0.9048
        assert score == pytest.approx(0.9048, abs=0.01)

    def test_week_old_memory_scores_medium(self, service):
        """Test that 7-day old memory scores ~0.50."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Week old memory",
            summary="Week old",
            timestamp=datetime.now(UTC) - timedelta(days=7),
        )

        score = service.score_recency(memory)

        # Expected: exp(-0.1 * 7) = exp(-0.7) ≈ 0.4966
        assert score == pytest.approx(0.4966, abs=0.01)

    def test_month_old_memory_scores_low(self, service):
        """Test that 30-day old memory scores ~0.05."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Month old memory",
            summary="Month old",
            timestamp=datetime.now(UTC) - timedelta(days=30),
        )

        score = service.score_recency(memory)

        # Expected: exp(-0.1 * 30) = exp(-3.0) ≈ 0.0498
        assert score == pytest.approx(0.0498, abs=0.01)

    def test_current_memory_scores_maximum(self, service):
        """Test that current memory scores 1.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.WORKING,
            content="Current memory",
            summary="Current",
            timestamp=datetime.now(UTC),
        )

        score = service.score_recency(memory)

        # Expected: exp(-0.1 * 0) = exp(0) = 1.0
        assert score == pytest.approx(1.0, abs=0.01)


class TestFrequencyScoring:
    """Test frequency scoring based on access count."""

    @pytest.fixture
    def service(self):
        config = RetrievalConfig(max_access_count=100)
        return EnhancedRetrievalService(config)

    def test_zero_access_scores_zero(self, service):
        """Test that zero access count scores 0.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Never accessed",
            summary="Never",
            access_count=0,
        )

        score = service.score_frequency(memory)
        assert score == 0.0

    def test_medium_frequency_scores_proportional(self, service):
        """Test that 50 accesses out of 100 max scores 0.5."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Medium frequency",
            summary="Medium",
            access_count=50,
        )

        score = service.score_frequency(memory)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_high_frequency_scores_high(self, service):
        """Test that 100 accesses scores 1.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="High frequency",
            summary="High",
            access_count=100,
        )

        score = service.score_frequency(memory)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_frequency_capped_at_one(self, service):
        """Test that frequency above max is capped at 1.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Very high frequency",
            summary="Very high",
            access_count=200,  # Above max
        )

        score = service.score_frequency(memory)
        assert score == 1.0


class TestStageRelevanceScoring:
    """Test stage relevance scoring."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    def test_no_current_stage_returns_neutral(self, service):
        """Test that no current stage returns 0.3."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Test",
            summary="Test",
            stage_id="stage-123",
        )

        score = service.score_stage_relevance(memory, current_stage=None)
        assert score == 0.3

    def test_no_memory_stage_returns_neutral(self, service):
        """Test that memory without stage returns 0.3."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Test",
            summary="Test",
            stage_id=None,
        )

        score = service.score_stage_relevance(memory, StageType.PLANNING)
        assert score == 0.3

    def test_memory_with_stage_scores_high(self, service):
        """Test that memory with stage ID scores high (0.8)."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Test",
            summary="Test",
            stage_id="stage-123",
        )

        score = service.score_stage_relevance(memory, StageType.PLANNING)
        assert score == 0.8


class TestCriticalityScoring:
    """Test criticality boost scoring."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    def test_critical_memory_scores_maximum(self, service):
        """Test that critical memory scores 1.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Critical information",
            summary="Critical",
            is_critical=True,
        )

        score = service.score_criticality(memory)
        assert score == 1.0

    def test_non_critical_memory_scores_medium(self, service):
        """Test that non-critical memory scores 0.5."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Regular information",
            summary="Regular",
            is_critical=False,
        )

        score = service.score_criticality(memory)
        assert score == 0.5


class TestErrorCorrectionScoring:
    """Test error correction relevance scoring."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    def test_error_memory_with_recent_errors_scores_maximum(self, service):
        """Test error-related memory with recent errors scores 1.0."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Fixed authentication error by updating JWT token",
            summary="JWT fix",
            keywords=["error", "fix", "jwt"],
        )

        score = service.score_error_correction(memory, has_recent_errors=True)
        assert score == 1.0

    def test_error_memory_without_recent_errors_scores_medium(self, service):
        """Test error-related memory without recent errors scores 0.5."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Fixed authentication error",
            summary="Auth fix",
            keywords=["error", "fix"],
        )

        score = service.score_error_correction(memory, has_recent_errors=False)
        assert score == 0.5

    def test_non_error_memory_scores_low(self, service):
        """Test non-error memory scores 0.3."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Implemented new feature",
            summary="Feature",
            keywords=["feature", "new"],
        )

        score = service.score_error_correction(memory, has_recent_errors=False)
        assert score == 0.3

    def test_error_keyword_detection_in_content(self, service):
        """Test error keyword detection in content field."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Encountered a problem with database connection",
            summary="DB issue",
            keywords=[],  # No keywords, but content has "problem"
        )

        score = service.score_error_correction(memory, has_recent_errors=True)
        assert score == 1.0  # Should detect "problem" in content


class TestCombinedScoring:
    """Test combined scoring with all factors."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    @pytest.fixture
    def memory(self):
        embedding = [1.0] + [0.0] * 767  # 768D
        return MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="JWT authentication implementation",
            summary="JWT auth",
            embedding=embedding,
            timestamp=datetime.now(UTC) - timedelta(days=1),
            access_count=50,
            stage_id="stage-123",
            is_critical=True,
            keywords=["jwt", "auth"],
        )

    @pytest.mark.asyncio
    async def test_combined_score_calculation(self, service, memory):
        """Test combined score with all factors."""
        query_embedding = [1.0] + [0.0] * 767  # Identical to memory

        score, breakdown = await service.score_memory(
            memory=memory,
            query_embedding=query_embedding,
            current_stage=StageType.EXECUTION,
            has_recent_errors=False,
        )

        # Verify breakdown components
        assert breakdown.embedding_similarity == pytest.approx(1.0, abs=0.01)
        assert breakdown.recency == pytest.approx(0.9048, abs=0.01)
        assert breakdown.frequency == pytest.approx(0.5, abs=0.01)
        assert breakdown.stage_relevance == 0.8
        assert breakdown.criticality == 1.0
        assert breakdown.error_correction == 0.3

        # Verify weighted total
        expected_total = (
            1.0 * 0.35  # embedding
            + 0.9048 * 0.15  # recency
            + 0.5 * 0.10  # frequency
            + 0.8 * 0.20  # stage
            + 1.0 * 0.10  # criticality
            + 0.3 * 0.10  # error
        )

        assert score == pytest.approx(expected_total, abs=0.01)
        assert breakdown.total_score == pytest.approx(expected_total, abs=0.01)

    @pytest.mark.asyncio
    async def test_score_breakdown_structure(self, service, memory):
        """Test that score breakdown contains all fields."""
        score, breakdown = await service.score_memory(memory=memory)

        assert isinstance(breakdown, ScoringBreakdown)
        assert 0.0 <= breakdown.embedding_similarity <= 1.0
        assert 0.0 <= breakdown.recency <= 1.0
        assert 0.0 <= breakdown.frequency <= 1.0
        assert 0.0 <= breakdown.stage_relevance <= 1.0
        assert 0.0 <= breakdown.criticality <= 1.0
        assert 0.0 <= breakdown.error_correction <= 1.0
        assert 0.0 <= breakdown.total_score <= 1.0


class TestTopKRetrieval:
    """Test top-K retrieval functionality."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    @pytest.fixture
    def memories(self):
        """Create diverse memory set for ranking."""
        now = datetime.now(UTC)

        # Create embeddings
        emb1 = [1.0] + [0.0] * 767  # Unit vector along axis 0
        emb2 = [0.5, 0.5] + [0.0] * 766  # Mixed vector
        emb3 = [0.0, 1.0] + [0.0] * 766  # Unit vector along axis 1

        return [
            # High score: recent, critical, high similarity
            MemoryRecord(
                memory_id="mem-1",
                memory_layer=MemoryLayer.SEMANTIC,
                content="Critical authentication fix",
                summary="Auth fix",
                embedding=emb1,
                timestamp=now - timedelta(hours=1),
                access_count=80,
                is_critical=True,
                keywords=["auth", "critical"],
            ),
            # Medium score: older, non-critical
            MemoryRecord(
                memory_id="mem-2",
                memory_layer=MemoryLayer.EPISODIC,
                content="Regular task completion",
                summary="Task done",
                embedding=emb2,
                timestamp=now - timedelta(days=10),
                access_count=20,
                is_critical=False,
            ),
            # Low score: old, low frequency
            MemoryRecord(
                memory_id="mem-3",
                memory_layer=MemoryLayer.SEMANTIC,
                content="Old information",
                summary="Old",
                embedding=emb3,
                timestamp=now - timedelta(days=30),
                access_count=5,
                is_critical=False,
            ),
        ]

    @pytest.mark.asyncio
    async def test_retrieve_top_k_returns_sorted_results(self, service, memories):
        """Test that retrieve_top_k returns results sorted by score."""
        query_embedding = [1.0] + [0.0] * 767

        results = await service.retrieve_top_k(
            memories=memories, k=3, query_embedding=query_embedding
        )

        assert len(results) == 3

        # Verify sorted order (descending)
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)

        # Verify highest scoring memory is first
        top_memory, top_score, _ = results[0]
        assert top_memory.memory_id == "mem-1"  # Critical, recent, high similarity

    @pytest.mark.asyncio
    async def test_retrieve_top_k_respects_k_parameter(self, service, memories):
        """Test that retrieve_top_k respects k parameter."""
        results = await service.retrieve_top_k(memories=memories, k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_top_k_with_empty_list(self, service):
        """Test that retrieve_top_k handles empty list."""
        results = await service.retrieve_top_k(memories=[], k=10)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_top_k_includes_breakdown(self, service, memories):
        """Test that retrieve_top_k includes scoring breakdown."""
        results = await service.retrieve_top_k(memories=memories, k=1)

        assert len(results) == 1
        memory, score, breakdown = results[0]

        assert isinstance(breakdown, ScoringBreakdown)
        assert breakdown.total_score == pytest.approx(score, abs=0.001)

    @pytest.mark.asyncio
    async def test_retrieve_top_k_with_all_scoring_context(self, service, memories):
        """Test retrieve_top_k with all scoring parameters."""
        query_embedding = [1.0] + [0.0] * 767

        results = await service.retrieve_top_k(
            memories=memories,
            k=3,
            query_embedding=query_embedding,
            current_stage=StageType.REFLECTION,
            has_recent_errors=True,
        )

        assert len(results) == 3

        # All results should have breakdowns
        for _memory, score, breakdown in results:
            assert breakdown.total_score == pytest.approx(score, abs=0.001)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def service(self):
        return EnhancedRetrievalService()

    @pytest.mark.asyncio
    async def test_zero_weights_scenario(self):
        """Test service with all weights zero (invalid but should fail validation)."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            RetrievalConfig(
                embedding_similarity_weight=0.0,
                recency_decay_weight=0.0,
                frequency_weight=0.0,
                stage_relevance_weight=0.0,
                criticality_weight=0.0,
                error_correction_weight=0.0,
            )

    @pytest.mark.asyncio
    async def test_memory_with_missing_fields(self, service):
        """Test scoring memory with minimal fields."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Minimal memory",
            summary="Minimal",
            # No embedding, stage_id, etc.
        )

        score, breakdown = await service.score_memory(memory=memory)

        # Should not crash, should return valid score
        assert 0.0 <= score <= 1.0
        assert isinstance(breakdown, ScoringBreakdown)

    def test_recency_with_future_timestamp(self, service):
        """Test recency score with future timestamp (should handle gracefully)."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.WORKING,
            content="Future memory",
            summary="Future",
            timestamp=datetime.now(UTC) + timedelta(days=1),  # Future
        )

        score = service.score_recency(memory)

        # Future memories should score even higher (> 1.0)
        # But we clamp to [0, 1] range
        assert 0.0 <= score <= 1.0
