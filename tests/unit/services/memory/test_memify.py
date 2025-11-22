"""
Unit Tests for Memify Graph Optimizer

Tests for entity consolidation, relationship pruning, pattern detection,
and main orchestrator functionality.

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.services.memory.memify import (
    EntityConsolidation,
    PatternDetection,
    RelationshipPruning,
)
from agentcore.a2a_protocol.services.memory.memify_optimizer import (
    MemifyOptimizer,
)


class TestEntityConsolidation:
    """Test entity consolidation algorithm."""

    @pytest.fixture
    def consolidation(self) -> EntityConsolidation:
        """Create EntityConsolidation instance."""
        return EntityConsolidation(similarity_threshold=0.90)

    async def test_init_valid_threshold(self):
        """Test initialization with valid similarity threshold."""
        consolidation = EntityConsolidation(similarity_threshold=0.90)
        assert consolidation.similarity_threshold == 0.90

    async def test_init_invalid_threshold_high(self):
        """Test initialization with threshold > 1.0 raises error."""
        with pytest.raises(ValueError, match="Similarity threshold must be 0.0-1.0"):
            EntityConsolidation(similarity_threshold=1.5)

    async def test_init_invalid_threshold_low(self):
        """Test initialization with threshold < 0.0 raises error."""
        with pytest.raises(ValueError, match="Similarity threshold must be 0.0-1.0"):
            EntityConsolidation(similarity_threshold=-0.1)

    async def test_consolidate_all_duplicates(
        self, consolidation: EntityConsolidation
    ):
        """Test consolidating all duplicates in batch."""
        mock_session = AsyncMock()

        # Mock find_duplicate_entities
        with patch.object(
            consolidation,
            "find_duplicate_entities",
            return_value=[("ent-001", "ent-002", 0.95), ("ent-003", "ent-004", 0.92)],
        ):
            # Mock merge_entities
            with patch.object(
                consolidation,
                "merge_entities",
                return_value={
                    "relationships_merged": 5,
                    "memory_refs_merged": 3,
                    "primary_id": "ent-001",
                    "duplicate_id": "ent-002",
                },
            ):
                stats = await consolidation.consolidate_all_duplicates(mock_session)

                assert stats["pairs_found"] == 2
                assert stats["pairs_merged"] == 2
                assert stats["failed_merges"] == 0
                assert stats["total_relationships_merged"] == 10
                assert stats["total_memory_refs_merged"] == 6
                assert stats["accuracy"] == 1.0


class TestRelationshipPruning:
    """Test relationship pruning algorithm."""

    @pytest.fixture
    def pruning(self) -> RelationshipPruning:
        """Create RelationshipPruning instance."""
        return RelationshipPruning(min_access_count=2)

    async def test_init_valid_min_access(self):
        """Test initialization with valid min access count."""
        pruning = RelationshipPruning(min_access_count=2)
        assert pruning.min_access_count == 2

    async def test_init_invalid_min_access(self):
        """Test initialization with invalid min access count."""
        with pytest.raises(ValueError, match="Min access count must be >= 0"):
            RelationshipPruning(min_access_count=-1)

    async def test_prune_relationships_batch(self, pruning: RelationshipPruning):
        """Test batch relationship pruning."""
        mock_session = AsyncMock()

        mock_record = MagicMock()
        mock_record.__getitem__.return_value = 5

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run.return_value = mock_result

        deleted = await pruning.prune_relationships_batch(
            mock_session, ["rel-001", "rel-002", "rel-003", "rel-004", "rel-005"]
        )

        assert deleted == 5


class TestPatternDetection:
    """Test pattern detection algorithm."""

    @pytest.fixture
    def detection(self) -> PatternDetection:
        """Create PatternDetection instance."""
        return PatternDetection(min_frequency=5)

    async def test_init_valid_min_frequency(self):
        """Test initialization with valid min frequency."""
        detection = PatternDetection(min_frequency=5)
        assert detection.min_frequency == 5

    async def test_init_invalid_min_frequency(self):
        """Test initialization with invalid min frequency."""
        with pytest.raises(ValueError, match="Min frequency must be >= 1"):
            PatternDetection(min_frequency=0)


class TestMemifyOptimizer:
    """Test main Memify optimizer orchestrator."""

    @pytest.fixture
    def mock_driver(self) -> MagicMock:
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session_instance = AsyncMock()

        # Create context manager class that supports async with
        class AsyncContextManager:
            async def __aenter__(self):
                return session_instance

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        # Make session() return an instance of the context manager
        driver.session = MagicMock(return_value=AsyncContextManager())
        return driver

    @pytest.fixture
    def optimizer(self, mock_driver: AsyncMock) -> MemifyOptimizer:
        """Create MemifyOptimizer instance."""
        return MemifyOptimizer(
            driver=mock_driver,
            similarity_threshold=0.90,
            min_access_count=2,
            pattern_min_frequency=5,
        )

    async def test_init(self, optimizer: MemifyOptimizer):
        """Test optimizer initialization."""
        assert optimizer.similarity_threshold == 0.90
        assert optimizer.min_access_count == 2
        assert optimizer.pattern_min_frequency == 5
        assert isinstance(optimizer.consolidation, EntityConsolidation)
        assert isinstance(optimizer.pruning, RelationshipPruning)
        assert isinstance(optimizer.pattern_detection, PatternDetection)

    async def test_optimize_full_pipeline(
        self, optimizer: MemifyOptimizer, mock_driver: AsyncMock
    ):
        """Test full optimization pipeline."""
        # Mock all optimization steps
        with patch.object(
            optimizer.consolidation,
            "consolidate_all_duplicates",
            return_value={
                "pairs_found": 10,
                "pairs_merged": 9,
                "accuracy": 0.90,
            },
        ):
            with patch.object(
                optimizer.pruning,
                "prune_all_low_value",
                return_value={
                    "low_value_pruned": 50,
                    "redundant_pruned": 25,
                    "total_pruned": 75,
                },
            ):
                with patch.object(
                    optimizer.pattern_detection,
                    "detect_all_patterns",
                    return_value={
                        "frequent_paths": [],
                        "entity_clusters": [],
                        "index_suggestions": [],
                    },
                ):
                    with patch.object(
                        optimizer,
                        "calculate_quality_metrics",
                        return_value={
                            "total_entities": 1000,
                            "connectivity_score": 0.98,
                            "duplicate_percentage": 0.02,
                        },
                    ):
                        results = await optimizer.optimize()

                        assert "consolidation" in results
                        assert "pruning" in results
                        assert "patterns" in results
                        assert "quality_metrics" in results
                        assert "duration_seconds" in results
                        assert results["consolidation"]["pairs_merged"] == 9
                        assert results["pruning"]["total_pruned"] == 75

    async def test_validate_optimization_quality_all_pass(
        self, optimizer: MemifyOptimizer, mock_driver: AsyncMock
    ):
        """Test optimization quality validation (all checks pass)."""
        with patch.object(
            optimizer,
            "calculate_quality_metrics",
            return_value={
                "duplicate_percentage": 0.03,  # <5%
                "connectivity_score": 0.98,  # >95%
                "relationship_density": 3.5,  # >2.0
            },
        ):
            validation = await optimizer.validate_optimization_quality()

            assert validation["duplicate_percentage_ok"] is True
            assert validation["connectivity_ok"] is True
            assert validation["density_ok"] is True
            assert validation["all_checks_passed"] is True

    async def test_validate_optimization_quality_some_fail(
        self, optimizer: MemifyOptimizer, mock_driver: AsyncMock
    ):
        """Test optimization quality validation (some checks fail)."""
        with patch.object(
            optimizer,
            "calculate_quality_metrics",
            return_value={
                "duplicate_percentage": 0.08,  # >5% (FAIL)
                "connectivity_score": 0.98,  # >95% (PASS)
                "relationship_density": 1.5,  # <2.0 (FAIL)
            },
        ):
            validation = await optimizer.validate_optimization_quality()

            assert validation["duplicate_percentage_ok"] is False
            assert validation["connectivity_ok"] is True
            assert validation["density_ok"] is False
            assert validation["all_checks_passed"] is False

    async def test_get_optimization_recommendations(
        self, optimizer: MemifyOptimizer, mock_driver: AsyncMock
    ):
        """Test getting optimization recommendations."""
        with patch.object(
            optimizer,
            "calculate_quality_metrics",
            return_value={
                "duplicate_percentage": 0.08,  # High duplicates
                "connectivity_score": 0.92,  # Low connectivity
                "relationship_density": 1.5,  # Low density
                "avg_access_count": 0.5,  # Low access
                "isolated_nodes": 10,
            },
        ):
            recommendations = await optimizer.get_optimization_recommendations()

            # Should have recommendations for all issues
            assert len(recommendations) >= 5  # 4 issues + pattern detection
            types = [r["type"] for r in recommendations]
            assert "consolidation" in types
            assert "connectivity" in types
            assert "density" in types
            assert "usage" in types
            assert "patterns" in types


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
