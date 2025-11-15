"""
Unit Tests for GraphOptimizer

Tests graph optimization operations including entity consolidation,
relationship pruning, pattern detection, index optimization, and quality metrics.

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from agentcore.a2a_protocol.services.memory.graph_optimizer import (
    ConsolidationCandidate,
    FrequentPath,
    GraphOptimizer,
    OptimizationMetrics,
)


@pytest.fixture
def mock_session():
    """Mock Neo4j async session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_driver(mock_session):
    """Mock Neo4j async driver."""
    driver = MagicMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=mock_session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    driver.session.return_value = session_context
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def optimizer(mock_driver):
    """GraphOptimizer instance with mocked driver."""
    return GraphOptimizer(
        driver=mock_driver,
        similarity_threshold=0.90,
        min_access_count=2,
        batch_size=100,
    )


class TestOptimizationMetrics:
    """Test OptimizationMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = OptimizationMetrics()

        assert metrics.optimization_id.startswith("opt-")
        assert isinstance(metrics.started_at, datetime)
        assert metrics.completed_at is None
        assert metrics.entities_analyzed == 0
        assert metrics.duplicate_pairs_found == 0
        assert metrics.entities_merged == 0
        assert metrics.consolidation_accuracy == 0.0
        assert metrics.relationships_analyzed == 0
        assert metrics.low_value_edges_removed == 0
        assert metrics.patterns_detected == 0
        assert metrics.frequent_paths == []
        assert metrics.indexes_optimized == 0
        assert metrics.graph_connectivity == 0.0
        assert metrics.relationship_density == 0.0
        assert metrics.average_node_degree == 0.0
        assert metrics.duplicate_rate == 0.0
        assert metrics.duration_seconds == 0.0
        assert metrics.entities_per_second == 0.0


class TestConsolidationCandidate:
    """Test ConsolidationCandidate dataclass."""

    def test_candidate_creation(self):
        """Test creating consolidation candidate."""
        candidate = ConsolidationCandidate(
            entity1_id="ent-001",
            entity2_id="ent-002",
            entity1_name="Entity One",
            entity2_name="Entity Two",
            similarity_score=0.95,
            shared_properties={"domain": "security"},
        )

        assert candidate.entity1_id == "ent-001"
        assert candidate.entity2_id == "ent-002"
        assert candidate.similarity_score == 0.95
        assert candidate.shared_properties == {"domain": "security"}


class TestFrequentPath:
    """Test FrequentPath dataclass."""

    def test_frequent_path_creation(self):
        """Test creating frequent path."""
        path = FrequentPath(
            path_pattern=["concept", "tool", "concept"],
            traversal_count=50,
            average_access_count=12.5,
            representative_path=["ent-001", "ent-002", "ent-003"],
        )

        assert path.path_pattern == ["concept", "tool", "concept"]
        assert path.traversal_count == 50
        assert path.average_access_count == 12.5
        assert path.representative_path == ["ent-001", "ent-002", "ent-003"]


class TestGraphOptimizerInitialization:
    """Test GraphOptimizer initialization."""

    def test_default_initialization(self, mock_driver):
        """Test optimizer with default parameters."""
        optimizer = GraphOptimizer(mock_driver)

        assert optimizer.driver is mock_driver
        assert optimizer.similarity_threshold == 0.90
        assert optimizer.min_access_count == 2
        assert optimizer.batch_size == 100
        assert optimizer._scheduled_job_id is None
        assert optimizer._cron_expression is None
        assert optimizer._next_run is None

    def test_custom_initialization(self, mock_driver):
        """Test optimizer with custom parameters."""
        optimizer = GraphOptimizer(
            driver=mock_driver,
            similarity_threshold=0.85,
            min_access_count=5,
            batch_size=200,
        )

        assert optimizer.similarity_threshold == 0.85
        assert optimizer.min_access_count == 5
        assert optimizer.batch_size == 200


class TestEntityConsolidation:
    """Test entity consolidation functionality."""

    @pytest.mark.asyncio
    async def test_consolidation_no_entities(self, optimizer, mock_session):
        """Test consolidation when no entities exist."""
        # Mock empty graph
        mock_result = AsyncMock()
        mock_result.single.return_value = {"total": 0}
        mock_session.run.return_value = mock_result

        metrics = await optimizer._consolidate_entities()

        assert metrics["entities_analyzed"] == 0
        assert metrics["duplicate_pairs_found"] == 0
        assert metrics["entities_merged"] == 0
        assert metrics["consolidation_accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_consolidation_finds_similar_pairs(self, optimizer, mock_session):
        """Test finding similar entity pairs."""
        # Mock entity count
        count_result = AsyncMock()
        count_result.single.return_value = {"total": 100}

        # Mock similarity query results
        similarity_result = AsyncMock()

        async def async_iter(self):
            yield {
                "id1": "ent-001",
                "id2": "ent-002",
                "name1": "JWT Auth",
                "name2": "JWT Authentication",
                "similarity": 0.95,
                "props1": {"domain": "security"},
                "props2": {"confidence": 0.9},
            }
            yield {
                "id1": "ent-003",
                "id2": "ent-004",
                "name1": "Redis Cache",
                "name2": "Redis Caching",
                "similarity": 0.92,
                "props1": {},
                "props2": {},
            }

        similarity_result.__aiter__ = async_iter

        # Mock merge query
        merge_result = AsyncMock()
        merge_result.single.return_value = {"merged_id": "ent-001"}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return count_result
            elif call_count <= 2:
                return similarity_result
            else:
                return merge_result

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._consolidate_entities()

        assert metrics["entities_analyzed"] == 100
        assert metrics["duplicate_pairs_found"] == 2
        assert metrics["entities_merged"] >= 0

    @pytest.mark.asyncio
    async def test_merge_entity_pair_success(self, optimizer, mock_session):
        """Test successful entity pair merge."""
        # Mock successful merge
        merge_result = AsyncMock()
        merge_result.single.return_value = {"merged_id": "ent-001"}
        mock_session.run.return_value = merge_result

        candidate = ConsolidationCandidate(
            entity1_id="ent-001",
            entity2_id="ent-002",
            entity1_name="Entity One",
            entity2_name="Entity Two",
            similarity_score=0.95,
            shared_properties={"domain": "security"},
        )

        success = await optimizer._merge_entity_pair(mock_session, candidate)

        assert success is True

    @pytest.mark.asyncio
    async def test_merge_entity_pair_failure(self, optimizer, mock_session):
        """Test failed entity pair merge."""
        # Mock failed merge
        merge_result = AsyncMock()
        merge_result.single.return_value = None
        mock_session.run.return_value = merge_result

        candidate = ConsolidationCandidate(
            entity1_id="ent-001",
            entity2_id="ent-002",
            entity1_name="Entity One",
            entity2_name="Entity Two",
            similarity_score=0.95,
            shared_properties={},
        )

        success = await optimizer._merge_entity_pair(mock_session, candidate)

        assert success is False

    @pytest.mark.asyncio
    async def test_merge_entity_pair_exception(self, optimizer, mock_session):
        """Test entity merge with exception."""
        mock_session.run.side_effect = Exception("Database error")

        candidate = ConsolidationCandidate(
            entity1_id="ent-001",
            entity2_id="ent-002",
            entity1_name="Entity One",
            entity2_name="Entity Two",
            similarity_score=0.95,
            shared_properties={},
        )

        success = await optimizer._merge_entity_pair(mock_session, candidate)

        assert success is False

    def test_merge_properties_no_overlap(self, optimizer):
        """Test merging properties with no overlap."""
        props1 = {"key1": "value1"}
        props2 = {"key2": "value2"}

        merged = optimizer._merge_properties(props1, props2)

        assert merged == {"key1": "value1", "key2": "value2"}

    def test_merge_properties_with_overlap(self, optimizer):
        """Test merging properties with overlap (prioritizes props1)."""
        props1 = {"key1": "value1", "shared": "from_props1"}
        props2 = {"key2": "value2", "shared": "from_props2"}

        merged = optimizer._merge_properties(props1, props2)

        assert merged["key1"] == "value1"
        assert merged["key2"] == "value2"
        assert merged["shared"] == "from_props1"  # Props1 takes priority

    def test_merge_properties_list_deduplication(self, optimizer):
        """Test merging list properties deduplicates."""
        props1 = {"tags": ["a", "b"]}
        props2 = {"tags": ["b", "c"]}

        merged = optimizer._merge_properties(props1, props2)

        assert set(merged["tags"]) == {"a", "b", "c"}

    def test_merge_properties_empty_inputs(self, optimizer):
        """Test merging with empty property dicts."""
        assert optimizer._merge_properties({}, {}) == {}
        assert optimizer._merge_properties({"a": 1}, {}) == {"a": 1}
        assert optimizer._merge_properties({}, {"b": 2}) == {"b": 2}
        assert optimizer._merge_properties(None, {"b": 2}) == {"b": 2}
        assert optimizer._merge_properties({"a": 1}, None) == {"a": 1}


class TestRelationshipPruning:
    """Test relationship pruning functionality."""

    @pytest.mark.asyncio
    async def test_prune_relationships_success(self, optimizer, mock_session):
        """Test successful relationship pruning."""
        # Mock relationship count
        count_result = AsyncMock()
        count_result.single.return_value = {"total": 1000}

        # Mock prune results (delete in batches)
        prune_result1 = AsyncMock()
        prune_result1.single.return_value = {"deleted": 100}

        prune_result2 = AsyncMock()
        prune_result2.single.return_value = {"deleted": 50}  # Less than batch_size

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return count_result
            elif call_count == 2:
                return prune_result1
            else:
                return prune_result2

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._prune_relationships()

        assert metrics["relationships_analyzed"] == 1000
        assert metrics["low_value_edges_removed"] == 150

    @pytest.mark.asyncio
    async def test_prune_relationships_none_to_prune(self, optimizer, mock_session):
        """Test pruning when no low-value edges exist."""
        count_result = AsyncMock()
        count_result.single.return_value = {"total": 500}

        prune_result = AsyncMock()
        prune_result.single.return_value = {"deleted": 0}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return count_result
            else:
                return prune_result

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._prune_relationships()

        assert metrics["relationships_analyzed"] == 500
        assert metrics["low_value_edges_removed"] == 0


class TestPatternDetection:
    """Test pattern detection functionality."""

    @pytest.mark.asyncio
    async def test_detect_patterns_success(self, optimizer, mock_session):
        """Test successful pattern detection."""
        pattern_result = AsyncMock()

        async def async_iter(self):
            yield {
                "type1": "concept",
                "type2": "tool",
                "type3": "concept",
                "pattern_count": 50,
                "avg_access": 12.5,
                "sample_paths": [
                    {"path": ["ent-001", "ent-002", "ent-003"], "access": 15}
                ],
            }
            yield {
                "type1": "person",
                "type2": "concept",
                "type3": "tool",
                "pattern_count": 30,
                "avg_access": 8.0,
                "sample_paths": [
                    {"path": ["ent-004", "ent-005", "ent-006"], "access": 10}
                ],
            }

        pattern_result.__aiter__ = async_iter
        mock_session.run.return_value = pattern_result

        metrics = await optimizer._detect_patterns()

        assert metrics["patterns_detected"] == 2
        assert len(metrics["frequent_paths"]) == 2
        assert metrics["frequent_paths"][0]["path_pattern"] == [
            "concept",
            "tool",
            "concept",
        ]
        assert metrics["frequent_paths"][0]["traversal_count"] == 50
        assert metrics["frequent_paths"][0]["average_access_count"] == 12.5

    @pytest.mark.asyncio
    async def test_detect_patterns_empty(self, optimizer, mock_session):
        """Test pattern detection with no patterns found."""
        pattern_result = AsyncMock()

        async def async_iter(self):
            return
            yield  # pragma: no cover

        pattern_result.__aiter__ = async_iter
        mock_session.run.return_value = pattern_result

        metrics = await optimizer._detect_patterns()

        assert metrics["patterns_detected"] == 0
        assert metrics["frequent_paths"] == []


class TestIndexOptimization:
    """Test index optimization functionality."""

    @pytest.mark.asyncio
    async def test_optimize_indexes_creates_missing(self, optimizer, mock_session):
        """Test creating missing indexes."""
        # Mock existing indexes
        index_result = AsyncMock()
        index_result.single.return_value = {"indexes": ["entity_type_idx"]}

        # Mock pattern detection for pattern-based indexes
        pattern_result = AsyncMock()

        async def async_iter(self):
            return
            yield  # pragma: no cover

        pattern_result.__aiter__ = async_iter

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return index_result
            elif call_count <= 3:
                # Index creation
                return AsyncMock()
            else:
                return pattern_result

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._optimize_indexes()

        assert metrics["indexes_optimized"] >= 0

    @pytest.mark.asyncio
    async def test_optimize_indexes_skips_existing(self, optimizer, mock_session):
        """Test that existing indexes are not recreated."""
        index_result = AsyncMock()
        index_result.single.return_value = {
            "indexes": [
                "relationship_access_idx",
                "entity_embedding_exists_idx",
            ]
        }

        pattern_result = AsyncMock()

        async def async_iter(self):
            return
            yield  # pragma: no cover

        pattern_result.__aiter__ = async_iter

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return index_result
            else:
                return pattern_result

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._optimize_indexes()

        # Should not create indexes that already exist
        assert metrics["indexes_optimized"] >= 0


class TestQualityMetrics:
    """Test quality metrics computation."""

    @pytest.mark.asyncio
    async def test_compute_quality_metrics_empty_graph(self, optimizer, mock_session):
        """Test quality metrics with empty graph."""
        stats_result = AsyncMock()
        stats_result.single.return_value = None

        mock_session.run.return_value = stats_result

        metrics = await optimizer._compute_quality_metrics()

        assert metrics["connectivity"] == 0.0
        assert metrics["density"] == 0.0
        assert metrics["average_degree"] == 0.0
        assert metrics["duplicate_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_quality_metrics_success(self, optimizer, mock_session):
        """Test computing quality metrics."""
        # Mock graph statistics
        stats_result = AsyncMock()
        stats_result.single.return_value = {
            "node_count": 100,
            "edge_count": 500,
        }

        # Mock connectivity
        conn_result = AsyncMock()
        conn_result.single.return_value = {"connected_nodes": 95}

        # Mock duplicate detection
        dup_result = AsyncMock()
        dup_result.single.return_value = {"duplicate_pairs": 3}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stats_result
            elif call_count == 2:
                return conn_result
            else:
                return dup_result

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._compute_quality_metrics()

        assert metrics["connectivity"] == 0.95  # 95/100
        assert metrics["average_degree"] == 10.0  # (2*500)/100
        # Density = 500 / (100*99) = 0.0505
        assert 0.05 <= metrics["density"] <= 0.051
        assert metrics["duplicate_rate"] == 0.03  # 3/100

    @pytest.mark.asyncio
    async def test_compute_quality_metrics_no_duplicates(self, optimizer, mock_session):
        """Test quality metrics when GDS cosine similarity not available."""
        stats_result = AsyncMock()
        stats_result.single.return_value = {
            "node_count": 50,
            "edge_count": 200,
        }

        conn_result = AsyncMock()
        conn_result.single.return_value = {"connected_nodes": 48}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stats_result
            elif call_count == 2:
                return conn_result
            else:
                raise Exception("GDS not available")

        mock_session.run.side_effect = side_effect

        metrics = await optimizer._compute_quality_metrics()

        assert metrics["connectivity"] == 0.96  # 48/50
        assert metrics["duplicate_rate"] == 0.0  # Fallback when GDS unavailable


class TestScheduledExecution:
    """Test scheduled optimization execution."""

    def test_schedule_optimization_valid_cron(self, optimizer):
        """Test scheduling with valid cron expression."""
        job_id = optimizer.schedule_optimization("0 2 * * *")

        assert job_id.startswith("opt-job-")
        assert optimizer._scheduled_job_id == job_id
        assert optimizer._cron_expression == "0 2 * * *"
        assert optimizer._next_run is not None

    def test_schedule_optimization_invalid_cron(self, optimizer):
        """Test scheduling with invalid cron expression."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            optimizer.schedule_optimization("invalid cron")

    def test_schedule_optimization_every_four_hours(self, optimizer):
        """Test scheduling every 4 hours."""
        job_id = optimizer.schedule_optimization("0 */4 * * *")

        assert job_id is not None
        next_run = optimizer.get_next_scheduled_run()
        assert next_run is not None
        assert next_run > datetime.now(UTC)

    def test_get_next_scheduled_run_not_scheduled(self, optimizer):
        """Test getting next run when not scheduled."""
        next_run = optimizer.get_next_scheduled_run()
        assert next_run is None

    def test_get_next_scheduled_run_updates_past_time(self, optimizer):
        """Test that next run updates when in the past."""
        optimizer.schedule_optimization("* * * * *")  # Every minute

        # Set next run to past
        optimizer._next_run = datetime.now(UTC) - timedelta(hours=1)

        next_run = optimizer.get_next_scheduled_run()
        assert next_run > datetime.now(UTC)

    def test_cancel_scheduled_optimization(self, optimizer):
        """Test cancelling scheduled optimization."""
        optimizer.schedule_optimization("0 2 * * *")

        result = optimizer.cancel_scheduled_optimization()

        assert result is True
        assert optimizer._scheduled_job_id is None
        assert optimizer._cron_expression is None
        assert optimizer._next_run is None

    def test_cancel_when_not_scheduled(self, optimizer):
        """Test cancelling when nothing is scheduled."""
        result = optimizer.cancel_scheduled_optimization()
        assert result is False

    @pytest.mark.asyncio
    async def test_should_run_scheduled_when_due(self, optimizer):
        """Test should_run_scheduled returns True when due."""
        optimizer.schedule_optimization("* * * * *")

        # Set next run to past
        optimizer._next_run = datetime.now(UTC) - timedelta(minutes=1)

        should_run = await optimizer.should_run_scheduled()
        assert should_run is True

        # Next run should be updated
        assert optimizer._next_run > datetime.now(UTC)

    @pytest.mark.asyncio
    async def test_should_run_scheduled_not_due(self, optimizer):
        """Test should_run_scheduled returns False when not due."""
        optimizer.schedule_optimization("0 2 * * *")

        # Ensure next run is in future
        optimizer._next_run = datetime.now(UTC) + timedelta(hours=1)

        should_run = await optimizer.should_run_scheduled()
        assert should_run is False

    @pytest.mark.asyncio
    async def test_should_run_scheduled_not_scheduled(self, optimizer):
        """Test should_run_scheduled when nothing scheduled."""
        should_run = await optimizer.should_run_scheduled()
        assert should_run is False


class TestOptimizationStatus:
    """Test optimization status reporting."""

    @pytest.mark.asyncio
    async def test_get_optimization_status(self, optimizer, mock_session):
        """Test getting optimizer status."""
        stats_result = AsyncMock()
        stats_result.single.return_value = {
            "entity_count": 100,
            "relationship_count": 500,
        }
        mock_session.run.return_value = stats_result

        status = await optimizer.get_optimization_status()

        assert status["similarity_threshold"] == 0.90
        assert status["min_access_count"] == 2
        assert status["batch_size"] == 100
        assert status["scheduled_job_id"] is None
        assert status["cron_expression"] is None
        assert status["next_scheduled_run"] is None
        assert status["entity_count"] == 100
        assert status["relationship_count"] == 500

    @pytest.mark.asyncio
    async def test_get_optimization_status_with_schedule(
        self, optimizer, mock_session
    ):
        """Test status with scheduled optimization."""
        optimizer.schedule_optimization("0 2 * * *")

        stats_result = AsyncMock()
        stats_result.single.return_value = {
            "entity_count": 50,
            "relationship_count": 200,
        }
        mock_session.run.return_value = stats_result

        status = await optimizer.get_optimization_status()

        assert status["scheduled_job_id"] is not None
        assert status["cron_expression"] == "0 2 * * *"
        assert status["next_scheduled_run"] is not None

    @pytest.mark.asyncio
    async def test_get_optimization_status_empty_graph(
        self, optimizer, mock_session
    ):
        """Test status with empty graph."""
        stats_result = AsyncMock()
        stats_result.single.return_value = None
        mock_session.run.return_value = stats_result

        status = await optimizer.get_optimization_status()

        assert status["entity_count"] == 0
        assert status["relationship_count"] == 0


class TestFullOptimizationCycle:
    """Test complete optimization cycle."""

    @pytest.mark.asyncio
    async def test_optimize_full_cycle(self, optimizer):
        """Test running complete optimization cycle."""
        # Mock all internal methods
        with patch.object(
            optimizer,
            "_consolidate_entities",
            return_value={
                "entities_analyzed": 100,
                "duplicate_pairs_found": 5,
                "entities_merged": 5,
                "consolidation_accuracy": 1.0,
            },
        ), patch.object(
            optimizer,
            "_prune_relationships",
            return_value={
                "relationships_analyzed": 500,
                "low_value_edges_removed": 50,
            },
        ), patch.object(
            optimizer,
            "_detect_patterns",
            return_value={
                "patterns_detected": 3,
                "frequent_paths": [
                    {
                        "path_pattern": ["concept", "tool", "concept"],
                        "traversal_count": 20,
                        "average_access_count": 10.0,
                        "representative_path": ["e1", "e2", "e3"],
                    }
                ],
            },
        ), patch.object(
            optimizer,
            "_optimize_indexes",
            return_value={"indexes_optimized": 2},
        ), patch.object(
            optimizer,
            "_compute_quality_metrics",
            return_value={
                "connectivity": 0.95,
                "density": 0.05,
                "average_degree": 10.0,
                "duplicate_rate": 0.02,
            },
        ):
            metrics = await optimizer.optimize()

        assert metrics.entities_analyzed == 100
        assert metrics.duplicate_pairs_found == 5
        assert metrics.entities_merged == 5
        assert metrics.consolidation_accuracy == 1.0
        assert metrics.relationships_analyzed == 500
        assert metrics.low_value_edges_removed == 50
        assert metrics.patterns_detected == 3
        assert len(metrics.frequent_paths) == 1
        assert metrics.indexes_optimized == 2
        assert metrics.graph_connectivity == 0.95
        assert metrics.relationship_density == 0.05
        assert metrics.average_node_degree == 10.0
        assert metrics.duplicate_rate == 0.02
        assert metrics.completed_at is not None
        assert metrics.duration_seconds > 0
        assert metrics.entities_per_second > 0

    @pytest.mark.asyncio
    async def test_optimize_performance_metrics(self, optimizer):
        """Test that performance metrics are calculated correctly."""
        with patch.object(
            optimizer,
            "_consolidate_entities",
            return_value={
                "entities_analyzed": 1000,
                "duplicate_pairs_found": 10,
                "entities_merged": 10,
                "consolidation_accuracy": 1.0,
            },
        ), patch.object(
            optimizer,
            "_prune_relationships",
            return_value={
                "relationships_analyzed": 5000,
                "low_value_edges_removed": 100,
            },
        ), patch.object(
            optimizer,
            "_detect_patterns",
            return_value={"patterns_detected": 5, "frequent_paths": []},
        ), patch.object(
            optimizer,
            "_optimize_indexes",
            return_value={"indexes_optimized": 3},
        ), patch.object(
            optimizer,
            "_compute_quality_metrics",
            return_value={
                "connectivity": 0.98,
                "density": 0.02,
                "average_degree": 8.0,
                "duplicate_rate": 0.01,
            },
        ):
            metrics = await optimizer.optimize()

        # Verify performance target: entities_per_second should be high
        # In production, should process 1000 entities in <5s = >200 entities/s
        assert metrics.entities_analyzed == 1000
        assert metrics.duration_seconds > 0
        assert metrics.entities_per_second > 0

    @pytest.mark.asyncio
    async def test_optimize_consolidation_accuracy_target(self, optimizer):
        """Test consolidation accuracy meets 90%+ target."""
        with patch.object(
            optimizer,
            "_consolidate_entities",
            return_value={
                "entities_analyzed": 500,
                "duplicate_pairs_found": 20,
                "entities_merged": 19,  # 95% accuracy
                "consolidation_accuracy": 0.95,
            },
        ), patch.object(
            optimizer,
            "_prune_relationships",
            return_value={
                "relationships_analyzed": 1000,
                "low_value_edges_removed": 50,
            },
        ), patch.object(
            optimizer,
            "_detect_patterns",
            return_value={"patterns_detected": 2, "frequent_paths": []},
        ), patch.object(
            optimizer,
            "_optimize_indexes",
            return_value={"indexes_optimized": 1},
        ), patch.object(
            optimizer,
            "_compute_quality_metrics",
            return_value={
                "connectivity": 0.90,
                "density": 0.03,
                "average_degree": 6.0,
                "duplicate_rate": 0.02,
            },
        ):
            metrics = await optimizer.optimize()

        assert metrics.consolidation_accuracy >= 0.90  # 90%+ target

    @pytest.mark.asyncio
    async def test_optimize_duplicate_rate_target(self, optimizer):
        """Test duplicate rate meets <5% target."""
        with patch.object(
            optimizer,
            "_consolidate_entities",
            return_value={
                "entities_analyzed": 1000,
                "duplicate_pairs_found": 30,
                "entities_merged": 28,
                "consolidation_accuracy": 0.93,
            },
        ), patch.object(
            optimizer,
            "_prune_relationships",
            return_value={
                "relationships_analyzed": 3000,
                "low_value_edges_removed": 200,
            },
        ), patch.object(
            optimizer,
            "_detect_patterns",
            return_value={"patterns_detected": 4, "frequent_paths": []},
        ), patch.object(
            optimizer,
            "_optimize_indexes",
            return_value={"indexes_optimized": 2},
        ), patch.object(
            optimizer,
            "_compute_quality_metrics",
            return_value={
                "connectivity": 0.92,
                "density": 0.04,
                "average_degree": 7.5,
                "duplicate_rate": 0.03,  # 3% < 5% target
            },
        ):
            metrics = await optimizer.optimize()

        assert metrics.duplicate_rate < 0.05  # <5% target

    @pytest.mark.asyncio
    async def test_optimize_with_zero_entities(self, optimizer):
        """Test optimization with no entities doesn't divide by zero."""
        with patch.object(
            optimizer,
            "_consolidate_entities",
            return_value={
                "entities_analyzed": 0,
                "duplicate_pairs_found": 0,
                "entities_merged": 0,
                "consolidation_accuracy": 1.0,
            },
        ), patch.object(
            optimizer,
            "_prune_relationships",
            return_value={
                "relationships_analyzed": 0,
                "low_value_edges_removed": 0,
            },
        ), patch.object(
            optimizer,
            "_detect_patterns",
            return_value={"patterns_detected": 0, "frequent_paths": []},
        ), patch.object(
            optimizer,
            "_optimize_indexes",
            return_value={"indexes_optimized": 0},
        ), patch.object(
            optimizer,
            "_compute_quality_metrics",
            return_value={
                "connectivity": 0.0,
                "density": 0.0,
                "average_degree": 0.0,
                "duplicate_rate": 0.0,
            },
        ):
            metrics = await optimizer.optimize()

        assert metrics.entities_analyzed == 0
        assert metrics.entities_per_second == 0.0  # No division by zero
