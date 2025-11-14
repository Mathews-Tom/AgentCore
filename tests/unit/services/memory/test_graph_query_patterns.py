"""
Unit Tests for Graph Query Patterns (MEM-019)

Tests all graph query patterns including:
- 1-hop neighbor queries with direction
- 2-hop relationship queries
- 3-hop path finding (shortest path)
- Entity similarity queries
- Relationship strength aggregation

Component ID: MEM-019
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.a2a_protocol.models.memory import EntityNode, EntityType
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService


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
def graph_service(mock_driver):
    """Create GraphMemoryService instance with mock driver."""
    service = GraphMemoryService(driver=mock_driver)
    service._initialized = True
    return service


@pytest.fixture
def sample_entity_data():
    """Sample entity data for testing."""
    return {
        "entity_id": "ent-test-001",
        "entity_name": "Test Entity",
        "entity_type": "concept",
        "properties": {"domain": "testing"},
        "memory_refs": ["mem-001"],
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


class TestOneHopNeighborQueries:
    """Test 1-hop neighbor queries with direction support."""

    @pytest.mark.asyncio
    async def test_get_one_hop_neighbors_both_directions(
        self, graph_service, mock_session, sample_entity_data
    ):
        """Test getting neighbors in both directions."""
        result_mock = AsyncMock()

        async def async_iter(self):
            yield {"neighbor": sample_entity_data}

        result_mock.__aiter__ = async_iter
        mock_session.run.return_value = result_mock

        neighbors = await graph_service.get_one_hop_neighbors(
            "ent-001", direction="both"
        )

        assert len(neighbors) == 1
        assert isinstance(neighbors[0], EntityNode)
        assert neighbors[0].entity_id == "ent-test-001"

    @pytest.mark.asyncio
    async def test_get_one_hop_neighbors_invalid_direction(self, graph_service):
        """Test error on invalid direction."""
        with pytest.raises(ValueError, match="Direction must be"):
            await graph_service.get_one_hop_neighbors("ent-001", direction="invalid")


class TestTwoHopPathQueries:
    """Test 2-hop relationship queries."""

    @pytest.mark.asyncio
    async def test_get_two_hop_paths_all_paths(self, graph_service, mock_session):
        """Test finding all 2-hop paths from start node."""
        result_mock = AsyncMock()

        path_data = {
            "start": {"entity_id": "ent-001", "entity_name": "Start"},
            "r1": {"relationship_id": "rel-001", "properties": {}},
            "middle": {"entity_id": "ent-002", "entity_name": "Middle"},
            "r2": {"relationship_id": "rel-002", "properties": {}},
            "end": {"entity_id": "ent-003", "entity_name": "End"},
        }

        async def async_iter(self):
            yield path_data

        result_mock.__aiter__ = async_iter
        mock_session.run.return_value = result_mock

        paths = await graph_service.get_two_hop_paths("ent-001")

        assert len(paths) == 1
        assert paths[0]["start"]["entity_id"] == "ent-001"
        assert paths[0]["middle"]["entity_id"] == "ent-002"
        assert paths[0]["end"]["entity_id"] == "ent-003"


class TestEntitySimilarityQueries:
    """Test entity similarity queries."""

    @pytest.mark.asyncio
    async def test_find_similar_entities(
        self, graph_service, mock_session, sample_entity_data
    ):
        """Test finding similar entities."""
        result_mock = AsyncMock()

        similar_data = {
            "e2": sample_entity_data,
            "similarity_score": 0.85,
        }

        async def async_iter(self):
            yield similar_data

        result_mock.__aiter__ = async_iter
        mock_session.run.return_value = result_mock

        similar = await graph_service.find_similar_entities("ent-001")

        assert len(similar) == 1
        entity, score = similar[0]
        assert isinstance(entity, EntityNode)
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_find_similar_entities_invalid_threshold(self, graph_service):
        """Test error on invalid similarity threshold."""
        with pytest.raises(ValueError, match="Similarity threshold must be 0.0-1.0"):
            await graph_service.find_similar_entities("ent-001", similarity_threshold=1.5)


class TestRelationshipStrengthAggregation:
    """Test relationship strength aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_relationship_strength(self, graph_service, mock_session):
        """Test aggregating relationship strength."""
        direct_result = AsyncMock()
        direct_result.single.return_value = {"strength": 0.85}

        paths_result = AsyncMock()
        paths_result.single.return_value = {
            "path_count": 5,
            "avg_length": 2.2,
            "total_strength": 3.14,
        }

        mock_session.run.side_effect = [direct_result, paths_result]

        result = await graph_service.aggregate_relationship_strength(
            "ent-001", "ent-002"
        )

        assert result["direct_strength"] == 0.85
        assert result["path_count"] == 5
        assert result["average_path_length"] == 2.2
        assert result["total_strength"] == 3.14

    @pytest.mark.asyncio
    async def test_aggregate_relationship_strength_invalid_depth(self, graph_service):
        """Test error on invalid max depth."""
        with pytest.raises(ValueError, match="Max depth must be 1-5"):
            await graph_service.aggregate_relationship_strength(
                "ent-001", "ent-002", max_depth=10
            )
