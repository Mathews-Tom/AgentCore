"""
Tests for Graph Repository

Comprehensive test suite for GraphRepository Neo4j operations
to achieve 90%+ coverage.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.database.graph_repository import GraphRepository
from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    RelationshipEdge,
    RelationshipType,
)


# ==================== GraphRepository Tests ====================


@pytest.mark.asyncio
async def test_graph_repository_create_node():
    """Test creating entity node in Neo4j graph."""
    mock_session = MagicMock()

    entity = EntityNode(
        entity_id=f"ent-{uuid4()}",
        entity_name="JWT Authentication",
        entity_type=EntityType.CONCEPT,
        properties={"domain": "security", "confidence": 0.95},
        embedding=[0.1] * 768,
        memory_refs=["mem-001", "mem-002"],
    )

    # Mock Neo4j result
    mock_node = {
        "entity_id": entity.entity_id,
        "entity_name": entity.entity_name,
        "entity_type": entity.entity_type.value,
        "properties": entity.properties,
    }
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: mock_node
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    node = await GraphRepository.create_node(mock_session, entity)

    assert node["entity_id"] == entity.entity_id
    assert node["entity_name"] == entity.entity_name
    mock_session.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_repository_create_node_failure():
    """Test creating node failure when no record returned."""
    mock_session = MagicMock()

    entity = EntityNode(
        entity_id=f"ent-{uuid4()}",
        entity_name="Test Entity",
        entity_type=EntityType.TOOL,
        properties={},
    )

    # Mock failure: no record returned
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=None)
    mock_session.run = AsyncMock(return_value=mock_result)

    with pytest.raises(RuntimeError, match="Failed to create entity node"):
        await GraphRepository.create_node(mock_session, entity)


@pytest.mark.asyncio
async def test_graph_repository_create_relationship():
    """Test creating relationship edge in Neo4j graph."""
    mock_session = MagicMock()

    relationship = RelationshipEdge(
        relationship_id=f"rel-{uuid4()}",
        source_entity_id="ent-001",
        target_entity_id="ent-002",
        relationship_type=RelationshipType.RELATES_TO,
        properties={"strength": 0.85},
        memory_refs=["mem-003"],
    )

    # Mock Neo4j result
    mock_rel = {
        "relationship_id": relationship.relationship_id,
        "properties": relationship.properties,
    }
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: mock_rel
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    rel = await GraphRepository.create_relationship(mock_session, relationship)

    assert rel["relationship_id"] == relationship.relationship_id
    mock_session.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_repository_create_relationship_failure():
    """Test creating relationship failure when no record returned."""
    mock_session = MagicMock()

    relationship = RelationshipEdge(
        relationship_id=f"rel-{uuid4()}",
        source_entity_id="ent-001",
        target_entity_id="ent-002",
        relationship_type=RelationshipType.MENTIONS,
        properties={},
    )

    # Mock failure
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=None)
    mock_session.run = AsyncMock(return_value=mock_result)

    with pytest.raises(RuntimeError, match="Failed to create relationship"):
        await GraphRepository.create_relationship(mock_session, relationship)


@pytest.mark.asyncio
async def test_graph_repository_get_node_by_id():
    """Test getting entity node by ID."""
    mock_session = MagicMock()
    entity_id = f"ent-{uuid4()}"

    # Mock Neo4j result
    mock_node = {
        "entity_id": entity_id,
        "entity_name": "Test Entity",
        "entity_type": "concept",
    }
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: mock_node
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    node = await GraphRepository.get_node_by_id(mock_session, entity_id)

    assert node is not None
    assert node["entity_id"] == entity_id


@pytest.mark.asyncio
async def test_graph_repository_get_node_by_id_not_found():
    """Test getting non-existent node returns None."""
    mock_session = MagicMock()

    # Mock no result
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=None)
    mock_session.run = AsyncMock(return_value=mock_result)

    node = await GraphRepository.get_node_by_id(mock_session, "nonexistent")

    assert node is None


@pytest.mark.asyncio
async def test_graph_repository_query_graph():
    """Test querying graph using traversal from starting entity."""
    mock_session = MagicMock()
    start_entity_id = f"ent-{uuid4()}"

    # Mock path data
    mock_node1 = {"entity_id": start_entity_id, "entity_name": "Start"}
    mock_node2 = {"entity_id": "ent-002", "entity_name": "Related"}
    mock_rel = {"relationship_id": "rel-001", "type": "RELATES_TO"}

    mock_path = MagicMock()
    mock_path.nodes = [MagicMock(__iter__=lambda self: iter(mock_node1.items())), MagicMock(__iter__=lambda self: iter(mock_node2.items()))]
    mock_path.relationships = [MagicMock(__iter__=lambda self: iter(mock_rel.items()))]

    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: mock_path

    # Create async iterator
    async def async_iter():
        yield mock_record

    mock_result = AsyncMock()
    mock_result.__aiter__ = lambda self: async_iter()
    mock_session.run = AsyncMock(return_value=mock_result)

    paths = await GraphRepository.query_graph(
        mock_session, start_entity_id, max_depth=2
    )

    assert len(paths) == 1
    assert "nodes" in paths[0]
    assert "relationships" in paths[0]


@pytest.mark.asyncio
async def test_graph_repository_query_graph_with_rel_types():
    """Test querying graph with relationship type filter."""
    mock_session = MagicMock()
    start_entity_id = f"ent-{uuid4()}"

    # Mock empty result
    async def async_iter():
        return
        yield  # Make it an async generator

    mock_result = AsyncMock()
    mock_result.__aiter__ = lambda self: async_iter()
    mock_session.run = AsyncMock(return_value=mock_result)

    paths = await GraphRepository.query_graph(
        mock_session,
        start_entity_id,
        max_depth=2,
        relationship_types=["RELATES_TO", "PART_OF"],
    )

    assert len(paths) == 0
    mock_session.run.assert_called_once()


@pytest.mark.asyncio
async def test_graph_repository_get_related_entities():
    """Test getting entities directly related to given entity."""
    mock_session = MagicMock()
    entity_id = f"ent-{uuid4()}"

    # Mock related entities
    mock_entity1 = {"entity_id": "ent-002", "entity_name": "Related 1"}
    mock_entity2 = {"entity_id": "ent-003", "entity_name": "Related 2"}

    mock_record1 = MagicMock()
    mock_record1.__getitem__ = lambda self, key: mock_entity1
    mock_record2 = MagicMock()
    mock_record2.__getitem__ = lambda self, key: mock_entity2

    # Create async iterator
    async def async_iter():
        yield mock_record1
        yield mock_record2

    mock_result = AsyncMock()
    mock_result.__aiter__ = lambda self: async_iter()
    mock_session.run = AsyncMock(return_value=mock_result)

    entities = await GraphRepository.get_related_entities(mock_session, entity_id)

    assert len(entities) == 2


@pytest.mark.asyncio
async def test_graph_repository_get_related_entities_with_type():
    """Test getting related entities with relationship type filter."""
    mock_session = MagicMock()
    entity_id = f"ent-{uuid4()}"

    # Mock empty result
    async def async_iter():
        return
        yield  # Make it an async generator

    mock_result = AsyncMock()
    mock_result.__aiter__ = lambda self: async_iter()
    mock_session.run = AsyncMock(return_value=mock_result)

    entities = await GraphRepository.get_related_entities(
        mock_session, entity_id, relationship_type="PART_OF"
    )

    assert len(entities) == 0


@pytest.mark.asyncio
async def test_graph_repository_update_node():
    """Test updating entity node properties."""
    mock_session = MagicMock()
    entity_id = f"ent-{uuid4()}"

    # Mock successful update
    mock_record = MagicMock()
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.update_node(
        mock_session, entity_id, {"entity_name": "Updated Name"}
    )

    assert success is True


@pytest.mark.asyncio
async def test_graph_repository_update_node_not_found():
    """Test updating non-existent node returns False."""
    mock_session = MagicMock()

    # Mock no result
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=None)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.update_node(
        mock_session, "nonexistent", {"entity_name": "Test"}
    )

    assert success is False


@pytest.mark.asyncio
async def test_graph_repository_update_relationship_access():
    """Test incrementing relationship access count."""
    mock_session = MagicMock()
    relationship_id = f"rel-{uuid4()}"

    # Mock successful update
    mock_record = MagicMock()
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.update_relationship_access(
        mock_session, relationship_id
    )

    assert success is True


@pytest.mark.asyncio
async def test_graph_repository_delete_node():
    """Test deleting entity node and all connected relationships."""
    mock_session = MagicMock()
    entity_id = f"ent-{uuid4()}"

    # Mock successful deletion
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 1  # deleted count
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.delete_node(mock_session, entity_id)

    assert success is True


@pytest.mark.asyncio
async def test_graph_repository_delete_node_not_found():
    """Test deleting non-existent node returns False."""
    mock_session = MagicMock()

    # Mock no deletion
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 0  # no nodes deleted
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.delete_node(mock_session, "nonexistent")

    assert success is False


@pytest.mark.asyncio
async def test_graph_repository_delete_relationship():
    """Test deleting relationship by ID."""
    mock_session = MagicMock()
    relationship_id = f"rel-{uuid4()}"

    # Mock successful deletion
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 1  # deleted count
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.delete_relationship(mock_session, relationship_id)

    assert success is True


@pytest.mark.asyncio
async def test_graph_repository_delete_relationship_not_found():
    """Test deleting non-existent relationship returns False."""
    mock_session = MagicMock()

    # Mock no deletion
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 0  # no relationships deleted
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    success = await GraphRepository.delete_relationship(mock_session, "nonexistent")

    assert success is False


@pytest.mark.asyncio
async def test_graph_repository_find_shortest_path():
    """Test finding shortest path between two entities."""
    mock_session = MagicMock()
    start_id = f"ent-{uuid4()}"
    end_id = f"ent-{uuid4()}"

    # Mock path
    mock_node1 = {"entity_id": start_id}
    mock_node2 = {"entity_id": end_id}
    mock_rel = {"relationship_id": "rel-001"}

    mock_path = MagicMock()
    mock_path.nodes = [MagicMock(__iter__=lambda self: iter(mock_node1.items())), MagicMock(__iter__=lambda self: iter(mock_node2.items()))]
    mock_path.relationships = [MagicMock(__iter__=lambda self: iter(mock_rel.items()))]

    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: mock_path
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    mock_session.run = AsyncMock(return_value=mock_result)

    path = await GraphRepository.find_shortest_path(mock_session, start_id, end_id)

    assert path is not None
    assert "nodes" in path
    assert "relationships" in path
    assert "length" in path


@pytest.mark.asyncio
async def test_graph_repository_find_shortest_path_not_found():
    """Test finding shortest path when no path exists."""
    mock_session = MagicMock()

    # Mock no path
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=None)
    mock_session.run = AsyncMock(return_value=mock_result)

    path = await GraphRepository.find_shortest_path(
        mock_session, "ent-001", "ent-999"
    )

    assert path is None


@pytest.mark.asyncio
async def test_graph_repository_search_entities_by_name():
    """Test searching entities by name pattern."""
    mock_session = MagicMock()

    # Mock search results
    mock_entity1 = {"entity_id": "ent-001", "entity_name": "JWT Authentication"}
    mock_entity2 = {"entity_id": "ent-002", "entity_name": "JWT Token"}

    mock_record1 = MagicMock()
    mock_record1.__getitem__ = lambda self, key: mock_entity1
    mock_record2 = MagicMock()
    mock_record2.__getitem__ = lambda self, key: mock_entity2

    # Create async iterator
    async def async_iter():
        yield mock_record1
        yield mock_record2

    mock_result = AsyncMock()
    mock_result.__aiter__ = lambda self: async_iter()
    mock_session.run = AsyncMock(return_value=mock_result)

    entities = await GraphRepository.search_entities_by_name(mock_session, "JWT")

    assert len(entities) == 2


@pytest.mark.asyncio
async def test_graph_repository_search_entities_by_name_no_results():
    """Test searching entities with no matching results."""
    mock_session = MagicMock()

    # Mock no results
    async def async_iter():
        return
        yield  # Make it an async generator

    mock_result = AsyncMock()
    mock_result.__aiter__ = lambda self: async_iter()
    mock_session.run = AsyncMock(return_value=mock_result)

    entities = await GraphRepository.search_entities_by_name(
        mock_session, "NonexistentPattern"
    )

    assert len(entities) == 0
