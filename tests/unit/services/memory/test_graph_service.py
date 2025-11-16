"""
Unit Tests for GraphMemoryService

Tests graph memory operations with mocked Neo4j driver.
Validates node creation, relationship management, graph traversal, and indexing.

Component ID: MEM-017
Ticket: MEM-017 (Implement GraphMemoryService - Neo4j Integration)
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    MemoryLayer,
    MemoryRecord,
    RelationshipType,
)
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService


@pytest.fixture
def mock_session():
    """Mock Neo4j async session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_driver(mock_session):
    """Mock Neo4j async driver."""
    driver = MagicMock()  # Use MagicMock instead of AsyncMock for driver
    # Create a proper async context manager mock
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=mock_session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    driver.session.return_value = session_context
    driver.close = AsyncMock()  # close() needs to be async
    return driver


@pytest.fixture
def graph_service(mock_driver):
    """GraphMemoryService instance with mocked driver."""
    service = GraphMemoryService(mock_driver)
    return service


@pytest.fixture
def sample_memory():
    """Sample memory record for testing."""
    return MemoryRecord(
        memory_id=f"mem-{uuid4()}",
        memory_layer=MemoryLayer.SEMANTIC,
        content="User prefers detailed technical explanations",
        summary="User preference: technical detail",
        embedding=[0.1] * 768,  # Valid 768-dim embedding
        agent_id="agent-123",
        session_id="session-456",
        task_id="task-789",
        timestamp=datetime.now(UTC),
        is_critical=True,
        relevance_score=0.9,
    )


@pytest.fixture
def sample_entity():
    """Sample entity node for testing."""
    return EntityNode(
        entity_id=f"ent-{uuid4()}",
        entity_name="JWT Authentication",
        entity_type=EntityType.CONCEPT,
        properties={"domain": "security", "confidence": 0.95},
        memory_refs=["mem-001", "mem-002"],
    )


class TestGraphMemoryServiceInitialization:
    """Test GraphMemoryService initialization and setup."""

    @pytest.mark.asyncio
    async def test_initialize_creates_indexes(self, graph_service, mock_session):
        """Test that initialize creates all required indexes."""
        # Mock successful index creation
        mock_session.run.return_value = AsyncMock()

        await graph_service.initialize()

        # Verify index creation calls
        assert mock_session.run.call_count >= 6  # At least 6 indexes
        assert graph_service._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_creates_entity_id_constraint(
        self, graph_service, mock_session
    ):
        """Test that unique constraint is created for entity_id."""
        mock_session.run.return_value = AsyncMock()

        await graph_service.initialize()

        # Check that constraint query was called
        calls = [call[0][0] for call in mock_session.run.call_args_list]
        constraint_calls = [
            call for call in calls if "CONSTRAINT entity_id_unique" in call
        ]
        assert len(constraint_calls) > 0


class TestMemoryNodeOperations:
    """Test memory node storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_memory_node_success(
        self, graph_service, mock_session, sample_memory
    ):
        """Test successful memory node creation."""
        # Mock query result
        mock_result = AsyncMock()
        mock_record = {"memory_id": sample_memory.memory_id}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        memory_id = await graph_service.store_memory_node(sample_memory)

        assert memory_id == sample_memory.memory_id
        mock_session.run.assert_called_once()

        # Verify query parameters
        call_args = mock_session.run.call_args
        params = call_args[0][1]
        assert params["memory_id"] == sample_memory.memory_id
        assert params["memory_layer"] == sample_memory.memory_layer.value
        assert params["is_critical"] == sample_memory.is_critical

    @pytest.mark.asyncio
    async def test_store_memory_node_failure_raises_error(
        self, graph_service, mock_session, sample_memory
    ):
        """Test that store_memory_node raises error on failure."""
        # Mock failed query (no record returned)
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        with pytest.raises(RuntimeError, match="Failed to store memory node"):
            await graph_service.store_memory_node(sample_memory)


class TestEntityNodeOperations:
    """Test entity node storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_entity_node_success(
        self, graph_service, mock_session, sample_entity
    ):
        """Test successful entity node creation."""
        # Mock GraphRepository.create_node
        with patch(
            "agentcore.a2a_protocol.services.memory.graph_service.GraphRepository.create_node"
        ) as mock_create:
            mock_create.return_value = {"entity_id": sample_entity.entity_id}

            entity_id = await graph_service.store_entity_node(sample_entity)

            assert entity_id == sample_entity.entity_id
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_concept_node_success(self, graph_service, mock_session):
        """Test successful concept node creation."""
        # Mock query result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"concept_id": "concept-123"}
        mock_session.run.return_value = mock_result

        concept_id = await graph_service.store_concept_node(
            name="authentication",
            properties={"domain": "security"},
            memory_refs=["mem-001"],
        )

        assert concept_id.startswith("concept-")
        mock_session.run.assert_called_once()

        # Verify query parameters
        call_args = mock_session.run.call_args
        params = call_args[0][1]
        assert params["name"] == "authentication"
        # Properties are serialized as JSON string
        import json
        assert params["properties_json"] == json.dumps({"domain": "security"})
        assert params["memory_refs"] == ["mem-001"]


class TestRelationshipOperations:
    """Test relationship creation and management."""

    @pytest.mark.asyncio
    async def test_create_mention_relationship_success(
        self, graph_service, mock_session
    ):
        """Test creating MENTIONS relationship."""
        # Mock query result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"relationship_id": "rel-123"}
        mock_session.run.return_value = mock_result

        rel_id = await graph_service.create_mention_relationship(
            memory_id="mem-001",
            entity_id="ent-002",
            properties={"confidence": 0.95},
        )

        assert rel_id.startswith("rel-")
        mock_session.run.assert_called_once()

        # Verify query contains MENTIONS
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "MENTIONS" in query

    @pytest.mark.asyncio
    async def test_create_relationship_relates_to(self, graph_service, mock_session):
        """Test creating RELATES_TO relationship."""
        # Mock query result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"relationship_id": "rel-456"}
        mock_session.run.return_value = mock_result

        rel_id = await graph_service.create_relationship(
            from_id="ent-001",
            to_id="ent-002",
            rel_type=RelationshipType.RELATES_TO,
        )

        assert rel_id.startswith("rel-")

        # Verify query contains RELATES_TO
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "RELATES_TO" in query

    @pytest.mark.asyncio
    async def test_create_relationship_follows_temporal(
        self, graph_service, mock_session
    ):
        """Test creating FOLLOWS temporal relationship."""
        # Mock query result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"relationship_id": "rel-789"}
        mock_session.run.return_value = mock_result

        rel_id = await graph_service.create_relationship(
            from_id="mem-001",
            to_id="mem-002",
            rel_type=RelationshipType.FOLLOWS,
            from_label="Memory",
            to_label="Memory",
        )

        assert rel_id.startswith("rel-")

        # Verify query uses Memory labels
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "Memory" in query
        assert "FOLLOWS" in query

    @pytest.mark.asyncio
    async def test_create_relationship_part_of(self, graph_service, mock_session):
        """Test creating PART_OF hierarchical relationship."""
        # Mock query result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"relationship_id": "rel-101"}
        mock_session.run.return_value = mock_result

        rel_id = await graph_service.create_relationship(
            from_id="ent-child",
            to_id="ent-parent",
            rel_type=RelationshipType.PART_OF,
        )

        assert rel_id.startswith("rel-")

        # Verify query contains PART_OF
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "PART_OF" in query


class TestGraphTraversal:
    """Test graph traversal and path finding."""

    @pytest.mark.asyncio
    async def test_traverse_graph_with_depth_limit(self, graph_service, mock_session):
        """Test graph traversal with max depth."""
        # Mock GraphRepository.query_graph
        mock_paths = [
            {
                "nodes": [
                    {"entity_id": "ent-001", "entity_name": "Entity 1"},
                    {"entity_id": "ent-002", "entity_name": "Entity 2"},
                ],
                "relationships": [
                    {"relationship_id": "rel-001", "relationship_type": "RELATES_TO"}
                ],
            }
        ]

        with patch(
            "agentcore.a2a_protocol.services.memory.graph_service.GraphRepository.query_graph"
        ) as mock_query:
            mock_query.return_value = mock_paths

            paths = await graph_service.traverse_graph(
                start_id="ent-001",
                max_depth=2,
                relationship_types=[RelationshipType.RELATES_TO],
            )

            assert len(paths) == 1
            assert len(paths[0]["nodes"]) == 2
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_traverse_graph_with_high_depth(
        self, graph_service, mock_session
    ):
        """Test graph traversal with depth > 3 (logs warning but still works)."""
        with patch(
            "agentcore.a2a_protocol.services.memory.graph_service.GraphRepository.query_graph"
        ) as mock_query:
            mock_query.return_value = []

            # Should still work even with depth > 3, just logs a warning
            paths = await graph_service.traverse_graph(start_id="ent-001", max_depth=5)

            assert paths == []
            # Verify query was called with max_depth=5
            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["max_depth"] == 5

    @pytest.mark.asyncio
    async def test_find_related_entities(self, graph_service, mock_session):
        """Test finding related entities."""
        # Mock GraphRepository.get_related_entities
        mock_entities = [
            {
                "entity_id": "ent-002",
                "entity_name": "Entity 2",
                "entity_type": "concept",
                "properties": {},
                "memory_refs": [],
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
        ]

        with patch(
            "agentcore.a2a_protocol.services.memory.graph_service.GraphRepository.get_related_entities"
        ) as mock_get:
            mock_get.return_value = mock_entities

            entities = await graph_service.find_related_entities(
                entity_id="ent-001",
                rel_type=RelationshipType.RELATES_TO,
            )

            assert len(entities) == 1
            assert isinstance(entities[0], EntityNode)
            assert entities[0].entity_id == "ent-002"


class TestTemporalOperations:
    """Test temporal relationship queries."""

    @pytest.mark.asyncio
    async def test_get_temporal_sequence(self, graph_service, mock_session):
        """Test retrieving temporal sequence of memories."""
        # Mock query result
        mock_memories = [
            {
                "memory_id": "mem-001",
                "timestamp": datetime.now(UTC),
                "content": "First memory",
            },
            {
                "memory_id": "mem-002",
                "timestamp": datetime.now(UTC),
                "content": "Second memory",
            },
        ]

        # Mock async iteration
        class MockResult:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, '_index'):
                    self._index = 0
                if self._index >= len(mock_memories):
                    raise StopAsyncIteration
                result = {"mem": mock_memories[self._index]}
                self._index += 1
                return result

        mock_session.run.return_value = MockResult()

        sequence = await graph_service.get_temporal_sequence(task_id="task-123")

        assert len(sequence) == 2
        assert sequence[0]["memory_id"] == "mem-001"

        # Verify query parameters
        call_args = mock_session.run.call_args
        params = call_args[0][1]
        assert params["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_find_memories_by_entity(self, graph_service, mock_session):
        """Test finding memories that mention an entity."""
        # Mock query result
        mock_memories = [
            {
                "memory_id": "mem-001",
                "content": "Memory mentioning entity",
            }
        ]

        # Mock async iteration
        class MockResult:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, '_index'):
                    self._index = 0
                if self._index >= len(mock_memories):
                    raise StopAsyncIteration
                result = {"m": mock_memories[self._index]}
                self._index += 1
                return result

        mock_session.run.return_value = MockResult()

        memories = await graph_service.find_memories_by_entity(entity_id="ent-123")

        assert len(memories) == 1
        assert memories[0]["memory_id"] == "mem-001"


class TestUtilityOperations:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_update_relationship_access(self, graph_service, mock_session):
        """Test incrementing relationship access count."""
        with patch(
            "agentcore.a2a_protocol.services.memory.graph_service.GraphRepository.update_relationship_access"
        ) as mock_update:
            mock_update.return_value = True

            success = await graph_service.update_relationship_access("rel-123")

            assert success is True
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_node_degree(self, graph_service, mock_session):
        """Test getting node connection degree."""
        # Mock query result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"degree": 5}
        mock_session.run.return_value = mock_result

        degree = await graph_service.get_node_degree("ent-001")

        assert degree == 5

    @pytest.mark.asyncio
    async def test_find_shortest_path(self, graph_service, mock_session):
        """Test finding shortest path between nodes."""
        mock_path = {
            "nodes": [
                {"entity_id": "ent-001"},
                {"entity_id": "ent-002"},
            ],
            "relationships": [{"relationship_id": "rel-001"}],
            "length": 1,
        }

        with patch(
            "agentcore.a2a_protocol.services.memory.graph_service.GraphRepository.find_shortest_path"
        ) as mock_find:
            mock_find.return_value = mock_path

            path = await graph_service.find_shortest_path("ent-001", "ent-002")

            assert path is not None
            assert path["length"] == 1
            assert len(path["nodes"]) == 2

    @pytest.mark.asyncio
    async def test_close_driver(self, graph_service, mock_driver):
        """Test closing Neo4j driver."""
        await graph_service.close()
        mock_driver.close.assert_called_once()

    def test_get_id_field_mapping(self):
        """Test ID field mapping for different labels."""
        assert GraphMemoryService._get_id_field("Memory") == "memory_id"
        assert GraphMemoryService._get_id_field("Entity") == "entity_id"
        assert GraphMemoryService._get_id_field("Concept") == "concept_id"
        assert (
            GraphMemoryService._get_id_field("Unknown") == "entity_id"
        )  # Default


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_create_relationship_failure_raises_error(
        self, graph_service, mock_session
    ):
        """Test that create_relationship raises error on failure."""
        # Mock failed query
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        with pytest.raises(RuntimeError, match="Failed to create .* relationship"):
            await graph_service.create_relationship(
                from_id="ent-001",
                to_id="ent-002",
                rel_type=RelationshipType.RELATES_TO,
            )

    @pytest.mark.asyncio
    async def test_store_concept_node_failure_raises_error(
        self, graph_service, mock_session
    ):
        """Test that store_concept_node raises error on failure."""
        # Mock failed query
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        with pytest.raises(RuntimeError, match="Failed to store concept node"):
            await graph_service.store_concept_node("test_concept")

    @pytest.mark.asyncio
    async def test_get_node_degree_returns_zero_for_nonexistent_node(
        self, graph_service, mock_session
    ):
        """Test that get_node_degree returns 0 for nonexistent node."""
        # Mock query result with no record
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        degree = await graph_service.get_node_degree("nonexistent-id")

        assert degree == 0
