"""Unit tests for GraphMemoryService.

Tests cover:
- Memory node creation
- Entity node creation with deduplication
- Concept node creation
- Relationship creation (MENTIONS, RELATES_TO, PART_OF, FOLLOWS, PRECEDES)
- Graph traversal queries (multi-hop entity retrieval, temporal chains)
- Error handling and validation
- Edge cases (empty values, out-of-range scores)

References:
    - MEM-017: GraphMemoryService (Neo4j Integration)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from neo4j.exceptions import Neo4jError

from agentcore.memory.graph.service import (
    EntityType,
    GraphMemoryService,
    MemoryLayer,
    MemoryStage,
    RelationshipType,
)


# Sample test data
SAMPLE_MEMORY_ID = str(uuid4())
SAMPLE_ENTITY_ID = str(uuid4())
SAMPLE_CONCEPT_ID = str(uuid4())
SAMPLE_AGENT_ID = "agent-123"
SAMPLE_SESSION_ID = "session-456"


class TestGraphMemoryService:
    """Test GraphMemoryService class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock Neo4j session."""
        session = AsyncMock()
        session.run = AsyncMock()
        return session

    @pytest.fixture
    def mock_get_session(self, mock_session):
        """Mock get_session context manager."""
        with patch("agentcore.memory.graph.service.get_session") as mock:
            mock.return_value.__aenter__.return_value = mock_session
            mock.return_value.__aexit__.return_value = None
            yield mock

    @pytest.mark.asyncio
    async def test_create_memory_node(self, mock_get_session, mock_session):
        """Test creating Memory node."""
        # Setup mock response
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"memory_id": SAMPLE_MEMORY_ID})
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        # Create memory node
        memory_id = await service.create_memory_node(
            agent_id=SAMPLE_AGENT_ID,
            session_id=SAMPLE_SESSION_ID,
            layer=MemoryLayer.EPISODIC,
            stage=MemoryStage.EXECUTION,
            content="Test memory content",
            criticality=0.8,
            memory_id=SAMPLE_MEMORY_ID,
        )

        # Verify result
        assert memory_id == SAMPLE_MEMORY_ID

        # Verify query execution
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (m:Memory" in call_args[0][0]
        assert call_args[1]["memory_id"] == SAMPLE_MEMORY_ID
        assert call_args[1]["agent_id"] == SAMPLE_AGENT_ID
        assert call_args[1]["layer"] == "episodic"
        assert call_args[1]["stage"] == "execution"
        assert call_args[1]["criticality"] == 0.8

    @pytest.mark.asyncio
    async def test_create_memory_node_generates_id(self, mock_get_session, mock_session):
        """Test Memory node creation generates UUID if not provided."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(
            return_value={"memory_id": "generated-uuid"}
        )
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        memory_id = await service.create_memory_node(
            agent_id=SAMPLE_AGENT_ID,
            session_id=SAMPLE_SESSION_ID,
            layer=MemoryLayer.SEMANTIC,
            stage=MemoryStage.PLANNING,
            content="Test content",
            criticality=0.5,
        )

        assert memory_id is not None
        assert len(memory_id) > 0

    @pytest.mark.asyncio
    async def test_create_memory_node_invalid_criticality(self, mock_get_session):
        """Test validation for criticality out of range."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Criticality must be in range"):
            await service.create_memory_node(
                agent_id=SAMPLE_AGENT_ID,
                session_id=SAMPLE_SESSION_ID,
                layer=MemoryLayer.EPISODIC,
                stage=MemoryStage.EXECUTION,
                content="Test",
                criticality=1.5,  # Invalid
            )

        with pytest.raises(ValueError, match="Criticality must be in range"):
            await service.create_memory_node(
                agent_id=SAMPLE_AGENT_ID,
                session_id=SAMPLE_SESSION_ID,
                layer=MemoryLayer.EPISODIC,
                stage=MemoryStage.EXECUTION,
                content="Test",
                criticality=-0.1,  # Invalid
            )

    @pytest.mark.asyncio
    async def test_create_memory_node_empty_content(self, mock_get_session):
        """Test validation for empty content."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Memory content cannot be empty"):
            await service.create_memory_node(
                agent_id=SAMPLE_AGENT_ID,
                session_id=SAMPLE_SESSION_ID,
                layer=MemoryLayer.EPISODIC,
                stage=MemoryStage.EXECUTION,
                content="",
                criticality=0.5,
            )

        with pytest.raises(ValueError, match="Memory content cannot be empty"):
            await service.create_memory_node(
                agent_id=SAMPLE_AGENT_ID,
                session_id=SAMPLE_SESSION_ID,
                layer=MemoryLayer.EPISODIC,
                stage=MemoryStage.EXECUTION,
                content="   ",
                criticality=0.5,
            )

    @pytest.mark.asyncio
    async def test_create_entity_node(self, mock_get_session, mock_session):
        """Test creating Entity node with MERGE deduplication."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"entity_id": SAMPLE_ENTITY_ID})
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        entity_id = await service.create_entity_node(
            name="Python",
            entity_type=EntityType.TOOL,
            confidence=0.95,
            entity_id=SAMPLE_ENTITY_ID,
            properties={"version": "3.12"},
        )

        assert entity_id == SAMPLE_ENTITY_ID

        # Verify MERGE query
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MERGE (e:Entity" in call_args[0][0]
        assert "ON CREATE SET" in call_args[0][0]
        assert "ON MATCH SET" in call_args[0][0]
        assert call_args[1]["name"] == "Python"
        assert call_args[1]["entity_type"] == "tool"
        assert call_args[1]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_create_entity_node_invalid_confidence(self, mock_get_session):
        """Test validation for confidence out of range."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Confidence must be in range"):
            await service.create_entity_node(
                name="Test",
                entity_type=EntityType.TOOL,
                confidence=1.5,
            )

    @pytest.mark.asyncio
    async def test_create_entity_node_empty_name(self, mock_get_session):
        """Test validation for empty entity name."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            await service.create_entity_node(
                name="",
                entity_type=EntityType.TOOL,
                confidence=0.9,
            )

    @pytest.mark.asyncio
    async def test_create_concept_node(self, mock_get_session, mock_session):
        """Test creating Concept node."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"concept_id": SAMPLE_CONCEPT_ID})
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        concept_id = await service.create_concept_node(
            name="Machine Learning",
            description="AI systems that learn from data",
            category="ai",
            concept_id=SAMPLE_CONCEPT_ID,
        )

        assert concept_id == SAMPLE_CONCEPT_ID

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (c:Concept" in call_args[0][0]
        assert call_args[1]["name"] == "Machine Learning"
        assert call_args[1]["category"] == "ai"

    @pytest.mark.asyncio
    async def test_create_concept_node_empty_description(self, mock_get_session):
        """Test validation for empty description."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Concept description cannot be empty"):
            await service.create_concept_node(
                name="Test",
                description="",
                category="test",
            )

    @pytest.mark.asyncio
    async def test_create_mentions_relationship(self, mock_get_session, mock_session):
        """Test creating MENTIONS relationship."""
        mock_session.run.return_value = AsyncMock()

        service = GraphMemoryService()

        await service.create_mentions_relationship(
            memory_id=SAMPLE_MEMORY_ID,
            entity_id=SAMPLE_ENTITY_ID,
            position=10,
            context="Python is used for backend",
            sentiment=0.5,
        )

        # Verify relationship creation
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (m)-[r:MENTIONS" in call_args[0][0]
        assert call_args[1]["memory_id"] == SAMPLE_MEMORY_ID
        assert call_args[1]["entity_id"] == SAMPLE_ENTITY_ID
        assert call_args[1]["position"] == 10
        assert call_args[1]["sentiment"] == 0.5

    @pytest.mark.asyncio
    async def test_create_mentions_relationship_invalid_sentiment(self, mock_get_session):
        """Test validation for sentiment out of range."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Sentiment must be in range"):
            await service.create_mentions_relationship(
                memory_id=SAMPLE_MEMORY_ID,
                entity_id=SAMPLE_ENTITY_ID,
                position=0,
                context="test",
                sentiment=2.0,
            )

    @pytest.mark.asyncio
    async def test_create_relates_to_relationship(self, mock_get_session, mock_session):
        """Test creating RELATES_TO relationship between entities."""
        mock_session.run.return_value = AsyncMock()

        service = GraphMemoryService()

        await service.create_relates_to_relationship(
            source_entity_id=SAMPLE_ENTITY_ID,
            target_entity_id=str(uuid4()),
            relationship_type=RelationshipType.DEPENDS_ON,
            strength=0.8,
            confidence=0.9,
        )

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MERGE (e1)-[r:RELATES_TO" in call_args[0][0]
        assert call_args[1]["relationship_type"] == "depends_on"
        assert call_args[1]["strength"] == 0.8
        assert call_args[1]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_create_relates_to_relationship_invalid_strength(self, mock_get_session):
        """Test validation for strength out of range."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Strength must be in range"):
            await service.create_relates_to_relationship(
                source_entity_id=SAMPLE_ENTITY_ID,
                target_entity_id=str(uuid4()),
                relationship_type=RelationshipType.DEPENDS_ON,
                strength=1.5,
                confidence=0.9,
            )

    @pytest.mark.asyncio
    async def test_create_part_of_relationship(self, mock_get_session, mock_session):
        """Test creating PART_OF relationship."""
        mock_session.run.return_value = AsyncMock()

        service = GraphMemoryService()

        await service.create_part_of_relationship(
            entity_id=SAMPLE_ENTITY_ID,
            concept_id=SAMPLE_CONCEPT_ID,
            relevance=0.85,
        )

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (e)-[r:PART_OF" in call_args[0][0]
        assert call_args[1]["relevance"] == 0.85

    @pytest.mark.asyncio
    async def test_create_follows_relationship(self, mock_get_session, mock_session):
        """Test creating FOLLOWS temporal relationship."""
        mock_session.run.return_value = AsyncMock()

        service = GraphMemoryService()

        await service.create_follows_relationship(
            source_memory_id=SAMPLE_MEMORY_ID,
            target_memory_id=str(uuid4()),
            time_delta=120,
            stage_transition=True,
        )

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (m1)-[r:FOLLOWS" in call_args[0][0]
        assert call_args[1]["time_delta"] == 120
        assert call_args[1]["stage_transition"] is True

    @pytest.mark.asyncio
    async def test_create_follows_relationship_negative_delta(self, mock_get_session):
        """Test validation for negative time delta."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Time delta must be non-negative"):
            await service.create_follows_relationship(
                source_memory_id=SAMPLE_MEMORY_ID,
                target_memory_id=str(uuid4()),
                time_delta=-10,
            )

    @pytest.mark.asyncio
    async def test_create_precedes_relationship(self, mock_get_session, mock_session):
        """Test creating PRECEDES temporal relationship."""
        mock_session.run.return_value = AsyncMock()

        service = GraphMemoryService()

        await service.create_precedes_relationship(
            source_memory_id=SAMPLE_MEMORY_ID,
            target_memory_id=str(uuid4()),
            time_delta=60,
            stage_transition=False,
        )

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (m1)-[r:PRECEDES" in call_args[0][0]
        assert call_args[1]["time_delta"] == 60
        assert call_args[1]["stage_transition"] is False

    @pytest.mark.asyncio
    async def test_get_related_entities(self, mock_get_session, mock_session):
        """Test multi-hop entity traversal query."""
        # Setup mock result
        mock_result = AsyncMock()
        mock_records = [
            {
                "entity_id": "entity-1",
                "name": "Python",
                "entity_type": "tool",
                "confidence": 0.95,
                "distance": 1,
            },
            {
                "entity_id": "entity-2",
                "name": "FastAPI",
                "entity_type": "tool",
                "confidence": 0.90,
                "distance": 2,
            },
        ]

        async def async_iter():
            for record in mock_records:
                yield record

        mock_result.__aiter__ = lambda self: async_iter()
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        entities = await service.get_related_entities(
            memory_id=SAMPLE_MEMORY_ID,
            max_depth=2,
            min_confidence=0.5,
        )

        assert len(entities) == 2
        assert entities[0]["name"] == "Python"
        assert entities[0]["distance"] == 1
        assert entities[1]["name"] == "FastAPI"
        assert entities[1]["distance"] == 2

        # Verify query
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH path = (m)-[:MENTIONS|RELATES_TO*1..2]" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_related_entities_invalid_depth(self, mock_get_session):
        """Test validation for max_depth out of range."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Max depth must be in range"):
            await service.get_related_entities(
                memory_id=SAMPLE_MEMORY_ID,
                max_depth=5,
            )

    @pytest.mark.asyncio
    async def test_get_temporal_chain_forward(self, mock_get_session, mock_session):
        """Test temporal chain traversal (forward direction)."""
        mock_result = AsyncMock()
        mock_records = [
            {
                "memory_id": "mem-1",
                "content": "First memory",
                "stage": "planning",
                "layer": "episodic",
                "created_at": datetime(2024, 1, 1, 10, 0),
            },
            {
                "memory_id": "mem-2",
                "content": "Second memory",
                "stage": "execution",
                "layer": "episodic",
                "created_at": datetime(2024, 1, 1, 10, 5),
            },
        ]

        async def async_iter():
            for record in mock_records:
                yield record

        mock_result.__aiter__ = lambda self: async_iter()
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        memories = await service.get_temporal_chain(
            memory_id=SAMPLE_MEMORY_ID,
            direction="forward",
            max_length=10,
        )

        assert len(memories) == 2
        assert memories[0]["content"] == "First memory"
        assert memories[1]["content"] == "Second memory"

        # Verify query
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH path = (m:Memory {memory_id: $memory_id})-[:FOLLOWS*1..10]" in call_args[0][0]
        assert "ORDER BY mem.created_at ASC" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_temporal_chain_backward(self, mock_get_session, mock_session):
        """Test temporal chain traversal (backward direction)."""
        mock_result = AsyncMock()
        mock_result.__aiter__ = lambda self: async_iter([])

        async def async_iter(items):
            for item in items:
                yield item

        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        memories = await service.get_temporal_chain(
            memory_id=SAMPLE_MEMORY_ID,
            direction="backward",
            max_length=5,
        )

        # Verify query uses PRECEDES
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "-[:PRECEDES*1..5]-" in call_args[0][0]
        assert "ORDER BY mem.created_at DESC" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_temporal_chain_invalid_direction(self, mock_get_session):
        """Test validation for invalid direction."""
        service = GraphMemoryService()

        with pytest.raises(ValueError, match="Direction must be"):
            await service.get_temporal_chain(
                memory_id=SAMPLE_MEMORY_ID,
                direction="sideways",
            )

    @pytest.mark.asyncio
    async def test_get_entity_by_name(self, mock_get_session, mock_session):
        """Test retrieving entity by name."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(
            return_value={
                "entity_id": SAMPLE_ENTITY_ID,
                "name": "Python",
                "entity_type": "tool",
                "confidence": 0.95,
                "properties": {"version": "3.12"},
                "mention_count": 10,
            }
        )
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        entity = await service.get_entity_by_name(
            name="Python",
            entity_type=EntityType.TOOL,
        )

        assert entity is not None
        assert entity["name"] == "Python"
        assert entity["entity_type"] == "tool"
        assert entity["mention_count"] == 10

    @pytest.mark.asyncio
    async def test_get_entity_by_name_not_found(self, mock_get_session, mock_session):
        """Test entity not found returns None."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run.return_value = mock_result

        service = GraphMemoryService()

        entity = await service.get_entity_by_name(name="NonExistent")

        assert entity is None

    @pytest.mark.asyncio
    async def test_neo4j_error_handling(self, mock_get_session, mock_session):
        """Test Neo4j error handling."""
        mock_session.run.side_effect = Neo4jError("Database connection failed")

        service = GraphMemoryService()

        with pytest.raises(Neo4jError):
            await service.create_memory_node(
                agent_id=SAMPLE_AGENT_ID,
                session_id=SAMPLE_SESSION_ID,
                layer=MemoryLayer.EPISODIC,
                stage=MemoryStage.EXECUTION,
                content="Test",
                criticality=0.5,
            )


class TestMemoryLayer:
    """Test MemoryLayer enum."""

    def test_memory_layers(self):
        """Test all memory layer values."""
        assert MemoryLayer.EPISODIC.value == "episodic"
        assert MemoryLayer.SEMANTIC.value == "semantic"
        assert MemoryLayer.PROCEDURAL.value == "procedural"


class TestMemoryStage:
    """Test MemoryStage enum."""

    def test_memory_stages(self):
        """Test all memory stage values."""
        assert MemoryStage.PLANNING.value == "planning"
        assert MemoryStage.EXECUTION.value == "execution"
        assert MemoryStage.REFLECTION.value == "reflection"
        assert MemoryStage.VERIFICATION.value == "verification"


class TestEntityType:
    """Test EntityType enum."""

    def test_entity_types(self):
        """Test all entity type values."""
        assert EntityType.PERSON.value == "person"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.TOOL.value == "tool"
        assert EntityType.CONSTRAINT.value == "constraint"
        assert EntityType.OTHER.value == "other"


class TestRelationshipType:
    """Test RelationshipType enum."""

    def test_relationship_types(self):
        """Test all relationship type values."""
        assert RelationshipType.DEPENDS_ON.value == "depends_on"
        assert RelationshipType.PART_OF.value == "part_of"
        assert RelationshipType.SIMILAR_TO.value == "similar_to"
        assert RelationshipType.CONTRADICTS.value == "contradicts"
        assert RelationshipType.CAUSES.value == "causes"
        assert RelationshipType.ENABLES.value == "enables"
