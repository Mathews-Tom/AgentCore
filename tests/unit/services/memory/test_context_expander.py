"""
Unit tests for GraphContextExpander service.

Tests context expansion using graph relationships for retrieval results.

Component ID: MEM-022
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    MemoryLayer,
    MemoryRecord,
    RelationshipEdge,
)
from agentcore.a2a_protocol.services.memory.context_expander import (
    ExpandedContext,
    GraphContextExpander,
)


@pytest.fixture
def mock_graph_service():
    """Create a mock GraphMemoryService."""
    service = AsyncMock()
    return service


@pytest.fixture
def sample_memory():
    """Create a sample memory record."""
    return MemoryRecord(
        memory_id="mem-001",
        memory_layer=MemoryLayer.SEMANTIC,
        content="Using Redis for session caching improves performance",
        summary="Redis caching for performance",
        embedding=[0.1] * 768,
        agent_id="agent-001",
        task_id="task-001",
        entities=["Redis", "caching", "performance"],
        facts=["Redis improves caching performance"],
        keywords=["redis", "cache", "performance"],
    )


@pytest.fixture
def critical_memory():
    """Create a critical memory record."""
    memory = MemoryRecord(
        memory_id="mem-002",
        memory_layer=MemoryLayer.SEMANTIC,
        content="Critical: Database connection timeout must be handled",
        summary="Database timeout handling critical",
        embedding=[0.2] * 768,
        agent_id="agent-001",
        task_id="task-001",
        entities=["database", "timeout", "connection"],
        facts=["Timeout handling is critical"],
        keywords=["database", "timeout", "critical"],
    )
    # Add is_critical attribute
    object.__setattr__(memory, "is_critical", True)
    return memory


@pytest.fixture
def sample_entities():
    """Create sample entity nodes."""
    return [
        EntityNode(
            entity_id="ent-001",
            entity_name="Redis",
            entity_type=EntityType.TOOL,
            properties={"version": "7.0"},
        ),
        EntityNode(
            entity_id="ent-002",
            entity_name="caching",
            entity_type=EntityType.CONCEPT,
            properties={},
        ),
        EntityNode(
            entity_id="ent-003",
            entity_name="performance",
            entity_type=EntityType.CONCEPT,
            properties={},
        ),
    ]


@pytest.fixture
def sample_relationships():
    """Create sample relationship edges."""
    return [
        RelationshipEdge(
            source_entity_id="Redis",
            target_entity_id="caching",
            relationship_type="relates_to",
            properties={"strength": 0.85},
        ),
        RelationshipEdge(
            source_entity_id="caching",
            target_entity_id="performance",
            relationship_type="relates_to",
            properties={"strength": 0.75},
        ),
    ]


class TestExpandedContext:
    """Tests for ExpandedContext container."""

    def test_init_default(self):
        """Test default initialization."""
        ctx = ExpandedContext(memory_id="mem-001")
        assert ctx.memory_id == "mem-001"
        assert ctx.entities == []
        assert ctx.relationships == []
        assert ctx.depth == 1

    def test_init_with_data(self, sample_entities, sample_relationships):
        """Test initialization with data."""
        ctx = ExpandedContext(
            memory_id="mem-001",
            entities=sample_entities,
            relationships=sample_relationships,
            depth=2,
        )
        assert ctx.memory_id == "mem-001"
        assert len(ctx.entities) == 3
        assert len(ctx.relationships) == 2
        assert ctx.depth == 2

    def test_format_triples(self, sample_relationships):
        """Test formatting relationships as triples."""
        ctx = ExpandedContext(
            memory_id="mem-001",
            relationships=sample_relationships,
        )
        triples = ctx.format_triples()

        assert len(triples) == 2
        assert "Redis -[RELATES_TO:0.85]-> caching" in triples
        assert "caching -[RELATES_TO:0.75]-> performance" in triples

    def test_format_triples_empty(self):
        """Test formatting with no relationships."""
        ctx = ExpandedContext(memory_id="mem-001")
        triples = ctx.format_triples()
        assert triples == []

    def test_to_dict(self, sample_entities, sample_relationships):
        """Test conversion to dictionary."""
        ctx = ExpandedContext(
            memory_id="mem-001",
            entities=sample_entities,
            relationships=sample_relationships,
            depth=2,
        )
        result = ctx.to_dict()

        assert result["memory_id"] == "mem-001"
        assert result["depth"] == 2
        assert len(result["entities"]) == 3
        assert len(result["relationships"]) == 2
        assert len(result["formatted_triples"]) == 2

        # Verify entity structure
        assert result["entities"][0]["name"] == "Redis"
        assert result["entities"][0]["type"] == "tool"

        # Verify relationship structure
        assert result["relationships"][0]["from"] == "Redis"
        assert result["relationships"][0]["to"] == "caching"
        assert result["relationships"][0]["type"] == "relates_to"
        assert result["relationships"][0]["strength"] == 0.85


class TestGraphContextExpanderInit:
    """Tests for GraphContextExpander initialization."""

    def test_init(self, mock_graph_service):
        """Test initialization."""
        expander = GraphContextExpander(mock_graph_service)
        assert expander.graph_service == mock_graph_service
        assert expander._logger is not None


class TestExpandSingleMemory:
    """Tests for single memory expansion."""

    @pytest.mark.asyncio
    async def test_expand_non_critical_memory(
        self, mock_graph_service, sample_memory, sample_entities
    ):
        """Test expanding non-critical memory uses depth=1."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            return_value=sample_entities[:2]
        )
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 1, "average_strength": 0.8}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_single_memory(sample_memory)

        assert result.depth == 1
        assert result.memory_id == "mem-001"
        # Should have called get_one_hop_neighbors for each entity
        assert mock_graph_service.get_one_hop_neighbors.call_count >= 1

    @pytest.mark.asyncio
    async def test_expand_critical_memory(
        self, mock_graph_service, critical_memory, sample_entities
    ):
        """Test expanding critical memory uses depth=2."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            return_value=sample_entities[:1]
        )
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_single_memory(critical_memory)

        assert result.depth == 2
        assert result.memory_id == "mem-002"

    @pytest.mark.asyncio
    async def test_expand_with_explicit_depth(
        self, mock_graph_service, sample_memory
    ):
        """Test expanding with explicit depth parameter."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(return_value=[])
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_single_memory(sample_memory, depth=3)

        assert result.depth == 3

    @pytest.mark.asyncio
    async def test_expand_handles_graph_error(
        self, mock_graph_service, sample_memory
    ):
        """Test graceful handling of graph service errors."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            side_effect=Exception("Graph service unavailable")
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_single_memory(sample_memory)

        # Should return empty context on error
        assert result.memory_id == "mem-001"
        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_expand_empty_entities(self, mock_graph_service):
        """Test expanding memory with no entities."""
        memory = MemoryRecord(
            memory_id="mem-003",
            memory_layer=MemoryLayer.EPISODIC,
            content="Simple memory",
            summary="Simple",
            embedding=[0.1] * 768,
            agent_id="agent-001",
            entities=[],
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_single_memory(memory)

        assert result.entities == []
        assert result.relationships == []
        mock_graph_service.get_one_hop_neighbors.assert_not_called()


class TestExpandContext:
    """Tests for batch context expansion."""

    @pytest.mark.asyncio
    async def test_expand_multiple_memories(
        self, mock_graph_service, sample_memory, critical_memory, sample_entities
    ):
        """Test expanding multiple memories."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            return_value=sample_entities[:1]
        )
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_context(
            memories=[sample_memory, critical_memory]
        )

        assert "memories" in result
        assert "expanded_context" in result
        assert len(result["memories"]) == 2
        assert len(result["expanded_context"]) == 2
        assert sample_memory.memory_id in result["expanded_context"]
        assert critical_memory.memory_id in result["expanded_context"]

    @pytest.mark.asyncio
    async def test_expand_respects_max_memories(
        self, mock_graph_service, sample_memory
    ):
        """Test max_memories limit is respected."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(return_value=[])
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        # Create 10 memories
        memories = [
            MemoryRecord(
                memory_id=f"mem-{i:03d}",
                memory_layer=MemoryLayer.SEMANTIC,
                content=f"Memory {i}",
                summary=f"Summary {i}",
                embedding=[0.1] * 768,
                agent_id="agent-001",
                entities=[],
            )
            for i in range(10)
        ]

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_context(memories=memories, max_memories=3)

        assert len(result["memories"]) == 3
        assert len(result["expanded_context"]) == 3

    @pytest.mark.asyncio
    async def test_expand_exclude_entities(
        self, mock_graph_service, sample_memory, sample_entities
    ):
        """Test excluding entities from result."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            return_value=sample_entities
        )
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_context(
            memories=[sample_memory], include_entities=False
        )

        context = result["expanded_context"][sample_memory.memory_id]
        assert context["entities"] == []

    @pytest.mark.asyncio
    async def test_expand_exclude_relationships(
        self, mock_graph_service, sample_memory, sample_entities
    ):
        """Test excluding relationships from result."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            return_value=sample_entities
        )
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_context(
            memories=[sample_memory], include_relationships=False
        )

        context = result["expanded_context"][sample_memory.memory_id]
        assert context["relationships"] == []
        assert context["formatted_triples"] == []

    @pytest.mark.asyncio
    async def test_expand_handles_individual_failure(
        self, mock_graph_service, sample_memory, critical_memory
    ):
        """Test handling failure for individual memory expansion."""
        # First memory succeeds, second fails
        call_count = 0

        async def mock_neighbors(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise Exception("Network error")
            return []

        mock_graph_service.get_one_hop_neighbors = AsyncMock(side_effect=mock_neighbors)

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_context(
            memories=[sample_memory, critical_memory]
        )

        # Both should have context (with or without error)
        assert len(result["expanded_context"]) == 2


class TestFormatGraphContext:
    """Tests for graph context formatting."""

    def test_format_with_relationships(
        self, mock_graph_service, sample_entities, sample_relationships
    ):
        """Test formatting with entities and relationships."""
        expander = GraphContextExpander(mock_graph_service)
        result = expander.format_graph_context(
            entities=sample_entities, relationships=sample_relationships
        )

        assert "Graph Context:" in result
        assert "Redis -[RELATES_TO:0.85]-> caching" in result
        assert "Related Entities:" in result
        assert "Redis (tool)" in result
        assert "caching (concept)" in result

    def test_format_empty_context(self, mock_graph_service):
        """Test formatting empty context."""
        expander = GraphContextExpander(mock_graph_service)
        result = expander.format_graph_context(entities=[], relationships=[])

        assert result == ""

    def test_format_entities_only(self, mock_graph_service, sample_entities):
        """Test formatting with only entities."""
        expander = GraphContextExpander(mock_graph_service)
        result = expander.format_graph_context(
            entities=sample_entities, relationships=[]
        )

        assert "Graph Context:" not in result
        assert "Related Entities:" in result
        assert "Redis (tool)" in result

    def test_format_relationships_only(
        self, mock_graph_service, sample_relationships
    ):
        """Test formatting with only relationships."""
        expander = GraphContextExpander(mock_graph_service)
        result = expander.format_graph_context(
            entities=[], relationships=sample_relationships
        )

        assert "Graph Context:" in result
        assert "Related Entities:" not in result


class TestExpandWithFormatting:
    """Tests for formatted expansion output."""

    @pytest.mark.asyncio
    async def test_expand_with_formatting(
        self, mock_graph_service, sample_memory, sample_entities
    ):
        """Test expansion with formatted output."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(
            return_value=sample_entities[:1]
        )
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_with_formatting(
            memories=[sample_memory], max_memories=10
        )

        assert "Memory:" in result
        assert sample_memory.summary in result
        assert sample_memory.memory_id in result

    @pytest.mark.asyncio
    async def test_expand_with_formatting_multiple(
        self, mock_graph_service, sample_memory, critical_memory
    ):
        """Test formatted expansion of multiple memories."""
        mock_graph_service.get_one_hop_neighbors = AsyncMock(return_value=[])
        mock_graph_service.aggregate_relationship_strength = AsyncMock(
            return_value={"path_count": 0}
        )

        expander = GraphContextExpander(mock_graph_service)
        result = await expander.expand_with_formatting(
            memories=[sample_memory, critical_memory], max_memories=2
        )

        # Should have separator between memories
        assert "---" in result
        assert sample_memory.summary in result
        assert critical_memory.summary in result
