"""
Unit Tests for HybridSearchService

Tests vector search, graph search, result merging, and scoring algorithms.
Uses mocks for Qdrant and Neo4j to enable fast, isolated testing.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from qdrant_client import models as qmodels

from agentcore.a2a_protocol.models.memory import MemoryRecord, StageType
from agentcore.a2a_protocol.services.memory.hybrid_search import (
    HybridSearchConfig,
    HybridSearchMetadata,
    HybridSearchService,
)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_graph_service():
    """Mock GraphMemoryService."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_retrieval_service():
    """Mock EnhancedRetrievalService."""
    service = AsyncMock()
    return service


@pytest.fixture
def sample_memory_1() -> MemoryRecord:
    """Create sample memory record."""
    return MemoryRecord(
        memory_id="mem-001",
        memory_layer="semantic",
        content="User prefers detailed technical explanations",
        summary="User technical preference",
        embedding=[0.1] * 768,
        agent_id="agent-001",
        session_id="session-001",
        task_id="task-001",
        timestamp=datetime.now(UTC),
        entities=["user", "preference"],
        facts=["prefers detailed explanations"],
        keywords=["technical", "detail"],
        stage_id="stage-planning",
        is_critical=True,
        access_count=10,
    )


@pytest.fixture
def sample_memory_2() -> MemoryRecord:
    """Create another sample memory record."""
    return MemoryRecord(
        memory_id="mem-002",
        memory_layer="semantic",
        content="System architecture uses microservices",
        summary="Microservices architecture",
        embedding=[0.2] * 768,
        agent_id="agent-001",
        session_id="session-001",
        task_id="task-001",
        timestamp=datetime.now(UTC),
        entities=["system", "architecture"],
        facts=["uses microservices"],
        keywords=["architecture", "microservices"],
        stage_id="stage-planning",
        is_critical=False,
        access_count=5,
    )


@pytest.fixture
def hybrid_search_service(
    mock_qdrant_client, mock_graph_service, mock_retrieval_service
):
    """Create HybridSearchService with mocks."""
    config = HybridSearchConfig(
        vector_weight=0.6,
        graph_weight=0.4,
        max_graph_depth=2,
        max_graph_seeds=10,
        enable_retrieval_scoring=False,  # Disable for simpler tests
    )

    return HybridSearchService(
        qdrant_client=mock_qdrant_client,
        graph_service=mock_graph_service,
        collection_name="test_memories",
        retrieval_service=mock_retrieval_service,
        config=config,
    )


class TestHybridSearchConfig:
    """Test HybridSearchConfig validation."""

    def test_config_default_values(self):
        """Test default configuration values."""
        config = HybridSearchConfig()

        assert config.vector_weight == 0.6
        assert config.graph_weight == 0.4
        assert config.max_graph_depth == 2
        assert config.max_graph_seeds == 10
        assert config.use_graph_expansion is True
        assert config.vector_score_threshold == 0.5

    def test_config_weights_sum_validation(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            HybridSearchConfig(vector_weight=0.5, graph_weight=0.3)

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = HybridSearchConfig(
            vector_weight=0.7,
            graph_weight=0.3,
            max_graph_depth=1,
            max_graph_seeds=5,
        )

        assert config.vector_weight == 0.7
        assert config.graph_weight == 0.3
        assert config.max_graph_depth == 1
        assert config.max_graph_seeds == 5


class TestVectorSearch:
    """Test vector search functionality."""

    @pytest.mark.asyncio
    async def test_vector_search_success(
        self, hybrid_search_service, mock_qdrant_client, sample_memory_1
    ):
        """Test successful vector search."""
        # Mock Qdrant search response
        mock_point = Mock()
        mock_point.id = "mem-001"
        mock_point.score = 0.95
        mock_point.payload = sample_memory_1.model_dump()

        mock_qdrant_client.search.return_value = [mock_point]

        # Execute vector search
        query_embedding = [0.1] * 768
        results = await hybrid_search_service.vector_search(
            query_embedding=query_embedding, limit=10
        )

        # Verify results
        assert len(results) == 1
        memory, score = results[0]
        assert memory.memory_id == "mem-001"
        assert score == 0.95

        # Verify Qdrant called correctly
        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs["collection_name"] == "test_memories"
        assert call_args.kwargs["query_vector"] == query_embedding
        assert call_args.kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_vector_search_with_filters(
        self, hybrid_search_service, mock_qdrant_client, sample_memory_1
    ):
        """Test vector search with filter conditions."""
        mock_point = Mock()
        mock_point.id = "mem-001"
        mock_point.score = 0.85
        mock_point.payload = sample_memory_1.model_dump()

        mock_qdrant_client.search.return_value = [mock_point]

        # Execute with filters
        query_embedding = [0.1] * 768
        filters = {"task_id": "task-001", "is_critical": True}

        results = await hybrid_search_service.vector_search(
            query_embedding=query_embedding, limit=10, filter_conditions=filters
        )

        assert len(results) == 1

        # Verify filter was built and passed
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(
        self, hybrid_search_service, mock_qdrant_client
    ):
        """Test vector search with no results."""
        mock_qdrant_client.search.return_value = []

        query_embedding = [0.1] * 768
        results = await hybrid_search_service.vector_search(
            query_embedding=query_embedding, limit=10
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_vector_search_error_handling(
        self, hybrid_search_service, mock_qdrant_client
    ):
        """Test vector search error handling."""
        mock_qdrant_client.search.side_effect = Exception("Qdrant connection failed")

        query_embedding = [0.1] * 768

        with pytest.raises(Exception, match="Qdrant connection failed"):
            await hybrid_search_service.vector_search(
                query_embedding=query_embedding, limit=10
            )


class TestGraphSearch:
    """Test graph search functionality."""

    @pytest.mark.asyncio
    async def test_graph_search_success(
        self, hybrid_search_service, mock_graph_service, sample_memory_1, sample_memory_2
    ):
        """Test successful graph search."""
        # Mock graph traversal response
        path_data = {
            "nodes": [
                {
                    "labels": ["Memory"],
                    "memory_id": "mem-seed",
                    "memory_layer": "semantic",
                    "content": "seed memory",
                    "summary": "seed",
                    "embedding": [],
                    "agent_id": "agent-001",
                },
                {
                    "labels": ["Memory"],
                    "memory_id": "mem-001",
                    **sample_memory_1.model_dump(),
                },
            ],
            "relationships": [],
        }

        mock_graph_service.traverse_graph.return_value = [path_data]

        # Execute graph search
        results = await hybrid_search_service.graph_search(
            seed_memory_ids=["mem-seed"], max_depth=2
        )

        # Verify results
        assert len(results) == 1
        memory, score, depth = results[0]
        assert memory.memory_id == "mem-001"
        assert score == 1.0  # 1-hop = 1.0 / 1
        assert depth == 1

    @pytest.mark.asyncio
    async def test_graph_search_multiple_depths(
        self, hybrid_search_service, mock_graph_service, sample_memory_1, sample_memory_2
    ):
        """Test graph search with multiple depth levels."""
        # Mock 2-hop path
        path_data = {
            "nodes": [
                {"labels": ["Memory"], "memory_id": "mem-seed"},
                {"labels": ["Entity"], "entity_id": "ent-001"},
                {
                    "labels": ["Memory"],
                    "memory_id": "mem-002",
                    **sample_memory_2.model_dump(),
                },
            ],
            "relationships": [],
        }

        mock_graph_service.traverse_graph.return_value = [path_data]

        results = await hybrid_search_service.graph_search(
            seed_memory_ids=["mem-seed"], max_depth=2
        )

        # 2-hop memory should have score 0.5
        assert len(results) == 1
        memory, score, depth = results[0]
        assert memory.memory_id == "mem-002"
        assert score == 0.5  # 1.0 / 2
        assert depth == 2

    @pytest.mark.asyncio
    async def test_graph_search_seed_limiting(
        self, hybrid_search_service, mock_graph_service
    ):
        """Test that graph search limits number of seeds."""
        # Create more seeds than max_graph_seeds (10)
        seed_ids = [f"mem-{i:03d}" for i in range(20)]

        mock_graph_service.traverse_graph.return_value = []

        await hybrid_search_service.graph_search(seed_memory_ids=seed_ids, max_depth=2)

        # Should only traverse from first 10 seeds
        assert mock_graph_service.traverse_graph.call_count == 10

    @pytest.mark.asyncio
    async def test_graph_search_error_handling(
        self, hybrid_search_service, mock_graph_service
    ):
        """Test graph search error handling."""
        mock_graph_service.traverse_graph.side_effect = Exception("Neo4j error")

        # Should return empty list on error, not raise
        results = await hybrid_search_service.graph_search(
            seed_memory_ids=["mem-001"], max_depth=2
        )

        assert results == []


class TestHybridSearch:
    """Test hybrid search combining vector and graph."""

    @pytest.mark.asyncio
    async def test_hybrid_search_vector_only(
        self, hybrid_search_service, mock_qdrant_client, sample_memory_1
    ):
        """Test hybrid search with only vector results."""
        mock_point = Mock()
        mock_point.id = "mem-001"
        mock_point.score = 0.9
        mock_point.payload = sample_memory_1.model_dump()

        mock_qdrant_client.search.return_value = [mock_point]

        query_embedding = [0.1] * 768
        results = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=10, use_graph_expansion=False
        )

        assert len(results) == 1
        memory, score, metadata = results[0]
        assert memory.memory_id == "mem-001"
        assert metadata.found_in_vector is True
        assert metadata.found_in_graph is False
        assert metadata.vector_score == 0.9
        assert metadata.graph_score == 0.0

    @pytest.mark.asyncio
    async def test_hybrid_search_combined(
        self,
        hybrid_search_service,
        mock_qdrant_client,
        mock_graph_service,
        sample_memory_1,
        sample_memory_2,
    ):
        """Test hybrid search combining vector and graph results."""
        # Mock vector search
        mock_point = Mock()
        mock_point.id = "mem-001"
        mock_point.score = 0.8
        mock_point.payload = sample_memory_1.model_dump()

        mock_qdrant_client.search.return_value = [mock_point]

        # Mock graph search returning different memory
        path_data = {
            "nodes": [
                {"labels": ["Memory"], "memory_id": "mem-001"},
                {
                    "labels": ["Memory"],
                    "memory_id": "mem-002",
                    **sample_memory_2.model_dump(),
                },
            ],
        }

        mock_graph_service.traverse_graph.return_value = [path_data]

        query_embedding = [0.1] * 768
        results = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=10, use_graph_expansion=True
        )

        # Should have 2 results: one from vector, one from graph
        assert len(results) == 2

        # Check scores are combined correctly
        for memory, score, metadata in results:
            if memory.memory_id == "mem-001":
                # Found in both vector and graph
                assert metadata.found_in_vector is True
                # Graph might find it too
            elif memory.memory_id == "mem-002":
                # Found only in graph
                assert metadata.found_in_graph is True

    @pytest.mark.asyncio
    async def test_hybrid_search_deduplication(
        self, hybrid_search_service, mock_qdrant_client, mock_graph_service, sample_memory_1
    ):
        """Test that hybrid search deduplicates results."""
        # Mock vector search
        mock_point = Mock()
        mock_point.id = "mem-001"
        mock_point.score = 0.9
        mock_point.payload = sample_memory_1.model_dump()

        mock_qdrant_client.search.return_value = [mock_point]

        # Mock graph search returning same memory
        path_data = {
            "nodes": [
                {"labels": ["Memory"], "memory_id": "mem-seed"},
                {
                    "labels": ["Memory"],
                    "memory_id": "mem-001",
                    **sample_memory_1.model_dump(),
                },
            ],
        }

        mock_graph_service.traverse_graph.return_value = [path_data]

        query_embedding = [0.1] * 768
        results = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=10, use_graph_expansion=True
        )

        # Should have only 1 result despite being in both
        assert len(results) == 1
        memory, score, metadata = results[0]
        assert memory.memory_id == "mem-001"
        assert metadata.found_in_vector is True
        assert metadata.found_in_graph is True

    @pytest.mark.asyncio
    async def test_hybrid_search_weighted_scoring(
        self, hybrid_search_service, mock_qdrant_client, sample_memory_1
    ):
        """Test weighted score combination."""
        mock_point = Mock()
        mock_point.id = "mem-001"
        mock_point.score = 1.0  # Perfect vector match
        mock_point.payload = sample_memory_1.model_dump()

        mock_qdrant_client.search.return_value = [mock_point]

        query_embedding = [0.1] * 768
        results = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding,
            limit=10,
            use_graph_expansion=False,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        memory, score, metadata = results[0]

        # With vector=1.0, graph=0.0, weights 0.6/0.4
        # hybrid_score = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert metadata.vector_score == 1.0
        assert metadata.graph_score == 0.0
        assert abs(metadata.hybrid_score - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_hybrid_search_no_query_embedding(self, hybrid_search_service):
        """Test hybrid search without query embedding."""
        results = await hybrid_search_service.hybrid_search(
            query_embedding=None, limit=10
        )

        # Should return empty without vector search
        assert results == []


class TestFilterBuilding:
    """Test Qdrant filter building."""

    def test_build_filter_single_condition(self, hybrid_search_service):
        """Test building filter with single condition."""
        conditions = {"task_id": "task-001"}
        qdrant_filter = hybrid_search_service._build_qdrant_filter(conditions)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 1
        assert qdrant_filter.must[0].key == "task_id"

    def test_build_filter_multiple_conditions(self, hybrid_search_service):
        """Test building filter with multiple conditions."""
        conditions = {"task_id": "task-001", "is_critical": True, "memory_layer": "semantic"}

        qdrant_filter = hybrid_search_service._build_qdrant_filter(conditions)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 3

    def test_build_filter_list_condition(self, hybrid_search_service):
        """Test building filter with list value."""
        conditions = {"stage_id": ["stage-001", "stage-002"]}

        qdrant_filter = hybrid_search_service._build_qdrant_filter(conditions)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 1

    def test_build_filter_none_values_ignored(self, hybrid_search_service):
        """Test that None values are ignored."""
        conditions = {"task_id": "task-001", "stage_id": None}

        qdrant_filter = hybrid_search_service._build_qdrant_filter(conditions)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 1  # Only task_id

    def test_build_filter_empty_conditions(self, hybrid_search_service):
        """Test building filter with empty conditions."""
        qdrant_filter = hybrid_search_service._build_qdrant_filter({})

        assert qdrant_filter is None

    def test_build_filter_none_conditions(self, hybrid_search_service):
        """Test building filter with None."""
        qdrant_filter = hybrid_search_service._build_qdrant_filter(None)

        assert qdrant_filter is None


class TestMetadata:
    """Test HybridSearchMetadata model."""

    def test_metadata_creation(self):
        """Test creating metadata object."""
        metadata = HybridSearchMetadata(
            vector_score=0.9,
            graph_score=0.7,
            hybrid_score=0.82,
            final_score=0.82,
            found_in_vector=True,
            found_in_graph=True,
            graph_depth=2,
            relationship_count=3,
        )

        assert metadata.vector_score == 0.9
        assert metadata.graph_score == 0.7
        assert metadata.hybrid_score == 0.82
        assert metadata.found_in_vector is True
        assert metadata.found_in_graph is True
        assert metadata.graph_depth == 2
        assert metadata.relationship_count == 3

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = HybridSearchMetadata()

        assert metadata.vector_score == 0.0
        assert metadata.graph_score == 0.0
        assert metadata.hybrid_score == 0.0
        assert metadata.final_score == 0.0
        assert metadata.found_in_vector is False
        assert metadata.found_in_graph is False
        assert metadata.graph_depth is None
        assert metadata.relationship_count == 0
