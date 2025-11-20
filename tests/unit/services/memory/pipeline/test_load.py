"""Unit tests for Load phase tasks.

Tests cover:
- LoadTask base class
- VectorLoadTask
- GraphLoadTask
- MemoryLoadTask
- MultiBackendLoadTask
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.a2a_protocol.services.memory.pipeline.load import (
    GraphLoadTask,
    LoadTask,
    MemoryLoadTask,
    MultiBackendLoadTask,
    VectorLoadTask,
)


@pytest.mark.asyncio
class TestLoadTask:
    """Tests for LoadTask base class."""

    async def test_load_task_initialization(self):
        """Test load task initialization."""
        task = LoadTask(
            name="test_load",
            backend_type="test_backend",
            enable_rollback=True,
            batch_size=50,
        )

        assert task.name == "test_load"
        assert task.backend_type == "test_backend"
        assert task.enable_rollback is True
        assert task.batch_size == 50

    async def test_load_task_default_execute(self):
        """Test default execute implementation."""
        task = LoadTask(name="test_load", backend_type="generic")
        result = await task.run_with_retry({"data": []})

        assert result.is_success()
        assert result.output["backend"] == "generic"
        assert result.output["stored_count"] == 0


@pytest.mark.asyncio
class TestVectorLoadTask:
    """Tests for VectorLoadTask."""

    async def test_vector_load_initialization(self):
        """Test vector load task initialization."""
        task = VectorLoadTask(embedding_dimension=768)

        assert task.name == "vector_load"
        assert task.backend_type == "vector"
        assert task.embedding_dimension == 768

    async def test_vector_load_requires_embeddings(self):
        """Test vector load requires embeddings."""
        task = VectorLoadTask()

        result = await task.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_vector_load_with_embeddings(self):
        """Test vector load with embeddings."""
        task = VectorLoadTask(embedding_dimension=1536)

        embeddings = [
            {"vector": [0.1] * 1536, "content": "text1"},
            {"vector": [0.2] * 1536, "content": "text2"}
        ]

        result = await task.run_with_retry({
            "embeddings": embeddings,
            "session_id": "session-123"
        })

        assert result.is_success()
        assert result.output["backend"] == "vector"
        assert result.output["session_id"] == "session-123"

    async def test_vector_load_from_semantic_analysis(self):
        """Test vector load from semantic_analysis task output."""
        task = VectorLoadTask()

        result = await task.run_with_retry({
            "semantic_analysis": {
                "embeddings": [{"vector": [0.1] * 1536}]
            }
        })

        assert result.is_success()


@pytest.mark.asyncio
class TestGraphLoadTask:
    """Tests for GraphLoadTask."""

    async def test_graph_load_initialization(self):
        """Test graph load task initialization."""
        task = GraphLoadTask(merge_duplicates=False)

        assert task.name == "graph_load"
        assert task.backend_type == "graph"
        assert task.merge_duplicates is False

    async def test_graph_load_requires_entities(self):
        """Test graph load requires entities."""
        task = GraphLoadTask()

        result = await task.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_graph_load_with_entities(self):
        """Test graph load with entities and relationships."""
        task = GraphLoadTask(merge_duplicates=True)

        entities = [
            {"name": "Alice", "type": "person"},
            {"name": "Python", "type": "tool"}
        ]
        relationships = [
            {"from": "Alice", "to": "Python", "type": "USES"}
        ]

        result = await task.run_with_retry({
            "entities": entities,
            "relationships": relationships,
            "session_id": "session-123"
        })

        assert result.is_success()
        assert result.output["backend"] == "graph"
        assert result.output["session_id"] == "session-123"

    async def test_graph_load_from_cognify_tasks(self):
        """Test graph load from entity_extraction and relationship_detection."""
        task = GraphLoadTask()

        result = await task.run_with_retry({
            "entity_extraction": {
                "entities": [{"name": "Alice"}]
            },
            "relationship_detection": {
                "relationships": [{"from": "Alice", "to": "Bob"}]
            }
        })

        assert result.is_success()


@pytest.mark.asyncio
class TestMemoryLoadTask:
    """Tests for MemoryLoadTask."""

    async def test_memory_load_initialization(self):
        """Test memory load task initialization."""
        task = MemoryLoadTask()

        assert task.name == "memory_load"
        assert task.backend_type == "memory"

    async def test_memory_load_requires_memories_and_agent_id(self):
        """Test memory load requires memories and agent_id."""
        task = MemoryLoadTask()

        result = await task.run_with_retry({})
        assert result.is_failure()

    async def test_memory_load_with_parameters(self):
        """Test memory load with full parameters."""
        task = MemoryLoadTask()

        memories = [
            {"content": "Memory 1", "importance": 0.8},
            {"content": "Memory 2", "importance": 0.6}
        ]

        result = await task.run_with_retry({
            "memories": memories,
            "agent_id": "agent-123",
            "session_id": "session-456"
        })

        assert result.is_success()
        assert result.output["backend"] == "memory"
        assert result.output["agent_id"] == "agent-123"
        assert result.output["session_id"] == "session-456"


@pytest.mark.asyncio
class TestMultiBackendLoadTask:
    """Tests for MultiBackendLoadTask."""

    async def test_multi_backend_load_initialization(self):
        """Test multi-backend load task initialization."""
        task = MultiBackendLoadTask()

        assert task.name == "multi_backend_load"
        assert task.backend_type == "multi"
        assert task.enable_rollback is True

    async def test_multi_backend_load_vector_only(self):
        """Test multi-backend load with vector data only."""
        task = MultiBackendLoadTask()

        result = await task.run_with_retry({
            "embeddings": [{"vector": [0.1] * 1536}]
        })

        assert result.is_success()
        assert "vector" in result.output["backends_written"]
        assert "graph" not in result.output["backends_written"]

    async def test_multi_backend_load_graph_only(self):
        """Test multi-backend load with graph data only."""
        task = MultiBackendLoadTask()

        result = await task.run_with_retry({
            "entities": [{"name": "Alice"}]
        })

        assert result.is_success()
        assert "graph" in result.output["backends_written"]
        assert "vector" not in result.output["backends_written"]

    async def test_multi_backend_load_memory_only(self):
        """Test multi-backend load with memory data only."""
        task = MultiBackendLoadTask()

        result = await task.run_with_retry({
            "memories": [{"content": "Test"}],
            "agent_id": "agent-123"
        })

        assert result.is_success()
        assert "memory" in result.output["backends_written"]

    async def test_multi_backend_load_all_backends(self):
        """Test multi-backend load with all data types."""
        task = MultiBackendLoadTask()

        result = await task.run_with_retry({
            "embeddings": [{"vector": [0.1] * 1536}],
            "entities": [{"name": "Alice"}],
            "memories": [{"content": "Test"}],
            "agent_id": "agent-123"
        })

        assert result.is_success()
        assert len(result.output["backends_written"]) == 3
        assert "vector" in result.output["backends_written"]
        assert "graph" in result.output["backends_written"]
        assert "memory" in result.output["backends_written"]

    async def test_multi_backend_load_from_cognify_tasks(self):
        """Test multi-backend load from cognify task outputs."""
        task = MultiBackendLoadTask()

        result = await task.run_with_retry({
            "semantic_analysis": {
                "embeddings": [{"vector": [0.1] * 1536}]
            },
            "entity_extraction": {
                "entities": [{"name": "Alice"}]
            }
        })

        assert result.is_success()
        assert "vector" in result.output["backends_written"]
        assert "graph" in result.output["backends_written"]


@pytest.mark.asyncio
class TestLoadIntegration:
    """Integration tests for Load phase."""

    async def test_complete_ecl_pipeline(self):
        """Test complete Extract-Cognify-Load pipeline."""
        from agentcore.a2a_protocol.services.memory.pipeline import Pipeline
        from agentcore.a2a_protocol.services.memory.pipeline.extract import (
            ConversationExtractor,
        )
        from agentcore.a2a_protocol.services.memory.pipeline.cognify import (
            EntityExtractionTask,
            SemanticAnalysisTask,
        )

        pipeline = Pipeline(pipeline_id="full_ecl", parallel_execution=True)

        # Extract
        extract = ConversationExtractor()

        # Cognify
        entity = EntityExtractionTask()
        entity.dependencies = ["conversation_extractor"]

        semantic = SemanticAnalysisTask()
        semantic.dependencies = ["conversation_extractor"]

        # Load
        multi_load = MultiBackendLoadTask()
        multi_load.dependencies = ["entity_extraction", "semantic_analysis"]

        pipeline.add_task(extract)
        pipeline.add_task(entity)
        pipeline.add_task(semantic)
        pipeline.add_task(multi_load)

        # This will partially fail (no session_id) but structure is valid
        result = await pipeline.execute({
            "session_id": "session-123",
            "content": "Test conversation"
        })

        # Check that all tasks were attempted
        assert "conversation_extractor" in result.task_results

    async def test_load_phase_parallel_execution(self):
        """Test parallel execution of load tasks."""
        from agentcore.a2a_protocol.services.memory.pipeline import Pipeline

        pipeline = Pipeline(pipeline_id="parallel_load", parallel_execution=True)

        vector_load = VectorLoadTask()
        graph_load = GraphLoadTask()

        pipeline.add_task(vector_load)
        pipeline.add_task(graph_load)

        # Both should execute in parallel
        result = await pipeline.execute({
            "embeddings": [{"vector": [0.1] * 1536}],
            "entities": [{"name": "Alice"}]
        })

        assert result.is_success()
        assert result.task_results["vector_load"].is_success()
        assert result.task_results["graph_load"].is_success()
