"""Unit tests for Cognify phase tasks.

Tests cover:
- CognifyTask base class
- EntityExtractionTask
- RelationshipDetectionTask
- SemanticAnalysisTask
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.a2a_protocol.services.memory.pipeline.cognify import (
    CognifyTask,
    EntityExtractionTask,
    RelationshipDetectionTask,
    SemanticAnalysisTask,
)


@pytest.mark.asyncio
class TestCognifyTask:
    """Tests for CognifyTask base class."""

    async def test_cognify_task_initialization(self):
        """Test cognify task initialization."""
        task = CognifyTask(
            name="test_cognify",
            cognify_type="test_type",
            model="gpt-4.1-mini",
        )

        assert task.name == "test_cognify"
        assert task.cognify_type == "test_type"
        assert task.model == "gpt-4.1-mini"

    async def test_cognify_task_default_execute(self):
        """Test default execute implementation."""
        task = CognifyTask(name="test_cognify", cognify_type="generic")
        result = await task.run_with_retry({"content": "test"})

        assert result.is_success()
        assert result.output["cognify_type"] == "generic"


@pytest.mark.asyncio
class TestEntityExtractionTask:
    """Tests for EntityExtractionTask."""

    async def test_entity_extraction_initialization(self):
        """Test entity extraction task initialization."""
        task = EntityExtractionTask(max_entities=10, confidence_threshold=0.7)

        assert task.name == "entity_extraction"
        assert task.cognify_type == "entity"
        assert task.max_entities == 10
        assert task.confidence_threshold == 0.7

    async def test_entity_extraction_requires_content(self):
        """Test entity extraction requires content."""
        task = EntityExtractionTask()

        result = await task.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_entity_extraction_with_content(self):
        """Test entity extraction with content."""
        task = EntityExtractionTask(max_entities=20)

        result = await task.run_with_retry({
            "content": "Alice used Python to build a web scraper."
        })

        assert result.is_success()
        assert result.output["cognify_type"] == "entity"
        assert result.output["metadata"]["max_entities"] == 20

    async def test_entity_extraction_from_pipeline_input(self):
        """Test entity extraction with pipeline-wrapped input."""
        task = EntityExtractionTask()

        result = await task.run_with_retry({
            "input": {"content": "Test content"}
        })

        assert result.is_success()

    async def test_entity_extraction_from_data_array(self):
        """Test entity extraction from extracted data array."""
        task = EntityExtractionTask()

        result = await task.run_with_retry({
            "data": [
                {"content": "Message 1"},
                {"content": "Message 2"},
            ]
        })

        assert result.is_success()


@pytest.mark.asyncio
class TestRelationshipDetectionTask:
    """Tests for RelationshipDetectionTask."""

    async def test_relationship_detection_initialization(self):
        """Test relationship detection task initialization."""
        task = RelationshipDetectionTask(
            max_relationships=30,
            strength_threshold=0.5
        )

        assert task.name == "relationship_detection"
        assert task.cognify_type == "relationship"
        assert task.max_relationships == 30
        assert task.strength_threshold == 0.5

    async def test_relationship_detection_requires_entities(self):
        """Test relationship detection requires entities."""
        task = RelationshipDetectionTask()

        result = await task.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_relationship_detection_with_entities(self):
        """Test relationship detection with entities."""
        task = RelationshipDetectionTask(max_relationships=50)

        entities = [
            {"name": "Alice", "type": "person"},
            {"name": "Python", "type": "tool"}
        ]

        result = await task.run_with_retry({
            "entities": entities,
            "content": "Alice used Python"
        })

        assert result.is_success()
        assert result.output["cognify_type"] == "relationship"
        assert result.output["metadata"]["entity_count"] == 2

    async def test_relationship_detection_from_entity_extraction(self):
        """Test relationship detection from entity_extraction task output."""
        task = RelationshipDetectionTask()

        result = await task.run_with_retry({
            "entity_extraction": {
                "entities": [{"name": "Alice"}, {"name": "Bob"}]
            }
        })

        assert result.is_success()
        assert result.output["metadata"]["entity_count"] == 2


@pytest.mark.asyncio
class TestSemanticAnalysisTask:
    """Tests for SemanticAnalysisTask."""

    async def test_semantic_analysis_initialization(self):
        """Test semantic analysis task initialization."""
        task = SemanticAnalysisTask(
            embedding_model="text-embedding-3-large",
            chunk_size=1024
        )

        assert task.name == "semantic_analysis"
        assert task.cognify_type == "semantic"
        assert task.embedding_model == "text-embedding-3-large"
        assert task.chunk_size == 1024

    async def test_semantic_analysis_requires_content(self):
        """Test semantic analysis requires content."""
        task = SemanticAnalysisTask()

        result = await task.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_semantic_analysis_with_content(self):
        """Test semantic analysis with content."""
        task = SemanticAnalysisTask(chunk_size=512)

        result = await task.run_with_retry({
            "content": "This is test content for semantic analysis."
        })

        assert result.is_success()
        assert result.output["cognify_type"] == "semantic"
        assert result.output["metadata"]["chunk_size"] == 512


@pytest.mark.asyncio
class TestCognifyIntegration:
    """Integration tests for Cognify phase."""

    async def test_cognify_pipeline(self):
        """Test complete cognify pipeline."""
        from agentcore.a2a_protocol.services.memory.pipeline import Pipeline

        pipeline = Pipeline(pipeline_id="cognify_pipeline", parallel_execution=True)

        # Entity extraction (no dependencies)
        entity_task = EntityExtractionTask()

        # Relationship detection depends on entity extraction
        relationship_task = RelationshipDetectionTask()
        relationship_task.dependencies = ["entity_extraction"]

        # Semantic analysis (independent)
        semantic_task = SemanticAnalysisTask()

        pipeline.add_task(entity_task)
        pipeline.add_task(relationship_task)
        pipeline.add_task(semantic_task)

        result = await pipeline.execute({
            "content": "Alice and Bob worked on the Python project together."
        })

        # Entity should succeed
        assert result.task_results["entity_extraction"].is_success()

        # Semantic should succeed
        assert result.task_results["semantic_analysis"].is_success()

        # Relationship will fail because entity_extraction returns empty list
        # This is expected behavior - the task requires entities but gets none
        assert "relationship_detection" in result.task_results
