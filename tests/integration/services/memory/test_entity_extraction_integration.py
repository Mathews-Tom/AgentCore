"""
Integration Tests for Entity Extraction with ECL Pipeline

Tests entity extractor integration with the ECL pipeline framework
and Neo4j graph storage.

Component ID: MEM-015
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.a2a_protocol.models.memory import EntityType
from agentcore.a2a_protocol.services.memory import (
    EntityExtractor,
    Pipeline,
    task_registry,
)


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI-compatible LLM client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


class TestEntityExtractionPipelineIntegration:
    """Test entity extractor integration with ECL pipeline."""

    @pytest.mark.asyncio
    async def test_entity_extractor_in_pipeline(self, mock_llm_client):
        """Test entity extractor can be added to and executed in pipeline."""
        # Create pipeline
        pipeline = Pipeline(pipeline_id="entity_extraction_test")

        # Create entity extractor task
        extractor = EntityExtractor(
            llm_client=mock_llm_client,
        )

        # Mock LLM response
        llm_response = {
            "entities": [
                {"name": "redis", "type": "tool", "confidence": 0.95},
                {"name": "authentication", "type": "concept", "confidence": 0.90},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Add task to pipeline
        pipeline.add_task(extractor)

        # Execute pipeline
        input_data = {
            "content": "Using Redis for authentication caching",
            "memory_id": "mem-test-001",
        }

        result = await pipeline.execute(input_data)

        # Assertions
        assert result.is_success()
        assert "entity_extractor" in result.task_results
        task_result = result.task_results["entity_extractor"]
        assert task_result.is_success()

        # Check extracted entities
        output = task_result.output
        assert output is not None
        assert "normalized_entities" in output
        assert len(output["normalized_entities"]) == 2

        # Verify entity types
        entity_names = {e.entity_name for e in output["normalized_entities"]}
        assert "redis" in entity_names
        assert "authentication" in entity_names

    @pytest.mark.asyncio
    async def test_entity_extractor_from_registry(self, mock_llm_client):
        """Test retrieving entity extractor from task registry."""
        # Get extractor from registry
        extractor = task_registry.get_task(
            "EntityExtractor",
            llm_client=mock_llm_client,
        )

        assert extractor is not None
        assert isinstance(extractor, EntityExtractor)
        assert extractor.llm_client == mock_llm_client

    @pytest.mark.asyncio
    async def test_multiple_memory_batch_extraction(self, mock_llm_client):
        """Test extracting entities from multiple memories in sequence."""
        # Create pipeline with entity extractor
        pipeline = Pipeline(pipeline_id="batch_extraction")

        extractor = EntityExtractor(
            llm_client=mock_llm_client,
        )

        # Mock LLM responses for different memories
        responses = [
            {
                "entities": [
                    {"name": "docker", "type": "tool", "confidence": 0.95},
                ]
            },
            {
                "entities": [
                    {"name": "kubernetes", "type": "tool", "confidence": 0.95},
                ]
            },
        ]

        async def mock_create(**kwargs):
            # Return different response each time
            response_idx = mock_create.call_count
            mock_create.call_count += 1
            response_data = responses[min(response_idx, len(responses) - 1)]

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = str(response_data).replace("'", '"')
            return mock_response

        mock_create.call_count = 0
        mock_llm_client.chat.completions.create = mock_create

        pipeline.add_task(extractor)

        # Process first memory
        result1 = await pipeline.execute({
            "content": "Deploy with Docker",
            "memory_id": "mem-001",
        })

        assert result1.is_success()
        entities1 = result1.task_results["entity_extractor"].output["normalized_entities"]
        assert len(entities1) == 1
        assert entities1[0].entity_name == "docker"

        # Process second memory
        result2 = await pipeline.execute({
            "content": "Use Kubernetes for orchestration",
            "memory_id": "mem-002",
        })

        assert result2.is_success()
        entities2 = result2.task_results["entity_extractor"].output["normalized_entities"]
        assert len(entities2) == 1
        assert entities2[0].entity_name == "kubernetes"

    @pytest.mark.asyncio
    async def test_entity_deduplication_across_memories(self, mock_llm_client):
        """Test entity deduplication when processing multiple memories."""
        extractor = EntityExtractor(llm_client=mock_llm_client)

        # Mock LLM to extract same entity from different memories
        llm_response = {
            "entities": [
                {"name": "Redis", "type": "tool", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Extract from first memory
        result1 = await extractor.execute({
            "content": "Using Redis for caching",
            "memory_id": "mem-001",
        })

        entities1 = result1["normalized_entities"]
        assert len(entities1) == 1
        assert entities1[0].entity_name == "redis"
        assert "mem-001" in entities1[0].memory_refs

        # Extract from second memory, pass existing entities
        result2 = await extractor.execute({
            "content": "Redis handles session storage",
            "memory_id": "mem-002",
            "existing_entities": entities1,
        })

        entities2 = result2["normalized_entities"]
        assert len(entities2) == 1  # Should be deduplicated
        assert entities2[0].entity_name == "redis"
        # Should have both memory references
        assert "mem-001" in entities2[0].memory_refs
        assert "mem-002" in entities2[0].memory_refs

    @pytest.mark.asyncio
    async def test_pipeline_handles_extraction_error(self, mock_llm_client):
        """Test pipeline error handling when extraction fails."""
        pipeline = Pipeline(pipeline_id="error_test")

        extractor = EntityExtractor(
            llm_client=mock_llm_client,
            max_retries=0,  # Disable retry for faster test
        )

        # Mock LLM to fail
        mock_llm_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM API error")
        )

        pipeline.add_task(extractor)

        # Execute pipeline
        result = await pipeline.execute({
            "content": "Test content",
            "memory_id": "mem-test",
        })

        # Pipeline should fail gracefully
        assert not result.is_success()
        assert "entity_extractor" in result.task_results
        assert result.task_results["entity_extractor"].is_failure()
        assert result.task_results["entity_extractor"].error is not None


class TestEntityExtractionWithFallback:
    """Test fallback extraction when LLM not available."""

    @pytest.mark.asyncio
    async def test_pipeline_with_fallback_extraction(self):
        """Test pipeline works with fallback extraction (no LLM)."""
        pipeline = Pipeline(pipeline_id="fallback_test")

        # Create extractor without LLM client
        extractor = EntityExtractor(
            llm_client=None,
        )

        pipeline.add_task(extractor)

        # Execute with content containing recognizable entities
        result = await pipeline.execute({
            "content": "Using Redis, PostgreSQL, and Docker for deployment",
            "memory_id": "mem-fallback-001",
        })

        # Should succeed with fallback
        assert result.is_success()
        task_result = result.task_results["entity_extractor"]
        assert task_result.is_success()

        entities = task_result.output["normalized_entities"]
        assert len(entities) > 0

        # Should extract at least some tools
        entity_names = {e.entity_name for e in entities}
        assert any(name in entity_names for name in ["redis", "postgresql", "docker"])


class TestEntityExtractionMetrics:
    """Test extraction metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_extraction_timing_metrics(self, mock_llm_client):
        """Test extraction tracks timing metrics."""
        extractor = EntityExtractor(llm_client=mock_llm_client)

        # Mock LLM response
        llm_response = {"entities": []}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Execute with retry tracking
        task_result = await extractor.run_with_retry({
            "content": "Test content",
        })

        # Should have timing information
        assert task_result.started_at is not None
        assert task_result.completed_at is not None
        assert task_result.execution_time_ms is not None
        assert task_result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_extraction_metadata_tracking(self, mock_llm_client):
        """Test extraction metadata is properly tracked."""
        extractor = EntityExtractor(
            llm_client=mock_llm_client,
            confidence_threshold=0.6,
        )

        # Mock LLM response with varied confidence
        llm_response = {
            "entities": [
                {"name": "high", "type": "tool", "confidence": 0.95},
                {"name": "medium", "type": "tool", "confidence": 0.70},
                {"name": "low", "type": "tool", "confidence": 0.40},  # Below threshold
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.execute({
            "content": "Test content",
            "memory_id": "mem-metrics",
        })

        metadata = result["extraction_metadata"]

        # Verify metadata accuracy
        assert metadata["total_extracted"] == 3
        assert metadata["after_confidence_filter"] == 2  # Only high and medium
        assert metadata["after_deduplication"] == 2
        assert metadata["confidence_threshold"] == 0.6
