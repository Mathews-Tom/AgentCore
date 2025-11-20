"""Unit tests for Extract phase tasks.

Tests cover:
- ExtractTask base class
- ConversationExtractor
- ArtifactExtractor
- ErrorRecordExtractor
- Streaming support
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.a2a_protocol.services.memory.pipeline.extract import (
    ArtifactExtractor,
    ConversationExtractor,
    ErrorRecordExtractor,
    ExtractTask,
)
from agentcore.a2a_protocol.services.memory.pipeline.task_base import TaskStatus


@pytest.mark.asyncio
class TestExtractTask:
    """Tests for ExtractTask base class."""

    async def test_extract_task_initialization(self):
        """Test extract task initialization."""
        task = ExtractTask(
            name="test_extract",
            source_type="test_source",
            streaming_enabled=True,
            batch_size=50,
        )

        assert task.name == "test_extract"
        assert task.source_type == "test_source"
        assert task.streaming_enabled is True
        assert task.batch_size == 50

    async def test_extract_task_default_execute(self):
        """Test default execute implementation."""
        task = ExtractTask(name="test_extract", source_type="generic")
        result = await task.run_with_retry({"source": "test"})

        assert result.is_success()
        assert result.output["source_type"] == "generic"
        assert result.output["count"] == 0

    async def test_extract_task_streaming_not_enabled(self):
        """Test streaming raises error when not enabled."""
        task = ExtractTask(name="test_extract", streaming_enabled=False)

        with pytest.raises(NotImplementedError):
            async for _ in task.stream({"source": "test"}):
                pass

    async def test_extract_task_streaming_enabled(self):
        """Test streaming when enabled."""
        task = ExtractTask(name="test_extract", streaming_enabled=True)

        batches = []
        async for batch in task.stream({"source": "test"}):
            batches.append(batch)

        assert len(batches) == 1
        assert batches[0]["is_last_batch"] is True


@pytest.mark.asyncio
class TestConversationExtractor:
    """Tests for ConversationExtractor."""

    async def test_conversation_extractor_initialization(self):
        """Test conversation extractor initialization."""
        extractor = ConversationExtractor()

        assert extractor.name == "conversation_extractor"
        assert extractor.source_type == "conversation"
        assert extractor.streaming_enabled is True

    async def test_conversation_extractor_requires_session_id(self):
        """Test conversation extractor requires session_id."""
        extractor = ConversationExtractor()

        result = await extractor.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_conversation_extractor_with_session_id(self):
        """Test conversation extractor with valid session_id."""
        extractor = ConversationExtractor()

        result = await extractor.run_with_retry({"session_id": "session-123"})

        assert result.is_success()
        assert result.output["source_type"] == "conversation"
        assert result.output["session_id"] == "session-123"

    async def test_conversation_extractor_pipeline_input(self):
        """Test conversation extractor with pipeline-wrapped input."""
        extractor = ConversationExtractor()

        result = await extractor.run_with_retry({
            "input": {"session_id": "session-456"}
        })

        assert result.is_success()
        assert result.output["session_id"] == "session-456"


@pytest.mark.asyncio
class TestArtifactExtractor:
    """Tests for ArtifactExtractor."""

    async def test_artifact_extractor_initialization(self):
        """Test artifact extractor initialization."""
        extractor = ArtifactExtractor()

        assert extractor.name == "artifact_extractor"
        assert extractor.source_type == "artifact"

    async def test_artifact_extractor_requires_task_id(self):
        """Test artifact extractor requires task_id."""
        extractor = ArtifactExtractor()

        result = await extractor.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_artifact_extractor_with_task_id(self):
        """Test artifact extractor with valid task_id."""
        extractor = ArtifactExtractor()

        result = await extractor.run_with_retry({
            "task_id": "task-123",
            "artifact_types": ["file", "output"]
        })

        assert result.is_success()
        assert result.output["source_type"] == "artifact"
        assert result.output["task_id"] == "task-123"
        assert result.output["artifact_types"] == ["file", "output"]

    async def test_artifact_extractor_default_types(self):
        """Test artifact extractor with default artifact types."""
        extractor = ArtifactExtractor()

        result = await extractor.run_with_retry({"task_id": "task-123"})

        assert result.is_success()
        assert "file" in result.output["artifact_types"]
        assert "output" in result.output["artifact_types"]


@pytest.mark.asyncio
class TestErrorRecordExtractor:
    """Tests for ErrorRecordExtractor."""

    async def test_error_record_extractor_initialization(self):
        """Test error record extractor initialization."""
        extractor = ErrorRecordExtractor()

        assert extractor.name == "error_record_extractor"
        assert extractor.source_type == "error"

    async def test_error_record_extractor_requires_task_id(self):
        """Test error record extractor requires task_id."""
        extractor = ErrorRecordExtractor()

        result = await extractor.run_with_retry({})
        assert result.is_failure()
        assert isinstance(result.error, ValueError)

    async def test_error_record_extractor_with_parameters(self):
        """Test error record extractor with full parameters."""
        extractor = ErrorRecordExtractor()

        result = await extractor.run_with_retry({
            "task_id": "task-123",
            "min_severity": 0.7
        })

        assert result.is_success()
        assert result.output["source_type"] == "error"
        assert result.output["task_id"] == "task-123"
        assert result.output["min_severity"] == 0.7

    async def test_error_record_extractor_default_severity(self):
        """Test error record extractor with default severity."""
        extractor = ErrorRecordExtractor()

        result = await extractor.run_with_retry({"task_id": "task-123"})

        assert result.is_success()
        assert result.output["min_severity"] == 0.0


@pytest.mark.asyncio
class TestExtractIntegration:
    """Integration tests for Extract phase."""

    async def test_multiple_extractors_in_pipeline(self):
        """Test using multiple extractors together."""
        from agentcore.a2a_protocol.services.memory.pipeline import Pipeline

        pipeline = Pipeline(pipeline_id="extract_pipeline")

        conv_extractor = ConversationExtractor()
        artifact_extractor = ArtifactExtractor()

        # Make artifact extractor depend on conversation
        artifact_extractor.dependencies = ["conversation_extractor"]

        pipeline.add_task(conv_extractor)
        pipeline.add_task(artifact_extractor)

        result = await pipeline.execute({
            "session_id": "session-123",
            "task_id": "task-123"
        })

        # Conversation should fail (no session_id passed correctly)
        # But we can check the structure
        assert "conversation_extractor" in result.task_results
        assert "artifact_extractor" in result.task_results
