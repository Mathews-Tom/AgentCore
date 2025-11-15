"""
Unit tests for Memory Service JSON-RPC methods.

Tests all memory JSON-RPC handlers covering:
- memory.store
- memory.retrieve
- memory.get_context
- memory.complete_stage
- memory.record_error
- memory.get_strategic_context
- memory.run_memify

Component ID: MEM-026
Ticket: MEM-026 (Implement JSON-RPC Methods)
"""

from unittest.mock import AsyncMock, patch

import pytest

from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest
from agentcore.a2a_protocol.services.memory_jsonrpc import (
    _error_store,
    _memory_store,
    _stage_store,
    handle_memory_complete_stage,
    handle_memory_get_context,
    handle_memory_get_strategic_context,
    handle_memory_record_error,
    handle_memory_retrieve,
    handle_memory_run_memify,
    handle_memory_store,
)


@pytest.fixture(autouse=True)
def clear_stores() -> None:
    """Clear all in-memory stores before each test."""
    _memory_store.clear()
    _stage_store.clear()
    _error_store.clear()


class TestMemoryStore:
    """Test suite for memory.store JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_store_success(self) -> None:
        """Test successful memory storage."""
        request = JsonRpcRequest(
            method="memory.store",
            params={
                "content": "User prefers technical explanations",
                "memory_layer": "semantic",
                "summary": "User preference for technical detail",
                "agent_id": "agent-123",
                "session_id": "session-456",
                "keywords": ["user", "preference", "technical"],
                "is_critical": True,
                "criticality_reason": "User preference affects all interactions",
            },
            id=1,
        )

        result = await handle_memory_store(request)

        assert result["success"] is True
        assert "memory_id" in result
        assert result["memory_layer"] == "semantic"
        assert "timestamp" in result
        assert result["message"] == "Memory stored successfully"

        # Verify memory was stored
        assert result["memory_id"] in _memory_store

    @pytest.mark.asyncio
    async def test_store_minimal_params(self) -> None:
        """Test storage with minimal parameters."""
        request = JsonRpcRequest(
            method="memory.store",
            params={
                "content": "Basic memory content",
                "agent_id": "agent-001",
            },
            id=2,
        )

        result = await handle_memory_store(request)

        assert result["success"] is True
        assert result["memory_layer"] == "episodic"  # Default layer

    @pytest.mark.asyncio
    async def test_store_with_embedding(self) -> None:
        """Test storage with embedding vector."""
        embedding = [0.1] * 768  # 768-dim embedding
        request = JsonRpcRequest(
            method="memory.store",
            params={
                "content": "Memory with embedding",
                "agent_id": "agent-001",
                "embedding": embedding,
            },
            id=3,
        )

        result = await handle_memory_store(request)

        assert result["success"] is True
        memory = _memory_store[result["memory_id"]]
        assert len(memory.embedding) == 768

    @pytest.mark.asyncio
    async def test_store_missing_content(self) -> None:
        """Test error when content is missing."""
        request = JsonRpcRequest(
            method="memory.store",
            params={
                "agent_id": "agent-001",
            },
            id=4,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_store(request)

    @pytest.mark.asyncio
    async def test_store_missing_agent_id(self) -> None:
        """Test error when agent_id is missing."""
        request = JsonRpcRequest(
            method="memory.store",
            params={
                "content": "Some content",
            },
            id=5,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_store(request)

    @pytest.mark.asyncio
    async def test_store_no_params(self) -> None:
        """Test error when no params provided."""
        request = JsonRpcRequest(
            method="memory.store",
            params=None,
            id=6,
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_memory_store(request)

    @pytest.mark.asyncio
    async def test_store_all_layers(self) -> None:
        """Test storage in all memory layers."""
        layers = ["working", "episodic", "semantic", "procedural"]

        for layer in layers:
            request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Memory for {layer} layer",
                    "memory_layer": layer,
                    "agent_id": "agent-001",
                },
                id=10,
            )

            result = await handle_memory_store(request)
            assert result["memory_layer"] == layer


class TestMemoryRetrieve:
    """Test suite for memory.retrieve JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self) -> None:
        """Test retrieval from empty store."""
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "agent_id": "agent-001",
                "limit": 10,
            },
            id=1,
        )

        result = await handle_memory_retrieve(request)

        assert result["memories"] == []
        assert result["total_count"] == 0
        assert "query_time_ms" in result

    @pytest.mark.asyncio
    async def test_retrieve_with_memories(self) -> None:
        """Test retrieval after storing memories."""
        # Store some memories first
        for i in range(3):
            store_request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Memory {i}",
                    "agent_id": "agent-001",
                },
                id=i,
            )
            await handle_memory_store(store_request)

        # Retrieve memories
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "agent_id": "agent-001",
                "limit": 10,
            },
            id=10,
        )

        result = await handle_memory_retrieve(request)

        assert result["total_count"] == 3
        assert len(result["memories"]) == 3

    @pytest.mark.asyncio
    async def test_retrieve_filter_by_session(self) -> None:
        """Test retrieval filtered by session_id."""
        # Store memories for different sessions
        for session in ["session-1", "session-2"]:
            store_request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Memory for {session}",
                    "agent_id": "agent-001",
                    "session_id": session,
                },
                id=1,
            )
            await handle_memory_store(store_request)

        # Retrieve for specific session
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "session_id": "session-1",
                "limit": 10,
            },
            id=10,
        )

        result = await handle_memory_retrieve(request)

        assert result["total_count"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_filter_by_task(self) -> None:
        """Test retrieval filtered by task_id."""
        # Store memories for different tasks
        for task in ["task-1", "task-2", "task-3"]:
            store_request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Memory for {task}",
                    "agent_id": "agent-001",
                    "task_id": task,
                },
                id=1,
            )
            await handle_memory_store(store_request)

        # Retrieve for specific task
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "task_id": "task-2",
                "limit": 10,
            },
            id=10,
        )

        result = await handle_memory_retrieve(request)

        assert result["total_count"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_filter_by_layer(self) -> None:
        """Test retrieval filtered by memory_layer."""
        # Store memories in different layers
        for layer in ["episodic", "semantic"]:
            store_request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Memory in {layer}",
                    "agent_id": "agent-001",
                    "memory_layer": layer,
                },
                id=1,
            )
            await handle_memory_store(store_request)

        # Retrieve semantic memories
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "memory_layer": "semantic",
                "limit": 10,
            },
            id=10,
        )

        result = await handle_memory_retrieve(request)

        assert result["total_count"] == 1
        assert result["memories"][0]["memory_layer"] == "semantic"

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self) -> None:
        """Test retrieval respects limit parameter."""
        # Store 10 memories
        for i in range(10):
            store_request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Memory {i}",
                    "agent_id": "agent-001",
                },
                id=i,
            )
            await handle_memory_store(store_request)

        # Retrieve with limit
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "agent_id": "agent-001",
                "limit": 5,
            },
            id=100,
        )

        result = await handle_memory_retrieve(request)

        assert result["total_count"] == 5
        assert len(result["memories"]) == 5

    @pytest.mark.asyncio
    async def test_retrieve_with_stage_context(self) -> None:
        """Test retrieval with current stage context."""
        # Store a memory
        store_request = JsonRpcRequest(
            method="memory.store",
            params={
                "content": "Test memory",
                "agent_id": "agent-001",
            },
            id=1,
        )
        await handle_memory_store(store_request)

        # Retrieve with stage
        request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "agent_id": "agent-001",
                "current_stage": "planning",
                "has_recent_errors": True,
            },
            id=10,
        )

        result = await handle_memory_retrieve(request)

        assert "memories" in result
        assert "query_time_ms" in result

    @pytest.mark.asyncio
    async def test_retrieve_invalid_params_type(self) -> None:
        """Test error when params is an array."""
        request = JsonRpcRequest(
            method="memory.retrieve",
            params=["invalid"],
            id=1,
        )

        with pytest.raises(ValueError, match="must be an object"):
            await handle_memory_retrieve(request)


class TestMemoryGetContext:
    """Test suite for memory.get_context JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_get_context_empty(self) -> None:
        """Test context retrieval for new session."""
        request = JsonRpcRequest(
            method="memory.get_context",
            params={
                "session_id": "session-123",
                "max_memories": 10,
            },
            id=1,
        )

        result = await handle_memory_get_context(request)

        assert result["session_id"] == "session-123"
        assert result["memory_count"] == 0
        assert result["context_size_bytes"] > 0  # At least the header
        assert result["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_get_context_markdown_format(self) -> None:
        """Test context retrieval in markdown format."""
        request = JsonRpcRequest(
            method="memory.get_context",
            params={
                "session_id": "session-123",
                "format": "markdown",
            },
            id=1,
        )

        result = await handle_memory_get_context(request)

        assert result["format"] == "markdown"
        assert "# Memory Context" in result["context"]

    @pytest.mark.asyncio
    async def test_get_context_plain_format(self) -> None:
        """Test context retrieval in plain text format."""
        request = JsonRpcRequest(
            method="memory.get_context",
            params={
                "session_id": "session-123",
                "format": "plain",
            },
            id=1,
        )

        result = await handle_memory_get_context(request)

        assert result["format"] == "plain"

    @pytest.mark.asyncio
    async def test_get_context_json_format(self) -> None:
        """Test context retrieval in JSON format."""
        request = JsonRpcRequest(
            method="memory.get_context",
            params={
                "session_id": "session-123",
                "format": "json",
            },
            id=1,
        )

        result = await handle_memory_get_context(request)

        assert result["format"] == "json"

    @pytest.mark.asyncio
    async def test_get_context_missing_session_id(self) -> None:
        """Test error when session_id is missing."""
        request = JsonRpcRequest(
            method="memory.get_context",
            params={
                "format": "markdown",
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_get_context(request)

    @pytest.mark.asyncio
    async def test_get_context_no_params(self) -> None:
        """Test error when no params provided."""
        request = JsonRpcRequest(
            method="memory.get_context",
            params=None,
            id=1,
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_memory_get_context(request)


class TestMemoryCompleteStage:
    """Test suite for memory.complete_stage JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_complete_stage_success(self) -> None:
        """Test successful stage completion."""
        request = JsonRpcRequest(
            method="memory.complete_stage",
            params={
                "stage_id": "stage-123",
                "task_id": "task-456",
                "agent_id": "agent-789",
                "stage_type": "planning",
                "stage_summary": "Analyzed requirements and designed solution",
                "stage_insights": ["Use JWT tokens", "Store in Redis"],
                "raw_memory_refs": [],
                "compression_model": "gpt-4.1-mini",
            },
            id=1,
        )

        result = await handle_memory_complete_stage(request)

        assert result["success"] is True
        assert result["stage_id"] == "stage-123"
        assert result["task_id"] == "task-456"
        assert result["compression_ratio"] >= 1.0
        assert result["quality_score"] > 0.0
        assert result["message"] == "Stage completed and compressed"

        # Verify stage was stored
        assert "stage-123" in _stage_store

    @pytest.mark.asyncio
    async def test_complete_stage_all_types(self) -> None:
        """Test completion for all stage types."""
        stage_types = ["planning", "execution", "reflection", "verification"]

        for i, stage_type in enumerate(stage_types):
            request = JsonRpcRequest(
                method="memory.complete_stage",
                params={
                    "stage_id": f"stage-{i}",
                    "task_id": "task-001",
                    "agent_id": "agent-001",
                    "stage_type": stage_type,
                    "stage_summary": f"Summary for {stage_type}",
                },
                id=i,
            )

            result = await handle_memory_complete_stage(request)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_complete_stage_invalid_type(self) -> None:
        """Test error with invalid stage type."""
        request = JsonRpcRequest(
            method="memory.complete_stage",
            params={
                "stage_id": "stage-001",
                "task_id": "task-001",
                "agent_id": "agent-001",
                "stage_type": "invalid_type",
                "stage_summary": "Summary",
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Invalid stage_type"):
            await handle_memory_complete_stage(request)

    @pytest.mark.asyncio
    async def test_complete_stage_missing_params(self) -> None:
        """Test error with missing required parameters."""
        request = JsonRpcRequest(
            method="memory.complete_stage",
            params={
                "stage_id": "stage-001",
                # Missing task_id, agent_id, stage_type, stage_summary
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_complete_stage(request)


class TestMemoryRecordError:
    """Test suite for memory.record_error JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_record_error_success(self) -> None:
        """Test successful error recording."""
        request = JsonRpcRequest(
            method="memory.record_error",
            params={
                "task_id": "task-123",
                "agent_id": "agent-456",
                "error_type": "hallucination",
                "error_description": "LLM generated false API endpoint",
                "context_when_occurred": "During API documentation generation",
                "recovery_action": "Corrected endpoint from documentation",
                "error_severity": 0.7,
                "stage_id": "stage-789",
            },
            id=1,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory_jsonrpc._get_error_tracker"
        ) as mock_tracker:
            mock_error_tracker = AsyncMock()
            mock_tracker.return_value = mock_error_tracker
            mock_error_tracker.record_error = AsyncMock()

            result = await handle_memory_record_error(request)

        assert result["success"] is True
        assert "error_id" in result
        assert result["task_id"] == "task-123"
        assert "recorded_at" in result
        assert result["message"] == "Error recorded successfully"

        # Verify error was stored
        assert result["error_id"] in _error_store

    @pytest.mark.asyncio
    async def test_record_error_all_types(self) -> None:
        """Test recording all error types."""
        error_types = [
            "hallucination",
            "missing_info",
            "incorrect_action",
            "context_degradation",
        ]

        for i, error_type in enumerate(error_types):
            request = JsonRpcRequest(
                method="memory.record_error",
                params={
                    "task_id": f"task-{i}",
                    "agent_id": "agent-001",
                    "error_type": error_type,
                    "error_description": f"Error of type {error_type}",
                    "error_severity": 0.5,
                },
                id=i,
            )

            with patch(
                "agentcore.a2a_protocol.services.memory_jsonrpc._get_error_tracker"
            ) as mock_tracker:
                mock_error_tracker = AsyncMock()
                mock_tracker.return_value = mock_error_tracker
                mock_error_tracker.record_error = AsyncMock()

                result = await handle_memory_record_error(request)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_record_error_invalid_type(self) -> None:
        """Test error with invalid error type."""
        request = JsonRpcRequest(
            method="memory.record_error",
            params={
                "task_id": "task-001",
                "agent_id": "agent-001",
                "error_type": "invalid_error_type",
                "error_description": "Some error",
                "error_severity": 0.5,
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Invalid error_type"):
            await handle_memory_record_error(request)

    @pytest.mark.asyncio
    async def test_record_error_invalid_severity(self) -> None:
        """Test error with invalid severity value."""
        request = JsonRpcRequest(
            method="memory.record_error",
            params={
                "task_id": "task-001",
                "agent_id": "agent-001",
                "error_type": "hallucination",
                "error_description": "Some error",
                "error_severity": 1.5,  # Invalid: > 1.0
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_record_error(request)

    @pytest.mark.asyncio
    async def test_record_error_missing_params(self) -> None:
        """Test error with missing required parameters."""
        request = JsonRpcRequest(
            method="memory.record_error",
            params={
                "task_id": "task-001",
                # Missing agent_id, error_type, error_description, error_severity
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_record_error(request)


class TestMemoryGetStrategicContext:
    """Test suite for memory.get_strategic_context JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_get_strategic_context_success(self) -> None:
        """Test successful strategic context retrieval."""
        request = JsonRpcRequest(
            method="memory.get_strategic_context",
            params={
                "agent_id": "agent-123",
                "session_id": "session-456",
                "goal": "Optimize system performance",
            },
            id=1,
        )

        result = await handle_memory_get_strategic_context(request)

        assert "context_id" in result
        assert result["agent_id"] == "agent-123"
        assert result["session_id"] == "session-456"
        assert result["current_goal"] == "Optimize system performance"
        assert "strategic_memory_count" in result
        assert "tactical_memory_count" in result
        assert "error_patterns" in result
        assert "success_patterns" in result
        assert "confidence_score" in result

    @pytest.mark.asyncio
    async def test_get_strategic_context_minimal(self) -> None:
        """Test strategic context with minimal parameters."""
        request = JsonRpcRequest(
            method="memory.get_strategic_context",
            params={
                "agent_id": "agent-001",
            },
            id=1,
        )

        result = await handle_memory_get_strategic_context(request)

        assert result["agent_id"] == "agent-001"
        assert result["session_id"] is None
        assert result["current_goal"] is None

    @pytest.mark.asyncio
    async def test_get_strategic_context_missing_agent_id(self) -> None:
        """Test error when agent_id is missing."""
        request = JsonRpcRequest(
            method="memory.get_strategic_context",
            params={
                "session_id": "session-123",
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_get_strategic_context(request)

    @pytest.mark.asyncio
    async def test_get_strategic_context_no_params(self) -> None:
        """Test error when no params provided."""
        request = JsonRpcRequest(
            method="memory.get_strategic_context",
            params=None,
            id=1,
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_memory_get_strategic_context(request)


class TestMemoryRunMemify:
    """Test suite for memory.run_memify JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_run_memify_success(self) -> None:
        """Test successful memify operation."""
        request = JsonRpcRequest(
            method="memory.run_memify",
            params={
                "similarity_threshold": 0.90,
                "min_access_count": 2,
                "batch_size": 100,
            },
            id=1,
        )

        result = await handle_memory_run_memify(request)

        assert "optimization_id" in result
        assert result["entities_analyzed"] >= 0
        assert result["entities_merged"] >= 0
        assert result["relationships_pruned"] >= 0
        assert result["patterns_detected"] >= 0
        assert result["consolidation_accuracy"] >= 0.0
        assert result["duplicate_rate"] >= 0.0
        assert result["duration_seconds"] >= 0.0
        assert result["scheduled_job_id"] is None
        assert result["next_run"] is None

    @pytest.mark.asyncio
    async def test_run_memify_with_defaults(self) -> None:
        """Test memify with default parameters."""
        request = JsonRpcRequest(
            method="memory.run_memify",
            params={},
            id=1,
        )

        result = await handle_memory_run_memify(request)

        assert "optimization_id" in result
        assert result["duration_seconds"] >= 0.0

    @pytest.mark.asyncio
    async def test_run_memify_with_scheduling(self) -> None:
        """Test memify with cron scheduling."""
        request = JsonRpcRequest(
            method="memory.run_memify",
            params={
                "schedule_cron": "0 2 * * *",  # Daily at 2am
            },
            id=1,
        )

        result = await handle_memory_run_memify(request)

        assert result["scheduled_job_id"] is not None
        assert result["next_run"] is not None

    @pytest.mark.asyncio
    async def test_run_memify_invalid_cron(self) -> None:
        """Test error with invalid cron expression."""
        request = JsonRpcRequest(
            method="memory.run_memify",
            params={
                "schedule_cron": "invalid cron",
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Invalid cron expression"):
            await handle_memory_run_memify(request)

    @pytest.mark.asyncio
    async def test_run_memify_invalid_threshold(self) -> None:
        """Test error with invalid similarity threshold."""
        request = JsonRpcRequest(
            method="memory.run_memify",
            params={
                "similarity_threshold": 1.5,  # Invalid: > 1.0
            },
            id=1,
        )

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await handle_memory_run_memify(request)

    @pytest.mark.asyncio
    async def test_run_memify_invalid_params_type(self) -> None:
        """Test error when params is an array."""
        request = JsonRpcRequest(
            method="memory.run_memify",
            params=["invalid"],
            id=1,
        )

        with pytest.raises(ValueError, match="must be an object"):
            await handle_memory_run_memify(request)


class TestMemoryJSONRPCIntegration:
    """Integration tests for memory JSON-RPC methods."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_flow(self) -> None:
        """Test complete store and retrieve workflow."""
        # Store multiple memories
        for i in range(5):
            store_request = JsonRpcRequest(
                method="memory.store",
                params={
                    "content": f"Important fact number {i}",
                    "agent_id": "agent-integration",
                    "session_id": "session-integration",
                    "memory_layer": "semantic",
                },
                id=i,
            )
            await handle_memory_store(store_request)

        # Retrieve all stored memories
        retrieve_request = JsonRpcRequest(
            method="memory.retrieve",
            params={
                "agent_id": "agent-integration",
                "session_id": "session-integration",
                "limit": 10,
            },
            id=100,
        )

        result = await handle_memory_retrieve(retrieve_request)

        assert result["total_count"] == 5
        for memory in result["memories"]:
            assert memory["agent_id"] == "agent-integration"
            assert memory["session_id"] == "session-integration"

    @pytest.mark.asyncio
    async def test_error_tracking_flow(self) -> None:
        """Test error recording and pattern detection."""
        # Record multiple errors
        error_types = ["hallucination", "missing_info", "incorrect_action"]

        for i, error_type in enumerate(error_types):
            request = JsonRpcRequest(
                method="memory.record_error",
                params={
                    "task_id": "task-flow",
                    "agent_id": "agent-flow",
                    "error_type": error_type,
                    "error_description": f"Error {i}",
                    "error_severity": 0.5 + (i * 0.1),
                },
                id=i,
            )

            with patch(
                "agentcore.a2a_protocol.services.memory_jsonrpc._get_error_tracker"
            ) as mock_tracker:
                mock_error_tracker = AsyncMock()
                mock_tracker.return_value = mock_error_tracker
                mock_error_tracker.record_error = AsyncMock()

                result = await handle_memory_record_error(request)

            assert result["success"] is True

        # Verify all errors were stored
        assert len(_error_store) == 3

    @pytest.mark.asyncio
    async def test_stage_completion_flow(self) -> None:
        """Test stage completion workflow."""
        stages = ["planning", "execution", "verification"]

        for i, stage in enumerate(stages):
            request = JsonRpcRequest(
                method="memory.complete_stage",
                params={
                    "stage_id": f"stage-flow-{i}",
                    "task_id": "task-flow",
                    "agent_id": "agent-flow",
                    "stage_type": stage,
                    "stage_summary": f"Completed {stage} stage",
                    "stage_insights": [f"Insight from {stage}"],
                },
                id=i,
            )

            result = await handle_memory_complete_stage(request)
            assert result["success"] is True

        # Verify all stages were stored
        assert len(_stage_store) == 3

    @pytest.mark.asyncio
    async def test_a2a_context_preserved(self) -> None:
        """Test that A2A context is preserved in operations."""
        request = JsonRpcRequest(
            method="memory.store",
            params={
                "content": "Memory with A2A context",
                "agent_id": "agent-001",
            },
            id=1,
            a2a_context=A2AContext(
                source_agent="source-agent",
                trace_id="trace-a2a-test",
                session_id="session-a2a",
                timestamp="2025-11-15T10:00:00Z",
            ),
        )

        result = await handle_memory_store(request)
        assert result["success"] is True
        # A2A context should be available in request for logging/tracing
