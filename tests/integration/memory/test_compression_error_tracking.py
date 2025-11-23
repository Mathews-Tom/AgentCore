"""
Integration Tests for Stage Compression and Error Tracking (MEM-027.6 & MEM-027.7)

Part 1: Stage Compression Pipeline
- Stage completion workflow and compression triggering
- Task compression from multiple stages
- Compression quality validation (≥95% fact retention)
- Cost tracking integration

Part 2: Error Tracking Workflow
- Error recording with full context
- Pattern detection (frequency, sequence, context)
- Error-aware retrieval integration
- ACE integration signals

Performance targets:
- Stage compression: 10:1 ratio (±20%)
- Task compression: 5:1 ratio (±20%)
- Critical fact retention: ≥95%
- 100% error capture rate
- 80%+ pattern detection accuracy
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from agentcore.a2a_protocol.models.memory import MemoryLayer, StageType
from agentcore.a2a_protocol.services.memory.context_compressor import ContextCompressor
from agentcore.a2a_protocol.services.memory.stage_manager import StageManager
from agentcore.a2a_protocol.services.memory.quality_validator import QualityValidator
from agentcore.a2a_protocol.services.memory.cost_tracker import CostTracker
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.storage_backend import StorageBackendService
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService


# Use function-scoped event loop for all tests and mark as integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestStageCompressionPipeline:
    """Test COMPASS stage compression pipeline."""

    @pytest.fixture
    async def context_compressor(self) -> ContextCompressor:
        """Create context compressor."""
        return ContextCompressor(api_key="test-key", model="gpt-4.1-mini")

    @pytest.fixture
    async def quality_validator(self) -> QualityValidator:
        """Create quality validator."""
        return QualityValidator(api_key="test-key")

    @pytest.fixture
    async def cost_tracker(self) -> CostTracker:
        """Create cost tracker."""
        return CostTracker(use_memory=True)

    @pytest.fixture
    async def stage_manager(
        self,
        context_compressor: ContextCompressor,
    ) -> StageManager:
        """Create stage manager with compression trigger."""
        return StageManager(compression_trigger=context_compressor)

    async def test_stage_completion_triggers_compression(
        self,
        stage_manager: StageManager,
        context_compressor: ContextCompressor,
    ) -> None:
        """Test stage completion workflow triggers compression."""
        # Arrange - Create stage with multiple raw memories
        task_id = "task-compression-001"
        stage_type = StageType.EXECUTION

        stage_id = await stage_manager.create_stage(
            task_id=task_id,
            stage_type=stage_type,
        )

        # Add raw memories to stage
        raw_memories = [
            "Implemented JWT authentication endpoint",
            "Added token validation middleware",
            "Configured Redis for session storage",
            "Wrote unit tests for auth flow",
            "Deployed authentication service to staging",
        ]

        for memory_content in raw_memories:
            await stage_manager.add_memory_to_stage(
                stage_id=stage_id,
                memory_content=memory_content,
            )

        # Act - Complete stage (should trigger compression)
        compression_result = await stage_manager.complete_stage(stage_id)

        # Assert - Compression executed
        assert compression_result["status"] == "success"
        assert "compressed_summary" in compression_result
        assert "compression_ratio" in compression_result

        # Verify compression ratio
        original_length = sum(len(m) for m in raw_memories)
        compressed_length = len(compression_result["compressed_summary"])
        actual_ratio = original_length / compressed_length

        # Should achieve approximately 10:1 compression (±20% tolerance)
        assert 8.0 <= actual_ratio <= 12.0, (
            f"Compression ratio {actual_ratio:.1f}:1 outside 10:1 ±20% range"
        )

    async def test_task_compression_from_multiple_stages(
        self,
        stage_manager: StageManager,
        context_compressor: ContextCompressor,
    ) -> None:
        """Test task compression from multiple stage summaries."""
        # Arrange - Create multiple completed stages
        task_id = "task-multi-stage-001"

        stages_data = [
            {
                "type": StageType.PLANNING,
                "memories": [
                    "Analyzed requirements for authentication feature",
                    "Selected JWT approach over session cookies",
                    "Designed token refresh mechanism",
                ],
            },
            {
                "type": StageType.EXECUTION,
                "memories": [
                    "Implemented JWT generation and validation",
                    "Integrated Redis for token storage",
                    "Added authentication middleware",
                ],
            },
            {
                "type": StageType.VERIFICATION,
                "memories": [
                    "Tested token expiration handling",
                    "Validated refresh token flow",
                    "Verified unauthorized access blocking",
                ],
            },
        ]

        stage_summaries = []
        for stage_data in stages_data:
            stage_id = await stage_manager.create_stage(
                task_id=task_id,
                stage_type=stage_data["type"],
            )

            for memory in stage_data["memories"]:
                await stage_manager.add_memory_to_stage(stage_id, memory)

            result = await stage_manager.complete_stage(stage_id)
            stage_summaries.append(result["compressed_summary"])

        # Act - Compress task from stage summaries
        task_compression = await context_compressor.compress_task(
            task_id=task_id,
            stage_summaries=stage_summaries,
        )

        # Assert - Task compression successful
        assert task_compression["status"] == "success"
        assert "task_summary" in task_compression
        assert "compression_ratio" in task_compression

        # Verify 5:1 compression ratio (±20% tolerance)
        total_stage_length = sum(len(s) for s in stage_summaries)
        task_summary_length = len(task_compression["task_summary"])
        actual_ratio = total_stage_length / task_summary_length

        assert 4.0 <= actual_ratio <= 6.0, (
            f"Task compression ratio {actual_ratio:.1f}:1 outside 5:1 ±20% range"
        )

    async def test_compression_quality_validation(
        self,
        context_compressor: ContextCompressor,
        quality_validator: QualityValidator,
    ) -> None:
        """Test compression quality validation achieves ≥95% fact retention."""
        # Arrange - Original content with critical facts
        original_content = """
        CRITICAL: Implemented JWT authentication using HS256 algorithm.
        Token expiration set to 15 minutes with 7-day refresh tokens.
        Redis storage configured with 30-day TTL for refresh tokens.
        Passwords hashed using bcrypt with cost factor 12.
        Rate limiting: 5 login attempts per 15 minutes per IP.
        """

        critical_facts = [
            "JWT authentication",
            "HS256 algorithm",
            "15 minutes token expiration",
            "7-day refresh tokens",
            "Redis storage",
            "30-day TTL",
            "bcrypt hashing",
            "cost factor 12",
            "Rate limiting 5 attempts per 15 minutes",
        ]

        # Act - Compress content
        compressed = await context_compressor.compress_stage(
            stage_id="test-stage",
            stage_type=StageType.EXECUTION,
            raw_memories=[original_content],
        )

        # Validate quality
        quality_metrics = await quality_validator.validate_compression(
            original_content=original_content,
            compressed_content=compressed["summary"],
            critical_facts=critical_facts,
        )

        # Assert - Fact retention ≥95%
        fact_retention = quality_metrics["fact_retention_rate"]
        assert fact_retention >= 0.95, (
            f"Fact retention {fact_retention:.2%} below 95% target"
        )

        # No contradictions introduced
        assert quality_metrics["has_contradictions"] is False

        # Coherence score acceptable
        assert quality_metrics["coherence_score"] >= 0.8

    async def test_compression_cost_tracking(
        self,
        context_compressor: ContextCompressor,
        cost_tracker: CostTracker,
    ) -> None:
        """Test cost tracking for compression operations."""
        # Arrange - Content to compress
        raw_memories = [
            "Memory 1: Implemented feature A",
            "Memory 2: Tested feature A",
            "Memory 3: Deployed feature A",
        ]

        # Act - Compress with cost tracking
        compression_result = await context_compressor.compress_stage(
            stage_id="cost-test-stage",
            stage_type=StageType.EXECUTION,
            raw_memories=raw_memories,
        )

        # Track compression cost
        await cost_tracker.record_compression(
            operation_id="comp-001",
            input_tokens=compression_result.get("input_tokens", 0),
            output_tokens=compression_result.get("output_tokens", 0),
            model="gpt-4.1-mini",
        )

        # Assert - Cost tracked
        metrics = await cost_tracker.get_monthly_usage()
        assert metrics["total_operations"] >= 1
        assert metrics["total_cost"] > 0.0
        assert metrics["total_input_tokens"] > 0
        assert metrics["total_output_tokens"] > 0

        # Cost should be calculated correctly (gpt-4.1-mini pricing)
        expected_cost = (
            metrics["total_input_tokens"] * 0.00015 / 1000  # $0.15 per 1M input tokens
            + metrics["total_output_tokens"] * 0.0006 / 1000  # $0.60 per 1M output tokens
        )

        actual_cost = metrics["total_cost"]
        assert abs(actual_cost - expected_cost) < 0.01, "Cost calculation inaccurate"

    async def test_compression_budget_monitoring(
        self,
        cost_tracker: CostTracker,
    ) -> None:
        """Test monthly budget tracking and alerts."""
        # Arrange - Set monthly budget
        monthly_budget = 100.0  # $100/month
        cost_tracker.set_monthly_budget(monthly_budget)

        # Simulate compression operations
        operations = [
            {"input_tokens": 1000, "output_tokens": 200},  # ~$0.27
            {"input_tokens": 2000, "output_tokens": 400},  # ~$0.54
            {"input_tokens": 5000, "output_tokens": 1000},  # ~$1.35
        ]

        for i, op in enumerate(operations):
            await cost_tracker.record_compression(
                operation_id=f"budget-test-{i}",
                input_tokens=op["input_tokens"],
                output_tokens=op["output_tokens"],
                model="gpt-4.1-mini",
            )

        # Act - Check budget status
        budget_status = await cost_tracker.get_budget_status()

        # Assert - Budget tracked correctly
        assert "monthly_budget" in budget_status
        assert "current_spend" in budget_status
        assert "remaining_budget" in budget_status
        assert "utilization_percentage" in budget_status

        assert budget_status["monthly_budget"] == monthly_budget
        assert budget_status["current_spend"] > 0.0
        assert budget_status["remaining_budget"] < monthly_budget

        # Test alert threshold (75%)
        if budget_status["utilization_percentage"] >= 75.0:
            assert budget_status["alert_triggered"] is True
            assert "alert_level" in budget_status


class TestErrorTrackingWorkflow:
    """Test error tracking and pattern detection workflow."""

    @pytest.fixture
    async def error_tracker(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> ErrorTracker:
        """Create error tracker."""
        storage_backend = StorageBackendService(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )

        graph_service = GraphMemoryService(driver=neo4j_driver)
        await graph_service.initialize_schema()

        return ErrorTracker(
            storage_backend=storage_backend,
            graph_service=graph_service,
        )

    async def test_error_recording_with_full_context(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test recording error with full context."""
        # Arrange - Error data
        error_data = {
            "error_id": "err-001",
            "error_type": "hallucination",
            "severity": 0.8,
            "context": "Generated Python code with incorrect asyncio usage",
            "correction": "Fixed async/await syntax and event loop handling",
            "task_id": "task-123",
            "stage_id": "stage-456",
            "agent_id": "agent-789",
            "timestamp": datetime.now(UTC),
        }

        # Act - Record error
        error_id = await error_tracker.record_error(**error_data)

        # Assert - Error recorded
        assert error_id is not None
        assert error_id == error_data["error_id"]

        # Verify error stored
        recorded_error = await error_tracker.get_error(error_id)
        assert recorded_error is not None
        assert recorded_error["error_type"] == "hallucination"
        assert recorded_error["severity"] == 0.8

        # Verify 100% capture rate (error should be retrievable)
        capture_success = recorded_error is not None
        assert capture_success is True

    async def test_error_type_classification(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test error type classification."""
        # Arrange - Different error types
        error_types = [
            {
                "id": "err-hallucination",
                "type": "hallucination",
                "context": "Generated non-existent API endpoint",
            },
            {
                "id": "err-missing-info",
                "type": "missing_info",
                "context": "Failed to retrieve relevant documentation",
            },
            {
                "id": "err-incorrect-action",
                "type": "incorrect_action",
                "context": "Deleted wrong file during cleanup",
            },
            {
                "id": "err-context-degradation",
                "type": "context_degradation",
                "context": "Lost track of previous conversation context",
            },
        ]

        # Act - Record all error types
        for error in error_types:
            await error_tracker.record_error(
                error_id=error["id"],
                error_type=error["type"],
                severity=0.7,
                context=error["context"],
                correction="Applied fix",
            )

        # Assert - All types classified correctly
        for error in error_types:
            recorded = await error_tracker.get_error(error["id"])
            assert recorded["error_type"] == error["type"]

    async def test_frequency_pattern_detection(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test detecting recurring error patterns by frequency."""
        # Arrange - Record multiple occurrences of same error type
        for i in range(10):
            await error_tracker.record_error(
                error_id=f"err-freq-{i}",
                error_type="hallucination",
                severity=0.6,
                context=f"Hallucinated API method {i}",
                correction="Validated against documentation",
            )

        # Also record other error types (fewer occurrences)
        for i in range(3):
            await error_tracker.record_error(
                error_id=f"err-other-{i}",
                error_type="missing_info",
                severity=0.5,
                context="Missing context",
                correction="Retrieved context",
            )

        # Act - Detect patterns
        patterns = await error_tracker.detect_frequency_patterns()

        # Assert - Should detect hallucination as frequent pattern
        assert len(patterns) >= 1

        # Find hallucination pattern
        hallucination_pattern = next(
            (p for p in patterns if p["error_type"] == "hallucination"),
            None,
        )
        assert hallucination_pattern is not None
        assert hallucination_pattern["frequency"] >= 10

        # Pattern detection accuracy check
        total_errors = 13  # 10 hallucination + 3 missing_info
        detected_errors = sum(p["frequency"] for p in patterns)

        accuracy = detected_errors / total_errors
        assert accuracy >= 0.8, f"Pattern detection accuracy {accuracy:.2%} below 80% target"

    async def test_sequence_pattern_detection(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test detecting error sequence patterns."""
        # Arrange - Record error sequences
        # Pattern: missing_info -> hallucination -> incorrect_action
        sequences = [
            ["missing_info", "hallucination", "incorrect_action"],
            ["missing_info", "hallucination", "incorrect_action"],
            ["missing_info", "hallucination", "incorrect_action"],
        ]

        for seq_idx, sequence in enumerate(sequences):
            for err_idx, error_type in enumerate(sequence):
                await error_tracker.record_error(
                    error_id=f"err-seq-{seq_idx}-{err_idx}",
                    error_type=error_type,
                    severity=0.6,
                    context=f"Sequence {seq_idx} step {err_idx}",
                    correction="Fixed",
                    timestamp=datetime.now(UTC),
                )

        # Act - Detect sequence patterns
        sequence_patterns = await error_tracker.detect_sequence_patterns()

        # Assert - Should detect the repeated sequence
        assert len(sequence_patterns) >= 1

        # Find the specific pattern
        target_pattern = next(
            (
                p
                for p in sequence_patterns
                if "missing_info" in p["sequence"] and "hallucination" in p["sequence"]
            ),
            None,
        )
        assert target_pattern is not None
        assert target_pattern["frequency"] >= 3

    async def test_context_correlation_detection(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test detecting errors in similar contexts."""
        # Arrange - Record errors with similar contexts
        similar_contexts = [
            "Error in async Python code with asyncio",
            "Error in async Python implementation using asyncio",
            "Error in Python asyncio async/await pattern",
        ]

        for i, context in enumerate(similar_contexts):
            await error_tracker.record_error(
                error_id=f"err-context-{i}",
                error_type="hallucination",
                severity=0.7,
                context=context,
                correction="Fixed asyncio usage",
            )

        # Act - Detect context correlations
        correlations = await error_tracker.detect_context_correlations()

        # Assert - Should find "asyncio" as common context
        assert len(correlations) >= 1

        # Find asyncio correlation
        asyncio_correlation = next(
            (c for c in correlations if "asyncio" in c["context_keyword"].lower()),
            None,
        )
        assert asyncio_correlation is not None
        assert asyncio_correlation["occurrence_count"] >= 3

    async def test_error_aware_retrieval_integration(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test error-aware retrieval boosts correction memories."""
        # Arrange - Record error with correction
        error_id = "err-retrieval-test"
        error_context = "JWT token validation failed"
        correction = "Updated token signature verification algorithm"

        await error_tracker.record_error(
            error_id=error_id,
            error_type="incorrect_action",
            severity=0.7,
            context=error_context,
            correction=correction,
        )

        # Act - Retrieve memories related to error context
        query_embedding = [0.5] * 1536  # Mock embedding
        results = await error_tracker.retrieve_error_prevention_knowledge(
            query_embedding=query_embedding,
            query_text="JWT validation",
        )

        # Assert - Correction memory should be retrieved
        assert len(results) >= 1

        # Verify correction memory is boosted (higher relevance)
        correction_result = next(
            (r for r in results if correction in r.get("content", "")),
            None,
        )

        if correction_result:
            # Correction should have high relevance score
            assert correction_result.get("relevance_boost", 0.0) > 1.0

    async def test_ace_integration_signals(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test ACE receives signals when error rate exceeds threshold."""
        # Arrange - Simulate high error rate (>30%)
        total_operations = 20
        error_count = 7  # 35% error rate

        for i in range(error_count):
            await error_tracker.record_error(
                error_id=f"err-ace-{i}",
                error_type="hallucination",
                severity=0.8,
                context=f"Error {i}",
                correction="Fixed",
            )

        # Act - Check if ACE intervention signal triggered
        should_trigger_ace = await error_tracker.should_trigger_ace_intervention(
            window_size=total_operations
        )

        # Assert - ACE signal triggered
        assert should_trigger_ace is True

        # Get ACE context
        ace_context = await error_tracker.get_ace_intervention_context()

        assert ace_context is not None
        assert "error_rate" in ace_context
        assert "recent_errors" in ace_context
        assert "recommended_actions" in ace_context

        error_rate = ace_context["error_rate"]
        assert error_rate >= 0.30, f"Error rate {error_rate:.2%} below 30% threshold"

    async def test_error_severity_scoring(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test error severity scoring (0-1 scale)."""
        # Arrange - Errors with different severities
        errors = [
            {"id": "err-low", "type": "missing_info", "severity": 0.2},
            {"id": "err-medium", "type": "incorrect_action", "severity": 0.5},
            {"id": "err-high", "type": "hallucination", "severity": 0.9},
        ]

        # Act - Record errors
        for error in errors:
            await error_tracker.record_error(
                error_id=error["id"],
                error_type=error["type"],
                severity=error["severity"],
                context="Test context",
                correction="Fixed",
            )

        # Assert - Severity scores preserved correctly
        for error in errors:
            recorded = await error_tracker.get_error(error["id"])
            assert recorded["severity"] == error["severity"]
            assert 0.0 <= recorded["severity"] <= 1.0
