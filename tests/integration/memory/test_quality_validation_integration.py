"""
Integration tests for QualityValidator with ContextCompressor.

Tests the integration of quality validation with compression operations,
including adaptive fallback scenarios and end-to-end quality validation flow.

Component ID: MEM-013
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage
from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord
from agentcore.a2a_protocol.services.memory.context_compressor import (
    ContextCompressor,
)
from agentcore.a2a_protocol.services.memory.quality_validator import (
    QualityMetrics,
    QualityValidator,
)


class TestQualityValidationIntegration:
    """Integration tests for quality validation with compression."""

    @pytest.fixture
    def compressor(self) -> ContextCompressor:
        """Create a ContextCompressor instance."""
        return ContextCompressor(trace_id="test-integration-trace")

    @pytest.fixture
    def validator(self) -> QualityValidator:
        """Create a QualityValidator instance."""
        return QualityValidator(trace_id="test-integration-trace")

    @pytest.fixture
    def sample_memories(self) -> list[MemoryRecord]:
        """Create sample memory records for testing."""
        return [
            MemoryRecord(
                memory_id="mem-001",
                memory_layer=MemoryLayer.EPISODIC,
                content="User requested JWT authentication with Redis storage.",
                summary="JWT auth request",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-002",
                memory_layer=MemoryLayer.EPISODIC,
                content="Decided on 1-hour TTL for tokens. Error threshold is 5%.",
                summary="Token TTL and error threshold",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-003",
                memory_layer=MemoryLayer.EPISODIC,
                content="Implemented /auth/login endpoint successfully.",
                summary="Login endpoint done",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-004",
                memory_layer=MemoryLayer.EPISODIC,
                content="Initial error rate was 8%, exceeding 5% threshold.",
                summary="High error rate",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-005",
                memory_layer=MemoryLayer.EPISODIC,
                content="Fixed connection pooling. Error rate now 2%.",
                summary="Error rate fixed",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
        ]

    @pytest.mark.asyncio
    async def test_compression_with_quality_validation(
        self,
        compressor: ContextCompressor,
        validator: QualityValidator,
        sample_memories: list[MemoryRecord],
    ):
        """Test full compression flow with enhanced quality validation."""
        # Define critical facts manually (simulating what would be extracted)
        critical_facts = [
            "JWT authentication required",
            "Redis storage with 1-hour TTL",
            "Error threshold: 5%",
            "Login endpoint: /auth/login",
            "Final error rate: 2%",
        ]

        original_content = compressor._combine_memories(sample_memories)
        compressed_content = """Implemented JWT authentication with Redis token storage (1h TTL).
Created /auth/login endpoint. Initial 8% error rate exceeded 5% threshold.
Fixed via connection pooling, now at 2%."""

        # Mock LLM responses for enhanced quality validation
        fact_retention_response = LLMResponse(
            content="""RETAINED: [1, 2, 3, 4, 5]
MISSING: []""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=250, completion_tokens=30, total_tokens=280),
            latency_ms=250,
        )

        contradiction_response = LLMResponse(
            content="""CONTRADICTIONS: 0
DETAILS: None""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=20, total_tokens=320),
            latency_ms=200,
        )

        content_quality_response = LLMResponse(
            content="0.96",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=10, total_tokens=210),
            latency_ms=150,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_validate_llm:
            # Setup validator mocks
            mock_validate_llm.side_effect = [
                fact_retention_response,
                contradiction_response,
                content_quality_response,
            ]

            # Step 1: Validate fact retention
            retention_rate, missing_facts = await validator.validate_fact_retention(
                original_content=original_content,
                compressed_content=compressed_content,
                critical_facts=critical_facts,
            )

            assert retention_rate == 1.0
            assert missing_facts == []

            # Step 2: Detect contradictions
            contradiction_count, contradictions = await validator.detect_contradictions(
                original_content=original_content,
                compressed_content=compressed_content,
            )

            assert contradiction_count == 0
            assert contradictions == []

            # Step 3: Calculate coherence score
            coherence_score = await validator.calculate_coherence_score(
                original_content=original_content,
                compressed_content=compressed_content,
                critical_facts=critical_facts,
                fact_retention_rate=retention_rate,
                contradiction_count=contradiction_count,
            )

            assert coherence_score > 0.9

            # Step 4: Create quality metrics and check for degradation
            quality_metrics = QualityMetrics(
                compression_ratio=13.9,  # Simulated ratio
                fact_retention_rate=retention_rate,
                coherence_score=coherence_score,
                contradiction_count=contradiction_count,
                quality_score=0.97,
            )

            # Check for quality degradation
            is_degraded, alerts = await validator.check_quality_degradation(
                quality_metrics
            )

            assert is_degraded is False
            assert alerts == []

    @pytest.mark.asyncio
    async def test_adaptive_compression_fallback_scenario(
        self,
        validator: QualityValidator,
    ):
        """Test adaptive compression fallback when quality degrades."""
        # Define critical facts (5 facts)
        critical_facts = [
            "JWT authentication",
            "Redis storage",
            "1-hour TTL",
            "Login endpoint",
            "Error rate fixed",
        ]

        # Simulated aggressive compression (missing facts)
        original_content = "JWT authentication with Redis storage, 1-hour TTL. Login endpoint at /auth/login. Error rate fixed."
        compressed_content = "JWT auth implemented with Redis."  # Very aggressive, lost critical info

        # Mock response showing poor retention (only 2/5 facts retained)
        poor_retention_response = LLMResponse(
            content="""RETAINED: [1, 2]
MISSING: [3, 4, 5]""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=250, completion_tokens=25, total_tokens=275),
            latency_ms=200,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_validate_llm:
            mock_validate_llm.return_value = poor_retention_response

            # Step 1: Validate quality
            retention_rate, missing_facts = await validator.validate_fact_retention(
                original_content=original_content,
                compressed_content=compressed_content,
                critical_facts=critical_facts,
            )

            # Poor retention (2/5 = 0.4)
            assert retention_rate == 0.4
            assert len(missing_facts) == 3

            # Step 2: Check for quality degradation
            quality_metrics = QualityMetrics(
                compression_ratio=5.2,
                fact_retention_rate=retention_rate,
                coherence_score=0.85,
                contradiction_count=0,
                quality_score=0.85,
            )

            is_degraded, alerts = await validator.check_quality_degradation(
                quality_metrics
            )

            assert is_degraded is True
            assert len(alerts) >= 2  # Low retention + low coherence
            assert any("Fact retention" in alert for alert in alerts)

            # Step 3: Get fallback ratio
            original_target = 10.0
            fallback_ratio = validator.get_fallback_ratio(original_target)

            assert fallback_ratio == 8.0

            # In production, we would retry compression with fallback_ratio

    @pytest.mark.asyncio
    async def test_compression_metrics_with_quality_fields(
        self, compressor: ContextCompressor, sample_memories: list[MemoryRecord]
    ):
        """Test that CompressionMetrics can store enhanced quality fields."""
        # Mock responses
        fact_response = LLMResponse(
            content="1. JWT authentication\n2. Redis storage",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            latency_ms=200,
        )

        compression_response = LLMResponse(
            content="JWT auth with Redis storage implemented.",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=50, total_tokens=550),
            latency_ms=400,
        )

        quality_response = LLMResponse(
            content="Quality Score: 0.96",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=15, total_tokens=315),
            latency_ms=150,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.side_effect = [
                fact_response,
                compression_response,
                quality_response,
            ]

            summary, metrics = await compressor.compress_task(
                task_id="task-789",
                stage_summaries=["Stage 1 summary", "Stage 2 summary"],
                task_goal="Implement auth",
            )

            # Verify metrics has base fields
            assert metrics.compression_ratio > 0
            assert metrics.quality_score >= 0.95
            assert metrics.cost_usd > 0

            # Enhanced quality fields should be None initially
            # (would be populated by integration with QualityValidator)
            assert metrics.coherence_score is None
            assert metrics.fact_retention_rate is None
            assert metrics.contradiction_count is None

            # Verify to_dict includes all fields
            metrics_dict = metrics.to_dict()
            assert "compression_ratio" in metrics_dict
            assert "quality_score" in metrics_dict
            # Optional fields only included if set
            assert "coherence_score" not in metrics_dict
            assert "fact_retention_rate" not in metrics_dict
            assert "contradiction_count" not in metrics_dict

    @pytest.mark.asyncio
    async def test_quality_validation_with_contradictions_detected(
        self, validator: QualityValidator
    ):
        """Test quality validation when contradictions are detected."""
        original_content = "Final error rate is 2%. System is stable."
        compressed_content = "Final error rate is 8%. System has issues."

        # Mock contradiction detection
        contradiction_response = LLMResponse(
            content="""CONTRADICTIONS: 2
DETAILS:
- Error rate changed from 2% to 8%
- System status changed from stable to having issues""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=40, total_tokens=340),
            latency_ms=250,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = contradiction_response

            count, contradictions = await validator.detect_contradictions(
                original_content=original_content,
                compressed_content=compressed_content,
            )

            assert count == 2
            assert len(contradictions) == 2

            # Create quality metrics with contradictions
            quality_metrics = QualityMetrics(
                compression_ratio=9.5,
                fact_retention_rate=0.96,
                coherence_score=0.85,
                contradiction_count=count,
                quality_score=0.88,
            )

            # Should trigger quality degradation alert
            is_degraded, alerts = await validator.check_quality_degradation(
                quality_metrics
            )

            assert is_degraded is True
            assert any("contradiction" in alert.lower() for alert in alerts)

    @pytest.mark.asyncio
    async def test_quality_validator_model_consistency(
        self, compressor: ContextCompressor, validator: QualityValidator
    ):
        """Test that both compressor and validator use approved models."""
        # Verify both use gpt-4.1-mini for cost efficiency
        assert compressor.COMPRESSION_MODEL == "gpt-4.1-mini"
        assert validator.VALIDATION_MODEL == "gpt-4.1-mini"

        # Verify targets align
        assert compressor.STAGE_COMPRESSION_TARGET == 10.0
        assert compressor.TASK_COMPRESSION_TARGET == 5.0

        # Verify validator has fallback for these targets
        assert validator.get_fallback_ratio(10.0) == 8.0
        assert validator.get_fallback_ratio(5.0) == 4.0
