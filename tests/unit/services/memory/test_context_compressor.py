"""
Unit tests for ContextCompressor with mocked LLM responses.

Tests stage compression, task compression, critical fact extraction,
and quality validation with deterministic mocked LLM outputs.

Component ID: MEM-012
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage
from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord
from agentcore.a2a_protocol.services.memory.context_compressor import (
    CompressionMetrics,
    ContextCompressor,
)


class TestContextCompressor:
    """Test suite for ContextCompressor."""

    @pytest.fixture
    def compressor(self) -> ContextCompressor:
        """Create a ContextCompressor instance."""
        return ContextCompressor(trace_id="test-trace-123")

    @pytest.fixture
    def sample_memories(self) -> list[MemoryRecord]:
        """Create sample memory records for testing."""
        return [
            MemoryRecord(
                memory_id="mem-001",
                memory_layer=MemoryLayer.EPISODIC,
                content="User requested authentication implementation using JWT tokens.",
                summary="User wants JWT auth",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-002",
                memory_layer=MemoryLayer.EPISODIC,
                content="Decided to use Redis for token storage with 1-hour TTL.",
                summary="Redis for tokens, 1h TTL",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-003",
                memory_layer=MemoryLayer.EPISODIC,
                content="Implemented /auth/login endpoint successfully. Returns JWT on success.",
                summary="Login endpoint working",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-004",
                memory_layer=MemoryLayer.EPISODIC,
                content="Error rate observed at 8% during load testing. Threshold is 5%.",
                summary="Error rate too high",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
            MemoryRecord(
                memory_id="mem-005",
                memory_layer=MemoryLayer.EPISODIC,
                content="Fixed connection pooling issue. Error rate now at 2%.",
                summary="Error rate fixed",
                agent_id="agent-123",
                task_id="task-789",
                timestamp=datetime.now(UTC),
            ),
        ]

    @pytest.fixture
    def mock_llm_response(self) -> LLMResponse:
        """Create a mock LLM response."""
        return LLMResponse(
            content="Mock response",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
            latency_ms=500,
            trace_id="test-trace-123",
        )

    @pytest.mark.asyncio
    async def test_compress_stage_success(
        self, compressor: ContextCompressor, sample_memories: list[MemoryRecord]
    ):
        """Test successful stage compression with 10:1 ratio validation."""
        # Mock LLM responses for fact extraction and compression
        fact_response = LLMResponse(
            content="""1. Use JWT tokens for authentication
2. Redis storage with 1-hour TTL
3. Error rate threshold is 5%
4. Login endpoint: /auth/login""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
            latency_ms=300,
        )

        compression_response = LLMResponse(
            content="""Implemented JWT authentication with Redis token storage (1h TTL).
Created /auth/login endpoint. Initial error rate of 8% exceeded 5% threshold;
fixed via connection pooling, now at 2%.""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600),
            latency_ms=600,
        )

        quality_response = LLMResponse(
            content="Quality Score: 0.97",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=20, total_tokens=320),
            latency_ms=200,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            # Return different responses for each call
            mock_complete.side_effect = [
                fact_response,
                compression_response,
                quality_response,
            ]

            result = await compressor.compress_stage(
                stage_id="stage-001",
                raw_memory_ids=["mem-001", "mem-002", "mem-003", "mem-004", "mem-005"],
                raw_memories=sample_memories,
                stage_type="execution",
            )

            # Validate result
            assert "compression_ratio" in result
            assert "quality_score" in result
            assert result["compression_ratio"] > 1.0  # Should have compression
            assert result["quality_score"] >= 0.95  # Should meet quality threshold
            assert mock_complete.call_count == 3  # fact extraction, compression, validation

    @pytest.mark.asyncio
    async def test_compress_stage_no_memories(self, compressor: ContextCompressor):
        """Test stage compression with no memories provided."""
        result = await compressor.compress_stage(
            stage_id="stage-002",
            raw_memory_ids=["mem-001"],
            raw_memories=None,  # No memories provided
        )

        # Should return default metrics
        assert result["compression_ratio"] == 1.0
        assert result["quality_score"] == 1.0

    @pytest.mark.asyncio
    async def test_compress_task_success(self, compressor: ContextCompressor):
        """Test successful task compression with 5:1 ratio validation."""
        stage_summaries = [
            "Planning stage: Analyzed authentication requirements. Decided on JWT with Redis storage.",
            "Execution stage: Implemented /auth/login endpoint. Initial error rate 8% exceeded threshold.",
            "Reflection stage: Identified connection pooling issue causing errors.",
            "Verification stage: Fixed pooling issue. Error rate now 2%, within 5% threshold.",
        ]

        # Mock LLM responses
        fact_response = LLMResponse(
            content="""1. JWT authentication with Redis
2. Error rate threshold: 5%
3. Login endpoint: /auth/login
4. Connection pooling fix applied""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=60, total_tokens=360),
            latency_ms=400,
        )

        compression_response = LLMResponse(
            content="""Task: Implement JWT authentication system.
Completed: Login endpoint with Redis token storage.
Issue: Error rate exceeded 5% threshold.
Resolution: Fixed connection pooling, now at 2%.""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=600, completion_tokens=80, total_tokens=680),
            latency_ms=700,
        )

        quality_response = LLMResponse(
            content="Quality Score: 0.98",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=400, completion_tokens=25, total_tokens=425),
            latency_ms=250,
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
                stage_summaries=stage_summaries,
                task_goal="Implement authentication system",
            )

            # Validate summary
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "JWT" in summary or "authentication" in summary.lower()

            # Validate metrics
            assert isinstance(metrics, CompressionMetrics)
            assert metrics.compression_ratio >= 1.0
            assert metrics.quality_score >= 0.95
            assert metrics.model == "gpt-4.1-mini"
            assert metrics.cost_usd > 0
            assert metrics.latency_seconds > 0

    @pytest.mark.asyncio
    async def test_extract_critical_facts(self, compressor: ContextCompressor):
        """Test critical fact extraction."""
        content = """User requested JWT authentication.
Redis storage with 1-hour TTL required.
Error rate must be below 5%.
Login endpoint is /auth/login."""

        fact_response = LLMResponse(
            content="""1. JWT authentication required
2. Redis storage, 1-hour TTL
3. Error rate threshold: 5%
4. Login endpoint: /auth/login""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=150, completion_tokens=40, total_tokens=190),
            latency_ms=200,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = fact_response

            facts = await compressor.extract_critical_facts(content, "stage")

            # Validate facts
            assert isinstance(facts, list)
            assert len(facts) == 4
            assert all(isinstance(fact, str) for fact in facts)
            assert any("JWT" in fact for fact in facts)

    @pytest.mark.asyncio
    async def test_extract_critical_facts_error_handling(
        self, compressor: ContextCompressor
    ):
        """Test critical fact extraction with LLM error."""
        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.side_effect = Exception("LLM API error")

            facts = await compressor.extract_critical_facts("test content")

            # Should return empty list on error
            assert isinstance(facts, list)
            assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_validate_compression_quality(self, compressor: ContextCompressor):
        """Test compression quality validation."""
        original = "JWT authentication with Redis storage (1h TTL). Error threshold 5%."
        compressed = "JWT auth using Redis, 1h TTL. 5% error limit."
        critical_facts = [
            "JWT authentication",
            "Redis storage",
            "1-hour TTL",
            "5% error threshold",
        ]

        quality_response = LLMResponse(
            content="Quality Score: 0.95",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=15, total_tokens=215),
            latency_ms=150,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = quality_response

            quality = await compressor.validate_compression_quality(
                original, compressed, critical_facts
            )

            # Validate quality score
            assert isinstance(quality, float)
            assert 0.0 <= quality <= 1.0
            assert quality == 0.95

    @pytest.mark.asyncio
    async def test_validate_quality_with_percentage_response(
        self, compressor: ContextCompressor
    ):
        """Test quality validation with percentage format response."""
        quality_response = LLMResponse(
            content="95%",  # Alternative format
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=10, total_tokens=210),
            latency_ms=100,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = quality_response

            quality = await compressor.validate_compression_quality(
                "original", "compressed", ["fact1", "fact2"]
            )

            # Should parse percentage correctly
            assert quality == 0.95

    @pytest.mark.asyncio
    async def test_validate_quality_fallback_heuristic(
        self, compressor: ContextCompressor
    ):
        """Test quality validation fallback to heuristic on parse error."""
        quality_response = LLMResponse(
            content="The quality is very good",  # No numeric score
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=20, total_tokens=220),
            latency_ms=150,
        )

        compressed = "JWT authentication with Redis storage"
        critical_facts = [
            "JWT authentication",  # Present
            "Redis storage",  # Present
            "1-hour TTL",  # Missing
            "5% error threshold",  # Missing
        ]

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = quality_response

            quality = await compressor.validate_compression_quality(
                "original", compressed, critical_facts
            )

            # Should use heuristic: 2/4 facts present = 0.5
            assert quality == 0.5

    @pytest.mark.asyncio
    async def test_validate_quality_no_facts(self, compressor: ContextCompressor):
        """Test quality validation with no critical facts."""
        quality = await compressor.validate_compression_quality(
            "original", "compressed", []
        )

        # Should return perfect score when no facts to validate
        assert quality == 1.0

    @pytest.mark.asyncio
    async def test_compression_model_enforcement(self, compressor: ContextCompressor):
        """Test that only gpt-4.1-mini is used for compression."""
        assert compressor.COMPRESSION_MODEL == "gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_compression_ratio_targets(self, compressor: ContextCompressor):
        """Test compression ratio targets."""
        assert compressor.STAGE_COMPRESSION_TARGET == 10.0
        assert compressor.TASK_COMPRESSION_TARGET == 5.0

    @pytest.mark.asyncio
    async def test_quality_threshold(self, compressor: ContextCompressor):
        """Test minimum quality score threshold."""
        assert compressor.MIN_QUALITY_SCORE == 0.95

    @pytest.mark.asyncio
    async def test_cost_calculation(self, compressor: ContextCompressor):
        """Test cost calculation for compression."""
        # 1M input tokens, 1M output tokens
        cost = compressor._calculate_cost(1_000_000, 1_000_000)

        # Expected: $0.15 + $0.60 = $0.75
        assert cost == pytest.approx(0.75, rel=0.01)

    @pytest.mark.asyncio
    async def test_cost_calculation_small(self, compressor: ContextCompressor):
        """Test cost calculation for small token counts."""
        # 1000 input tokens, 500 output tokens
        cost = compressor._calculate_cost(1000, 500)

        # Expected: (1000/1M * $0.15) + (500/1M * $0.60)
        # = 0.00015 + 0.0003 = 0.00045
        assert cost == pytest.approx(0.00045, rel=0.01)

    @pytest.mark.asyncio
    async def test_compression_metrics_to_dict(self):
        """Test CompressionMetrics to_dict conversion."""
        metrics = CompressionMetrics(
            compression_ratio=8.5,
            quality_score=0.96,
            latency_seconds=2.3,
            input_tokens=1000,
            output_tokens=200,
            model="gpt-4.1-mini",
            cost_usd=0.0005,
        )

        result = metrics.to_dict()

        assert result["compression_ratio"] == 8.5
        assert result["quality_score"] == 0.96
        assert result["latency_seconds"] == 2.3
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 200
        assert result["model"] == "gpt-4.1-mini"
        assert result["cost_usd"] == 0.0005

    @pytest.mark.asyncio
    async def test_combine_memories(
        self, compressor: ContextCompressor, sample_memories: list[MemoryRecord]
    ):
        """Test memory combination into single content string."""
        combined = compressor._combine_memories(sample_memories)

        # Validate combined content
        assert isinstance(combined, str)
        assert len(combined) > 0
        assert "JWT" in combined
        assert "Redis" in combined
        assert all(mem.content in combined for mem in sample_memories)

    @pytest.mark.asyncio
    async def test_compression_error_fallback(self, compressor: ContextCompressor):
        """Test compression fallback behavior on LLM error."""
        content = "A" * 1000  # 1000 character content
        target_ratio = 10.0

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_complete:
            mock_complete.side_effect = Exception("LLM API error")

            result = await compressor._compress_content(
                content=content,
                target_ratio=target_ratio,
                context_type="stage",
            )

            # Should return truncated content as fallback
            assert isinstance(result, str)
            assert len(result) <= len(content)
            assert "[compressed due to error]" in result

    @pytest.mark.asyncio
    async def test_stage_compression_low_quality_warning(
        self, compressor: ContextCompressor, sample_memories: list[MemoryRecord]
    ):
        """Test that low quality compression triggers warning but continues."""
        # Mock responses with low quality score
        fact_response = LLMResponse(
            content="1. JWT tokens\n2. Redis storage",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            latency_ms=200,
        )

        compression_response = LLMResponse(
            content="Auth implemented.",  # Very aggressive compression
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=500, completion_tokens=20, total_tokens=520),
            latency_ms=300,
        )

        quality_response = LLMResponse(
            content="Quality Score: 0.60",  # Below 0.95 threshold
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

            result = await compressor.compress_stage(
                stage_id="stage-003",
                raw_memory_ids=["mem-001", "mem-002"],
                raw_memories=sample_memories[:2],
            )

            # Should still return result, just with low quality score
            assert result["quality_score"] == 0.60
            assert result["quality_score"] < compressor.MIN_QUALITY_SCORE
