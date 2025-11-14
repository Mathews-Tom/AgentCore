"""
Unit tests for QualityValidator with mocked LLM responses.

Tests fact retention tracking, contradiction detection, coherence scoring,
quality degradation alerts, and adaptive compression fallback.

Component ID: MEM-013
"""

from unittest.mock import patch

import pytest

from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage
from agentcore.a2a_protocol.services.memory.quality_validator import (
    QualityMetrics,
    QualityValidator,
)


class TestQualityValidator:
    """Test suite for QualityValidator."""

    @pytest.fixture
    def validator(self) -> QualityValidator:
        """Create a QualityValidator instance."""
        return QualityValidator(trace_id="test-trace-123")

    @pytest.fixture
    def sample_original_content(self) -> str:
        """Create sample original content for testing."""
        return """User requested JWT authentication implementation.
System will use Redis for token storage with 1-hour TTL.
Error rate threshold is 5%.
Login endpoint is /auth/login.
Connection pooling issue was identified and fixed.
Final error rate is 2%."""

    @pytest.fixture
    def sample_compressed_content(self) -> str:
        """Create sample compressed content for testing."""
        return """Implemented JWT authentication with Redis (1h TTL).
Login endpoint: /auth/login. Error rate: 2% (threshold: 5%).
Fixed connection pooling issue."""

    @pytest.fixture
    def sample_critical_facts(self) -> list[str]:
        """Create sample critical facts for testing."""
        return [
            "JWT authentication required",
            "Redis storage with 1-hour TTL",
            "Error rate threshold is 5%",
            "Login endpoint: /auth/login",
            "Error rate is 2%",
        ]

    @pytest.mark.asyncio
    async def test_validate_fact_retention_success(
        self,
        validator: QualityValidator,
        sample_compressed_content: str,
        sample_critical_facts: list[str],
    ):
        """Test successful fact retention validation with high retention."""
        # Mock LLM response indicating high retention
        llm_response = LLMResponse(
            content="""RETAINED: [1, 2, 3, 4, 5]
MISSING: []""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            latency_ms=300,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = llm_response

            retention_rate, missing_facts = await validator.validate_fact_retention(
                original_content="original",
                compressed_content=sample_compressed_content,
                critical_facts=sample_critical_facts,
            )

            # Validate retention rate
            assert retention_rate == 1.0  # All facts retained
            assert missing_facts == []
            assert mock_complete.call_count == 1

    @pytest.mark.asyncio
    async def test_validate_fact_retention_partial(
        self,
        validator: QualityValidator,
        sample_compressed_content: str,
        sample_critical_facts: list[str],
    ):
        """Test fact retention validation with partial retention."""
        # Mock LLM response indicating partial retention
        llm_response = LLMResponse(
            content="""RETAINED: [1, 2, 4]
MISSING: [3, 5]""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            latency_ms=300,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = llm_response

            retention_rate, missing_facts = await validator.validate_fact_retention(
                original_content="original",
                compressed_content=sample_compressed_content,
                critical_facts=sample_critical_facts,
            )

            # Validate retention rate: 3/5 = 0.6
            assert retention_rate == 0.6
            assert len(missing_facts) == 2
            assert sample_critical_facts[2] in missing_facts  # Fact 3
            assert sample_critical_facts[4] in missing_facts  # Fact 5

    @pytest.mark.asyncio
    async def test_validate_fact_retention_no_facts(
        self, validator: QualityValidator
    ):
        """Test fact retention validation with no critical facts."""
        retention_rate, missing_facts = await validator.validate_fact_retention(
            original_content="original",
            compressed_content="compressed",
            critical_facts=[],
        )

        # Should return perfect retention when no facts to validate
        assert retention_rate == 1.0
        assert missing_facts == []

    @pytest.mark.asyncio
    async def test_validate_fact_retention_error_fallback(
        self,
        validator: QualityValidator,
        sample_compressed_content: str,
        sample_critical_facts: list[str],
    ):
        """Test fact retention validation fallback on LLM error."""
        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.side_effect = Exception("LLM API error")

            retention_rate, missing_facts = await validator.validate_fact_retention(
                original_content="original",
                compressed_content=sample_compressed_content,
                critical_facts=sample_critical_facts,
            )

            # Should use heuristic fallback
            assert isinstance(retention_rate, float)
            assert 0.0 <= retention_rate <= 1.0
            assert isinstance(missing_facts, list)

    @pytest.mark.asyncio
    async def test_validate_compression_ratio_valid(
        self, validator: QualityValidator
    ):
        """Test compression ratio validation within tolerance."""
        # Test 10:1 target with ratio of 9.5 (within 20% tolerance)
        is_valid = await validator.validate_compression_ratio(
            compression_ratio=9.5,
            target_ratio=10.0,
            tolerance=0.2,
        )

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_compression_ratio_invalid_low(
        self, validator: QualityValidator
    ):
        """Test compression ratio validation below tolerance."""
        # Test 10:1 target with ratio of 7.0 (below 20% tolerance)
        is_valid = await validator.validate_compression_ratio(
            compression_ratio=7.0,
            target_ratio=10.0,
            tolerance=0.2,
        )

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_compression_ratio_invalid_high(
        self, validator: QualityValidator
    ):
        """Test compression ratio validation above tolerance."""
        # Test 10:1 target with ratio of 13.0 (above 20% tolerance)
        is_valid = await validator.validate_compression_ratio(
            compression_ratio=13.0,
            target_ratio=10.0,
            tolerance=0.2,
        )

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_detect_contradictions_none(
        self,
        validator: QualityValidator,
        sample_original_content: str,
        sample_compressed_content: str,
    ):
        """Test contradiction detection with no contradictions."""
        llm_response = LLMResponse(
            content="""CONTRADICTIONS: 0
DETAILS: None""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=20, total_tokens=320),
            latency_ms=250,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = llm_response

            count, contradictions = await validator.detect_contradictions(
                original_content=sample_original_content,
                compressed_content=sample_compressed_content,
            )

            assert count == 0
            assert contradictions == []

    @pytest.mark.asyncio
    async def test_detect_contradictions_found(
        self,
        validator: QualityValidator,
        sample_original_content: str,
    ):
        """Test contradiction detection with contradictions found."""
        # Compressed content with contradiction
        contradictory_content = """Implemented JWT authentication with Redis (1h TTL).
Login endpoint: /auth/login. Error rate: 8% (threshold: 5%).
No issues found."""

        llm_response = LLMResponse(
            content="""CONTRADICTIONS: 2
DETAILS:
- Error rate of 8% contradicts original final error rate of 2%
- "No issues found" contradicts the connection pooling issue mentioned""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=50, total_tokens=350),
            latency_ms=300,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = llm_response

            count, contradictions = await validator.detect_contradictions(
                original_content=sample_original_content,
                compressed_content=contradictory_content,
            )

            assert count == 2
            assert len(contradictions) == 2
            assert any("8%" in c for c in contradictions)
            assert any("No issues" in c for c in contradictions)

    @pytest.mark.asyncio
    async def test_detect_contradictions_error_fallback(
        self,
        validator: QualityValidator,
        sample_original_content: str,
        sample_compressed_content: str,
    ):
        """Test contradiction detection fallback on LLM error."""
        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.side_effect = Exception("LLM API error")

            count, contradictions = await validator.detect_contradictions(
                original_content=sample_original_content,
                compressed_content=sample_compressed_content,
            )

            # Conservative fallback: assume no contradictions
            assert count == 0
            assert contradictions == []

    @pytest.mark.asyncio
    async def test_calculate_coherence_score_high(
        self,
        validator: QualityValidator,
        sample_original_content: str,
        sample_compressed_content: str,
        sample_critical_facts: list[str],
    ):
        """Test coherence score calculation with high quality."""
        # Mock content quality assessment
        llm_response = LLMResponse(
            content="0.95",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=10, total_tokens=210),
            latency_ms=150,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = llm_response

            coherence_score = await validator.calculate_coherence_score(
                original_content=sample_original_content,
                compressed_content=sample_compressed_content,
                critical_facts=sample_critical_facts,
                fact_retention_rate=0.98,  # High retention
                contradiction_count=0,  # No contradictions
            )

            # Expected: (0.98 * 0.5) + (1.0 * 0.3) + (0.95 * 0.2) = 0.49 + 0.3 + 0.19 = 0.98
            assert coherence_score > 0.9
            assert coherence_score <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_coherence_score_low(
        self,
        validator: QualityValidator,
        sample_original_content: str,
        sample_compressed_content: str,
        sample_critical_facts: list[str],
    ):
        """Test coherence score calculation with low quality."""
        # Mock content quality assessment
        llm_response = LLMResponse(
            content="0.60",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=10, total_tokens=210),
            latency_ms=150,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.quality_validator.llm_service.complete"
        ) as mock_complete:
            mock_complete.return_value = llm_response

            coherence_score = await validator.calculate_coherence_score(
                original_content=sample_original_content,
                compressed_content=sample_compressed_content,
                critical_facts=sample_critical_facts,
                fact_retention_rate=0.75,  # Lower retention
                contradiction_count=2,  # 2 contradictions
            )

            # Expected: (0.75 * 0.5) + (0.6 * 0.3) + (0.6 * 0.2)
            # = 0.375 + 0.18 + 0.12 = 0.675
            assert coherence_score < 0.8
            assert coherence_score > 0.0

    @pytest.mark.asyncio
    async def test_check_quality_degradation_none(
        self, validator: QualityValidator
    ):
        """Test quality degradation check with good quality."""
        metrics = QualityMetrics(
            compression_ratio=9.8,
            fact_retention_rate=0.98,
            coherence_score=0.96,
            contradiction_count=0,
            quality_score=0.97,
        )

        is_degraded, alerts = await validator.check_quality_degradation(metrics)

        assert is_degraded is False
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_quality_degradation_low_retention(
        self, validator: QualityValidator
    ):
        """Test quality degradation detection with low fact retention."""
        metrics = QualityMetrics(
            compression_ratio=9.5,
            fact_retention_rate=0.90,  # Below 0.95 threshold
            coherence_score=0.92,
            contradiction_count=0,
            quality_score=0.91,
        )

        is_degraded, alerts = await validator.check_quality_degradation(metrics)

        assert is_degraded is True
        assert len(alerts) == 1
        assert "Fact retention" in alerts[0]
        assert "0.95" in alerts[0] or "95" in alerts[0]

    @pytest.mark.asyncio
    async def test_check_quality_degradation_low_coherence(
        self, validator: QualityValidator
    ):
        """Test quality degradation detection with low coherence."""
        metrics = QualityMetrics(
            compression_ratio=9.5,
            fact_retention_rate=0.96,
            coherence_score=0.85,  # Below 0.90 threshold
            contradiction_count=0,
            quality_score=0.90,
        )

        is_degraded, alerts = await validator.check_quality_degradation(metrics)

        assert is_degraded is True
        assert len(alerts) == 1
        assert "Coherence" in alerts[0]
        assert "0.90" in alerts[0] or "90" in alerts[0]

    @pytest.mark.asyncio
    async def test_check_quality_degradation_contradictions(
        self, validator: QualityValidator
    ):
        """Test quality degradation detection with contradictions."""
        metrics = QualityMetrics(
            compression_ratio=9.5,
            fact_retention_rate=0.96,
            coherence_score=0.92,
            contradiction_count=2,  # Above 0 threshold
            quality_score=0.88,
        )

        is_degraded, alerts = await validator.check_quality_degradation(metrics)

        assert is_degraded is True
        assert len(alerts) == 1
        assert "contradiction" in alerts[0].lower()
        assert "2" in alerts[0]

    @pytest.mark.asyncio
    async def test_check_quality_degradation_multiple_issues(
        self, validator: QualityValidator
    ):
        """Test quality degradation detection with multiple issues."""
        metrics = QualityMetrics(
            compression_ratio=9.5,
            fact_retention_rate=0.90,  # Low retention
            coherence_score=0.85,  # Low coherence
            contradiction_count=1,  # Has contradictions
            quality_score=0.85,
        )

        is_degraded, alerts = await validator.check_quality_degradation(metrics)

        assert is_degraded is True
        assert len(alerts) == 3
        assert any("Fact retention" in a for a in alerts)
        assert any("Coherence" in a for a in alerts)
        assert any("contradiction" in a.lower() for a in alerts)

    @pytest.mark.asyncio
    async def test_get_fallback_ratio_stage(self, validator: QualityValidator):
        """Test fallback ratio retrieval for stage compression."""
        fallback = validator.get_fallback_ratio(10.0)

        assert fallback == 8.0

    @pytest.mark.asyncio
    async def test_get_fallback_ratio_task(self, validator: QualityValidator):
        """Test fallback ratio retrieval for task compression."""
        fallback = validator.get_fallback_ratio(5.0)

        assert fallback == 4.0

    @pytest.mark.asyncio
    async def test_get_fallback_ratio_unknown(self, validator: QualityValidator):
        """Test fallback ratio retrieval for unknown target."""
        fallback = validator.get_fallback_ratio(15.0)

        assert fallback is None

    @pytest.mark.asyncio
    async def test_quality_metrics_to_dict(self):
        """Test QualityMetrics to_dict conversion."""
        metrics = QualityMetrics(
            compression_ratio=9.8,
            fact_retention_rate=0.97,
            coherence_score=0.95,
            contradiction_count=0,
            quality_score=0.96,
            quality_degraded=False,
            fallback_triggered=False,
            original_target_ratio=10.0,
            adjusted_target_ratio=None,
        )

        result = metrics.to_dict()

        assert result["compression_ratio"] == 9.8
        assert result["fact_retention_rate"] == 0.97
        assert result["coherence_score"] == 0.95
        assert result["contradiction_count"] == 0
        assert result["quality_score"] == 0.96
        assert result["quality_degraded"] is False
        assert result["fallback_triggered"] is False
        assert result["original_target_ratio"] == 10.0
        assert result["adjusted_target_ratio"] is None

    @pytest.mark.asyncio
    async def test_quality_metrics_with_fallback(self):
        """Test QualityMetrics with fallback triggered."""
        metrics = QualityMetrics(
            compression_ratio=8.2,
            fact_retention_rate=0.96,
            coherence_score=0.94,
            contradiction_count=0,
            quality_score=0.95,
            quality_degraded=True,
            fallback_triggered=True,
            original_target_ratio=10.0,
            adjusted_target_ratio=8.0,
        )

        assert metrics.fallback_triggered is True
        assert metrics.original_target_ratio == 10.0
        assert metrics.adjusted_target_ratio == 8.0

    @pytest.mark.asyncio
    async def test_validation_thresholds(self, validator: QualityValidator):
        """Test validation threshold constants."""
        assert validator.MIN_FACT_RETENTION == 0.95
        assert validator.MIN_COHERENCE_SCORE == 0.90
        assert validator.MAX_CONTRADICTIONS == 0

    @pytest.mark.asyncio
    async def test_fallback_ratios_configuration(self, validator: QualityValidator):
        """Test fallback ratios are correctly configured."""
        assert 10.0 in validator.FALLBACK_RATIOS
        assert 5.0 in validator.FALLBACK_RATIOS
        assert validator.FALLBACK_RATIOS[10.0] == 8.0
        assert validator.FALLBACK_RATIOS[5.0] == 4.0

    @pytest.mark.asyncio
    async def test_validation_model_enforcement(self, validator: QualityValidator):
        """Test that only gpt-4.1-mini is used for validation."""
        assert validator.VALIDATION_MODEL == "gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_fact_present_in_content_exact_match(
        self, validator: QualityValidator
    ):
        """Test fact presence check with exact match."""
        fact = "JWT authentication"
        content = "Implemented JWT authentication with Redis storage"

        is_present = validator._fact_present_in_content(fact, content)

        assert is_present is True

    @pytest.mark.asyncio
    async def test_fact_present_in_content_partial_match(
        self, validator: QualityValidator
    ):
        """Test fact presence check with partial match."""
        fact = "Redis storage with 1-hour TTL"
        content = "Using Redis for token storage, TTL set to 1 hour"

        is_present = validator._fact_present_in_content(fact, content)

        # Should detect presence based on key terms (Redis, storage, TTL, hour)
        assert is_present is True

    @pytest.mark.asyncio
    async def test_fact_present_in_content_no_match(
        self, validator: QualityValidator
    ):
        """Test fact presence check with no match."""
        fact = "PostgreSQL database required"
        content = "Using Redis for token storage"

        is_present = validator._fact_present_in_content(fact, content)

        assert is_present is False

    @pytest.mark.asyncio
    async def test_estimate_content_quality_heuristic_ideal_ratio(
        self, validator: QualityValidator
    ):
        """Test content quality heuristic with ideal compression ratio."""
        original = "A" * 1000
        compressed = "A" * 150  # 15% of original (ideal range: 10-20%)

        quality = validator._estimate_content_quality_heuristic(original, compressed)

        assert quality == 0.9

    @pytest.mark.asyncio
    async def test_estimate_content_quality_heuristic_acceptable_ratio(
        self, validator: QualityValidator
    ):
        """Test content quality heuristic with acceptable compression ratio."""
        original = "A" * 1000
        compressed = "A" * 250  # 25% of original (acceptable range: 5-30%)

        quality = validator._estimate_content_quality_heuristic(original, compressed)

        assert quality == 0.75

    @pytest.mark.asyncio
    async def test_estimate_content_quality_heuristic_poor_ratio(
        self, validator: QualityValidator
    ):
        """Test content quality heuristic with poor compression ratio."""
        original = "A" * 1000
        compressed = "A" * 500  # 50% of original (too high)

        quality = validator._estimate_content_quality_heuristic(original, compressed)

        assert quality == 0.6
