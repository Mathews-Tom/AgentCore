"""
Integration tests for DecisionMaker (COMPASS ACE-2 - ACE-017).

Tests intervention decision making with mocked Portkey client responses.
Validates all trigger types, decision latency, and decision quality.

Coverage target: 95%+
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.ace.intervention.decision import DecisionMaker
from agentcore.ace.models.ace_models import (
    InterventionType,
    StrategicContext,
    TriggerSignal,
    TriggerType,
)
from agentcore.llm_gateway.exceptions import (
    LLMGatewayError,
    LLMGatewayTimeoutError,
)
from agentcore.llm_gateway.models import LLMResponse


def _create_mock_llm_response(intervention_data: dict[str, Any], response_id: str = "test") -> LLMResponse:
    """Helper to create mock LLM response with correct structure."""
    response_content = json.dumps(intervention_data)
    return LLMResponse(
        id=response_id,
        model="gpt-4.1",
        choices=[{
            "message": {"content": response_content},
            "finish_reason": "stop",
        }],
        usage={"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500},
        latency_ms=140,
    )


class TestDecisionMakerInit:
    """Test DecisionMaker initialization."""

    def test_init_success_default_params(self):
        """Test successful initialization with default parameters."""
        mock_client = MagicMock()
        decision_maker = DecisionMaker(llm_client=mock_client)

        assert decision_maker.model == "gpt-4.1"
        assert decision_maker.temperature == 0.3
        assert decision_maker.max_tokens == 500
        assert decision_maker.llm_client is mock_client

    def test_init_from_env_config(self, monkeypatch):
        """Test initialization with client created from environment config."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-key-123")
        monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "test-virtual-key")

        # Initialize without providing a client
        decision_maker = DecisionMaker()

        assert decision_maker.model == "gpt-4.1"
        assert decision_maker.llm_client is not None

    def test_init_success_custom_params(self):
        """Test successful initialization with custom parameters."""
        mock_client = MagicMock()
        decision_maker = DecisionMaker(
            llm_client=mock_client,
            model="gpt-5",
            temperature=0.5,
            max_tokens=1000,
        )

        assert decision_maker.model == "gpt-5"
        assert decision_maker.temperature == 0.5
        assert decision_maker.max_tokens == 1000

    def test_init_invalid_temperature_negative(self):
        """Test initialization with invalid temperature (negative)."""
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="temperature must be in"):
            DecisionMaker(llm_client=mock_client, temperature=-0.1)

    def test_init_invalid_temperature_too_high(self):
        """Test initialization with invalid temperature (>2.0)."""
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="temperature must be in"):
            DecisionMaker(llm_client=mock_client, temperature=2.5)

    def test_init_invalid_max_tokens(self):
        """Test initialization with invalid max_tokens."""
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="max_tokens must be >= 100"):
            DecisionMaker(llm_client=mock_client, max_tokens=50)


class TestDecisionMakerPerformanceDegradation:
    """Test decision making for performance degradation triggers."""

    @pytest.fixture
    def decision_maker(self):
        """DecisionMaker instance with mocked client."""
        mock_client = MagicMock()
        return DecisionMaker(llm_client=mock_client)

    @pytest.fixture
    def degradation_trigger(self):
        """Performance degradation trigger signal."""
        return TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["velocity_drop_below_threshold", "error_rate_spike"],
            rationale="Task velocity dropped 60% below baseline (0.6 -> 0.24 actions/min) and error rate increased 2.5x (0.1 -> 0.25)",
            confidence=0.92,
            metric_values={
                "baseline_velocity": 0.6,
                "current_velocity": 0.24,
                "velocity_ratio": 0.4,
                "baseline_error_rate": 0.1,
                "current_error_rate": 0.25,
                "error_rate_ratio": 2.5,
            },
        )

    @pytest.fixture
    def strategic_context(self):
        """Strategic context with moderate health."""
        return StrategicContext(
            relevant_stage_summaries=[
                "Planning stage completed with 85% confidence",
                "Execution stage showing 2.5x error rate increase",
            ],
            critical_facts=[
                "Task requires complex data transformation",
                "Agent has limited error recovery mechanisms",
            ],
            error_patterns=[
                "Repeated file parsing failures",
                "Timeout errors on API calls",
            ],
            successful_patterns=[
                "Initial planning was accurate",
            ],
            context_health_score=0.65,
        )

    @pytest.mark.asyncio
    async def test_decide_degradation_replan(
        self,
        decision_maker,
        degradation_trigger,
        strategic_context,
    ):
        """Test decision for degradation trigger -> REPLAN intervention."""
        llm_response = _create_mock_llm_response({
            "intervention_type": "REPLAN",
            "rationale": "Velocity dropped 60% with 2.5x error rate. Replanning will help agent reassess approach and address root causes of errors.",
            "confidence": 0.88,
            "expected_impact": "Velocity should return to baseline within 2-3 stages. Error rate should drop to <15%.",
            "alternative_interventions": [
                "reflect - useful but less urgent than addressing velocity",
                "context_refresh - won't fix root planning issues",
            ],
        }, "replan-1")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            degradation_trigger,
            strategic_context,
        )

        assert decision.intervention_type == InterventionType.REPLAN
        assert "velocity" in decision.rationale.lower()
        assert "error" in decision.rationale.lower()
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.expected_impact) >= 10
        assert len(decision.alternative_interventions) > 0
        assert "trigger_confidence" in decision.metadata
        assert "context_health" in decision.metadata
        assert "decision_latency_ms" in decision.metadata
        assert decision.metadata["decision_latency_ms"] < 200

    @pytest.mark.asyncio
    async def test_decide_degradation_reflect(
        self,
        decision_maker,
        degradation_trigger,
        strategic_context,
    ):
        """Test decision for degradation trigger -> REFLECT intervention."""
        strategic_context.error_patterns = [
            "Same error pattern repeated 5 times",
            "Agent not learning from failures",
        ]

        llm_response = _create_mock_llm_response({
            "intervention_type": "REFLECT",
            "rationale": "Error patterns show agent is not learning from repeated failures. Reflection will build meta-cognitive understanding.",
            "confidence": 0.82,
            "expected_impact": "Agent will identify failure patterns and adapt strategy. Error rate should decrease by 40%+.",
            "alternative_interventions": [
                "replan - considered but reflection better for learning",
            ],
        }, "reflect-1")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            degradation_trigger,
            strategic_context,
        )

        assert decision.intervention_type == InterventionType.REFLECT
        assert "reflection" in decision.rationale.lower() or "reflect" in decision.rationale.lower()
        assert 0.0 <= decision.confidence <= 1.0


class TestDecisionMakerErrorAccumulation:
    """Test decision making for error accumulation triggers."""

    @pytest.fixture
    def decision_maker(self):
        """DecisionMaker instance with mocked client."""
        mock_client = MagicMock()
        return DecisionMaker(llm_client=mock_client)

    @pytest.fixture
    def error_trigger(self):
        """Error accumulation trigger signal."""
        return TriggerSignal(
            trigger_type=TriggerType.ERROR_ACCUMULATION,
            signals=["high_error_count_in_stage", "compounding_error_pattern"],
            rationale="5 errors in execution stage (threshold: 3); Compounding error patterns detected: sequential",
            confidence=0.85,
            metric_values={
                "stage_error_count": 5.0,
                "pattern_count": 2.0,
            },
        )

    @pytest.fixture
    def strategic_context(self):
        """Strategic context with error focus."""
        return StrategicContext(
            relevant_stage_summaries=[
                "Execution stage has 5 errors",
                "Error patterns are compounding",
            ],
            critical_facts=[
                "Same error type repeated 3 times",
                "Agent failing to recover from errors",
            ],
            error_patterns=[
                "FileNotFoundError on same path",
                "API authentication failures",
            ],
            successful_patterns=[],
            context_health_score=0.45,
        )

    @pytest.mark.asyncio
    async def test_decide_error_reflect(
        self,
        decision_maker,
        error_trigger,
        strategic_context,
    ):
        """Test decision for error accumulation -> REFLECT intervention."""
        llm_response = _create_mock_llm_response({
            "intervention_type": "REFLECT",
            "rationale": "5 errors with compounding patterns indicate agent needs to reflect on failure patterns and learn.",
            "confidence": 0.90,
            "expected_impact": "Agent will identify root causes and avoid repeated failures. Error count should drop to <2 per stage.",
            "alternative_interventions": [
                "replan - may help but doesn't address learning",
            ],
        }, "reflect-2")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            error_trigger,
            strategic_context,
        )

        assert decision.intervention_type == InterventionType.REFLECT
        assert "error" in decision.rationale.lower()
        assert 0.0 <= decision.confidence <= 1.0


class TestDecisionMakerContextStaleness:
    """Test decision making for context staleness triggers."""

    @pytest.fixture
    def decision_maker(self):
        """DecisionMaker instance with mocked client."""
        mock_client = MagicMock()
        return DecisionMaker(llm_client=mock_client)

    @pytest.fixture
    def staleness_trigger(self):
        """Context staleness trigger signal."""
        return TriggerSignal(
            trigger_type=TriggerType.CONTEXT_STALENESS,
            signals=["context_age_exceeded", "low_retrieval_relevance"],
            rationale="Context not refreshed for 25 steps (threshold: 20); Memory retrieval relevance at 35.0% (threshold: 40%)",
            confidence=0.78,
            metric_values={
                "context_age": 25.0,
                "low_confidence_ratio": 0.55,
                "retrieval_relevance": 0.35,
            },
        )

    @pytest.fixture
    def strategic_context(self):
        """Strategic context with staleness indicators."""
        return StrategicContext(
            relevant_stage_summaries=[
                "Context not updated for 25 steps",
                "Memory retrieval returning irrelevant results",
            ],
            critical_facts=[
                "Task context has drifted from initial state",
                "Agent making decisions on outdated information",
            ],
            error_patterns=[
                "Decisions based on stale data",
            ],
            successful_patterns=[
                "Initial context was accurate",
            ],
            context_health_score=0.55,
        )

    @pytest.mark.asyncio
    async def test_decide_staleness_context_refresh(
        self,
        decision_maker,
        staleness_trigger,
        strategic_context,
    ):
        """Test decision for staleness trigger -> CONTEXT_REFRESH intervention."""
        llm_response = _create_mock_llm_response({
            "intervention_type": "CONTEXT_REFRESH",
            "rationale": "Context is 25 steps old with low retrieval relevance (35%). Refreshing will update stale information.",
            "confidence": 0.85,
            "expected_impact": "Context health score should increase to >0.8. Retrieval relevance should return to >60%.",
            "alternative_interventions": [
                "replan - could help but won't fix stale context",
            ],
        }, "refresh-1")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            staleness_trigger,
            strategic_context,
        )

        assert decision.intervention_type == InterventionType.CONTEXT_REFRESH
        assert "context" in decision.rationale.lower() or "refresh" in decision.rationale.lower()
        assert 0.0 <= decision.confidence <= 1.0


class TestDecisionMakerCapabilityMismatch:
    """Test decision making for capability mismatch triggers."""

    @pytest.fixture
    def decision_maker(self):
        """DecisionMaker instance with mocked client."""
        mock_client = MagicMock()
        return DecisionMaker(llm_client=mock_client)

    @pytest.fixture
    def capability_trigger(self):
        """Capability mismatch trigger signal."""
        return TriggerSignal(
            trigger_type=TriggerType.CAPABILITY_MISMATCH,
            signals=["low_capability_coverage", "high_action_failure_rate"],
            rationale="Only 40.0% capability coverage (3 missing: data_processing, file_handling, api_integration...); Action failure rate at 65.0% (threshold: 50%)",
            confidence=0.88,
            metric_values={
                "capability_coverage": 0.4,
                "missing_count": 3.0,
                "action_failure_rate": 0.65,
            },
        )

    @pytest.fixture
    def strategic_context(self):
        """Strategic context with capability issues."""
        return StrategicContext(
            relevant_stage_summaries=[
                "Task requires data processing capabilities",
                "Agent has 40% capability coverage",
            ],
            critical_facts=[
                "Task needs: data_processing, file_handling, api_integration",
                "Agent only has: basic_reasoning, text_generation",
            ],
            error_patterns=[
                "Tool not found errors",
                "Capability not available errors",
            ],
            successful_patterns=[],
            context_health_score=0.50,
        )

    @pytest.mark.asyncio
    async def test_decide_capability_switch(
        self,
        decision_maker,
        capability_trigger,
        strategic_context,
    ):
        """Test decision for capability mismatch -> CAPABILITY_SWITCH intervention."""
        llm_response = _create_mock_llm_response({
            "intervention_type": "CAPABILITY_SWITCH",
            "rationale": "Only 40% capability coverage with 65% action failure rate. Switching to agent with required capabilities will resolve mismatch.",
            "confidence": 0.92,
            "expected_impact": "Action failure rate should drop to <15%. Capability coverage should reach >90%.",
            "alternative_interventions": [
                "replan - won't fix fundamental capability gap",
            ],
        }, "switch-1")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            capability_trigger,
            strategic_context,
        )

        assert decision.intervention_type == InterventionType.CAPABILITY_SWITCH
        assert "capability" in decision.rationale.lower() or "capabilities" in decision.rationale.lower()
        assert 0.0 <= decision.confidence <= 1.0


class TestDecisionMakerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def decision_maker(self):
        """DecisionMaker instance with mocked client."""
        mock_client = MagicMock()
        return DecisionMaker(llm_client=mock_client)

    @pytest.fixture
    def minimal_trigger(self):
        """Minimal trigger signal."""
        return TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["test_signal"],
            rationale="Test rationale for minimal trigger",
            confidence=0.5,
            metric_values={},
        )

    @pytest.fixture
    def minimal_context(self):
        """Minimal strategic context."""
        return StrategicContext(
            context_health_score=0.5,
        )

    @pytest.mark.asyncio
    async def test_decide_with_minimal_context(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision with minimal context (empty lists)."""
        llm_response = _create_mock_llm_response({
            "intervention_type": "REPLAN",
            "rationale": "Performance degradation detected with limited context. Replanning is safest option.",
            "confidence": 0.70,
            "expected_impact": "Should stabilize performance.",
            "alternative_interventions": [],
        }, "minimal-1")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            minimal_trigger,
            minimal_context,
        )

        assert decision.intervention_type == InterventionType.REPLAN
        assert len(decision.rationale) >= 10
        assert len(decision.expected_impact) >= 10

    @pytest.mark.asyncio
    async def test_decide_llm_timeout_error(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision when LLM request times out."""
        decision_maker.llm_client.complete = AsyncMock(
            side_effect=LLMGatewayTimeoutError("Request timed out", timeout=60.0)
        )

        with pytest.raises(LLMGatewayTimeoutError):
            await decision_maker.decide_intervention(
                minimal_trigger,
                minimal_context,
            )

    @pytest.mark.asyncio
    async def test_decide_llm_generic_error(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision when LLM request fails."""
        decision_maker.llm_client.complete = AsyncMock(
            side_effect=LLMGatewayError("Connection error")
        )

        with pytest.raises(LLMGatewayError):
            await decision_maker.decide_intervention(
                minimal_trigger,
                minimal_context,
            )

    @pytest.mark.asyncio
    async def test_decide_invalid_json_response(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision with invalid JSON response."""
        llm_response = LLMResponse(
            id="invalid-1",
            model="gpt-4.1",
            choices=[{
                "message": {"content": "This is not JSON at all"},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            latency_ms=100,
        )

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        with pytest.raises(ValueError, match="Invalid JSON response"):
            await decision_maker.decide_intervention(
                minimal_trigger,
                minimal_context,
            )

    @pytest.mark.asyncio
    async def test_decide_missing_required_field(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision with missing required field in response."""
        # Missing 'rationale' field
        invalid_data = {
            "intervention_type": "REPLAN",
            "confidence": 0.80,
            "expected_impact": "Should improve performance.",
        }

        llm_response = LLMResponse(
            id="missing-field-1",
            model="gpt-4.1",
            choices=[{
                "message": {"content": json.dumps(invalid_data)},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            latency_ms=100,
        )

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        with pytest.raises(ValueError, match="Missing required fields"):
            await decision_maker.decide_intervention(
                minimal_trigger,
                minimal_context,
            )

    @pytest.mark.asyncio
    async def test_decide_invalid_intervention_type(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision with invalid intervention type."""
        invalid_data = {
            "intervention_type": "INVALID_TYPE",
            "rationale": "This is a test",
            "confidence": 0.80,
            "expected_impact": "Should work",
        }

        llm_response = LLMResponse(
            id="invalid-type-1",
            model="gpt-4.1",
            choices=[{
                "message": {"content": json.dumps(invalid_data)},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            latency_ms=100,
        )

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        with pytest.raises(ValueError, match="Invalid intervention_type"):
            await decision_maker.decide_intervention(
                minimal_trigger,
                minimal_context,
            )

    @pytest.mark.asyncio
    async def test_decide_markdown_json_response(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision with JSON wrapped in markdown code blocks."""
        markdown_content = """```json
{
    "intervention_type": "REPLAN",
    "rationale": "Test rationale with markdown wrapper",
    "confidence": 0.75,
    "expected_impact": "Should handle markdown",
    "alternative_interventions": []
}
```"""

        llm_response = LLMResponse(
            id="markdown-1",
            model="gpt-4.1",
            choices=[{
                "message": {"content": markdown_content},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
            latency_ms=120,
        )

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            minimal_trigger,
            minimal_context,
        )

        assert decision.intervention_type == InterventionType.REPLAN
        assert "markdown" in decision.rationale.lower()

    @pytest.mark.asyncio
    async def test_decide_markdown_without_json_prefix(
        self,
        decision_maker,
        minimal_trigger,
        minimal_context,
    ):
        """Test decision with JSON in markdown without 'json' prefix."""
        markdown_content = """```
{
    "intervention_type": "REPLAN",
    "rationale": "Test without json prefix",
    "confidence": 0.72,
    "expected_impact": "Should still work",
    "alternative_interventions": []
}
```"""

        llm_response = LLMResponse(
            id="markdown-2",
            model="gpt-4.1",
            choices=[{
                "message": {"content": markdown_content},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 300, "completion_tokens": 70, "total_tokens": 370},
            latency_ms=125,
        )

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            minimal_trigger,
            minimal_context,
        )

        assert decision.intervention_type == InterventionType.REPLAN
        assert "prefix" in decision.rationale.lower()


class TestDecisionMakerLatencyValidation:
    """Test decision latency validation (<200ms target)."""

    @pytest.fixture
    def decision_maker(self):
        """DecisionMaker instance with mocked client."""
        mock_client = MagicMock()
        return DecisionMaker(llm_client=mock_client)

    @pytest.fixture
    def test_trigger(self):
        """Test trigger signal."""
        return TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["test"],
            rationale="Test latency validation",
            confidence=0.8,
            metric_values={},
        )

    @pytest.fixture
    def test_context(self):
        """Test strategic context."""
        return StrategicContext(context_health_score=0.7)

    @pytest.mark.asyncio
    async def test_decision_latency_under_200ms(
        self,
        decision_maker,
        test_trigger,
        test_context,
    ):
        """Test that decision latency is under 200ms (p95 target)."""
        llm_response = _create_mock_llm_response({
            "intervention_type": "REPLAN",
            "rationale": "Quick decision test",
            "confidence": 0.80,
            "expected_impact": "Fast response",
        }, "latency-1")

        decision_maker.llm_client.complete = AsyncMock(return_value=llm_response)

        decision = await decision_maker.decide_intervention(
            test_trigger,
            test_context,
        )

        assert "decision_latency_ms" in decision.metadata
        latency_ms = decision.metadata["decision_latency_ms"]
        assert latency_ms < 200  # <200ms target (p95)
        assert latency_ms >= 0  # Sanity check
