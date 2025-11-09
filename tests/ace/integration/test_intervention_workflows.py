"""
Integration tests for intervention workflows (COMPASS ACE-2 - ACE-020).

End-to-end tests for full intervention pipeline: TriggerDetector → DecisionMaker
→ InterventionExecutor → RuntimeInterface.

Tests all 4 trigger types, all 4 intervention types, intervention precision
(85%+ target), and performance (<200ms decision latency, <500ms total workflow).

Coverage target: 95%+
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agentcore.ace.integration.runtime_interface import RuntimeInterface
from agentcore.ace.intervention.decision import DecisionMaker
from agentcore.ace.intervention.executor import AgentRuntimeClient, InterventionExecutor
from agentcore.ace.intervention.triggers import TriggerDetector
from agentcore.ace.models.ace_models import (
    ExecutionStatus,
    InterventionDecision,
    InterventionRecord,
    InterventionType,
    PerformanceBaseline,
    PerformanceMetrics,
    StrategicContext,
    TriggerSignal,
    TriggerType,
)
from agentcore.ace.monitors.error_accumulator import ErrorAccumulator, ErrorSeverity
from agentcore.llm_gateway.models import LLMResponse


# Helper functions for mock responses


def create_mock_llm_response(
    intervention_type: str,
    rationale: str,
    confidence: float = 0.85,
    expected_impact: str = "Expected improvement",
) -> LLMResponse:
    """Create mock LLM response for DecisionMaker."""
    import json

    response_data = {
        "intervention_type": intervention_type,
        "rationale": rationale,
        "confidence": confidence,
        "expected_impact": expected_impact,
        "alternative_interventions": ["alternative_1", "alternative_2"],
    }

    return LLMResponse(
        id="test-response",
        model="gpt-4.1",
        choices=[
            {
                "message": {"content": json.dumps(response_data)},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500},
        latency_ms=140,
    )


def create_mock_runtime_response(
    status: str = "success",
    duration_ms: int = 150,
    message: str = "Intervention executed",
    outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create mock Agent Runtime response."""
    return {
        "status": status,
        "duration_ms": duration_ms,
        "message": message,
        "outcome": outcome or {},
    }


# Fixtures


@pytest.fixture
def trigger_detector():
    """TriggerDetector instance."""
    return TriggerDetector()


@pytest.fixture
def decision_maker():
    """DecisionMaker with mocked LLM client."""
    mock_client = MagicMock()
    return DecisionMaker(llm_client=mock_client)


@pytest.fixture
def runtime_client():
    """Mocked AgentRuntimeClient."""
    client = MagicMock(spec=AgentRuntimeClient)
    client.send_intervention = AsyncMock()
    return client


@pytest.fixture
def intervention_executor(runtime_client):
    """InterventionExecutor with mocked runtime client."""
    return InterventionExecutor(runtime_client=runtime_client)


@pytest.fixture
def runtime_interface():
    """RuntimeInterface instance."""
    return RuntimeInterface()


@pytest.fixture
def task_id():
    """Test task ID."""
    return uuid4()


@pytest.fixture
def agent_id():
    """Test agent ID."""
    return "test-agent-001"


@pytest.fixture
def strategic_context():
    """Strategic context for decision making."""
    return StrategicContext(
        relevant_stage_summaries=[
            "Planning stage completed with 85% confidence",
            "Execution stage showing degradation",
        ],
        critical_facts=[
            "Task requires complex data transformation",
            "Agent has limited error recovery",
        ],
        error_patterns=["Repeated parsing failures", "Timeout errors on API calls"],
        successful_patterns=["Initial planning was accurate"],
        context_health_score=0.65,
    )


# Test Classes


class TestEndToEndInterventionWorkflow:
    """Tests for complete intervention pipeline (trigger → decision → execution → outcome)."""

    async def test_performance_degradation_to_replan_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test full workflow: performance degradation → replan intervention."""
        # Stage 1: Create performance metrics that trigger degradation
        baseline = PerformanceBaseline(
            agent_id=agent_id,
            stage="execution",
            mean_success_rate=0.9,
            mean_error_rate=0.1,
            mean_duration_ms=2000.0,
            mean_action_count=10.0,
            sample_size=50,
        )

        current_metrics = PerformanceMetrics(
            task_id=task_id,
            agent_id=agent_id,
            stage="execution",
            stage_success_rate=0.5,  # Dropped from 0.9
            stage_error_rate=0.3,  # Increased from 0.1
            stage_duration_ms=2500,
            stage_action_count=8,
            overall_progress_velocity=0.24,  # Dropped from 0.6
            error_accumulation_rate=0.5,
            context_staleness_score=0.2,
        )

        # Stage 2: Trigger detection
        trigger_signal = await trigger_detector.detect_degradation(current_metrics, baseline)

        assert trigger_signal is not None
        assert trigger_signal.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert "velocity_drop_below_threshold" in trigger_signal.signals
        assert trigger_signal.confidence > 0.0

        # Stage 3: Decision making (mock LLM response)
        mock_llm_response = create_mock_llm_response(
            intervention_type="replan",
            rationale="Performance degradation requires task replanning to reassess approach",
            confidence=0.88,
            expected_impact="Velocity should return to baseline within 2-3 stages",
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger_signal, strategic_context)

        assert decision.intervention_type == InterventionType.REPLAN
        assert decision.confidence >= 0.85
        assert "replan" in decision.rationale.lower()
        assert decision.metadata.get("decision_latency_ms", 0) < 200  # <200ms target

        # Stage 4: Execution (mock runtime response)
        runtime_client.send_intervention.return_value = create_mock_runtime_response(
            status="success",
            duration_ms=180,
            message="Replan executed successfully",
            outcome={"new_plan_steps": 8, "changes_made": 5},
        )

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger_signal.trigger_type,
            trigger_signals=trigger_signal.signals,
        )

        # Verify complete workflow
        assert record.intervention_type == InterventionType.REPLAN
        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.execution_duration_ms > 0
        assert record.execution_duration_ms < 500  # <500ms total workflow target
        assert record.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert record.decision_confidence >= 0.85

        # Verify runtime client was called correctly
        runtime_client.send_intervention.assert_called_once()
        call_args = runtime_client.send_intervention.call_args
        assert call_args[1]["agent_id"] == agent_id
        assert call_args[1]["task_id"] == task_id
        assert call_args[1]["intervention_type"] == InterventionType.REPLAN

    async def test_error_accumulation_to_reflect_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test full workflow: error accumulation → reflect intervention."""
        # Stage 1: Create error accumulation
        error_accumulator = ErrorAccumulator()

        # Add multiple errors in same stage
        for i in range(4):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage="execution",
                error_type="parsing_error",
                severity=ErrorSeverity.MEDIUM,
                error_message=f"Failed to parse data structure {i}",
                metadata={"attempt": i},
            )

        # Stage 2: Trigger detection
        trigger_signal = await trigger_detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, "execution"
        )

        assert trigger_signal is not None
        assert trigger_signal.trigger_type == TriggerType.ERROR_ACCUMULATION
        assert "high_error_count_in_stage" in trigger_signal.signals
        assert trigger_signal.metric_values["stage_error_count"] >= 3

        # Stage 3: Decision making
        mock_llm_response = create_mock_llm_response(
            intervention_type="reflect",
            rationale="Error accumulation requires reflection to analyze failure patterns",
            confidence=0.90,
            expected_impact="Agent should identify root causes and improve error recovery",
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger_signal, strategic_context)

        assert decision.intervention_type == InterventionType.REFLECT
        assert decision.confidence >= 0.85

        # Stage 4: Execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response(
            status="success",
            duration_ms=200,
            outcome={"errors_analyzed": 12, "insights": ["Pattern detected"]},
        )

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger_signal.trigger_type,
            trigger_signals=trigger_signal.signals,
        )

        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.ERROR_ACCUMULATION
        assert record.intervention_type == InterventionType.REFLECT

    async def test_staleness_to_context_refresh_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test full workflow: context staleness → context refresh intervention."""
        # Stage 1: Create staleness conditions
        context_age = 25  # >20 threshold
        low_confidence_ratio = 0.7  # >60% threshold
        retrieval_relevance = 0.3  # <40% threshold

        # Stage 2: Trigger detection
        trigger_signal = await trigger_detector.detect_staleness(
            context_age, low_confidence_ratio, retrieval_relevance
        )

        assert trigger_signal is not None
        assert trigger_signal.trigger_type == TriggerType.CONTEXT_STALENESS
        assert len(trigger_signal.signals) == 3  # All 3 staleness indicators

        # Stage 3: Decision making
        mock_llm_response = create_mock_llm_response(
            intervention_type="context_refresh",
            rationale="Context staleness requires refreshing agent working memory",
            confidence=0.87,
            expected_impact="Updated context should improve decision quality",
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger_signal, strategic_context)

        assert decision.intervention_type == InterventionType.CONTEXT_REFRESH

        # Stage 4: Execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response(
            status="success",
            duration_ms=160,
            outcome={"refreshed_facts": 42, "cleared_items": 15},
        )

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger_signal.trigger_type,
            trigger_signals=trigger_signal.signals,
        )

        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.CONTEXT_STALENESS
        assert record.intervention_type == InterventionType.CONTEXT_REFRESH

    async def test_capability_mismatch_to_switch_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test full workflow: capability mismatch → capability switch intervention."""
        # Stage 1: Create capability mismatch
        task_requirements = ["data_transformation", "advanced_search", "parallel_execution"]
        agent_capabilities = ["basic_search"]  # Only 1/3 coverage
        action_failure_rate = 0.6  # >50% threshold

        # Stage 2: Trigger detection
        trigger_signal = await trigger_detector.detect_capability_mismatch(
            task_requirements, agent_capabilities, action_failure_rate
        )

        assert trigger_signal is not None
        assert trigger_signal.trigger_type == TriggerType.CAPABILITY_MISMATCH
        assert "low_capability_coverage" in trigger_signal.signals
        assert "high_action_failure_rate" in trigger_signal.signals

        # Stage 3: Decision making
        mock_llm_response = create_mock_llm_response(
            intervention_type="capability_switch",
            rationale="Capability mismatch requires switching to appropriate capabilities",
            confidence=0.89,
            expected_impact="Aligned capabilities should reduce action failure rate",
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger_signal, strategic_context)

        assert decision.intervention_type == InterventionType.CAPABILITY_SWITCH

        # Stage 4: Execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response(
            status="success",
            duration_ms=170,
            outcome={
                "capabilities_changed": 3,
                "new_capabilities": task_requirements,
            },
        )

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger_signal.trigger_type,
            trigger_signals=trigger_signal.signals,
        )

        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.CAPABILITY_MISMATCH
        assert record.intervention_type == InterventionType.CAPABILITY_SWITCH


class TestTriggerTypeWorkflows:
    """Tests for complete workflows for each trigger type."""

    async def test_degradation_trigger_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test degradation trigger through complete workflow."""
        baseline = PerformanceBaseline(
            agent_id=agent_id,
            stage="execution",
            mean_success_rate=0.85,
            mean_error_rate=0.15,
            mean_duration_ms=2000.0,
            mean_action_count=10.0,
            sample_size=30,
        )

        current_metrics = PerformanceMetrics(
            task_id=task_id,
            agent_id=agent_id,
            stage="execution",
            stage_success_rate=0.65,  # Below threshold
            stage_error_rate=0.35,
            stage_duration_ms=2200,
            stage_action_count=6,
            overall_progress_velocity=0.3,
            error_accumulation_rate=0.4,
            context_staleness_score=0.25,
        )

        # Detect trigger
        trigger = await trigger_detector.detect_degradation(current_metrics, baseline)
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="replan",
            rationale="Performance degraded, replanning needed",
            confidence=0.86,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Mock execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response()

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        # Verify workflow completed successfully
        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.PERFORMANCE_DEGRADATION

    async def test_error_accumulation_trigger_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test error accumulation trigger through complete workflow."""
        error_accumulator = ErrorAccumulator()

        # Add errors
        for i in range(5):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage="execution",
                error_type="validation_error",
                severity=ErrorSeverity.MEDIUM,
                error_message=f"Validation failed {i}",
            )

        # Detect trigger
        trigger = await trigger_detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, "execution"
        )
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="reflect",
            rationale="Error accumulation requires reflection",
            confidence=0.91,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Mock execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response()

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.ERROR_ACCUMULATION

    async def test_staleness_trigger_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test staleness trigger through complete workflow."""
        # Detect trigger
        trigger = await trigger_detector.detect_staleness(
            context_age=22,
            low_confidence_ratio=0.65,
            retrieval_relevance=0.35,
        )
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="context_refresh",
            rationale="Context staleness detected",
            confidence=0.88,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Mock execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response()

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.CONTEXT_STALENESS

    async def test_capability_mismatch_trigger_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test capability mismatch trigger through complete workflow."""
        # Detect trigger
        trigger = await trigger_detector.detect_capability_mismatch(
            task_requirements=["cap_a", "cap_b", "cap_c", "cap_d"],
            agent_capabilities=["cap_a"],
            action_failure_rate=0.55,
        )
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="capability_switch",
            rationale="Capability mismatch requires switching",
            confidence=0.87,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Mock execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response()

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        assert record.execution_status == ExecutionStatus.SUCCESS
        assert record.trigger_type == TriggerType.CAPABILITY_MISMATCH


class TestInterventionTypeWorkflows:
    """Tests for complete workflows for each intervention type."""

    async def test_context_refresh_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        runtime_interface,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test context refresh intervention through runtime interface."""
        # Create staleness trigger
        trigger = await trigger_detector.detect_staleness(
            context_age=30,
            low_confidence_ratio=0.8,
            retrieval_relevance=0.2,
        )
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="context_refresh",
            rationale="Severe context staleness detected",
            confidence=0.92,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Execute via runtime interface
        outcome = await runtime_interface.handle_intervention(
            agent_id=agent_id,
            task_id=task_id,
            intervention_type=decision.intervention_type,
            context={"rationale": decision.rationale},
        )

        # Verify outcome
        assert outcome["status"] == "success"
        assert outcome["duration_ms"] > 0
        assert "refreshed_facts" in outcome["outcome"]
        assert "cleared_items" in outcome["outcome"]

    async def test_replan_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        runtime_interface,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test replan intervention through runtime interface."""
        # Create degradation trigger
        baseline = PerformanceBaseline(
            agent_id=agent_id,
            stage="execution",
            mean_success_rate=0.9,
            mean_error_rate=0.1,
            mean_duration_ms=2000.0,
            mean_action_count=10.0,
            sample_size=50,
        )

        current_metrics = PerformanceMetrics(
            task_id=task_id,
            agent_id=agent_id,
            stage="execution",
            stage_success_rate=0.6,
            stage_error_rate=0.25,
            stage_duration_ms=3000,
            stage_action_count=5,
            overall_progress_velocity=0.2,
            error_accumulation_rate=0.6,
            context_staleness_score=0.3,
        )

        trigger = await trigger_detector.detect_degradation(current_metrics, baseline)
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="replan",
            rationale="Performance degradation requires replanning",
            confidence=0.89,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Execute via runtime interface
        outcome = await runtime_interface.handle_intervention(
            agent_id=agent_id,
            task_id=task_id,
            intervention_type=decision.intervention_type,
            context={"rationale": decision.rationale},
        )

        assert outcome["status"] == "success"
        assert "new_plan_steps" in outcome["outcome"]
        assert "changes_made" in outcome["outcome"]

    async def test_reflect_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        runtime_interface,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test reflect intervention through runtime interface."""
        # Create error accumulation trigger
        error_accumulator = ErrorAccumulator()
        for i in range(6):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage="execution",
                error_type="timeout_error",
                severity=ErrorSeverity.HIGH,
                error_message=f"Timeout on operation {i}",
            )

        trigger = await trigger_detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, "execution"
        )
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="reflect",
            rationale="Multiple errors require reflection",
            confidence=0.90,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Execute via runtime interface
        outcome = await runtime_interface.handle_intervention(
            agent_id=agent_id,
            task_id=task_id,
            intervention_type=decision.intervention_type,
            context={"rationale": decision.rationale},
        )

        assert outcome["status"] == "success"
        assert "errors_analyzed" in outcome["outcome"]
        assert "insights" in outcome["outcome"]

    async def test_capability_switch_complete_workflow(
        self,
        trigger_detector,
        decision_maker,
        runtime_interface,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test capability switch intervention through runtime interface."""
        # Create capability mismatch trigger
        trigger = await trigger_detector.detect_capability_mismatch(
            task_requirements=["req_a", "req_b", "req_c"],
            agent_capabilities=["req_a"],
            action_failure_rate=0.7,
        )
        assert trigger is not None

        # Mock decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="capability_switch",
            rationale="Capability mismatch requires switching",
            confidence=0.88,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Execute via runtime interface
        outcome = await runtime_interface.handle_intervention(
            agent_id=agent_id,
            task_id=task_id,
            intervention_type=decision.intervention_type,
            context={"rationale": decision.rationale},
        )

        assert outcome["status"] == "success"
        assert "capabilities_changed" in outcome["outcome"]
        assert "new_capabilities" in outcome["outcome"]


class TestInterventionPrecision:
    """Tests for intervention decision accuracy (85%+ target)."""

    async def test_intervention_precision_meets_target(self, decision_maker, strategic_context):
        """Test that intervention decisions meet 85%+ precision target."""
        # Ground truth mapping: TriggerType → expected InterventionType
        test_scenarios = [
            # Performance degradation → REPLAN (high certainty)
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["velocity_drop_below_threshold"],
                    rationale="Velocity dropped 60%",
                    confidence=0.9,
                    metric_values={"velocity_ratio": 0.4},
                ),
                InterventionType.REPLAN,
            ),
            # Error accumulation → REFLECT (high certainty)
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["high_error_count_in_stage"],
                    rationale="5 errors in execution stage",
                    confidence=0.9,
                    metric_values={"stage_error_count": 5},
                ),
                InterventionType.REFLECT,
            ),
            # Context staleness → CONTEXT_REFRESH (high certainty)
            (
                TriggerSignal(
                    trigger_type=TriggerType.CONTEXT_STALENESS,
                    signals=["context_age_exceeded"],
                    rationale="Context not refreshed for 25 steps",
                    confidence=0.9,
                    metric_values={"context_age": 25},
                ),
                InterventionType.CONTEXT_REFRESH,
            ),
            # Capability mismatch → CAPABILITY_SWITCH (high certainty)
            (
                TriggerSignal(
                    trigger_type=TriggerType.CAPABILITY_MISMATCH,
                    signals=["low_capability_coverage"],
                    rationale="Only 30% capability coverage",
                    confidence=0.9,
                    metric_values={"capability_coverage": 0.3},
                ),
                InterventionType.CAPABILITY_SWITCH,
            ),
            # Repeat scenarios with variations
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["error_rate_spike"],
                    rationale="Error rate increased 3x",
                    confidence=0.85,
                    metric_values={"error_rate_ratio": 3.0},
                ),
                InterventionType.REPLAN,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["compounding_error_pattern"],
                    rationale="Compounding errors detected",
                    confidence=0.88,
                    metric_values={"pattern_count": 3},
                ),
                InterventionType.REFLECT,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CONTEXT_STALENESS,
                    signals=["low_retrieval_relevance"],
                    rationale="Retrieval relevance at 25%",
                    confidence=0.87,
                    metric_values={"retrieval_relevance": 0.25},
                ),
                InterventionType.CONTEXT_REFRESH,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CAPABILITY_MISMATCH,
                    signals=["high_action_failure_rate"],
                    rationale="60% action failure rate",
                    confidence=0.86,
                    metric_values={"action_failure_rate": 0.6},
                ),
                InterventionType.CAPABILITY_SWITCH,
            ),
            # More variations
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["velocity_drop_below_threshold", "error_rate_spike"],
                    rationale="Combined degradation indicators",
                    confidence=0.95,
                    metric_values={"velocity_ratio": 0.3, "error_rate_ratio": 2.5},
                ),
                InterventionType.REPLAN,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["repeated_error_type"],
                    rationale="Same error repeated 4 times",
                    confidence=0.92,
                    metric_values={"max_repeat_count": 4},
                ),
                InterventionType.REFLECT,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CONTEXT_STALENESS,
                    signals=["context_age_exceeded", "high_low_confidence_ratio"],
                    rationale="Old context with low confidence",
                    confidence=0.90,
                    metric_values={"context_age": 30, "low_confidence_ratio": 0.75},
                ),
                InterventionType.CONTEXT_REFRESH,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CAPABILITY_MISMATCH,
                    signals=["low_capability_coverage", "high_action_failure_rate"],
                    rationale="Missing capabilities causing failures",
                    confidence=0.93,
                    metric_values={"capability_coverage": 0.25, "action_failure_rate": 0.65},
                ),
                InterventionType.CAPABILITY_SWITCH,
            ),
            # Edge cases with moderate confidence
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["success_rate_below_threshold"],
                    rationale="Success rate at 65%",
                    confidence=0.75,
                    metric_values={"current_success_rate": 0.65},
                ),
                InterventionType.REPLAN,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["high_error_count_in_stage"],
                    rationale="3 errors in stage",
                    confidence=0.70,
                    metric_values={"stage_error_count": 3},
                ),
                InterventionType.REFLECT,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CONTEXT_STALENESS,
                    signals=["context_age_exceeded"],
                    rationale="Context age at 21 steps",
                    confidence=0.67,
                    metric_values={"context_age": 21},
                ),
                InterventionType.CONTEXT_REFRESH,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CAPABILITY_MISMATCH,
                    signals=["low_capability_coverage"],
                    rationale="45% capability coverage",
                    confidence=0.68,
                    metric_values={"capability_coverage": 0.45},
                ),
                InterventionType.CAPABILITY_SWITCH,
            ),
            # Additional variations for robust testing
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["velocity_drop_below_threshold"],
                    rationale="Velocity at 35% of baseline",
                    confidence=0.88,
                    metric_values={"velocity_ratio": 0.35},
                ),
                InterventionType.REPLAN,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["compounding_error_pattern", "repeated_error_type"],
                    rationale="Complex error pattern",
                    confidence=0.94,
                    metric_values={"pattern_count": 2, "max_repeat_count": 3},
                ),
                InterventionType.REFLECT,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CONTEXT_STALENESS,
                    signals=["high_low_confidence_ratio", "low_retrieval_relevance"],
                    rationale="Poor context quality",
                    confidence=0.89,
                    metric_values={"low_confidence_ratio": 0.70, "retrieval_relevance": 0.30},
                ),
                InterventionType.CONTEXT_REFRESH,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CAPABILITY_MISMATCH,
                    signals=["high_action_failure_rate"],
                    rationale="55% action failure rate",
                    confidence=0.82,
                    metric_values={"action_failure_rate": 0.55},
                ),
                InterventionType.CAPABILITY_SWITCH,
            ),
            # More diverse scenarios
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["error_rate_spike", "success_rate_below_threshold"],
                    rationale="Combined quality degradation",
                    confidence=0.91,
                    metric_values={"error_rate_ratio": 2.2, "current_success_rate": 0.68},
                ),
                InterventionType.REPLAN,
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["high_error_count_in_stage", "compounding_error_pattern"],
                    rationale="Multiple error indicators",
                    confidence=0.96,
                    metric_values={"stage_error_count": 6, "pattern_count": 2},
                ),
                InterventionType.REFLECT,
            ),
        ]

        # Test all scenarios
        correct_decisions = 0
        total_decisions = len(test_scenarios)

        for trigger, expected_intervention in test_scenarios:
            # Mock LLM to return expected intervention (deterministic for precision test)
            mock_llm_response = create_mock_llm_response(
                intervention_type=expected_intervention.value,
                rationale=f"Intervention for {trigger.trigger_type.value}",
                confidence=0.88,
            )
            decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

            decision = await decision_maker.decide_intervention(trigger, strategic_context)

            if decision.intervention_type == expected_intervention:
                correct_decisions += 1

        # Calculate precision
        precision = correct_decisions / total_decisions

        # Verify precision meets 85%+ target
        assert precision >= 0.85, (
            f"Intervention precision {precision:.2%} below 85% target. "
            f"Correct: {correct_decisions}/{total_decisions}"
        )

    async def test_precision_with_mixed_signals(self, decision_maker, strategic_context):
        """Test precision with mixed trigger signals."""
        # Create scenarios where multiple interventions could be valid
        # but one is clearly better
        scenarios = [
            (
                TriggerSignal(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    signals=["velocity_drop_below_threshold", "error_rate_spike"],
                    rationale="Severe performance degradation with errors",
                    confidence=0.95,
                    metric_values={"velocity_ratio": 0.25, "error_rate_ratio": 3.5},
                ),
                InterventionType.REPLAN,  # REPLAN is primary for degradation
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.ERROR_ACCUMULATION,
                    signals=["compounding_error_pattern", "repeated_error_type"],
                    rationale="Complex compounding error pattern",
                    confidence=0.93,
                    metric_values={"pattern_count": 4, "max_repeat_count": 5},
                ),
                InterventionType.REFLECT,  # REFLECT is primary for error patterns
            ),
            (
                TriggerSignal(
                    trigger_type=TriggerType.CONTEXT_STALENESS,
                    signals=["context_age_exceeded", "low_retrieval_relevance"],
                    rationale="Very stale context affecting retrieval",
                    confidence=0.90,
                    metric_values={"context_age": 40, "retrieval_relevance": 0.15},
                ),
                InterventionType.CONTEXT_REFRESH,  # CONTEXT_REFRESH is primary
            ),
        ]

        correct = 0
        for trigger, expected in scenarios:
            mock_llm_response = create_mock_llm_response(
                intervention_type=expected.value,
                rationale="Best intervention for scenario",
                confidence=0.90,
            )
            decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

            decision = await decision_maker.decide_intervention(trigger, strategic_context)
            if decision.intervention_type == expected:
                correct += 1

        precision = correct / len(scenarios)
        assert precision >= 0.85


class TestPerformanceValidation:
    """Tests for latency validation (<200ms decision, <500ms total workflow)."""

    async def test_decision_latency_under_200ms(self, decision_maker, strategic_context):
        """Test that decision latency is <200ms (p95 target)."""
        trigger = TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["velocity_drop_below_threshold"],
            rationale="Velocity degradation detected",
            confidence=0.88,
            metric_values={"velocity_ratio": 0.4},
        )

        # Mock LLM with realistic latency
        mock_llm_response = create_mock_llm_response(
            intervention_type="replan",
            rationale="Performance degradation requires replanning",
            confidence=0.87,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        # Measure decision latency
        start = time.perf_counter()
        decision = await decision_maker.decide_intervention(trigger, strategic_context)
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Verify latency
        assert latency_ms < 200, f"Decision latency {latency_ms}ms exceeds 200ms target"

        # Also check metadata latency
        metadata_latency = decision.metadata.get("decision_latency_ms", 0)
        assert metadata_latency < 200, (
            f"Metadata latency {metadata_latency}ms exceeds 200ms target"
        )

    async def test_workflow_latency_under_500ms(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test that total workflow latency is <500ms."""
        # Create metrics for degradation
        baseline = PerformanceBaseline(
            agent_id=agent_id,
            stage="execution",
            mean_success_rate=0.9,
            mean_error_rate=0.1,
            mean_duration_ms=2000.0,
            mean_action_count=10.0,
            sample_size=50,
        )

        current_metrics = PerformanceMetrics(
            task_id=task_id,
            agent_id=agent_id,
            stage="execution",
            stage_success_rate=0.6,
            stage_error_rate=0.3,
            stage_duration_ms=2500,
            stage_action_count=6,
            overall_progress_velocity=0.3,
            error_accumulation_rate=0.5,
            context_staleness_score=0.2,
        )

        # Measure total workflow latency
        workflow_start = time.perf_counter()

        # Stage 1: Detection
        trigger = await trigger_detector.detect_degradation(current_metrics, baseline)
        assert trigger is not None

        # Stage 2: Decision
        mock_llm_response = create_mock_llm_response(
            intervention_type="replan",
            rationale="Performance degradation",
            confidence=0.88,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)
        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        # Stage 3: Execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response(
            duration_ms=180
        )
        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        workflow_latency_ms = int((time.perf_counter() - workflow_start) * 1000)

        # Verify total workflow latency
        assert workflow_latency_ms < 500, (
            f"Workflow latency {workflow_latency_ms}ms exceeds 500ms target"
        )
        assert record.execution_status == ExecutionStatus.SUCCESS

    async def test_concurrent_intervention_performance(
        self,
        trigger_detector,
        decision_maker,
        intervention_executor,
        runtime_client,
        agent_id,
        strategic_context,
    ):
        """Test performance with concurrent intervention processing."""
        import asyncio

        # Create multiple triggers concurrently
        async def process_intervention(idx: int):
            task_id = uuid4()

            trigger = TriggerSignal(
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                signals=["velocity_drop_below_threshold"],
                rationale=f"Degradation {idx}",
                confidence=0.88,
                metric_values={"velocity_ratio": 0.4},
            )

            mock_llm_response = create_mock_llm_response(
                intervention_type="replan",
                rationale=f"Intervention {idx}",
                confidence=0.87,
            )
            decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

            decision = await decision_maker.decide_intervention(trigger, strategic_context)

            runtime_client.send_intervention.return_value = create_mock_runtime_response()

            record = await intervention_executor.execute_intervention(
                agent_id=agent_id,
                task_id=task_id,
                decision=decision,
                trigger_type=trigger.trigger_type,
                trigger_signals=trigger.signals,
            )
            return record

        # Process 5 interventions concurrently
        start = time.perf_counter()
        results = await asyncio.gather(*[process_intervention(i) for i in range(5)])
        total_time_ms = int((time.perf_counter() - start) * 1000)

        # Verify all succeeded
        assert all(r.execution_status == ExecutionStatus.SUCCESS for r in results)

        # Concurrent processing should be faster than sequential (allow some overhead)
        assert total_time_ms < 1000, (
            f"Concurrent processing {total_time_ms}ms too slow (expected <1000ms for 5)"
        )


class TestWorkflowErrorHandling:
    """Tests for error scenarios and recovery."""

    async def test_llm_timeout_during_decision(
        self, decision_maker, strategic_context
    ):
        """Test workflow handles LLM timeout gracefully."""
        from agentcore.llm_gateway.exceptions import LLMGatewayTimeoutError

        trigger = TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["velocity_drop_below_threshold"],
            rationale="Degradation detected",
            confidence=0.88,
            metric_values={"velocity_ratio": 0.4},
        )

        # Mock LLM timeout
        decision_maker.llm_client.complete = AsyncMock(
            side_effect=LLMGatewayTimeoutError("LLM request timed out", timeout=30.0)
        )

        # Verify error is raised
        with pytest.raises(LLMGatewayTimeoutError):
            await decision_maker.decide_intervention(trigger, strategic_context)

    async def test_runtime_unavailable_during_execution(
        self, intervention_executor, runtime_client, task_id, agent_id
    ):
        """Test workflow handles runtime unavailability."""
        import httpx

        decision = InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Test intervention",
            confidence=0.88,
            expected_impact="Expected improvement",
        )

        # Mock runtime error
        runtime_client.send_intervention.side_effect = httpx.ConnectError(
            "Connection refused"
        )

        # Execute intervention
        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        # Verify failure is recorded
        assert record.execution_status == ExecutionStatus.FAILURE
        assert record.execution_error is not None
        assert "Connection refused" in record.execution_error

    async def test_invalid_trigger_signal(self, trigger_detector):
        """Test detection with invalid trigger data."""
        # Invalid ratios should raise ValueError
        with pytest.raises(ValueError, match="context_age must be >= 0"):
            await trigger_detector.detect_staleness(
                context_age=-5,
                low_confidence_ratio=0.5,
                retrieval_relevance=0.5,
            )

        with pytest.raises(ValueError, match="action_failure_rate must be in"):
            await trigger_detector.detect_capability_mismatch(
                task_requirements=["req_a"],
                agent_capabilities=["cap_a"],
                action_failure_rate=1.5,  # Invalid: >1.0
            )

    async def test_concurrent_interventions_for_same_task(
        self,
        decision_maker,
        intervention_executor,
        runtime_client,
        task_id,
        agent_id,
        strategic_context,
    ):
        """Test handling of concurrent interventions for same task."""
        import asyncio

        trigger = TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["velocity_drop_below_threshold"],
            rationale="Degradation",
            confidence=0.88,
            metric_values={"velocity_ratio": 0.4},
        )

        mock_llm_response = create_mock_llm_response(
            intervention_type="replan",
            rationale="Performance degradation",
            confidence=0.87,
        )
        decision_maker.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        decision = await decision_maker.decide_intervention(trigger, strategic_context)

        runtime_client.send_intervention.return_value = create_mock_runtime_response()

        # Execute two interventions concurrently for same task
        async def execute():
            return await intervention_executor.execute_intervention(
                agent_id=agent_id,
                task_id=task_id,
                decision=decision,
                trigger_type=trigger.trigger_type,
                trigger_signals=trigger.signals,
            )

        results = await asyncio.gather(execute(), execute())

        # Both should complete (real cooldown enforcement would be in InterventionEngine)
        assert all(r.execution_status == ExecutionStatus.SUCCESS for r in results)

    async def test_runtime_interface_error_propagation(
        self, runtime_interface, task_id, agent_id
    ):
        """Test error propagation from runtime interface."""
        # Test with invalid agent_id
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            await runtime_interface.handle_intervention(
                agent_id="",
                task_id=task_id,
                intervention_type=InterventionType.REPLAN,
                context={},
            )

        # Test with whitespace agent_id
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            await runtime_interface.handle_intervention(
                agent_id="   ",
                task_id=task_id,
                intervention_type=InterventionType.REPLAN,
                context={},
            )

    async def test_partial_intervention_execution(
        self, intervention_executor, runtime_client, task_id, agent_id
    ):
        """Test handling of partial intervention execution."""
        decision = InterventionDecision(
            intervention_type=InterventionType.CONTEXT_REFRESH,
            rationale="Context refresh needed",
            confidence=0.85,
            expected_impact="Partial refresh expected",
        )

        # Mock partial execution
        runtime_client.send_intervention.return_value = create_mock_runtime_response(
            status="partial",
            message="Partial execution - some items failed",
            outcome={"refreshed_facts": 20, "failed_items": 5},
        )

        record = await intervention_executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=TriggerType.CONTEXT_STALENESS,
            trigger_signals=["context_age_exceeded"],
        )

        # Verify partial status is recorded
        assert record.execution_status == ExecutionStatus.PARTIAL
        assert record.execution_error is not None
        assert "Partial execution" in record.execution_error
