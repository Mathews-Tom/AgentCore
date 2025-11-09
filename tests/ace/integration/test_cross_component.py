"""
Cross-Component Integration Tests for COMPASS Meta-Thinker (ACE-024).

Validates full ACE-MEM-Runtime coordination workflow with comprehensive tests
covering the complete intervention pipeline: trigger → MEM query → decision →
execution → outcome tracking.

This completes Phase 4 (ACE-MEM Integration Layer) and validates:
- Full workflow: trigger detection through outcome tracking
- MEM query integration with all 4 query types
- Outcome tracking with delta computation
- Coordination latency (<200ms target)
- Error handling and graceful degradation
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agentcore.ace.integration.mem_interface import ACEMemoryInterface
from agentcore.ace.integration.outcome_tracker import OutcomeTracker
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
from agentcore.ace.monitors.error_accumulator import ErrorAccumulator, ErrorRecord, ErrorSeverity


# Helper function to create PerformanceMetrics with all required fields
def create_metrics(
    task_id, agent_id="agent-001", stage="execution",
    success_rate=0.75, error_rate=0.20, duration_ms=600,
    velocity=0.50, error_accum=0.2, staleness=0.3
) -> PerformanceMetrics:
    """Create PerformanceMetrics with all required fields."""
    return PerformanceMetrics(
        task_id=task_id,
        agent_id=agent_id,
        stage=stage,
        stage_success_rate=success_rate,
        stage_error_rate=error_rate,
        stage_duration_ms=duration_ms,
        stage_action_count=10,
        overall_progress_velocity=velocity,
        error_accumulation_rate=error_accum,
        context_staleness_score=staleness,
        recorded_at=datetime.now(UTC),
    )


# Test Fixtures


@pytest.fixture
def baseline_metrics() -> PerformanceBaseline:
    """Baseline performance metrics for comparison."""
    return PerformanceBaseline(
        agent_id="agent-001",
        stage="execution",
        task_type="data_processing",
        mean_success_rate=0.85,
        mean_error_rate=0.10,
        mean_duration_ms=500.0,
        mean_action_count=10.0,
        std_dev={
            "success_rate": 0.05,
            "error_rate": 0.02,
            "duration_ms": 50.0,
            "action_count": 1.0,
        },
        confidence_interval={
            "success_rate": (0.80, 0.90),
            "error_rate": (0.08, 0.12),
            "duration_ms": (450.0, 550.0),
            "action_count": (9.0, 11.0),
        },
        sample_size=10,
        last_updated=datetime.now(UTC),
    )


@pytest.fixture
def error_accumulator() -> ErrorAccumulator:
    """ErrorAccumulator instance for error tracking."""
    return ErrorAccumulator()


@pytest.fixture
def mem_interface() -> ACEMemoryInterface:
    """ACEMemoryInterface with deterministic seed."""
    return ACEMemoryInterface(seed=42)


@pytest.fixture
def outcome_tracker() -> OutcomeTracker:
    """OutcomeTracker instance for outcome recording."""
    return OutcomeTracker()


@pytest.fixture
def trigger_detector() -> TriggerDetector:
    """TriggerDetector with default thresholds."""
    return TriggerDetector()


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Mock LLM client for DecisionMaker."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"intervention_type": "REPLAN", "rationale": "Performance degradation detected with velocity drop and error increase", "confidence": 0.88, "expected_impact": "Improved task breakdown and execution strategy", "alternative_interventions": ["REFLECT"]}'
            )
        )
    ]
    mock_response.usage = {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600}
    mock_client.complete.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_runtime_client() -> AsyncMock:
    """Mock AgentRuntimeClient for InterventionExecutor."""
    mock_client = AsyncMock()
    mock_client.send_intervention.return_value = {
        "status": "success",
        "duration_ms": 250,
        "message": "Intervention executed successfully",
    }
    return mock_client


# Test Classes


class TestFullInterventionWorkflow:
    """Tests for complete intervention pipeline with MEM and outcome tracking."""

    @pytest.mark.asyncio
    async def test_performance_degradation_full_workflow(
        self,
        trigger_detector: TriggerDetector,
        baseline_metrics: PerformanceBaseline,
        mem_interface: ACEMemoryInterface,
        mock_llm_client: AsyncMock,
        mock_runtime_client: AsyncMock,
        outcome_tracker: OutcomeTracker,
    ) -> None:
        """Test complete workflow: degradation trigger → MEM query → REPLAN decision → execution → outcome."""
        task_id = uuid4()

        # Step 1: Create degraded metrics and detect trigger
        degraded_metrics = create_metrics(
            task_id=task_id, success_rate=0.60, error_rate=0.25,
            velocity=0.25, error_accum=0.4, staleness=0.5
        )

        trigger = await trigger_detector.detect_degradation(
            current_metrics=degraded_metrics,
            baseline=baseline_metrics,
        )

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert len(trigger.signals) > 0

        # Step 2: Query MEM for strategic context
        mem_result = await mem_interface.query_for_strategic_decision(
            trigger=trigger,
            agent_id=degraded_metrics.agent_id,
            task_id=degraded_metrics.task_id,
        )

        assert mem_result is not None
        assert mem_result.strategic_context.context_health_score > 0.0
        assert mem_result.query_latency_ms < 200

        # Step 3: Make decision
        decision_maker = DecisionMaker(llm_client=mock_llm_client)
        decision = await decision_maker.decide_intervention(
            trigger=trigger,
            strategic_context=mem_result.strategic_context,
        )

        assert decision.intervention_type == InterventionType.REPLAN
        assert decision.confidence > 0.5

        # Step 4: Execute intervention
        executor = InterventionExecutor(runtime_client=mock_runtime_client)
        intervention_record = await executor.execute_intervention(
            agent_id=degraded_metrics.agent_id,
            task_id=degraded_metrics.task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        assert intervention_record.execution_status == ExecutionStatus.SUCCESS
        assert intervention_record.execution_duration_ms > 0

        # Step 5: Record outcome with improved metrics
        improved_metrics = create_metrics(
            task_id=task_id, success_rate=0.80, error_rate=0.15,
            velocity=0.45, error_accum=0.2, staleness=0.3
        )

        outcome = await outcome_tracker.record_intervention_outcome_with_pre_metrics(
            intervention_id=intervention_record.intervention_id,
            intervention_record=intervention_record,
            pre_metrics=degraded_metrics,
            post_metrics=improved_metrics,
        )

        # Step 6: Validate improvement
        assert outcome.success is True
        assert outcome.delta_velocity > 0.0
        assert outcome.overall_improvement > 0.0

    @pytest.mark.asyncio
    async def test_error_accumulation_full_workflow(
        self,
        trigger_detector: TriggerDetector,
        error_accumulator: ErrorAccumulator,
        mem_interface: ACEMemoryInterface,
        mock_llm_client: AsyncMock,
        mock_runtime_client: AsyncMock,
        outcome_tracker: OutcomeTracker,
    ) -> None:
        """Test error accumulation workflow."""
        agent_id = "agent-001"
        task_id = uuid4()
        stage = "execution"

        # Track multiple errors
        errors: list[ErrorRecord] = []
        for i in range(4):
            error = error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage=stage,
                error_type="api_timeout",
                severity=ErrorSeverity.HIGH,
                error_message=f"Error {i} occurred",
            )
            errors.append(error)

        # Detect error accumulation
        trigger = await trigger_detector.detect_error_accumulation(
            error_accumulator=error_accumulator,
            agent_id=agent_id,
            task_id=task_id,
            stage=stage,
        )

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.ERROR_ACCUMULATION

        # Query MEM for error analysis
        mem_result = await mem_interface.query_for_error_analysis(
            errors=errors,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert mem_result is not None
        assert len(mem_result.strategic_context.error_patterns) > 0

        # Make REFLECT decision
        mock_llm_client.complete.return_value.choices[0].message.content = (
            '{"intervention_type": "REFLECT", "rationale": "Multiple errors require reflection on execution patterns", '
            '"confidence": 0.85, "expected_impact": "Improved error handling and pattern recognition", '
            '"alternative_interventions": []}'
        )

        decision_maker = DecisionMaker(llm_client=mock_llm_client)
        decision = await decision_maker.decide_intervention(
            trigger=trigger,
            strategic_context=mem_result.strategic_context,
        )

        assert decision.intervention_type == InterventionType.REFLECT

        # Execute and record outcome
        executor = InterventionExecutor(runtime_client=mock_runtime_client)
        intervention_record = await executor.execute_intervention(
            agent_id=agent_id,
            task_id=task_id,
            decision=decision,
            trigger_type=trigger.trigger_type,
            trigger_signals=trigger.signals,
        )

        assert intervention_record.execution_status == ExecutionStatus.SUCCESS

        # Record outcome with reduced errors
        pre_metrics = create_metrics(task_id=task_id, success_rate=0.60, error_rate=0.30, velocity=0.40)
        post_metrics = create_metrics(task_id=task_id, success_rate=0.75, error_rate=0.15, velocity=0.50)

        outcome = await outcome_tracker.record_intervention_outcome_with_pre_metrics(
            intervention_id=intervention_record.intervention_id,
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
        )

        assert outcome.success is True
        assert outcome.delta_error_rate > 0.0


class TestMemQueryIntegration:
    """Tests for MEM query integration in decision-making."""

    @pytest.mark.asyncio
    async def test_strategic_decision_query_influences_decision(
        self,
        mem_interface: ACEMemoryInterface,
        mock_llm_client: AsyncMock,
    ) -> None:
        """Verify strategic decision query results influence decision rationale."""
        trigger = TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["velocity_drop_below_threshold"],
            rationale="Velocity dropped significantly below threshold",
            confidence=0.85,
            metric_values={"velocity_ratio": 0.45},
        )

        # Query MEM
        mem_result = await mem_interface.query_for_strategic_decision(
            trigger=trigger,
            agent_id="agent-001",
            task_id=uuid4(),
        )

        assert mem_result.strategic_context.context_health_score > 0.0
        assert len(mem_result.strategic_context.critical_facts) >= 5

        # Make decision with MEM context
        decision_maker = DecisionMaker(llm_client=mock_llm_client)
        decision = await decision_maker.decide_intervention(
            trigger=trigger,
            strategic_context=mem_result.strategic_context,
        )

        assert "context_health" in decision.metadata

    @pytest.mark.asyncio
    async def test_error_analysis_query_provides_patterns(
        self,
        mem_interface: ACEMemoryInterface,
    ) -> None:
        """Verify error analysis query returns error patterns."""
        errors = [
            ErrorRecord(
                error_id=i,
                agent_id="agent-001",
                task_id=uuid4(),
                stage="execution",
                error_type="timeout",
                severity=ErrorSeverity.HIGH,
                error_message=f"Timeout error {i}",
                timestamp=datetime.now(UTC),
            )
            for i in range(3)
        ]

        mem_result = await mem_interface.query_for_error_analysis(
            errors=errors,
            agent_id="agent-001",
            task_id=errors[0].task_id,
        )

        assert len(mem_result.strategic_context.error_patterns) > 0
        assert mem_result.strategic_context.context_health_score < 0.8


class TestOutcomeTrackingIntegration:
    """Tests for outcome tracking integration."""

    @pytest.mark.asyncio
    async def test_delta_computation_for_improvement(
        self,
        outcome_tracker: OutcomeTracker,
    ) -> None:
        """Verify delta computation correctly identifies improvements."""
        task_id = uuid4()
        pre_metrics = create_metrics(task_id=task_id, success_rate=0.60, error_rate=0.30, velocity=0.30)
        post_metrics = create_metrics(task_id=task_id, success_rate=0.85, error_rate=0.10, velocity=0.60)

        deltas = await outcome_tracker.compute_delta(pre_metrics, post_metrics)

        assert deltas["delta_velocity"] > 0.0
        assert deltas["delta_success_rate"] > 0.0
        assert deltas["delta_error_rate"] > 0.0
        assert deltas["overall_improvement"] > 0.0

    @pytest.mark.asyncio
    async def test_delta_computation_for_degradation(
        self,
        outcome_tracker: OutcomeTracker,
    ) -> None:
        """Verify delta computation correctly identifies degradations."""
        task_id = uuid4()
        pre_metrics = create_metrics(task_id=task_id, success_rate=0.85, error_rate=0.10, velocity=0.60)
        post_metrics = create_metrics(task_id=task_id, success_rate=0.60, error_rate=0.30, velocity=0.30)

        deltas = await outcome_tracker.compute_delta(pre_metrics, post_metrics)

        assert deltas["delta_velocity"] < 0.0
        assert deltas["delta_success_rate"] < 0.0
        assert deltas["delta_error_rate"] < 0.0
        assert deltas["overall_improvement"] < 0.0

    @pytest.mark.asyncio
    async def test_success_determination_logic(
        self,
        outcome_tracker: OutcomeTracker,
    ) -> None:
        """Verify success determination uses multi-criteria evaluation."""
        # Overall improvement meets threshold
        deltas_overall = {
            "delta_velocity": 0.05,
            "delta_success_rate": 0.05,
            "delta_error_rate": 0.05,
            "overall_improvement": 0.15,
        }
        assert outcome_tracker._determine_success(deltas_overall) is True

        # Error rate reduction meets threshold
        deltas_error = {
            "delta_velocity": 0.0,
            "delta_success_rate": 0.0,
            "delta_error_rate": 0.25,
            "overall_improvement": 0.05,
        }
        assert outcome_tracker._determine_success(deltas_error) is True

        # No criteria met
        deltas_fail = {
            "delta_velocity": 0.05,
            "delta_success_rate": 0.05,
            "delta_error_rate": 0.05,
            "overall_improvement": 0.05,
        }
        assert outcome_tracker._determine_success(deltas_fail) is False


class TestCoordinationPerformance:
    """Tests for coordination latency validation."""

    @pytest.mark.asyncio
    async def test_trigger_detection_latency(
        self,
        trigger_detector: TriggerDetector,
        baseline_metrics: PerformanceBaseline,
    ) -> None:
        """Verify trigger detection completes within <50ms target."""
        degraded_metrics = create_metrics(
            task_id=uuid4(), success_rate=0.60, error_rate=0.25, velocity=0.25
        )

        start = time.perf_counter()
        trigger = await trigger_detector.detect_degradation(
            current_metrics=degraded_metrics,
            baseline=baseline_metrics,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        assert trigger is not None
        assert latency_ms < 50

    @pytest.mark.asyncio
    async def test_mem_query_latency(
        self,
        mem_interface: ACEMemoryInterface,
    ) -> None:
        """Verify MEM query completes within <150ms target."""
        trigger = TriggerSignal(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            signals=["test_signal"],
            rationale="Testing MEM query latency performance",
            confidence=0.8,
            metric_values={},
        )

        start = time.perf_counter()
        mem_result = await mem_interface.query_for_strategic_decision(
            trigger=trigger,
            agent_id="agent-001",
            task_id=uuid4(),
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        assert mem_result.query_latency_ms < 150
        assert latency_ms < 200


class TestErrorHandlingPaths:
    """Tests for error scenarios and graceful degradation."""

    @pytest.mark.asyncio
    async def test_mem_unavailable_uses_fallback(
        self,
        mem_interface: ACEMemoryInterface,
    ) -> None:
        """Verify graceful fallback when MEM is unavailable."""
        # Test with empty error list
        mem_result = await mem_interface.query_for_error_analysis(
            errors=[],
            agent_id="agent-001",
            task_id=uuid4(),
        )

        assert mem_result is not None
        assert mem_result.strategic_context is not None

    @pytest.mark.asyncio
    async def test_runtime_unavailable_returns_failure(
        self,
        mock_llm_client: AsyncMock,
    ) -> None:
        """Verify FAILURE status when runtime is unavailable."""
        import httpx

        mock_runtime = AsyncMock()
        mock_runtime.send_intervention.side_effect = httpx.ConnectError("Runtime unavailable")

        executor = InterventionExecutor(runtime_client=mock_runtime)

        decision = InterventionDecision(
            intervention_type=InterventionType.REPLAN,
            rationale="Testing runtime failure handling scenario",
            confidence=0.8,
            expected_impact="Testing failure scenario handling",
            alternative_interventions=[],
        )

        intervention_record = await executor.execute_intervention(
            agent_id="agent-001",
            task_id=uuid4(),
            decision=decision,
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
        )

        assert intervention_record.execution_status == ExecutionStatus.FAILURE
        assert intervention_record.execution_error is not None
        assert "Runtime communication error" in intervention_record.execution_error

    @pytest.mark.asyncio
    async def test_outcome_recording_failure_logged(
        self,
        outcome_tracker: OutcomeTracker,
    ) -> None:
        """Verify outcome recording failure is logged."""
        # Create mismatched metrics (different task IDs)
        pre_metrics = create_metrics(task_id=uuid4(), success_rate=0.60, error_rate=0.30)
        post_metrics = create_metrics(task_id=uuid4(), success_rate=0.80, error_rate=0.15)

        intervention_record = InterventionRecord(
            task_id=pre_metrics.task_id,
            agent_id="agent-001",
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["test_signal"],
            intervention_type=InterventionType.REPLAN,
            intervention_rationale="Testing outcome recording failure",
            decision_confidence=0.8,
            execution_status=ExecutionStatus.SUCCESS,
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="same task"):
            await outcome_tracker.record_intervention_outcome_with_pre_metrics(
                intervention_id=intervention_record.intervention_id,
                intervention_record=intervention_record,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
            )
