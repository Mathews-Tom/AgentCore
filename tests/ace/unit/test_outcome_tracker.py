"""
Unit tests for OutcomeTracker (COMPASS ACE-2 - ACE-023).

Tests outcome recording, delta computation, success determination,
learning data extraction, effectiveness tracking, and threshold updates.

Coverage target: 95%+
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from agentcore.ace.integration.outcome_tracker import OutcomeTracker
from agentcore.ace.models.ace_models import (
    ExecutionStatus,
    InterventionOutcome,
    InterventionRecord,
    InterventionType,
    PerformanceMetrics,
    TriggerType,
)


class TestOutcomeTrackerInit:
    """Test OutcomeTracker initialization."""

    def test_init_success_default(self):
        """Test successful initialization with defaults."""
        tracker = OutcomeTracker()
        assert tracker.mem_interface is None
        assert tracker.outcomes == {}
        assert tracker.logger is not None

    def test_init_success_with_mem_interface(self):
        """Test successful initialization with MEM interface."""
        mock_mem = object()
        tracker = OutcomeTracker(mem_interface=mock_mem)
        assert tracker.mem_interface is mock_mem
        assert tracker.outcomes == {}


class TestComputeDelta:
    """Test delta computation with various scenarios."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    @pytest.fixture
    def base_pre_metrics(self):
        """Base pre-intervention metrics."""
        return PerformanceMetrics(
            metric_id=uuid4(),
            task_id=uuid4(),
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
            recorded_at=datetime.now(UTC),
        )

    @pytest.fixture
    def base_post_metrics(self, base_pre_metrics):
        """Base post-intervention metrics (improved)."""
        return PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

    @pytest.mark.asyncio
    async def test_compute_delta_improvement(self, tracker, base_pre_metrics, base_post_metrics):
        """Test delta computation with improvement."""
        deltas = await tracker.compute_delta(base_pre_metrics, base_post_metrics)

        # Verify delta structure
        assert "delta_velocity" in deltas
        assert "delta_success_rate" in deltas
        assert "delta_error_rate" in deltas
        assert "overall_improvement" in deltas

        # Verify improvement detected
        assert deltas["delta_velocity"] > 0  # Velocity increased from 3.0 to 4.2 (40%)
        assert deltas["delta_success_rate"] > 0  # Success rate increased from 0.70 to 0.85
        assert deltas["delta_error_rate"] > 0  # Error rate decreased from 0.30 to 0.15
        assert deltas["overall_improvement"] > 0

        # Verify ranges
        assert -1.0 <= deltas["delta_velocity"] <= 1.0
        assert -1.0 <= deltas["delta_success_rate"] <= 1.0
        assert -1.0 <= deltas["delta_error_rate"] <= 1.0
        assert -1.0 <= deltas["overall_improvement"] <= 1.0

    @pytest.mark.asyncio
    async def test_compute_delta_degradation(self, tracker, base_pre_metrics):
        """Test delta computation with degradation."""
        # Create post-metrics with worse performance
        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=0.50,  # Decreased from 0.70
            stage_error_rate=0.50,  # Increased from 0.30
            stage_duration_ms=4000,
            stage_action_count=8,
            overall_progress_velocity=2.0,  # Decreased from 3.0
            error_accumulation_rate=0.7,
            context_staleness_score=0.6,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

        deltas = await tracker.compute_delta(base_pre_metrics, post_metrics)

        # Verify degradation detected
        assert deltas["delta_velocity"] < 0
        assert deltas["delta_success_rate"] < 0
        assert deltas["delta_error_rate"] < 0
        assert deltas["overall_improvement"] < 0

    @pytest.mark.asyncio
    async def test_compute_delta_no_change(self, tracker, base_pre_metrics):
        """Test delta computation with identical metrics."""
        # Create post-metrics identical to pre-metrics
        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=base_pre_metrics.stage_success_rate,
            stage_error_rate=base_pre_metrics.stage_error_rate,
            stage_duration_ms=base_pre_metrics.stage_duration_ms,
            stage_action_count=base_pre_metrics.stage_action_count,
            overall_progress_velocity=base_pre_metrics.overall_progress_velocity,
            error_accumulation_rate=base_pre_metrics.error_accumulation_rate,
            context_staleness_score=base_pre_metrics.context_staleness_score,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

        deltas = await tracker.compute_delta(base_pre_metrics, post_metrics)

        # Verify no change detected
        assert deltas["delta_velocity"] == 0.0
        assert deltas["delta_success_rate"] == 0.0
        assert deltas["delta_error_rate"] == 0.0
        assert deltas["overall_improvement"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_delta_zero_pre_velocity(self, tracker, base_pre_metrics):
        """Test delta computation with zero pre-velocity (edge case)."""
        # Set pre-velocity to zero
        base_pre_metrics.overall_progress_velocity = 0.0

        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,  # Non-zero post-velocity
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

        deltas = await tracker.compute_delta(base_pre_metrics, post_metrics)

        # Verify absolute change used instead of percentage
        assert deltas["delta_velocity"] >= 0.0  # Positive change
        assert -1.0 <= deltas["delta_velocity"] <= 1.0  # Clamped

    @pytest.mark.asyncio
    async def test_compute_delta_zero_velocities(self, tracker, base_pre_metrics):
        """Test delta computation with zero pre and post velocities."""
        base_pre_metrics.overall_progress_velocity = 0.0

        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=0.0,  # Also zero
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

        deltas = await tracker.compute_delta(base_pre_metrics, post_metrics)

        # Verify zero delta for velocity
        assert deltas["delta_velocity"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_delta_extreme_positive_values(self, tracker, base_pre_metrics):
        """Test delta computation with extreme positive changes (clamping)."""
        # Create post-metrics with extreme improvements
        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=1.0,  # Maximum
            stage_error_rate=0.0,  # Minimum
            stage_duration_ms=500,
            stage_action_count=50,
            overall_progress_velocity=30.0,  # 10x increase (1000%)
            error_accumulation_rate=0.0,
            context_staleness_score=0.0,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

        deltas = await tracker.compute_delta(base_pre_metrics, post_metrics)

        # Verify clamping to [-1, 1]
        assert -1.0 <= deltas["delta_velocity"] <= 1.0
        assert -1.0 <= deltas["delta_success_rate"] <= 1.0
        assert -1.0 <= deltas["delta_error_rate"] <= 1.0
        assert -1.0 <= deltas["overall_improvement"] <= 1.0

    @pytest.mark.asyncio
    async def test_compute_delta_extreme_negative_values(self, tracker, base_pre_metrics):
        """Test delta computation with extreme negative changes (clamping)."""
        # Create post-metrics with extreme degradation
        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=base_pre_metrics.task_id,
            agent_id=base_pre_metrics.agent_id,
            stage=base_pre_metrics.stage,
            stage_success_rate=0.0,  # Minimum
            stage_error_rate=1.0,  # Maximum
            stage_duration_ms=10000,
            stage_action_count=2,
            overall_progress_velocity=0.1,  # 97% decrease
            error_accumulation_rate=1.0,
            context_staleness_score=1.0,
            recorded_at=datetime.now(UTC) + timedelta(seconds=5),
        )

        deltas = await tracker.compute_delta(base_pre_metrics, post_metrics)

        # Verify clamping to [-1, 1]
        assert -1.0 <= deltas["delta_velocity"] <= 1.0
        assert -1.0 <= deltas["delta_success_rate"] <= 1.0
        assert -1.0 <= deltas["delta_error_rate"] <= 1.0
        assert -1.0 <= deltas["overall_improvement"] <= 1.0


class TestSuccessDetermination:
    """Test success determination logic."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    def test_success_overall_improvement_criterion(self, tracker):
        """Test success determination via overall improvement >= 0.1."""
        deltas = {
            "delta_velocity": 0.15,
            "delta_success_rate": 0.05,
            "delta_error_rate": 0.05,
            "overall_improvement": 0.12,  # >= 0.1
        }

        success = tracker._determine_success(deltas)
        assert success is True

    def test_success_error_reduction_criterion(self, tracker):
        """Test success determination via error rate reduction >= 0.2."""
        deltas = {
            "delta_velocity": 0.05,
            "delta_success_rate": 0.05,
            "delta_error_rate": 0.25,  # >= 0.2
            "overall_improvement": 0.08,
        }

        success = tracker._determine_success(deltas)
        assert success is True

    def test_success_success_rate_criterion(self, tracker):
        """Test success determination via success rate increase >= 0.15."""
        deltas = {
            "delta_velocity": 0.05,
            "delta_success_rate": 0.18,  # >= 0.15
            "delta_error_rate": 0.05,
            "overall_improvement": 0.08,
        }

        success = tracker._determine_success(deltas)
        assert success is True

    def test_success_multiple_criteria(self, tracker):
        """Test success determination with multiple criteria met."""
        deltas = {
            "delta_velocity": 0.30,
            "delta_success_rate": 0.20,
            "delta_error_rate": 0.25,
            "overall_improvement": 0.28,
        }

        success = tracker._determine_success(deltas)
        assert success is True

    def test_failure_no_criteria_met(self, tracker):
        """Test failure determination when no criteria met."""
        deltas = {
            "delta_velocity": 0.05,
            "delta_success_rate": 0.08,
            "delta_error_rate": 0.10,
            "overall_improvement": 0.05,
        }

        success = tracker._determine_success(deltas)
        assert success is False

    def test_failure_negative_deltas(self, tracker):
        """Test failure determination with negative deltas."""
        deltas = {
            "delta_velocity": -0.20,
            "delta_success_rate": -0.10,
            "delta_error_rate": -0.15,
            "overall_improvement": -0.18,
        }

        success = tracker._determine_success(deltas)
        assert success is False

    def test_success_boundary_values(self, tracker):
        """Test success determination at exact boundary values."""
        # Test overall_improvement boundary
        deltas_1 = {
            "delta_velocity": 0.0,
            "delta_success_rate": 0.0,
            "delta_error_rate": 0.0,
            "overall_improvement": 0.1,  # Exactly at boundary
        }
        assert tracker._determine_success(deltas_1) is True

        # Test error_rate boundary
        deltas_2 = {
            "delta_velocity": 0.0,
            "delta_success_rate": 0.0,
            "delta_error_rate": 0.2,  # Exactly at boundary
            "overall_improvement": 0.0,
        }
        assert tracker._determine_success(deltas_2) is True

        # Test success_rate boundary
        deltas_3 = {
            "delta_velocity": 0.0,
            "delta_success_rate": 0.15,  # Exactly at boundary
            "delta_error_rate": 0.0,
            "overall_improvement": 0.0,
        }
        assert tracker._determine_success(deltas_3) is True


class TestLearningDataExtraction:
    """Test learning data extraction."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    @pytest.fixture
    def intervention_record(self):
        """Sample intervention record."""
        return InterventionRecord(
            intervention_id=uuid4(),
            task_id=uuid4(),
            agent_id="agent-001",
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["velocity_drop_50pct", "error_rate_2x"],
            intervention_type=InterventionType.REPLAN,
            intervention_rationale="Performance degraded significantly",
            decision_confidence=0.92,
            execution_status=ExecutionStatus.SUCCESS,
        )

    @pytest.fixture
    def pre_metrics(self):
        """Sample pre-metrics."""
        return PerformanceMetrics(
            metric_id=uuid4(),
            task_id=uuid4(),
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
            recorded_at=datetime.now(UTC),
        )

    @pytest.fixture
    def post_metrics(self, pre_metrics):
        """Sample post-metrics."""
        return PerformanceMetrics(
            metric_id=uuid4(),
            task_id=pre_metrics.task_id,
            agent_id=pre_metrics.agent_id,
            stage=pre_metrics.stage,
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
            recorded_at=pre_metrics.recorded_at + timedelta(seconds=5),
        )

    @pytest.fixture
    def deltas(self):
        """Sample deltas."""
        return {
            "delta_velocity": 0.40,
            "delta_success_rate": 0.15,
            "delta_error_rate": 0.15,
            "overall_improvement": 0.35,
        }

    def test_extract_learning_data_structure(
        self, tracker, intervention_record, pre_metrics, post_metrics, deltas
    ):
        """Test learning data extraction returns correct structure."""
        learning_data = tracker._extract_learning_data(
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            deltas=deltas,
            success=True,
        )

        # Verify required fields
        assert "trigger_type" in learning_data
        assert "intervention_type" in learning_data
        assert "effectiveness" in learning_data
        assert "success" in learning_data
        assert "context_conditions" in learning_data
        assert "time_to_improvement_ms" in learning_data
        assert "trigger_confidence" in learning_data
        assert "trigger_signals" in learning_data

    def test_extract_learning_data_values(
        self, tracker, intervention_record, pre_metrics, post_metrics, deltas
    ):
        """Test learning data extraction contains correct values."""
        learning_data = tracker._extract_learning_data(
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            deltas=deltas,
            success=True,
        )

        # Verify values
        assert learning_data["trigger_type"] == "performance_degradation"
        assert learning_data["intervention_type"] == "replan"
        assert learning_data["effectiveness"] == 0.35
        assert learning_data["success"] is True
        assert learning_data["trigger_confidence"] == 0.92
        assert learning_data["trigger_signals"] == ["velocity_drop_50pct", "error_rate_2x"]

    def test_extract_learning_data_context_conditions(
        self, tracker, intervention_record, pre_metrics, post_metrics, deltas
    ):
        """Test learning data extraction context conditions."""
        learning_data = tracker._extract_learning_data(
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            deltas=deltas,
            success=True,
        )

        context = learning_data["context_conditions"]
        assert context["stage"] == "execution"
        assert context["agent_id"] == "agent-001"
        assert "task_id" in context

    def test_extract_learning_data_time_to_improvement(
        self, tracker, intervention_record, pre_metrics, post_metrics, deltas
    ):
        """Test learning data extraction time to improvement calculation."""
        learning_data = tracker._extract_learning_data(
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            deltas=deltas,
            success=True,
        )

        # Should be ~5000ms (5 seconds between metrics)
        assert learning_data["time_to_improvement_ms"] > 0
        assert learning_data["time_to_improvement_ms"] >= 5000


class TestRecordInterventionOutcome:
    """Test intervention outcome recording."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    @pytest.fixture
    def intervention_id(self):
        """Sample intervention ID."""
        return uuid4()

    @pytest.fixture
    def intervention_record(self):
        """Sample intervention record."""
        return InterventionRecord(
            intervention_id=uuid4(),
            task_id=uuid4(),
            agent_id="agent-001",
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            trigger_signals=["velocity_drop_50pct"],
            intervention_type=InterventionType.REPLAN,
            intervention_rationale="Performance degraded",
            decision_confidence=0.85,
            execution_status=ExecutionStatus.SUCCESS,
        )

    @pytest.fixture
    def pre_metrics(self, intervention_record):
        """Sample pre-metrics."""
        return PerformanceMetrics(
            metric_id=uuid4(),
            task_id=intervention_record.task_id,
            agent_id=intervention_record.agent_id,
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
            recorded_at=datetime.now(UTC),
        )

    @pytest.fixture
    def post_metrics(self, pre_metrics):
        """Sample post-metrics."""
        return PerformanceMetrics(
            metric_id=uuid4(),
            task_id=pre_metrics.task_id,
            agent_id=pre_metrics.agent_id,
            stage=pre_metrics.stage,
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
            recorded_at=pre_metrics.recorded_at + timedelta(seconds=5),
        )

    @pytest.mark.asyncio
    async def test_record_outcome_success(
        self, tracker, intervention_id, intervention_record, pre_metrics, post_metrics
    ):
        """Test successful outcome recording."""
        outcome = await tracker.record_intervention_outcome_with_pre_metrics(
            intervention_id=intervention_id,
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
        )

        # Verify outcome structure
        assert outcome.outcome_id is not None
        assert outcome.intervention_id == intervention_id
        assert outcome.success in [True, False]
        assert outcome.pre_metrics == pre_metrics
        assert outcome.post_metrics == post_metrics
        assert -1.0 <= outcome.delta_velocity <= 1.0
        assert -1.0 <= outcome.delta_success_rate <= 1.0
        assert -1.0 <= outcome.delta_error_rate <= 1.0
        assert -1.0 <= outcome.overall_improvement <= 1.0
        assert outcome.learning_data is not None
        assert outcome.recorded_at is not None

    @pytest.mark.asyncio
    async def test_record_outcome_stored(
        self, tracker, intervention_id, intervention_record, pre_metrics, post_metrics
    ):
        """Test outcome is stored in tracker."""
        outcome = await tracker.record_intervention_outcome_with_pre_metrics(
            intervention_id=intervention_id,
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
        )

        # Verify stored
        stored_outcome = tracker.get_outcome_by_intervention_id(intervention_id)
        assert stored_outcome is not None
        assert stored_outcome.outcome_id == outcome.outcome_id

    @pytest.mark.asyncio
    async def test_record_outcome_mismatched_task_id(
        self, tracker, intervention_id, intervention_record, pre_metrics, post_metrics
    ):
        """Test outcome recording fails with mismatched task IDs."""
        post_metrics.task_id = uuid4()  # Different task ID

        with pytest.raises(ValueError, match="same task"):
            await tracker.record_intervention_outcome_with_pre_metrics(
                intervention_id=intervention_id,
                intervention_record=intervention_record,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
            )

    @pytest.mark.asyncio
    async def test_record_outcome_mismatched_agent_id(
        self, tracker, intervention_id, intervention_record, pre_metrics, post_metrics
    ):
        """Test outcome recording fails with mismatched agent IDs."""
        post_metrics.agent_id = "agent-002"  # Different agent ID

        with pytest.raises(ValueError, match="same agent"):
            await tracker.record_intervention_outcome_with_pre_metrics(
                intervention_id=intervention_id,
                intervention_record=intervention_record,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
            )

    @pytest.mark.asyncio
    async def test_record_outcome_mismatched_stage(
        self, tracker, intervention_id, intervention_record, pre_metrics, post_metrics
    ):
        """Test outcome recording fails with mismatched stages."""
        post_metrics.stage = "planning"  # Different stage

        with pytest.raises(ValueError, match="same stage"):
            await tracker.record_intervention_outcome_with_pre_metrics(
                intervention_id=intervention_id,
                intervention_record=intervention_record,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
            )


class TestGetInterventionEffectiveness:
    """Test intervention effectiveness tracking."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    @pytest.mark.asyncio
    async def test_effectiveness_no_outcomes(self, tracker):
        """Test effectiveness returns 0.0 when no outcomes found."""
        effectiveness = await tracker.get_intervention_effectiveness(
            intervention_type=InterventionType.REPLAN,
            window_days=7,
        )

        assert effectiveness == 0.0

    @pytest.mark.asyncio
    async def test_effectiveness_single_outcome(self, tracker):
        """Test effectiveness with single outcome."""
        # Create and store outcome
        intervention_id = uuid4()
        outcome_data = {
            "outcome_id": uuid4(),
            "intervention_id": intervention_id,
            "success": True,
            "learning_data": {
                "intervention_type": "replan",
                "effectiveness": 0.5,
            },
            "recorded_at": datetime.now(UTC),
        }

        # Manually create outcome (simplified for test)
        # Need to create with all required fields
        task_id = uuid4()
        pre_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
        )
        post_metrics = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        outcome = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id,
            success=True,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            delta_velocity=0.4,
            delta_success_rate=0.15,
            delta_error_rate=0.15,
            overall_improvement=0.35,
            learning_data={
                "intervention_type": "replan",
                "effectiveness": 0.5,
            },
            recorded_at=datetime.now(UTC),
        )

        tracker.outcomes[intervention_id] = outcome

        # Get effectiveness
        effectiveness = await tracker.get_intervention_effectiveness(
            intervention_type=InterventionType.REPLAN,
            window_days=7,
        )

        # Effectiveness should be normalized (0.5 + 1.0) / 2.0 = 0.75
        assert effectiveness == 0.75

    @pytest.mark.asyncio
    async def test_effectiveness_multiple_outcomes(self, tracker):
        """Test effectiveness with multiple outcomes."""
        # Create multiple outcomes
        task_id = uuid4()
        base_pre = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
        )
        base_post = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        for i, eff in enumerate([0.3, 0.5, 0.7]):
            intervention_id = uuid4()
            outcome = InterventionOutcome(
                outcome_id=uuid4(),
                intervention_id=intervention_id,
                success=True,
                pre_metrics=base_pre,
                post_metrics=base_post,
                delta_velocity=0.4,
                delta_success_rate=0.15,
                delta_error_rate=0.15,
                overall_improvement=0.35,
                learning_data={
                    "intervention_type": "replan",
                    "effectiveness": eff,
                },
                recorded_at=datetime.now(UTC),
            )
            tracker.outcomes[intervention_id] = outcome

        # Get effectiveness (should be average: (0.3 + 0.5 + 0.7) / 3 = 0.5)
        # Normalized: (0.5 + 1.0) / 2.0 = 0.75
        effectiveness = await tracker.get_intervention_effectiveness(
            intervention_type=InterventionType.REPLAN,
            window_days=7,
        )

        assert effectiveness == 0.75

    @pytest.mark.asyncio
    async def test_effectiveness_time_window_filtering(self, tracker):
        """Test effectiveness filters by time window."""
        task_id = uuid4()
        base_pre = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
        )
        base_post = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        # Create outcome within window
        intervention_id_1 = uuid4()
        outcome_1 = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id_1,
            success=True,
            pre_metrics=base_pre,
            post_metrics=base_post,
            delta_velocity=0.4,
            delta_success_rate=0.15,
            delta_error_rate=0.15,
            overall_improvement=0.35,
            learning_data={
                "intervention_type": "replan",
                "effectiveness": 0.5,
            },
            recorded_at=datetime.now(UTC),
        )
        tracker.outcomes[intervention_id_1] = outcome_1

        # Create outcome outside window (8 days ago)
        intervention_id_2 = uuid4()
        outcome_2 = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id_2,
            success=True,
            pre_metrics=base_pre,
            post_metrics=base_post,
            delta_velocity=0.4,
            delta_success_rate=0.15,
            delta_error_rate=0.15,
            overall_improvement=0.35,
            learning_data={
                "intervention_type": "replan",
                "effectiveness": 0.9,  # High effectiveness but old
            },
            recorded_at=datetime.now(UTC) - timedelta(days=8),
        )
        tracker.outcomes[intervention_id_2] = outcome_2

        # Get effectiveness with 7-day window (should only include outcome_1)
        effectiveness = await tracker.get_intervention_effectiveness(
            intervention_type=InterventionType.REPLAN,
            window_days=7,
        )

        # Should only include outcome_1 with effectiveness 0.5
        # Normalized: (0.5 + 1.0) / 2.0 = 0.75
        assert effectiveness == 0.75


class TestUpdateInterventionThresholds:
    """Test intervention threshold updates."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    @pytest.mark.asyncio
    async def test_threshold_update_low_effectiveness(self, tracker):
        """Test threshold increase with low effectiveness (<0.7)."""
        # Manually set low effectiveness outcome
        task_id = uuid4()
        base_pre = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
        )
        base_post = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        intervention_id = uuid4()
        outcome = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id,
            success=False,
            pre_metrics=base_pre,
            post_metrics=base_post,
            delta_velocity=0.1,
            delta_success_rate=0.05,
            delta_error_rate=0.05,
            overall_improvement=0.05,
            learning_data={
                "intervention_type": "replan",
                "effectiveness": -0.3,  # Low effectiveness (normalized to 0.35)
            },
            recorded_at=datetime.now(UTC),
        )
        tracker.outcomes[intervention_id] = outcome

        # Update thresholds
        thresholds = await tracker.update_intervention_thresholds(
            intervention_type=InterventionType.REPLAN
        )

        # Thresholds should increase (more cautious)
        assert thresholds["trigger_threshold"] > 0.3  # Base threshold
        assert thresholds["confidence_threshold"] > 0.7  # Base threshold

    @pytest.mark.asyncio
    async def test_threshold_update_high_effectiveness(self, tracker):
        """Test threshold decrease with high effectiveness (>0.9)."""
        # Manually set high effectiveness outcome
        task_id = uuid4()
        base_pre = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
        )
        base_post = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        intervention_id = uuid4()
        outcome = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id,
            success=True,
            pre_metrics=base_pre,
            post_metrics=base_post,
            delta_velocity=0.5,
            delta_success_rate=0.25,
            delta_error_rate=0.25,
            overall_improvement=0.45,
            learning_data={
                "intervention_type": "replan",
                "effectiveness": 0.85,  # High effectiveness (normalized to 0.925)
            },
            recorded_at=datetime.now(UTC),
        )
        tracker.outcomes[intervention_id] = outcome

        # Update thresholds
        thresholds = await tracker.update_intervention_thresholds(
            intervention_type=InterventionType.REPLAN
        )

        # Thresholds should decrease (more proactive)
        assert thresholds["trigger_threshold"] < 0.3  # Base threshold
        assert thresholds["confidence_threshold"] < 0.7  # Base threshold

    @pytest.mark.asyncio
    async def test_threshold_update_moderate_effectiveness(self, tracker):
        """Test threshold maintenance with moderate effectiveness (0.7-0.9)."""
        # Manually set moderate effectiveness outcome
        task_id = uuid4()
        base_pre = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.70,
            stage_error_rate=0.30,
            stage_duration_ms=3000,
            stage_action_count=10,
            overall_progress_velocity=3.0,
            error_accumulation_rate=0.5,
            context_staleness_score=0.4,
        )
        base_post = PerformanceMetrics(
            metric_id=uuid4(),
            task_id=task_id,
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=12,
            overall_progress_velocity=4.2,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        intervention_id = uuid4()
        outcome = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id,
            success=True,
            pre_metrics=base_pre,
            post_metrics=base_post,
            delta_velocity=0.3,
            delta_success_rate=0.15,
            delta_error_rate=0.15,
            overall_improvement=0.25,
            learning_data={
                "intervention_type": "replan",
                "effectiveness": 0.5,  # Moderate effectiveness (normalized to 0.75)
            },
            recorded_at=datetime.now(UTC),
        )
        tracker.outcomes[intervention_id] = outcome

        # Update thresholds
        thresholds = await tracker.update_intervention_thresholds(
            intervention_type=InterventionType.REPLAN
        )

        # Thresholds should stay at base values
        assert thresholds["trigger_threshold"] == 0.3
        assert thresholds["confidence_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_threshold_update_no_outcomes(self, tracker):
        """Test threshold update with no historical outcomes."""
        # Update thresholds with no outcomes
        thresholds = await tracker.update_intervention_thresholds(
            intervention_type=InterventionType.REPLAN
        )

        # Should return base thresholds (effectiveness = 0.0, triggers increase)
        assert thresholds["trigger_threshold"] > 0.3
        assert thresholds["confidence_threshold"] > 0.7


class TestGetOutcomeMethods:
    """Test outcome retrieval methods."""

    @pytest.fixture
    def tracker(self):
        """OutcomeTracker instance."""
        return OutcomeTracker()

    @pytest.fixture
    def sample_outcome(self):
        """Sample outcome."""
        task_id = uuid4()
        return InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=uuid4(),
            success=True,
            pre_metrics=PerformanceMetrics(
                metric_id=uuid4(),
                task_id=task_id,
                agent_id="agent-001",
                stage="execution",
                stage_success_rate=0.70,
                stage_error_rate=0.30,
                stage_duration_ms=3000,
                stage_action_count=10,
                overall_progress_velocity=3.0,
                error_accumulation_rate=0.5,
                context_staleness_score=0.4,
            ),
            post_metrics=PerformanceMetrics(
                metric_id=uuid4(),
                task_id=task_id,
                agent_id="agent-001",
                stage="execution",
                stage_success_rate=0.85,
                stage_error_rate=0.15,
                stage_duration_ms=2500,
                stage_action_count=12,
                overall_progress_velocity=4.2,
                error_accumulation_rate=0.3,
                context_staleness_score=0.2,
            ),
            delta_velocity=0.4,
            delta_success_rate=0.15,
            delta_error_rate=0.15,
            overall_improvement=0.35,
            learning_data={},
            recorded_at=datetime.now(UTC),
        )

    def test_get_outcome_by_intervention_id_success(self, tracker, sample_outcome):
        """Test getting outcome by intervention ID."""
        tracker.outcomes[sample_outcome.intervention_id] = sample_outcome

        outcome = tracker.get_outcome_by_intervention_id(sample_outcome.intervention_id)
        assert outcome is not None
        assert outcome.outcome_id == sample_outcome.outcome_id

    def test_get_outcome_by_intervention_id_not_found(self, tracker):
        """Test getting outcome by non-existent intervention ID."""
        outcome = tracker.get_outcome_by_intervention_id(uuid4())
        assert outcome is None

    def test_get_all_outcomes_empty(self, tracker):
        """Test getting all outcomes when empty."""
        outcomes = tracker.get_all_outcomes()
        assert outcomes == []

    def test_get_all_outcomes_multiple(self, tracker, sample_outcome):
        """Test getting all outcomes with multiple stored."""
        # Store multiple outcomes
        outcome_1 = sample_outcome
        tracker.outcomes[outcome_1.intervention_id] = outcome_1

        task_id = uuid4()
        outcome_2 = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=uuid4(),
            success=False,
            pre_metrics=PerformanceMetrics(
                metric_id=uuid4(),
                task_id=task_id,
                agent_id="agent-002",
                stage="planning",
                stage_success_rate=0.60,
                stage_error_rate=0.40,
                stage_duration_ms=4000,
                stage_action_count=8,
                overall_progress_velocity=2.5,
                error_accumulation_rate=0.6,
                context_staleness_score=0.5,
            ),
            post_metrics=PerformanceMetrics(
                metric_id=uuid4(),
                task_id=task_id,
                agent_id="agent-002",
                stage="planning",
                stage_success_rate=0.55,
                stage_error_rate=0.45,
                stage_duration_ms=4500,
                stage_action_count=7,
                overall_progress_velocity=2.0,
                error_accumulation_rate=0.7,
                context_staleness_score=0.6,
            ),
            delta_velocity=-0.2,
            delta_success_rate=-0.05,
            delta_error_rate=-0.05,
            overall_improvement=-0.15,
            learning_data={},
            recorded_at=datetime.now(UTC),
        )
        tracker.outcomes[outcome_2.intervention_id] = outcome_2

        outcomes = tracker.get_all_outcomes()
        assert len(outcomes) == 2
        assert outcome_1 in outcomes
        assert outcome_2 in outcomes
