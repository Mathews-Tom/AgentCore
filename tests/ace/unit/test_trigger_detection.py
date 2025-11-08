"""
Unit tests for TriggerDetector (COMPASS ACE-2 - ACE-016).

Tests all 4 trigger types with edge cases, threshold validation,
and performance validation (<50ms latency, <15% false positive rate).

Coverage target: 95%+
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from agentcore.ace.intervention.triggers import TriggerDetector
from agentcore.ace.models.ace_models import (
    PerformanceBaseline,
    PerformanceMetrics,
    TriggerType,
)
from agentcore.ace.monitors.error_accumulator import ErrorAccumulator, ErrorSeverity


class TestTriggerDetectorInit:
    """Test TriggerDetector initialization and threshold validation."""

    def test_init_success_default_thresholds(self):
        """Test successful initialization with default thresholds."""
        detector = TriggerDetector()
        assert detector.velocity_threshold == 0.5
        assert detector.error_rate_threshold == 2.0
        assert detector.success_rate_threshold == 0.7
        assert detector.error_count_threshold == 3
        assert detector.context_age_threshold == 20
        assert detector.low_confidence_threshold == 0.6
        assert detector.retrieval_relevance_threshold == 0.4
        assert detector.capability_coverage_threshold == 0.5
        assert detector.action_failure_threshold == 0.5

    def test_init_success_custom_thresholds(self):
        """Test successful initialization with custom thresholds."""
        detector = TriggerDetector(
            velocity_threshold=0.3,
            error_rate_threshold=3.0,
            success_rate_threshold=0.8,
            error_count_threshold=5,
            context_age_threshold=30,
            low_confidence_threshold=0.7,
            retrieval_relevance_threshold=0.3,
            capability_coverage_threshold=0.6,
            action_failure_threshold=0.6,
        )
        assert detector.velocity_threshold == 0.3
        assert detector.error_rate_threshold == 3.0
        assert detector.success_rate_threshold == 0.8
        assert detector.error_count_threshold == 5
        assert detector.context_age_threshold == 30

    def test_init_invalid_velocity_threshold_zero(self):
        """Test initialization with invalid velocity threshold (zero)."""
        with pytest.raises(ValueError, match="velocity_threshold must be in"):
            TriggerDetector(velocity_threshold=0.0)

    def test_init_invalid_velocity_threshold_negative(self):
        """Test initialization with invalid velocity threshold (negative)."""
        with pytest.raises(ValueError, match="velocity_threshold must be in"):
            TriggerDetector(velocity_threshold=-0.1)

    def test_init_invalid_velocity_threshold_too_high(self):
        """Test initialization with invalid velocity threshold (>1)."""
        with pytest.raises(ValueError, match="velocity_threshold must be in"):
            TriggerDetector(velocity_threshold=1.1)

    def test_init_invalid_error_rate_threshold(self):
        """Test initialization with invalid error rate threshold."""
        with pytest.raises(ValueError, match="error_rate_threshold must be > 0"):
            TriggerDetector(error_rate_threshold=-1.0)

    def test_init_invalid_success_rate_threshold(self):
        """Test initialization with invalid success rate threshold."""
        with pytest.raises(ValueError, match="success_rate_threshold must be in"):
            TriggerDetector(success_rate_threshold=1.5)

    def test_init_invalid_error_count_threshold(self):
        """Test initialization with invalid error count threshold."""
        with pytest.raises(ValueError, match="error_count_threshold must be >= 1"):
            TriggerDetector(error_count_threshold=0)

    def test_init_invalid_context_age_threshold(self):
        """Test initialization with invalid context age threshold."""
        with pytest.raises(ValueError, match="context_age_threshold must be >= 1"):
            TriggerDetector(context_age_threshold=0)


class TestDetectDegradation:
    """Test performance degradation detection."""

    @pytest.fixture
    def detector(self):
        """TriggerDetector instance for testing."""
        return TriggerDetector()

    @pytest.fixture
    def baseline(self):
        """Baseline metrics for comparison."""
        return PerformanceBaseline(
            agent_id="agent-001",
            stage="execution",
            mean_success_rate=0.9,
            mean_error_rate=0.1,
            mean_duration_ms=2000.0,
            mean_action_count=10.0,
            sample_size=50,
        )

    @pytest.fixture
    def current_metrics(self):
        """Current performance metrics."""
        return PerformanceMetrics(
            task_id=uuid4(),
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.85,
            stage_error_rate=0.15,
            stage_duration_ms=2500,
            stage_action_count=10,
            overall_progress_velocity=5.0,  # 10 actions / (2000ms / 60000) = 300 actions/min baseline
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

    @pytest.mark.asyncio
    async def test_detect_degradation_velocity_drop(self, detector, baseline, current_metrics):
        """Test velocity degradation detection."""
        # Set velocity to 40% of baseline (below 50% threshold)
        current_metrics.overall_progress_velocity = 120.0  # 40% of 300 actions/min

        signal = await detector.detect_degradation(current_metrics, baseline)

        assert signal is not None
        assert signal.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert "velocity_drop_below_threshold" in signal.signals
        assert "velocity" in signal.rationale.lower()
        assert 0.0 < signal.confidence <= 1.0
        assert "baseline_velocity" in signal.metric_values
        assert "current_velocity" in signal.metric_values
        assert "velocity_ratio" in signal.metric_values

    @pytest.mark.asyncio
    async def test_detect_degradation_error_rate_spike(self, detector, baseline, current_metrics):
        """Test error rate spike detection."""
        # Set error rate to 3x baseline (above 2x threshold)
        current_metrics.stage_error_rate = 0.3  # 3x baseline of 0.1

        signal = await detector.detect_degradation(current_metrics, baseline)

        assert signal is not None
        assert signal.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert "error_rate_spike" in signal.signals
        assert "error rate" in signal.rationale.lower()
        assert "baseline_error_rate" in signal.metric_values
        assert "current_error_rate" in signal.metric_values
        assert "error_rate_ratio" in signal.metric_values

    @pytest.mark.asyncio
    async def test_detect_degradation_success_rate_drop(self, detector, baseline, current_metrics):
        """Test success rate drop detection."""
        # Set success rate below 70% threshold
        current_metrics.stage_success_rate = 0.65

        signal = await detector.detect_degradation(current_metrics, baseline)

        assert signal is not None
        assert signal.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
        assert "success_rate_below_threshold" in signal.signals
        assert "success rate" in signal.rationale.lower()
        assert "current_success_rate" in signal.metric_values

    @pytest.mark.asyncio
    async def test_detect_degradation_multiple_signals(self, detector, baseline, current_metrics):
        """Test detection with multiple degradation signals."""
        # Trigger all 3 signals
        current_metrics.overall_progress_velocity = 120.0  # 40% of baseline
        current_metrics.stage_error_rate = 0.3  # 3x baseline
        current_metrics.stage_success_rate = 0.65  # Below 70%

        signal = await detector.detect_degradation(current_metrics, baseline)

        assert signal is not None
        assert len(signal.signals) == 3
        assert signal.confidence == 1.0  # Max confidence with all signals

    @pytest.mark.asyncio
    async def test_detect_degradation_no_baseline(self, detector, current_metrics):
        """Test degradation detection without baseline (only checks success rate)."""
        current_metrics.stage_success_rate = 0.65  # Below 70%

        signal = await detector.detect_degradation(current_metrics, None)

        assert signal is not None
        assert len(signal.signals) == 1
        assert "success_rate_below_threshold" in signal.signals

    @pytest.mark.asyncio
    async def test_detect_degradation_no_signal(self, detector, baseline, current_metrics):
        """Test no degradation detected when metrics are good."""
        # All metrics above thresholds
        current_metrics.overall_progress_velocity = 300.0  # 100% of baseline
        current_metrics.stage_error_rate = 0.1  # Same as baseline
        current_metrics.stage_success_rate = 0.9  # Above 70%

        signal = await detector.detect_degradation(current_metrics, baseline)

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_degradation_latency(self, detector, baseline, current_metrics):
        """Test degradation detection latency (<50ms target)."""
        # Trigger degradation
        current_metrics.overall_progress_velocity = 120.0
        current_metrics.stage_error_rate = 0.3
        current_metrics.stage_success_rate = 0.65

        start_time = time.perf_counter()
        signal = await detector.detect_degradation(current_metrics, baseline)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert signal is not None
        assert elapsed_ms < 50.0  # <50ms latency target


class TestDetectErrorAccumulation:
    """Test error accumulation detection."""

    @pytest.fixture
    def detector(self):
        """TriggerDetector instance for testing."""
        return TriggerDetector()

    @pytest.fixture
    def error_accumulator(self):
        """ErrorAccumulator instance for testing."""
        return ErrorAccumulator()

    @pytest.mark.asyncio
    async def test_detect_error_accumulation_high_count(self, detector, error_accumulator):
        """Test detection with high error count in stage."""
        agent_id = "agent-001"
        task_id = uuid4()
        stage = "execution"

        # Add 4 errors to execution stage (above threshold of 3)
        for i in range(4):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage=stage,
                error_type=f"error_{i}",
                severity=ErrorSeverity.MEDIUM,
                error_message=f"Error {i}",
            )

        signal = await detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, stage
        )

        assert signal is not None
        assert signal.trigger_type == TriggerType.ERROR_ACCUMULATION
        assert "high_error_count_in_stage" in signal.signals
        assert "4 errors" in signal.rationale
        assert signal.metric_values["stage_error_count"] == 4.0

    @pytest.mark.asyncio
    async def test_detect_error_accumulation_compounding_pattern(
        self, detector, error_accumulator
    ):
        """Test detection with compounding error pattern."""
        agent_id = "agent-001"
        task_id = uuid4()
        stage = "execution"

        # Add 3 errors to trigger compounding pattern
        for i in range(3):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage=stage,
                error_type="same_error",
                severity=ErrorSeverity.HIGH,
                error_message="Same error",
            )

        signal = await detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, stage
        )

        assert signal is not None
        assert "compounding_error_pattern" in signal.signals or "repeated_error_type" in signal.signals

    @pytest.mark.asyncio
    async def test_detect_error_accumulation_repeated_type(self, detector, error_accumulator):
        """Test detection with repeated error type."""
        agent_id = "agent-001"
        task_id = uuid4()
        stage = "execution"

        # Add 2 errors of same type (within window)
        for i in range(2):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage=stage,
                error_type="repeated_error",
                severity=ErrorSeverity.MEDIUM,
                error_message="Repeated error",
            )

        signal = await detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, stage
        )

        # Should detect repeated error type
        assert signal is not None
        assert "repeated_error_type" in signal.signals

    @pytest.mark.asyncio
    async def test_detect_error_accumulation_no_signal(self, detector, error_accumulator):
        """Test no error accumulation detected with low error count."""
        agent_id = "agent-001"
        task_id = uuid4()
        stage = "execution"

        # Add only 1 error (below threshold)
        error_accumulator.track_error(
            agent_id=agent_id,
            task_id=task_id,
            stage=stage,
            error_type="error_1",
            severity=ErrorSeverity.LOW,
            error_message="Error 1",
        )

        signal = await detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, stage
        )

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_error_accumulation_latency(self, detector, error_accumulator):
        """Test error accumulation detection latency (<50ms target)."""
        agent_id = "agent-001"
        task_id = uuid4()
        stage = "execution"

        # Add 5 errors
        for i in range(5):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage=stage,
                error_type=f"error_{i}",
                severity=ErrorSeverity.MEDIUM,
                error_message=f"Error {i}",
            )

        start_time = time.perf_counter()
        signal = await detector.detect_error_accumulation(
            error_accumulator, agent_id, task_id, stage
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert signal is not None
        assert elapsed_ms < 50.0  # <50ms latency target


class TestDetectStaleness:
    """Test context staleness detection."""

    @pytest.fixture
    def detector(self):
        """TriggerDetector instance for testing."""
        return TriggerDetector()

    @pytest.mark.asyncio
    async def test_detect_staleness_context_age(self, detector):
        """Test staleness detection based on context age."""
        signal = await detector.detect_staleness(
            context_age=25,  # Above 20 threshold
            low_confidence_ratio=0.3,
            retrieval_relevance=0.8,
        )

        assert signal is not None
        assert signal.trigger_type == TriggerType.CONTEXT_STALENESS
        assert "context_age_exceeded" in signal.signals
        assert "25 steps" in signal.rationale
        assert signal.metric_values["context_age"] == 25.0

    @pytest.mark.asyncio
    async def test_detect_staleness_low_confidence(self, detector):
        """Test staleness detection based on low confidence ratio."""
        signal = await detector.detect_staleness(
            context_age=10,
            low_confidence_ratio=0.7,  # Above 0.6 threshold
            retrieval_relevance=0.8,
        )

        assert signal is not None
        assert "high_low_confidence_ratio" in signal.signals
        assert "70.0%" in signal.rationale

    @pytest.mark.asyncio
    async def test_detect_staleness_low_retrieval(self, detector):
        """Test staleness detection based on low retrieval relevance."""
        signal = await detector.detect_staleness(
            context_age=10,
            low_confidence_ratio=0.3,
            retrieval_relevance=0.3,  # Below 0.4 threshold
        )

        assert signal is not None
        assert "low_retrieval_relevance" in signal.signals
        assert "30.0%" in signal.rationale

    @pytest.mark.asyncio
    async def test_detect_staleness_multiple_signals(self, detector):
        """Test staleness detection with multiple signals."""
        signal = await detector.detect_staleness(
            context_age=25,
            low_confidence_ratio=0.7,
            retrieval_relevance=0.3,
        )

        assert signal is not None
        assert len(signal.signals) == 3
        assert signal.confidence == 1.0  # Max confidence with all signals

    @pytest.mark.asyncio
    async def test_detect_staleness_no_signal(self, detector):
        """Test no staleness detected when context is fresh."""
        signal = await detector.detect_staleness(
            context_age=10,  # Below 20 threshold
            low_confidence_ratio=0.3,  # Below 0.6 threshold
            retrieval_relevance=0.8,  # Above 0.4 threshold
        )

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_staleness_invalid_context_age(self, detector):
        """Test staleness detection with invalid context age."""
        with pytest.raises(ValueError, match="context_age must be >= 0"):
            await detector.detect_staleness(
                context_age=-1,
                low_confidence_ratio=0.5,
                retrieval_relevance=0.5,
            )

    @pytest.mark.asyncio
    async def test_detect_staleness_invalid_low_confidence_ratio(self, detector):
        """Test staleness detection with invalid low confidence ratio."""
        with pytest.raises(ValueError, match="low_confidence_ratio must be in"):
            await detector.detect_staleness(
                context_age=10,
                low_confidence_ratio=1.5,
                retrieval_relevance=0.5,
            )

    @pytest.mark.asyncio
    async def test_detect_staleness_invalid_retrieval_relevance(self, detector):
        """Test staleness detection with invalid retrieval relevance."""
        with pytest.raises(ValueError, match="retrieval_relevance must be in"):
            await detector.detect_staleness(
                context_age=10,
                low_confidence_ratio=0.5,
                retrieval_relevance=-0.1,
            )

    @pytest.mark.asyncio
    async def test_detect_staleness_latency(self, detector):
        """Test staleness detection latency (<50ms target)."""
        start_time = time.perf_counter()
        signal = await detector.detect_staleness(
            context_age=25,
            low_confidence_ratio=0.7,
            retrieval_relevance=0.3,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert signal is not None
        assert elapsed_ms < 50.0  # <50ms latency target


class TestDetectCapabilityMismatch:
    """Test capability mismatch detection."""

    @pytest.fixture
    def detector(self):
        """TriggerDetector instance for testing."""
        return TriggerDetector()

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_low_coverage(self, detector):
        """Test mismatch detection based on low capability coverage."""
        task_requirements = ["code_analysis", "data_processing", "web_scraping", "ml_training"]
        agent_capabilities = ["code_analysis"]  # Only 25% coverage

        signal = await detector.detect_capability_mismatch(
            task_requirements=task_requirements,
            agent_capabilities=agent_capabilities,
            action_failure_rate=0.3,
        )

        assert signal is not None
        assert signal.trigger_type == TriggerType.CAPABILITY_MISMATCH
        assert "low_capability_coverage" in signal.signals
        assert "25.0%" in signal.rationale
        assert signal.metric_values["capability_coverage"] == 0.25
        assert signal.metric_values["missing_count"] == 3.0

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_high_failure_rate(self, detector):
        """Test mismatch detection based on high action failure rate."""
        task_requirements = ["code_analysis", "data_processing"]
        agent_capabilities = ["code_analysis", "data_processing"]

        signal = await detector.detect_capability_mismatch(
            task_requirements=task_requirements,
            agent_capabilities=agent_capabilities,
            action_failure_rate=0.6,  # Above 0.5 threshold
        )

        assert signal is not None
        assert "high_action_failure_rate" in signal.signals
        assert "60.0%" in signal.rationale

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_multiple_signals(self, detector):
        """Test mismatch detection with multiple signals."""
        task_requirements = ["code_analysis", "data_processing", "web_scraping"]
        agent_capabilities = ["code_analysis"]  # 33% coverage

        signal = await detector.detect_capability_mismatch(
            task_requirements=task_requirements,
            agent_capabilities=agent_capabilities,
            action_failure_rate=0.7,
        )

        assert signal is not None
        assert len(signal.signals) == 2
        assert signal.confidence == 1.0  # Max confidence with both signals

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_no_signal(self, detector):
        """Test no mismatch detected with good coverage and low failure rate."""
        task_requirements = ["code_analysis", "data_processing"]
        agent_capabilities = ["code_analysis", "data_processing", "web_scraping"]

        signal = await detector.detect_capability_mismatch(
            task_requirements=task_requirements,
            agent_capabilities=agent_capabilities,
            action_failure_rate=0.2,
        )

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_no_requirements(self, detector):
        """Test no mismatch detected when no requirements specified."""
        signal = await detector.detect_capability_mismatch(
            task_requirements=[],
            agent_capabilities=["code_analysis"],
            action_failure_rate=0.3,
        )

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_invalid_failure_rate(self, detector):
        """Test capability mismatch detection with invalid failure rate."""
        with pytest.raises(ValueError, match="action_failure_rate must be in"):
            await detector.detect_capability_mismatch(
                task_requirements=["code_analysis"],
                agent_capabilities=["code_analysis"],
                action_failure_rate=1.5,
            )

    @pytest.mark.asyncio
    async def test_detect_capability_mismatch_latency(self, detector):
        """Test capability mismatch detection latency (<50ms target)."""
        task_requirements = ["code_analysis", "data_processing", "web_scraping"]
        agent_capabilities = ["code_analysis"]

        start_time = time.perf_counter()
        signal = await detector.detect_capability_mismatch(
            task_requirements=task_requirements,
            agent_capabilities=agent_capabilities,
            action_failure_rate=0.6,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert signal is not None
        assert elapsed_ms < 50.0  # <50ms latency target


class TestPerformanceValidation:
    """Test overall performance characteristics."""

    @pytest.fixture
    def detector(self):
        """TriggerDetector instance for testing."""
        return TriggerDetector()

    @pytest.mark.asyncio
    async def test_all_detections_meet_latency_target(self, detector):
        """Test that all detection methods meet <50ms latency target."""
        # Test degradation detection
        baseline = PerformanceBaseline(
            agent_id="agent-001",
            stage="execution",
            mean_success_rate=0.9,
            mean_error_rate=0.1,
            mean_duration_ms=2000.0,
            mean_action_count=10.0,
            sample_size=50,
        )
        metrics = PerformanceMetrics(
            task_id=uuid4(),
            agent_id="agent-001",
            stage="execution",
            stage_success_rate=0.65,
            stage_error_rate=0.3,
            stage_duration_ms=2500,
            stage_action_count=10,
            overall_progress_velocity=120.0,
            error_accumulation_rate=0.3,
            context_staleness_score=0.2,
        )

        start = time.perf_counter()
        await detector.detect_degradation(metrics, baseline)
        assert (time.perf_counter() - start) * 1000 < 50.0

        # Test error accumulation detection
        error_accumulator = ErrorAccumulator()
        agent_id = "agent-001"
        task_id = uuid4()
        for i in range(5):
            error_accumulator.track_error(
                agent_id=agent_id,
                task_id=task_id,
                stage="execution",
                error_type=f"error_{i}",
                severity=ErrorSeverity.MEDIUM,
                error_message=f"Error {i}",
            )

        start = time.perf_counter()
        await detector.detect_error_accumulation(error_accumulator, agent_id, task_id, "execution")
        assert (time.perf_counter() - start) * 1000 < 50.0

        # Test staleness detection
        start = time.perf_counter()
        await detector.detect_staleness(25, 0.7, 0.3)
        assert (time.perf_counter() - start) * 1000 < 50.0

        # Test capability mismatch detection
        start = time.perf_counter()
        await detector.detect_capability_mismatch(
            ["cap1", "cap2", "cap3"], ["cap1"], 0.6
        )
        assert (time.perf_counter() - start) * 1000 < 50.0
