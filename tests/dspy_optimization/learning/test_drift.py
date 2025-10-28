"""Tests for drift detection"""

import pytest
from datetime import datetime, timedelta

from agentcore.dspy_optimization.learning.drift import (
    DriftDetector,
    DriftConfig,
    DriftStatus,
)
from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
    PerformanceMetrics,
)


class TestDriftDetector:
    """Tests for drift detector"""

    @pytest.fixture
    def detector(self) -> DriftDetector:
        """Create drift detector for testing"""
        config = DriftConfig(
            performance_threshold=0.10,
            critical_threshold=0.20,
            window_size=100,
            min_samples=10,
        )
        return DriftDetector(config)

    @pytest.fixture
    def target(self) -> OptimizationTarget:
        """Create optimization target for testing"""
        return OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent",
            scope=OptimizationScope.INDIVIDUAL,
        )

    @pytest.fixture
    def baseline_metrics(self) -> PerformanceMetrics:
        """Create baseline metrics for testing"""
        return PerformanceMetrics(
            success_rate=0.85,
            avg_cost_per_task=0.10,
            avg_latency_ms=2000,
            quality_score=0.80,
        )

    @pytest.mark.asyncio
    async def test_set_baseline(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test setting baseline metrics"""
        await detector.set_baseline(target, baseline_metrics)

        target_key = detector._get_target_key(target)
        assert target_key in detector._baseline_metrics
        assert detector._baseline_metrics[target_key] == baseline_metrics

    @pytest.mark.asyncio
    async def test_record_metrics(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
    ) -> None:
        """Test recording metrics"""
        metrics = PerformanceMetrics(
            success_rate=0.80,
            avg_cost_per_task=0.12,
            avg_latency_ms=2200,
            quality_score=0.75,
        )

        await detector.record_metrics(target, metrics)

        target_key = detector._get_target_key(target)
        assert target_key in detector._recent_metrics
        assert len(detector._recent_metrics[target_key]) == 1

    @pytest.mark.asyncio
    async def test_no_drift_detection(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test no drift when performance is stable"""
        await detector.set_baseline(target, baseline_metrics)

        # Add metrics with slight variation
        for _ in range(15):
            metrics = PerformanceMetrics(
                success_rate=0.84,  # Slight decrease
                avg_cost_per_task=0.11,
                avg_latency_ms=2100,
                quality_score=0.79,
            )
            await detector.record_metrics(target, metrics)

        result = await detector.check_drift(target)
        assert result is None or result.status == DriftStatus.NO_DRIFT

    @pytest.mark.asyncio
    async def test_performance_degradation(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test performance degradation detection"""
        await detector.set_baseline(target, baseline_metrics)

        # Add degraded metrics
        for _ in range(15):
            metrics = PerformanceMetrics(
                success_rate=0.75,  # 11.8% degradation
                avg_cost_per_task=0.15,
                avg_latency_ms=2500,
                quality_score=0.70,
            )
            await detector.record_metrics(target, metrics)

        result = await detector.check_drift(target)
        assert result is not None
        assert result.status == DriftStatus.PERFORMANCE_DEGRADATION
        assert result.degradation_percentage > 0.0

    @pytest.mark.asyncio
    async def test_critical_drift(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test critical drift detection"""
        await detector.set_baseline(target, baseline_metrics)

        # Add severely degraded metrics
        for _ in range(15):
            metrics = PerformanceMetrics(
                success_rate=0.65,  # 23.5% degradation
                avg_cost_per_task=0.20,
                avg_latency_ms=3000,
                quality_score=0.60,
            )
            await detector.record_metrics(target, metrics)

        result = await detector.check_drift(target)
        assert result is not None
        assert result.status == DriftStatus.CRITICAL_DRIFT
        assert result.degradation_percentage >= 0.20

    @pytest.mark.asyncio
    async def test_insufficient_samples(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test no drift detection with insufficient samples"""
        await detector.set_baseline(target, baseline_metrics)

        # Add only a few metrics
        for _ in range(5):
            metrics = PerformanceMetrics(
                success_rate=0.70,
                avg_cost_per_task=0.15,
                avg_latency_ms=2500,
                quality_score=0.65,
            )
            await detector.record_metrics(target, metrics)

        result = await detector.check_drift(target)
        assert result is None

    @pytest.mark.asyncio
    async def test_reset_baseline(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test baseline reset"""
        await detector.set_baseline(target, baseline_metrics)

        # Add some metrics
        for _ in range(10):
            metrics = PerformanceMetrics(
                success_rate=0.80,
                avg_cost_per_task=0.12,
                avg_latency_ms=2200,
                quality_score=0.75,
            )
            await detector.record_metrics(target, metrics)

        # Reset with new baseline
        new_baseline = PerformanceMetrics(
            success_rate=0.90,
            avg_cost_per_task=0.08,
            avg_latency_ms=1800,
            quality_score=0.85,
        )
        await detector.reset_baseline(target, new_baseline)

        target_key = detector._get_target_key(target)
        assert detector._baseline_metrics[target_key] == new_baseline
        assert len(detector._recent_metrics[target_key]) == 0

    @pytest.mark.asyncio
    async def test_window_size_enforcement(
        self,
        detector: DriftDetector,
        target: OptimizationTarget,
    ) -> None:
        """Test window size limit enforcement"""
        # Add more metrics than window size
        for i in range(150):
            metrics = PerformanceMetrics(
                success_rate=0.80,
                avg_cost_per_task=0.12,
                avg_latency_ms=2200,
                quality_score=0.75,
            )
            await detector.record_metrics(target, metrics)

        target_key = detector._get_target_key(target)
        assert len(detector._recent_metrics[target_key]) <= detector.config.window_size
