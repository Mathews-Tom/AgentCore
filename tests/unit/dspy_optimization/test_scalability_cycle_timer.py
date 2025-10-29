"""
Tests for optimization cycle timer
"""

import asyncio
import time
from datetime import datetime, timedelta

import pytest

from agentcore.dspy_optimization.scalability.cycle_timer import (
    OptimizationTimer,
    CycleMetrics,
    PerformanceAlert,
    AlertSeverity,
)


class TestOptimizationTimer:
    """Test optimization timer functionality"""

    def test_start_cycle(self):
        """Test starting cycle timing"""
        timer = OptimizationTimer()
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)

        assert optimization_id in timer.get_active_cycles()
        assert timer.get_elapsed_time(optimization_id) is not None
        assert timer.get_elapsed_time(optimization_id) >= 0

    def test_start_duplicate_cycle(self):
        """Test starting duplicate cycle logs warning"""
        timer = OptimizationTimer()
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        timer.start_cycle(optimization_id)  # Should log warning

        assert optimization_id in timer.get_active_cycles()

    def test_update_progress(self):
        """Test updating cycle progress"""
        timer = OptimizationTimer()
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        timer.update_progress(optimization_id, iterations=50)

        # Should update internal state
        assert optimization_id in timer.get_active_cycles()

    def test_update_progress_nonexistent(self):
        """Test updating progress for non-existent cycle"""
        timer = OptimizationTimer()
        timer.update_progress("nonexistent", iterations=10)
        # Should log warning but not crash

    def test_end_cycle(self):
        """Test ending cycle and computing metrics"""
        timer = OptimizationTimer(target_duration_seconds=10)
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        timer.update_progress(optimization_id, iterations=100)
        time.sleep(0.1)  # Simulate some work
        metrics = timer.end_cycle(optimization_id, status="completed")

        assert metrics.optimization_id == optimization_id
        assert metrics.status == "completed"
        assert metrics.duration_seconds > 0
        assert metrics.iterations == 100
        assert metrics.throughput > 0
        assert optimization_id not in timer.get_active_cycles()

    def test_end_cycle_not_started(self):
        """Test ending cycle that wasn't started"""
        timer = OptimizationTimer()

        with pytest.raises(ValueError, match="not started"):
            timer.end_cycle("nonexistent")

    def test_exceeded_target_duration(self):
        """Test detection of exceeded target duration"""
        timer = OptimizationTimer(target_duration_seconds=0.05)
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.1)  # Exceed target
        metrics = timer.end_cycle(optimization_id)

        assert metrics.exceeded_target is True
        assert metrics.duration_seconds > metrics.target_duration_seconds

    def test_under_target_duration(self):
        """Test completion under target duration"""
        timer = OptimizationTimer(target_duration_seconds=10)
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.05)
        metrics = timer.end_cycle(optimization_id)

        assert metrics.exceeded_target is False
        assert metrics.duration_seconds < metrics.target_duration_seconds

    def test_get_elapsed_time(self):
        """Test getting elapsed time for active cycle"""
        timer = OptimizationTimer()
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.05)
        elapsed = timer.get_elapsed_time(optimization_id)

        assert elapsed is not None
        assert elapsed >= 0.05

    def test_get_elapsed_time_nonexistent(self):
        """Test getting elapsed time for non-existent cycle"""
        timer = OptimizationTimer()
        elapsed = timer.get_elapsed_time("nonexistent")

        assert elapsed is None

    def test_check_time_remaining(self):
        """Test checking time remaining"""
        timer = OptimizationTimer(target_duration_seconds=10)
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.1)
        remaining = timer.check_time_remaining(optimization_id)

        assert remaining is not None
        assert remaining < 10
        assert remaining > 0

    def test_is_approaching_limit(self):
        """Test detection of approaching time limit"""
        timer = OptimizationTimer(
            target_duration_seconds=1.0, warning_threshold=0.5
        )
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)

        # Should not be approaching initially
        assert timer.is_approaching_limit(optimization_id) is False

        # Wait to approach threshold
        time.sleep(0.6)
        assert timer.is_approaching_limit(optimization_id) is True

    def test_warning_alert_generation(self):
        """Test generation of warning alerts"""
        timer = OptimizationTimer(
            target_duration_seconds=1.0,
            warning_threshold=0.5,
            enable_alerts=True,
        )
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.6)  # Trigger warning
        timer.update_progress(optimization_id, iterations=10)

        alerts = timer.get_recent_alerts(severity=AlertSeverity.WARNING)
        assert len(alerts) > 0
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].optimization_id == optimization_id

    def test_critical_alert_generation(self):
        """Test generation of critical alerts"""
        timer = OptimizationTimer(
            target_duration_seconds=0.5, enable_alerts=True
        )
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.6)  # Exceed target
        timer.update_progress(optimization_id, iterations=10)

        alerts = timer.get_recent_alerts(severity=AlertSeverity.CRITICAL)
        assert len(alerts) > 0
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_alerts_disabled(self):
        """Test that alerts are not generated when disabled"""
        timer = OptimizationTimer(
            target_duration_seconds=0.5, enable_alerts=False
        )
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.6)
        timer.update_progress(optimization_id, iterations=10)

        alerts = timer.get_recent_alerts()
        assert len(alerts) == 0

    def test_get_cycle_statistics(self):
        """Test getting cycle statistics"""
        timer = OptimizationTimer(target_duration_seconds=10)

        # Run multiple cycles
        for i in range(5):
            optimization_id = f"test-opt-{i}"
            timer.start_cycle(optimization_id)
            timer.update_progress(optimization_id, iterations=50 + i * 10)
            time.sleep(0.05)
            timer.end_cycle(optimization_id, status="completed")

        stats = timer.get_cycle_statistics()

        assert stats["total_cycles"] == 5
        assert stats["avg_duration"] > 0
        assert stats["avg_throughput"] > 0
        assert stats["success_rate"] == 1.0
        assert "min_duration" in stats
        assert "max_duration" in stats

    def test_get_cycle_statistics_empty(self):
        """Test getting statistics with no completed cycles"""
        timer = OptimizationTimer()
        stats = timer.get_cycle_statistics()

        assert stats["total_cycles"] == 0
        assert stats["avg_duration"] == 0.0

    def test_get_active_cycles(self):
        """Test getting list of active cycles"""
        timer = OptimizationTimer()

        timer.start_cycle("opt-1")
        timer.start_cycle("opt-2")
        timer.start_cycle("opt-3")

        active = timer.get_active_cycles()
        assert len(active) == 3
        assert "opt-1" in active
        assert "opt-2" in active
        assert "opt-3" in active

    def test_throughput_calculation(self):
        """Test throughput calculation"""
        timer = OptimizationTimer()
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        timer.update_progress(optimization_id, iterations=1000)
        time.sleep(0.1)
        metrics = timer.end_cycle(optimization_id)

        # Should calculate iterations per second
        assert metrics.throughput > 0
        assert metrics.throughput == metrics.iterations / metrics.duration_seconds

    def test_clear_alerts(self):
        """Test clearing alerts"""
        timer = OptimizationTimer(
            target_duration_seconds=0.5, enable_alerts=True
        )

        # Generate some alerts
        timer.start_cycle("opt-1")
        time.sleep(0.6)
        timer.update_progress("opt-1", iterations=10)

        assert len(timer.get_recent_alerts()) > 0

        timer.clear_alerts()
        assert len(timer.get_recent_alerts()) == 0

    def test_reset(self):
        """Test resetting timer"""
        timer = OptimizationTimer()

        # Create some cycles and alerts
        timer.start_cycle("opt-1")
        timer.update_progress("opt-1", iterations=10)

        timer.reset()

        assert len(timer.get_active_cycles()) == 0
        assert timer.get_cycle_statistics()["total_cycles"] == 0
        assert len(timer.get_recent_alerts()) == 0

    def test_duplicate_alert_suppression(self):
        """Test that duplicate alerts are suppressed"""
        timer = OptimizationTimer(
            target_duration_seconds=0.5, enable_alerts=True
        )
        optimization_id = "test-opt-1"

        timer.start_cycle(optimization_id)
        time.sleep(0.6)

        # Trigger multiple updates
        timer.update_progress(optimization_id, iterations=10)
        timer.update_progress(optimization_id, iterations=20)
        timer.update_progress(optimization_id, iterations=30)

        # Should not create duplicate critical alerts
        alerts = timer.get_recent_alerts(severity=AlertSeverity.CRITICAL)
        assert len(alerts) == 1  # Only one alert despite multiple updates
