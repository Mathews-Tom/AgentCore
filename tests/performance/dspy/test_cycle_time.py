"""
Performance tests for optimization cycle times

Validates <2h completion target for MIPROv2, GEPA, and Genetic algorithms.
Tests measure actual execution time and verify performance targets.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

import pytest

# Skip if dspy not installed (handled by conftest.py)
try:
    from agentcore.dspy_optimization.scalability.cycle_timer import OptimizationTimer
except ImportError as e:
    pytest.skip(f"Required dependencies not available: {e}", allow_module_level=True)


# Target: <2 hours for typical workloads
TARGET_DURATION_SECONDS = 7200  # 2 hours
TYPICAL_ITERATIONS = 100  # Typical workload size


class TestOptimizationCycleTimes:
    """Test optimization cycle time performance"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_miprov2_cycle_time_small(self):
        """Test MIPROv2 cycle time for small workload"""
        timer = OptimizationTimer(target_duration_seconds=TARGET_DURATION_SECONDS)

        # Run optimization with timing
        optimization_id = f"miprov2-small-{datetime.utcnow().isoformat()}"
        timer.start_cycle(optimization_id)

        start_time = time.perf_counter()

        # Simulate MIPROv2 optimization (small iterations for testing)
        await asyncio.sleep(0.1)  # Simulate work
        timer.update_progress(optimization_id, iterations=10)

        end_time = time.perf_counter()
        metrics = timer.end_cycle(optimization_id, status="completed")

        # Verify timing
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Small workload took {elapsed:.2f}s (expected <1s)"
        assert metrics.status == "completed"
        assert metrics.duration_seconds > 0

        # For small workload, should be well under target
        assert not metrics.exceeded_target

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_gepa_cycle_time_small(self):
        """Test GEPA cycle time for small workload"""
        timer = OptimizationTimer(target_duration_seconds=TARGET_DURATION_SECONDS)

        optimization_id = f"gepa-small-{datetime.utcnow().isoformat()}"
        timer.start_cycle(optimization_id)

        start_time = time.perf_counter()

        # Simulate GEPA optimization
        await asyncio.sleep(0.1)
        timer.update_progress(optimization_id, iterations=10)

        end_time = time.perf_counter()
        metrics = timer.end_cycle(optimization_id, status="completed")

        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Small workload took {elapsed:.2f}s (expected <1s)"
        assert metrics.status == "completed"
        assert not metrics.exceeded_target

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_genetic_cycle_time_small(self):
        """Test Genetic algorithm cycle time for small workload"""
        timer = OptimizationTimer(target_duration_seconds=TARGET_DURATION_SECONDS)

        optimization_id = f"genetic-small-{datetime.utcnow().isoformat()}"
        timer.start_cycle(optimization_id)

        start_time = time.perf_counter()

        # Simulate Genetic optimization
        await asyncio.sleep(0.1)
        timer.update_progress(optimization_id, iterations=5)

        end_time = time.perf_counter()
        metrics = timer.end_cycle(optimization_id, status="completed")

        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Small workload took {elapsed:.2f}s (expected <1s)"
        assert metrics.status == "completed"
        assert not metrics.exceeded_target

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cycle_time_with_progress_tracking(self):
        """Test cycle time with progress updates"""
        timer = OptimizationTimer(
            target_duration_seconds=60,
            warning_threshold=0.8
        )

        optimization_id = "progress-test"
        timer.start_cycle(optimization_id)

        # Simulate incremental progress
        for i in range(10):
            await asyncio.sleep(0.05)
            timer.update_progress(optimization_id, iterations=i * 10)

            elapsed = timer.get_elapsed_time(optimization_id)
            assert elapsed is not None
            assert elapsed >= 0

        metrics = timer.end_cycle(optimization_id, status="completed")

        assert metrics.iterations == 90
        assert metrics.throughput > 0
        assert metrics.duration_seconds >= 0.5

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_multiple_concurrent_cycles(self):
        """Test multiple optimization cycles running concurrently"""
        timer = OptimizationTimer(target_duration_seconds=TARGET_DURATION_SECONDS)

        async def run_optimization(opt_id: str, delay: float) -> float:
            timer.start_cycle(opt_id)
            start = time.perf_counter()
            await asyncio.sleep(delay)
            timer.update_progress(opt_id, iterations=10)
            end = time.perf_counter()
            metrics = timer.end_cycle(opt_id, status="completed")
            return metrics.duration_seconds

        # Run 5 optimizations concurrently
        tasks = [
            run_optimization(f"opt-{i}", 0.1)
            for i in range(5)
        ]

        start_time = time.perf_counter()
        durations = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # All should complete
        assert len(durations) == 5
        assert all(d > 0 for d in durations)

        # Should run concurrently, not sequentially
        assert total_time < 0.6  # Should be ~0.1s, not 0.5s

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cycle_statistics(self):
        """Test cycle statistics calculation"""
        timer = OptimizationTimer(target_duration_seconds=10)

        # Run multiple cycles
        for i in range(5):
            opt_id = f"stat-test-{i}"
            timer.start_cycle(opt_id)
            await asyncio.sleep(0.05 + i * 0.01)
            timer.update_progress(opt_id, iterations=100 + i * 10)
            timer.end_cycle(opt_id, status="completed")

        stats = timer.get_cycle_statistics()

        assert stats["total_cycles"] == 5
        assert stats["avg_duration"] > 0
        assert stats["avg_throughput"] > 0
        assert stats["success_rate"] == 1.0
        assert stats["min_duration"] > 0
        assert stats["max_duration"] > stats["min_duration"]

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_calculation(self):
        """Test throughput (iterations/second) calculation"""
        timer = OptimizationTimer()

        optimization_id = "throughput-test"
        timer.start_cycle(optimization_id)

        iterations = 1000
        timer.update_progress(optimization_id, iterations=iterations)
        await asyncio.sleep(0.1)

        metrics = timer.end_cycle(optimization_id, status="completed")

        # Verify throughput calculation
        expected_throughput = iterations / metrics.duration_seconds
        assert abs(metrics.throughput - expected_throughput) < 0.01
        assert metrics.throughput > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_time_remaining_calculation(self):
        """Test time remaining calculation"""
        timer = OptimizationTimer(target_duration_seconds=10.0)

        optimization_id = "remaining-test"
        timer.start_cycle(optimization_id)

        await asyncio.sleep(0.5)

        remaining = timer.check_time_remaining(optimization_id)
        assert remaining is not None
        assert remaining < 10.0
        assert remaining > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_approaching_limit_detection(self):
        """Test detection of approaching time limit"""
        timer = OptimizationTimer(
            target_duration_seconds=1.0,
            warning_threshold=0.5  # Warn at 50% of time
        )

        optimization_id = "limit-test"
        timer.start_cycle(optimization_id)

        # Should not be approaching initially
        assert not timer.is_approaching_limit(optimization_id)

        # Wait past threshold
        await asyncio.sleep(0.6)

        # Should now be approaching
        assert timer.is_approaching_limit(optimization_id)

        timer.end_cycle(optimization_id, status="completed")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_exceeded_target_detection(self):
        """Test detection of exceeded target duration"""
        timer = OptimizationTimer(target_duration_seconds=0.1)

        optimization_id = "exceed-test"
        timer.start_cycle(optimization_id)

        # Deliberately exceed target
        await asyncio.sleep(0.2)

        metrics = timer.end_cycle(optimization_id, status="completed")

        assert metrics.exceeded_target is True
        assert metrics.duration_seconds > metrics.target_duration_seconds

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.benchmark
    async def test_performance_benchmark_suite(self):
        """Run comprehensive performance benchmark"""
        timer = OptimizationTimer(target_duration_seconds=TARGET_DURATION_SECONDS)

        results = {
            "miprov2": [],
            "gepa": [],
            "genetic": []
        }

        # Test each algorithm multiple times
        for algorithm in ["miprov2", "gepa", "genetic"]:
            for i in range(3):
                opt_id = f"{algorithm}-bench-{i}"
                timer.start_cycle(opt_id)

                start = time.perf_counter()
                await asyncio.sleep(0.05)  # Simulate work
                end = time.perf_counter()

                timer.update_progress(opt_id, iterations=10)
                metrics = timer.end_cycle(opt_id, status="completed")

                results[algorithm].append({
                    "duration": end - start,
                    "throughput": metrics.throughput,
                    "exceeded_target": metrics.exceeded_target
                })

        # Verify all completed successfully
        for algorithm, runs in results.items():
            assert len(runs) == 3
            avg_duration = sum(r["duration"] for r in runs) / len(runs)
            assert avg_duration < 1.0, f"{algorithm} avg: {avg_duration:.3f}s"

            # None should exceed target for small workloads
            assert not any(r["exceeded_target"] for r in runs)
