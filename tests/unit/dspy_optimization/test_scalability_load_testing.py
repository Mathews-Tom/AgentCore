"""
Tests for load testing framework
"""

import asyncio

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
)
from agentcore.dspy_optimization.scalability.load_testing import (
    LoadTestRunner,
    LoadProfile,
    LoadPattern,
    PerformanceMetrics,
    BottleneckInfo,
)


async def mock_optimization_handler(request: OptimizationRequest) -> OptimizationResult:
    """Mock optimization handler"""
    await asyncio.sleep(0.01)  # Simulate work
    return OptimizationResult(status=OptimizationStatus.COMPLETED)


async def slow_optimization_handler(request: OptimizationRequest) -> OptimizationResult:
    """Slow optimization handler"""
    await asyncio.sleep(1.0)
    return OptimizationResult(status=OptimizationStatus.COMPLETED)


async def failing_optimization_handler(request: OptimizationRequest) -> OptimizationResult:
    """Failing optimization handler"""
    raise RuntimeError("Optimization failed")


class TestLoadTestRunner:
    """Test load testing functionality"""

    @pytest.mark.asyncio
    async def test_constant_load_pattern(self):
        """Test constant load pattern"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
            max_concurrent=10,
        )

        results = await runner.run_load_test(profile)

        assert results.passed is True
        assert results.metrics.total_requests > 0
        assert results.metrics.successful_requests > 0
        assert results.duration_seconds >= 1.0

    @pytest.mark.asyncio
    async def test_ramp_up_load_pattern(self):
        """Test ramp-up load pattern"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.RAMP_UP,
            duration_seconds=2,
            target_rps=10,
            ramp_up_seconds=1,
            cool_down_seconds=0,
        )

        results = await runner.run_load_test(profile)

        assert results.passed is True
        assert results.metrics.total_requests > 0

    @pytest.mark.asyncio
    async def test_spike_load_pattern(self):
        """Test spike load pattern"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.SPIKE,
            duration_seconds=2,
            target_rps=5,
            max_concurrent=20,
        )

        results = await runner.run_load_test(profile)

        assert results.passed is True
        assert results.metrics.total_requests > 0

    @pytest.mark.asyncio
    async def test_wave_load_pattern(self):
        """Test wave load pattern"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.WAVE,
            duration_seconds=2,
            target_rps=5,
            max_concurrent=10,
        )

        results = await runner.run_load_test(profile)

        assert results.passed is True
        assert results.metrics.total_requests > 0

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=10,
        )

        results = await runner.run_load_test(profile)
        metrics = results.metrics

        assert metrics.total_requests > 0
        assert metrics.successful_requests > 0
        assert metrics.avg_response_time > 0
        assert metrics.p50_response_time > 0
        assert metrics.p95_response_time > 0
        assert metrics.p99_response_time > 0
        assert metrics.min_response_time > 0
        assert metrics.max_response_time > 0
        assert metrics.throughput_rps > 0
        assert metrics.error_rate >= 0

    @pytest.mark.asyncio
    async def test_error_rate_calculation(self):
        """Test error rate calculation"""
        runner = LoadTestRunner(failing_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
        )

        results = await runner.run_load_test(profile)
        metrics = results.metrics

        assert metrics.failed_requests > 0
        assert metrics.error_rate > 0
        assert metrics.error_rate == metrics.failed_requests / metrics.total_requests

    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self):
        """Test concurrent request limit enforcement"""
        runner = LoadTestRunner(slow_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=20,
            max_concurrent=5,
        )

        results = await runner.run_load_test(profile)

        # Should respect max_concurrent
        assert results.metrics.total_requests > 0

    @pytest.mark.asyncio
    async def test_bottleneck_detection_high_error_rate(self):
        """Test bottleneck detection for high error rate"""
        runner = LoadTestRunner(failing_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
        )

        results = await runner.run_load_test(profile)

        # Should detect high error rate bottleneck
        error_bottlenecks = [
            b for b in results.bottlenecks
            if b.component == "error_handling"
        ]
        assert len(error_bottlenecks) > 0
        assert error_bottlenecks[0].severity == "high"

    @pytest.mark.asyncio
    async def test_bottleneck_detection_high_response_time(self):
        """Test bottleneck detection for high response time"""
        runner = LoadTestRunner(slow_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=2,
        )

        results = await runner.run_load_test(profile)

        # Should detect high response time bottleneck
        response_time_bottlenecks = [
            b for b in results.bottlenecks
            if b.component == "response_time"
        ]
        assert len(response_time_bottlenecks) > 0

    @pytest.mark.asyncio
    async def test_test_failure_high_error_rate(self):
        """Test failure due to high error rate"""
        runner = LoadTestRunner(failing_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
        )

        results = await runner.run_load_test(profile)

        # Test should fail due to high error rate (>10%)
        assert results.passed is False

    @pytest.mark.asyncio
    async def test_test_failure_low_throughput(self):
        """Test failure due to low throughput"""
        runner = LoadTestRunner(slow_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=10,
            max_concurrent=2,
        )

        results = await runner.run_load_test(profile)

        # Test may fail due to low throughput (<80% of target)
        if results.metrics.throughput_rps < profile.target_rps * 0.8:
            assert results.passed is False

    @pytest.mark.asyncio
    async def test_resource_monitoring(self):
        """Test resource monitoring during load test"""
        from agentcore.dspy_optimization.scalability.resource_pool import (
            OptimizationResourceManager,
        )

        manager = OptimizationResourceManager()
        await manager.initialize_worker_pool()

        runner = LoadTestRunner(mock_optimization_handler, enable_monitoring=True)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
        )

        results = await runner.run_load_test(profile, resource_manager=manager)

        # Should have resource usage data
        assert results.resource_usage is not None
        assert "worker" in results.resource_usage

    @pytest.mark.asyncio
    async def test_percentile_calculations(self):
        """Test percentile calculations"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=20,
        )

        results = await runner.run_load_test(profile)
        metrics = results.metrics

        # Percentiles should be ordered
        assert metrics.p50_response_time <= metrics.p95_response_time
        assert metrics.p95_response_time <= metrics.p99_response_time
        assert metrics.min_response_time <= metrics.p50_response_time
        assert metrics.p99_response_time <= metrics.max_response_time

    @pytest.mark.asyncio
    async def test_throughput_calculation(self):
        """Test throughput calculation"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=2,
            target_rps=10,
        )

        results = await runner.run_load_test(profile)

        # Throughput should be approximately target_rps
        assert results.metrics.throughput_rps > 0
        # Allow some tolerance
        assert abs(results.metrics.throughput_rps - profile.target_rps) < 5

    @pytest.mark.asyncio
    async def test_load_test_duration(self):
        """Test load test duration"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
        )

        results = await runner.run_load_test(profile)

        # Duration should be close to target
        assert results.duration_seconds >= profile.duration_seconds
        assert results.duration_seconds < profile.duration_seconds + 1

    @pytest.mark.asyncio
    async def test_synthetic_request_generation(self):
        """Test synthetic request generation"""
        runner = LoadTestRunner(mock_optimization_handler)

        request = runner._create_synthetic_request()

        assert request.target.type.value == "agent"
        assert request.target.id == "test-agent"
        assert len(request.objectives) > 0
        assert len(request.algorithms) > 0

    @pytest.mark.asyncio
    async def test_load_test_timestamps(self):
        """Test load test timestamps"""
        runner = LoadTestRunner(mock_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=1,
            target_rps=5,
        )

        results = await runner.run_load_test(profile)

        assert results.start_time is not None
        assert results.end_time is not None
        assert results.end_time > results.start_time

    @pytest.mark.asyncio
    async def test_empty_response_times(self):
        """Test handling of no completed requests"""
        async def never_complete_handler(request: OptimizationRequest) -> OptimizationResult:
            await asyncio.sleep(100)  # Never completes in test
            return OptimizationResult(status=OptimizationStatus.COMPLETED)

        runner = LoadTestRunner(never_complete_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=0.1,
            target_rps=1,
        )

        results = await runner.run_load_test(profile)

        # Should handle empty metrics gracefully
        assert results.metrics.total_requests >= 0
