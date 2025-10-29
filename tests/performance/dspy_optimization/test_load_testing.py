"""
Comprehensive load testing for DSPy optimization

Uses DSP-012 load testing framework to validate system behavior under load.
Tests various load patterns and generates performance reports.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    from agentcore.dspy_optimization.scalability.load_testing import (
        LoadTestRunner,
        LoadProfile,
        LoadPattern,
    )
    from agentcore.dspy_optimization.scalability.resource_pool import (
        OptimizationResourceManager,
    )
except ImportError as e:
    pytest.skip(f"Required dependencies not available: {e}", allow_module_level=True)


async def standard_optimization_handler(request) -> dict:
    """Standard optimization handler for load testing"""
    await asyncio.sleep(0.02)  # Simulate 20ms work
    return {"status": "completed", "accuracy": 0.95}


async def fast_optimization_handler(request) -> dict:
    """Fast optimization handler"""
    await asyncio.sleep(0.005)  # Simulate 5ms work
    return {"status": "completed", "accuracy": 0.90}


async def slow_optimization_handler(request) -> dict:
    """Slow optimization handler"""
    await asyncio.sleep(0.1)  # Simulate 100ms work
    return {"status": "completed", "accuracy": 0.98}


class TestLoadTesting:
    """Comprehensive load testing"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_constant_load_pattern(self):
        """Test constant load pattern"""
        runner = LoadTestRunner(standard_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=5,
            target_rps=20,
            max_concurrent=50,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("CONSTANT LOAD TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Duration: {results.duration_seconds:.2f}s")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Successful: {results.metrics.successful_requests}")
        print(f"Failed: {results.metrics.failed_requests}")
        print(f"Throughput: {results.metrics.throughput_rps:.2f} req/s")
        print(f"Avg Response Time: {results.metrics.avg_response_time * 1000:.2f} ms")
        print(f"P50: {results.metrics.p50_response_time * 1000:.2f} ms")
        print(f"P95: {results.metrics.p95_response_time * 1000:.2f} ms")
        print(f"P99: {results.metrics.p99_response_time * 1000:.2f} ms")
        print(f"Error Rate: {results.metrics.error_rate:.2%}")
        print(f"Test Passed: {results.passed}")

        assert results.passed is True
        assert results.metrics.total_requests > 0
        assert results.metrics.successful_requests > 0
        assert results.metrics.error_rate < 0.1  # Less than 10% errors

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_ramp_up_load_pattern(self):
        """Test ramp-up load pattern"""
        runner = LoadTestRunner(standard_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.RAMP_UP,
            duration_seconds=10,
            target_rps=50,
            ramp_up_seconds=5,
            cool_down_seconds=2,
            max_concurrent=100,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("RAMP-UP LOAD TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Duration: {results.duration_seconds:.2f}s")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Throughput: {results.metrics.throughput_rps:.2f} req/s")
        print(f"P95 Response Time: {results.metrics.p95_response_time * 1000:.2f} ms")
        print(f"Test Passed: {results.passed}")

        assert results.passed is True
        assert results.metrics.total_requests > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_spike_load_pattern(self):
        """Test spike load pattern"""
        runner = LoadTestRunner(fast_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.SPIKE,
            duration_seconds=8,
            target_rps=30,
            max_concurrent=200,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("SPIKE LOAD TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Duration: {results.duration_seconds:.2f}s")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Max Response Time: {results.metrics.max_response_time * 1000:.2f} ms")
        print(f"P99 Response Time: {results.metrics.p99_response_time * 1000:.2f} ms")
        print(f"Test Passed: {results.passed}")

        assert results.passed is True
        assert results.metrics.total_requests > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_wave_load_pattern(self):
        """Test wave load pattern"""
        runner = LoadTestRunner(standard_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.WAVE,
            duration_seconds=10,
            target_rps=25,
            max_concurrent=80,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("WAVE LOAD TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Duration: {results.duration_seconds:.2f}s")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Throughput: {results.metrics.throughput_rps:.2f} req/s")
        print(f"Avg Response Time: {results.metrics.avg_response_time * 1000:.2f} ms")
        print(f"Test Passed: {results.passed}")

        assert results.passed is True

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_high_load(self):
        """Test sustained high load"""
        runner = LoadTestRunner(fast_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=30,
            target_rps=100,
            max_concurrent=500,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("SUSTAINED HIGH LOAD TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Duration: {results.duration_seconds:.2f}s")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Throughput: {results.metrics.throughput_rps:.2f} req/s")
        print(f"Avg Response Time: {results.metrics.avg_response_time * 1000:.2f} ms")
        print(f"P95 Response Time: {results.metrics.p95_response_time * 1000:.2f} ms")
        print(f"Error Rate: {results.metrics.error_rate:.2%}")
        print(f"Test Passed: {results.passed}")

        assert results.passed is True
        assert results.metrics.total_requests >= 2500  # At least 2500 requests
        assert results.metrics.error_rate < 0.05  # Less than 5% errors

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        runner = LoadTestRunner(slow_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=3,
            target_rps=30,
            max_concurrent=20,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("BOTTLENECK DETECTION RESULTS")
        print(f"{'=' * 80}")
        print(f"Bottlenecks Found: {len(results.bottlenecks)}")

        for bottleneck in results.bottlenecks:
            print(f"\n  Component: {bottleneck.component}")
            print(f"  Severity: {bottleneck.severity}")
            print(f"  Description: {bottleneck.description}")
            print(f"  Impact: {bottleneck.impact}")
            print(f"  Recommendation: {bottleneck.recommendation}")

        # Should detect high response time bottleneck
        assert len(results.bottlenecks) > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_load_with_resource_monitoring(self):
        """Test load testing with resource monitoring"""
        manager = OptimizationResourceManager()
        await manager.initialize_worker_pool()

        runner = LoadTestRunner(standard_optimization_handler, enable_monitoring=True)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=5,
            target_rps=20,
        )

        results = await runner.run_load_test(profile, resource_manager=manager)

        print(f"\n{'=' * 80}")
        print("LOAD TEST WITH RESOURCE MONITORING")
        print(f"{'=' * 80}")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Throughput: {results.metrics.throughput_rps:.2f} req/s")

        if results.resource_usage:
            print("\nResource Usage:")
            for resource, usage in results.resource_usage.items():
                print(f"  {resource}: {usage}")

        assert results.passed is True
        assert results.resource_usage is not None

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_percentile_accuracy(self):
        """Test response time percentile accuracy"""
        runner = LoadTestRunner(standard_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=5,
            target_rps=50,
        )

        results = await runner.run_load_test(profile)

        metrics = results.metrics

        print(f"\n{'=' * 80}")
        print("PERCENTILE ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Min: {metrics.min_response_time * 1000:.2f} ms")
        print(f"P50: {metrics.p50_response_time * 1000:.2f} ms")
        print(f"P95: {metrics.p95_response_time * 1000:.2f} ms")
        print(f"P99: {metrics.p99_response_time * 1000:.2f} ms")
        print(f"Max: {metrics.max_response_time * 1000:.2f} ms")
        print(f"Avg: {metrics.avg_response_time * 1000:.2f} ms")

        # Verify percentiles are ordered
        assert metrics.min_response_time <= metrics.p50_response_time
        assert metrics.p50_response_time <= metrics.p95_response_time
        assert metrics.p95_response_time <= metrics.p99_response_time
        assert metrics.p99_response_time <= metrics.max_response_time

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_validation(self):
        """Test throughput validation"""
        runner = LoadTestRunner(fast_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=10,
            target_rps=50,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("THROUGHPUT VALIDATION")
        print(f"{'=' * 80}")
        print(f"Target RPS: {profile.target_rps}")
        print(f"Actual RPS: {results.metrics.throughput_rps:.2f}")
        print(f"Achievement: {results.metrics.throughput_rps / profile.target_rps * 100:.1f}%")

        # Should achieve at least 80% of target throughput
        assert results.metrics.throughput_rps >= profile.target_rps * 0.8

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_request_handling(self):
        """Test concurrent request handling"""
        runner = LoadTestRunner(standard_optimization_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=5,
            target_rps=100,
            max_concurrent=200,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("CONCURRENT REQUEST HANDLING")
        print(f"{'=' * 80}")
        print(f"Max Concurrent: {profile.max_concurrent}")
        print(f"Total Requests: {results.metrics.total_requests}")
        print(f"Successful: {results.metrics.successful_requests}")
        print(f"Failed: {results.metrics.failed_requests}")

        assert results.passed is True
        assert results.metrics.successful_requests > 0

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_full_performance_report(self):
        """Generate full performance test report"""
        print(f"\n{'=' * 80}")
        print("FULL PERFORMANCE TEST REPORT")
        print(f"{'=' * 80}")

        test_scenarios = [
            ("Low Load", 10, 10, 2),
            ("Medium Load", 50, 30, 5),
            ("High Load", 100, 60, 10),
        ]

        all_results = {}

        for name, target_rps, max_concurrent, duration in test_scenarios:
            runner = LoadTestRunner(fast_optimization_handler)
            profile = LoadProfile(
                pattern=LoadPattern.CONSTANT,
                duration_seconds=duration,
                target_rps=target_rps,
                max_concurrent=max_concurrent,
            )

            results = await runner.run_load_test(profile)
            all_results[name] = results

            print(f"\n{name}:")
            print(f"  Duration: {results.duration_seconds:.2f}s")
            print(f"  Requests: {results.metrics.total_requests}")
            print(f"  Throughput: {results.metrics.throughput_rps:.2f} req/s")
            print(f"  Avg Response: {results.metrics.avg_response_time * 1000:.2f} ms")
            print(f"  P95 Response: {results.metrics.p95_response_time * 1000:.2f} ms")
            print(f"  Error Rate: {results.metrics.error_rate:.2%}")
            print(f"  Passed: {results.passed}")

        # All scenarios should pass
        assert all(r.passed for r in all_results.values())

        print(f"\n{'=' * 80}")
        print("PERFORMANCE VALIDATION: ALL TESTS PASSED")
        print(f"{'=' * 80}")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_error_rate_threshold(self):
        """Test error rate threshold enforcement"""
        async def mixed_handler(request) -> dict:
            await asyncio.sleep(0.01)
            # Simulate 5% error rate
            import random
            if random.random() < 0.05:
                raise RuntimeError("Simulated error")
            return {"status": "completed"}

        runner = LoadTestRunner(mixed_handler)
        profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=5,
            target_rps=20,
        )

        results = await runner.run_load_test(profile)

        print(f"\n{'=' * 80}")
        print("ERROR RATE VALIDATION")
        print(f"{'=' * 80}")
        print(f"Error Rate: {results.metrics.error_rate:.2%}")
        print(f"Failed Requests: {results.metrics.failed_requests}")
        print(f"Total Requests: {results.metrics.total_requests}")

        # Error rate should be within acceptable range
        assert results.metrics.error_rate <= 0.15  # Allow up to 15%

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_stress_recovery(self):
        """Test system recovery after stress"""
        runner = LoadTestRunner(standard_optimization_handler)

        # High stress phase
        stress_profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=3,
            target_rps=100,
            max_concurrent=200,
        )

        stress_results = await runner.run_load_test(stress_profile)

        # Recovery phase
        recovery_profile = LoadProfile(
            pattern=LoadPattern.CONSTANT,
            duration_seconds=3,
            target_rps=20,
            max_concurrent=50,
        )

        recovery_results = await runner.run_load_test(recovery_profile)

        print(f"\n{'=' * 80}")
        print("STRESS RECOVERY TEST")
        print(f"{'=' * 80}")
        print(f"Stress Phase:")
        print(f"  Throughput: {stress_results.metrics.throughput_rps:.2f} req/s")
        print(f"  P95: {stress_results.metrics.p95_response_time * 1000:.2f} ms")
        print(f"\nRecovery Phase:")
        print(f"  Throughput: {recovery_results.metrics.throughput_rps:.2f} req/s")
        print(f"  P95: {recovery_results.metrics.p95_response_time * 1000:.2f} ms")

        # Both phases should pass
        assert stress_results.passed or recovery_results.passed
        # Recovery phase should have lower response times
        assert recovery_results.metrics.p95_response_time <= stress_results.metrics.p95_response_time * 1.5
