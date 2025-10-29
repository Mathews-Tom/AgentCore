"""
Performance tests for concurrent optimization handling

Validates 1000+ concurrent optimizations target using job queue infrastructure.
Tests 100, 500, and 1000 concurrent jobs with throughput validation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import pytest

try:
    from agentcore.dspy_optimization.scalability.job_queue import (
        JobQueue,
        JobStatus,
        OptimizationJob,
        QueueConfig,
    )
except ImportError as e:
    pytest.skip(f"Required dependencies not available: {e}", allow_module_level=True)


async def mock_optimization_handler(job: OptimizationJob) -> dict:
    """Mock optimization handler for testing"""
    await asyncio.sleep(0.01)  # Simulate work
    return {"status": "completed", "accuracy": 0.95}


async def fast_optimization_handler(job: OptimizationJob) -> dict:
    """Fast optimization handler"""
    await asyncio.sleep(0.001)
    return {"status": "completed", "accuracy": 0.90}


class TestConcurrentOptimizations:
    """Test concurrent optimization performance"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_100_concurrent_optimizations(self):
        """Test 100 concurrent optimizations"""
        config = QueueConfig(
            max_concurrent_jobs=100,
            max_queue_size=1000,
            worker_count=20,
            enable_rate_limiting=False
        )

        queue = JobQueue(config)
        await queue.start()

        try:
            start_time = time.perf_counter()

            # Submit 100 jobs
            job_ids = []
            for i in range(100):
                job = OptimizationJob(
                    optimization_id=f"opt-{i}",
                    priority=i % 10
                )
                job_id = await queue.submit_job(job, mock_optimization_handler)
                job_ids.append(job_id)

            # Wait for all to complete
            results = await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids],
                return_exceptions=True
            )

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Verify all completed
            assert len(results) == 100
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful == 100

            # Check throughput
            stats = queue.get_queue_stats()
            assert stats["completed"] == 100
            assert stats["failed"] == 0

            # Should complete in reasonable time
            assert elapsed < 10.0, f"100 jobs took {elapsed:.2f}s (expected <10s)"

            # Calculate throughput
            throughput = 100 / elapsed
            print(f"\n100 concurrent jobs: {elapsed:.2f}s, {throughput:.2f} jobs/sec")
            assert throughput > 10  # At least 10 jobs/sec

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_500_concurrent_optimizations(self):
        """Test 500 concurrent optimizations"""
        config = QueueConfig(
            max_concurrent_jobs=200,
            max_queue_size=1000,
            worker_count=50,
            enable_rate_limiting=False
        )

        queue = JobQueue(config)
        await queue.start()

        try:
            start_time = time.perf_counter()

            # Submit 500 jobs
            job_ids = []
            for i in range(500):
                job = OptimizationJob(
                    optimization_id=f"opt-{i}",
                    priority=i % 10
                )
                job_id = await queue.submit_job(job, fast_optimization_handler)
                job_ids.append(job_id)

            # Wait for all to complete
            results = await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids],
                return_exceptions=True
            )

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Verify all completed
            assert len(results) == 500
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful == 500

            stats = queue.get_queue_stats()
            assert stats["completed"] == 500

            # Should complete in reasonable time
            assert elapsed < 30.0, f"500 jobs took {elapsed:.2f}s (expected <30s)"

            throughput = 500 / elapsed
            print(f"\n500 concurrent jobs: {elapsed:.2f}s, {throughput:.2f} jobs/sec")
            assert throughput > 15  # At least 15 jobs/sec

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_1000_concurrent_optimizations(self):
        """Test 1000+ concurrent optimizations (validation target)"""
        config = QueueConfig(
            max_concurrent_jobs=500,
            max_queue_size=2000,
            worker_count=100,
            enable_rate_limiting=False
        )

        queue = JobQueue(config)
        await queue.start()

        try:
            start_time = time.perf_counter()

            # Submit 1000 jobs
            job_ids = []
            for i in range(1000):
                job = OptimizationJob(
                    optimization_id=f"opt-{i}",
                    priority=i % 10
                )
                job_id = await queue.submit_job(job, fast_optimization_handler)
                job_ids.append(job_id)

            # Wait for all to complete
            results = await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids],
                return_exceptions=True
            )

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Verify all completed
            assert len(results) == 1000
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful >= 990, f"Only {successful}/1000 jobs succeeded"

            stats = queue.get_queue_stats()
            assert stats["completed"] >= 990

            # Should complete in reasonable time
            assert elapsed < 60.0, f"1000 jobs took {elapsed:.2f}s (expected <60s)"

            throughput = 1000 / elapsed
            print(f"\n1000 concurrent jobs: {elapsed:.2f}s, {throughput:.2f} jobs/sec")
            assert throughput > 15  # At least 15 jobs/sec

            # VALIDATION TARGET: 1000+ concurrent optimizations
            print(f"\nVALIDATION: Successfully handled {successful} concurrent optimizations")
            assert successful >= 1000, "Target: 1000+ concurrent optimizations"

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_queue_backpressure(self):
        """Test queue backpressure handling"""
        config = QueueConfig(
            max_concurrent_jobs=10,
            max_queue_size=50,
            worker_count=5,
            enable_backpressure=True,
            backpressure_threshold=0.9  # Reject at 90% capacity
        )

        queue = JobQueue(config)
        await queue.start()

        try:
            # Fill queue to trigger backpressure
            job_ids = []
            backpressure_triggered = False

            for i in range(100):
                try:
                    job = OptimizationJob(optimization_id=f"opt-{i}")
                    job_id = await queue.submit_job(job, mock_optimization_handler)
                    job_ids.append(job_id)
                except RuntimeError as e:
                    if "capacity" in str(e):
                        backpressure_triggered = True
                        break

            # Backpressure should have triggered
            assert backpressure_triggered, "Backpressure should trigger at 90% capacity"

            # Let queue drain
            await asyncio.sleep(0.5)

            stats = queue.get_queue_stats()
            assert stats["queue_utilization"] < 1.0

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_priority_scheduling(self):
        """Test priority-based job scheduling"""
        config = QueueConfig(
            max_concurrent_jobs=1,  # Force sequential execution
            worker_count=1,
            enable_rate_limiting=False
        )

        queue = JobQueue(config)
        await queue.start()

        completion_order = []

        async def tracking_handler(job: OptimizationJob) -> dict:
            completion_order.append(job.optimization_id)
            await asyncio.sleep(0.01)
            return {"status": "completed"}

        try:
            # Submit jobs with different priorities
            jobs = [
                (OptimizationJob(optimization_id="low", priority=1), tracking_handler),
                (OptimizationJob(optimization_id="high", priority=10), tracking_handler),
                (OptimizationJob(optimization_id="medium", priority=5), tracking_handler),
            ]

            job_ids = []
            for job, handler in jobs:
                job_id = await queue.submit_job(job, handler)
                job_ids.append(job_id)

            # Wait for completion
            await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids]
            )

            # High priority should complete first
            assert completion_order[0] == "high"
            assert completion_order[1] == "medium"
            assert completion_order[2] == "low"

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_with_failures(self):
        """Test concurrent processing with some failures"""
        config = QueueConfig(
            max_concurrent_jobs=50,
            worker_count=10,
            enable_rate_limiting=False
        )

        queue = JobQueue(config)
        await queue.start()

        async def mixed_handler(job: OptimizationJob) -> dict:
            await asyncio.sleep(0.01)
            # Fail 10% of jobs
            if int(job.optimization_id.split("-")[1]) % 10 == 0:
                raise RuntimeError("Simulated failure")
            return {"status": "completed"}

        try:
            # Submit 100 jobs
            job_ids = []
            for i in range(100):
                job = OptimizationJob(
                    optimization_id=f"opt-{i}",
                    max_retries=0  # No retries for this test
                )
                job_id = await queue.submit_job(job, mixed_handler)
                job_ids.append(job_id)

            # Wait for all to complete or fail
            results = await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids],
                return_exceptions=True
            )

            # Check results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = sum(1 for r in results if isinstance(r, Exception))

            assert successful >= 80, f"Expected ~90 successes, got {successful}"
            assert failed <= 20, f"Expected ~10 failures, got {failed}"

            stats = queue.get_queue_stats()
            assert stats["completed"] >= 80
            assert stats["failed"] >= 5

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_scaling(self):
        """Test throughput scaling with worker count"""
        results = {}

        for worker_count in [5, 10, 20]:
            config = QueueConfig(
                max_concurrent_jobs=100,
                worker_count=worker_count,
                enable_rate_limiting=False
            )

            queue = JobQueue(config)
            await queue.start()

            try:
                start_time = time.perf_counter()

                # Submit 100 jobs
                job_ids = []
                for i in range(100):
                    job = OptimizationJob(optimization_id=f"opt-{i}")
                    job_id = await queue.submit_job(job, fast_optimization_handler)
                    job_ids.append(job_id)

                await asyncio.gather(
                    *[queue.get_job_result(job_id) for job_id in job_ids]
                )

                elapsed = time.perf_counter() - start_time
                throughput = 100 / elapsed

                results[worker_count] = {
                    "elapsed": elapsed,
                    "throughput": throughput
                }

            finally:
                await queue.stop()

        # More workers should increase throughput
        print(f"\nThroughput scaling:")
        for workers, metrics in results.items():
            print(f"  {workers} workers: {metrics['throughput']:.2f} jobs/sec")

        # Verify scaling improvement
        assert results[10]["throughput"] > results[5]["throughput"]
        assert results[20]["throughput"] > results[10]["throughput"]

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_queue_utilization_metrics(self):
        """Test queue utilization metrics during load"""
        config = QueueConfig(
            max_concurrent_jobs=50,
            max_queue_size=200,
            worker_count=10
        )

        queue = JobQueue(config)
        await queue.start()

        try:
            # Submit jobs gradually
            job_ids = []
            for i in range(100):
                job = OptimizationJob(optimization_id=f"opt-{i}")
                job_id = await queue.submit_job(job, mock_optimization_handler)
                job_ids.append(job_id)

                if i % 20 == 0:
                    stats = queue.get_queue_stats()
                    print(f"\nAfter {i} jobs:")
                    print(f"  Queue utilization: {stats['queue_utilization']:.2%}")
                    print(f"  Running: {stats['running']}")
                    print(f"  Queued: {stats['queued']}")

                    assert stats["queue_utilization"] <= 1.0
                    assert stats["running"] <= config.max_concurrent_jobs

            # Wait for completion
            await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids]
            )

            final_stats = queue.get_queue_stats()
            assert final_stats["completed"] == 100
            assert final_stats["queued"] == 0
            assert final_stats["running"] == 0

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        config = QueueConfig(
            max_concurrent_jobs=100,
            worker_count=10,
            enable_rate_limiting=True,
            rate_limit_per_second=10  # 10 jobs/sec limit
        )

        queue = JobQueue(config)
        await queue.start()

        try:
            start_time = time.perf_counter()

            # Submit 50 jobs
            job_ids = []
            for i in range(50):
                job = OptimizationJob(optimization_id=f"opt-{i}")
                job_id = await queue.submit_job(job, fast_optimization_handler)
                job_ids.append(job_id)

            await asyncio.gather(
                *[queue.get_job_result(job_id) for job_id in job_ids]
            )

            elapsed = time.perf_counter() - start_time

            # With rate limiting, should take at least 5 seconds (50 jobs / 10 per sec)
            # Allow some tolerance
            assert elapsed >= 4.0, f"Rate limiting too loose: {elapsed:.2f}s"

            stats = queue.get_queue_stats()
            assert stats["completed"] == 50

        finally:
            await queue.stop()
