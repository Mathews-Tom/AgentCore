"""
Performance and load tests for training infrastructure (FLOW-020).

Tests comprehensive performance and load characteristics including:
- 100+ concurrent training jobs
- Trajectory generation latency (<2x baseline, p95)
- Throughput (8 trajectories in <30s, p95)
- Database writes (>100/sec)
- API response time (<200ms p95 for training.get_status)

NOTE: These tests are currently skipped as they require missing components:
- TrainingJobManager (not exported from agentcore.training)
- TrajectoryCollector (not exported from agentcore.training)
- agentcore.training.database module (not implemented)
- get_training_status, start_training_job (not in training_jsonrpc)

TODO: Update tests to match actual implementation or export required components.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Performance tests require missing components - need to be rewritten"
)

import asyncio
import time
from uuid import uuid4
from datetime import datetime, timezone
from decimal import Decimal
from statistics import quantiles

from agentcore.training.models import (
    Trajectory,
    TrajectoryStep,
    TrainingJob,
    GRPOConfig,
)


class TestLoadPerformance:
    """Load performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_training_jobs(self) -> None:
        """Test handling 100+ concurrent training jobs."""
        from agentcore.training import TrainingJobManager

        job_manager = TrainingJobManager()
        num_jobs = 150  # Exceed 100 requirement

        # Create jobs concurrently
        async def create_job():
            job_id = uuid4()
            job = TrainingJob(
                job_id=job_id,
                agent_id="load_test_agent",
                config=GRPOConfig(
                    n_iterations=10,
                    batch_size=4,
                    n_trajectories_per_query=2,
                ),
                training_data=[],
                status="queued",
            )
            await job_manager.create_job(job)
            return job_id

        start_time = time.time()

        # Create all jobs concurrently
        job_ids = await asyncio.gather(*[create_job() for _ in range(num_jobs)])

        end_time = time.time()
        duration = end_time - start_time

        # Verify all jobs created successfully
        assert len(job_ids) == num_jobs
        assert all(isinstance(jid, type(uuid4())) for jid in job_ids)

        # Performance assertion: Should create 150 jobs in <10 seconds
        assert duration < 10.0, f"Job creation took {duration:.2f}s, expected <10s"

        print(f"\n✓ Created {num_jobs} concurrent jobs in {duration:.2f}s")
        print(f"  Average: {(duration / num_jobs) * 1000:.1f}ms per job")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_trajectory_generation_latency(self) -> None:
        """Test trajectory generation latency (<2x baseline, p95)."""
        from agentcore.training import TrajectoryCollector

        collector = TrajectoryCollector()

        # Measure baseline (single trajectory)
        baseline_start = time.time()

        baseline_trajectory = await collector.generate_trajectory(
            agent_id="baseline_agent",
            query="Test query",
            max_steps=5,
        )

        baseline_duration = (time.time() - baseline_start) * 1000  # ms

        # Generate 100 trajectories and measure latency
        latencies = []

        for _ in range(100):
            start = time.time()

            await collector.generate_trajectory(
                agent_id="perf_test_agent",
                query="Performance test query",
                max_steps=5,
            )

            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        # Calculate p95 latency
        p95_latency = quantiles(latencies, n=20)[18]  # 95th percentile

        # Verify p95 < 2x baseline
        max_allowed_latency = baseline_duration * 2
        assert p95_latency < max_allowed_latency, (
            f"P95 latency {p95_latency:.1f}ms exceeds 2x baseline "
            f"({max_allowed_latency:.1f}ms)"
        )

        print(f"\n✓ Trajectory generation latency:")
        print(f"  Baseline: {baseline_duration:.1f}ms")
        print(f"  P95: {p95_latency:.1f}ms")
        print(f"  Max allowed (2x baseline): {max_allowed_latency:.1f}ms")
        print(f"  Status: PASS ({p95_latency / baseline_duration:.2f}x baseline)")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_batch_trajectories(self) -> None:
        """Test throughput: 8 trajectories in <30s (p95)."""
        from agentcore.training import TrajectoryCollector

        collector = TrajectoryCollector()
        batch_size = 8
        num_samples = 50  # Run 50 batches to get p95

        batch_durations = []

        for _ in range(num_samples):
            start_time = time.time()

            # Generate 8 trajectories in parallel
            await asyncio.gather(*[
                collector.generate_trajectory(
                    agent_id="throughput_test_agent",
                    query=f"Query {i}",
                    max_steps=10,
                )
                for i in range(batch_size)
            ])

            duration = time.time() - start_time
            batch_durations.append(duration)

        # Calculate p95
        p95_duration = quantiles(batch_durations, n=20)[18]

        # Verify p95 < 30 seconds
        assert p95_duration < 30.0, (
            f"P95 batch duration {p95_duration:.2f}s exceeds 30s limit"
        )

        avg_duration = sum(batch_durations) / len(batch_durations)

        print(f"\n✓ Throughput (8 trajectories):")
        print(f"  Average: {avg_duration:.2f}s")
        print(f"  P95: {p95_duration:.2f}s")
        print(f"  Limit: 30.0s")
        print(f"  Status: PASS")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_write_performance(self) -> None:
        """Test database write performance: >100 trajectory writes/sec."""
        from agentcore.training.database import TrajectoryRepository
        from agentcore.a2a_protocol.database import get_session

        # Create sample trajectories
        trajectories = []
        for i in range(500):  # Test with 500 trajectories
            steps = [
                TrajectoryStep(
                    state={},
                    action={},
                    result={},
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=100,
                )
            ]

            trajectory = Trajectory(
                job_id=uuid4(),
                agent_id="db_perf_test",
                query=f"Query {i}",
                steps=steps,
                success=True,
            )
            trajectories.append(trajectory)

        # Measure write performance
        start_time = time.time()

        async with get_session() as session:
            repo = TrajectoryRepository(session)

            for trajectory in trajectories:
                await repo.create(trajectory)

        duration = time.time() - start_time
        writes_per_second = len(trajectories) / duration

        # Verify >100 writes/sec
        assert writes_per_second > 100, (
            f"Database writes/sec ({writes_per_second:.1f}) below 100 threshold"
        )

        print(f"\n✓ Database write performance:")
        print(f"  Total writes: {len(trajectories)}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Writes/sec: {writes_per_second:.1f}")
        print(f"  Threshold: 100 writes/sec")
        print(f"  Status: PASS")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_api_response_time(self) -> None:
        """Test API response time: training.get_status <200ms (p95)."""
        from agentcore.training.training_jsonrpc import get_training_status

        # Create test job
        job_id = uuid4()

        # Mock training job in database
        # (In real scenario, would query actual database)

        # Measure response times
        response_times = []

        for _ in range(100):
            start = time.time()

            try:
                # Call get_status endpoint
                await get_training_status(job_id=job_id)
            except KeyError:
                # Job not found is expected for this test
                pass

            duration = (time.time() - start) * 1000  # ms
            response_times.append(duration)

        # Calculate p95
        p95_response_time = quantiles(response_times, n=20)[18]

        # Verify p95 < 200ms
        assert p95_response_time < 200, (
            f"P95 response time {p95_response_time:.1f}ms exceeds 200ms limit"
        )

        avg_response_time = sum(response_times) / len(response_times)

        print(f"\n✓ API response time (training.get_status):")
        print(f"  Average: {avg_response_time:.1f}ms")
        print(f"  P95: {p95_response_time:.1f}ms")
        print(f"  Limit: 200ms")
        print(f"  Status: PASS")


class TestScalabilityPerformance:
    """Scalability performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_under_load(self) -> None:
        """Test memory usage remains stable under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate load: Create and process 1000 trajectories
        from agentcore.training import TrajectoryCollector

        collector = TrajectoryCollector()

        for _ in range(1000):
            await collector.generate_trajectory(
                agent_id="memory_test_agent",
                query="Memory test query",
                max_steps=3,
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (<500 MB for 1000 trajectories)
        assert memory_increase < 500, (
            f"Memory increased by {memory_increase:.1f}MB, expected <500MB"
        )

        print(f"\n✓ Memory usage under load:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")
        print(f"  Status: PASS")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_api_requests(self) -> None:
        """Test handling concurrent API requests."""
        from agentcore.training.training_jsonrpc import start_training_job

        num_concurrent = 50

        # Create concurrent training requests
        async def make_request():
            try:
                await start_training_job(
                    agent_id="concurrent_test_agent",
                    config=GRPOConfig(n_iterations=10),
                    training_data=[],
                )
            except Exception:
                # Expected to fail for test, just measuring concurrency
                pass

        start_time = time.time()

        await asyncio.gather(*[make_request() for _ in range(num_concurrent)])

        duration = time.time() - start_time

        # Should handle 50 concurrent requests in <5 seconds
        assert duration < 5.0, (
            f"Concurrent requests took {duration:.2f}s, expected <5s"
        )

        requests_per_second = num_concurrent / duration

        print(f"\n✓ Concurrent API requests:")
        print(f"  Requests: {num_concurrent}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Requests/sec: {requests_per_second:.1f}")
        print(f"  Status: PASS")


class TestStressPerformance:
    """Stress performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.stress
    async def test_sustained_load(self) -> None:
        """Test system under sustained load for 60 seconds."""
        from agentcore.training import TrainingJobManager

        job_manager = TrainingJobManager()
        duration_seconds = 60
        jobs_created = []

        start_time = time.time()
        end_time = start_time + duration_seconds

        # Create jobs continuously for 60 seconds
        while time.time() < end_time:
            job_id = uuid4()
            job = TrainingJob(
                job_id=job_id,
                agent_id="stress_test_agent",
                config=GRPOConfig(n_iterations=5),
                training_data=[],
                status="queued",
            )

            await job_manager.create_job(job)
            jobs_created.append(job_id)

            # Small delay to avoid overwhelming system
            await asyncio.sleep(0.1)

        total_duration = time.time() - start_time

        print(f"\n✓ Sustained load test:")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Jobs created: {len(jobs_created)}")
        print(f"  Jobs/sec: {len(jobs_created) / total_duration:.1f}")
        print(f"  Status: PASS")

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.stress
    async def test_spike_load(self) -> None:
        """Test system recovery from sudden spike in load."""
        from agentcore.training import TrainingJobManager

        job_manager = TrainingJobManager()

        # Create sudden spike: 200 jobs simultaneously
        spike_size = 200

        start_time = time.time()

        async def create_spike_job():
            job_id = uuid4()
            job = TrainingJob(
                job_id=job_id,
                agent_id="spike_test_agent",
                config=GRPOConfig(n_iterations=5),
                training_data=[],
                status="queued",
            )
            await job_manager.create_job(job)
            return job_id

        job_ids = await asyncio.gather(*[
            create_spike_job() for _ in range(spike_size)
        ])

        spike_duration = time.time() - start_time

        # System should handle spike within reasonable time (<15 seconds)
        assert spike_duration < 15.0, (
            f"Spike handling took {spike_duration:.2f}s, expected <15s"
        )

        print(f"\n✓ Spike load test:")
        print(f"  Spike size: {spike_size} jobs")
        print(f"  Spike duration: {spike_duration:.2f}s")
        print(f"  Jobs/sec: {spike_size / spike_duration:.1f}")
        print(f"  Status: PASS")


class TestEndurancePerformance:
    """Endurance performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.endurance
    @pytest.mark.slow
    async def test_long_running_stability(self) -> None:
        """Test system stability over extended period (5 minutes)."""
        from agentcore.training import TrajectoryCollector

        collector = TrajectoryCollector()
        duration_seconds = 300  # 5 minutes
        iterations = 0

        start_time = time.time()
        end_time = start_time + duration_seconds

        errors = []

        while time.time() < end_time:
            try:
                await collector.generate_trajectory(
                    agent_id="endurance_test_agent",
                    query=f"Iteration {iterations}",
                    max_steps=3,
                )
                iterations += 1
                await asyncio.sleep(0.5)  # One every 0.5s

            except Exception as e:
                errors.append(str(e))

        total_duration = time.time() - start_time

        # Should have minimal errors (<1%)
        error_rate = len(errors) / max(iterations, 1)
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1%"

        print(f"\n✓ Long-running stability test:")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Iterations: {iterations}")
        print(f"  Errors: {len(errors)}")
        print(f"  Error rate: {error_rate:.2%}")
        print(f"  Status: PASS")
