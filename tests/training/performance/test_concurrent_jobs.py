"""
Performance tests for concurrent training job scheduling.

Tests job queue prioritization, worker pool management, and handling of
100+ concurrent jobs.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest
import redis.asyncio as aioredis

from agentcore.a2a_protocol.config import settings
from agentcore.training.job_manager import TrainingJobManager
from agentcore.training.models import (
    GRPOConfig,
    TrainingJobStatus,
    TrainingQuery,
)
from agentcore.training.scheduler import (
    JobPriority,
    TrainingJobScheduler,
)


@pytest.fixture
async def redis_client():
    """Create Redis client for testing."""
    try:
        client = await aioredis.from_url(settings.REDIS_URL, decode_responses=False)

        # Clean up test queues before test
        await client.flushdb()

        yield client

        # Clean up after test
        await client.flushdb()
        await client.aclose()
    except Exception:
        pytest.skip("Redis not available")


@pytest.fixture
async def job_manager():
    """Create training job manager."""
    return TrainingJobManager()


@pytest.fixture
async def scheduler(job_manager, redis_client):
    """Create and initialize training job scheduler."""
    sched = TrainingJobScheduler(job_manager, settings.REDIS_URL)
    await sched.connect()

    yield sched

    # Stop all workers and disconnect
    await sched.stop_worker_pool()
    await sched.disconnect()


@pytest.fixture
def sample_config():
    """Create sample GRPO configuration."""
    return GRPOConfig(
        n_iterations=2,  # Minimal iterations for fast testing
        group_size=4,  # Small group for faster execution
        learning_rate=0.0001,
        max_budget_usd=Decimal("10.0"),
        checkpoint_interval=10,
    )


@pytest.fixture
def sample_training_data():
    """Create sample training data (minimum 100 required)."""
    return [
        TrainingQuery(
            query=f"Test query {i}",
            expected_outcome={"output": f"Test output {i}"},
        )
        for i in range(100)
    ]


@pytest.mark.asyncio
async def test_concurrent_jobs_p2_priority(
    scheduler, job_manager, sample_config, sample_training_data
):
    """Test handling 100+ concurrent jobs with P2 priority."""
    # Create 100 jobs
    num_jobs = 100
    jobs = []

    for i in range(num_jobs):
        job = await job_manager.create_job(
            agent_id=f"test-agent-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        jobs.append(job)

    # Enqueue all jobs with P2 priority
    for job in jobs:
        await scheduler.enqueue_job(job, priority=JobPriority.P2)

    # Verify queue length
    queue_lengths = await scheduler.get_queue_lengths()
    assert queue_lengths["P2"] == num_jobs
    assert queue_lengths["P0"] == 0
    assert queue_lengths["P1"] == 0

    # Start worker pool (10 workers)
    await scheduler.start_worker_pool(pool_size=10)

    # Verify workers started
    assert scheduler.get_worker_count() == 10

    # Wait for all jobs to complete (with timeout)
    timeout = 60  # 60 seconds
    start_time = asyncio.get_event_loop().time()

    while True:
        # Check if all jobs completed
        completed_count = sum(
            1 for j in jobs if j.status == TrainingJobStatus.COMPLETED
        )

        if completed_count == num_jobs:
            break

        # Check timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            pytest.fail(
                f"Timeout: Only {completed_count}/{num_jobs} jobs completed in {timeout}s"
            )

        await asyncio.sleep(1)

    # Verify all jobs completed
    for job in jobs:
        assert job.status == TrainingJobStatus.COMPLETED

    # Verify queue is empty
    queue_lengths = await scheduler.get_queue_lengths()
    assert sum(queue_lengths.values()) == 0

    # Stop workers
    await scheduler.stop_worker_pool()


@pytest.mark.asyncio
async def test_priority_ordering(
    scheduler, job_manager, sample_config, sample_training_data
):
    """Test that jobs are executed in priority order (P0 > P1 > P2)."""
    # Create jobs for each priority
    job_p2 = await job_manager.create_job(
        agent_id="test-agent-p2",
        training_data=sample_training_data,
        config=sample_config,
    )

    job_p1 = await job_manager.create_job(
        agent_id="test-agent-p1",
        training_data=sample_training_data,
        config=sample_config,
    )

    job_p0 = await job_manager.create_job(
        agent_id="test-agent-p0",
        training_data=sample_training_data,
        config=sample_config,
    )

    # Enqueue in reverse priority order (P2, P1, P0)
    await scheduler.enqueue_job(job_p2, priority=JobPriority.P2)
    await scheduler.enqueue_job(job_p1, priority=JobPriority.P1)
    await scheduler.enqueue_job(job_p0, priority=JobPriority.P0)

    # Verify queue lengths
    queue_lengths = await scheduler.get_queue_lengths()
    assert queue_lengths["P0"] == 1
    assert queue_lengths["P1"] == 1
    assert queue_lengths["P2"] == 1

    # Dequeue jobs and verify priority order
    job_id_1, priority_1 = await scheduler.dequeue_job()
    assert priority_1 == JobPriority.P0
    assert job_id_1 == job_p0.job_id

    job_id_2, priority_2 = await scheduler.dequeue_job()
    assert priority_2 == JobPriority.P1
    assert job_id_2 == job_p1.job_id

    job_id_3, priority_3 = await scheduler.dequeue_job()
    assert priority_3 == JobPriority.P2
    assert job_id_3 == job_p2.job_id


@pytest.mark.asyncio
async def test_worker_pool_scaling(scheduler, job_manager, sample_config, sample_training_data):
    """Test worker pool scaling (start, stop, scale up, scale down)."""
    # Start with 5 workers
    await scheduler.start_worker_pool(pool_size=5)
    assert scheduler.get_worker_count() == 5

    # Scale up to 10 workers
    await scheduler.scale_worker_pool(target_size=10)
    assert scheduler.get_worker_count() == 10

    # Scale down to 3 workers
    await scheduler.scale_worker_pool(target_size=3)
    assert scheduler.get_worker_count() == 3

    # Stop all workers
    await scheduler.stop_worker_pool()
    assert scheduler.get_worker_count() == 0


@pytest.mark.asyncio
async def test_worker_health_checks(scheduler):
    """Test worker health check functionality."""
    # Start worker pool
    await scheduler.start_worker_pool(pool_size=3)

    # Wait for workers to update health
    await asyncio.sleep(1)

    # Get worker health
    health = await scheduler.get_worker_health()

    # Verify 3 workers are healthy
    assert len(health) == 3

    for worker_id, worker_health in health.items():
        assert worker_health["status"] == "healthy"
        assert "last_heartbeat" in worker_health
        assert worker_id.startswith("worker-")

    # Stop workers
    await scheduler.stop_worker_pool()

    # Wait for cleanup
    await asyncio.sleep(1)

    # Verify health keys removed
    health = await scheduler.get_worker_health()
    assert len(health) == 0


@pytest.mark.asyncio
async def test_scheduler_health_check(scheduler, job_manager, sample_config, sample_training_data):
    """Test scheduler health check endpoint."""
    # Create and enqueue jobs
    for i in range(5):
        job = await job_manager.create_job(
            agent_id=f"test-agent-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        await scheduler.enqueue_job(job, priority=JobPriority.P1)

    # Start workers
    await scheduler.start_worker_pool(pool_size=2)

    # Wait for health updates
    await asyncio.sleep(1)

    # Perform health check
    health = await scheduler.health_check()

    # Verify health response
    assert health["status"] == "healthy"
    assert health["redis_connected"] is True
    assert "queue_lengths" in health
    assert "active_workers" in health
    assert health["active_workers"] == 2

    # Stop workers
    await scheduler.stop_worker_pool()


@pytest.mark.asyncio
async def test_queue_fifo_within_priority(
    scheduler, job_manager, sample_config, sample_training_data
):
    """Test FIFO ordering within same priority level."""
    # Create 5 jobs with same priority
    jobs = []
    for i in range(5):
        job = await job_manager.create_job(
            agent_id=f"test-agent-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        jobs.append(job)

        # Enqueue with P1 priority
        await scheduler.enqueue_job(job, priority=JobPriority.P1)

    # Dequeue all and verify FIFO order
    for expected_job in jobs:
        job_id, priority = await scheduler.dequeue_job()
        assert job_id == expected_job.job_id
        assert priority == JobPriority.P1


@pytest.mark.asyncio
async def test_mixed_priority_concurrent_execution(
    scheduler, job_manager, sample_config, sample_training_data
):
    """Test concurrent execution with mixed priorities."""
    # Create jobs with mixed priorities
    jobs_p0 = []
    jobs_p1 = []
    jobs_p2 = []

    # Create 10 P0, 20 P1, 30 P2 jobs (60 total)
    for i in range(10):
        job = await job_manager.create_job(
            agent_id=f"test-agent-p0-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        jobs_p0.append(job)
        await scheduler.enqueue_job(job, priority=JobPriority.P0)

    for i in range(20):
        job = await job_manager.create_job(
            agent_id=f"test-agent-p1-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        jobs_p1.append(job)
        await scheduler.enqueue_job(job, priority=JobPriority.P1)

    for i in range(30):
        job = await job_manager.create_job(
            agent_id=f"test-agent-p2-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        jobs_p2.append(job)
        await scheduler.enqueue_job(job, priority=JobPriority.P2)

    # Verify queue lengths
    queue_lengths = await scheduler.get_queue_lengths()
    assert queue_lengths["P0"] == 10
    assert queue_lengths["P1"] == 20
    assert queue_lengths["P2"] == 30

    # Start worker pool
    await scheduler.start_worker_pool(pool_size=5)

    # Wait for all jobs to complete
    timeout = 60
    start_time = asyncio.get_event_loop().time()
    all_jobs = jobs_p0 + jobs_p1 + jobs_p2

    while True:
        completed_count = sum(
            1 for j in all_jobs if j.status == TrainingJobStatus.COMPLETED
        )

        if completed_count == len(all_jobs):
            break

        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            pytest.fail(
                f"Timeout: Only {completed_count}/{len(all_jobs)} jobs completed"
            )

        await asyncio.sleep(1)

    # Verify all jobs completed
    for job in all_jobs:
        assert job.status == TrainingJobStatus.COMPLETED

    # Stop workers
    await scheduler.stop_worker_pool()


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow test
async def test_high_load_150_concurrent_jobs(
    scheduler, job_manager, sample_config, sample_training_data
):
    """Test handling 150 concurrent jobs (exceeds 100+ requirement)."""
    # Create 150 jobs
    num_jobs = 150
    jobs = []

    for i in range(num_jobs):
        job = await job_manager.create_job(
            agent_id=f"test-agent-{i}",
            training_data=sample_training_data,
            config=sample_config,
        )
        jobs.append(job)

        # Mix priorities
        if i % 3 == 0:
            priority = JobPriority.P0
        elif i % 3 == 1:
            priority = JobPriority.P1
        else:
            priority = JobPriority.P2

        await scheduler.enqueue_job(job, priority=priority)

    # Start large worker pool (20 workers)
    await scheduler.start_worker_pool(pool_size=20)

    # Wait for all jobs to complete
    timeout = 120  # 2 minutes
    start_time = asyncio.get_event_loop().time()

    while True:
        completed_count = sum(
            1 for j in jobs if j.status == TrainingJobStatus.COMPLETED
        )

        if completed_count == num_jobs:
            break

        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            pytest.fail(
                f"Timeout: Only {completed_count}/{num_jobs} jobs completed"
            )

        await asyncio.sleep(2)

    # Verify all jobs completed
    for job in jobs:
        assert job.status == TrainingJobStatus.COMPLETED

    # Measure execution time
    total_time = asyncio.get_event_loop().time() - start_time
    avg_time_per_job = total_time / num_jobs

    print(f"\nHigh Load Test Results:")
    print(f"  Total jobs: {num_jobs}")
    print(f"  Workers: 20")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per job: {avg_time_per_job:.2f}s")

    # Stop workers
    await scheduler.stop_worker_pool()
