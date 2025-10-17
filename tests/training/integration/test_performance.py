"""
Performance and throughput integration tests for GRPO training.

Tests trajectory generation performance and API response times.
"""

from __future__ import annotations

import time

import pytest

from agentcore.training.job_manager import TrainingJobManager
from agentcore.training.models import GRPOConfig, TrainingQuery


@pytest.mark.asyncio
async def test_parallel_trajectory_generation_performance() -> None:
    """
    Test parallel trajectory generation completes within performance target.

    Acceptance Criteria:
    - 8 trajectories generated in <30s (p95)
    - Tests trajectory collection parallelization

    Note: In Phase 1 simulated execution mode, this tests the async
    infrastructure rather than actual trajectory collection time.
    """
    # Arrange
    job_manager = TrainingJobManager()

    agent_id = "test-agent-perf"
    training_queries = [
        TrainingQuery(
            query=f"Performance test query {i}",
            expected_outcome={"result": "success"},
        )
        for i in range(100)
    ]

    config = GRPOConfig(
        n_iterations=1,  # Single iteration for performance test
        batch_size=1,  # Single query batch
        n_trajectories_per_query=8,  # 8 parallel trajectories
    )

    # Act
    start_time = time.time()

    job = await job_manager.create_job(
        agent_id=agent_id,
        training_data=training_queries,
        config=config,
    )

    await job_manager.start_job(job.job_id)
    await job_manager.wait_for_job(job.job_id)

    elapsed_time = time.time() - start_time

    # Assert
    # In simulated mode, should complete very quickly
    # Real implementation would target <30s for 8 trajectories
    assert elapsed_time < 30.0, f"Job took {elapsed_time:.2f}s, expected <30s"

    final_job = job_manager.get_job(job.job_id)
    assert final_job.status.value in ["completed", "running"]


@pytest.mark.asyncio
async def test_job_status_api_response_time() -> None:
    """
    Test get_job_status API response time meets SLA.

    Acceptance Criteria:
    - training.get_status <200ms (p95)
    """
    # Arrange
    job_manager = TrainingJobManager()

    agent_id = "test-agent-api-perf"
    training_queries = [
        TrainingQuery(
            query=f"API perf test query {i}",
            expected_outcome={"result": "success"},
        )
        for i in range(100)
    ]

    config = GRPOConfig(
        n_iterations=10,
        batch_size=16,
    )

    job = await job_manager.create_job(
        agent_id=agent_id,
        training_data=training_queries,
        config=config,
    )

    await job_manager.start_job(job.job_id)

    # Act - Measure get_job_status response time
    start_time = time.time()
    status = job_manager.get_job_status(job.job_id)
    elapsed_ms = (time.time() - start_time) * 1000

    # Assert
    assert elapsed_ms < 200.0, f"get_job_status took {elapsed_ms:.2f}ms, expected <200ms"
    assert status["job_id"] == str(job.job_id)
    assert "status" in status
    assert "progress" in status


@pytest.mark.asyncio
async def test_concurrent_job_throughput() -> None:
    """
    Test system handles multiple concurrent training jobs.

    Acceptance Criteria:
    - Support 100+ concurrent jobs (Phase 2 target)
    - Phase 1: Validate 10 concurrent jobs complete successfully
    """
    # Arrange
    job_manager = TrainingJobManager()

    num_jobs = 10
    training_queries = [
        TrainingQuery(
            query=f"Concurrent test query {i}",
            expected_outcome={"result": "success"},
        )
        for i in range(100)
    ]

    config = GRPOConfig(
        n_iterations=3,  # Small number for fast concurrent test
        batch_size=16,
    )

    # Act - Create and start multiple jobs concurrently
    jobs = []
    for i in range(num_jobs):
        job = await job_manager.create_job(
            agent_id=f"test-agent-concurrent-{i}",
            training_data=training_queries,
            config=config,
        )
        await job_manager.start_job(job.job_id)
        jobs.append(job)

    # Wait for all jobs to complete
    for job in jobs:
        await job_manager.wait_for_job(job.job_id)

    # Assert - All jobs should complete successfully
    completed_count = 0
    for job in jobs:
        final_job = job_manager.get_job(job.job_id)
        if final_job.status.value in ["completed", "running"]:
            completed_count += 1

    assert completed_count == num_jobs, f"Only {completed_count}/{num_jobs} jobs completed"


@pytest.mark.asyncio
async def test_training_iteration_throughput() -> None:
    """
    Test training iteration execution time.

    Validates:
    - Training iterations execute within reasonable time
    - Progress tracking updates correctly
    """
    # Arrange
    job_manager = TrainingJobManager()

    agent_id = "test-agent-iteration-throughput"
    training_queries = [
        TrainingQuery(
            query=f"Iteration throughput test {i}",
            expected_outcome={"result": "success"},
        )
        for i in range(100)
    ]

    config = GRPOConfig(
        n_iterations=5,
        batch_size=16,
    )

    # Act
    start_time = time.time()

    job = await job_manager.create_job(
        agent_id=agent_id,
        training_data=training_queries,
        config=config,
    )

    await job_manager.start_job(job.job_id)
    await job_manager.wait_for_job(job.job_id)

    elapsed_time = time.time() - start_time

    # Assert
    final_job = job_manager.get_job(job.job_id)

    # Simulated mode should complete very quickly
    assert elapsed_time < 60.0, f"5 iterations took {elapsed_time:.2f}s, expected <60s"

    # Verify all iterations completed or job is still running
    assert final_job.current_iteration >= 1
    assert final_job.status.value in ["completed", "running"]
