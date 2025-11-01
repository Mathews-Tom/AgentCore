"""
Tests for job queue management
"""

import asyncio
from datetime import UTC, datetime

import pytest

from agentcore.dspy_optimization.scalability.job_queue import (
    JobQueue,
    JobStatus,
    OptimizationJob,
    QueueConfig,
)


@pytest.fixture
async def job_queue():
    """Create and start job queue"""
    config = QueueConfig(
        max_concurrent_jobs=10,
        worker_count=3,
        enable_rate_limiting=False,
        enable_backpressure=True,
        backpressure_threshold=0.8,
    )
    queue = JobQueue(config)
    await queue.start()
    yield queue
    await queue.stop(graceful=False)


async def dummy_handler(job: OptimizationJob) -> str:
    """Dummy job handler"""
    await asyncio.sleep(0.01)
    return f"completed-{job.job_id}"


async def failing_handler(job: OptimizationJob) -> None:
    """Handler that always fails"""
    raise RuntimeError("Intentional failure")


class TestJobQueue:
    """Test job queue functionality"""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping queue"""
        queue = JobQueue()
        await queue.start()

        assert queue._running is True
        assert len(queue._workers) > 0

        await queue.stop()
        assert queue._running is False

    @pytest.mark.asyncio
    async def test_submit_job(self, job_queue):
        """Test submitting job to queue"""
        job = OptimizationJob(optimization_id="test-opt-1")

        job_id = await job_queue.submit_job(job, dummy_handler)

        assert job_id == job.job_id
        assert job.status == JobStatus.QUEUED

    @pytest.mark.asyncio
    async def test_job_execution(self, job_queue):
        """Test job execution"""
        job = OptimizationJob(optimization_id="test-opt-1")

        job_id = await job_queue.submit_job(job, dummy_handler)

        # Wait for completion
        result = await job_queue.get_job_result(job_id)

        assert result == f"completed-{job_id}"
        assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_job_priority(self, job_queue):
        """Test job priority ordering"""
        results = []

        async def tracking_handler(job: OptimizationJob) -> str:
            results.append(job.priority)
            await asyncio.sleep(0.01)
            return "done"

        # Submit jobs with different priorities
        high_priority = OptimizationJob(optimization_id="high", priority=10)
        low_priority = OptimizationJob(optimization_id="low", priority=1)
        medium_priority = OptimizationJob(optimization_id="medium", priority=5)

        await job_queue.submit_job(low_priority, tracking_handler)
        await job_queue.submit_job(high_priority, tracking_handler)
        await job_queue.submit_job(medium_priority, tracking_handler)

        # Wait for all to complete
        await asyncio.sleep(0.5)

        # High priority should be executed first
        assert results[0] == 10

    @pytest.mark.asyncio
    async def test_concurrent_job_limit(self, job_queue):
        """Test concurrent job limit enforcement"""
        config = QueueConfig(max_concurrent_jobs=2, worker_count=5)
        limited_queue = JobQueue(config)
        await limited_queue.start()

        try:
            active_count = []

            async def monitor_handler(job: OptimizationJob) -> str:
                active_count.append(len(limited_queue._active_jobs))
                await asyncio.sleep(0.05)
                return "done"

            # Submit many jobs
            jobs = [OptimizationJob(optimization_id=f"job-{i}") for i in range(10)]
            for job in jobs:
                await limited_queue.submit_job(job, monitor_handler)

            # Wait for completion
            await asyncio.sleep(0.8)

            # Active jobs should never exceed limit
            assert all(count <= 2 for count in active_count)

        finally:
            await limited_queue.stop()

    @pytest.mark.asyncio
    async def test_job_failure_and_retry(self, job_queue):
        """Test job retry on failure"""
        job = OptimizationJob(optimization_id="failing-job", max_retries=2)

        await job_queue.submit_job(job, failing_handler)

        # Wait for retries to complete
        await asyncio.sleep(0.5)

        assert job.status == JobStatus.FAILED
        assert job.retries == 2
        assert job.error is not None

    @pytest.mark.asyncio
    async def test_get_job_status(self, job_queue):
        """Test getting job status"""
        job = OptimizationJob(optimization_id="test-opt-1")

        job_id = await job_queue.submit_job(job, dummy_handler)

        # Initially queued or running
        status = await job_queue.get_job_status(job_id)
        assert status in (JobStatus.QUEUED, JobStatus.RUNNING)

        # Wait for completion
        await job_queue.get_job_result(job_id)

        status = await job_queue.get_job_status(job_id)
        assert status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_job(self, job_queue):
        """Test cancelling job"""

        async def slow_handler(job: OptimizationJob) -> str:
            await asyncio.sleep(1.0)
            return "done"

        job = OptimizationJob(optimization_id="slow-job")
        job_id = await job_queue.submit_job(job, slow_handler)

        # Cancel immediately
        cancelled = await job_queue.cancel_job(job_id)

        assert cancelled is True
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self, job_queue):
        """Test cancelling already completed job"""
        job = OptimizationJob(optimization_id="fast-job")
        job_id = await job_queue.submit_job(job, dummy_handler)

        # Wait for completion
        await job_queue.get_job_result(job_id)

        # Try to cancel
        cancelled = await job_queue.cancel_job(job_id)

        assert cancelled is False

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, job_queue):
        """Test getting queue statistics"""
        # Submit some jobs
        for i in range(5):
            job = OptimizationJob(optimization_id=f"job-{i}")
            await job_queue.submit_job(job, dummy_handler)

        stats = job_queue.get_queue_stats()

        assert stats["total_jobs"] == 5
        assert "queued" in stats
        assert "running" in stats
        assert "completed" in stats
        assert "failed" in stats
        assert "queue_utilization" in stats

    @pytest.mark.asyncio
    async def test_backpressure(self):
        """Test backpressure rejection"""
        config = QueueConfig(
            max_queue_size=10,
            enable_backpressure=True,
            backpressure_threshold=0.5,  # Reject at 50%
            worker_count=1,  # Single worker to ensure queue fills up
        )
        queue = JobQueue(config)
        await queue.start()

        try:
            # Use a slow handler to keep jobs in queue
            async def blocking_handler(job: OptimizationJob) -> str:
                await asyncio.sleep(2.0)  # Long enough to keep jobs queued
                return "done"

            # Fill queue rapidly to trigger backpressure
            # Submit first job and let worker pick it up
            job = OptimizationJob(optimization_id="job-0")
            await queue.submit_job(job, blocking_handler)
            await asyncio.sleep(0.1)  # Let worker pick up the job

            # Now submit 5 more to fill queue to 50% (5/10)
            for i in range(1, 6):
                job = OptimizationJob(optimization_id=f"job-{i}")
                await queue.submit_job(job, blocking_handler)

            # 7th job should be rejected (qsize=5, which is 50%)
            job = OptimizationJob(optimization_id="job-overflow")
            with pytest.raises(RuntimeError, match="capacity"):
                await queue.submit_job(job, blocking_handler)

        finally:
            await queue.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting"""
        config = QueueConfig(
            enable_rate_limiting=True,
            rate_limit_per_second=5,
            worker_count=2,
        )
        queue = JobQueue(config)
        await queue.start()

        try:
            execution_times = []

            async def timed_handler(job: OptimizationJob) -> str:
                execution_times.append(datetime.now(UTC))
                return "done"

            # Submit 10 jobs
            for i in range(10):
                job = OptimizationJob(optimization_id=f"job-{i}")
                await queue.submit_job(job, timed_handler)

            # Wait for completion
            await asyncio.sleep(3.0)

            # Jobs should be rate-limited to ~5 per second
            assert len(execution_times) == 10

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_clear_completed_jobs(self, job_queue):
        """Test clearing old completed jobs"""
        # Submit and complete jobs
        for i in range(5):
            job = OptimizationJob(optimization_id=f"job-{i}")
            await job_queue.submit_job(job, dummy_handler)

        # Wait for completion
        await asyncio.sleep(0.2)

        # Clear recent jobs (should not clear since they're fresh)
        cleared = job_queue.clear_completed_jobs(older_than_seconds=1000)
        assert cleared == 0

        # Clear old jobs
        cleared = job_queue.clear_completed_jobs(older_than_seconds=0)
        assert cleared == 5

    @pytest.mark.asyncio
    async def test_multiple_workers(self):
        """Test multiple workers processing jobs"""
        config = QueueConfig(worker_count=5)
        queue = JobQueue(config)
        await queue.start()

        try:
            # Submit many jobs
            jobs = []
            for i in range(20):
                job = OptimizationJob(optimization_id=f"job-{i}")
                await queue.submit_job(job, dummy_handler)
                jobs.append(job)

            # Wait for all to complete
            for job in jobs:
                await queue.get_job_result(job.job_id)

            # All should be completed
            assert all(job.status == JobStatus.COMPLETED for job in jobs)

        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown waits for jobs"""
        queue = JobQueue()
        await queue.start()

        async def slow_handler(job: OptimizationJob) -> str:
            await asyncio.sleep(0.2)
            return "done"

        # Submit jobs
        jobs = []
        for i in range(3):
            job = OptimizationJob(optimization_id=f"job-{i}")
            await queue.submit_job(job, slow_handler)
            jobs.append(job)

        # Graceful shutdown
        await queue.stop(graceful=True)

        # All jobs should complete
        assert all(
            job.status in (JobStatus.COMPLETED, JobStatus.RUNNING) for job in jobs
        )

    @pytest.mark.asyncio
    async def test_job_not_found(self, job_queue):
        """Test getting result for non-existent job"""
        with pytest.raises(ValueError, match="not found"):
            await job_queue.get_job_result("nonexistent")

    @pytest.mark.asyncio
    async def test_failed_job_result(self, job_queue):
        """Test getting result for failed job"""
        job = OptimizationJob(optimization_id="failing", max_retries=0)
        job_id = await job_queue.submit_job(job, failing_handler)

        with pytest.raises(RuntimeError, match="failed"):
            await job_queue.get_job_result(job_id)
