"""
Concurrent optimization job queue management

Manages async job processing with resource pooling, rate limiting,
and support for 1000+ concurrent optimizations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueConfig:
    """Configuration for job queue"""

    max_concurrent_jobs: int = 100
    max_queue_size: int = 10000
    worker_count: int = 10
    enable_rate_limiting: bool = True
    rate_limit_per_second: int = 50
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.9  # Reject new jobs at 90% capacity


@dataclass
class OptimizationJob:
    """Optimization job metadata"""

    job_id: str = field(default_factory=lambda: str(uuid4()))
    optimization_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    priority: int = 0  # Higher priority = executed first
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    retries: int = 0
    max_retries: int = 3


class JobQueue:
    """
    Async job queue for concurrent optimization processing

    Key features:
    - 1000+ concurrent job support
    - Priority-based scheduling
    - Rate limiting and backpressure
    - Automatic retry on failure
    - Worker pool management
    - Resource pooling
    """

    def __init__(self, config: QueueConfig | None = None) -> None:
        """
        Initialize job queue

        Args:
            config: Queue configuration
        """
        self.config = config or QueueConfig()
        self._queue: asyncio.PriorityQueue[tuple[int, OptimizationJob]] = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        self._jobs: dict[str, OptimizationJob] = {}
        self._active_jobs: set[str] = set()
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._rate_limiter: asyncio.Semaphore | None = None
        self._job_handlers: dict[str, Callable[[OptimizationJob], Awaitable[Any]]] = {}

        if self.config.enable_rate_limiting:
            self._rate_limiter = asyncio.Semaphore(self.config.rate_limit_per_second)

    async def start(self) -> None:
        """Start job queue workers"""
        if self._running:
            logger.warning("Job queue already running")
            return

        self._running = True

        # Start worker tasks
        for i in range(self.config.worker_count):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(
            f"Started job queue with {self.config.worker_count} workers "
            f"(max concurrent: {self.config.max_concurrent_jobs})"
        )

    async def stop(self, graceful: bool = True) -> None:
        """
        Stop job queue

        Args:
            graceful: Wait for active jobs to complete
        """
        if not self._running:
            return

        self._running = False

        if graceful:
            # Wait for queue to drain
            await self._queue.join()

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Stopped job queue")

    async def submit_job(
        self,
        job: OptimizationJob,
        handler: Callable[[OptimizationJob], Awaitable[Any]],
    ) -> str:
        """
        Submit job to queue

        Args:
            job: Job to submit
            handler: Async function to execute job

        Returns:
            Job ID

        Raises:
            RuntimeError: If queue is full or backpressure threshold exceeded
        """
        # Check backpressure
        if self.config.enable_backpressure:
            queue_utilization = self._queue.qsize() / self.config.max_queue_size
            if queue_utilization >= self.config.backpressure_threshold:
                raise RuntimeError(
                    f"Queue at {queue_utilization * 100:.1f}% capacity - rejecting new jobs"
                )

        # Register job handler
        self._job_handlers[job.job_id] = handler

        # Add to tracking
        self._jobs[job.job_id] = job

        # Add to queue (priority, insertion order, job)
        priority = -job.priority  # Negative for max-heap behavior
        try:
            await self._queue.put((priority, job))
            logger.debug(f"Queued job {job.job_id} (priority: {job.priority})")
            return job.job_id
        except asyncio.QueueFull:
            raise RuntimeError("Job queue is full")

    async def get_job_status(self, job_id: str) -> JobStatus | None:
        """
        Get job status

        Args:
            job_id: Job identifier

        Returns:
            Job status or None if not found
        """
        job = self._jobs.get(job_id)
        return job.status if job else None

    async def get_job_result(self, job_id: str) -> Any:
        """
        Get job result (blocks until complete)

        Args:
            job_id: Job identifier

        Returns:
            Job result

        Raises:
            ValueError: If job not found
            RuntimeError: If job failed
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]

        # Wait for completion
        while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            await asyncio.sleep(0.1)

        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job failed: {job.error}")

        if job.status == JobStatus.CANCELLED:
            raise RuntimeError("Job was cancelled")

        return job.result

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel queued or running job

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled
        """
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        job.status = JobStatus.CANCELLED
        logger.info(f"Cancelled job {job_id}")
        return True

    def get_queue_stats(self) -> dict[str, Any]:
        """
        Get queue statistics

        Returns:
            Dictionary with queue stats
        """
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(
                1 for job in self._jobs.values() if job.status == status
            )

        return {
            "total_jobs": len(self._jobs),
            "queued": status_counts[JobStatus.QUEUED.value],
            "running": status_counts[JobStatus.RUNNING.value],
            "completed": status_counts[JobStatus.COMPLETED.value],
            "failed": status_counts[JobStatus.FAILED.value],
            "cancelled": status_counts[JobStatus.CANCELLED.value],
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.config.max_queue_size,
            "queue_utilization": self._queue.qsize() / self.config.max_queue_size,
            "active_workers": len([w for w in self._workers if not w.done()]),
        }

    async def _worker(self, worker_id: int) -> None:
        """
        Worker task that processes jobs from queue

        Args:
            worker_id: Worker identifier
        """
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get next job (with timeout to allow shutdown)
                try:
                    priority, job = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                # Check concurrent limit
                while len(self._active_jobs) >= self.config.max_concurrent_jobs:
                    await asyncio.sleep(0.1)

                # Rate limiting
                if self._rate_limiter:
                    await self._rate_limiter.acquire()
                    # Schedule release after 1 second
                    asyncio.create_task(self._release_rate_limit())

                # Execute job
                self._active_jobs.add(job.job_id)
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()

                try:
                    handler = self._job_handlers.get(job.job_id)
                    if not handler:
                        raise ValueError(f"No handler for job {job.job_id}")

                    # Execute handler
                    result = await handler(job)
                    job.result = result
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()

                    logger.debug(f"Worker {worker_id} completed job {job.job_id}")

                except Exception as e:
                    logger.error(f"Worker {worker_id} job {job.job_id} failed: {e}")

                    # Retry logic
                    if job.retries < job.max_retries:
                        job.retries += 1
                        job.status = JobStatus.QUEUED
                        await self._queue.put((priority, job))
                        logger.info(
                            f"Retrying job {job.job_id} (attempt {job.retries}/{job.max_retries})"
                        )
                    else:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.completed_at = datetime.utcnow()

                finally:
                    self._active_jobs.discard(job.job_id)
                    self._queue.task_done()

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    async def _release_rate_limit(self) -> None:
        """Release rate limiter after delay"""
        await asyncio.sleep(1.0)
        if self._rate_limiter:
            self._rate_limiter.release()

    def clear_completed_jobs(self, older_than_seconds: int = 3600) -> int:
        """
        Clear completed/failed jobs older than threshold

        Args:
            older_than_seconds: Age threshold in seconds

        Returns:
            Number of jobs cleared
        """
        now = datetime.utcnow()
        cleared = 0

        job_ids = list(self._jobs.keys())
        for job_id in job_ids:
            job = self._jobs[job_id]

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds()
                    if age > older_than_seconds:
                        del self._jobs[job_id]
                        if job_id in self._job_handlers:
                            del self._job_handlers[job_id]
                        cleared += 1

        if cleared > 0:
            logger.info(f"Cleared {cleared} old jobs")

        return cleared
