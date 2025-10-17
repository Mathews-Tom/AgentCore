"""
Training job scheduler with Redis-based priority queue.

Implements job queue prioritization, worker pool management, and health checks.
"""

from __future__ import annotations

import asyncio
import enum
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
import structlog

from agentcore.a2a_protocol.config import settings
from agentcore.training.job_manager import TrainingJobManager
from agentcore.training.models import TrainingJob, TrainingJobStatus

logger = structlog.get_logger()


class JobPriority(enum.Enum):
    """Job priority levels."""

    P0 = 0  # Critical - highest priority
    P1 = 1  # High priority
    P2 = 2  # Normal priority


class TrainingJobScheduler:
    """
    Training job scheduler with Redis-based priority queue.

    Implements priority-based job scheduling, worker pool management,
    and health monitoring for distributed training execution.
    """

    # Redis queue names by priority
    QUEUE_NAMES = {
        JobPriority.P0: "training:queue:p0",
        JobPriority.P1: "training:queue:p1",
        JobPriority.P2: "training:queue:p2",
    }

    # Worker health tracking key prefix
    WORKER_HEALTH_PREFIX = "training:worker:health:"

    def __init__(
        self,
        job_manager: TrainingJobManager,
        redis_url: str | None = None,
    ) -> None:
        """
        Initialize training job scheduler.

        Args:
            job_manager: Training job manager for execution
            redis_url: Redis connection URL (uses config default if None)
        """
        self.job_manager = job_manager
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis: aioredis.Redis[bytes] | None = None
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._running = False

        logger.info("training_job_scheduler_initialized", redis_url=self.redis_url)

    async def connect(self) -> None:
        """Connect to Redis."""
        self._redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=False,
            max_connections=20,
        )

        # Verify connection
        await self._redis.ping()

        logger.info("scheduler_redis_connected")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

        logger.info("scheduler_redis_disconnected")

    @property
    def redis(self) -> aioredis.Redis[bytes]:
        """Get Redis client."""
        if not self._redis:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._redis

    async def enqueue_job(
        self,
        job: TrainingJob,
        priority: JobPriority = JobPriority.P2,
    ) -> None:
        """
        Enqueue training job to priority queue.

        Args:
            job: Training job to enqueue
            priority: Job priority level (default: P2)
        """
        queue_name = self.QUEUE_NAMES[priority]

        # Serialize job to JSON
        job_data = {
            "job_id": str(job.job_id),
            "agent_id": job.agent_id,
            "priority": priority.value,
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        }

        # Push to priority queue (RPUSH for FIFO within priority)
        await self.redis.rpush(queue_name, json.dumps(job_data))

        # Update job status to QUEUED
        job.status = TrainingJobStatus.QUEUED

        logger.info(
            "job_enqueued",
            job_id=str(job.job_id),
            priority=priority.name,
            queue=queue_name,
        )

    async def dequeue_job(self) -> tuple[UUID, JobPriority] | None:
        """
        Dequeue next job from highest priority queue.

        Returns:
            Tuple of (job_id, priority) if job available, None otherwise
        """
        # Check queues in priority order (P0 -> P1 -> P2)
        for priority in [JobPriority.P0, JobPriority.P1, JobPriority.P2]:
            queue_name = self.QUEUE_NAMES[priority]

            # BLPOP with 1 second timeout
            result = await self.redis.blpop(queue_name, timeout=1)

            if result:
                _, job_data_bytes = result
                job_data = json.loads(job_data_bytes.decode("utf-8"))
                job_id = UUID(job_data["job_id"])

                logger.info(
                    "job_dequeued",
                    job_id=str(job_id),
                    priority=priority.name,
                    queue=queue_name,
                )

                return job_id, priority

        return None

    async def get_queue_lengths(self) -> dict[str, int]:
        """
        Get current queue lengths for all priorities.

        Returns:
            Dictionary mapping priority name to queue length
        """
        lengths = {}
        for priority, queue_name in self.QUEUE_NAMES.items():
            length = await self.redis.llen(queue_name)
            lengths[priority.name] = length

        return lengths

    async def start_worker(self, worker_id: str) -> None:
        """
        Start worker process.

        Args:
            worker_id: Unique worker identifier
        """
        if worker_id in self._workers:
            raise ValueError(f"Worker {worker_id} already running")

        task = asyncio.create_task(self._worker_loop(worker_id))
        self._workers[worker_id] = task

        logger.info("worker_started", worker_id=worker_id)

    async def stop_worker(self, worker_id: str) -> None:
        """
        Stop worker process.

        Args:
            worker_id: Worker identifier
        """
        if worker_id not in self._workers:
            raise ValueError(f"Worker {worker_id} not found")

        task = self._workers[worker_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        del self._workers[worker_id]

        logger.info("worker_stopped", worker_id=worker_id)

    async def start_worker_pool(self, pool_size: int) -> None:
        """
        Start worker pool with specified size.

        Args:
            pool_size: Number of workers to start
        """
        self._running = True

        for i in range(pool_size):
            worker_id = f"worker-{i}"
            await self.start_worker(worker_id)

        logger.info("worker_pool_started", pool_size=pool_size)

    async def stop_worker_pool(self) -> None:
        """Stop all workers in the pool."""
        self._running = False

        # Cancel all workers
        worker_ids = list(self._workers.keys())
        for worker_id in worker_ids:
            await self.stop_worker(worker_id)

        logger.info("worker_pool_stopped")

    async def scale_worker_pool(self, target_size: int) -> None:
        """
        Scale worker pool to target size.

        Args:
            target_size: Target number of workers
        """
        current_size = len(self._workers)

        if target_size > current_size:
            # Scale up
            for i in range(current_size, target_size):
                worker_id = f"worker-{i}"
                await self.start_worker(worker_id)

            logger.info(
                "worker_pool_scaled_up",
                from_size=current_size,
                to_size=target_size,
            )

        elif target_size < current_size:
            # Scale down
            worker_ids = list(self._workers.keys())
            for i in range(target_size, current_size):
                worker_id = worker_ids[i]
                await self.stop_worker(worker_id)

            logger.info(
                "worker_pool_scaled_down",
                from_size=current_size,
                to_size=target_size,
            )

    async def _worker_loop(self, worker_id: str) -> None:
        """
        Worker main loop - dequeue and execute jobs.

        Args:
            worker_id: Worker identifier
        """
        logger.info("worker_loop_started", worker_id=worker_id)

        try:
            while self._running:
                # Update worker health
                await self._update_worker_health(worker_id)

                # Dequeue next job
                job_info = await self.dequeue_job()

                if not job_info:
                    # No jobs available, continue loop
                    await asyncio.sleep(0.1)
                    continue

                job_id, priority = job_info

                try:
                    # Get job from job manager
                    job = self.job_manager.get_job(job_id)

                    logger.info(
                        "worker_executing_job",
                        worker_id=worker_id,
                        job_id=str(job_id),
                        priority=priority.name,
                    )

                    # Execute job
                    await self.job_manager.start_job(job_id)

                    # Wait for job completion
                    await self.job_manager.wait_for_job(job_id)

                    logger.info(
                        "worker_job_completed",
                        worker_id=worker_id,
                        job_id=str(job_id),
                        status=job.status.value,
                    )

                except Exception as e:
                    logger.error(
                        "worker_job_failed",
                        worker_id=worker_id,
                        job_id=str(job_id),
                        error=str(e),
                    )

        except asyncio.CancelledError:
            logger.info("worker_loop_cancelled", worker_id=worker_id)
            raise

        except Exception as e:
            logger.error("worker_loop_error", worker_id=worker_id, error=str(e))
            raise

        finally:
            # Remove worker health key
            await self._remove_worker_health(worker_id)
            logger.info("worker_loop_stopped", worker_id=worker_id)

    async def _update_worker_health(self, worker_id: str) -> None:
        """
        Update worker health status in Redis.

        Args:
            worker_id: Worker identifier
        """
        health_key = f"{self.WORKER_HEALTH_PREFIX}{worker_id}"

        health_data = {
            "worker_id": worker_id,
            "status": "healthy",
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        }

        # Set with 30 second TTL (worker should update every 10s)
        await self.redis.setex(
            health_key,
            30,
            json.dumps(health_data),
        )

    async def _remove_worker_health(self, worker_id: str) -> None:
        """
        Remove worker health status from Redis.

        Args:
            worker_id: Worker identifier
        """
        health_key = f"{self.WORKER_HEALTH_PREFIX}{worker_id}"
        await self.redis.delete(health_key)

    async def get_worker_health(self) -> dict[str, Any]:
        """
        Get health status of all workers.

        Returns:
            Dictionary mapping worker_id to health status
        """
        # Scan for all worker health keys
        pattern = f"{self.WORKER_HEALTH_PREFIX}*"
        cursor = 0
        workers_health = {}

        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:
                health_data_bytes = await self.redis.get(key)
                if health_data_bytes:
                    health_data = json.loads(health_data_bytes.decode("utf-8"))
                    worker_id = health_data["worker_id"]
                    workers_health[worker_id] = health_data

            if cursor == 0:
                break

        return workers_health

    async def health_check(self) -> dict[str, Any]:
        """
        Perform scheduler health check.

        Returns:
            Dictionary with health status information
        """
        try:
            # Check Redis connectivity
            await self.redis.ping()

            # Get queue lengths
            queue_lengths = await self.get_queue_lengths()

            # Get worker health
            workers_health = await self.get_worker_health()

            # Count active workers
            active_workers = len([w for w in workers_health.values() if w["status"] == "healthy"])

            return {
                "status": "healthy",
                "redis_connected": True,
                "queue_lengths": queue_lengths,
                "total_queued_jobs": sum(queue_lengths.values()),
                "active_workers": active_workers,
                "workers": workers_health,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_worker_count(self) -> int:
        """
        Get current number of workers.

        Returns:
            Worker count
        """
        return len(self._workers)


# Singleton scheduler instance (initialized by application)
_scheduler: TrainingJobScheduler | None = None


def get_scheduler() -> TrainingJobScheduler:
    """
    Get global scheduler instance.

    Returns:
        Training job scheduler

    Raises:
        RuntimeError: If scheduler not initialized
    """
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized. Call init_scheduler() first.")

    return _scheduler


async def init_scheduler(job_manager: TrainingJobManager, redis_url: str | None = None) -> TrainingJobScheduler:
    """
    Initialize global scheduler instance.

    Args:
        job_manager: Training job manager
        redis_url: Redis connection URL (optional)

    Returns:
        Initialized scheduler
    """
    global _scheduler

    _scheduler = TrainingJobScheduler(job_manager, redis_url)
    await _scheduler.connect()

    logger.info("scheduler_initialized")

    return _scheduler


async def close_scheduler() -> None:
    """Close global scheduler instance."""
    global _scheduler

    if _scheduler:
        await _scheduler.stop_worker_pool()
        await _scheduler.disconnect()
        _scheduler = None

    logger.info("scheduler_closed")
