"""
Prometheus metrics for GRPO training.

Implements metrics export for training jobs, performance, and budget tracking.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

# Training job metrics
training_jobs_created = Counter(
    "training_jobs_created_total",
    "Total number of training jobs created",
    ["agent_id"],
)

training_jobs_completed = Counter(
    "training_jobs_completed_total",
    "Total number of training jobs completed successfully",
    ["agent_id"],
)

training_jobs_failed = Counter(
    "training_jobs_failed_total",
    "Total number of training jobs failed",
    ["agent_id"],
)

training_jobs_cancelled = Counter(
    "training_jobs_cancelled_total",
    "Total number of training jobs cancelled",
    ["agent_id"],
)

training_jobs_active = Gauge(
    "training_jobs_active",
    "Number of currently active training jobs",
)

# Performance metrics
trajectory_generation_duration = Histogram(
    "training_trajectory_generation_duration_seconds",
    "Time to generate trajectory batch",
    ["agent_id"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

policy_update_duration = Histogram(
    "training_policy_update_duration_seconds",
    "Time to update policy after training iteration",
    ["agent_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

training_iteration_duration = Histogram(
    "training_iteration_duration_seconds",
    "Time to complete one training iteration",
    ["agent_id"],
    buckets=[5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

checkpoint_save_duration = Histogram(
    "training_checkpoint_save_duration_seconds",
    "Time to save checkpoint",
    ["agent_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Budget metrics
training_budget_usage = Gauge(
    "training_budget_usage_usd",
    "Current budget usage in USD",
    ["agent_id", "job_id"],
)

training_budget_limit = Gauge(
    "training_budget_limit_usd",
    "Budget limit in USD",
    ["agent_id", "job_id"],
)

training_budget_utilization = Gauge(
    "training_budget_utilization_percent",
    "Budget utilization percentage (0-100)",
    ["agent_id", "job_id"],
)

# Training progress metrics
training_iteration_current = Gauge(
    "training_iteration_current",
    "Current training iteration",
    ["agent_id", "job_id"],
)

training_iteration_total = Gauge(
    "training_iteration_total",
    "Total training iterations planned",
    ["agent_id", "job_id"],
)

training_loss = Gauge(
    "training_loss",
    "Current training loss",
    ["agent_id", "job_id"],
)

training_reward_mean = Gauge(
    "training_reward_mean",
    "Mean reward across trajectories",
    ["agent_id", "job_id"],
)


class TrainingMetrics:
    """
    Training metrics collector for Prometheus export.

    Provides methods to record training job lifecycle, performance,
    and budget metrics.
    """

    @staticmethod
    def job_created(agent_id: str) -> None:
        """
        Record training job creation.

        Args:
            agent_id: Agent identifier
        """
        training_jobs_created.labels(agent_id=agent_id).inc()
        training_jobs_active.inc()

    @staticmethod
    def job_completed(agent_id: str) -> None:
        """
        Record training job completion.

        Args:
            agent_id: Agent identifier
        """
        training_jobs_completed.labels(agent_id=agent_id).inc()
        training_jobs_active.dec()

    @staticmethod
    def job_failed(agent_id: str) -> None:
        """
        Record training job failure.

        Args:
            agent_id: Agent identifier
        """
        training_jobs_failed.labels(agent_id=agent_id).inc()
        training_jobs_active.dec()

    @staticmethod
    def job_cancelled(agent_id: str) -> None:
        """
        Record training job cancellation.

        Args:
            agent_id: Agent identifier
        """
        training_jobs_cancelled.labels(agent_id=agent_id).inc()
        training_jobs_active.dec()

    @staticmethod
    @contextmanager
    def measure_trajectory_generation(agent_id: str) -> Iterator[None]:
        """
        Measure trajectory generation duration.

        Args:
            agent_id: Agent identifier

        Yields:
            None

        Example:
            with TrainingMetrics.measure_trajectory_generation(agent_id):
                trajectories = generate_trajectories(...)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            trajectory_generation_duration.labels(agent_id=agent_id).observe(duration)

    @staticmethod
    @contextmanager
    def measure_policy_update(agent_id: str) -> Iterator[None]:
        """
        Measure policy update duration.

        Args:
            agent_id: Agent identifier

        Yields:
            None

        Example:
            with TrainingMetrics.measure_policy_update(agent_id):
                update_policy(...)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            policy_update_duration.labels(agent_id=agent_id).observe(duration)

    @staticmethod
    @contextmanager
    def measure_training_iteration(agent_id: str) -> Iterator[None]:
        """
        Measure training iteration duration.

        Args:
            agent_id: Agent identifier

        Yields:
            None

        Example:
            with TrainingMetrics.measure_training_iteration(agent_id):
                run_iteration(...)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            training_iteration_duration.labels(agent_id=agent_id).observe(duration)

    @staticmethod
    @contextmanager
    def measure_checkpoint_save(agent_id: str) -> Iterator[None]:
        """
        Measure checkpoint save duration.

        Args:
            agent_id: Agent identifier

        Yields:
            None

        Example:
            with TrainingMetrics.measure_checkpoint_save(agent_id):
                save_checkpoint(...)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            checkpoint_save_duration.labels(agent_id=agent_id).observe(duration)

    @staticmethod
    def update_budget(
        agent_id: str,
        job_id: str,
        usage_usd: float,
        limit_usd: float,
    ) -> None:
        """
        Update budget metrics.

        Args:
            agent_id: Agent identifier
            job_id: Training job ID
            usage_usd: Current budget usage in USD
            limit_usd: Budget limit in USD
        """
        training_budget_usage.labels(agent_id=agent_id, job_id=job_id).set(usage_usd)
        training_budget_limit.labels(agent_id=agent_id, job_id=job_id).set(limit_usd)

        # Calculate utilization percentage
        utilization = (usage_usd / limit_usd * 100) if limit_usd > 0 else 0.0
        training_budget_utilization.labels(agent_id=agent_id, job_id=job_id).set(
            utilization
        )

    @staticmethod
    def update_progress(
        agent_id: str,
        job_id: str,
        current_iteration: int,
        total_iterations: int,
    ) -> None:
        """
        Update training progress metrics.

        Args:
            agent_id: Agent identifier
            job_id: Training job ID
            current_iteration: Current iteration number
            total_iterations: Total iterations planned
        """
        training_iteration_current.labels(agent_id=agent_id, job_id=job_id).set(
            current_iteration
        )
        training_iteration_total.labels(agent_id=agent_id, job_id=job_id).set(
            total_iterations
        )

    @staticmethod
    def update_training_metrics(
        agent_id: str,
        job_id: str,
        loss: float | None = None,
        mean_reward: float | None = None,
    ) -> None:
        """
        Update training performance metrics.

        Args:
            agent_id: Agent identifier
            job_id: Training job ID
            loss: Current training loss
            mean_reward: Mean reward across trajectories
        """
        if loss is not None:
            training_loss.labels(agent_id=agent_id, job_id=job_id).set(loss)

        if mean_reward is not None:
            training_reward_mean.labels(agent_id=agent_id, job_id=job_id).set(
                mean_reward
            )
