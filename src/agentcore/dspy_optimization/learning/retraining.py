"""
Automatic retraining triggers

Manages automatic retraining based on drift detection, schedules,
and performance thresholds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import OptimizationTarget


class TriggerCondition(str, Enum):
    """Retraining trigger conditions"""

    DRIFT_DETECTED = "drift_detected"
    SCHEDULED = "scheduled"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    MANUAL = "manual"
    SAMPLE_COUNT = "sample_count"


class RetrainingStatus(str, Enum):
    """Status of retraining job"""

    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetrainingConfig(BaseModel):
    """Configuration for automatic retraining"""

    enable_drift_triggers: bool = Field(
        default=True,
        description="Enable drift-based triggers",
    )
    enable_scheduled_triggers: bool = Field(
        default=True,
        description="Enable scheduled triggers",
    )
    schedule_interval_hours: int = Field(
        default=168,
        description="Schedule interval in hours (default: 1 week)",
    )
    performance_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Performance degradation threshold",
    )
    min_samples_for_retraining: int = Field(
        default=1000,
        description="Minimum samples required for retraining",
    )
    max_concurrent_retraining: int = Field(
        default=3,
        description="Maximum concurrent retraining jobs",
    )
    validation_required: bool = Field(
        default=True,
        description="Require validation before deployment",
    )
    min_validation_improvement: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum improvement for validation (5%)",
    )


class RetrainingTrigger(BaseModel):
    """Retraining trigger event"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    condition: TriggerCondition
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrainingJob(BaseModel):
    """Retraining job"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    trigger: RetrainingTrigger
    status: RetrainingStatus = RetrainingStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    samples_used: int = 0
    validation_improvement: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_running(self) -> bool:
        """Check if job is running"""
        return self.status in (RetrainingStatus.RUNNING, RetrainingStatus.VALIDATING)

    def is_completed(self) -> bool:
        """Check if job is completed"""
        return self.status in (
            RetrainingStatus.COMPLETED,
            RetrainingStatus.FAILED,
            RetrainingStatus.CANCELLED,
        )


class RetrainingManager:
    """
    Manages automatic retraining triggers and jobs

    Monitors conditions and triggers retraining when needed,
    managing the full retraining lifecycle.

    Key features:
    - Multiple trigger conditions
    - Scheduled and event-driven retraining
    - Validation before deployment
    - Concurrent job management
    """

    def __init__(self, config: RetrainingConfig | None = None) -> None:
        """
        Initialize retraining manager

        Args:
            config: Retraining configuration
        """
        self.config = config or RetrainingConfig()
        self._triggers: dict[str, list[RetrainingTrigger]] = {}
        self._jobs: dict[str, RetrainingJob] = {}
        self._last_retraining: dict[str, datetime] = {}
        self._sample_counts: dict[str, int] = {}

    async def check_triggers(
        self,
        target: OptimizationTarget,
        drift_detected: bool = False,
        performance_degradation: float = 0.0,
        sample_count: int = 0,
    ) -> RetrainingTrigger | None:
        """
        Check if retraining should be triggered

        Args:
            target: Optimization target
            drift_detected: Whether drift was detected
            performance_degradation: Performance degradation percentage
            sample_count: Current sample count

        Returns:
            Trigger if retraining should start, None otherwise
        """
        target_key = self._get_target_key(target)

        # Update sample count
        self._sample_counts[target_key] = sample_count

        # Check drift trigger
        if self.config.enable_drift_triggers and drift_detected:
            return await self._create_trigger(
                target,
                TriggerCondition.DRIFT_DETECTED,
                {"drift_detected": True},
            )

        # Check performance threshold trigger
        if performance_degradation >= self.config.performance_threshold:
            return await self._create_trigger(
                target,
                TriggerCondition.PERFORMANCE_THRESHOLD,
                {"degradation": performance_degradation},
            )

        # Check scheduled trigger
        if self.config.enable_scheduled_triggers:
            last_retraining = self._last_retraining.get(target_key)

            if not last_retraining:
                # Never retrained, trigger if we have enough samples
                if sample_count >= self.config.min_samples_for_retraining:
                    return await self._create_trigger(
                        target,
                        TriggerCondition.SCHEDULED,
                        {"reason": "initial_training"},
                    )
            else:
                # Check if schedule interval has elapsed
                elapsed = datetime.utcnow() - last_retraining
                schedule_interval = timedelta(hours=self.config.schedule_interval_hours)

                if elapsed >= schedule_interval:
                    return await self._create_trigger(
                        target,
                        TriggerCondition.SCHEDULED,
                        {"elapsed_hours": elapsed.total_seconds() / 3600},
                    )

        # Check sample count trigger
        if sample_count >= self.config.min_samples_for_retraining:
            last_count = self._sample_counts.get(f"{target_key}_last_training", 0)
            new_samples = sample_count - last_count

            # Trigger if we have significant new samples (2x minimum)
            if new_samples >= self.config.min_samples_for_retraining * 2:
                return await self._create_trigger(
                    target,
                    TriggerCondition.SAMPLE_COUNT,
                    {"new_samples": new_samples},
                )

        return None

    async def create_manual_trigger(
        self,
        target: OptimizationTarget,
        metadata: dict[str, Any] | None = None,
    ) -> RetrainingTrigger:
        """
        Create manual retraining trigger

        Args:
            target: Optimization target
            metadata: Optional trigger metadata

        Returns:
            Created trigger
        """
        return await self._create_trigger(
            target,
            TriggerCondition.MANUAL,
            metadata or {},
        )

    async def start_retraining(
        self,
        trigger: RetrainingTrigger,
        sample_count: int,
    ) -> RetrainingJob:
        """
        Start retraining job

        Args:
            trigger: Retraining trigger
            sample_count: Number of samples for training

        Returns:
            Created retraining job

        Raises:
            ValueError: If too many concurrent jobs
        """
        # Check concurrent job limit
        running_jobs = [j for j in self._jobs.values() if j.is_running()]
        if len(running_jobs) >= self.config.max_concurrent_retraining:
            raise ValueError(
                f"Maximum concurrent retraining jobs reached: {self.config.max_concurrent_retraining}"
            )

        # Create job
        job = RetrainingJob(
            target=trigger.target,
            trigger=trigger,
            samples_used=sample_count,
        )

        # Store job
        self._jobs[job.id] = job

        return job

    async def update_job_status(
        self,
        job_id: str,
        status: RetrainingStatus,
        validation_improvement: float = 0.0,
        error_message: str | None = None,
    ) -> RetrainingJob:
        """
        Update retraining job status

        Args:
            job_id: Job ID
            status: New status
            validation_improvement: Validation improvement percentage
            error_message: Optional error message

        Returns:
            Updated job

        Raises:
            ValueError: If job not found
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Retraining job not found: {job_id}")

        job.status = status

        if status == RetrainingStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()

        if status in (RetrainingStatus.COMPLETED, RetrainingStatus.FAILED, RetrainingStatus.CANCELLED):
            job.completed_at = datetime.utcnow()

            if status == RetrainingStatus.COMPLETED:
                # Update last retraining time
                target_key = self._get_target_key(job.target)
                self._last_retraining[target_key] = datetime.utcnow()
                self._sample_counts[f"{target_key}_last_training"] = job.samples_used

        if validation_improvement:
            job.validation_improvement = validation_improvement

        if error_message:
            job.error_message = error_message

        return job

    async def validate_job(
        self,
        job_id: str,
        improvement: float,
    ) -> bool:
        """
        Validate retraining job results

        Args:
            job_id: Job ID
            improvement: Improvement percentage

        Returns:
            True if validation passes
        """
        if not self.config.validation_required:
            return True

        return improvement >= self.config.min_validation_improvement

    async def cancel_job(
        self,
        job_id: str,
    ) -> RetrainingJob:
        """
        Cancel retraining job

        Args:
            job_id: Job ID

        Returns:
            Cancelled job

        Raises:
            ValueError: If job not found or cannot be cancelled
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Retraining job not found: {job_id}")

        if job.is_completed():
            raise ValueError(f"Cannot cancel completed job: {job_id}")

        job.status = RetrainingStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        return job

    async def get_job(
        self,
        job_id: str,
    ) -> RetrainingJob | None:
        """
        Get retraining job by ID

        Args:
            job_id: Job ID

        Returns:
            Retraining job or None
        """
        return self._jobs.get(job_id)

    async def list_jobs(
        self,
        target: OptimizationTarget | None = None,
        status: RetrainingStatus | None = None,
    ) -> list[RetrainingJob]:
        """
        List retraining jobs

        Args:
            target: Optional target filter
            status: Optional status filter

        Returns:
            List of retraining jobs
        """
        jobs = list(self._jobs.values())

        if target:
            target_key = self._get_target_key(target)
            jobs = [
                j for j in jobs if self._get_target_key(j.target) == target_key
            ]

        if status:
            jobs = [j for j in jobs if j.status == status]

        return jobs

    async def get_trigger_history(
        self,
        target: OptimizationTarget,
        limit: int | None = None,
    ) -> list[RetrainingTrigger]:
        """
        Get trigger history for target

        Args:
            target: Optimization target
            limit: Optional result limit

        Returns:
            List of triggers
        """
        target_key = self._get_target_key(target)
        triggers = self._triggers.get(target_key, [])

        # Sort by timestamp (newest first)
        triggers.sort(key=lambda t: t.timestamp, reverse=True)

        if limit:
            return triggers[:limit]

        return triggers

    async def _create_trigger(
        self,
        target: OptimizationTarget,
        condition: TriggerCondition,
        metadata: dict[str, Any],
    ) -> RetrainingTrigger:
        """
        Create retraining trigger

        Args:
            target: Optimization target
            condition: Trigger condition
            metadata: Trigger metadata

        Returns:
            Created trigger
        """
        trigger = RetrainingTrigger(
            target=target,
            condition=condition,
            metadata=metadata,
        )

        # Store trigger
        target_key = self._get_target_key(target)
        if target_key not in self._triggers:
            self._triggers[target_key] = []

        self._triggers[target_key].append(trigger)

        return trigger

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """
        Get target storage key

        Args:
            target: Optimization target

        Returns:
            Target key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
