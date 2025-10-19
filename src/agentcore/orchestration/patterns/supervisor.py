"""
Supervisor Pattern Implementation

Master-worker coordination with task distribution, monitoring, and failure handling.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.streams.models import (
    AgentFailedEvent,
    OrchestrationEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
)
from agentcore.orchestration.streams.producer import StreamProducer


class WorkerStatus(str, Enum):
    """Status of a worker agent."""

    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    OFFLINE = "offline"


class LoadBalancingStrategy(str, Enum):
    """Strategy for distributing tasks to workers."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    PRIORITY_BASED = "priority_based"


class WorkerState(BaseModel):
    """Runtime state of a worker agent."""

    worker_id: str = Field(description="Unique worker identifier")
    status: WorkerStatus = Field(default=WorkerStatus.IDLE)
    current_task_id: UUID | None = Field(default=None)
    tasks_completed: int = Field(default=0)
    tasks_failed: int = Field(default=0)
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    capabilities: list[str] = Field(default_factory=list)
    load_score: float = Field(default=0.0, description="Current load score (0.0-1.0)")

    model_config = {"frozen": False}


class TaskAssignment(BaseModel):
    """Assignment of a task to a worker."""

    assignment_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    worker_id: str
    assigned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)


class SupervisorConfig(BaseModel):
    """Configuration for supervisor pattern."""

    max_workers: int = Field(default=10, ge=1, description="Maximum number of workers")
    load_balancing_strategy: LoadBalancingStrategy = Field(
        default=LoadBalancingStrategy.LEAST_LOADED
    )
    worker_timeout_seconds: int = Field(
        default=30, ge=1, description="Worker heartbeat timeout"
    )
    task_timeout_seconds: int = Field(
        default=300, ge=1, description="Task execution timeout"
    )
    enable_auto_recovery: bool = Field(
        default=True, description="Enable automatic worker failure recovery"
    )
    max_task_retries: int = Field(default=3, ge=0, description="Max task retry attempts")


class SupervisorCoordinator:
    """
    Supervisor pattern coordinator.

    Implements master-worker coordination with:
    - Task distribution and load balancing
    - Worker health monitoring
    - Failure detection and recovery
    - Task retry and reassignment
    """

    def __init__(
        self,
        supervisor_id: str,
        config: SupervisorConfig | None = None,
        event_producer: StreamProducer | None = None,
    ) -> None:
        """
        Initialize supervisor coordinator.

        Args:
            supervisor_id: Unique identifier for this supervisor
            config: Supervisor configuration
            event_producer: Event stream producer for publishing events
        """
        self.supervisor_id = supervisor_id
        self.config = config or SupervisorConfig()
        self.event_producer = event_producer

        # Worker registry
        self._workers: dict[str, WorkerState] = {}

        # Task tracking
        self._pending_tasks: list[UUID] = []
        self._active_assignments: dict[UUID, TaskAssignment] = {}
        self._completed_tasks: set[UUID] = set()
        self._failed_tasks: dict[UUID, str] = {}

        # Round-robin index for load balancing
        self._round_robin_index = 0

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def register_worker(
        self,
        worker_id: str,
        capabilities: list[str] | None = None,
    ) -> None:
        """
        Register a new worker agent.

        Args:
            worker_id: Unique worker identifier
            capabilities: List of worker capabilities
        """
        async with self._lock:
            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                capabilities=capabilities or [],
                status=WorkerStatus.IDLE,
            )

    async def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker agent.

        Args:
            worker_id: Worker identifier to unregister
        """
        async with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = WorkerStatus.OFFLINE

                # Reassign active task if any
                if worker.current_task_id:
                    await self._reassign_task(worker.current_task_id, worker_id)

                del self._workers[worker_id]

    async def submit_task(
        self,
        task_id: UUID,
        task_type: str,
        input_data: dict[str, Any] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> None:
        """
        Submit a task for execution.

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            input_data: Task input data
            required_capabilities: Required worker capabilities
        """
        async with self._lock:
            self._pending_tasks.append(task_id)

        # Publish task created event
        if self.event_producer:
            event = TaskCreatedEvent(
                task_id=task_id,
                task_type=task_type,
                input_data=input_data or {},
                timeout_seconds=self.config.task_timeout_seconds,
                metadata={"required_capabilities": required_capabilities or []},
            )
            await self.event_producer.publish(event.model_dump())

        # Try to assign immediately
        await self._assign_pending_tasks()

    async def handle_task_completion(
        self,
        task_id: UUID,
        worker_id: str,
        result_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Handle task completion from a worker.

        Args:
            task_id: Completed task identifier
            worker_id: Worker that completed the task
            result_data: Task result data
        """
        async with self._lock:
            if task_id in self._active_assignments:
                assignment = self._active_assignments[task_id]
                assignment.completed_at = datetime.now(UTC)

                # Update worker state
                if worker_id in self._workers:
                    worker = self._workers[worker_id]
                    worker.status = WorkerStatus.IDLE
                    worker.current_task_id = None
                    worker.tasks_completed += 1
                    worker.load_score = max(0.0, worker.load_score - 0.1)

                # Mark task as completed
                self._completed_tasks.add(task_id)
                del self._active_assignments[task_id]

        # Publish completion event
        if self.event_producer:
            event = TaskCompletedEvent(
                task_id=task_id,
                agent_id=worker_id,
                result_data=result_data or {},
                execution_time_ms=int(
                    (
                        (assignment.completed_at or datetime.now(UTC))
                        - (assignment.started_at or assignment.assigned_at)
                    ).total_seconds()
                    * 1000
                ),
            )
            await self.event_producer.publish(event.model_dump())

        # Assign next pending task
        await self._assign_pending_tasks()

    async def handle_task_failure(
        self,
        task_id: UUID,
        worker_id: str,
        error_message: str,
        error_type: str,
    ) -> None:
        """
        Handle task failure from a worker.

        Args:
            task_id: Failed task identifier
            worker_id: Worker that failed the task
            error_message: Error message
            error_type: Error type/class
        """
        async with self._lock:
            if task_id not in self._active_assignments:
                return

            assignment = self._active_assignments[task_id]
            assignment.retry_count += 1

            # Update worker state
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.status = WorkerStatus.IDLE
                worker.current_task_id = None
                worker.tasks_failed += 1

            # Check if retry limit reached
            if assignment.retry_count >= assignment.max_retries:
                # Mark task as permanently failed
                self._failed_tasks[task_id] = error_message
                del self._active_assignments[task_id]

                # Publish failure event
                if self.event_producer:
                    event = TaskFailedEvent(
                        task_id=task_id,
                        agent_id=worker_id,
                        error_message=error_message,
                        error_type=error_type,
                        retry_count=assignment.retry_count,
                    )
                    await self.event_producer.publish(event.model_dump())
            else:
                # Retry the task
                await self._reassign_task(task_id, worker_id)

    async def handle_worker_heartbeat(
        self,
        worker_id: str,
        load_score: float | None = None,
    ) -> None:
        """
        Handle heartbeat from a worker.

        Args:
            worker_id: Worker identifier
            load_score: Current load score (0.0-1.0)
        """
        async with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.last_heartbeat = datetime.now(UTC)
                if load_score is not None:
                    worker.load_score = max(0.0, min(1.0, load_score))

    async def monitor_workers(self) -> None:
        """
        Monitor worker health and handle timeouts.

        Should be called periodically to detect failed workers.
        """
        now = datetime.now(UTC)
        timeout_threshold = now.timestamp() - self.config.worker_timeout_seconds

        async with self._lock:
            for worker_id, worker in list(self._workers.items()):
                if worker.last_heartbeat.timestamp() < timeout_threshold:
                    # Worker timeout detected
                    worker.status = WorkerStatus.FAILED

                    # Publish failure event
                    if self.event_producer:
                        event = AgentFailedEvent(
                            agent_id=worker_id,
                            error_message="Worker heartbeat timeout",
                            error_type="TimeoutError",
                        )
                        await self.event_producer.publish(event.model_dump())

                    # Reassign active task if any
                    if worker.current_task_id and self.config.enable_auto_recovery:
                        await self._reassign_task(worker.current_task_id, worker_id)

                    # Remove failed worker
                    del self._workers[worker_id]

    async def _assign_pending_tasks(self) -> None:
        """Assign pending tasks to available workers."""
        async with self._lock:
            while self._pending_tasks and self._has_available_worker():
                task_id = self._pending_tasks.pop(0)

                # Select worker using load balancing strategy
                worker_id = await self._select_worker()

                if worker_id:
                    # Create assignment
                    assignment = TaskAssignment(
                        task_id=task_id,
                        worker_id=worker_id,
                        max_retries=self.config.max_task_retries,
                    )
                    self._active_assignments[task_id] = assignment

                    # Update worker state
                    worker = self._workers[worker_id]
                    worker.status = WorkerStatus.BUSY
                    worker.current_task_id = task_id
                    worker.load_score = min(1.0, worker.load_score + 0.1)

    async def _reassign_task(self, task_id: UUID, failed_worker_id: str) -> None:
        """
        Reassign a task after worker failure.

        Args:
            task_id: Task identifier to reassign
            failed_worker_id: ID of the failed worker
        """
        if task_id in self._active_assignments:
            assignment = self._active_assignments[task_id]
            assignment.retry_count += 1

            if assignment.retry_count < assignment.max_retries:
                # Put task back in pending queue
                self._pending_tasks.insert(0, task_id)
            else:
                # Max retries reached
                self._failed_tasks[task_id] = (
                    f"Max retries reached after worker {failed_worker_id} failure"
                )
                del self._active_assignments[task_id]

    def _has_available_worker(self) -> bool:
        """Check if there are available workers."""
        return any(w.status == WorkerStatus.IDLE for w in self._workers.values())

    async def _select_worker(self) -> str | None:
        """
        Select a worker using the configured load balancing strategy.

        Returns:
            Worker ID or None if no worker available
        """
        idle_workers = [
            w for w in self._workers.values() if w.status == WorkerStatus.IDLE
        ]

        if not idle_workers:
            return None

        if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round-robin selection
            self._round_robin_index = (self._round_robin_index + 1) % len(idle_workers)
            return idle_workers[self._round_robin_index].worker_id

        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select worker with lowest load score
            return min(idle_workers, key=lambda w: w.load_score).worker_id

        elif self.config.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            # Random selection
            import random

            return random.choice(idle_workers).worker_id

        else:
            # Default to round-robin
            return idle_workers[0].worker_id

    async def get_supervisor_status(self) -> dict[str, Any]:
        """
        Get current supervisor status.

        Returns:
            Status dictionary with workers and tasks information
        """
        async with self._lock:
            return {
                "supervisor_id": self.supervisor_id,
                "workers": {
                    "total": len(self._workers),
                    "idle": sum(
                        1 for w in self._workers.values() if w.status == WorkerStatus.IDLE
                    ),
                    "busy": sum(
                        1 for w in self._workers.values() if w.status == WorkerStatus.BUSY
                    ),
                    "failed": sum(
                        1
                        for w in self._workers.values()
                        if w.status == WorkerStatus.FAILED
                    ),
                },
                "tasks": {
                    "pending": len(self._pending_tasks),
                    "active": len(self._active_assignments),
                    "completed": len(self._completed_tasks),
                    "failed": len(self._failed_tasks),
                },
                "config": self.config.model_dump(),
            }

    async def get_worker_states(self) -> list[WorkerState]:
        """
        Get states of all registered workers.

        Returns:
            List of worker states
        """
        async with self._lock:
            return list(self._workers.values())

    async def get_task_assignments(self) -> list[TaskAssignment]:
        """
        Get all active task assignments.

        Returns:
            List of active assignments
        """
        async with self._lock:
            return list(self._active_assignments.values())
