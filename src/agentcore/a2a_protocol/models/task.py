"""
Task Management Models

A2A Protocol v0.2 compliant task definition, execution, and lifecycle management models.
Implements task state transitions, dependency tracking, and agent assignment.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    """Task execution status states."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyType(str, Enum):
    """Task dependency relationship types."""

    PREDECESSOR = "predecessor"  # Must complete before this task can start
    SUCCESSOR = "successor"  # Will start after this task completes
    PARALLEL = "parallel"  # Can run concurrently
    CONDITIONAL = "conditional"  # Conditional dependency based on outcome


class TaskDependency(BaseModel):
    """Task dependency definition."""

    task_id: str = Field(..., description="Dependent task ID")
    type: DependencyType = Field(..., description="Dependency type")
    condition: dict[str, Any] | None = Field(
        None, description="Optional condition for dependency"
    )


class TaskArtifact(BaseModel):
    """Task execution artifact."""

    name: str = Field(..., description="Artifact name")
    type: str = Field(..., description="Artifact type (file, data, url, etc.)")
    content: Any = Field(..., description="Artifact content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Artifact metadata"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )

    # A2A-018: Context engineering fields
    context_lineage: list[str] | None = Field(
        None,
        description="Lineage of context transformations (task_ids that contributed)",
    )
    context_summary: str | None = Field(
        None, description="Summary of accumulated context for this artifact"
    )

    @field_validator("type")
    @classmethod
    def validate_artifact_type(cls, v: str) -> str:
        """Validate artifact type."""
        valid_types = [
            "file",
            "data",
            "url",
            "json",
            "text",
            "binary",
            "image",
            "document",
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid artifact type. Must be one of: {valid_types}")
        return v


class TaskRequirement(BaseModel):
    """Task execution requirements."""

    required_capabilities: list[str] = Field(
        default_factory=list, description="Required agent capabilities"
    )
    preferred_agents: list[str] = Field(
        default_factory=list, description="Preferred agent IDs"
    )
    excluded_agents: list[str] = Field(
        default_factory=list, description="Excluded agent IDs"
    )
    max_execution_time: int | None = Field(
        None, description="Maximum execution time in seconds"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    require_authentication: bool = Field(
        default=False, description="Whether authentication is required"
    )


class TaskDefinition(BaseModel):
    """
    A2A Protocol task definition.

    Defines what needs to be executed, requirements, and dependencies.
    """

    task_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique task identifier"
    )
    task_type: str = Field(..., description="Type of task to execute")
    title: str = Field(..., description="Human-readable task title")
    description: str | None = Field(None, description="Task description")

    # Task data and parameters
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for task execution"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Task-specific parameters"
    )

    # Requirements and constraints
    requirements: TaskRequirement = Field(
        default_factory=TaskRequirement, description="Task requirements"
    )
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL, description="Task priority"
    )

    # Dependencies
    dependencies: list[TaskDependency] = Field(
        default_factory=list, description="Task dependencies"
    )

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Task tags")
    created_by: str | None = Field(None, description="Creator agent/user ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task type format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Task type cannot be empty")
        # Task types should follow a namespace pattern like "text.generation" or "data.analysis"
        if "." not in v:
            raise ValueError(
                "Task type should follow namespace.action format (e.g., 'text.generation')"
            )
        return v.strip()

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate task title."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Task title cannot be empty")
        if len(v) > 200:
            raise ValueError("Task title cannot exceed 200 characters")
        return v.strip()

    def has_dependency(
        self, task_id: str, dependency_type: DependencyType | None = None
    ) -> bool:
        """Check if task has a specific dependency."""
        for dep in self.dependencies:
            if dep.task_id == task_id:
                if dependency_type is None or dep.type == dependency_type:
                    return True
        return False

    def get_predecessor_tasks(self) -> list[str]:
        """Get list of predecessor task IDs."""
        return [
            dep.task_id
            for dep in self.dependencies
            if dep.type == DependencyType.PREDECESSOR
        ]


class TaskExecution(BaseModel):
    """
    Task execution instance with state and progress tracking.

    Represents the runtime execution of a TaskDefinition.
    """

    execution_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique execution identifier"
    )
    task_definition: TaskDefinition = Field(
        ..., description="Associated task definition"
    )

    # Execution state
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current execution status"
    )
    assigned_agent: str | None = Field(None, description="Assigned agent ID")

    # Timing
    assigned_at: datetime | None = Field(None, description="Assignment timestamp")
    started_at: datetime | None = Field(None, description="Execution start timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")

    # Progress tracking
    progress_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Execution progress (0-100)"
    )
    current_step: str | None = Field(None, description="Current execution step")

    # Results and artifacts
    result_data: dict[str, Any] = Field(
        default_factory=dict, description="Execution result data"
    )
    artifacts: list[TaskArtifact] = Field(
        default_factory=list, description="Generated artifacts"
    )

    # Error handling
    error_message: str | None = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retry attempts made")

    # Metadata
    execution_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution-specific metadata"
    )

    @property
    def task_id(self) -> str:
        """Get the task ID from the definition."""
        return self.task_definition.task_id

    @property
    def task_type(self) -> str:
        """Get the task type from the definition."""
        return self.task_definition.task_type

    @property
    def is_pending(self) -> bool:
        """Check if task is pending."""
        return self.status == TaskStatus.PENDING

    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status in [
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
        ]

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        ]

    @property
    def execution_duration(self) -> float | None:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def can_transition_to(self, new_status: TaskStatus) -> bool:
        """Check if task can transition to a new status."""
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.ASSIGNED, TaskStatus.CANCELLED],
            TaskStatus.ASSIGNED: [TaskStatus.RUNNING, TaskStatus.CANCELLED],
            TaskStatus.RUNNING: [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            ],
            TaskStatus.COMPLETED: [],  # Terminal state
            TaskStatus.FAILED: [TaskStatus.PENDING]
            if self.retry_count < self.task_definition.requirements.retry_count
            else [],
            TaskStatus.CANCELLED: [],  # Terminal state
            TaskStatus.TIMEOUT: [TaskStatus.PENDING]
            if self.retry_count < self.task_definition.requirements.retry_count
            else [],
        }
        return new_status in valid_transitions.get(self.status, [])

    def assign_to_agent(self, agent_id: str) -> None:
        """Assign task to an agent."""
        if not self.can_transition_to(TaskStatus.ASSIGNED):
            raise ValueError(f"Cannot assign task in {self.status} status")

        self.assigned_agent = agent_id
        self.assigned_at = datetime.now(UTC)
        self.status = TaskStatus.ASSIGNED

    def start_execution(self) -> None:
        """Start task execution."""
        if not self.can_transition_to(TaskStatus.RUNNING):
            raise ValueError(f"Cannot start task in {self.status} status")

        self.started_at = datetime.now(UTC)
        self.status = TaskStatus.RUNNING
        self.progress_percentage = 0.0

    def complete_execution(
        self,
        result_data: dict[str, Any],
        artifacts: list[TaskArtifact] | None = None,
    ) -> None:
        """Complete task execution successfully."""
        if not self.can_transition_to(TaskStatus.COMPLETED):
            raise ValueError(f"Cannot complete task in {self.status} status")

        self.completed_at = datetime.now(UTC)
        self.status = TaskStatus.COMPLETED
        self.progress_percentage = 100.0
        self.result_data = result_data
        if artifacts:
            self.artifacts.extend(artifacts)

    def fail_execution(self, error_message: str, should_retry: bool = True) -> None:
        """Mark task execution as failed."""
        if not self.can_transition_to(TaskStatus.FAILED):
            raise ValueError(f"Cannot fail task in {self.status} status")

        self.completed_at = datetime.now(UTC)
        self.error_message = error_message

        # Check if we should retry
        if (
            should_retry
            and self.retry_count < self.task_definition.requirements.retry_count
        ):
            self.status = TaskStatus.PENDING
            self.retry_count += 1
            self.assigned_agent = None
            self.assigned_at = None
            self.started_at = None
            self.completed_at = None
            self.progress_percentage = 0.0
        else:
            self.status = TaskStatus.FAILED

    def cancel_execution(self) -> None:
        """Cancel task execution."""
        if not self.can_transition_to(TaskStatus.CANCELLED):
            raise ValueError(f"Cannot cancel task in {self.status} status")

        self.completed_at = datetime.now(UTC)
        self.status = TaskStatus.CANCELLED

    def update_progress(
        self, percentage: float, current_step: str | None = None
    ) -> None:
        """Update task execution progress."""
        if not self.is_running:
            raise ValueError("Can only update progress for running tasks")

        self.progress_percentage = max(0.0, min(100.0, percentage))
        if current_step:
            self.current_step = current_step

    def add_artifact(
        self,
        name: str,
        artifact_type: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an execution artifact."""
        artifact = TaskArtifact(
            name=name, type=artifact_type, content=content, metadata=metadata or {}
        )
        self.artifacts.append(artifact)

    def to_summary(self) -> dict[str, Any]:
        """Create a task execution summary."""
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "title": self.task_definition.title,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "assigned_agent": self.assigned_agent,
            "created_at": self.task_definition.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "execution_duration": self.execution_duration,
            "retry_count": self.retry_count,
            "artifacts_count": len(self.artifacts),
            "error_message": self.error_message,
            "priority": self.task_definition.priority.value,
            "tags": self.task_definition.tags,
        }


class TaskCreateRequest(BaseModel):
    """Request to create a new task."""

    task_definition: TaskDefinition = Field(..., description="Task definition")
    auto_assign: bool = Field(
        default=True, description="Automatically assign to capable agent"
    )
    preferred_agent: str | None = Field(
        None, description="Preferred agent ID for assignment"
    )


class TaskCreateResponse(BaseModel):
    """Response for task creation."""

    execution_id: str = Field(..., description="Task execution ID")
    task_id: str = Field(..., description="Task definition ID")
    status: str = Field(..., description="Initial task status")
    assigned_agent: str | None = Field(None, description="Assigned agent ID")
    message: str = Field(..., description="Creation status message")


class TaskQuery(BaseModel):
    """Task query parameters for filtering and search."""

    status: TaskStatus | None = Field(None, description="Filter by task status")
    task_type: str | None = Field(None, description="Filter by task type")
    assigned_agent: str | None = Field(None, description="Filter by assigned agent")
    created_by: str | None = Field(None, description="Filter by creator")
    tags: list[str] | None = Field(None, description="Filter by tags (all must match)")
    priority: TaskPriority | None = Field(None, description="Filter by priority")
    created_after: datetime | None = Field(None, description="Filter by creation time")
    created_before: datetime | None = Field(None, description="Filter by creation time")
    limit: int = Field(
        default=50, ge=1, le=1000, description="Maximum number of results"
    )
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")


class TaskQueryResponse(BaseModel):
    """Response for task queries."""

    tasks: list[dict[str, Any]] = Field(..., description="Task execution summaries")
    total_count: int = Field(..., description="Total number of matching tasks")
    has_more: bool = Field(..., description="Whether more results are available")
    query: TaskQuery = Field(..., description="Original query parameters")
