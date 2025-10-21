"""
Workflow Hooks Models

Event-driven hooks for automated workflow enhancement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class HookTrigger(str, Enum):
    """Hook trigger types for workflow automation."""

    # Pre-operation hooks
    PRE_TASK = "pre_task"
    PRE_SEARCH = "pre_search"
    PRE_EDIT = "pre_edit"
    PRE_COMMAND = "pre_command"

    # Post-operation hooks
    POST_EDIT = "post_edit"
    POST_TASK = "post_task"
    POST_COMMAND = "post_command"
    NOTIFICATION = "notification"

    # Session hooks
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_RESTORE = "session_restore"


class HookStatus(str, Enum):
    """Hook execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class HookExecutionMode(str, Enum):
    """Hook execution mode."""

    SYNC = "sync"  # Blocking execution
    ASYNC = "async"  # Non-blocking via queue
    FIRE_AND_FORGET = "fire_and_forget"  # No result tracking


class HookConfig(BaseModel):
    """
    Hook configuration.

    Defines a workflow hook for automated enhancement.
    """

    hook_id: UUID = Field(default_factory=uuid4, description="Unique hook identifier")
    name: str = Field(description="Hook name")
    trigger: HookTrigger = Field(description="Event trigger type")

    # Execution
    command: str = Field(
        description="Shell command or Python function reference (module:function)"
    )
    args: list[str] = Field(default_factory=list, description="Command arguments")
    always_run: bool = Field(
        default=False, description="Run even if previous hooks fail"
    )
    timeout_ms: int = Field(default=30000, description="Execution timeout (ms)")

    # Control
    enabled: bool = Field(default=True, description="Hook enabled status")
    priority: int = Field(
        default=100, description="Execution priority (lower runs first)"
    )
    execution_mode: HookExecutionMode = Field(
        default=HookExecutionMode.ASYNC, description="Execution mode"
    )

    # Filtering
    event_filters: dict[str, Any] = Field(
        default_factory=dict, description="Event data filters"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Hook metadata")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Update timestamp"
    )

    # Retry configuration
    retry_enabled: bool = Field(default=True, description="Enable retries on failure")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_ms: int = Field(
        default=1000, description="Initial retry delay (ms), doubles each retry"
    )

    model_config = {"frozen": False}

    def matches_event(self, event_data: dict[str, Any]) -> bool:
        """
        Check if hook matches event data.

        Args:
            event_data: Event data to match against filters

        Returns:
            True if event matches all filters
        """
        if not self.event_filters:
            return True

        for key, expected_value in self.event_filters.items():
            actual_value = self._get_nested_value(event_data, key)
            if actual_value != expected_value:
                return False

        return True

    @staticmethod
    def _get_nested_value(data: dict[str, Any], key: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = key.split(".")
        value = data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value


class HookExecution(BaseModel):
    """Hook execution record."""

    execution_id: UUID = Field(
        default_factory=uuid4, description="Unique execution identifier"
    )
    hook_id: UUID = Field(description="Hook that was executed")
    trigger: HookTrigger = Field(description="Trigger that fired the hook")

    # Execution details
    status: HookStatus = Field(description="Execution status")
    started_at: datetime = Field(description="Execution start time")
    completed_at: datetime | None = Field(None, description="Execution completion time")
    duration_ms: int | None = Field(None, description="Execution duration (ms)")

    # Input/Output
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Hook input data"
    )
    output_data: dict[str, Any] | None = Field(None, description="Hook output data")
    error_message: str | None = Field(None, description="Error message if failed")
    error_traceback: str | None = Field(None, description="Error traceback if failed")

    # Retry tracking
    retry_count: int = Field(default=0, description="Number of retry attempts")
    is_retry: bool = Field(default=False, description="Is this a retry attempt")

    # Context
    workflow_id: str | None = Field(None, description="Associated workflow ID")
    task_id: str | None = Field(None, description="Associated task ID")
    session_id: str | None = Field(None, description="Associated session ID")

    model_config = {"frozen": False}

    def mark_completed(self, output: dict[str, Any] | None = None) -> None:
        """Mark execution as completed."""
        self.status = HookStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.output_data = output
        if self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds() * 1000
            self.duration_ms = int(duration)

    def mark_failed(self, error: str, traceback: str | None = None) -> None:
        """Mark execution as failed."""
        self.status = HookStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error_message = error
        self.error_traceback = traceback
        if self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds() * 1000
            self.duration_ms = int(duration)

    def mark_timeout(self) -> None:
        """Mark execution as timed out."""
        self.status = HookStatus.TIMEOUT
        self.completed_at = datetime.now(UTC)
        self.error_message = f"Hook execution exceeded timeout of {self.duration_ms}ms"
        if self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds() * 1000
            self.duration_ms = int(duration)


class HookRegistrationRequest(BaseModel):
    """Hook registration request."""

    name: str = Field(description="Hook name")
    trigger: HookTrigger = Field(description="Event trigger type")
    command: str = Field(description="Command to execute")
    args: list[str] | None = Field(None, description="Command arguments")
    always_run: bool = Field(default=False, description="Always run flag")
    timeout_ms: int = Field(default=30000, description="Timeout in milliseconds")
    priority: int = Field(default=100, description="Execution priority")
    event_filters: dict[str, Any] | None = Field(None, description="Event filters")
    metadata: dict[str, Any] | None = Field(None, description="Hook metadata")


class HookRegistrationResponse(BaseModel):
    """Hook registration response."""

    success: bool = Field(description="Registration success status")
    hook_id: str = Field(description="Registered hook ID")
    message: str | None = Field(None, description="Response message")


class HookExecutionResult(BaseModel):
    """Result of hook execution."""

    execution_id: str = Field(description="Execution ID")
    hook_id: str = Field(description="Hook ID")
    status: HookStatus = Field(description="Execution status")
    duration_ms: int | None = Field(None, description="Execution duration")
    output: dict[str, Any] | None = Field(None, description="Execution output")
    error: str | None = Field(None, description="Error message if failed")


class HookEvent(BaseModel):
    """Event data for hook triggering."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Event identifier"
    )
    trigger: HookTrigger = Field(description="Hook trigger type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )
    source: str = Field(description="Event source")
    data: dict[str, Any] = Field(description="Event data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Event metadata")

    # Context
    workflow_id: str | None = Field(None, description="Workflow context")
    task_id: str | None = Field(None, description="Task context")
    session_id: str | None = Field(None, description="Session context")
