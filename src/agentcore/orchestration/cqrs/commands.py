"""
CQRS Commands Module

Command models and handlers for write operations.
Commands represent intent to change state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CommandType(str, Enum):
    """Types of commands."""

    # Workflow commands
    CREATE_WORKFLOW = "create_workflow"
    UPDATE_WORKFLOW = "update_workflow"
    DELETE_WORKFLOW = "delete_workflow"
    START_WORKFLOW = "start_workflow"
    PAUSE_WORKFLOW = "pause_workflow"
    RESUME_WORKFLOW = "resume_workflow"
    CANCEL_WORKFLOW = "cancel_workflow"

    # Agent commands
    ASSIGN_AGENT = "assign_agent"
    UNASSIGN_AGENT = "unassign_agent"
    REPLACE_AGENT = "replace_agent"

    # Task commands
    SCHEDULE_TASK = "schedule_task"
    RETRY_TASK = "retry_task"
    CANCEL_TASK = "cancel_task"

    # Handoff commands
    INITIATE_HANDOFF = "initiate_handoff"
    EXECUTE_HANDOFF = "execute_handoff"
    ROLLBACK_HANDOFF = "rollback_handoff"


class Command(BaseModel):
    """
    Base command for CQRS write operations.

    Commands are requests to change state.
    """

    command_id: UUID = Field(
        default_factory=uuid4, description="Unique command identifier"
    )
    command_type: CommandType = Field(description="Type of command")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Command timestamp"
    )
    user_id: str | None = Field(default=None, description="User issuing command")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional command metadata"
    )


class CommandResult(BaseModel):
    """Result of command execution."""

    command_id: UUID = Field(description="Command identifier")
    success: bool = Field(description="Whether command succeeded")
    aggregate_id: UUID | None = Field(
        default=None, description="ID of affected aggregate"
    )
    events_produced: list[UUID] = Field(
        default_factory=list, description="IDs of events produced"
    )
    error_message: str | None = Field(default=None, description="Error message if failed")
    error_type: str | None = Field(default=None, description="Error type if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class CreateWorkflowCommand(Command):
    """Command to create a new workflow."""

    command_type: CommandType = Field(default=CommandType.CREATE_WORKFLOW, frozen=True)
    workflow_name: str = Field(description="Workflow name")
    orchestration_pattern: str = Field(description="Orchestration pattern type")
    agent_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Required agent capabilities"
    )
    task_definitions: list[dict[str, Any]] = Field(
        default_factory=list, description="Task definitions"
    )
    workflow_config: dict[str, Any] = Field(
        default_factory=dict, description="Workflow configuration"
    )


class UpdateWorkflowCommand(Command):
    """Command to update an existing workflow."""

    command_type: CommandType = Field(default=CommandType.UPDATE_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    changes: dict[str, Any] = Field(description="Fields to update with new values")


class DeleteWorkflowCommand(Command):
    """Command to delete a workflow."""

    command_type: CommandType = Field(default=CommandType.DELETE_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    force: bool = Field(
        default=False, description="Force deletion even if workflow is running"
    )


class StartWorkflowCommand(Command):
    """Command to start workflow execution."""

    command_type: CommandType = Field(default=CommandType.START_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Workflow input data"
    )
    execution_options: dict[str, Any] = Field(
        default_factory=dict, description="Execution options (timeout, retries, etc)"
    )


class PauseWorkflowCommand(Command):
    """Command to pause workflow execution."""

    command_type: CommandType = Field(default=CommandType.PAUSE_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    reason: str | None = Field(default=None, description="Reason for pausing")


class ResumeWorkflowCommand(Command):
    """Command to resume paused workflow execution."""

    command_type: CommandType = Field(default=CommandType.RESUME_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")


class CancelWorkflowCommand(Command):
    """Command to cancel workflow execution."""

    command_type: CommandType = Field(default=CommandType.CANCEL_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    reason: str | None = Field(default=None, description="Cancellation reason")


class AssignAgentCommand(Command):
    """Command to assign an agent to a workflow."""

    command_type: CommandType = Field(default=CommandType.ASSIGN_AGENT, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    agent_id: str = Field(description="Agent identifier")
    agent_role: str = Field(description="Role to assign to agent")
    capabilities: list[str] = Field(
        default_factory=list, description="Agent capabilities"
    )


class UnassignAgentCommand(Command):
    """Command to unassign an agent from a workflow."""

    command_type: CommandType = Field(default=CommandType.UNASSIGN_AGENT, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    agent_id: str = Field(description="Agent identifier")
    agent_role: str = Field(description="Role of agent to unassign")
    reason: str | None = Field(default=None, description="Unassignment reason")


class ReplaceAgentCommand(Command):
    """Command to replace an agent in a workflow."""

    command_type: CommandType = Field(default=CommandType.REPLACE_AGENT, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    old_agent_id: str = Field(description="Agent to replace")
    new_agent_id: str = Field(description="Replacement agent")
    agent_role: str = Field(description="Role of agent")
    reason: str = Field(description="Replacement reason")


class ScheduleTaskCommand(Command):
    """Command to schedule a task for execution."""

    command_type: CommandType = Field(default=CommandType.SCHEDULE_TASK, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_type: str = Field(description="Type of task")
    agent_id: str = Field(description="Agent to execute task")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Task input data"
    )
    dependencies: list[UUID] = Field(
        default_factory=list, description="Task dependencies"
    )
    timeout_seconds: int = Field(default=300, description="Task timeout")


class RetryTaskCommand(Command):
    """Command to retry a failed task."""

    command_type: CommandType = Field(default=CommandType.RETRY_TASK, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier to retry")
    reason: str = Field(description="Reason for retry")


class CancelTaskCommand(Command):
    """Command to cancel a task."""

    command_type: CommandType = Field(default=CommandType.CANCEL_TASK, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier to cancel")
    reason: str | None = Field(default=None, description="Cancellation reason")


class CommandHandler(ABC):
    """
    Abstract base class for command handlers.

    Each command handler processes one type of command.
    """

    @abstractmethod
    async def handle(self, command: Command) -> CommandResult:
        """
        Handle a command and return result.

        Args:
            command: Command to handle

        Returns:
            Command execution result
        """
        pass

    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """
        Check if this handler can handle the given command.

        Args:
            command: Command to check

        Returns:
            True if handler can handle this command
        """
        pass


CommandHandlerType = Callable[[Command], CommandResult]


class InitiateHandoffCommand(Command):
    """Command to initiate a task handoff."""

    command_type: CommandType = Field(default=CommandType.INITIATE_HANDOFF, frozen=True)
    task_id: UUID = Field(description="Task identifier")
    task_type: str = Field(description="Type of task")
    source_agent_id: str = Field(description="Source agent identifier")
    target_agent_id: str = Field(description="Target agent identifier")
    task_data: dict[str, Any] = Field(
        default_factory=dict, description="Task data to transfer"
    )
    handoff_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Handoff metadata"
    )


class ExecuteHandoffCommand(Command):
    """Command to execute a pending handoff."""

    command_type: CommandType = Field(default=CommandType.EXECUTE_HANDOFF, frozen=True)
    handoff_id: UUID = Field(description="Handoff identifier to execute")


class RollbackHandoffCommand(Command):
    """Command to rollback a handoff."""

    command_type: CommandType = Field(default=CommandType.ROLLBACK_HANDOFF, frozen=True)
    handoff_id: UUID = Field(description="Handoff identifier to rollback")
    reason: str = Field(description="Reason for rollback")


class CommandBus:
    """
    Command bus for routing commands to handlers.

    Implements command dispatching with registration and routing.
    """

    def __init__(self) -> None:
        """Initialize command bus."""
        self._handlers: dict[CommandType, CommandHandler] = {}

    def register(
        self, command_type: CommandType, handler: CommandHandler
    ) -> None:
        """
        Register a command handler.

        Args:
            command_type: Type of command to handle
            handler: Handler instance

        Raises:
            ValueError: If handler already registered for command type
        """
        if command_type in self._handlers:
            raise ValueError(f"Handler already registered for {command_type}")

        self._handlers[command_type] = handler

    def unregister(self, command_type: CommandType) -> None:
        """
        Unregister a command handler.

        Args:
            command_type: Type of command to unregister
        """
        if command_type in self._handlers:
            del self._handlers[command_type]

    async def dispatch(self, command: Command) -> CommandResult:
        """
        Dispatch command to appropriate handler.

        Args:
            command: Command to dispatch

        Returns:
            Command execution result

        Raises:
            ValueError: If no handler registered for command type
        """
        handler = self._handlers.get(command.command_type)
        if not handler:
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error_message=f"No handler registered for command type: {command.command_type}",
                error_type="UnhandledCommandError",
            )

        try:
            result = await handler.handle(command)
            return result
        except Exception as e:
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
            )

    def get_registered_handlers(self) -> dict[CommandType, CommandHandler]:
        """
        Get all registered handlers.

        Returns:
            Dictionary mapping command types to handlers
        """
        return self._handlers.copy()
