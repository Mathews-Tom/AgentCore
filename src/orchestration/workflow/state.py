"""Workflow state management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orchestration.workflow.models import (
    NodeExecutionState,
    NodeStatus,
    WorkflowExecution,
    WorkflowStatus,
)


class WorkflowStateManager:
    """Manages workflow execution state."""

    def __init__(self) -> None:
        """Initialize the state manager."""
        self._executions: dict[UUID, WorkflowExecution] = {}

    def create_execution(
        self,
        workflow_id: UUID,
        input_data: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Create a new workflow execution.

        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow

        Returns:
            New WorkflowExecution instance
        """
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            status=WorkflowStatus.PLANNING,
            input_data=input_data or {},
            started_at=datetime.now(UTC),
        )
        self._executions[execution.execution_id] = execution
        return execution

    def get_execution(self, execution_id: UUID) -> WorkflowExecution | None:
        """Get execution by ID.

        Args:
            execution_id: ID of the execution

        Returns:
            WorkflowExecution if found, None otherwise
        """
        return self._executions.get(execution_id)

    def update_execution_status(
        self,
        execution_id: UUID,
        status: WorkflowStatus,
    ) -> None:
        """Update the status of a workflow execution.

        Args:
            execution_id: ID of the execution
            status: New status
        """
        execution = self._executions.get(execution_id)
        if execution:
            execution.status = status
            if status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED):
                execution.completed_at = datetime.now(UTC)
            elif status == WorkflowStatus.PAUSED:
                execution.paused_at = datetime.now(UTC)

    def initialize_node_states(
        self,
        execution_id: UUID,
        node_ids: list[str],
    ) -> None:
        """Initialize state for all nodes in the workflow.

        Args:
            execution_id: ID of the execution
            node_ids: List of node IDs to initialize
        """
        execution = self._executions.get(execution_id)
        if execution:
            for node_id in node_ids:
                execution.node_states[node_id] = NodeExecutionState(
                    node_id=node_id, status=NodeStatus.PENDING
                )

    def update_node_status(
        self,
        execution_id: UUID,
        node_id: str,
        status: NodeStatus,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Update the status of a node.

        Args:
            execution_id: ID of the execution
            node_id: ID of the node
            status: New status
            error_message: Error message if failed
            result: Result data if completed
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return

        node_state = execution.node_states.get(node_id)
        if not node_state:
            node_state = NodeExecutionState(node_id=node_id)
            execution.node_states[node_id] = node_state

        node_state.status = status

        if status == NodeStatus.RUNNING:
            node_state.started_at = datetime.now(UTC)
        elif status in (NodeStatus.COMPLETED, NodeStatus.FAILED):
            node_state.completed_at = datetime.now(UTC)

        if error_message:
            node_state.error_message = error_message
        if result:
            node_state.result = result

    def increment_node_attempts(
        self,
        execution_id: UUID,
        node_id: str,
    ) -> int:
        """Increment the attempt count for a node.

        Args:
            execution_id: ID of the execution
            node_id: ID of the node

        Returns:
            New attempt count
        """
        execution = self._executions.get(execution_id)
        if execution and node_id in execution.node_states:
            node_state = execution.node_states[node_id]
            node_state.attempt_count += 1
            return node_state.attempt_count
        return 0

    def get_node_state(
        self,
        execution_id: UUID,
        node_id: str,
    ) -> NodeExecutionState | None:
        """Get the state of a specific node.

        Args:
            execution_id: ID of the execution
            node_id: ID of the node

        Returns:
            NodeExecutionState if found, None otherwise
        """
        execution = self._executions.get(execution_id)
        if execution:
            return execution.node_states.get(node_id)
        return None

    def get_completed_nodes(self, execution_id: UUID) -> set[str]:
        """Get all completed node IDs.

        Args:
            execution_id: ID of the execution

        Returns:
            Set of completed node IDs
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return set()

        return {
            node_id
            for node_id, state in execution.node_states.items()
            if state.status == NodeStatus.COMPLETED
        }

    def get_failed_nodes(self, execution_id: UUID) -> set[str]:
        """Get all failed node IDs.

        Args:
            execution_id: ID of the execution

        Returns:
            Set of failed node IDs
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return set()

        return {
            node_id
            for node_id, state in execution.node_states.items()
            if state.status == NodeStatus.FAILED
        }

    def get_running_nodes(self, execution_id: UUID) -> set[str]:
        """Get all running node IDs.

        Args:
            execution_id: ID of the execution

        Returns:
            Set of running node IDs
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return set()

        return {
            node_id
            for node_id, state in execution.node_states.items()
            if state.status == NodeStatus.RUNNING
        }

    def allocate_agent(
        self,
        execution_id: UUID,
        node_id: str,
        agent_id: str,
    ) -> None:
        """Allocate an agent to a node.

        Args:
            execution_id: ID of the execution
            node_id: ID of the node
            agent_id: ID of the agent to allocate
        """
        execution = self._executions.get(execution_id)
        if execution:
            execution.allocated_agents[node_id] = agent_id

    def update_coordination_overhead(
        self,
        execution_id: UUID,
        overhead_ms: float,
    ) -> None:
        """Update coordination overhead metric.

        Args:
            execution_id: ID of the execution
            overhead_ms: Coordination overhead in milliseconds
        """
        execution = self._executions.get(execution_id)
        if execution:
            execution.coordination_overhead_ms += overhead_ms

    def set_output_data(
        self,
        execution_id: UUID,
        output_data: dict[str, Any],
    ) -> None:
        """Set output data for the workflow execution.

        Args:
            execution_id: ID of the execution
            output_data: Output data to set
        """
        execution = self._executions.get(execution_id)
        if execution:
            execution.output_data = output_data

    def checkpoint(self, execution_id: UUID) -> dict[str, Any]:
        """Create a checkpoint of the execution state.

        Args:
            execution_id: ID of the execution

        Returns:
            Serialized execution state
        """
        execution = self._executions.get(execution_id)
        if execution:
            return execution.model_dump()
        return {}

    def restore(self, checkpoint_data: dict[str, Any]) -> UUID | None:
        """Restore execution from checkpoint.

        Args:
            checkpoint_data: Serialized execution state

        Returns:
            Execution ID if successful, None otherwise
        """
        try:
            execution = WorkflowExecution(**checkpoint_data)
            self._executions[execution.execution_id] = execution
            return execution.execution_id
        except Exception:
            return None
