"""Tests for workflow state management."""

from __future__ import annotations

from uuid import uuid4

import pytest

from orchestration.workflow.models import NodeStatus, WorkflowStatus
from orchestration.workflow.state import WorkflowStateManager


def test_create_execution() -> None:
    """Test creating a workflow execution."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)

    assert execution.workflow_id == workflow_id
    assert execution.status == WorkflowStatus.PLANNING


def test_get_execution() -> None:
    """Test getting execution by ID."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    retrieved = manager.get_execution(execution.execution_id)

    assert retrieved is not None
    assert retrieved.execution_id == execution.execution_id


def test_update_execution_status() -> None:
    """Test updating execution status."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.update_execution_status(
        execution.execution_id, WorkflowStatus.EXECUTING
    )

    updated = manager.get_execution(execution.execution_id)
    assert updated is not None
    assert updated.status == WorkflowStatus.EXECUTING


def test_initialize_node_states() -> None:
    """Test initializing node states."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(
        execution.execution_id, ["task1", "task2"]
    )

    updated = manager.get_execution(execution.execution_id)
    assert updated is not None
    assert len(updated.node_states) == 2
    assert "task1" in updated.node_states
    assert "task2" in updated.node_states


def test_update_node_status() -> None:
    """Test updating node status."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(execution.execution_id, ["task1"])
    manager.update_node_status(
        execution.execution_id, "task1", NodeStatus.RUNNING
    )

    state = manager.get_node_state(execution.execution_id, "task1")
    assert state is not None
    assert state.status == NodeStatus.RUNNING
    assert state.started_at is not None


def test_update_node_with_result() -> None:
    """Test updating node with result data."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(execution.execution_id, ["task1"])

    result = {"output": "test result"}
    manager.update_node_status(
        execution.execution_id,
        "task1",
        NodeStatus.COMPLETED,
        result=result,
    )

    state = manager.get_node_state(execution.execution_id, "task1")
    assert state is not None
    assert state.status == NodeStatus.COMPLETED
    assert state.result == result


def test_increment_node_attempts() -> None:
    """Test incrementing node attempt count."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(execution.execution_id, ["task1"])

    count1 = manager.increment_node_attempts(execution.execution_id, "task1")
    count2 = manager.increment_node_attempts(execution.execution_id, "task1")

    assert count1 == 1
    assert count2 == 2


def test_get_completed_nodes() -> None:
    """Test getting completed nodes."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(
        execution.execution_id, ["task1", "task2", "task3"]
    )

    manager.update_node_status(
        execution.execution_id, "task1", NodeStatus.COMPLETED
    )
    manager.update_node_status(
        execution.execution_id, "task2", NodeStatus.RUNNING
    )
    manager.update_node_status(
        execution.execution_id, "task3", NodeStatus.COMPLETED
    )

    completed = manager.get_completed_nodes(execution.execution_id)
    assert completed == {"task1", "task3"}


def test_get_failed_nodes() -> None:
    """Test getting failed nodes."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(
        execution.execution_id, ["task1", "task2"]
    )

    manager.update_node_status(
        execution.execution_id,
        "task1",
        NodeStatus.FAILED,
        error_message="Task failed",
    )

    failed = manager.get_failed_nodes(execution.execution_id)
    assert failed == {"task1"}


def test_get_running_nodes() -> None:
    """Test getting running nodes."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(
        execution.execution_id, ["task1", "task2"]
    )

    manager.update_node_status(
        execution.execution_id, "task1", NodeStatus.RUNNING
    )

    running = manager.get_running_nodes(execution.execution_id)
    assert running == {"task1"}


def test_allocate_agent() -> None:
    """Test allocating agent to node."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.allocate_agent(execution.execution_id, "task1", "agent_123")

    updated = manager.get_execution(execution.execution_id)
    assert updated is not None
    assert updated.allocated_agents["task1"] == "agent_123"


def test_update_coordination_overhead() -> None:
    """Test updating coordination overhead."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.update_coordination_overhead(execution.execution_id, 50.0)
    manager.update_coordination_overhead(execution.execution_id, 30.0)

    updated = manager.get_execution(execution.execution_id)
    assert updated is not None
    assert updated.coordination_overhead_ms == 80.0


def test_checkpoint_and_restore() -> None:
    """Test checkpoint and restore functionality."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    manager.initialize_node_states(execution.execution_id, ["task1"])
    manager.update_node_status(
        execution.execution_id, "task1", NodeStatus.COMPLETED
    )

    # Create checkpoint
    checkpoint = manager.checkpoint(execution.execution_id)
    assert checkpoint is not None

    # Create new manager and restore
    new_manager = WorkflowStateManager()
    restored_id = new_manager.restore(checkpoint)

    assert restored_id == execution.execution_id
    restored = new_manager.get_execution(restored_id)
    assert restored is not None
    assert restored.workflow_id == workflow_id
    assert restored.node_states["task1"].status == NodeStatus.COMPLETED


def test_set_output_data() -> None:
    """Test setting output data."""
    manager = WorkflowStateManager()
    workflow_id = uuid4()

    execution = manager.create_execution(workflow_id)
    output_data = {"result": "test output", "metrics": {"count": 42}}

    manager.set_output_data(execution.execution_id, output_data)

    updated = manager.get_execution(execution.execution_id)
    assert updated is not None
    assert updated.output_data == output_data
