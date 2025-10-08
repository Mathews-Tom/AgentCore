"""Tests for workflow models."""

from __future__ import annotations

import pytest
from uuid import uuid4

from orchestration.workflow.models import (
    NodeStatus,
    NodeType,
    RetryPolicy,
    TaskNode,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
)


def test_task_node_creation() -> None:
    """Test creating a task node."""
    node = TaskNode(
        node_id="task1",
        agent_role="researcher",
        timeout_seconds=300,
    )

    assert node.node_id == "task1"
    assert node.agent_role == "researcher"
    assert node.node_type == NodeType.TASK
    assert node.timeout_seconds == 300
    assert len(node.depends_on) == 0


def test_task_node_with_dependencies() -> None:
    """Test task node with dependencies."""
    node = TaskNode(
        node_id="task2",
        agent_role="analyzer",
        depends_on=["task1"],
    )

    assert node.depends_on == ["task1"]


def test_retry_policy_defaults() -> None:
    """Test retry policy defaults."""
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.backoff_multiplier == 2.0
    assert policy.initial_delay_seconds == 1.0
    assert policy.max_delay_seconds == 60.0


def test_workflow_definition() -> None:
    """Test workflow definition creation."""
    workflow = WorkflowDefinition(
        name="test_workflow",
        version="1.0.0",
    )

    assert workflow.name == "test_workflow"
    assert workflow.version == "1.0.0"
    assert len(workflow.nodes) == 0


def test_workflow_execution() -> None:
    """Test workflow execution state."""
    workflow_id = uuid4()
    execution = WorkflowExecution(
        workflow_id=workflow_id,
        status=WorkflowStatus.PLANNING,
    )

    assert execution.workflow_id == workflow_id
    assert execution.status == WorkflowStatus.PLANNING
    assert len(execution.node_states) == 0


def test_node_status_enum() -> None:
    """Test node status enum values."""
    assert NodeStatus.PENDING == "pending"
    assert NodeStatus.RUNNING == "running"
    assert NodeStatus.COMPLETED == "completed"
    assert NodeStatus.FAILED == "failed"


def test_workflow_status_enum() -> None:
    """Test workflow status enum values."""
    assert WorkflowStatus.PLANNING == "planning"
    assert WorkflowStatus.EXECUTING == "executing"
    assert WorkflowStatus.COMPLETED == "completed"
    assert WorkflowStatus.FAILED == "failed"
