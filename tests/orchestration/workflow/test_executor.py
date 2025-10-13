"""Tests for workflow executor."""

from __future__ import annotations

import pytest

from orchestration.workflow.builder import WorkflowBuilder
from orchestration.workflow.executor import WorkflowExecutor
from orchestration.workflow.models import NodeStatus, WorkflowStatus
from orchestration.workflow.state import WorkflowStateManager


@pytest.mark.asyncio
async def test_execute_simple_workflow() -> None:
    """Test executing a simple workflow."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .add_task("task2", "analyzer")
        .depends_on("task1")
        .build()
    )

    executor = WorkflowExecutor()
    execution = await executor.execute(workflow)

    assert execution.status == WorkflowStatus.COMPLETED
    assert execution.node_states["task1"].status == NodeStatus.COMPLETED
    assert execution.node_states["task2"].status == NodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_parallel_workflow() -> None:
    """Test executing workflow with parallel tasks."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .add_task("task2", "analyzer1")
        .depends_on("task1")
        .add_task("task3", "analyzer2")
        .depends_on("task1")
        .set_coordination(coordination_type="hybrid")
        .build()
    )

    executor = WorkflowExecutor()
    execution = await executor.execute(workflow)

    assert execution.status == WorkflowStatus.COMPLETED
    # All tasks should complete
    assert execution.node_states["task1"].status == NodeStatus.COMPLETED
    assert execution.node_states["task2"].status == NodeStatus.COMPLETED
    assert execution.node_states["task3"].status == NodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_state_management() -> None:
    """Test workflow state management during execution."""
    builder = WorkflowBuilder("test_workflow")
    workflow = builder.add_task("task1", "researcher").build()

    state_manager = WorkflowStateManager()
    executor = WorkflowExecutor(state_manager)

    execution = await executor.execute(workflow)

    # Check state was properly tracked
    assert state_manager.get_execution(execution.execution_id) is not None
    completed = state_manager.get_completed_nodes(execution.execution_id)
    assert "task1" in completed


@pytest.mark.asyncio
async def test_get_execution_status() -> None:
    """Test getting execution status."""
    builder = WorkflowBuilder("test_workflow")
    workflow = builder.add_task("task1", "researcher").build()

    executor = WorkflowExecutor()
    execution = await executor.execute(workflow)

    # Get status after completion
    status = executor.get_execution_status(execution.execution_id)
    assert status is not None
    assert status.status == WorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_sequential_execution() -> None:
    """Test sequential execution maintains order."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .add_task("task2", "analyzer")
        .depends_on("task1")
        .add_task("task3", "summarizer")
        .depends_on("task2")
        .set_coordination(coordination_type="graph_based")
        .build()
    )

    executor = WorkflowExecutor()
    execution = await executor.execute(workflow)

    assert execution.status == WorkflowStatus.COMPLETED

    # Check all tasks completed in order
    task1_state = execution.node_states["task1"]
    task2_state = execution.node_states["task2"]
    task3_state = execution.node_states["task3"]

    assert task1_state.status == NodeStatus.COMPLETED
    assert task2_state.status == NodeStatus.COMPLETED
    assert task3_state.status == NodeStatus.COMPLETED

    # Verify timestamps show sequential execution
    assert task1_state.completed_at is not None
    assert task2_state.started_at is not None
    assert task1_state.completed_at <= task2_state.started_at


@pytest.mark.asyncio
async def test_execution_with_input_data() -> None:
    """Test execution with input data."""
    builder = WorkflowBuilder("test_workflow")
    workflow = builder.add_task("task1", "researcher").build()

    input_data = {"query": "test query", "max_results": 10}

    executor = WorkflowExecutor()
    execution = await executor.execute(workflow, input_data=input_data)

    assert execution.input_data == input_data


@pytest.mark.asyncio
async def test_coordination_overhead_tracking() -> None:
    """Test coordination overhead is tracked."""
    builder = WorkflowBuilder("test_workflow")
    workflow = builder.add_task("task1", "researcher").build()

    executor = WorkflowExecutor()
    execution = await executor.execute(workflow)

    # Coordination overhead should be tracked
    assert execution.coordination_overhead_ms > 0
