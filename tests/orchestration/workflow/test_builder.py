"""Tests for workflow builder."""

from __future__ import annotations

import pytest

from orchestration.workflow.builder import WorkflowBuilder
from orchestration.workflow.models import RetryPolicy


def test_create_simple_workflow() -> None:
    """Test creating a simple workflow."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .add_task("task2", "analyzer")
        .depends_on("task1")
        .build()
    )

    assert workflow.name == "test_workflow"
    assert len(workflow.nodes) == 2
    assert workflow.nodes[0].node_id == "task1"
    assert workflow.nodes[1].node_id == "task2"
    assert workflow.nodes[1].depends_on == ["task1"]


def test_builder_with_description() -> None:
    """Test adding description to workflow."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.description("Test workflow description")
        .add_task("task1", "researcher")
        .build()
    )

    assert workflow.description == "Test workflow description"


def test_builder_with_metadata() -> None:
    """Test adding metadata to tasks."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .with_metadata(priority="high", category="research")
        .build()
    )

    assert workflow.nodes[0].metadata["priority"] == "high"
    assert workflow.nodes[0].metadata["category"] == "research"


def test_builder_with_compensation() -> None:
    """Test adding compensation action."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .with_compensation("rollback_task1")
        .build()
    )

    assert workflow.nodes[0].compensation_action == "rollback_task1"


def test_builder_multiple_dependencies() -> None:
    """Test task with multiple dependencies."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .add_task("task2", "analyzer")
        .add_task("task3", "summarizer")
        .depends_on("task1", "task2")
        .build()
    )

    assert set(workflow.nodes[2].depends_on) == {"task1", "task2"}


def test_builder_with_coordination() -> None:
    """Test setting coordination configuration."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .set_coordination(
            coordination_type="hybrid",
            event_driven_events=["status_update"],
            max_parallel_tasks=5,
        )
        .build()
    )

    assert workflow.coordination.coordination_type == "hybrid"
    assert workflow.coordination.event_driven_events == ["status_update"]
    assert workflow.coordination.max_parallel_tasks == 5


def test_builder_validation_no_nodes() -> None:
    """Test validation fails with no nodes."""
    builder = WorkflowBuilder("test_workflow")

    with pytest.raises(ValueError, match="at least one node"):
        builder.build()


def test_builder_validation_duplicate_node_ids() -> None:
    """Test validation fails with duplicate node IDs."""
    builder = WorkflowBuilder("test_workflow")

    with pytest.raises(ValueError, match="must be unique"):
        builder.add_task("task1", "researcher").add_task(
            "task1", "analyzer"
        ).build()


def test_builder_validation_invalid_dependency() -> None:
    """Test validation fails with invalid dependency."""
    builder = WorkflowBuilder("test_workflow")

    with pytest.raises(ValueError, match="non-existent node"):
        builder.add_task("task1", "researcher").depends_on(
            "nonexistent"
        ).build()


def test_builder_conditional_edge() -> None:
    """Test adding conditional edge."""
    builder = WorkflowBuilder("test_workflow")
    workflow = (
        builder.add_task("task1", "researcher")
        .add_task("task2", "analyzer")
        .add_task("task3", "summarizer")
        .add_conditional_edge("task1", "task2", "result > 0.5")
        .add_conditional_edge("task1", "task3", "result <= 0.5")
        .build()
    )

    assert len(workflow.conditional_edges) == 2
    assert workflow.conditional_edges[0].from_node == "task1"
    assert workflow.conditional_edges[0].to_node == "task2"
    assert workflow.conditional_edges[0].condition == "result > 0.5"


def test_from_json() -> None:
    """Test creating workflow from JSON."""
    json_str = """
    {
        "name": "test_workflow",
        "version": "1.0.0",
        "nodes": [
            {
                "node_id": "task1",
                "agent_role": "researcher",
                "timeout_seconds": 300
            }
        ]
    }
    """

    workflow = WorkflowBuilder.from_json(json_str)

    assert workflow.name == "test_workflow"
    assert len(workflow.nodes) == 1
    assert workflow.nodes[0].node_id == "task1"


def test_from_yaml() -> None:
    """Test creating workflow from YAML."""
    yaml_str = """
    name: test_workflow
    version: 1.0.0
    nodes:
      - node_id: task1
        agent_role: researcher
        timeout_seconds: 300
    """

    workflow = WorkflowBuilder.from_yaml(yaml_str)

    assert workflow.name == "test_workflow"
    assert len(workflow.nodes) == 1
    assert workflow.nodes[0].node_id == "task1"


def test_builder_with_custom_retry_policy() -> None:
    """Test adding custom retry policy."""
    builder = WorkflowBuilder("test_workflow")
    retry_policy = RetryPolicy(
        max_attempts=5, backoff_multiplier=1.5, initial_delay_seconds=2.0
    )

    workflow = builder.add_task(
        "task1", "researcher", retry_policy=retry_policy
    ).build()

    assert workflow.nodes[0].retry_policy.max_attempts == 5
    assert workflow.nodes[0].retry_policy.backoff_multiplier == 1.5
