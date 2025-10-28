"""Tests for workflow graph."""

from __future__ import annotations

import pytest

from orchestration.workflow.graph import (
    CycleDetectedError,
    InvalidGraphError,
    WorkflowGraph)
from orchestration.workflow.models import (
    EdgeType,
    TaskNode,
    WorkflowDefinition)


def test_empty_graph() -> None:
    """Test creating an empty graph."""
    graph = WorkflowGraph()
    assert graph._graph.number_of_nodes() == 0
    assert graph._graph.number_of_edges() == 0


def test_add_node() -> None:
    """Test adding a node to the graph."""
    graph = WorkflowGraph()
    node = TaskNode(node_id="task1", agent_role="researcher")

    graph.add_node(node)

    assert graph._graph.number_of_nodes() == 1
    assert "task1" in graph._graph.nodes()


def test_add_edge() -> None:
    """Test adding an edge between nodes."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)

    assert graph._graph.number_of_edges() == 1


def test_get_dependencies() -> None:
    """Test getting node dependencies."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)

    deps = graph.get_dependencies("task2")
    assert deps == ["task1"]


def test_get_entry_nodes() -> None:
    """Test getting entry nodes."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)

    entry_nodes = graph.get_entry_nodes()
    assert entry_nodes == ["task1"]


def test_get_exit_nodes() -> None:
    """Test getting exit nodes."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)

    exit_nodes = graph.get_exit_nodes()
    assert exit_nodes == ["task2"]


def test_topological_sort() -> None:
    """Test topological sorting of nodes."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])
    node3 = TaskNode(node_id="task3", agent_role="summarizer", depends_on=["task2"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)
    graph.add_edge("task2", "task3", EdgeType.SEQUENTIAL)

    sorted_nodes = graph.topological_sort()
    assert sorted_nodes == ["task1", "task2", "task3"]


def test_cycle_detection() -> None:
    """Test cycle detection."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)
    graph.add_edge("task2", "task1", EdgeType.SEQUENTIAL)  # Creates cycle

    cycles = graph.detect_cycles()
    assert len(cycles) > 0


def test_validate_with_cycle() -> None:
    """Test validation fails with cycle."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)
    graph.add_edge("task2", "task1", EdgeType.SEQUENTIAL)

    with pytest.raises(CycleDetectedError):
        graph.validate()


def test_validate_disconnected_components() -> None:
    """Test validation fails with disconnected components."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer")
    node3 = TaskNode(node_id="task3", agent_role="summarizer")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    # task1 and task2 connected, task3 disconnected
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)

    with pytest.raises(InvalidGraphError):
        graph.validate()


def test_get_ready_nodes() -> None:
    """Test getting ready nodes."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])
    node3 = TaskNode(node_id="task3", agent_role="summarizer", depends_on=["task2"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)
    graph.add_edge("task2", "task3", EdgeType.SEQUENTIAL)

    # Initially, only task1 is ready
    ready = graph.get_ready_nodes(set())
    assert ready == ["task1"]

    # After task1 completes, task2 is ready
    ready = graph.get_ready_nodes({"task1"})
    assert ready == ["task2"]

    # After task2 completes, task3 is ready
    ready = graph.get_ready_nodes({"task1", "task2"})
    assert ready == ["task3"]


def test_parallel_execution_groups() -> None:
    """Test getting parallel execution groups."""
    graph = WorkflowGraph()
    # Create a diamond-shaped graph:
    # task1 -> task2, task3 -> task4
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer1", depends_on=["task1"])
    node3 = TaskNode(node_id="task3", agent_role="analyzer2", depends_on=["task1"])
    node4 = TaskNode(node_id="task4", agent_role="summarizer", depends_on=["task2", "task3"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)
    graph.add_edge("task1", "task3", EdgeType.SEQUENTIAL)
    graph.add_edge("task2", "task4", EdgeType.SEQUENTIAL)
    graph.add_edge("task3", "task4", EdgeType.SEQUENTIAL)

    groups = graph.get_parallel_execution_groups()
    assert len(groups) == 3
    assert groups[0] == ["task1"]
    assert set(groups[1]) == {"task2", "task3"}  # Can execute in parallel
    assert groups[2] == ["task4"]


def test_from_definition() -> None:
    """Test creating graph from workflow definition."""
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])

    definition = WorkflowDefinition(
        name="test_workflow",
        nodes=[node1, node2])

    graph = WorkflowGraph.from_definition(definition)

    assert graph._graph.number_of_nodes() == 2
    assert graph._graph.number_of_edges() == 1


def test_graph_metrics() -> None:
    """Test graph metrics."""
    graph = WorkflowGraph()
    node1 = TaskNode(node_id="task1", agent_role="researcher")
    node2 = TaskNode(node_id="task2", agent_role="analyzer", depends_on=["task1"])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("task1", "task2", EdgeType.SEQUENTIAL)

    metrics = graph.get_graph_metrics()

    assert metrics["node_count"] == 2
    assert metrics["edge_count"] == 1
    assert metrics["is_dag"] is True
    assert metrics["max_depth"] == 1
    assert metrics["entry_nodes"] == 1
    assert metrics["exit_nodes"] == 1
