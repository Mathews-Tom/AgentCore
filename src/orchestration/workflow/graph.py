"""Workflow graph implementation using NetworkX."""

from __future__ import annotations

from typing import Any

import networkx as nx

from orchestration.workflow.models import (
    ConditionalEdge,
    EdgeType,
    TaskNode,
    WorkflowDefinition,
)


class WorkflowGraphError(Exception):
    """Base exception for workflow graph errors."""


class CycleDetectedError(WorkflowGraphError):
    """Exception raised when a cycle is detected in the workflow graph."""


class InvalidGraphError(WorkflowGraphError):
    """Exception raised when the graph structure is invalid."""


class WorkflowGraph:
    """DAG-based workflow graph using NetworkX."""

    def __init__(self) -> None:
        """Initialize an empty workflow graph."""
        self._graph: nx.DiGraph = nx.DiGraph()
        self._node_data: dict[str, TaskNode] = {}

    @classmethod
    def from_definition(cls, definition: WorkflowDefinition) -> WorkflowGraph:
        """Create a workflow graph from a workflow definition.

        Args:
            definition: Workflow definition to convert to graph

        Returns:
            WorkflowGraph instance

        Raises:
            CycleDetectedError: If the graph contains cycles
            InvalidGraphError: If the graph structure is invalid
        """
        graph = cls()

        # Add all nodes
        for node in definition.nodes:
            graph.add_node(node)

        # Add sequential edges based on dependencies
        for node in definition.nodes:
            for dep_id in node.depends_on:
                graph.add_edge(dep_id, node.node_id, EdgeType.SEQUENTIAL)

        # Add conditional edges
        for edge in definition.conditional_edges:
            graph.add_edge(
                edge.from_node,
                edge.to_node,
                EdgeType.CONDITIONAL,
                condition=edge.condition,
            )

        # Validate the graph
        graph.validate()

        return graph

    def add_node(self, node: TaskNode) -> None:
        """Add a node to the graph.

        Args:
            node: Node to add to the graph
        """
        self._graph.add_node(node.node_id)
        self._node_data[node.node_id] = node

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType,
        **attributes: Any,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            from_node: ID of the source node
            to_node: ID of the destination node
            edge_type: Type of edge (sequential or conditional)
            **attributes: Additional edge attributes
        """
        self._graph.add_edge(from_node, to_node, edge_type=edge_type, **attributes)

    def get_node(self, node_id: str) -> TaskNode:
        """Get node data by ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            TaskNode data

        Raises:
            KeyError: If node not found
        """
        return self._node_data[node_id]

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get all dependencies (predecessors) of a node.

        Args:
            node_id: ID of the node

        Returns:
            List of node IDs that this node depends on
        """
        return list(self._graph.predecessors(node_id))

    def get_dependents(self, node_id: str) -> list[str]:
        """Get all dependents (successors) of a node.

        Args:
            node_id: ID of the node

        Returns:
            List of node IDs that depend on this node
        """
        return list(self._graph.successors(node_id))

    def get_entry_nodes(self) -> list[str]:
        """Get all entry nodes (nodes with no dependencies).

        Returns:
            List of node IDs with no predecessors
        """
        return [
            node for node in self._graph.nodes() if self._graph.in_degree(node) == 0
        ]

    def get_exit_nodes(self) -> list[str]:
        """Get all exit nodes (nodes with no dependents).

        Returns:
            List of node IDs with no successors
        """
        return [
            node for node in self._graph.nodes() if self._graph.out_degree(node) == 0
        ]

    def topological_sort(self) -> list[str]:
        """Get nodes in topological order.

        Returns:
            List of node IDs in topological order

        Raises:
            CycleDetectedError: If the graph contains cycles
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXError as e:
            raise CycleDetectedError(
                "Workflow graph contains cycles and cannot be executed"
            ) from e

    def detect_cycles(self) -> list[list[str]]:
        """Detect cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        try:
            cycles = list(nx.simple_cycles(self._graph))
            return cycles
        except Exception:
            return []

    def validate(self) -> None:
        """Validate the workflow graph structure.

        Raises:
            CycleDetectedError: If the graph contains cycles
            InvalidGraphError: If the graph structure is invalid
        """
        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            cycle_str = ", ".join(" -> ".join(cycle) for cycle in cycles)
            raise CycleDetectedError(f"Workflow graph contains cycles: {cycle_str}")

        # Check that all referenced nodes exist
        for node_id in self._graph.nodes():
            if node_id not in self._node_data:
                raise InvalidGraphError(
                    f"Node {node_id} referenced in graph but not defined"
                )

        # Check for disconnected components
        if not nx.is_weakly_connected(self._graph):
            raise InvalidGraphError("Workflow graph contains disconnected components")

    def get_ready_nodes(self, completed_nodes: set[str]) -> list[str]:
        """Get nodes that are ready to execute.

        Args:
            completed_nodes: Set of node IDs that have completed

        Returns:
            List of node IDs ready to execute
        """
        ready = []
        for node_id in self._graph.nodes():
            if node_id in completed_nodes:
                continue

            # Check if all dependencies are completed
            dependencies = self.get_dependencies(node_id)
            if all(dep in completed_nodes for dep in dependencies):
                ready.append(node_id)

        return ready

    def get_parallel_execution_groups(self) -> list[list[str]]:
        """Get groups of nodes that can be executed in parallel.

        Returns:
            List of parallel execution groups, where each group
            contains node IDs that can execute concurrently
        """
        # Use topological generations to identify parallel groups
        try:
            generations = list(nx.topological_generations(self._graph))
            return [list(gen) for gen in generations]
        except nx.NetworkXError:
            # If graph has cycles, return empty list
            return []

    def get_graph_metrics(self) -> dict[str, Any]:
        """Get metrics about the workflow graph.

        Returns:
            Dictionary containing graph metrics
        """
        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
            "max_depth": (
                nx.dag_longest_path_length(self._graph)
                if nx.is_directed_acyclic_graph(self._graph)
                else None
            ),
            "entry_nodes": len(self.get_entry_nodes()),
            "exit_nodes": len(self.get_exit_nodes()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": [
                {
                    "id": node_id,
                    "data": self._node_data[node_id].model_dump(),
                }
                for node_id in self._graph.nodes()
            ],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "data": data,
                }
                for u, v, data in self._graph.edges(data=True)
            ],
            "metrics": self.get_graph_metrics(),
        }
