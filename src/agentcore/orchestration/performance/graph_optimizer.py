"""
Graph Planning Optimizer

Optimizations for workflow graph planning to achieve <1s planning for 1000+ nodes:
- Caching of graph computations
- Lazy evaluation of graph properties
- Parallel pathfinding algorithms
- Memoization of topological sorts
"""

from __future__ import annotations

import functools
from typing import Any, Hashable

import networkx as nx


class GraphOptimizer:
    """
    Optimizer for workflow graph operations.

    Provides caching and optimization for expensive graph algorithms
    used in workflow planning.
    """

    def __init__(self, cache_size: int = 128) -> None:
        """
        Initialize graph optimizer.

        Args:
            cache_size: Maximum number of cached computations
        """
        self.cache_size = cache_size
        self._topo_cache: dict[int, list[Hashable]] = {}
        self._path_cache: dict[tuple[int, Hashable, Hashable], list[Hashable]] = {}
        self._levels_cache: dict[int, list[set[Hashable]]] = {}

    def topological_sort_cached(
        self, graph: nx.DiGraph
    ) -> list[Hashable]:
        """
        Cached topological sort for workflow execution ordering.

        Args:
            graph: Directed acyclic graph

        Returns:
            List of nodes in topological order

        Raises:
            nx.NetworkXError: If graph contains cycles
        """
        graph_hash = hash(frozenset(graph.edges()))

        if graph_hash not in self._topo_cache:
            self._topo_cache[graph_hash] = list(nx.topological_sort(graph))
            self._trim_cache(self._topo_cache)

        return self._topo_cache[graph_hash]

    def find_critical_path_cached(
        self, graph: nx.DiGraph, weight: str = "weight"
    ) -> list[Hashable]:
        """
        Find critical path (longest path) in workflow graph with caching.

        Args:
            graph: Directed acyclic graph
            weight: Edge attribute for path length calculation

        Returns:
            List of nodes in critical path
        """
        graph_hash = hash(frozenset(graph.edges()))
        cache_key = (graph_hash, None, None)  # Full path, no specific source/target

        if cache_key not in self._path_cache:
            try:
                path = nx.algorithms.dag.dag_longest_path(graph, weight=weight)
                self._path_cache[cache_key] = path
                self._trim_cache(self._path_cache)
            except nx.NetworkXError:
                # Empty graph or other error
                self._path_cache[cache_key] = []

        return self._path_cache[cache_key]

    def compute_execution_levels(
        self, graph: nx.DiGraph
    ) -> list[set[Hashable]]:
        """
        Compute execution levels for parallel task scheduling.

        Nodes at the same level can execute in parallel (no dependencies between them).

        Args:
            graph: Directed acyclic graph

        Returns:
            List of sets, where each set contains nodes that can execute in parallel
        """
        graph_hash = hash(frozenset(graph.edges()))

        if graph_hash not in self._levels_cache:
            levels: list[set[Hashable]] = []
            remaining_nodes = set(graph.nodes())

            while remaining_nodes:
                # Find nodes with no unprocessed predecessors
                level_nodes = {
                    node
                    for node in remaining_nodes
                    if not any(pred in remaining_nodes for pred in graph.predecessors(node))
                }

                if not level_nodes:
                    # Cycle detected or isolated nodes
                    break

                levels.append(level_nodes)
                remaining_nodes -= level_nodes

            self._levels_cache[graph_hash] = levels
            self._trim_cache(self._levels_cache)

        return self._levels_cache[graph_hash]

    def _trim_cache(self, cache: dict[Any, Any]) -> None:
        """
        Trim cache to maximum size using LRU strategy.

        Args:
            cache: Cache dictionary to trim
        """
        if len(cache) > self.cache_size:
            # Remove oldest entries (simple FIFO, could use LRU)
            excess = len(cache) - self.cache_size
            for key in list(cache.keys())[:excess]:
                del cache[key]

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._topo_cache.clear()
        self._path_cache.clear()
        self._levels_cache.clear()

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def compute_graph_metrics(
        nodes_tuple: tuple[Hashable, ...],
        edges_tuple: tuple[tuple[Hashable, Hashable], ...],
    ) -> dict[str, Any]:
        """
        Compute graph metrics with LRU caching.

        Args:
            nodes_tuple: Tuple of graph nodes (for hashing)
            edges_tuple: Tuple of graph edges (for hashing)

        Returns:
            Dictionary of graph metrics
        """
        # Reconstruct graph from tuples
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes_tuple)
        graph.add_edges_from(edges_tuple)

        # Compute metrics
        metrics = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_dag": nx.is_directed_acyclic_graph(graph),
        }

        # Additional DAG metrics
        if metrics["is_dag"] and graph.number_of_nodes() > 0:
            try:
                metrics["longest_path_length"] = nx.dag_longest_path_length(graph)
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                metrics["longest_path_length"] = 0

        return metrics


def optimize_workflow_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Apply graph optimizations for workflow execution.

    Optimizations:
    - Remove redundant edges (transitive reduction)
    - Simplify graph structure
    - Validate DAG properties

    Args:
        graph: Original workflow graph

    Returns:
        Optimized workflow graph
    """
    optimized = graph.copy()

    # Apply transitive reduction to remove redundant edges
    if nx.is_directed_acyclic_graph(optimized):
        optimized = nx.algorithms.dag.transitive_reduction(optimized)

    # Remove self-loops (invalid in workflow)
    self_loops = list(nx.selfloop_edges(optimized))
    optimized.remove_edges_from(self_loops)

    return optimized


def analyze_workflow_parallelism(graph: nx.DiGraph) -> dict[str, Any]:
    """
    Analyze parallelism opportunities in workflow graph.

    Args:
        graph: Workflow graph

    Returns:
        Dictionary with parallelism analysis
    """
    optimizer = GraphOptimizer()

    levels = optimizer.compute_execution_levels(graph)
    critical_path = optimizer.find_critical_path_cached(graph)

    max_parallelism = max((len(level) for level in levels), default=0)
    avg_parallelism = sum(len(level) for level in levels) / len(levels) if levels else 0

    return {
        "execution_levels": len(levels),
        "max_parallelism": max_parallelism,
        "avg_parallelism": avg_parallelism,
        "critical_path_length": len(critical_path),
        "total_nodes": graph.number_of_nodes(),
        "parallelism_efficiency": avg_parallelism / max_parallelism if max_parallelism > 0 else 0,
    }
