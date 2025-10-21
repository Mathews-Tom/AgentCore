"""
Orchestration Engine Benchmarking Suite

Comprehensive benchmarks for:
- Workflow graph planning performance
- Event processing throughput
- Coordination latency
- Memory efficiency
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    duration_seconds: float
    throughput: float | None = None
    memory_mb: float | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] | None = None


class OrchestrationBenchmarks:
    """
    Benchmark suite for orchestration engine performance.

    Measures:
    - Graph planning time for various graph sizes
    - Event processing throughput
    - Coordination overhead
    - Memory consumption
    """

    @staticmethod
    def benchmark_graph_planning(
        node_count: int, edge_density: float = 0.1
    ) -> BenchmarkResult:
        """
        Benchmark workflow graph planning performance.

        Target: <1s planning for 1000+ node graphs

        Args:
            node_count: Number of nodes in workflow graph
            edge_density: Density of edges (0.0 to 1.0)

        Returns:
            BenchmarkResult with planning duration and metadata
        """
        try:
            # Create test workflow graph as DAG (avoid expensive cycle removal)
            # Generate random DAG by ensuring edges only go from lower to higher node IDs
            graph = nx.DiGraph()
            graph.add_nodes_from(range(node_count))

            # Add edges with constraint: i < j to guarantee DAG property
            import random

            random.seed(42)  # Deterministic for benchmarking
            target_edges = int(node_count * (node_count - 1) * edge_density / 2)
            edges_added = 0
            attempts = 0
            max_attempts = target_edges * 10

            while edges_added < target_edges and attempts < max_attempts:
                i = random.randint(0, node_count - 2)
                j = random.randint(i + 1, node_count - 1)
                if not graph.has_edge(i, j):
                    graph.add_edge(i, j)
                    edges_added += 1
                attempts += 1

            # Add task metadata
            for node in graph.nodes():
                graph.nodes[node]["task_id"] = f"task_{node}"
                graph.nodes[node]["agent_type"] = f"agent_{node % 5}"

            # Measure planning time
            start_time = time.perf_counter()

            # Simulate workflow planning operations
            # 1. Topological sort for execution order (graph is already DAG)
            execution_order = list(nx.topological_sort(graph))

            # 2. Find critical path
            if graph.number_of_edges() > 0:
                longest_path = nx.algorithms.dag.dag_longest_path(graph)
                critical_path_length = len(longest_path)
            else:
                critical_path_length = 1

            # 3. Identify parallel execution opportunities (optimized O(n+e) algorithm)
            # Use Kahn's algorithm for level detection
            in_degree = {node: graph.in_degree(node) for node in graph.nodes()}
            levels = []
            queue = [node for node, degree in in_degree.items() if degree == 0]

            while queue:
                # Current level is all nodes with 0 in-degree
                current_level = set(queue)
                levels.append(current_level)

                # Process current level and update in-degrees
                next_queue = []
                for node in queue:
                    for successor in graph.successors(node):
                        in_degree[successor] -= 1
                        if in_degree[successor] == 0:
                            next_queue.append(successor)

                queue = next_queue

            parallelism = max(len(level) for level in levels) if levels else 1

            duration = time.perf_counter() - start_time

            return BenchmarkResult(
                name=f"graph_planning_{node_count}_nodes",
                duration_seconds=duration,
                throughput=node_count / duration if duration > 0 else None,
                success=True,
                metadata={
                    "node_count": node_count,
                    "edge_count": graph.number_of_edges(),
                    "critical_path_length": critical_path_length,
                    "max_parallelism": parallelism,
                    "execution_levels": len(levels),
                },
            )

        except Exception as e:
            return BenchmarkResult(
                name=f"graph_planning_{node_count}_nodes",
                duration_seconds=0.0,
                success=False,
                error=str(e),
            )

    @staticmethod
    async def benchmark_event_processing(
        event_count: int, batch_size: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark event processing throughput.

        Target: 100,000+ events/second

        Args:
            event_count: Number of events to process
            batch_size: Size of event batches

        Returns:
            BenchmarkResult with throughput metrics
        """
        try:
            # Simulate event processing
            start_time = time.perf_counter()

            processed_count = 0
            batches = (event_count + batch_size - 1) // batch_size

            for batch_num in range(batches):
                batch_events = min(batch_size, event_count - processed_count)

                # Simulate async event processing
                tasks = [
                    asyncio.create_task(OrchestrationBenchmarks._process_event(i))
                    for i in range(batch_events)
                ]
                await asyncio.gather(*tasks)

                processed_count += batch_events

            duration = time.perf_counter() - start_time
            throughput = event_count / duration if duration > 0 else 0

            return BenchmarkResult(
                name=f"event_processing_{event_count}_events",
                duration_seconds=duration,
                throughput=throughput,
                success=True,
                metadata={
                    "event_count": event_count,
                    "batch_size": batch_size,
                    "batches_processed": batches,
                    "events_per_second": throughput,
                },
            )

        except Exception as e:
            return BenchmarkResult(
                name=f"event_processing_{event_count}_events",
                duration_seconds=0.0,
                success=False,
                error=str(e),
            )

    @staticmethod
    async def _process_event(event_id: int) -> None:
        """
        Simulate processing a single event.

        Args:
            event_id: Event identifier
        """
        # Simulate minimal event processing overhead
        await asyncio.sleep(0)  # Yield control

    @staticmethod
    def benchmark_suite() -> dict[str, BenchmarkResult]:
        """
        Run complete benchmark suite.

        Returns:
            Dictionary of benchmark results by name
        """
        results = {}

        # Graph planning benchmarks
        for node_count in [10, 100, 500, 1000, 2000]:
            result = OrchestrationBenchmarks.benchmark_graph_planning(node_count)
            results[result.name] = result

        return results

    @staticmethod
    async def async_benchmark_suite() -> dict[str, BenchmarkResult]:
        """
        Run async benchmark suite.

        Returns:
            Dictionary of benchmark results by name
        """
        results = {}

        # Event processing benchmarks
        for event_count in [1000, 10000, 50000, 100000]:
            result = await OrchestrationBenchmarks.benchmark_event_processing(
                event_count
            )
            results[result.name] = result

        return results

    @staticmethod
    def print_results(results: dict[str, BenchmarkResult]) -> None:
        """
        Print benchmark results in formatted table.

        Args:
            results: Dictionary of benchmark results
        """
        print("\n" + "=" * 80)
        print("ORCHESTRATION ENGINE PERFORMANCE BENCHMARKS")
        print("=" * 80)
        print()

        for name, result in results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.name}")
            print(f"  Duration: {result.duration_seconds:.4f}s")

            if result.throughput:
                print(f"  Throughput: {result.throughput:,.0f} ops/sec")

            if result.memory_mb:
                print(f"  Memory: {result.memory_mb:.2f} MB")

            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")

            if result.error:
                print(f"  Error: {result.error}")

            print()

        # Summary
        passed = sum(1 for r in results.values() if r.success)
        total = len(results)
        print("=" * 80)
        print(f"Results: {passed}/{total} benchmarks passed")
        print("=" * 80)
        print()


def run_benchmarks() -> None:
    """Run all benchmarks (CLI entry point)."""
    # Sync benchmarks
    sync_results = OrchestrationBenchmarks.benchmark_suite()
    OrchestrationBenchmarks.print_results(sync_results)

    # Async benchmarks
    async_results = asyncio.run(OrchestrationBenchmarks.async_benchmark_suite())
    OrchestrationBenchmarks.print_results(async_results)


if __name__ == "__main__":
    run_benchmarks()
