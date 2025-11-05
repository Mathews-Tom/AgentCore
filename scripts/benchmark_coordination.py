#!/usr/bin/env python3
"""Performance benchmark script for Coordination Service.

This script measures the performance characteristics of the coordination service
to validate SLO requirements:
- Signal registration latency: <5ms (p95)
- Routing score retrieval: <2ms (p95)
- Optimal agent selection: <10ms for 100 candidates (p95)
- Throughput: 10,000 signals/sec

Usage:
    uv run python scripts/benchmark_coordination.py
    uv run python scripts/benchmark_coordination.py --agents 1000 --signals 10000
"""

import argparse
import asyncio
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import coordination_service


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    total_operations: int
    duration_seconds: float
    throughput: float  # operations/second
    latencies_ms: list[float]
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


def calculate_percentiles(latencies: list[float]) -> dict[str, float]:
    """Calculate percentile statistics from latency measurements.

    Args:
        latencies: List of latency measurements in milliseconds

    Returns:
        Dictionary with p50, p90, p95, p99 percentiles
    """
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    if len(latencies) == 1:
        val = latencies[0]
        return {"p50": val, "p90": val, "p95": val, "p99": val}

    sorted_latencies = sorted(latencies)
    return {
        "p50": statistics.quantiles(sorted_latencies, n=100)[49],
        "p90": statistics.quantiles(sorted_latencies, n=100)[89],
        "p95": statistics.quantiles(sorted_latencies, n=100)[94],
        "p99": statistics.quantiles(sorted_latencies, n=100)[98],
    }


def benchmark_operation(
    operation_name: str, operation: Callable[[], Any], iterations: int
) -> BenchmarkResult:
    """Benchmark a synchronous operation.

    Args:
        operation_name: Name of the operation being benchmarked
        operation: Function to benchmark
        iterations: Number of iterations to run

    Returns:
        BenchmarkResult with performance statistics
    """
    latencies: list[float] = []

    start_time = time.perf_counter()

    for _ in range(iterations):
        op_start = time.perf_counter()
        operation()
        op_end = time.perf_counter()
        latencies.append((op_end - op_start) * 1000)  # Convert to ms

    end_time = time.perf_counter()
    duration = end_time - start_time

    percentiles = calculate_percentiles(latencies)

    return BenchmarkResult(
        operation=operation_name,
        total_operations=iterations,
        duration_seconds=duration,
        throughput=iterations / duration if duration > 0 else 0,
        latencies_ms=latencies,
        p50_ms=percentiles["p50"],
        p90_ms=percentiles["p90"],
        p95_ms=percentiles["p95"],
        p99_ms=percentiles["p99"],
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
        min_ms=min(latencies) if latencies else 0.0,
        max_ms=max(latencies) if latencies else 0.0,
    )


def print_result(result: BenchmarkResult, slo_p95_ms: float | None = None) -> None:
    """Print benchmark result in formatted table.

    Args:
        result: Benchmark result to print
        slo_p95_ms: Optional SLO threshold for p95 latency
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: {result.operation}")
    print(f"{'='*70}")
    print(f"Total Operations:  {result.total_operations:,}")
    print(f"Duration:          {result.duration_seconds:.3f} seconds")
    print(f"Throughput:        {result.throughput:,.2f} ops/sec")
    print(f"\nLatency Distribution:")
    print(f"  Mean:            {result.mean_ms:.3f} ms")
    print(f"  Min:             {result.min_ms:.3f} ms")
    print(f"  Max:             {result.max_ms:.3f} ms")
    print(f"  p50:             {result.p50_ms:.3f} ms")
    print(f"  p90:             {result.p90_ms:.3f} ms")
    print(f"  p95:             {result.p95_ms:.3f} ms", end="")

    if slo_p95_ms is not None:
        status = "✓ PASS" if result.p95_ms < slo_p95_ms else "✗ FAIL"
        print(f"  (SLO: <{slo_p95_ms}ms) [{status}]")
    else:
        print()

    print(f"  p99:             {result.p99_ms:.3f} ms")


def benchmark_signal_registration(num_agents: int, signals_per_agent: int) -> BenchmarkResult:
    """Benchmark signal registration performance.

    Args:
        num_agents: Number of unique agents to register signals for
        signals_per_agent: Number of signals to register per agent

    Returns:
        BenchmarkResult for signal registration
    """
    coordination_service.clear_state()

    agent_ids = [f"bench-agent-{i:04d}" for i in range(num_agents)]
    signal_types = list(SignalType)
    latencies: list[float] = []

    start_time = time.perf_counter()

    for i, agent_id in enumerate(agent_ids):
        for j in range(signals_per_agent):
            signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=signal_types[j % len(signal_types)],
                value=0.5 + (j * 0.05),
                ttl_seconds=300,
            )

            op_start = time.perf_counter()
            coordination_service.register_signal(signal)
            op_end = time.perf_counter()
            latencies.append((op_end - op_start) * 1000)

    end_time = time.perf_counter()
    duration = end_time - start_time

    total_signals = num_agents * signals_per_agent
    percentiles = calculate_percentiles(latencies)

    return BenchmarkResult(
        operation="Signal Registration",
        total_operations=total_signals,
        duration_seconds=duration,
        throughput=total_signals / duration if duration > 0 else 0,
        latencies_ms=latencies,
        p50_ms=percentiles["p50"],
        p90_ms=percentiles["p90"],
        p95_ms=percentiles["p95"],
        p99_ms=percentiles["p99"],
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
        min_ms=min(latencies) if latencies else 0.0,
        max_ms=max(latencies) if latencies else 0.0,
    )


def benchmark_routing_score_retrieval(num_agents: int) -> BenchmarkResult:
    """Benchmark routing score retrieval performance.

    Args:
        num_agents: Number of agents to retrieve scores for

    Returns:
        BenchmarkResult for score retrieval
    """
    # Pre-populate with signals
    coordination_service.clear_state()
    for i in range(num_agents):
        signal = SensitivitySignal(
            agent_id=f"bench-agent-{i:04d}",
            signal_type=SignalType.LOAD,
            value=0.5,
            ttl_seconds=300,
        )
        coordination_service.register_signal(signal)

    agent_ids = [f"bench-agent-{i:04d}" for i in range(num_agents)]

    def get_score_op() -> None:
        for agent_id in agent_ids:
            coordination_service.compute_routing_score(agent_id)

    result = benchmark_operation(
        "Routing Score Retrieval", get_score_op, iterations=100
    )

    # Adjust for per-retrieval latency
    total_retrievals = num_agents * 100
    result.total_operations = total_retrievals
    result.latencies_ms = [lat / num_agents for lat in result.latencies_ms]
    percentiles = calculate_percentiles(result.latencies_ms)
    result.p50_ms = percentiles["p50"]
    result.p90_ms = percentiles["p90"]
    result.p95_ms = percentiles["p95"]
    result.p99_ms = percentiles["p99"]
    result.mean_ms = statistics.mean(result.latencies_ms)

    return result


def benchmark_optimal_agent_selection(num_candidates: int) -> BenchmarkResult:
    """Benchmark optimal agent selection performance.

    Args:
        num_candidates: Number of candidate agents to select from

    Returns:
        BenchmarkResult for agent selection
    """
    # Pre-populate with signals
    coordination_service.clear_state()
    for i in range(num_candidates):
        signal = SensitivitySignal(
            agent_id=f"bench-agent-{i:04d}",
            signal_type=SignalType.LOAD,
            value=0.3 + (i * 0.001),  # Varied load
            ttl_seconds=300,
        )
        coordination_service.register_signal(signal)

    candidates = [f"bench-agent-{i:04d}" for i in range(num_candidates)]

    def select_agent_op() -> None:
        coordination_service.select_optimal_agent(candidates)

    result = benchmark_operation(
        f"Optimal Agent Selection ({num_candidates} candidates)",
        select_agent_op,
        iterations=1000,
    )

    return result


def benchmark_throughput(target_signals_per_sec: int, duration_seconds: int) -> BenchmarkResult:
    """Benchmark sustained throughput under load.

    Args:
        target_signals_per_sec: Target signals per second to achieve
        duration_seconds: Duration to sustain the load

    Returns:
        BenchmarkResult for throughput test
    """
    coordination_service.clear_state()

    signal_types = list(SignalType)
    agent_counter = 0
    signal_counter = 0

    interval = 1.0 / target_signals_per_sec  # Time between signals
    latencies: list[float] = []

    start_time = time.perf_counter()
    end_target = start_time + duration_seconds

    while time.perf_counter() < end_target:
        op_start = time.perf_counter()

        signal = SensitivitySignal(
            agent_id=f"load-agent-{agent_counter % 1000:04d}",
            signal_type=signal_types[signal_counter % len(signal_types)],
            value=0.5,
            ttl_seconds=60,
        )
        coordination_service.register_signal(signal)

        op_end = time.perf_counter()
        latencies.append((op_end - op_start) * 1000)

        agent_counter += 1
        signal_counter += 1

        # Sleep to maintain target rate
        elapsed = op_end - op_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    actual_duration = time.perf_counter() - start_time
    percentiles = calculate_percentiles(latencies)

    return BenchmarkResult(
        operation=f"Sustained Throughput ({target_signals_per_sec} signals/sec target)",
        total_operations=signal_counter,
        duration_seconds=actual_duration,
        throughput=signal_counter / actual_duration,
        latencies_ms=latencies,
        p50_ms=percentiles["p50"],
        p90_ms=percentiles["p90"],
        p95_ms=percentiles["p95"],
        p99_ms=percentiles["p99"],
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
        min_ms=min(latencies) if latencies else 0.0,
        max_ms=max(latencies) if latencies else 0.0,
    )


def main() -> None:
    """Run all coordination service benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark coordination service performance")
    parser.add_argument(
        "--agents", type=int, default=1000, help="Number of agents (default: 1000)"
    )
    parser.add_argument(
        "--signals-per-sec",
        type=int,
        default=10000,
        help="Target signals per second for throughput test (default: 10000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration in seconds for throughput test (default: 10)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("COORDINATION SERVICE PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Benchmark 1: Signal Registration
    print("\n[1/4] Running signal registration benchmark...")
    result_registration = benchmark_signal_registration(
        num_agents=args.agents, signals_per_agent=5
    )
    print_result(result_registration, slo_p95_ms=5.0)

    # Benchmark 2: Routing Score Retrieval
    print("\n[2/4] Running routing score retrieval benchmark...")
    result_scores = benchmark_routing_score_retrieval(num_agents=args.agents)
    print_result(result_scores, slo_p95_ms=2.0)

    # Benchmark 3: Optimal Agent Selection
    print("\n[3/4] Running optimal agent selection benchmark...")
    result_selection = benchmark_optimal_agent_selection(num_candidates=100)
    print_result(result_selection, slo_p95_ms=10.0)

    # Benchmark 4: Sustained Throughput
    print("\n[4/4] Running sustained throughput benchmark...")
    result_throughput = benchmark_throughput(
        target_signals_per_sec=args.signals_per_sec, duration_seconds=args.duration
    )
    print_result(result_throughput)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Signal Registration p95:        {result_registration.p95_ms:.3f} ms (SLO: <5ms)")
    print(f"Routing Score Retrieval p95:    {result_scores.p95_ms:.3f} ms (SLO: <2ms)")
    print(f"Optimal Agent Selection p95:    {result_selection.p95_ms:.3f} ms (SLO: <10ms)")
    print(
        f"Sustained Throughput:           {result_throughput.throughput:,.2f} signals/sec "
        f"(target: {args.signals_per_sec:,})"
    )

    # Overall SLO check
    slo_pass = (
        result_registration.p95_ms < 5.0
        and result_scores.p95_ms < 2.0
        and result_selection.p95_ms < 10.0
    )

    print(f"\nOverall SLO Status: {'✓ PASS' if slo_pass else '✗ FAIL'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
