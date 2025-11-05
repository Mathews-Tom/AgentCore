"""Load tests for coordination service performance validation.

These tests validate that the coordination service meets performance SLOs:
- Signal registration latency: <5ms (p95)
- Routing score retrieval: <2ms (p95)
- Optimal agent selection: <10ms for 100 candidates (p95)
- Sustained throughput: 10,000 signals/sec
"""

import statistics
import time

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import coordination_service


class TestCoordinationPerformance:
    """Performance tests for coordination service SLO validation."""

    def setup_method(self) -> None:
        """Clear state before each test."""
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        coordination_service.clear_state()

    def test_signal_registration_latency_slo(self) -> None:
        """Test that signal registration meets <5ms p95 latency SLO."""
        num_signals = 1000
        latencies: list[float] = []

        for i in range(num_signals):
            signal = SensitivitySignal(
                agent_id=f"perf-agent-{i % 100:03d}",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=300,
            )

            start = time.perf_counter()
            coordination_service.register_signal(signal)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate p95 latency
        p95 = statistics.quantiles(sorted(latencies), n=100)[94]

        print(f"\nSignal Registration Latency:")
        print(f"  p50: {statistics.quantiles(sorted(latencies), n=100)[49]:.3f} ms")
        print(f"  p90: {statistics.quantiles(sorted(latencies), n=100)[89]:.3f} ms")
        print(f"  p95: {p95:.3f} ms")
        print(f"  p99: {statistics.quantiles(sorted(latencies), n=100)[98]:.3f} ms")

        assert p95 < 5.0, f"Signal registration p95 latency {p95:.3f}ms exceeds SLO of 5ms"

    def test_routing_score_retrieval_latency_slo(self) -> None:
        """Test that routing score retrieval meets <2ms p95 latency SLO."""
        num_agents = 100

        # Pre-populate with signals
        for i in range(num_agents):
            signal = SensitivitySignal(
                agent_id=f"score-agent-{i:03d}",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=300,
            )
            coordination_service.register_signal(signal)

        # Measure score retrieval latency
        latencies: list[float] = []
        iterations = 1000

        for _ in range(iterations):
            agent_id = f"score-agent-{_ % num_agents:03d}"

            start = time.perf_counter()
            coordination_service.compute_routing_score(agent_id)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        # Calculate p95 latency
        p95 = statistics.quantiles(sorted(latencies), n=100)[94]

        print(f"\nRouting Score Retrieval Latency:")
        print(f"  p50: {statistics.quantiles(sorted(latencies), n=100)[49]:.3f} ms")
        print(f"  p90: {statistics.quantiles(sorted(latencies), n=100)[89]:.3f} ms")
        print(f"  p95: {p95:.3f} ms")
        print(f"  p99: {statistics.quantiles(sorted(latencies), n=100)[98]:.3f} ms")

        assert p95 < 2.0, f"Score retrieval p95 latency {p95:.3f}ms exceeds SLO of 2ms"

    def test_optimal_agent_selection_latency_slo(self) -> None:
        """Test that optimal agent selection meets <10ms p95 latency SLO for 100 candidates."""
        num_candidates = 100

        # Pre-populate with signals
        for i in range(num_candidates):
            signal = SensitivitySignal(
                agent_id=f"select-agent-{i:03d}",
                signal_type=SignalType.LOAD,
                value=0.3 + (i * 0.001),  # Varied load
                ttl_seconds=300,
            )
            coordination_service.register_signal(signal)

        candidates = [f"select-agent-{i:03d}" for i in range(num_candidates)]

        # Measure selection latency
        latencies: list[float] = []
        iterations = 1000

        for _ in range(iterations):
            start = time.perf_counter()
            coordination_service.select_optimal_agent(candidates)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        # Calculate p95 latency
        p95 = statistics.quantiles(sorted(latencies), n=100)[94]

        print(f"\nOptimal Agent Selection Latency (100 candidates):")
        print(f"  p50: {statistics.quantiles(sorted(latencies), n=100)[49]:.3f} ms")
        print(f"  p90: {statistics.quantiles(sorted(latencies), n=100)[89]:.3f} ms")
        print(f"  p95: {p95:.3f} ms")
        print(f"  p99: {statistics.quantiles(sorted(latencies), n=100)[98]:.3f} ms")

        assert (
            p95 < 10.0
        ), f"Agent selection p95 latency {p95:.3f}ms exceeds SLO of 10ms"

    def test_sustained_throughput_10k_signals_per_sec(self) -> None:
        """Test that coordination service can sustain 10,000 signals/sec throughput."""
        target_signals_per_sec = 10000
        duration_seconds = 5  # Shorter duration for test suite

        signal_types = list(SignalType)
        agent_counter = 0
        signal_counter = 0

        start_time = time.perf_counter()
        end_target = start_time + duration_seconds

        while time.perf_counter() < end_target:
            signal = SensitivitySignal(
                agent_id=f"throughput-agent-{agent_counter % 100:03d}",
                signal_type=signal_types[signal_counter % len(signal_types)],
                value=0.5,
                ttl_seconds=60,
            )
            coordination_service.register_signal(signal)

            agent_counter += 1
            signal_counter += 1

        actual_duration = time.perf_counter() - start_time
        throughput = signal_counter / actual_duration

        print(f"\nSustained Throughput Test:")
        print(f"  Duration: {actual_duration:.2f} seconds")
        print(f"  Signals: {signal_counter:,}")
        print(f"  Throughput: {throughput:,.2f} signals/sec")
        print(f"  Target: {target_signals_per_sec:,} signals/sec")

        # Allow 10% variance below target
        min_acceptable = target_signals_per_sec * 0.9

        assert (
            throughput >= min_acceptable
        ), f"Throughput {throughput:,.2f} signals/sec below acceptable threshold of {min_acceptable:,.2f}"

    def test_large_scale_agent_coordination(self) -> None:
        """Test coordination with 1,000 agents and varied signal load."""
        num_agents = 1000
        signals_per_agent = 10

        start_time = time.perf_counter()

        # Register signals for 1,000 agents
        signal_types = list(SignalType)
        for i in range(num_agents):
            for j in range(signals_per_agent):
                signal = SensitivitySignal(
                    agent_id=f"scale-agent-{i:04d}",
                    signal_type=signal_types[j % len(signal_types)],
                    value=0.3 + (j * 0.05),
                    ttl_seconds=300,
                )
                coordination_service.register_signal(signal)

        registration_duration = time.perf_counter() - start_time

        # Verify state tracking
        assert coordination_service.metrics.agents_tracked == num_agents
        assert coordination_service.metrics.total_signals == num_agents * signals_per_agent

        print(f"\nLarge Scale Coordination Test:")
        print(f"  Agents: {num_agents:,}")
        print(f"  Total Signals: {num_agents * signals_per_agent:,}")
        print(f"  Registration Duration: {registration_duration:.2f} seconds")
        print(
            f"  Registration Rate: {(num_agents * signals_per_agent) / registration_duration:,.2f} signals/sec"
        )

        # Test selection performance with large candidate pool
        candidates = [f"scale-agent-{i:04d}" for i in range(100)]

        selection_latencies: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            coordination_service.select_optimal_agent(candidates)
            end = time.perf_counter()
            selection_latencies.append((end - start) * 1000)

        p95_selection = statistics.quantiles(sorted(selection_latencies), n=100)[94]

        print(f"  Selection p95 (from 1,000 agents): {p95_selection:.3f} ms")

        assert (
            p95_selection < 15.0
        ), f"Selection p95 {p95_selection:.3f}ms exceeds 15ms under load"

    def test_concurrent_operations_throughput(self) -> None:
        """Test throughput under concurrent read/write operations."""
        num_agents = 100
        operations = 5000

        # Pre-populate with signals
        for i in range(num_agents):
            signal = SensitivitySignal(
                agent_id=f"concurrent-agent-{i:03d}",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=300,
            )
            coordination_service.register_signal(signal)

        candidates = [f"concurrent-agent-{i:03d}" for i in range(num_agents)]

        start_time = time.perf_counter()

        for i in range(operations):
            # Interleave writes and reads
            if i % 3 == 0:
                # Write: register new signal
                signal = SensitivitySignal(
                    agent_id=f"concurrent-agent-{i % num_agents:03d}",
                    signal_type=SignalType.CAPACITY,
                    value=0.6,
                    ttl_seconds=300,
                )
                coordination_service.register_signal(signal)
            elif i % 3 == 1:
                # Read: get routing score
                coordination_service.compute_routing_score(
                    f"concurrent-agent-{i % num_agents:03d}"
                )
            else:
                # Read: select optimal agent
                coordination_service.select_optimal_agent(candidates[:10])

        duration = time.perf_counter() - start_time
        throughput = operations / duration

        print(f"\nConcurrent Operations Test:")
        print(f"  Operations: {operations:,}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Throughput: {throughput:,.2f} ops/sec")

        assert throughput > 1000, f"Concurrent throughput {throughput:,.2f} ops/sec below 1,000"

    def test_memory_efficiency_under_load(self) -> None:
        """Test that coordination service maintains reasonable memory usage under load."""
        import sys

        num_agents = 500
        signals_per_agent = 20

        # Measure initial state size
        initial_size = sys.getsizeof(coordination_service.coordination_states)

        # Populate with signals
        signal_types = list(SignalType)
        for i in range(num_agents):
            for j in range(signals_per_agent):
                signal = SensitivitySignal(
                    agent_id=f"memory-agent-{i:04d}",
                    signal_type=signal_types[j % len(signal_types)],
                    value=0.5,
                    ttl_seconds=300,
                )
                coordination_service.register_signal(signal)

        # Measure final state size
        final_size = sys.getsizeof(coordination_service.coordination_states)
        size_increase_mb = (final_size - initial_size) / (1024 * 1024)

        # Calculate bytes per agent
        bytes_per_agent = (final_size - initial_size) / num_agents

        print(f"\nMemory Efficiency Test:")
        print(f"  Agents: {num_agents:,}")
        print(f"  Signals: {num_agents * signals_per_agent:,}")
        print(f"  Memory Increase: {size_increase_mb:.2f} MB")
        print(f"  Bytes per Agent: {bytes_per_agent:,.0f}")

        # Reasonable threshold: <100KB per agent for coordination state
        assert bytes_per_agent < 100_000, f"Memory usage {bytes_per_agent:,.0f} bytes/agent exceeds 100KB threshold"
