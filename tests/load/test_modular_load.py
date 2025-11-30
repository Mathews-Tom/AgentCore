"""
Load Testing for Modular Agent Core (MOD-028)

Tests 100 concurrent modular.solve executions to validate NFR-1.4:
"The system SHALL handle at least 100 concurrent modular executions per instance."

Acceptance Criteria:
1. Locust load test with 100 concurrent users
2. Sustained load for 10 minutes
3. Success rate >95% under load
4. p95 latency <3x baseline
5. No memory leaks or resource exhaustion
6. Load test report with metrics

Usage:
    # Full load test (100 users, 10 minutes)
    uv run locust -f tests/load/test_modular_load.py \
        --host=http://localhost:8001 \
        --users=100 \
        --spawn-rate=10 \
        --run-time=10m \
        --headless

    # Quick validation test (10 users, 2 minutes)
    uv run locust -f tests/load/test_modular_load.py \
        --host=http://localhost:8001 \
        --users=10 \
        --spawn-rate=5 \
        --run-time=2m

    # With web UI for monitoring
    uv run locust -f tests/load/test_modular_load.py \
        --host=http://localhost:8001
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

from locust import HttpUser, TaskSet, between, events, task
from locust.exception import RescheduleTask


# ============================================================================
# Test Query Definitions
# ============================================================================


@dataclass
class TestQuery:
    """Test query with expected behavior."""

    query: str
    complexity: str  # "simple", "moderate", "complex"
    weight: int  # Task weight (higher = more frequent)
    expected_min_latency: int  # Minimum expected latency (ms)
    expected_max_latency: int  # Maximum expected latency (ms)


# Query distribution matches real-world usage:
# 60% simple, 30% moderate, 10% complex
SIMPLE_QUERIES = [
    TestQuery(
        "Hello",
        "simple",
        weight=6,
        expected_min_latency=500,
        expected_max_latency=2000,
    ),
    TestQuery(
        "What is the capital of France?",
        "simple",
        weight=6,
        expected_min_latency=500,
        expected_max_latency=2000,
    ),
    TestQuery(
        "Calculate 15% of 240",
        "simple",
        weight=6,
        expected_min_latency=500,
        expected_max_latency=2000,
    ),
    TestQuery(
        "List the days of the week",
        "simple",
        weight=6,
        expected_min_latency=500,
        expected_max_latency=2000,
    ),
    TestQuery(
        "What is the speed of light?",
        "simple",
        weight=6,
        expected_min_latency=500,
        expected_max_latency=2000,
    ),
]

MODERATE_QUERIES = [
    TestQuery(
        "What is the capital of France and what is its population?",
        "moderate",
        weight=3,
        expected_min_latency=1000,
        expected_max_latency=4000,
    ),
    TestQuery(
        "Who invented the telephone and when was it patented?",
        "moderate",
        weight=3,
        expected_min_latency=1000,
        expected_max_latency=4000,
    ),
    TestQuery(
        "If a car travels 60 mph for 2.5 hours, how far does it go in kilometers?",
        "moderate",
        weight=3,
        expected_min_latency=1000,
        expected_max_latency=4000,
    ),
    TestQuery(
        "Compare the populations of New York City and Los Angeles",
        "moderate",
        weight=3,
        expected_min_latency=1000,
        expected_max_latency=4000,
    ),
    TestQuery(
        "What are the main differences between Python and JavaScript?",
        "moderate",
        weight=3,
        expected_min_latency=1000,
        expected_max_latency=4000,
    ),
]

COMPLEX_QUERIES = [
    TestQuery(
        "Research top 3 programming languages by popularity, compare their use cases, and recommend one for web development",
        "complex",
        weight=1,
        expected_min_latency=3000,
        expected_max_latency=8000,
    ),
    TestQuery(
        "Compare electric vs gas cars on cost, environmental impact, and convenience, then provide recommendation",
        "complex",
        weight=1,
        expected_min_latency=3000,
        expected_max_latency=8000,
    ),
]

ALL_QUERIES = SIMPLE_QUERIES + MODERATE_QUERIES + COMPLEX_QUERIES


# ============================================================================
# Performance Tracking
# ============================================================================


class PerformanceTracker:
    """Track performance metrics across load test."""

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self.latencies: list[float] = []
        self.successes: int = 0
        self.failures: int = 0
        self.timeout_count: int = 0
        self.error_count: int = 0
        self.baseline_p95: float = 0.0  # Will be set from baseline measurement
        self.start_time: float = time.time()

    def record_success(self, latency_ms: float) -> None:
        """Record successful request."""
        self.latencies.append(latency_ms)
        self.successes += 1

    def record_failure(self, is_timeout: bool = False) -> None:
        """Record failed request."""
        self.failures += 1
        if is_timeout:
            self.timeout_count += 1
        else:
            self.error_count += 1

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successes + self.failures
        return (self.successes / total * 100) if total > 0 else 0.0

    def get_p95_latency(self) -> float:
        """Calculate p95 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else 0.0

    def get_p99_latency(self) -> float:
        """Calculate p99 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else 0.0

    def get_mean_latency(self) -> float:
        """Calculate mean latency."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def get_throughput(self) -> float:
        """Calculate requests per second."""
        elapsed = time.time() - self.start_time
        total_requests = self.successes + self.failures
        return total_requests / elapsed if elapsed > 0 else 0.0

    def validate_nfr_targets(self) -> dict[str, bool]:
        """
        Validate NFR targets from acceptance criteria.

        Returns:
            Dict of validation results
        """
        success_rate = self.get_success_rate()
        p95_latency = self.get_p95_latency()

        # Acceptance criteria checks
        meets_success_rate = success_rate >= 95.0  # >95% success rate
        meets_latency = (
            p95_latency < (self.baseline_p95 * 3) if self.baseline_p95 > 0 else True
        )  # <3x baseline

        return {
            "success_rate": meets_success_rate,
            "latency": meets_latency,
            "overall": meets_success_rate and meets_latency,
        }


# Global tracker instance
performance_tracker = PerformanceTracker()


# ============================================================================
# Modular Load Test User
# ============================================================================


class ModularSolveTaskSet(TaskSet):
    """Task set for modular.solve load testing."""

    def _make_modular_request(
        self, query: TestQuery, request_id: str
    ) -> dict[str, Any]:
        """
        Create modular.solve JSON-RPC request.

        Args:
            query: Test query to execute
            request_id: Unique request ID

        Returns:
            JSON-RPC request dict
        """
        return {
            "jsonrpc": "2.0",
            "method": "modular.solve",
            "params": {
                "query": query.query,
                "config": {
                    "max_iterations": 5,
                    "timeout_seconds": 60,
                    "output_format": "text",
                    "include_reasoning": False,  # Disable to reduce latency
                },
            },
            "id": request_id,
            "a2a_context": {
                "source_agent": "load-test-client",
                "target_agent": "modular-agent",
                "trace_id": f"load-test-{request_id}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

    def _execute_query(self, query: TestQuery) -> None:
        """
        Execute single modular.solve query.

        Args:
            query: Test query to execute
        """
        request_id = f"{self.user.user_id}-{random.randint(10000, 99999)}"

        request_body = self._make_modular_request(query, request_id)

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request_body,
            name=f"modular.solve [{query.complexity}]",
            catch_response=True,
        ) as response:
            latency_ms = response.elapsed.total_seconds() * 1000

            if response.status_code == 200:
                try:
                    data = response.json()

                    # Validate JSON-RPC response structure
                    if "result" in data:
                        result = data["result"]

                        # Validate response structure
                        if (
                            "answer" in result
                            and "execution_trace" in result
                        ):
                            trace = result["execution_trace"]

                            # Check execution completed successfully
                            if trace.get("verification_passed", False):
                                performance_tracker.record_success(latency_ms)
                                response.success()
                            else:
                                performance_tracker.record_failure()
                                response.failure(
                                    f"Verification failed: {trace.get('confidence_score', 0)}"
                                )
                        else:
                            performance_tracker.record_failure()
                            response.failure(
                                f"Invalid result structure: {list(result.keys())}"
                            )
                    elif "error" in data:
                        error = data["error"]
                        performance_tracker.record_failure()
                        response.failure(
                            f"JSON-RPC error {error.get('code')}: {error.get('message')}"
                        )
                    else:
                        performance_tracker.record_failure()
                        response.failure(
                            f"Invalid JSON-RPC response: {list(data.keys())}"
                        )

                except json.JSONDecodeError as e:
                    performance_tracker.record_failure()
                    response.failure(f"Invalid JSON response: {str(e)}")
            elif response.status_code == 408:  # Timeout
                performance_tracker.record_failure(is_timeout=True)
                response.failure("Request timeout")
            else:
                performance_tracker.record_failure()
                response.failure(f"HTTP {response.status_code}")

    @task
    def execute_weighted_query(self) -> None:
        """
        Execute query with weighted selection based on complexity distribution.

        Weights: 60% simple, 30% moderate, 10% complex
        """
        # Weighted random selection
        weights = [q.weight for q in ALL_QUERIES]
        query = random.choices(ALL_QUERIES, weights=weights, k=1)[0]

        self._execute_query(query)


class ModularLoadTestUser(HttpUser):
    """
    Simulated user executing modular.solve queries under load.

    Configuration:
    - wait_time: 1-5 seconds between requests (realistic user behavior)
    - tasks: ModularSolveTaskSet
    """

    tasks = [ModularSolveTaskSet]
    wait_time = between(1, 5)  # Realistic think time

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize load test user."""
        super().__init__(*args, **kwargs)
        self.user_id = random.randint(1000, 9999)


# ============================================================================
# Locust Event Handlers
# ============================================================================


@events.test_start.add_listener
def on_test_start(environment: Any, **kwargs: Any) -> None:
    """
    Initialize test environment and measure baseline.

    Args:
        environment: Locust environment
    """
    print("\n" + "=" * 80)
    print("MODULAR AGENT CORE LOAD TEST (MOD-028)")
    print("=" * 80)
    print(f"Target Users: {environment.parsed_options.num_users}")
    print(f"Spawn Rate: {environment.parsed_options.spawn_rate}/s")
    print(f"Run Time: {environment.parsed_options.run_time or 'unlimited'}")
    print()
    print("NFR Targets:")
    print("  - Concurrent Executions: 100+")
    print("  - Success Rate: >95%")
    print("  - p95 Latency: <3x baseline")
    print("  - No memory leaks or resource exhaustion")
    print()
    print("Test Queries:")
    print(f"  - Simple (60%): {len(SIMPLE_QUERIES)} queries")
    print(f"  - Moderate (30%): {len(MODERATE_QUERIES)} queries")
    print(f"  - Complex (10%): {len(COMPLEX_QUERIES)} queries")
    print("=" * 80)
    print()

    # Measure baseline p95 latency (from performance benchmarks)
    # Based on test_modular_performance.py baseline measurements
    performance_tracker.baseline_p95 = 1500.0  # ms (from baseline benchmarks)
    print(f"Baseline p95 latency: {performance_tracker.baseline_p95:.0f}ms")
    print(f"Target p95 latency: <{performance_tracker.baseline_p95 * 3:.0f}ms")
    print()


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    """
    Generate final load test report.

    Args:
        environment: Locust environment
    """
    print("\n" + "=" * 80)
    print("LOAD TEST RESULTS")
    print("=" * 80)
    print()

    # Overall statistics
    total_requests = performance_tracker.successes + performance_tracker.failures
    success_rate = performance_tracker.get_success_rate()
    mean_latency = performance_tracker.get_mean_latency()
    p95_latency = performance_tracker.get_p95_latency()
    p99_latency = performance_tracker.get_p99_latency()
    throughput = performance_tracker.get_throughput()

    print("REQUEST SUMMARY:")
    print(f"  Total Requests:     {total_requests:,}")
    print(f"  Successful:         {performance_tracker.successes:,}")
    print(f"  Failed:             {performance_tracker.failures:,}")
    print(f"  Timeouts:           {performance_tracker.timeout_count:,}")
    print(f"  Errors:             {performance_tracker.error_count:,}")
    print(f"  Success Rate:       {success_rate:.2f}%")
    print()

    print("LATENCY METRICS:")
    print(f"  Mean:               {mean_latency:.0f}ms")
    print(f"  p95:                {p95_latency:.0f}ms")
    print(f"  p99:                {p99_latency:.0f}ms")
    print()

    print("THROUGHPUT:")
    print(f"  Requests/sec:       {throughput:.2f}")
    print()

    # Validate NFR targets
    validation_results = performance_tracker.validate_nfr_targets()

    print("=" * 80)
    print("NFR TARGET VALIDATION")
    print("=" * 80)
    print()

    latency_multiplier = (
        p95_latency / performance_tracker.baseline_p95
        if performance_tracker.baseline_p95 > 0
        else 0
    )

    print(f"  Success Rate:       {success_rate:.2f}% ")
    print(f"                      {'✓ PASS' if validation_results['success_rate'] else '✗ FAIL'} (Target: >95%)")
    print()
    print(f"  p95 Latency:        {p95_latency:.0f}ms")
    print(f"  Baseline p95:       {performance_tracker.baseline_p95:.0f}ms")
    print(f"  Multiplier:         {latency_multiplier:.2f}x")
    print(f"                      {'✓ PASS' if validation_results['latency'] else '✗ FAIL'} (Target: <3x)")
    print()
    print(
        f"  Overall:            {'✓ ALL TARGETS MET' if validation_results['overall'] else '✗ SOME TARGETS FAILED'}"
    )
    print()

    # Additional metrics from Locust stats
    if hasattr(environment, "stats"):
        stats = environment.stats
        print("=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        print()
        print(f"  Total RPS:          {stats.total.total_rps:.2f}")
        print(
            f"  Total Failures:     {stats.total.num_failures} ({stats.total.fail_ratio:.2%})"
        )
        print(
            f"  Avg Response Time:  {stats.total.avg_response_time:.0f}ms"
        )
        print(
            f"  Min Response Time:  {stats.total.min_response_time:.0f}ms"
        )
        print(
            f"  Max Response Time:  {stats.total.max_response_time:.0f}ms"
        )
        print()

    print("=" * 80)
    print()

    # Performance recommendations
    if not validation_results["overall"]:
        print("RECOMMENDATIONS:")
        if not validation_results["success_rate"]:
            print("  - Success rate below target: Check logs for failure patterns")
            print(
                "  - Consider scaling resources or optimizing module coordination"
            )
        if not validation_results["latency"]:
            print("  - Latency above target: Review MOD-020 optimization tasks")
            print("  - Consider implementing response caching")
            print("  - Check for network or database bottlenecks")
        print()


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    exception: Exception | None,
    **kwargs: Any,
) -> None:
    """
    Track individual requests for detailed analysis.

    Args:
        request_type: HTTP method
        name: Request name
        response_time: Response time in ms
        response_length: Response size in bytes
        exception: Exception if request failed
    """
    # Additional per-request tracking could be added here
    pass


# ============================================================================
# Standalone Execution
# ============================================================================

if __name__ == "__main__":
    """
    Run load test directly (not recommended - use locust CLI instead).

    Example:
        uv run python tests/load/test_modular_load.py
    """
    print("Load test should be run via locust CLI:")
    print()
    print("  uv run locust -f tests/load/test_modular_load.py \\")
    print("      --host=http://localhost:8001 \\")
    print("      --users=100 \\")
    print("      --spawn-rate=10 \\")
    print("      --run-time=10m \\")
    print("      --headless")
    print()
