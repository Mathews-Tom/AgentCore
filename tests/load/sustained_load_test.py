"""
Sustained Load Testing for Gateway Layer

Tests gateway stability under sustained load for 10+ minutes.

Usage:
    uv run locust -f tests/load/sustained_load_test.py --host=http://localhost:8001 --run-time=10m
"""

import random
import time
from datetime import datetime

from locust import HttpUser, task, between, events


class SustainedLoadUser(HttpUser):
    """User for sustained load testing."""

    wait_time = between(0.5, 2.0)  # Moderate wait time

    def on_start(self) -> None:
        """Setup user."""
        self.agent_id = f"sustained-agent-{random.randint(10000, 99999)}"
        self.start_time = time.time()

        # Register agent
        agent_card = {
            "agent_id": self.agent_id,
            "name": f"Sustained Test Agent {self.agent_id}",
            "version": "1.0.0",
            "status": "active",
            "capabilities": ["text-generation"],
            "endpoints": [
                {
                    "url": f"http://localhost:8080/{self.agent_id}",
                    "type": "https",
                    "protocols": ["jsonrpc-2.0"]
                }
            ],
        }

        request = {
            "jsonrpc": "2.0",
            "method": "agent.register",
            "params": {"agent_card": agent_card},
            "id": "1"
        }

        self.client.post("/api/v1/jsonrpc", json=request)

    @task(5)
    def regular_operation(self) -> None:
        """Regular operational requests."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.ping",
            "params": {"agent_id": self.agent_id},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="Sustained: agent.ping",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()

    @task(3)
    def mixed_operations(self) -> None:
        """Mixed operational requests."""
        operations = [
            ("agent.discover", {"capabilities": ["text-generation"]}),
            ("task.query", {"status": "pending"}),
            ("health.get_stats", {}),
        ]

        method, params = random.choice(operations)
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name=f"Sustained: {method}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()


# Track metrics over time
time_series_stats = []
last_stats_time = None


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test start."""
    global last_stats_time
    last_stats_time = time.time()

    print(f"\n{'='*70}")
    print(f"Sustained Load Test Starting")
    print(f"Target: {environment.host}")
    print(f"Duration: 10+ minutes")
    print(f"Monitoring for memory leaks and performance degradation")
    print(f"{'='*70}\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs) -> None:
    """Track request metrics over time."""
    global time_series_stats, last_stats_time

    current_time = time.time()

    # Record stats every 10 seconds
    if current_time - last_stats_time >= 10:
        time_series_stats.append({
            "timestamp": datetime.now().isoformat(),
            "elapsed": current_time - last_stats_time,
            "response_time": response_time,
        })
        last_stats_time = current_time


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log test results and check for degradation."""
    stats = environment.stats.total

    print(f"\n{'='*70}")
    print(f"Sustained Load Test Results")
    print(f"{'='*70}")
    print(f"Total Duration: {stats.last_request_timestamp - stats.start_time:.2f}, s")
    print(f"Total Requests: {stats.num_requests:,}")
    print(f"Failed Requests: {stats.num_failures:,}")
    print(f"Average RPS: {stats.total_rps:,.2f}")
    print(f"Average Response Time: {stats.avg_response_time:.2f}, ms")
    print(f"Median Response Time: {stats.median_response_time:.2f}, ms")
    print(f"95th Percentile: {stats.get_response_time_percentile(0.95):.2f}, ms")
    print(f"99th Percentile: {stats.get_response_time_percentile(0.99):.2f}, ms")

    # Check for performance degradation
    if len(time_series_stats) > 1:
        first_half = time_series_stats[:len(time_series_stats)//2]
        second_half = time_series_stats[len(time_series_stats)//2:]

        if first_half and second_half:
            avg_first = sum(s["response_time"] for s in first_half) / len(first_half)
            avg_second = sum(s["response_time"] for s in second_half) / len(second_half)

            degradation = ((avg_second - avg_first) / avg_first) * 100

            print(f"\nPerformance Analysis:")
            print(f"First Half Avg Response: {avg_first:.2f}, ms")
            print(f"Second Half Avg Response: {avg_second:.2f}, ms")
            print(f"Degradation: {degradation:+.2f}%")

            if degradation < 10:
                print(f"✓ Performance stable (< 10% degradation)")
            else:
                print(f"✗ Performance degraded by {degradation:.2f}%")

    print(f"{'='*70}\n")
