"""
Burst Traffic Load Testing for Gateway Layer

Tests gateway behavior under sudden traffic spikes and burst patterns.

Usage:
    uv run locust -f tests/load/burst_traffic_test.py --host=http://localhost:8001
"""

import random
import time

from locust import HttpUser, task, between, events


class BurstTrafficUser(HttpUser):
    """User simulating burst traffic patterns."""

    # Variable wait time to create burst patterns
    wait_time = between(0.01, 5.0)

    def on_start(self) -> None:
        """Setup user."""
        self.agent_id = f"burst-agent-{random.randint(10000, 99999)}"
        self.burst_mode = False
        self.last_burst = time.time()

        # Register agent
        agent_card = {
            "agent_id": self.agent_id,
            "name": f"Burst Test Agent {self.agent_id}",
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

    @task
    def burst_request(self) -> None:
        """Generate burst traffic pattern."""
        # Simulate burst every 30 seconds
        current_time = time.time()
        if current_time - self.last_burst > 30:
            self.burst_mode = True
            self.last_burst = current_time

        # In burst mode, send rapid requests
        if self.burst_mode:
            # Send 10 rapid requests
            for _ in range(10):
                request = {
                    "jsonrpc": "2.0",
                    "method": "agent.ping",
                    "params": {"agent_id": self.agent_id},
                    "id": str(random.randint(1, 10000))
                }

                with self.client.post(
                    "/api/v1/jsonrpc",
                    json=request,
                    name="Burst: agent.ping",
                    catch_response=True
                ) as response:
                    if response.status_code == 200:
                        response.success()
                    elif response.status_code == 429:
                        # Rate limited - expected during burst
                        response.success()

            self.burst_mode = False
        else:
            # Normal traffic
            request = {
                "jsonrpc": "2.0",
                "method": "agent.discover",
                "params": {"capabilities": ["text-generation"]},
                "id": str(random.randint(1, 10000))
            }

            with self.client.post(
                "/api/v1/jsonrpc",
                json=request,
                name="Normal: agent.discover",
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    response.success()


# Track burst statistics
burst_stats = {
    "total_bursts": 0,
    "rate_limited": 0,
    "successful_burst_requests": 0,
}


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test start."""
    print(f"\n{'='*70}")
    print(f"Burst Traffic Load Test Starting")
    print(f"Target: {environment.host}")
    print(f"Testing sudden traffic spikes and rate limiting")
    print(f"{'='*70}\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, exception, **kwargs) -> None:
    """Track burst statistics."""
    if "Burst:" in name:
        if response and response.status_code == 429:
            burst_stats["rate_limited"] += 1
        elif response and response.status_code == 200:
            burst_stats["successful_burst_requests"] += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log burst test results."""
    stats = environment.stats.total

    print(f"\n{'='*70}")
    print(f"Burst Traffic Test Results")
    print(f"{'='*70}")
    print(f"Total Requests: {stats.num_requests:,}")
    print(f"Failed Requests: {stats.num_failures:,}")
    print(f"Successful Burst Requests: {burst_stats['successful_burst_requests']:,}")
    print(f"Rate Limited Requests: {burst_stats['rate_limited']:,}")
    print(f"Average Response Time: {stats.avg_response_time:.2f}ms")
    print(f"Peak RPS: {stats.max_requests_per_second:,.2f}")

    # Check rate limiting behavior
    if burst_stats["rate_limited"] > 0:
        print(f"\n✓ Rate limiting activated during bursts")
    else:
        print(f"\n✗ No rate limiting detected (may indicate misconfiguration)")

    print(f"{'='*70}\n")
