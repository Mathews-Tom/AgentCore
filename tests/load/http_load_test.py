"""
HTTP Load Testing for Gateway Layer

Tests gateway performance with high-throughput HTTP requests targeting 60,000+ req/sec.

Usage:
    uv run locust -f tests/load/http_load_test.py --host=http://localhost:8001
"""

import random
from locust import HttpUser, task, between, events


class GatewayAPIUser(HttpUser):
    """Gateway API load testing user."""

    wait_time = between(0.01, 0.05)  # Very short wait time for high throughput

    def on_start(self) -> None:
        """Setup: Register agent."""
        self.agent_id = f"load-agent-{random.randint(10000, 99999)}"

        agent_card = {
            "agent_id": self.agent_id,
            "name": f"Load Test Agent {self.agent_id}",
            "version": "1.0.0",
            "status": "active",
            "description": "High-throughput load testing agent",
            "capabilities": ["text-generation", "analysis"],
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

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    response.success()

    @task(10)
    def health_check(self) -> None:
        """High-frequency health check."""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(8)
    def agent_ping(self) -> None:
        """Agent heartbeat ping."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.ping",
            "params": {"agent_id": self.agent_id},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="/api/v1/jsonrpc (agent.ping)",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()

    @task(5)
    def agent_discovery(self) -> None:
        """Agent discovery by capability."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.discover",
            "params": {"capabilities": ["text-generation"]},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="/api/v1/jsonrpc (agent.discover)",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()

    @task(3)
    def task_creation(self) -> None:
        """Create task for agent."""
        task_def = {
            "task_id": f"task-{random.randint(10000, 99999)}",
            "name": "Load Test Task",
            "description": "Performance testing task",
            "required_capabilities": ["text-generation"],
            "parameters": {"input": "test", "max_tokens": 100},
            "priority": random.randint(1, 10)
        }

        request = {
            "jsonrpc": "2.0",
            "method": "task.create",
            "params": {"task_definition": task_def},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="/api/v1/jsonrpc (task.create)",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()

    @task(2)
    def task_query(self) -> None:
        """Query tasks by status."""
        request = {
            "jsonrpc": "2.0",
            "method": "task.query",
            "params": {"status": random.choice(["pending", "running", "completed"])},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="/api/v1/jsonrpc (task.query)",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test start."""
    print(f"\n{'='*70}")
    print(f"HTTP Load Test Starting")
    print(f"Target: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users if environment.parsed_options else 'N/A'}")
    print(f"Target: 60,000+ requests per second")
    print(f"{'='*70}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log test results."""
    stats = environment.stats.total
    print(f"\n{'='*70}")
    print(f"HTTP Load Test Results")
    print(f"{'='*70}")
    print(f"Total Requests: {stats.num_requests:,}")
    print(f"Failed Requests: {stats.num_failures:,}")
    print(f"Requests Per Second: {stats.total_rps:,.2f}")
    print(f"Average Response Time: {stats.avg_response_time:.2f}, ms")
    print(f"Median Response Time: {stats.median_response_time:.2f}, ms")
    print(f"95th Percentile: {stats.get_response_time_percentile(0.95):.2f}, ms")
    print(f"99th Percentile: {stats.get_response_time_percentile(0.99):.2f}, ms")
    print(f"{'='*70}")

    # Check if target RPS achieved
    if stats.total_rps >= 60000:
        print(f"✓ Target 60,000+ req/sec ACHIEVED!")
    else:
        print(f"✗ Target 60,000+ req/sec not reached (got {stats.total_rps:,.2f})")
    print(f"{'='*70}\n")
