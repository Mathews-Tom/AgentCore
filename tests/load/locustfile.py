"""
Load Testing for A2A Protocol Layer

Locust load testing scenario for testing 1000+ concurrent connections.

Usage:
    locust -f tests/load/locustfile.py --host=http://localhost:8001
"""

import random
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask


class A2AProtocolUser(HttpUser):
    """Simulated A2A protocol user for load testing."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    agent_id = None
    task_id = None

    def on_start(self):
        """Setup: Register agent when user starts."""
        self.agent_id = f"load-test-agent-{self.environment.runner.user_count}-{random.randint(1000, 9999)}"

        agent_card = {
            "agent_id": self.agent_id,
            "name": f"Load Test Agent {self.agent_id}",
            "version": "1.0.0",
            "status": "active",
            "description": "Load testing agent",
            "capabilities": ["text-generation", "summarization"],
            "endpoints": [
                {
                    "url": f"http://localhost:8080/{self.agent_id}",
                    "type": "https",
                    "protocols": ["jsonrpc-2.0"]
                }
            ],
            "authentication": {
                "type": "jwt",
                "metadata": {}
            }
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
                if "result" in data and data["result"].get("success"):
                    response.success()
                else:
                    response.failure(f"Agent registration failed: {data}")
            else:
                response.failure(f"HTTP {response.status_code}")

    def on_stop(self):
        """Cleanup: Unregister agent when user stops."""
        if self.agent_id:
            request = {
                "jsonrpc": "2.0",
                "method": "agent.unregister",
                "params": {"agent_id": self.agent_id},
                "id": "999"
            }
            self.client.post("/api/v1/jsonrpc", json=request)

    @task(5)
    def ping_agent(self):
        """Agent heartbeat - most frequent operation."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.ping",
            "params": {"agent_id": self.agent_id},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="agent.ping",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def discover_agents(self):
        """Discover agents by capability."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.discover",
            "params": {"capabilities": ["text-generation"]},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="agent.discover",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def create_task(self):
        """Create a task."""
        self.task_id = f"load-test-task-{random.randint(10000, 99999)}"

        task_def = {
            "task_id": self.task_id,
            "name": f"Load Test Task {self.task_id}",
            "description": "Load testing task",
            "required_capabilities": ["text-generation"],
            "parameters": {"input": "Test input", "max_tokens": 100},
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
            name="task.create",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def query_tasks(self):
        """Query tasks by status."""
        statuses = ["pending", "running", "completed"]
        status = random.choice(statuses)

        request = {
            "jsonrpc": "2.0",
            "method": "task.query",
            "params": {"status": status},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="task.query",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_health_stats(self):
        """Get health monitoring stats."""
        request = {
            "jsonrpc": "2.0",
            "method": "health.get_stats",
            "params": {},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="health.get_stats",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def route_message(self):
        """Route a message to an agent."""
        message_envelope = {
            "message_id": f"msg-{random.randint(10000, 99999)}",
            "sender_id": self.agent_id,
            "recipient_id": "any",
            "message_type": "request",
            "payload": {"action": "generate", "input": "test"},
            "timestamp": "2025-09-30T00:00:00Z"
        }

        request = {
            "jsonrpc": "2.0",
            "method": "route.message",
            "params": {
                "envelope": message_envelope,
                "required_capabilities": ["text-generation"]
            },
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="route.message",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class WebSocketUser(HttpUser):
    """Test WebSocket connections for events."""

    wait_time = between(5, 15)

    @task
    def health_check(self):
        """Simple health check to maintain connection."""
        self.client.get("/api/v1/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log when test starts."""
    print(f"Load test starting with {environment.parsed_options.num_users} users")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test results."""
    print(f"Load test completed")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Failed requests: {environment.stats.total.num_failures}")
    print(f"RPS: {environment.stats.total.total_rps}")
    print(f"Avg response time: {environment.stats.total.avg_response_time}, ms")