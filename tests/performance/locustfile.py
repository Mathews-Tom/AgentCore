"""
Locust Load Testing for Orchestration Engine

Load tests for ORCH-010 acceptance criteria:
- 100,000+ events/second processing
- Linear scaling validation
- Coordination latency testing

Usage:
    locust -f tests/performance/locustfile.py --host=http://localhost:8002
"""

from __future__ import annotations

import json
import random
import time
import uuid

from locust import HttpUser, between, task


class OrchestrationUser(HttpUser):
    """
    Simulated user for orchestration engine load testing.

    Tests workflow creation, execution, and event processing under load.
    """

    wait_time = between(0.1, 0.5)  # 100-500ms between requests

    def on_start(self) -> None:
        """Initialize user session."""
        self.workflow_ids: list[str] = []
        self.execution_ids: list[str] = []

    @task(3)
    def create_workflow(self) -> None:
        """
        Create workflow definition.

        Weight: 3 (most common operation)
        """
        workflow_id = str(uuid.uuid4())

        # Generate random workflow
        node_count = random.randint(10, 100)
        workflow = {
            "workflow_id": workflow_id,
            "name": f"load_test_workflow_{workflow_id[:8]}",
            "version": "1.0",
            "orchestration_pattern": random.choice(
                ["supervisor", "hierarchical", "handoff", "swarm"]
            ),
            "agents": self._generate_agents(node_count),
            "tasks": self._generate_tasks(node_count),
        }

        with self.client.post(
            "/api/v1/workflows",
            json=workflow,
            catch_response=True,
            name="create_workflow"
        ) as response:
            if response.status_code == 201:
                self.workflow_ids.append(workflow_id)
                response.success()
            else:
                response.failure(f"Failed to create workflow: {response.status_code}")

    @task(2)
    def execute_workflow(self) -> None:
        """
        Execute workflow.

        Weight: 2 (common operation)
        """
        if not self.workflow_ids:
            return

        workflow_id = random.choice(self.workflow_ids)

        execution_request = {
            "workflow_id": workflow_id,
            "input_data": {"test_data": "load_test"},
            "execution_options": {
                "timeout": 300,
                "retry_policy": "exponential_backoff",
            },
        }

        start_time = time.time()

        with self.client.post(
            f"/api/v1/workflows/{workflow_id}/execute",
            json=execution_request,
            catch_response=True,
            name="execute_workflow"
        ) as response:
            if response.status_code == 200:
                execution_id = response.json().get("execution_id")
                if execution_id:
                    self.execution_ids.append(execution_id)

                # Measure coordination latency (ORCH-010 acceptance: <100ms)
                latency_ms = (time.time() - start_time) * 1000
                if latency_ms > 100:
                    response.failure(
                        f"Coordination latency {latency_ms:.1f}, ms exceeds 100ms target"
                    )
                else:
                    response.success()
            else:
                response.failure(f"Failed to execute workflow: {response.status_code}")

    @task(5)
    def get_workflow_status(self) -> None:
        """
        Query workflow status.

        Weight: 5 (most frequent operation - monitoring)
        """
        if not self.workflow_ids:
            return

        workflow_id = random.choice(self.workflow_ids)

        with self.client.get(
            f"/api/v1/workflows/{workflow_id}/status",
            catch_response=True,
            name="get_workflow_status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get status: {response.status_code}")

    @task(1)
    def publish_event_batch(self) -> None:
        """
        Publish batch of orchestration events.

        Weight: 1 (periodic batch publishing)
        Tests event processing throughput.
        """
        batch_size = 1000  # Publish 1000 events per batch

        events = [
            {
                "event_id": str(uuid.uuid4()),
                "event_type": random.choice([
                    "task_started",
                    "task_completed",
                    "agent_assigned",
                    "workflow_state_changed",
                ]),
                "workflow_id": random.choice(self.workflow_ids) if self.workflow_ids else str(uuid.uuid4()),
                "timestamp": time.time(),
                "data": {"test": "load_test"},
            }
            for _ in range(batch_size)
        ]

        start_time = time.time()

        with self.client.post(
            "/api/v1/events/batch",
            json={"events": events},
            catch_response=True,
            name="publish_event_batch"
        ) as response:
            if response.status_code == 200:
                duration = time.time() - start_time
                throughput = batch_size / duration if duration > 0 else 0

                # Check if we're meeting 100k events/sec target
                # (per batch, extrapolated)
                if throughput < 100000:
                    response.failure(
                        f"Event throughput {throughput:,.0f} events/sec below 100k target"
                    )
                else:
                    response.success()
            else:
                response.failure(f"Failed to publish events: {response.status_code}")

    def _generate_agents(self, count: int) -> dict[str, dict[str, list[str]]]:
        """Generate agent requirements for workflow."""
        agents = {}
        for i in range(min(count, 10)):  # Max 10 agent types
            agents[f"agent_{i}"] = {
                "type": f"agent_type_{i % 5}",
                "capabilities": [f"cap_{j}" for j in range(3)],
            }
        return agents

    def _generate_tasks(self, count: int) -> list[dict]:
        """Generate task definitions for workflow."""
        tasks = []
        for i in range(count):
            task = {
                "task_id": f"task_{i}",
                "agent_role": f"agent_{i % 10}",
                "depends_on": [f"task_{i-1}"] if i > 0 else [],
                "parallel": random.random() > 0.7,  # 30% parallel tasks
            }
            tasks.append(task)
        return tasks


class HighThroughputUser(HttpUser):
    """
    High-throughput user for stress testing event processing.

    Focuses on event publishing to validate 100k+ events/sec target.
    """

    wait_time = between(0.01, 0.05)  # Minimal wait time for max throughput

    @task
    def publish_large_event_batch(self) -> None:
        """Publish very large event batches for throughput testing."""
        batch_size = 10000  # 10k events per request

        events = [
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "performance_test",
                "workflow_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "data": {},
            }
            for _ in range(batch_size)
        ]

        with self.client.post(
            "/api/v1/events/batch",
            json={"events": events},
            catch_response=True,
            name="publish_large_batch"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")


# Locust configuration for different test scenarios
# Run with:
#   locust -f locustfile.py --users=100 --spawn-rate=10 --run-time=5m
