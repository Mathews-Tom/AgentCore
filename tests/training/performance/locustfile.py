"""
Locust load testing configuration for training API (FLOW-020).

Simulates real-world load patterns for training infrastructure:
- Start training jobs
- Poll job status
- Cancel jobs
- Export trajectories

Run with:
    locust -f tests/training/performance/locustfile.py --host=http://localhost:8001

TODO: Stress Test Performance Optimization
    Current stress test (500 users) shows degraded performance:
    - 6.00% failure rate (target: <2%)
    - 3,580ms P95 response time (target: <1000ms)

    Optimization path to support 500+ concurrent users:
    1. Implement PgBouncer for database connection pooling
    2. Scale worker pods from 20 to 30-50 (HPA adjustment)
    3. Increase database max_connections to 500
    4. Add Redis connection pool tuning
    5. Implement request queuing for burst protection

    References:
    - PR description: .docs/PR_DESCRIPTION.md:106-113
    - Deployment recommendations: .docs/PR_DESCRIPTION.md:194-210
    - Current capacity: 150-200 concurrent jobs (acceptable for initial deployment)
"""

from __future__ import annotations

import json
import random
from uuid import uuid4

from locust import HttpUser, between, events, task


class TrainingAPIUser(HttpUser):
    """
    Simulates a user interacting with the training API.

    Behavior:
    - 40% of requests: Check job status (most common)
    - 30% of requests: Start new training jobs
    - 20% of requests: Export trajectory data
    - 10% of requests: Cancel jobs
    """

    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    host = "http://localhost:8001"

    def on_start(self):
        """Setup: Authenticate and get JWT token."""
        self.jwt_token = "test-jwt-token"  # In real scenario, obtain via auth
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json",
        }
        self.job_ids = []  # Track created job IDs

    @task(30)
    def start_training_job(self):
        """Start a new training job (30% of requests)."""
        # Create minimal training data (100 queries minimum)
        training_data = [
            {"query": f"Task {i}", "expected_outcome": {"success": True}}
            for i in range(100)
        ]

        request_data = {
            "jsonrpc": "2.0",
            "method": "training.start_grpo",
            "params": {
                "agent_id": f"load-test-agent-{random.randint(1, 10)}",
                "config": {
                    "n_iterations": random.randint(10, 50),
                    "batch_size": 16,
                    "n_trajectories_per_query": 4,
                    "learning_rate": 0.0001,
                    "max_budget_usd": 5.00,
                },
                "training_data": training_data,
            },
            "id": str(uuid4()),
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request_data,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    job_id = result["result"]["job_id"]
                    self.job_ids.append(job_id)
                    response.success()
                else:
                    response.failure(f"No result in response: {result}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(40)
    def get_job_status(self):
        """Check status of training job (40% of requests - most common)."""
        if not self.job_ids:
            # Use a random job ID if none created yet
            job_id = str(uuid4())
        else:
            job_id = random.choice(self.job_ids)

        request_data = {
            "jsonrpc": "2.0",
            "method": "training.get_status",
            "params": {"job_id": job_id},
            "id": str(uuid4()),
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request_data,
            headers=self.headers,
            catch_response=True,
            name="/training.get_status",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "result" in result or "error" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status {response.status_code}")

    @task(20)
    def export_trajectories(self):
        """Export trajectory data (20% of requests)."""
        if not self.job_ids:
            return  # Skip if no jobs created

        job_id = random.choice(self.job_ids)

        request_data = {
            "jsonrpc": "2.0",
            "method": "training.export_trajectories",
            "params": {
                "job_id": job_id,
                "success_only": True,
                "limit": 50,
            },
            "id": str(uuid4()),
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request_data,
            headers=self.headers,
            catch_response=True,
            name="/training.export_trajectories",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(10)
    def cancel_job(self):
        """Cancel a training job (10% of requests)."""
        if not self.job_ids:
            return  # Skip if no jobs created

        job_id = random.choice(self.job_ids)

        request_data = {
            "jsonrpc": "2.0",
            "method": "training.cancel",
            "params": {"job_id": job_id, "reason": "Load test cancellation"},
            "id": str(uuid4()),
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request_data,
            headers=self.headers,
            catch_response=True,
            name="/training.cancel",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class BurstTrainingUser(HttpUser):
    """
    Simulates burst traffic patterns - sudden spikes in load.

    Used for stress testing system resilience.
    """

    wait_time = between(0.1, 1)  # Very short wait times for burst
    host = "http://localhost:8001"

    def on_start(self):
        """Setup authentication."""
        self.jwt_token = "test-jwt-token"
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json",
        }

    @task(100)
    def rapid_status_checks(self):
        """Rapid fire status checks to simulate burst."""
        request_data = {
            "jsonrpc": "2.0",
            "method": "training.get_status",
            "params": {"job_id": str(uuid4())},
            "id": str(uuid4()),
        }

        self.client.post(
            "/api/v1/jsonrpc",
            json=request_data,
            headers=self.headers,
            name="/training.get_status (burst)",
        )


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Hook called before test starts."""
    print(f"\n{'=' * 60}")
    print("Starting Training API Load Test")
    print(f"Host: {environment.host}")
    print(f"{'=' * 60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Hook called after test stops - print summary."""
    print(f"\n{'=' * 60}")
    print("Load Test Complete")
    print(f"{'=' * 60}")

    stats = environment.stats
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0

    print(f"\nSummary:")
    print(f"  Total Requests: {total_requests}")
    print(f"  Total Failures: {total_failures}")
    print(f"  Failure Rate: {failure_rate:.2f}%")
    print(f"  Average Response Time: {stats.total.avg_response_time:.1f}ms")
    print(f"  Median Response Time: {stats.total.median_response_time:.1f}ms")
    print(
        f"  P95 Response Time: {stats.total.get_response_time_percentile(0.95):.1f}ms"
    )
    print(
        f"  P99 Response Time: {stats.total.get_response_time_percentile(0.99):.1f}ms"
    )
    print(f"  Requests/sec: {stats.total.total_rps:.2f}")

    print(f"\nStatus:")
    if failure_rate < 1.0 and stats.total.get_response_time_percentile(0.95) < 200:
        print("  ✓ PASS - All SLA targets met")
    else:
        print("  ✗ FAIL - SLA targets not met")

    print(f"{'=' * 60}\n")


# Custom load shape for gradual ramp-up
class GradualRampUp:
    """
    Gradually ramps up load to test system under increasing pressure.

    Pattern:
    - 0-2 min: 10 users
    - 2-4 min: 50 users
    - 4-6 min: 100 users
    - 6-8 min: 150 users
    - 8-10 min: 200 users (peak)
    """

    def tick(self):
        """Define number of users at each time point."""
        run_time = self.get_run_time()

        if run_time < 120:
            return (10, 5)  # 10 users, spawn rate 5/sec
        elif run_time < 240:
            return (50, 10)
        elif run_time < 360:
            return (100, 10)
        elif run_time < 480:
            return (150, 10)
        elif run_time < 600:
            return (200, 10)
        else:
            return None  # Stop test after 10 minutes


# Configuration for different test scenarios
SCENARIOS = {
    "light": {
        "description": "Light load - typical usage",
        "users": 20,
        "spawn_rate": 5,
        "duration": "5m",
    },
    "moderate": {
        "description": "Moderate load - busy period",
        "users": 100,
        "spawn_rate": 10,
        "duration": "10m",
    },
    "heavy": {
        "description": "Heavy load - peak usage",
        "users": 200,
        "spawn_rate": 20,
        "duration": "15m",
    },
    "stress": {
        "description": "Stress test - beyond capacity",
        "users": 500,
        "spawn_rate": 50,
        "duration": "20m",
    },
}


def print_scenario_info():
    """Print available test scenarios."""
    print("\nAvailable Load Test Scenarios:")
    print("=" * 60)
    for name, config in SCENARIOS.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Users: {config['users']}")
        print(f"  Spawn Rate: {config['spawn_rate']}/sec")
        print(f"  Duration: {config['duration']}")
    print("\n" + "=" * 60)
    print("\nUsage:")
    print("  locust -f locustfile.py --users=100 --spawn-rate=10 --run-time=10m")
    print("  locust -f locustfile.py --headless --users=200 --spawn-rate=20")
    print("\n")


if __name__ == "__main__":
    print_scenario_info()
    print_scenario_info()
