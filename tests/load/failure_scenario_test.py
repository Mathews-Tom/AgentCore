"""
Failure Scenario Load Testing for Gateway Layer

Tests gateway behavior under failure conditions and recovery.

Usage:
    uv run locust -f tests/load/failure_scenario_test.py --host=http://localhost:8001
"""

import random
import time

from locust import HttpUser, task, between, events


class FailureScenarioUser(HttpUser):
    """User for testing failure scenarios."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Setup user."""
        self.agent_id = f"failure-agent-{random.randint(10000, 99999)}"

    @task(3)
    def normal_request(self) -> None:
        """Normal successful request."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.ping",
            "params": {"agent_id": self.agent_id},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="Normal Request",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()

    @task(1)
    def invalid_method_request(self) -> None:
        """Request with invalid method."""
        request = {
            "jsonrpc": "2.0",
            "method": "nonexistent.method",
            "params": {},
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="Invalid Method",
            catch_response=True
        ) as response:
            # Should return error but not crash
            if response.status_code in [200, 400, 404]:
                response.success()

    @task(1)
    def malformed_json_request(self) -> None:
        """Request with malformed JSON."""
        with self.client.post(
            "/api/v1/jsonrpc",
            data="{'invalid': json}",  # Intentionally malformed
            headers={"Content-Type": "application/json"},
            name="Malformed JSON",
            catch_response=True
        ) as response:
            # Should return error but not crash
            if response.status_code in [400, 422]:
                response.success()
            else:
                response.failure(f"Expected 400/422, got {response.status_code}")

    @task(1)
    def missing_auth_request(self) -> None:
        """Request without authentication."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.register",
            "params": {"agent_card": {"agent_id": "test"}},
            "id": "1"
        }

        # Don't include auth header
        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="Missing Auth",
            catch_response=True
        ) as response:
            # Should handle gracefully
            if response.status_code in [200, 401, 403]:
                response.success()

    @task(1)
    def large_payload_request(self) -> None:
        """Request with very large payload."""
        request = {
            "jsonrpc": "2.0",
            "method": "task.create",
            "params": {
                "task_definition": {
                    "task_id": f"large-task-{random.randint(1, 10000)}",
                    "name": "Large Task",
                    "description": "x" * 10000,  # 10KB description
                    "required_capabilities": ["text-generation"],
                    "parameters": {"input": "test" * 1000},
                }
            },
            "id": str(random.randint(1, 10000))
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            json=request,
            name="Large Payload",
            catch_response=True
        ) as response:
            # Should handle or reject gracefully
            if response.status_code in [200, 413]:
                response.success()


# Track error statistics
error_stats = {
    "malformed_json": 0,
    "invalid_method": 0,
    "missing_auth": 0,
    "large_payload": 0,
    "server_errors": 0,
}


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test start."""
    print(f"\n{'='*70}")
    print(f"Failure Scenario Load Test Starting")
    print(f"Target: {environment.host}")
    print(f"Testing error handling and recovery")
    print(f"{'='*70}\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, exception, **kwargs) -> None:
    """Track error statistics."""
    if response:
        if "Malformed JSON" in name and response.status_code in [400, 422]:
            error_stats["malformed_json"] += 1
        elif "Invalid Method" in name and response.status_code in [400, 404]:
            error_stats["invalid_method"] += 1
        elif "Missing Auth" in name and response.status_code in [401, 403]:
            error_stats["missing_auth"] += 1
        elif "Large Payload" in name and response.status_code == 413:
            error_stats["large_payload"] += 1
        elif response.status_code >= 500:
            error_stats["server_errors"] += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log failure scenario results."""
    stats = environment.stats.total

    print(f"\n{'='*70}")
    print(f"Failure Scenario Test Results")
    print(f"{'='*70}")
    print(f"Total Requests: {stats.num_requests:,}")
    print(f"Error Handling Statistics:")
    print(f"  Malformed JSON Handled: {error_stats['malformed_json']:,}")
    print(f"  Invalid Method Handled: {error_stats['invalid_method']:,}")
    print(f"  Missing Auth Handled: {error_stats['missing_auth']:,}")
    print(f"  Large Payload Handled: {error_stats['large_payload']:,}")
    print(f"  Server Errors (5xx): {error_stats['server_errors']:,}")

    # Check graceful error handling
    if error_stats["server_errors"] == 0:
        print(f"\n✓ No server errors - graceful error handling working")
    else:
        print(f"\n✗ Server errors detected: {error_stats['server_errors']:,}")

    print(f"{'='*70}\n")
