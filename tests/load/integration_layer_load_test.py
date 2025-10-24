"""Load testing for Integration Layer.

Tests external integration performance including:
- LLM provider requests through Portkey
- Webhook delivery with retries
- Storage adapter operations
- Database connector queries
- API integration calls

Target: 10,000+ external requests per second
"""

import json
import random
import time
from uuid import uuid4

from locust import HttpUser, TaskSet, between, events, task

# Test data
SAMPLE_LLM_REQUEST = {
    "model_requirements": {
        "capabilities": ["text_generation"],
        "max_cost_per_token": 0.001,
        "max_latency_ms": 2000,
    },
    "request": {
        "prompt": "Summarize the key points from this data",
        "max_tokens": 100,
        "temperature": 0.7,
    },
    "context": {
        "agent_id": "load-test-agent",
        "tenant_id": "load-test-tenant",
    },
}

SAMPLE_WEBHOOK_REGISTRATION = {
    "name": "Load Test Webhook",
    "url": "https://webhook-test.example.com/endpoint",
    "events": ["integration.created", "task.completed"],
}

SAMPLE_EVENT_PUBLISH = {
    "event_type": "task.completed",
    "data": {"task_id": "load-test-123", "status": "completed"},
    "source": "load-test",
}


class IntegrationLayerTaskSet(TaskSet):
    """Task set for integration layer load testing."""

    def on_start(self):
        """Initialize test data."""
        self.webhook_id = None
        self.request_count = 0
        self.error_count = 0
        self.latencies = []

    @task(40)
    def test_llm_provider_request(self):
        """Test LLM provider request routing (40% of load)."""
        start_time = time.time()

        # Simulate LLM completion request
        response = self.client.post(
            "/api/v1/integration/llm/complete",
            json=SAMPLE_LLM_REQUEST,
            headers={"Content-Type": "application/json"},
            name="LLM Provider Request",
        )

        latency = (time.time() - start_time) * 1000  # ms
        self.latencies.append(latency)

        if response.status_code != 200:
            self.error_count += 1

        self.request_count += 1

    @task(20)
    def test_webhook_registration(self):
        """Test webhook registration (20% of load)."""
        webhook_data = SAMPLE_WEBHOOK_REGISTRATION.copy()
        webhook_data["name"] = f"Load Test Webhook {uuid4()}"

        response = self.client.post(
            "/api/v1/integration/webhooks",
            json=webhook_data,
            headers={"Content-Type": "application/json"},
            name="Webhook Registration",
        )

        if response.status_code == 201:
            data = response.json()
            self.webhook_id = data.get("id")

    @task(30)
    def test_event_publishing(self):
        """Test event publishing (30% of load)."""
        event_data = SAMPLE_EVENT_PUBLISH.copy()
        event_data["data"]["task_id"] = f"load-test-{uuid4()}"

        self.client.post(
            "/api/v1/integration/events/publish",
            json=event_data,
            headers={"Content-Type": "application/json"},
            name="Event Publishing",
        )

    @task(5)
    def test_webhook_list(self):
        """Test webhook listing (5% of load)."""
        self.client.get(
            "/api/v1/integration/webhooks",
            name="Webhook List",
        )

    @task(3)
    def test_webhook_stats(self):
        """Test webhook statistics (3% of load)."""
        if self.webhook_id:
            self.client.get(
                f"/api/v1/integration/webhooks/{self.webhook_id}/stats",
                name="Webhook Stats",
            )

    @task(2)
    def test_storage_upload(self):
        """Test storage adapter upload (2% of load)."""
        file_data = {
            "file_name": f"load-test-{uuid4()}.txt",
            "content": "Load test file content",
            "metadata": {"test": "true", "timestamp": time.time()},
        }

        self.client.post(
            "/api/v1/integration/storage/upload",
            json=file_data,
            name="Storage Upload",
        )

    def on_stop(self):
        """Cleanup and report statistics."""
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            p95_latency = sorted(self.latencies)[int(len(self.latencies) * 0.95)]

            print(f"\n=== Load Test Statistics ===")
            print(f"Total Requests: {self.request_count}")
            print(f"Errors: {self.error_count}")
            print(f"Error Rate: {self.error_count / self.request_count * 100:.2f}%")
            print(f"Avg Latency: {avg_latency:.2f}ms")
            print(f"P95 Latency: {p95_latency:.2f}ms")


class IntegrationLayerUser(HttpUser):
    """Locust user for integration layer testing."""

    tasks = [IntegrationLayerTaskSet]
    wait_time = between(0.1, 0.5)  # Fast request rate for high throughput
    host = "http://localhost:8001"  # AgentCore API endpoint


class HighThroughputUser(HttpUser):
    """High throughput user for stress testing."""

    tasks = [IntegrationLayerTaskSet]
    wait_time = between(0.01, 0.1)  # Very fast for 10k+ req/s
    host = "http://localhost:8001"


# Performance metrics collection
request_latencies = []
request_counts = {"success": 0, "failure": 0}


@events.request.add_listener
def on_request(
    request_type,
    name,
    response_time,
    response_length,
    exception,
    context,
    **kwargs,
):
    """Track request performance metrics."""
    request_latencies.append(response_time)

    if exception:
        request_counts["failure"] += 1
    else:
        request_counts["success"] += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report final performance statistics."""
    if not request_latencies:
        print("No requests recorded")
        return

    total_requests = sum(request_counts.values())
    success_rate = request_counts["success"] / total_requests * 100 if total_requests > 0 else 0

    sorted_latencies = sorted(request_latencies)
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

    print("\n" + "=" * 60)
    print("INTEGRATION LAYER LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests:        {total_requests:,}")
    print(f"Successful Requests:   {request_counts['success']:,}")
    print(f"Failed Requests:       {request_counts['failure']:,}")
    print(f"Success Rate:          {success_rate:.2f}%")
    print(f"\nLatency Statistics:")
    print(f"  P50 (Median):        {p50:.2f}ms")
    print(f"  P95:                 {p95:.2f}ms")
    print(f"  P99:                 {p99:.2f}ms")
    print(f"  Min:                 {min(request_latencies):.2f}ms")
    print(f"  Max:                 {max(request_latencies):.2f}ms")
    print(f"  Average:             {sum(request_latencies) / len(request_latencies):.2f}ms")

    # Performance targets validation
    print(f"\n{'=' * 60}")
    print("PERFORMANCE TARGETS VALIDATION")
    print("=" * 60)

    targets_met = True

    # Target: 10,000+ requests per second
    if environment.stats.total.num_requests > 0:
        duration = environment.stats.total.last_request_timestamp - environment.stats.total.start_time
        rps = environment.stats.total.num_requests / duration if duration > 0 else 0
        print(f"Throughput:            {rps:.0f} req/s")
        if rps >= 10000:
            print("  ✅ Target met: 10,000+ req/s")
        else:
            print(f"  ❌ Target not met: {rps:.0f} < 10,000 req/s")
            targets_met = False

    # Target: <100ms P95 latency
    if p95 < 100:
        print(f"P95 Latency:           {p95:.2f}ms")
        print("  ✅ Target met: <100ms P95 latency")
    else:
        print(f"P95 Latency:           {p95:.2f}ms")
        print(f"  ❌ Target not met: {p95:.2f}ms > 100ms")
        targets_met = False

    # Target: 99.9% success rate
    if success_rate >= 99.9:
        print(f"Success Rate:          {success_rate:.2f}%")
        print("  ✅ Target met: 99.9%+ success rate")
    else:
        print(f"Success Rate:          {success_rate:.2f}%")
        print(f"  ❌ Target not met: {success_rate:.2f}% < 99.9%")
        targets_met = False

    print("=" * 60)

    if targets_met:
        print("✅ ALL PERFORMANCE TARGETS MET")
    else:
        print("❌ SOME PERFORMANCE TARGETS NOT MET")

    print("=" * 60 + "\n")
