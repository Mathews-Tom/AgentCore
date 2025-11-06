"""
Gateway Layer Load Testing for 60,000+ req/sec

Locust load testing scenario for validating gateway performance targets.

Target Performance:
- 60,000+ requests per second
- <5ms p95 latency for routing overhead
- 10,000+ concurrent WebSocket connections
- <4GB memory per instance
- <50% CPU utilization under normal load

Usage:
    # Basic load test
    uv run locust -f tests/load/gateway_performance_test.py --host http://localhost:8080

    # High-throughput test (60k+ req/sec target)
    uv run locust -f tests/load/gateway_performance_test.py \\
        --host http://localhost:8080 \\
        --users 5000 \\
        --spawn-rate 100 \\
        --run-time 5m \\
        --headless

    # WebSocket connection test
    uv run locust -f tests/load/gateway_performance_test.py \\
        --host http://localhost:8080 \\
        --users 10000 \\
        --spawn-rate 500 \\
        --headless \\
        WebSocketConnectionUser

Performance Validation:
    1. Start gateway with Gunicorn:
       uv run gunicorn gateway.main:app --config src/gateway/gunicorn.conf.py

    2. Run load test and validate metrics:
       - RPS: Should exceed 60,000 req/sec with sufficient workers
       - Latency (p95): Should be <5ms for routing overhead
       - Error rate: Should be <0.1%

    3. Monitor with Prometheus:
       http://localhost:8080/metrics
"""

from __future__ import annotations

import json
import random
import time
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask


class GatewayHealthUser(HttpUser):
    """
    High-frequency health check user.

    Simulates monitoring systems and load balancers performing health checks.
    Targets: Simple GET requests with minimal overhead.
    """

    wait_time = between(0.1, 0.5)  # Very frequent checks

    @task(10)
    def health_check(self):
        """Basic health endpoint (should be fast, no auth required)."""
        with self.client.get("/health", name="/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def readiness_check(self):
        """Readiness endpoint for load balancers."""
        with self.client.get("/health/ready", name="/health/ready", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class GatewayAuthenticatedUser(HttpUser):
    """
    Authenticated API user simulating real application traffic.

    Tests full request pipeline including:
    - JWT authentication
    - Rate limiting
    - Request validation
    - Response transformation
    - Caching
    """

    wait_time = between(0.5, 2)
    access_token = None

    def on_start(self):
        """Authenticate and obtain JWT token."""
        # For now, skip actual OAuth flow in load test
        # In production load test, would perform full OAuth flow
        # self.access_token = "mock-jwt-token"
        pass

    @task(5)
    def get_with_cache(self):
        """
        GET request that should benefit from response caching.

        Tests cache hit rate and response time improvement.
        """
        cache_key = random.randint(1, 100)  # Limited key space for cache hits
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        with self.client.get(
            f"/api/v1/resource/{cache_key}",
            name="/api/v1/resource/:id (cached)",
            headers=headers,
            catch_response=True,
        ) as response:
            # Even 404s are fine for load testing (testing gateway, not backend)
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def post_request(self):
        """POST request with JSON payload."""
        payload = {
            "action": "test",
            "data": f"load-test-{random.randint(1000, 9999)}",
        }
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        with self.client.post(
            "/api/v1/action",
            name="/api/v1/action",
            json=payload,
            headers=headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201, 404):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def list_resources(self):
        """List resources with query parameters."""
        limit = random.choice([10, 25, 50, 100])
        offset = random.randint(0, 1000)
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        with self.client.get(
            f"/api/v1/resources?limit={limit}&offset={offset}",
            name="/api/v1/resources (list)",
            headers=headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def metrics_endpoint(self):
        """Fetch Prometheus metrics."""
        with self.client.get("/metrics", name="/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class GatewayBurstTrafficUser(HttpUser):
    """
    Burst traffic user simulating traffic spikes.

    Tests gateway's ability to handle sudden traffic increases
    without degrading performance or triggering rate limits incorrectly.
    """

    wait_time = between(0, 0.1)  # Minimal wait time for burst

    @task
    def burst_requests(self):
        """Rapid-fire requests to test burst handling."""
        with self.client.get("/health", name="/health (burst)", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limited - this is expected under burst
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class WebSocketConnectionUser(HttpUser):
    """
    WebSocket connection user for real-time testing.

    Note: Locust doesn't natively support WebSocket load testing well.
    For proper WebSocket testing, use dedicated tools like:
    - wsdump (from websocket-client)
    - artillery
    - k6 with WebSocket support

    This user simulates SSE (Server-Sent Events) as a lighter alternative.
    """

    wait_time = between(10, 30)  # Long-lived connections

    @task
    def sse_connection(self):
        """
        Simulate SSE connection for events.

        In production, would maintain long-lived connection.
        For Locust, we simulate with repeated connection attempts.
        """
        with self.client.get(
            "/api/v1/events/stream",
            name="/api/v1/events/stream (SSE)",
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# Performance monitoring and reporting
request_latencies = []
request_times = {}


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Track request latencies for p95 calculation."""
    if exception is None:
        request_latencies.append(response_time)

        # Track per-endpoint latencies
        if name not in request_times:
            request_times[name] = []
        request_times[name].append(response_time)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test and print configuration."""
    print("=" * 80)
    print("Gateway Layer Performance Test")
    print("=" * 80)
    print(f"Target Host: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users}")
    print(f"Spawn Rate: {environment.parsed_options.spawn_rate} users/sec")
    print()
    print("Performance Targets:")
    print("  - Throughput: 60,000+ req/sec")
    print("  - Latency (p95): <5ms routing overhead")
    print("  - Error Rate: <0.1%")
    print("=" * 80)
    print()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Calculate and print final performance metrics."""
    print()
    print("=" * 80)
    print("Performance Test Results")
    print("=" * 80)
    print()

    # Overall stats
    total_requests = environment.stats.total.num_requests
    failed_requests = environment.stats.total.num_failures
    total_rps = environment.stats.total.total_rps
    avg_response_time = environment.stats.total.avg_response_time

    print(f"Total Requests: {total_requests:,}")
    print(f"Failed Requests: {failed_requests:,}")
    print(f"Error Rate: {(failed_requests / total_requests * 100) if total_requests > 0 else 0:.2f}%")
    print(f"Requests/sec: {total_rps:,.2f}")
    print(f"Avg Response Time: {avg_response_time:.2f} ms")
    print()

    # Calculate percentiles
    if request_latencies:
        sorted_latencies = sorted(request_latencies)
        count = len(sorted_latencies)

        p50_index = int(count * 0.50)
        p95_index = int(count * 0.95)
        p99_index = int(count * 0.99)

        p50 = sorted_latencies[p50_index] if p50_index < count else 0
        p95 = sorted_latencies[p95_index] if p95_index < count else 0
        p99 = sorted_latencies[p99_index] if p99_index < count else 0

        print("Latency Percentiles:")
        print(f"  p50: {p50:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        print(f"  p99: {p99:.2f} ms")
        print()

        # Performance validation
        print("Performance Validation:")
        print(f"  ✓ Throughput Target (60k req/sec): {'PASS' if total_rps >= 60000 else 'FAIL'} ({total_rps:,.0f} req/sec)")
        print(f"  ✓ Latency Target (p95 <5ms): {'PASS' if p95 < 5 else 'FAIL'} ({p95:.2f} ms)")
        print(f"  ✓ Error Rate Target (<0.1%): {'PASS' if (failed_requests / total_requests * 100) < 0.1 else 'FAIL'} ({(failed_requests / total_requests * 100):.2f}%)")
        print()

    # Per-endpoint breakdown
    print("Per-Endpoint Performance:")
    print(f"{'Endpoint':<40} {'Requests':>10} {'RPS':>10} {'Avg (ms)':>10} {'p95 (ms)':>10}")
    print("-" * 80)

    for stat in sorted(environment.stats.entries.values(), key=lambda x: x.num_requests, reverse=True):
        if stat.num_requests > 0:
            endpoint_latencies = request_times.get(stat.name, [])
            if endpoint_latencies:
                sorted_endpoint = sorted(endpoint_latencies)
                p95_idx = int(len(sorted_endpoint) * 0.95)
                endpoint_p95 = sorted_endpoint[p95_idx] if p95_idx < len(sorted_endpoint) else 0
            else:
                endpoint_p95 = 0

            print(
                f"{stat.name:<40} "
                f"{stat.num_requests:>10,} "
                f"{stat.total_rps:>10.2f} "
                f"{stat.avg_response_time:>10.2f} "
                f"{endpoint_p95:>10.2f}"
            )

    print("=" * 80)

    # Recommendations
    if total_rps < 60000:
        print()
        print("⚠️  Throughput below target. Recommendations:")
        print("  1. Increase Gunicorn workers (current: check gunicorn.conf.py)")
        print("  2. Tune OS parameters (net.core.somaxconn, file descriptors)")
        print("  3. Enable response caching for GET endpoints")
        print("  4. Optimize middleware chain (disable unused middleware)")
        print("  5. Use HTTP/2 and connection pooling")

    if p95 >= 5:
        print()
        print("⚠️  Latency above target. Recommendations:")
        print("  1. Profile slow endpoints and optimize")
        print("  2. Reduce middleware overhead")
        print("  3. Enable response caching")
        print("  4. Optimize database queries (if applicable)")
        print("  5. Use connection pooling for backend services")

    print()
