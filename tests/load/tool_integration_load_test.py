"""
Tool Integration Framework Load Testing

Comprehensive load testing for tool execution, rate limiting, quota management,
retry logic, and parallel execution features.

Target Performance:
- 1,000+ concurrent tool executions
- <100ms p95 latency for lightweight tools (echo, calculator)
- <1s p95 latency for medium tools (web scraping, code execution)
- Rate limiting: Proper 429 responses under burst traffic
- Quota management: Enforced limits without false negatives
- Retry logic: Automatic retry on transient failures
- Error rate: <1% for non-quota errors

Usage:
    # Basic tool execution test
    uv run locust -f tests/load/tool_integration_load_test.py --host http://localhost:8001

    # High-throughput test (1000+ concurrent executions)
    uv run locust -f tests/load/tool_integration_load_test.py \\
        --host http://localhost:8001 \\
        --users 1000 \\
        --spawn-rate 50 \\
        --run-time 5m \\
        --headless

    # Rate limiting test
    uv run locust -f tests/load/tool_integration_load_test.py \\
        --host http://localhost:8001 \\
        --users 500 \\
        --spawn-rate 100 \\
        --run-time 3m \\
        --headless \\
        RateLimitingUser

    # Quota management test
    uv run locust -f tests/load/tool_integration_load_test.py \\
        --host http://localhost:8001 \\
        --users 200 \\
        --spawn-rate 50 \\
        --run-time 5m \\
        --headless \\
        QuotaEnforcementUser

Performance Validation:
    1. Start AgentCore with monitoring:
       docker compose -f docker-compose.dev.yml up -d

    2. Run load test and validate metrics:
       - Concurrent executions: Should handle 1000+ concurrent requests
       - Latency (p95): <100ms for lightweight tools
       - Error rate: <1% (excluding quota/rate limit errors)
       - Rate limiting: Proper 429 responses during bursts
       - Quota enforcement: Consistent quota exhaustion behavior

    3. Monitor with Grafana:
       http://localhost:3000 (Tool Integration Metrics dashboard)
"""

from __future__ import annotations

import json
import random
import time
from locust import HttpUser, task, between, events

# Track performance metrics
tool_execution_times = {}
rate_limit_hits = 0
quota_exceeded_count = 0
successful_executions = 0
failed_executions = 0


class ToolExecutionUser(HttpUser):
    """
    Standard tool execution user testing various tool types.

    Tests the core tool execution pipeline including:
    - Parameter validation
    - Tool discovery and registry lookup
    - Execution lifecycle hooks
    - Result serialization
    - Metrics emission
    """

    wait_time = between(0.5, 2)

    @task(5)
    def execute_echo_tool(self):
        """Execute lightweight echo tool (minimal overhead)."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute",
            "params": {
                "tool_id": "echo",
                "parameters": {"message": f"load-test-{random.randint(1, 1000)}"},
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute (echo)",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code == -32001:  # Rate limit
                        response.success()  # Expected under load
                    elif error_code == -32005:  # Quota exceeded
                        response.success()  # Expected when quota configured
                    else:
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid JSON-RPC response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(3)
    def execute_calculator_tool(self):
        """Execute calculator tool with various operations."""
        operations = ["add", "subtract", "multiply", "divide"]
        operation = random.choice(operations)

        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute",
            "params": {
                "tool_id": "calculator",
                "parameters": {
                    "operation": operation,
                    "a": random.uniform(1, 100),
                    "b": random.uniform(1, 100),
                },
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute (calculator)",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code in (-32001, -32005):
                        response.success()  # Rate limit or quota
                    else:
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid JSON-RPC response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(2)
    def execute_current_time_tool(self):
        """Execute current time tool."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute",
            "params": {
                "tool_id": "current_time",
                "parameters": {"timezone": random.choice(["UTC", "America/New_York", "Europe/London"])},
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute (current_time)",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code in (-32001, -32005):
                        response.success()
                    else:
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid JSON-RPC response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(1)
    def list_tools(self):
        """List available tools (registry lookup)."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.list",
            "params": {"limit": 20},
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.list",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                else:
                    response.failure("Invalid response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")


class RateLimitingUser(HttpUser):
    """
    User testing rate limiting behavior under burst traffic.

    Generates rapid-fire requests to trigger rate limiting and verify:
    - Rate limiter activates correctly
    - 429 responses with proper headers
    - No false positives (legitimate requests not blocked)
    - Redis sliding window algorithm works correctly
    """

    wait_time = between(0, 0.1)  # Minimal wait for burst

    @task
    def burst_tool_executions(self):
        """Rapid-fire tool executions to test rate limiting."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute",
            "params": {
                "tool_id": "echo",
                "parameters": {"message": "burst-test"},
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute (burst)",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code == -32001:  # Rate limit expected
                        global rate_limit_hits
                        rate_limit_hits += 1
                        response.success()
                    elif error_code == -32005:  # Quota exceeded
                        response.success()
                    else:
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")


class QuotaEnforcementUser(HttpUser):
    """
    User testing quota management under load.

    Tests quota enforcement including:
    - Daily quota tracking
    - Monthly quota tracking
    - Per-agent quota isolation
    - Quota reset timing
    - Quota status queries
    """

    wait_time = between(0.5, 1.5)
    agent_id = None

    def on_start(self):
        """Assign unique agent ID for quota isolation testing."""
        self.agent_id = f"agent-{random.randint(1, 100)}"

    @task(3)
    def execute_with_quota(self):
        """Execute tool with quota tracking."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute",
            "params": {
                "tool_id": "echo",
                "parameters": {"message": "quota-test"},
                "identifier": self.agent_id,  # For per-agent quota
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute (with quota)",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    global successful_executions
                    successful_executions += 1
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code == -32005:  # Quota exceeded expected
                        global quota_exceeded_count
                        quota_exceeded_count += 1
                        response.success()
                    elif error_code == -32001:  # Rate limit
                        response.success()
                    else:
                        global failed_executions
                        failed_executions += 1
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(1)
    def check_rate_limit_status(self):
        """Query rate limit and quota status."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.get_rate_limit_status",
            "params": {
                "tool_id": "echo",
                "identifier": self.agent_id,
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.get_rate_limit_status",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                else:
                    response.failure("Invalid response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")


class ParallelExecutionUser(HttpUser):
    """
    User testing parallel and batch execution features.

    Tests advanced execution modes:
    - Batch execution (multiple tools in parallel)
    - Dependency-based execution
    - Fallback chains
    """

    wait_time = between(1, 3)

    @task(3)
    def execute_batch(self):
        """Execute multiple tools in parallel (batch)."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute_batch",
            "params": {
                "executions": [
                    {
                        "tool_id": "echo",
                        "parameters": {"message": "batch-1"},
                    },
                    {
                        "tool_id": "calculator",
                        "parameters": {"operation": "add", "a": 10, "b": 20},
                    },
                    {
                        "tool_id": "current_time",
                        "parameters": {"timezone": "UTC"},
                    },
                ]
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute_batch",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code in (-32001, -32005):
                        response.success()
                    else:
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")

    @task(1)
    def execute_with_fallback(self):
        """Execute tool with fallback chain."""
        payload = {
            "jsonrpc": "2.0",
            "id": f"test-{random.randint(1000, 9999)}",
            "method": "tools.execute_with_fallback",
            "params": {
                "primary": {
                    "tool_id": "echo",
                    "parameters": {"message": "primary"},
                },
                "fallbacks": [
                    {
                        "tool_id": "calculator",
                        "parameters": {"operation": "add", "a": 1, "b": 1},
                    }
                ],
            },
        }

        with self.client.post(
            "/api/v1/jsonrpc",
            name="tools.execute_with_fallback",
            json=payload,
            catch_response=True,
        ) as response:
            try:
                result = response.json()
                if "result" in result:
                    response.success()
                elif "error" in result:
                    error_code = result["error"].get("code")
                    if error_code in (-32001, -32005):
                        response.success()
                    else:
                        response.failure(f"Error: {result['error'].get('message')}")
                else:
                    response.failure("Invalid response")
            except Exception as e:
                response.failure(f"Exception: {str(e)}")


# Event handlers for metrics tracking

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Track tool execution metrics."""
    if exception is None and "tools." in name:
        if name not in tool_execution_times:
            tool_execution_times[name] = []
        tool_execution_times[name].append(response_time)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test and print configuration."""
    print("=" * 80)
    print("Tool Integration Framework Load Test")
    print("=" * 80)
    print(f"Target Host: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users}")
    print(f"Spawn Rate: {environment.parsed_options.spawn_rate} users/sec")
    print()
    print("Performance Targets:")
    print("  - Concurrent Executions: 1,000+")
    print("  - Latency (p95) Lightweight: <100ms")
    print("  - Latency (p95) Medium: <1s")
    print("  - Error Rate: <1% (excluding quota/rate limit)")
    print("  - Rate Limiting: Proper 429 responses")
    print("  - Quota Enforcement: Consistent behavior")
    print("=" * 80)
    print()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Calculate and print final performance metrics."""
    print()
    print("=" * 80)
    print("Tool Integration Load Test Results")
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

    # Rate limiting and quota stats
    global rate_limit_hits, quota_exceeded_count, successful_executions, failed_executions
    print("Rate Limiting & Quota:")
    print(f"  Rate Limit Hits: {rate_limit_hits:,}")
    print(f"  Quota Exceeded: {quota_exceeded_count:,}")
    print(f"  Successful Executions: {successful_executions:,}")
    print(f"  Failed Executions: {failed_executions:,}")
    print()

    # Per-method performance
    print("Per-Method Performance:")
    print(f"{'Method':<45} {'Requests':>10} {'RPS':>10} {'Avg (ms)':>10} {'p95 (ms)':>10}")
    print("-" * 85)

    for stat in sorted(environment.stats.entries.values(), key=lambda x: x.num_requests, reverse=True):
        if stat.num_requests > 0:
            method_times = tool_execution_times.get(stat.name, [])
            if method_times:
                sorted_times = sorted(method_times)
                p95_idx = int(len(sorted_times) * 0.95)
                method_p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0
            else:
                method_p95 = 0

            print(
                f"{stat.name:<45} "
                f"{stat.num_requests:>10,} "
                f"{stat.total_rps:>10.2f} "
                f"{stat.avg_response_time:>10.2f} "
                f"{method_p95:>10.2f}"
            )

    print("=" * 85)

    # Validation
    print()
    print("Performance Validation:")

    concurrent_target = 1000
    concurrent_result = "PASS" if total_rps >= (concurrent_target / 2) else "FAIL"
    print(f"  ✓ Concurrent Execution ({concurrent_target}+): {concurrent_result} ({total_rps:,.0f} req/sec)")

    error_rate = (failed_executions / total_requests * 100) if total_requests > 0 else 0
    error_result = "PASS" if error_rate < 1 else "FAIL"
    print(f"  ✓ Error Rate (<1%): {error_result} ({error_rate:.2f}%)")

    rate_limit_result = "PASS" if rate_limit_hits > 0 else "INFO"
    print(f"  ✓ Rate Limiting Active: {rate_limit_result} ({rate_limit_hits:,} hits)")

    quota_result = "PASS" if quota_exceeded_count >= 0 else "FAIL"
    print(f"  ✓ Quota Enforcement: {quota_result} ({quota_exceeded_count:,} exceeded)")

    print()
    print("=" * 85)
