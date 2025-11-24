# Load Testing for AgentCore

Comprehensive load testing suite for gateway layer, tool integration framework, and modular agent core performance validation.

## Test Scenarios

### 0. Modular Agent Core Load Test (`test_modular_load.py`)
**Target:** 100+ concurrent modular executions per instance

Tests modular.solve API under sustained load to validate NFR-1.4.

```bash
# Full load test (100 users, 10 minutes) - MOD-028 acceptance criteria
./tests/load/run_modular_load_test.sh full

# Quick validation (10 users, 2 minutes)
./tests/load/run_modular_load_test.sh quick

# Direct locust usage
uv run locust -f tests/load/test_modular_load.py \
    --host=http://localhost:8001 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=10m
```

**Expected Results:**
- Success rate: >95%
- p95 latency: <4500ms (3x baseline of 1500ms)
- Throughput: 15-25 requests/second
- No memory leaks or resource exhaustion

**Documentation:** See `tests/load/MODULAR_LOAD_TESTING.md` for detailed information.

## Test Scenarios

### 1. HTTP Load Test (`http_load_test.py`)
**Target:** 60,000+ requests per second

Tests high-throughput HTTP requests to validate gateway performance.

```bash
# Run with 1000 users, spawn rate of 100/sec
uv run locust -f tests/load/http_load_test.py \
    --host=http://localhost:8001 \
    --users=1000 \
    --spawn-rate=100 \
    --run-time=5m
```

**Expected Results:**
- RPS: 60,000+
- P95 latency: < 50ms
- P99 latency: < 100ms
- Failure rate: < 0.1%

### 2. WebSocket Connection Test (`websocket_load_test.py`)
**Target:** 10,000+ concurrent connections

Tests concurrent WebSocket connections and connection stability.

```bash
# Run with 10,000 users for concurrent connections
uv run locust -f tests/load/websocket_load_test.py \
    --host=http://localhost:8001 \
    --users=10000 \
    --spawn-rate=100 \
    --run-time=5m
```

**Expected Results:**
- Concurrent connections: 10,000+
- Connection stability: > 99%
- Memory usage: Stable over time

### 3. Sustained Load Test (`sustained_load_test.py`)
**Target:** 10+ minutes of stable operation

Tests gateway stability under sustained load.

```bash
# Run for 10 minutes minimum
uv run locust -f tests/load/sustained_load_test.py \
    --host=http://localhost:8001 \
    --users=500 \
    --spawn-rate=50 \
    --run-time=10m
```

**Expected Results:**
- Performance degradation: < 10%
- Memory leaks: None
- Stable RPS throughout test
- No crash or restart required

### 4. Burst Traffic Test (`burst_traffic_test.py`)
**Target:** Handle sudden traffic spikes

Tests gateway behavior under burst traffic patterns.

```bash
# Run with burst patterns
uv run locust -f tests/load/burst_traffic_test.py \
    --host=http://localhost:8001 \
    --users=1000 \
    --spawn-rate=200 \
    --run-time=5m
```

**Expected Results:**
- Rate limiting activates during bursts
- No crash or failure during spikes
- Recovery time: < 5 seconds
- Graceful degradation

### 5. Failure Scenario Test (`failure_scenario_test.py`)
**Target:** Graceful error handling

Tests gateway behavior under various failure conditions.

```bash
# Run failure scenarios
uv run locust -f tests/load/failure_scenario_test.py \
    --host=http://localhost:8001 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=5m
```

**Expected Results:**
- No 5xx errors from gateway
- Proper error responses (4xx)
- No memory leaks
- System remains responsive

### 6. Tool Integration Load Test (`tool_integration_load_test.py`)
**Target:** 1,000+ concurrent tool executions

Comprehensive load testing for tool execution, rate limiting, quota management, retry logic, and parallel execution features.

```bash
# Run with 1000 users for concurrent tool executions
uv run locust -f tests/load/tool_integration_load_test.py \
    --host=http://localhost:8001 \
    --users=1000 \
    --spawn-rate=50 \
    --run-time=5m
```

**Test Users:**
- **ToolExecutionUser**: Standard tool execution testing (echo, calculator, current_time)
- **RateLimitingUser**: Burst traffic to test rate limiting (--users=500 --spawn-rate=100)
- **QuotaEnforcementUser**: Quota management testing (--users=200 --spawn-rate=50)
- **ParallelExecutionUser**: Batch and fallback execution testing

**Run specific user classes:**
```bash
# Rate limiting test
uv run locust -f tests/load/tool_integration_load_test.py \
    --host=http://localhost:8001 \
    --users=500 \
    --spawn-rate=100 \
    --run-time=3m \
    --headless \
    RateLimitingUser

# Quota management test
uv run locust -f tests/load/tool_integration_load_test.py \
    --host=http://localhost:8001 \
    --users=200 \
    --spawn-rate=50 \
    --run-time=5m \
    --headless \
    QuotaEnforcementUser
```

**Expected Results:**
- Concurrent executions: 1,000+
- P95 latency (lightweight): < 100ms
- P95 latency (medium): < 1s
- Error rate: < 1% (excluding quota/rate limit errors)
- Rate limiting: Proper 429 responses during bursts
- Quota enforcement: Consistent behavior with configured limits

**Monitored Metrics:**
- Tool execution rate (executions/sec)
- Rate limit hit count
- Quota exceeded count
- Successful vs failed executions
- Per-method latency percentiles

**Visualization:**
Monitor with Grafana Tool Integration Metrics dashboard at http://localhost:3000

## Running All Tests

### Sequential Execution
```bash
# Run each test for 5 minutes
for test in http_load_test websocket_load_test sustained_load_test burst_traffic_test failure_scenario_test tool_integration_load_test; do
    echo "Running $test..."
    uv run locust -f tests/load/${test}.py \
        --host=http://localhost:8001 \
        --users=1000 \
        --spawn-rate=100 \
        --run-time=5m \
        --headless
    sleep 30  # Cool-down between tests
done
```

### With Monitoring
```bash
# Start monitoring
docker-compose -f docker-compose.dev.yml up -d prometheus grafana

# Run load tests
uv run locust -f tests/load/http_load_test.py \
    --host=http://localhost:8001 \
    --users=1000 \
    --spawn-rate=100 \
    --run-time=5m
```

## Performance Targets

| Metric | Target | Critical |
|--------|--------|----------|
| HTTP RPS | 60,000+ | 40,000+ |
| WebSocket Connections | 10,000+ | 5,000+ |
| P95 Latency (Gateway) | < 50ms | < 100ms |
| P99 Latency (Gateway) | < 100ms | < 200ms |
| Tool Executions (concurrent) | 1,000+ | 500+ |
| P95 Latency (Lightweight Tools) | < 100ms | < 200ms |
| P95 Latency (Medium Tools) | < 1s | < 2s |
| Error Rate | < 0.1% | < 1% |
| CPU Usage | < 50% | < 80% |
| Memory Usage | < 4GB | < 8GB |

## Test Environment

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Network: Gigabit

### Recommended Setup
- CPU: 8+ cores
- RAM: 16GB
- Network: 10 Gigabit
- SSD storage

### Docker Setup
```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# Verify services
docker-compose -f docker-compose.dev.yml ps

# Stop services
docker-compose -f docker-compose.dev.yml down
```

## Analyzing Results

### Web UI
Access Locust web UI at: http://localhost:8089

### Command Line
Results are printed to stdout at test completion.

### Metrics Export
Export results to CSV:
```bash
uv run locust -f tests/load/http_load_test.py \
    --host=http://localhost:8001 \
    --csv=results/http_load \
    --headless
```

### Prometheus Metrics
Query gateway metrics:
```bash
curl http://localhost:8001/metrics
```

## Troubleshooting

### Connection Refused
- Verify gateway is running: `curl http://localhost:8001/api/v1/health`
- Check ports: `netstat -an | grep 8001`

### Low RPS
- Increase users: `--users=2000`
- Increase spawn rate: `--spawn-rate=200`
- Check system resources: `top`, `htop`

### High Error Rate
- Check gateway logs
- Verify database connectivity
- Check Redis connection
- Review rate limiting configuration

### Memory Issues
- Monitor with: `docker stats`
- Check for leaks: Run sustained test for 30+ minutes
- Review connection pool settings

## Best Practices

1. **Warm-up:** Always include 1-2 minute warm-up period
2. **Cool-down:** Wait 30 seconds between different test scenarios
3. **Monitoring:** Use Prometheus + Grafana for detailed metrics
4. **Baseline:** Establish baseline performance before changes
5. **Isolation:** Run tests in isolated environment
6. **Repeatability:** Run tests 3+ times for statistical significance

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Load Tests
  run: |
    uv run locust -f tests/load/http_load_test.py \
      --host=http://localhost:8001 \
      --users=100 \
      --spawn-rate=10 \
      --run-time=2m \
      --headless \
      --only-summary
```

### Performance Regression Detection
```bash
# Compare with baseline
uv run locust -f tests/load/http_load_test.py \
    --host=http://localhost:8001 \
    --users=1000 \
    --run-time=5m \
    --csv=results/current \
    --headless

# Compare with baseline results
python scripts/compare_results.py \
    results/baseline_stats.csv \
    results/current_stats.csv
```
