# Load Testing for AgentCore Gateway

Comprehensive load testing suite for gateway layer performance validation.

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

## Running All Tests

### Sequential Execution
```bash
# Run each test for 5 minutes
for test in http_load_test websocket_load_test sustained_load_test burst_traffic_test failure_scenario_test; do
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
| P95 Latency | < 50ms | < 100ms |
| P99 Latency | < 100ms | < 200ms |
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
