## Integration Layer Load Testing

Comprehensive load testing and performance validation for the Integration Layer.

### Test Coverage

**Load Tests (Locust):**
- `integration_layer_load_test.py` - High-throughput load testing (10,000+ req/s)
  - LLM provider requests (40% of load)
  - Webhook registration/management (20% of load)
  - Event publishing (30% of load)
  - Storage operations (2% of load)

**Performance Benchmarks (pytest-benchmark):**
- `integration_performance_benchmarks.py` - Component-level performance validation
  - Webhook manager: 10,000 webhook registrations
  - Event publisher: 50,000 event publications
  - Delivery service: 5,000 concurrent deliveries
  - Resource utilization: Memory and CPU efficiency

### Performance Targets

**Throughput:**
- ✅ 10,000+ external requests per second
- ✅ 1,000+ webhook registrations per second
- ✅ 1,000+ events published per second

**Latency:**
- ✅ <100ms P95 latency for LLM requests
- ✅ <50ms P95 latency for webhook operations
- ✅ <10ms P95 latency for event publishing

**Reliability:**
- ✅ 99.9%+ success rate under load
- ✅ Graceful degradation under extreme load
- ✅ No memory leaks during sustained operations

**Resource Efficiency:**
- ✅ <500MB peak memory usage for 100K operations
- ✅ <80% CPU usage under sustained load
- ✅ <200MB memory growth for 10K webhook registrations

### Running Load Tests

**Prerequisites:**
```bash
# Install dependencies
uv add locust pytest-benchmark memory-profiler psutil

# Start AgentCore server
uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001
```

**Run Locust Load Tests:**

```bash
# Run integration layer load test
uv run locust -f tests/load/integration_layer_load_test.py

# Or headless mode with specific parameters
uv run locust -f tests/load/integration_layer_load_test.py \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --host http://localhost:8001

# High throughput test (target 10k+ req/s)
uv run locust -f tests/load/integration_layer_load_test.py \
  --headless \
  --users 1000 \
  --spawn-rate 100 \
  --run-time 2m \
  --host http://localhost:8001
```

**Run Performance Benchmarks:**

```bash
# Run all benchmarks
uv run pytest tests/load/integration_performance_benchmarks.py -v -m benchmark -s

# Run specific benchmark
uv run pytest tests/load/integration_performance_benchmarks.py::TestWebhookManagerPerformance::test_high_volume_registration -v -s

# Run with memory profiling
uv run pytest tests/load/integration_performance_benchmarks.py::TestResourceUtilization::test_memory_efficiency_under_load -v -s
```

### Test Scenarios

#### 1. Baseline Performance Test
**Goal:** Establish baseline performance metrics
**Users:** 50 | **Duration:** 5 minutes
```bash
uv run locust -f tests/load/integration_layer_load_test.py \
  --headless --users 50 --spawn-rate 5 --run-time 5m
```

#### 2. Target Throughput Test
**Goal:** Validate 10,000+ req/s target
**Users:** 1000 | **Duration:** 2 minutes
```bash
uv run locust -f tests/load/integration_layer_load_test.py \
  --headless --users 1000 --spawn-rate 100 --run-time 2m
```

#### 3. Sustained Load Test
**Goal:** Validate stability under sustained load
**Users:** 500 | **Duration:** 30 minutes
```bash
uv run locust -f tests/load/integration_layer_load_test.py \
  --headless --users 500 --spawn-rate 50 --run-time 30m
```

#### 4. Spike Test
**Goal:** Validate handling of traffic spikes
**Users:** 2000 | **Duration:** 1 minute
```bash
uv run locust -f tests/load/integration_layer_load_test.py \
  --headless --users 2000 --spawn-rate 500 --run-time 1m
```

### Interpreting Results

**Locust Web UI:**
- Access at `http://localhost:8089` when running without `--headless`
- Monitor real-time metrics: RPS, response times, failure rate
- View charts for requests/s and response time over time

**Performance Benchmarks:**
- Check console output for detailed metrics
- Compare against performance targets
- Look for memory leaks (growing memory delta)
- Monitor CPU usage patterns

**Success Criteria:**
- ✅ All load tests complete without errors
- ✅ Target throughput achieved (10,000+ req/s)
- ✅ P95 latency within targets
- ✅ Success rate ≥99.9%
- ✅ No memory leaks detected
- ✅ CPU usage remains reasonable (<80%)

### Optimization Tips

**If throughput is below target:**
1. Increase `max_concurrent_deliveries` in webhook config
2. Tune event queue and batch sizes
3. Enable connection pooling for HTTP clients
4. Consider horizontal scaling (multiple instances)

**If latency is high:**
1. Check database query performance
2. Enable caching for frequently accessed data
3. Optimize webhook delivery logic
4. Review event processing batching

**If memory usage is high:**
1. Reduce event queue size
2. Implement event cleanup for old deliveries
3. Tune garbage collection settings
4. Review object retention patterns

### Continuous Monitoring

**Production Metrics:**
- Monitor throughput: `/metrics` endpoint (Prometheus format)
- Track latency percentiles (P50, P95, P99)
- Alert on error rate spikes (>0.1%)
- Monitor resource utilization (memory, CPU)

**Automated Testing:**
- Run load tests in CI/CD pipeline
- Set performance regression thresholds
- Alert on performance degradation
- Track historical performance trends

### Troubleshooting

**"Connection refused" errors:**
- Ensure AgentCore server is running
- Check port 8001 is not in use
- Verify firewall/security group settings

**High error rates:**
- Check server logs for errors
- Verify database connectivity
- Monitor server resource usage
- Consider rate limiting configuration

**Inconsistent results:**
- Warm up server before measuring
- Increase test duration for stability
- Isolate test environment
- Disable other background processes

### References

- Locust Documentation: https://docs.locust.io/
- pytest-benchmark: https://pytest-benchmark.readthedocs.io/
- Integration Layer Spec: `docs/specs/integration-layer/spec.md`
- Performance Targets: See INT-011 acceptance criteria
