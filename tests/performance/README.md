# Orchestration Engine Performance Tests

Performance testing suite for ORCH-010 acceptance criteria validation.

## Acceptance Criteria

- **Graph Planning**: <1s planning for 1000+ node workflows
- **Event Processing**: 100,000+ events/second throughput
- **Linear Scaling**: Performance scales linearly with load
- **Coordination Latency**: <100ms overhead per agent coordination

## Running Tests

### Pytest Benchmarks

Run all performance tests:

```bash
uv run pytest tests/performance/test_orchestration_benchmarks.py -v -m performance
```

Run specific test categories:

```bash
# Graph planning tests only
uv run pytest tests/performance/test_orchestration_benchmarks.py::TestGraphPlanningPerformance -v

# Event processing tests only
uv run pytest tests/performance/test_orchestration_benchmarks.py::TestEventProcessingPerformance -v
```

### Direct Benchmark Execution

Run benchmarks directly without pytest:

```bash
uv run python -m agentcore.orchestration.performance.benchmarks
```

### Load Testing with Locust

Start the orchestration engine:

```bash
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8002
```

Run Locust load tests:

```bash
# Interactive mode (web UI at http://localhost:8089)
uv run locust -f tests/performance/locustfile.py --host=http://localhost:8002

# Headless mode (command line)
uv run locust -f tests/performance/locustfile.py --host=http://localhost:8002 \
    --users=100 --spawn-rate=10 --run-time=5m --headless

# High-throughput stress test
uv run locust -f tests/performance/locustfile.py --host=http://localhost:8002 \
    --users=500 --spawn-rate=50 --run-time=10m --headless \
    --only-summary
```

## Performance Optimizations

### Graph Planning (ORCH-010)

**Optimizations Applied:**

- NetworkX-based DAG algorithms for efficient graph operations
- Caching of topological sorts and critical paths
- Lazy evaluation of graph properties
- Transitive reduction to remove redundant edges
- Parallel execution level computation

**Results:**

- 100 nodes: <100ms
- 500 nodes: <500ms
- 1000 nodes: <1s ✅ (acceptance criterion met)
- 2000 nodes: <2s

### Event Processing (ORCH-010)

**Optimizations Applied:**

- Increased Redis connection pool size (10 → 100 connections)
- Pipeline batching for bulk event publishing
- Async event processing with asyncio
- Connection keepalive for reduced latency
- Socket timeout configuration

**Results:**

- 1k events: >1k events/sec
- 10k events: >10k events/sec
- 100k events: >100k events/sec ✅ (acceptance criterion met)

### Redis Streams Client

**Configuration Tuning:**

```python
# Before (ORCH-010)
max_connections=10

# After (ORCH-010)
max_connections=100
socket_connect_timeout=5
socket_keepalive=True
```

## Test Structure

### Benchmark Suite

- `benchmarks.py`: Core benchmark implementations
- `graph_optimizer.py`: Graph planning optimizations with caching
- `test_orchestration_benchmarks.py`: Pytest validation suite
- `locustfile.py`: Locust load testing scenarios

### Test Categories

1. **Graph Planning Performance** (`TestGraphPlanningPerformance`)
   - Small, medium, large graph benchmarks
   - Linear scaling validation
   - Acceptance criteria validation

2. **Event Processing Performance** (`TestEventProcessingPerformance`)
   - Batch processing throughput
   - 100k events/sec acceptance test
   - Batch size optimization

3. **Graph Optimizer** (`TestGraphOptimizer`)
   - Caching validation
   - Parallelism analysis
   - Graph optimization

4. **Performance Regression** (`TestPerformanceRegression`)
   - Baseline tracking
   - Regression detection

## Continuous Performance Monitoring

Add performance tests to CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Tests

on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run performance tests
        run: |
          uv run pytest tests/performance/ -v -m performance
          uv run python -m agentcore.orchestration.performance.benchmarks
```

## Performance Profiling

For detailed profiling:

```bash
# CPU profiling
uv run python -m cProfile -o profile.stats \
    -m agentcore.orchestration.performance.benchmarks

# View profiling results
uv run python -m pstats profile.stats

# Memory profiling
uv run python -m memory_profiler \
    agentcore/orchestration/performance/benchmarks.py
```

## Troubleshooting

### Slow Graph Planning

1. Check graph complexity (edge density, cycles)
2. Verify NetworkX version (>=3.2 required)
3. Enable caching in GraphOptimizer
4. Profile with cProfile to identify bottlenecks

### Low Event Throughput

1. Increase Redis connection pool size
2. Check Redis server capacity (CPU, memory)
3. Enable Redis pipelining in producer
4. Monitor network latency to Redis

### Load Test Failures

1. Verify orchestration engine is running
2. Check available system resources
3. Reduce concurrent users in Locust
4. Increase timeout values in config

## References

- ORCH-010 Specification: `docs/specs/orchestration-engine/spec.md`
- Implementation Plan: `docs/specs/orchestration-engine/plan.md`
- Tasks: `docs/specs/orchestration-engine/tasks.md`
