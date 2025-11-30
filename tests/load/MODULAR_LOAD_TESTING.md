# Modular Agent Core Load Testing (MOD-028)

Load testing implementation for validating NFR-1.4: "The system SHALL handle at least 100 concurrent modular executions per instance."

## Overview

This load test validates the modular agent core's ability to handle 100+ concurrent `modular.solve` executions with:
- **Success rate >95%** under sustained load
- **p95 latency <3x baseline** (baseline: ~1500ms from performance benchmarks)
- **No memory leaks** or resource exhaustion
- **10-minute sustained load** without degradation

## Files

- `test_modular_load.py` - Locust load test implementation
- `run_modular_load_test.sh` - Test runner script with scenario presets
- `MODULAR_LOAD_TESTING.md` - This documentation

## Test Scenarios

### Quick Validation
**Purpose:** Fast validation during development
**Configuration:** 10 users, 2 minutes
```bash
./tests/load/run_modular_load_test.sh quick
```

### Standard Load Test
**Purpose:** Regular load testing
**Configuration:** 50 users, 5 minutes
```bash
./tests/load/run_modular_load_test.sh standard
```

### Full Load Test (Default)
**Purpose:** MOD-028 acceptance criteria validation
**Configuration:** 100 users, 10 minutes
```bash
./tests/load/run_modular_load_test.sh full
# or simply:
./tests/load/run_modular_load_test.sh
```

### Stress Test
**Purpose:** Identify breaking points
**Configuration:** 150 users, 15 minutes
```bash
./tests/load/run_modular_load_test.sh stress
```

## Direct Locust Usage

### Headless Mode (CI/CD)
```bash
uv run locust -f tests/load/test_modular_load.py \
    --host=http://localhost:8001 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=10m \
    --headless \
    --csv=results/modular_load \
    --html=results/modular_load_report.html
```

### Web UI Mode (Interactive)
```bash
# Start Locust web UI
uv run locust -f tests/load/test_modular_load.py \
    --host=http://localhost:8001

# Open browser to http://localhost:8089
# Configure users and spawn rate in web UI
```

## Test Queries

The load test uses realistic query distribution:

| Complexity | Weight | Count | Example |
|------------|--------|-------|---------|
| **Simple** | 60% | 5 queries | "What is the capital of France?" |
| **Moderate** | 30% | 5 queries | "Compare populations of NYC and LA" |
| **Complex** | 10% | 2 queries | "Research top 3 programming languages..." |

### Expected Latency Ranges

- **Simple:** 500-2000ms
- **Moderate:** 1000-4000ms
- **Complex:** 3000-8000ms

## Acceptance Criteria Validation

### 1. Concurrent Executions (100+)
✓ Test spawns 100 concurrent users
✓ Each user executes queries with 1-5s think time
✓ Sustained load for 10 minutes

### 2. Success Rate (>95%)
```
Target: >95% successful requests
Tracked: (successful_requests / total_requests) * 100
```

Validation includes:
- HTTP 200 response
- Valid JSON-RPC structure
- `result.answer` present
- `execution_trace.verification_passed == true`

### 3. p95 Latency (<3x Baseline)
```
Baseline p95: ~1500ms (from test_modular_performance.py)
Target p95: <4500ms (3x baseline)
Tracked: 95th percentile response time
```

### 4. No Memory Leaks
Monitor with:
```bash
# In separate terminal
docker stats

# Or for standalone server
ps aux | grep uvicorn
```

### 5. No Resource Exhaustion
Monitor Prometheus metrics:
```bash
curl http://localhost:8001/metrics | grep modular
```

## Results Analysis

### Locust Output

After test completion, results are saved to `tests/load/results/`:

- `modular_load_<scenario>_<timestamp>_stats.csv` - Request statistics
- `modular_load_<scenario>_<timestamp>_failures.csv` - Failure details
- `modular_load_<scenario>_<timestamp>_exceptions.csv` - Exception details
- `modular_load_<scenario>_<timestamp>.html` - HTML report with charts

### Key Metrics

```
REQUEST SUMMARY:
  Total Requests:     12,450
  Successful:         12,012
  Failed:             438
  Success Rate:       96.48%

LATENCY METRICS:
  Mean:               2,341ms
  p95:                4,123ms
  p99:                5,876ms

THROUGHPUT:
  Requests/sec:       20.75

NFR TARGET VALIDATION:
  Success Rate:       96.48% ✓ PASS (Target: >95%)
  p95 Latency:        4,123ms ✓ PASS (Target: <4500ms)
  Overall:            ✓ ALL TARGETS MET
```

## Prerequisites

### 1. Server Running
```bash
# Start AgentCore server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# Verify health
curl http://localhost:8001/api/v1/health
```

### 2. Dependencies Installed
```bash
# Locust should be in pyproject.toml dependencies
uv add locust

# Verify installation
uv run locust --version
```

### 3. Resources Available
**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Network: Stable connection

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- SSD storage

## Monitoring During Test

### Server Logs
```bash
# Follow server logs
tail -f logs/agentcore.log
```

### System Resources
```bash
# CPU and memory
htop

# Docker stats (if using Docker)
docker stats
```

### Prometheus Metrics
```bash
# View all metrics
curl http://localhost:8001/metrics

# Filter modular metrics
curl http://localhost:8001/metrics | grep modular_

# Key metrics to watch:
# - modular_requests_total
# - modular_request_duration_seconds
# - modular_active_executions
# - modular_verification_confidence
```

### Grafana Dashboards
If using Grafana (from MOD-027):
- Navigate to http://localhost:3000
- Open "Modular Agent Health" dashboard
- Monitor real-time metrics during load test

## Troubleshooting

### High Failure Rate (>5%)

**Possible Causes:**
- Server overloaded (CPU/memory exhaustion)
- Database connection pool exhausted
- LLM provider rate limiting
- Network issues

**Solutions:**
1. Check server logs for errors
2. Increase connection pool size
3. Scale server resources
4. Reduce concurrent users

### High Latency (p95 >4500ms)

**Possible Causes:**
- Slow LLM responses
- Database query performance
- Module coordination overhead
- Network latency

**Solutions:**
1. Review MOD-020 optimization tasks
2. Enable response caching
3. Optimize module transitions
4. Profile with OpenTelemetry traces

### Memory Leaks

**Symptoms:**
- Memory usage grows continuously
- Server becomes unresponsive
- Out of memory errors

**Solutions:**
1. Check for unclosed connections
2. Review async task cleanup
3. Monitor with `memory_profiler`:
   ```bash
   uv add memory_profiler
   uv run python -m memory_profiler your_script.py
   ```

### Connection Refused

**Causes:**
- Server not running
- Wrong host/port
- Firewall blocking

**Solutions:**
```bash
# Verify server running
curl http://localhost:8001/api/v1/health

# Check port
netstat -an | grep 8001

# Try localhost vs 127.0.0.1
curl http://127.0.0.1:8001/api/v1/health
```

## Performance Baselines

From `test_modular_performance.py` (MOD-025):

| Metric | Baseline | Modular | Target |
|--------|----------|---------|--------|
| Success Rate | 77% | 88% | +15% |
| Tool Accuracy | 75% | 89% | +10% |
| p95 Latency | 1500ms | 2400ms | <2x |
| Cost per Query | $0.036 | $0.025 | -30% |
| Error Recovery | 65% | 82% | >80% |

**Load Test Expectations:**
- Under 100 concurrent users, expect p95 latency to increase to ~4000ms
- Success rate should remain >95% despite increased load
- Throughput should be 15-25 requests/second

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Modular Load Test

on:
  push:
    branches: [main]
  pull_request:
    paths:
      - 'src/agentcore/modular/**'
      - 'tests/load/test_modular_load.py'

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Start Services
        run: |
          docker-compose -f docker-compose.dev.yml up -d
          sleep 10

      - name: Run Quick Load Test
        run: |
          ./tests/load/run_modular_load_test.sh quick

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: tests/load/results/
```

## Performance Regression Detection

### Establish Baseline
```bash
# Run full load test
./tests/load/run_modular_load_test.sh full

# Save baseline results
cp tests/load/results/modular_load_full_*_stats.csv \
   tests/load/baselines/modular_load_baseline.csv
```

### Compare Against Baseline
```bash
# Run new load test
./tests/load/run_modular_load_test.sh full

# Compare results
python scripts/compare_load_test_results.py \
    tests/load/baselines/modular_load_baseline.csv \
    tests/load/results/modular_load_full_*_stats.csv
```

## Related Documentation

- **Spec:** `docs/specs/modular-agent-core/spec.md`
- **Plan:** `docs/specs/modular-agent-core/plan.md`
- **Tasks:** `docs/specs/modular-agent-core/tasks.md` (MOD-028)
- **Performance Benchmarks:** `tests/benchmarks/test_modular_performance.py` (MOD-025)
- **Integration Tests:** `tests/integration/modular/test_pipeline.py` (MOD-019)
- **Optimization:** `src/agentcore/modular/optimizer.py` (MOD-020)
- **Metrics:** `src/agentcore/modular/metrics.py` (MOD-026)

## Success Criteria Checklist

- [ ] Load test executes successfully with 100 concurrent users
- [ ] Test runs for sustained 10-minute period
- [ ] Success rate >95% achieved
- [ ] p95 latency <3x baseline (4500ms) achieved
- [ ] No memory leaks detected (stable memory over 10 minutes)
- [ ] No resource exhaustion (CPU/connections stay below limits)
- [ ] Comprehensive load test report generated
- [ ] Results documented and compared to baseline
- [ ] All NFR targets validated and documented

## Next Steps

After successful load test completion:

1. **Document Results:** Update MOD-026 metrics dashboard with load test findings
2. **Performance Tuning:** If targets not met, review MOD-020 optimization tasks
3. **Scalability Testing:** Consider horizontal scaling tests with multiple instances
4. **Stress Testing:** Run stress scenario to identify breaking points
5. **Production Readiness:** Mark MOD-028 complete and proceed to MOD-029 (Error Recovery Testing)
