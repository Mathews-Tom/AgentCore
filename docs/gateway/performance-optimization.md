# Gateway Layer Performance Optimization

**Ticket:** GATE-010
**Target:** 60,000+ requests/second with <5ms p95 latency
**Status:** Implemented

## Overview

This document describes the performance optimizations implemented to achieve 60,000+ req/sec throughput with minimal latency overhead.

## Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Throughput | 60,000+ req/sec | TBD (load test) |
| Latency (p95) | <5ms routing overhead | TBD (load test) |
| Concurrent Connections | 10,000+ WebSocket | TBD (load test) |
| Memory Usage | <4GB per instance | TBD (monitoring) |
| CPU Utilization | <50% under normal load | TBD (monitoring) |

## Implemented Optimizations

### 1. Multi-Worker Deployment with Gunicorn

**Configuration:** `src/gateway/gunicorn.conf.py`

**Key Settings:**
- **Workers:** `CPU cores × 2 + 1` (dynamic based on hardware)
- **Worker Class:** `uvicorn.workers.UvicornWorker` (async support)
- **Worker Connections:** 1,000 concurrent connections per worker
- **Backlog:** 4,096 (increased from default 2,048)
- **Preload App:** Enabled (reduces memory footprint)
- **Reuse Port:** Enabled (better load balancing via SO_REUSEPORT)

**Benefit:** Linear scaling with CPU cores, automatic worker recovery

**Usage:**
```bash
uv run gunicorn gateway.main:app --config src/gateway/gunicorn.conf.py
```

### 2. HTTP Connection Pooling

**Implementation:** `src/gateway/performance/connection_pool.py`
**Integration:** `src/gateway/main.py` (lifespan management)

**Configuration:**
- **Max Connections:** 1,000 total
- **Max Keep-Alive:** 500 persistent connections
- **Keep-Alive Expiry:** 30 seconds
- **HTTP/2:** Enabled for multiplexing
- **Connection Timeout:** 5 seconds
- **Read/Write Timeout:** 30 seconds

**Benefit:** Connection reuse eliminates TCP handshake overhead (~40-100ms saved per request)

**Usage:**
```python
# Automatically available in app.state.http_pool
http_pool = request.app.state.http_pool
response = await http_pool.get("http://backend-service/api/endpoint")
```

### 3. Response Caching

**Implementation:** `src/gateway/performance/response_cache.py`
**Integration:** `src/gateway/main.py` (lifespan management)

**Configuration:**
- **Cache Size:** 10,000 responses
- **TTL:** 300 seconds (5 minutes)
- **Eviction Strategy:** LRU (Least Recently Used)
- **Cached Methods:** GET only
- **Cached Status Codes:** 200, 301, 302

**Benefit:** Eliminates backend calls for cached responses (100-500ms saved per cache hit)

**Cache Hit Rate Target:** >70% for common endpoints

**Usage:**
```python
# Automatically available in app.state.response_cache
cache = request.app.state.response_cache

# Check cache
cached = cache.get("GET", "/api/v1/resource/123")
if cached:
    return Response(content=cached.content, status_code=cached.status_code)

# Store in cache
cache.set("GET", "/api/v1/resource/123", 200, headers, content)
```

### 4. Uvicorn Worker Optimization

**Configuration:** `src/gateway/deployment.py`

**Key Settings:**
- **Event Loop:** `uvloop` (70% faster than asyncio)
- **HTTP Parser:** `httptools` (faster than h11)
- **Backlog:** 4,096 socket backlog
- **Keep-Alive Timeout:** 5 seconds
- **Max Requests:** 10,000 per worker (automatic restart prevents memory leaks)

**Environment Variables:**
```bash
export UVICORN_LOOP=uvloop
export UVICORN_HTTP=httptools
export UVICORN_BACKLOG=4096
export UVICORN_TIMEOUT_KEEP_ALIVE=5
```

**Benefit:** uvloop + httptools provide up to 2× performance improvement over defaults

### 5. Operating System Tuning

**Required:** Apply these kernel parameters for production deployment

```bash
# Increase listen backlog
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=8192

# Optimize ephemeral port range
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Enable TIME_WAIT socket reuse
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# Faster FIN timeout
sudo sysctl -w net.ipv4.tcp_fin_timeout=15

# TCP keep-alive tuning
sudo sysctl -w net.ipv4.tcp_keepalive_time=300
sudo sysctl -w net.ipv4.tcp_keepalive_probes=3
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=15

# Increase max open files
sudo sysctl -w fs.file-max=2097152
ulimit -n 65535
```

**Benefit:** Prevents socket exhaustion, improves connection handling under high load

### 6. Middleware Optimization

**Order:** Middleware are ordered for performance (fastest first)

1. **CORS** - Simple header checks
2. **Security Headers** - Static header injection
3. **Input Validation** - Request validation
4. **Transformation** - Trace ID injection
5. **Cache Control** - ETag handling
6. **Compression** - Gzip/Brotli (only for large responses)
7. **Rate Limiting** - Redis-based (deferred to avoid startup dependency)
8. **Logging** - Async structured logging
9. **Metrics** - Prometheus instrumentation

**Optimization:** Response cache bypasses middleware for cache hits

**Benefit:** Minimal middleware overhead (<1ms per request)

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

# Install dependencies
COPY pyproject.toml .
RUN pip install uv && uv pip install .

# Copy application
COPY src/ src/

# Apply OS tuning (requires privileged container)
RUN echo "net.core.somaxconn=65535" >> /etc/sysctl.conf

# Run with Gunicorn
CMD ["uv", "run", "gunicorn", "gateway.main:app", "--config", "src/gateway/gunicorn.conf.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentcore-gateway
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: gateway
        image: agentcore/gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: GUNICORN_WORKERS
          value: "9"  # Adjust based on node CPU
        - name: UVICORN_LOOP
          value: "uvloop"
        - name: UVICORN_HTTP
          value: "httptools"
        resources:
          requests:
            memory: "2Gi"
            cpu: "2000m"
          limits:
            memory: "4Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 3
```

## Load Testing

### Running Load Tests

**Basic Load Test:**
```bash
uv run locust -f tests/load/gateway_performance_test.py --host http://localhost:8080
```

**High-Throughput Test (60k+ req/sec target):**
```bash
uv run locust -f tests/load/gateway_performance_test.py \
    --host http://localhost:8080 \
    --users 5000 \
    --spawn-rate 100 \
    --run-time 5m \
    --headless
```

**Expected Results:**
- **RPS:** 60,000+ requests/second
- **Latency (p95):** <5ms routing overhead
- **Error Rate:** <0.1%

### Interpreting Results

**Good Performance:**
```
Total Requests: 18,000,000
Requests/sec: 60,000.00
Avg Response Time: 2.50 ms
p95 Latency: 4.80 ms
Error Rate: 0.05%
```

**Poor Performance (needs optimization):**
```
Total Requests: 9,000,000
Requests/sec: 30,000.00
Avg Response Time: 15.00 ms
p95 Latency: 50.00 ms
Error Rate: 2.50%
```

**Troubleshooting:**
- If RPS < 60k: Increase workers, check OS limits, enable caching
- If latency > 5ms: Profile middleware, optimize slow endpoints, enable HTTP/2
- If error rate > 0.1%: Check backend service health, rate limit configuration

## Monitoring

### Prometheus Metrics

**Endpoint:** `http://localhost:8080/metrics`

**Key Metrics:**
- `gateway_requests_total` - Total requests processed
- `gateway_request_duration_seconds` - Request latency histogram
- `gateway_active_connections` - Current active connections
- `gateway_cache_hits_total` - Response cache hit count
- `gateway_cache_misses_total` - Response cache miss count
- `gateway_pool_connections_active` - Active HTTP pool connections

### Grafana Dashboard

**Queries:**
```promql
# Requests per second
rate(gateway_requests_total[1m])

# p95 latency
histogram_quantile(0.95, rate(gateway_request_duration_seconds_bucket[1m]))

# Cache hit rate
rate(gateway_cache_hits_total[1m]) / (rate(gateway_cache_hits_total[1m]) + rate(gateway_cache_misses_total[1m]))

# Error rate
rate(gateway_requests_total{status=~"5.."}[1m]) / rate(gateway_requests_total[1m])
```

## Performance Checklist

Before deploying to production:

- [ ] Gunicorn configured with optimal worker count
- [ ] HTTP connection pooling enabled
- [ ] Response caching enabled for GET endpoints
- [ ] Uvicorn using uvloop and httptools
- [ ] OS kernel parameters tuned
- [ ] File descriptor limits increased (ulimit -n 65535)
- [ ] Load testing validates 60k+ req/sec
- [ ] p95 latency <5ms confirmed
- [ ] Monitoring dashboards configured
- [ ] Alerting rules defined for performance degradation
- [ ] Horizontal scaling tested (multiple replicas)

## Troubleshooting Guide

### Problem: Low Throughput (<60k req/sec)

**Diagnosis:**
```bash
# Check active workers
ps aux | grep gunicorn

# Check open connections
netstat -an | grep ESTABLISHED | wc -l

# Check system limits
ulimit -a
```

**Solutions:**
1. Increase Gunicorn workers
2. Verify OS tuning applied
3. Enable response caching
4. Optimize slow middleware

### Problem: High Latency (>5ms p95)

**Diagnosis:**
```bash
# Profile requests
curl -w "@curl-format.txt" http://localhost:8080/api/v1/resource

# Check middleware overhead
# (review logs for slow middleware)
```

**Solutions:**
1. Enable response caching
2. Optimize middleware order
3. Use HTTP/2 and connection pooling
4. Profile and optimize slow backend calls

### Problem: Connection Exhaustion

**Diagnosis:**
```bash
# Check socket states
ss -s

# Check TIME_WAIT sockets
ss -tan | grep TIME_WAIT | wc -l
```

**Solutions:**
1. Enable tcp_tw_reuse
2. Reduce tcp_fin_timeout
3. Increase ephemeral port range
4. Use connection pooling

## References

- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/settings.html)
- [Uvicorn Performance Tips](https://www.uvicorn.org/deployment/)
- [Linux Kernel Tuning](https://www.kernel.org/doc/Documentation/networking/ip-sysctl.txt)
- [FastAPI Performance](https://fastapi.tiangolo.com/deployment/concepts/)
- [HTTP/2 Performance](https://http2.github.io/)

## Changelog

- **2025-11-07:** Initial implementation (GATE-010)
  - Gunicorn multi-worker configuration
  - HTTP connection pooling
  - Response caching
  - Uvicorn worker optimization
  - Load testing framework
