# Gateway Monitoring & Observability

This directory contains configuration files for monitoring and observability infrastructure.

## Directory Structure

```
deploy/
├── grafana/           # Grafana dashboard JSON files
│   ├── gateway-overview-dashboard.json
│   ├── service-performance-dashboard.json
│   ├── security-dashboard.json
│   └── realtime-dashboard.json
└── prometheus/        # Prometheus configuration
    └── alert-rules.yml
```

## Dashboards

### Gateway Overview Dashboard
- Request rate and throughput
- Request latency (p95)
- Active requests gauge
- Error rate tracking
- WebSocket connections
- Health status
- Top endpoints by traffic
- Rate limit hits
- Backend service latency

### Service Performance Dashboard
- Request latency distribution (heatmap)
- Latency percentiles (p50, p90, p95, p99)
- Throughput by HTTP method
- Request/response size distribution
- Backend service metrics
- Circuit breaker state
- Cache hit rate
- Compression savings

### Security Dashboard
- Authentication success/failure rates
- Authentication failure reasons
- Authorization denials
- Rate limit hits by type
- DDoS blocks and suspicious patterns
- TLS handshakes and errors
- Active sessions
- Session duration

### Real-time Dashboard
- Current request rate (live)
- Current latency (p95)
- Active requests count
- Error rate percentage
- Live request/latency graphs
- WebSocket connections and messages
- Status code distribution
- Top endpoints (last minute)
- Component health status
- Recent errors log

## Prometheus Alerts

The `prometheus/alert-rules.yml` file contains 18 alerting rules:

### Critical Alerts
- **HighErrorRate**: Error rate > 5% for 2 minutes
- **CriticalLatency**: P95 latency > 1.0s for 2 minutes
- **DDoSAttackDetected**: DDoS blocks > 10/sec
- **SuspiciousAuthActivity**: Auth failures > 100/sec
- **GatewayUnhealthy**: Overall health check failing
- **GatewayNotReady**: Readiness check failing

### Warning Alerts
- **HighLatency**: P95 latency > 0.5s for 5 minutes
- **CircuitBreakerOpen**: Circuit breaker opened for backend service
- **HighAuthFailureRate**: Auth failures > 30% for 5 minutes
- **ComponentUnhealthy**: Individual component unhealthy
- **HighRateLimitHits**: Rate limit hits > 100/sec for 5 minutes
- **BackendServiceErrors**: Backend error rate > 10%
- **BackendServiceHighLatency**: Backend P95 latency > 2.0s
- **WebSocketConnectionsHigh**: Active connections > 8000
- **HighMemoryUsage**: Memory usage > 90%
- **TLSHandshakeErrors**: TLS errors > 10/sec

### Info Alerts
- **LowCacheHitRate**: Cache hit rate < 50%
- **RequestRateSpike**: Request rate 2x average

## Usage

### Import Grafana Dashboards

1. Open Grafana UI
2. Navigate to Dashboards → Import
3. Upload or paste the JSON content
4. Select your Prometheus data source
5. Click Import

### Configure Prometheus Alerts

1. Add to your `prometheus.yml`:
```yaml
rule_files:
  - "alert-rules.yml"
```

2. Configure Alertmanager:
```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
```

3. Reload Prometheus configuration:
```bash
curl -X POST http://localhost:9090/-/reload
```

## Metrics Endpoint

The gateway exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

## Health Checks

- **Health**: `GET /health` - Comprehensive health check with dependencies
- **Readiness**: `GET /ready` - Readiness for traffic (critical dependencies)
- **Liveness**: `GET /live` - Simple liveness probe

## Distributed Tracing

The gateway supports OpenTelemetry distributed tracing:

### Configuration

Set environment variables:
```bash
GATEWAY_TRACING_ENABLED=true
GATEWAY_TRACING_EXPORT_ENDPOINT=http://jaeger:4317
GATEWAY_TRACING_SAMPLE_RATE=0.1
```

### Supported Exporters

- OTLP (Jaeger, Tempo, etc.)
- Console (for debugging)

### Trace Attributes

Automatic attributes:
- `http.method`
- `http.url`
- `http.status_code`
- `trace_id`
- `span_id`

Custom attributes can be added via the tracing API.

## Development

### Running Locally

```bash
# Start Prometheus
docker run -p 9090:9090 -v ./prometheus:/etc/prometheus prom/prometheus

# Start Grafana
docker run -p 3000:3000 grafana/grafana

# Start Jaeger (optional, for tracing)
docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

### Testing Metrics

```python
from gateway.monitoring.metrics import REQUEST_COUNT

# Record a request
REQUEST_COUNT.labels(
    method="GET",
    path="/api/agents",
    status_code="200"
).inc()
```

### Testing Tracing

```python
from gateway.monitoring.tracing import get_tracer

tracer = get_tracer()
with tracer.start_as_current_span("operation"):
    # Your code here
    pass
```

## Production Deployment

### Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gateway'
    static_configs:
      - targets: ['gateway:8080']
```

### Grafana

1. Install dashboards via ConfigMap or provisioning
2. Configure Prometheus data source
3. Set up SMTP for alert notifications
4. Configure user access and permissions

### Alertmanager

```yaml
# alertmanager.yml
route:
  receiver: 'team-notifications'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

receivers:
  - name: 'team-notifications'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#alerts'
```

## Troubleshooting

### Metrics not showing in Grafana

1. Check Prometheus is scraping the gateway:
   ```bash
   curl http://prometheus:9090/api/v1/targets
   ```

2. Verify metrics endpoint is accessible:
   ```bash
   curl http://gateway:8080/metrics
   ```

3. Check Grafana data source configuration

### Alerts not firing

1. Verify alert rules are loaded:
   ```bash
   curl http://prometheus:9090/api/v1/rules
   ```

2. Check Alertmanager connectivity:
   ```bash
   curl http://prometheus:9090/api/v1/alertmanagers
   ```

### Traces not appearing

1. Check OTLP endpoint is reachable
2. Verify `TRACING_ENABLED=true`
3. Check trace sampling rate
4. Review gateway logs for tracing errors
