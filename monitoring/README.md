# AgentCore Monitoring

This directory contains monitoring configurations for AgentCore services including Grafana dashboards and Prometheus alerting rules.

## Overview

AgentCore uses Prometheus for metrics collection and Grafana for visualization. The monitoring stack provides:

- **Real-time metrics**: Request rates, latency, errors, resource usage
- **Historical analysis**: Performance trends over time
- **Alerting**: Automated notifications for critical issues
- **Cost tracking**: Token usage and estimated costs

## Directory Structure

```
monitoring/
├── grafana/
│   ├── dspy-dashboard.json         # DSPy optimization service dashboard
│   └── reasoning-dashboard.json    # Reasoning framework dashboard
├── prometheus-rules.yaml           # DSPy alerting rules
└── prometheus-reasoning-rules.yaml # Reasoning framework alerting rules
```

## Reasoning Framework Metrics

### Collected Metrics

The reasoning framework exposes the following Prometheus metrics:

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `reasoning_bounded_context_requests_total` | Counter | Total reasoning requests | `status` (success/error) |
| `reasoning_bounded_context_errors_total` | Counter | Total errors by type | `error_type` |
| `reasoning_bounded_context_llm_failures_total` | Counter | LLM service failures | - |
| `reasoning_bounded_context_duration_seconds` | Histogram | Request duration | - |
| `reasoning_bounded_context_tokens_total` | Histogram | Tokens used per request | - |
| `reasoning_bounded_context_compute_savings_pct` | Histogram | Compute savings percentage | - |
| `reasoning_bounded_context_iterations_total` | Histogram | Iterations per request | - |

### Dashboard Panels

The reasoning dashboard (`grafana/reasoning-dashboard.json`) includes:

1. **Request Success Rate**: Overall success rate with color-coded thresholds
2. **Total Requests (1h)**: Request volume over last hour
3. **Average Compute Savings**: Efficiency metric for bounded context strategy
4. **LLM Failures (24h)**: LLM service reliability indicator
5. **Request Rate by Status**: Success vs error trends
6. **Response Latency (Percentiles)**: p50, p95, p99 latency
7. **Token Usage Distribution**: Token consumption patterns
8. **Average Tokens per Request**: Token efficiency over time
9. **Compute Savings Distribution**: Savings achieved by bounded context
10. **Iterations per Request**: Iteration count distribution
11. **Error Types Breakdown**: Error classification
12. **LLM Failure Rate**: LLM reliability trends
13. **Token Cost Estimation**: Estimated cost based on GPT-4 pricing
14. **Request Success/Error Ratio**: Visual ratio of success vs errors
15. **Duration Heatmap**: Latency distribution heatmap
16. **Average Compute Savings Over Time**: Historical savings trend

### Alerting Rules

The reasoning alerting rules (`prometheus-reasoning-rules.yaml`) include:

**Error Rate Alerts:**
- `ReasoningHighErrorRate`: Warning at >10% error rate
- `ReasoningCriticalErrorRate`: Critical at >25% error rate

**Performance Alerts:**
- `ReasoningSlowResponseTime`: Warning at p95 >30s
- `ReasoningVerySlowResponseTime`: Critical at p95 >60s

**LLM Reliability Alerts:**
- `ReasoningHighLLMFailureRate`: Warning at >0.1 failures/sec
- `ReasoningCriticalLLMFailureRate`: Critical at >0.5 failures/sec

**Error Type Alerts:**
- `ReasoningHighValidationErrors`: Warning at >0.5 validation errors/sec
- `ReasoningHighTimeoutErrors`: Warning at >0.2 timeout errors/sec

**Cost & Efficiency Alerts:**
- `ReasoningHighTokenUsage`: Warning at p95 >50K tokens
- `ReasoningVeryHighTokenUsage`: Critical at p95 >100K tokens (cost concern)
- `ReasoningLowComputeSavings`: Info at <20% compute savings

**Operational Alerts:**
- `ReasoningHighIterationCount`: Warning at p95 >20 iterations
- `ReasoningNoRecentRequests`: Info alert when no requests for 15m
- `ReasoningRequestSpike`: Warning when rate is 3x 1h average

## Deployment

### Kubernetes Deployment

#### 1. Apply Prometheus Rules

```bash
# Apply reasoning framework alerting rules
kubectl apply -f monitoring/prometheus-reasoning-rules.yaml

# Apply DSPy optimization alerting rules (if using)
kubectl apply -f monitoring/prometheus-rules.yaml
```

#### 2. Import Grafana Dashboards

**Using Grafana UI:**
1. Navigate to Grafana → Dashboards → Import
2. Upload JSON file or paste contents
3. Select Prometheus data source
4. Click "Import"

**Using ConfigMap (recommended for GitOps):**

```bash
# Create ConfigMap for reasoning dashboard
kubectl create configmap reasoning-dashboard \
  --from-file=reasoning-dashboard.json=monitoring/grafana/reasoning-dashboard.json \
  -n agentcore

# Label for automatic provisioning (if using grafana-operator)
kubectl label configmap reasoning-dashboard \
  grafana_dashboard=1 \
  -n agentcore
```

#### 3. Configure Prometheus Scraping

Ensure your Prometheus configuration includes the AgentCore metrics endpoint:

```yaml
scrape_configs:
  - job_name: 'agentcore-reasoning'
    static_configs:
      - targets: ['agentcore-api:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Docker Compose Deployment

For local development with docker-compose:

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus-reasoning-rules.yaml:/etc/prometheus/rules/reasoning.yaml
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
```

## Usage

### Accessing Dashboards

**Local Development:**
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

**Kubernetes:**
```bash
# Port forward Grafana
kubectl port-forward -n agentcore svc/grafana 3000:3000

# Port forward Prometheus
kubectl port-forward -n agentcore svc/prometheus 9090:9090
```

### Querying Metrics

**Example PromQL queries:**

```promql
# Current success rate
rate(reasoning_bounded_context_requests_total{status="success"}[5m]) /
rate(reasoning_bounded_context_requests_total[5m]) * 100

# Average latency (p95)
histogram_quantile(0.95, rate(reasoning_bounded_context_duration_seconds_bucket[5m]))

# Total tokens per hour
increase(reasoning_bounded_context_tokens_total_sum[1h])

# Compute savings percentage
avg(reasoning_bounded_context_compute_savings_pct)

# Error rate by type
rate(reasoning_bounded_context_errors_total[5m])
```

### Interpreting Metrics

**Success Rate:**
- **>95%**: Healthy
- **90-95%**: Warning - investigate errors
- **<90%**: Critical - immediate attention needed

**Response Time (p95):**
- **<10s**: Excellent
- **10-30s**: Acceptable for complex reasoning
- **>30s**: Investigate bottlenecks

**Compute Savings:**
- **>50%**: Excellent efficiency
- **30-50%**: Good efficiency
- **<30%**: Consider tuning chunk size or strategy

**Token Usage:**
- Monitor trends to estimate costs
- Spikes may indicate large queries or inefficient prompts
- Use for capacity planning

## Alerting Integration

### Slack Integration

Configure Alertmanager to send alerts to Slack:

```yaml
# alertmanager.yml
receivers:
  - name: 'slack-reasoning'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#reasoning-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .Annotations.description }}'

route:
  receiver: 'slack-reasoning'
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  routes:
    - match:
        service: reasoning-framework
      receiver: 'slack-reasoning'
```

### PagerDuty Integration

For critical alerts:

```yaml
receivers:
  - name: 'pagerduty-reasoning'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .Annotations.summary }}'

route:
  routes:
    - match:
        severity: critical
        service: reasoning-framework
      receiver: 'pagerduty-reasoning'
```

## Troubleshooting

### No Metrics Appearing

1. **Check metrics endpoint:**
   ```bash
   curl http://localhost:8001/metrics | grep reasoning
   ```

2. **Verify Prometheus scraping:**
   - Navigate to Prometheus → Status → Targets
   - Check if `agentcore-reasoning` target is UP

3. **Check logs:**
   ```bash
   kubectl logs -n agentcore -l app=agentcore-api
   ```

### Dashboard Shows No Data

1. **Verify Prometheus data source in Grafana:**
   - Grafana → Configuration → Data Sources
   - Test connection

2. **Check time range:**
   - Ensure time range includes period with activity
   - Try "Last 1 hour" or "Last 24 hours"

3. **Verify metrics exist:**
   ```bash
   # Query Prometheus directly
   curl 'http://localhost:9090/api/v1/query?query=reasoning_bounded_context_requests_total'
   ```

### Alerts Not Firing

1. **Check PrometheusRule is loaded:**
   ```bash
   kubectl get prometheusrules -n agentcore
   ```

2. **Verify alert conditions:**
   - Navigate to Prometheus → Alerts
   - Check alert state (Inactive, Pending, Firing)

3. **Check Alertmanager:**
   ```bash
   kubectl port-forward -n agentcore svc/alertmanager 9093:9093
   # Visit http://localhost:9093
   ```

## Best Practices

1. **Set appropriate retention:**
   - Prometheus: 15-30 days for operational metrics
   - Long-term storage: Export to Thanos or Cortex

2. **Dashboard organization:**
   - Create folders for different services
   - Use consistent naming conventions
   - Add annotations for deployments

3. **Alert tuning:**
   - Start with higher thresholds
   - Reduce based on baseline behavior
   - Avoid alert fatigue

4. **Cost monitoring:**
   - Track token usage trends
   - Set budgets and alerts
   - Optimize prompts based on metrics

5. **Regular review:**
   - Weekly: Review error patterns
   - Monthly: Analyze trends and capacity
   - Quarterly: Update thresholds and add new metrics

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Alerting Best Practices](https://prometheus.io/docs/practices/alerting/)

## Support

For issues or questions:
- File issue: https://github.com/AetherForge/AgentCore/issues
- Documentation: https://docs.agentcore.dev/monitoring
