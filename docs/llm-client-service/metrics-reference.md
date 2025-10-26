# Metrics Reference

Complete reference for all Prometheus metrics exposed by the LLM Client Service.

## Table of Contents

- [Overview](#overview)
- [Metrics Summary](#metrics-summary)
- [Request Metrics](#request-metrics)
- [Duration Metrics](#duration-metrics)
- [Token Metrics](#token-metrics)
- [Error Metrics](#error-metrics)
- [Active Requests](#active-requests)
- [Governance Metrics](#governance-metrics)
- [Query Examples](#query-examples)
- [Grafana Dashboards](#grafana-dashboards)
- [Alerting Rules](#alerting-rules)

---

## Overview

The LLM Client Service exposes Prometheus metrics for:
- Request counts and success rates
- Request latency percentiles (P50, P95, P99)
- Token usage tracking (cost estimation)
- Error tracking by type
- Active request monitoring
- Model governance violations

**Metrics Endpoint:** `http://localhost:9090/metrics`

---

## Metrics Summary

| Metric | Type | Description |
|--------|------|-------------|
| `llm_requests_total` | Counter | Total number of LLM requests |
| `llm_requests_duration_seconds` | Histogram | Request duration in seconds |
| `llm_tokens_total` | Counter | Total tokens used |
| `llm_errors_total` | Counter | Total errors by type |
| `llm_active_requests` | Gauge | Currently active requests |
| `llm_governance_violations_total` | Counter | Model governance violations |

---

## Request Metrics

### llm_requests_total

**Type:** Counter

**Description:** Total number of LLM requests by provider, model, and status.

**Labels:**
- `provider` - LLM provider (`openai`, `anthropic`, `gemini`)
- `model` - Model identifier (e.g., `gpt-4.1-mini`)
- `status` - Request status (`success`, `error`)

**Example Values:**
```promql
llm_requests_total{provider="openai",model="gpt-4.1-mini",status="success"} 1234
llm_requests_total{provider="anthropic",model="claude-3-5-haiku-20241022",status="success"} 567
llm_requests_total{provider="openai",model="gpt-4.1-mini",status="error"} 12
```

**Usage:**

```promql
# Total successful requests
sum(llm_requests_total{status="success"})

# Request rate by provider
rate(llm_requests_total[5m])

# Success rate
sum(rate(llm_requests_total{status="success"}[5m]))
  /
sum(rate(llm_requests_total[5m]))

# Requests by model
sum by (model) (llm_requests_total{status="success"})
```

---

## Duration Metrics

### llm_requests_duration_seconds

**Type:** Histogram

**Description:** Request duration in seconds with percentile buckets.

**Labels:**
- `provider` - LLM provider (`openai`, `anthropic`, `gemini`)
- `model` - Model identifier (e.g., `gpt-4.1-mini`)

**Buckets:** `[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]`

**Bucket Meanings:**
- `0.1s` - Very fast cached/simple responses
- `0.5s` - Fast responses
- `1.0s` - Normal responses
- `2.5s` - Longer responses
- `5.0s` - Complex responses
- `10.0s` - Very complex responses
- `30.0s` - Extended reasoning
- `60.0s` - Max typical timeout
- `120.0s` - Extended timeout
- `300.0s` - Maximum timeout (5 minutes)

**Example Values:**
```promql
llm_requests_duration_seconds_bucket{provider="openai",model="gpt-4.1-mini",le="0.5"} 100
llm_requests_duration_seconds_bucket{provider="openai",model="gpt-4.1-mini",le="1.0"} 250
llm_requests_duration_seconds_bucket{provider="openai",model="gpt-4.1-mini",le="+Inf"} 500
llm_requests_duration_seconds_sum{provider="openai",model="gpt-4.1-mini"} 875.3
llm_requests_duration_seconds_count{provider="openai",model="gpt-4.1-mini"} 500
```

**Usage:**

```promql
# Average latency
rate(llm_requests_duration_seconds_sum[5m])
  /
rate(llm_requests_duration_seconds_count[5m])

# P50 latency (median)
histogram_quantile(0.5, rate(llm_requests_duration_seconds_bucket[5m]))

# P95 latency
histogram_quantile(0.95, rate(llm_requests_duration_seconds_bucket[5m]))

# P99 latency
histogram_quantile(0.99, rate(llm_requests_duration_seconds_bucket[5m]))

# Latency by provider
histogram_quantile(0.95,
  sum by (provider, le) (rate(llm_requests_duration_seconds_bucket[5m]))
)

# Slow requests (>5s)
sum(rate(llm_requests_duration_seconds_bucket{le="5.0"}[5m]))
  -
sum(rate(llm_requests_duration_seconds_bucket{le="2.5"}[5m]))
```

---

## Token Metrics

### llm_tokens_total

**Type:** Counter

**Description:** Total number of tokens used by provider, model, and type.

**Labels:**
- `provider` - LLM provider (`openai`, `anthropic`, `gemini`)
- `model` - Model identifier (e.g., `gpt-4.1-mini`)
- `token_type` - Token type (`prompt`, `completion`)

**Example Values:**
```promql
llm_tokens_total{provider="openai",model="gpt-4.1-mini",token_type="prompt"} 500000
llm_tokens_total{provider="openai",model="gpt-4.1-mini",token_type="completion"} 250000
llm_tokens_total{provider="anthropic",model="claude-3-5-haiku-20241022",token_type="prompt"} 100000
```

**Usage:**

```promql
# Total tokens used
sum(llm_tokens_total)

# Tokens by provider
sum by (provider) (llm_tokens_total)

# Tokens by model
sum by (model) (llm_tokens_total)

# Prompt vs completion tokens
sum by (token_type) (llm_tokens_total)

# Token usage rate (tokens/sec)
rate(llm_tokens_total[5m])

# Average tokens per request
sum(rate(llm_tokens_total[5m]))
  /
sum(rate(llm_requests_total{status="success"}[5m]))

# Cost estimation (OpenAI gpt-4.1-mini: $0.15/1M input, $0.60/1M output)
(
  sum(rate(llm_tokens_total{provider="openai",model="gpt-4.1-mini",token_type="prompt"}[5m])) * 0.15 / 1000000
)
+
(
  sum(rate(llm_tokens_total{provider="openai",model="gpt-4.1-mini",token_type="completion"}[5m])) * 0.60 / 1000000
)
```

---

## Error Metrics

### llm_errors_total

**Type:** Counter

**Description:** Total number of errors by provider, model, and error type.

**Labels:**
- `provider` - LLM provider (`openai`, `anthropic`, `gemini`)
- `model` - Model identifier (e.g., `gpt-4.1-mini`)
- `error_type` - Error type name (e.g., `ProviderError`, `ProviderTimeoutError`)

**Example Values:**
```promql
llm_errors_total{provider="openai",model="gpt-4.1-mini",error_type="ProviderError"} 15
llm_errors_total{provider="openai",model="gpt-4.1-mini",error_type="ProviderTimeoutError"} 3
llm_errors_total{provider="anthropic",model="claude-3-5-haiku-20241022",error_type="ProviderError"} 7
```

**Usage:**

```promql
# Total errors
sum(llm_errors_total)

# Error rate
rate(llm_errors_total[5m])

# Errors by type
sum by (error_type) (llm_errors_total)

# Errors by provider
sum by (provider) (llm_errors_total)

# Timeout errors specifically
sum(llm_errors_total{error_type="ProviderTimeoutError"})

# Error rate as percentage of total requests
sum(rate(llm_errors_total[5m]))
  /
sum(rate(llm_requests_total[5m])) * 100
```

---

## Active Requests

### llm_active_requests

**Type:** Gauge

**Description:** Number of currently active LLM requests by provider.

**Labels:**
- `provider` - LLM provider (`openai`, `anthropic`, `gemini`)

**Example Values:**
```promql
llm_active_requests{provider="openai"} 5
llm_active_requests{provider="anthropic"} 2
llm_active_requests{provider="gemini"} 1
```

**Usage:**

```promql
# Total active requests
sum(llm_active_requests)

# Active requests by provider
llm_active_requests

# Max active requests in last 5 minutes
max_over_time(llm_active_requests[5m])

# Average active requests
avg_over_time(llm_active_requests[5m])
```

---

## Governance Metrics

### llm_governance_violations_total

**Type:** Counter

**Description:** Total number of governance violations (disallowed model attempts).

**Labels:**
- `model` - Model that was attempted (not in ALLOWED_MODELS)
- `source_agent` - Agent that attempted the request (for accountability)

**Example Values:**
```promql
llm_governance_violations_total{model="gpt-3.5-turbo",source_agent="agent-001"} 12
llm_governance_violations_total{model="gpt-4",source_agent="agent-002"} 5
llm_governance_violations_total{model="text-davinci-003",source_agent="unknown"} 3
```

**Usage:**

```promql
# Total governance violations
sum(llm_governance_violations_total)

# Violations by model
sum by (model) (llm_governance_violations_total)

# Violations by agent
sum by (source_agent) (llm_governance_violations_total)

# Violation rate
rate(llm_governance_violations_total[5m])

# Top violating agents
topk(5, sum by (source_agent) (llm_governance_violations_total))

# Top attempted disallowed models
topk(5, sum by (model) (llm_governance_violations_total))
```

---

## Query Examples

### Performance Monitoring

```promql
# Average latency by provider
sum by (provider) (rate(llm_requests_duration_seconds_sum[5m]))
  /
sum by (provider) (rate(llm_requests_duration_seconds_count[5m]))

# Request throughput (requests/sec)
sum(rate(llm_requests_total[5m]))

# Success rate percentage
sum(rate(llm_requests_total{status="success"}[5m]))
  /
sum(rate(llm_requests_total[5m])) * 100
```

### Cost Analysis

```promql
# Total tokens per hour
sum(increase(llm_tokens_total[1h]))

# Cost per hour (OpenAI gpt-4.1-mini)
(
  sum(increase(llm_tokens_total{provider="openai",model="gpt-4.1-mini",token_type="prompt"}[1h])) * 0.15 / 1000000
) + (
  sum(increase(llm_tokens_total{provider="openai",model="gpt-4.1-mini",token_type="completion"}[1h])) * 0.60 / 1000000
)

# Daily cost projection
sum(increase(llm_tokens_total[1h])) * 24 * 0.001  # Simplified example
```

### Error Analysis

```promql
# Error rate by provider
sum by (provider) (rate(llm_errors_total[5m]))

# Most common error types
topk(5, sum by (error_type) (llm_errors_total))

# Timeout error percentage
sum(llm_errors_total{error_type="ProviderTimeoutError"})
  /
sum(llm_errors_total) * 100
```

---

## Grafana Dashboards

### Sample Dashboard Panels

#### 1. Request Rate Panel

```json
{
  "title": "LLM Request Rate",
  "targets": [
    {
      "expr": "sum(rate(llm_requests_total[5m])) by (provider)",
      "legendFormat": "{{provider}}"
    }
  ],
  "type": "graph"
}
```

#### 2. Latency Percentiles Panel

```json
{
  "title": "Request Latency (P50, P95, P99)",
  "targets": [
    {
      "expr": "histogram_quantile(0.5, rate(llm_requests_duration_seconds_bucket[5m]))",
      "legendFormat": "P50"
    },
    {
      "expr": "histogram_quantile(0.95, rate(llm_requests_duration_seconds_bucket[5m]))",
      "legendFormat": "P95"
    },
    {
      "expr": "histogram_quantile(0.99, rate(llm_requests_duration_seconds_bucket[5m]))",
      "legendFormat": "P99"
    }
  ],
  "type": "graph"
}
```

#### 3. Token Usage Panel

```json
{
  "title": "Token Usage by Type",
  "targets": [
    {
      "expr": "sum(rate(llm_tokens_total[5m])) by (token_type)",
      "legendFormat": "{{token_type}}"
    }
  ],
  "type": "graph"
}
```

#### 4. Error Rate Panel

```json
{
  "title": "Error Rate",
  "targets": [
    {
      "expr": "sum(rate(llm_errors_total[5m])) by (error_type)",
      "legendFormat": "{{error_type}}"
    }
  ],
  "type": "graph"
}
```

#### 5. Active Requests Gauge

```json
{
  "title": "Active Requests",
  "targets": [
    {
      "expr": "sum(llm_active_requests)",
      "legendFormat": "Active"
    }
  ],
  "type": "gauge"
}
```

---

## Alerting Rules

### Prometheus Alert Rules

```yaml
groups:
  - name: llm_client_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighLLMErrorRate
        expr: |
          sum(rate(llm_errors_total[5m]))
            /
          sum(rate(llm_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM error rate (>5%)"
          description: "LLM error rate is {{ $value | humanizePercentage }}"

      # High latency
      - alert: HighLLMLatency
        expr: |
          histogram_quantile(0.95,
            rate(llm_requests_duration_seconds_bucket[5m])
          ) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM P95 latency (>10s)"
          description: "P95 latency is {{ $value }}s"

      # Timeout errors
      - alert: LLMTimeoutErrors
        expr: |
          sum(rate(llm_errors_total{error_type="ProviderTimeoutError"}[5m])) > 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM timeout errors detected"
          description: "{{ $value }} timeout errors per second"

      # Governance violations
      - alert: HighGovernanceViolations
        expr: |
          rate(llm_governance_violations_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of model governance violations"
          description: "{{ $value }} violations per second"

      # High token usage
      - alert: HighTokenUsage
        expr: |
          sum(rate(llm_tokens_total[1h])) > 1000000
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "High token usage (>1M/hour)"
          description: "Token usage is {{ $value }}/hour"

      # Provider down
      - alert: LLMProviderDown
        expr: |
          sum(rate(llm_requests_total{status="success"}[5m])) by (provider) == 0
          and
          sum(rate(llm_requests_total[5m])) by (provider) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM provider {{ $labels.provider }} appears down"
          description: "No successful requests in 5 minutes"
```

---

## Metric Collection

### Enabling Metrics

Metrics are automatically collected when using `llm_service`. No additional configuration required.

### Metrics Endpoint

```bash
# Default Prometheus metrics endpoint
curl http://localhost:9090/metrics | grep llm_
```

### Scraping Configuration

**prometheus.yml:**
```yaml
scrape_configs:
  - job_name: 'agentcore'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
```

---

## Best Practices

1. **Monitor P95 and P99 latency** - Better indicators than average
2. **Track token usage for cost control** - Set budgets and alerts
3. **Alert on governance violations** - Enforce model policies
4. **Monitor error rates by provider** - Identify provider issues
5. **Track active requests** - Prevent resource exhaustion
6. **Use rate() for counters** - Get per-second rates
7. **Use histogram_quantile() for latency** - Calculate percentiles
8. **Set appropriate alert thresholds** - Based on SLAs and usage patterns
9. **Create dashboards by provider and model** - Granular visibility
10. **Export metrics to long-term storage** - Historical analysis

---

For pre-built Grafana dashboards, see `docs/grafana/llm-client-dashboard.json`.
