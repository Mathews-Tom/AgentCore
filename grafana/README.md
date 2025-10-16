# Grafana Dashboards for AgentCore

This directory contains Grafana dashboard definitions for monitoring AgentCore services.

## Available Dashboards

### Bounded Context Reasoning Dashboard

**File:** `dashboards/reasoning-dashboard.json`

Comprehensive monitoring dashboard for the bounded context reasoning feature, tracking:

#### Reasoning Overview
- **Request Rate**: Successful vs error requests per second with 5-minute rate
- **Request Latency**: p50, p95, p99 percentiles for response times
- **Error Rate**: Percentage of failed requests (gauge)
- **Avg Compute Savings**: Average percentage of compute saved vs traditional reasoning
- **Requests (Last Hour)**: Total request count in the past hour
- **Avg Iterations per Request**: Mean number of reasoning iterations

#### Token Usage & Compute Savings
- **Token Usage Rate**: Tokens consumed per second
- **Compute Savings Distribution**: Percentile distribution of savings across requests

#### Iteration Analysis
- **Iterations per Request Distribution**: p50, p95, p99 for iteration counts
- **Iteration Count Distribution**: Bucketed histogram showing iteration frequency

#### Errors & LLM Health
- **Errors by Type**: Breakdown of validation, LLM, timeout, internal, authentication, and authorization errors
- **LLM Service Failures**: Rate of LLM API failures

## Prerequisites

1. **Prometheus** configured to scrape AgentCore metrics endpoint (`/metrics`)
2. **Grafana** instance (v9.0+)
3. **Prometheus datasource** configured in Grafana

## Import Instructions

### Method 1: Via Grafana UI

1. Log in to your Grafana instance
2. Navigate to **Dashboards** → **Import**
3. Click **Upload JSON file**
4. Select the dashboard JSON file from this directory
5. Configure the Prometheus datasource (select from dropdown or use default)
6. Click **Import**

### Method 2: Via API

```bash
# Set your Grafana URL and API key
GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY="your-api-key"
DASHBOARD_FILE="dashboards/reasoning-dashboard.json"

# Import the dashboard
curl -X POST "${GRAFANA_URL}/api/dashboards/db" \
  -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @"${DASHBOARD_FILE}"
```

### Method 3: Provisioning (Recommended for Production)

1. Copy dashboard JSON files to Grafana provisioning directory:
   ```bash
   cp dashboards/*.json /etc/grafana/provisioning/dashboards/
   ```

2. Create a provisioning config file `/etc/grafana/provisioning/dashboards/agentcore.yaml`:
   ```yaml
   apiVersion: 1

   providers:
     - name: 'AgentCore'
       orgId: 1
       folder: 'AgentCore'
       type: file
       disableDeletion: false
       updateIntervalSeconds: 10
       allowUiUpdates: true
       options:
         path: /etc/grafana/provisioning/dashboards
         foldersFromFilesStructure: true
   ```

3. Restart Grafana:
   ```bash
   systemctl restart grafana-server
   ```

## Dashboard Configuration

### Datasource Variable

The dashboards use a templated Prometheus datasource variable `${DS_PROMETHEUS}`. When importing:

- If you have a single Prometheus datasource, it will be auto-selected
- If you have multiple datasources, select the correct one during import
- The datasource selection is saved with the dashboard

### Time Range

Default time range: **Last 3 hours**

Recommended ranges for different use cases:
- Development/Testing: Last 1 hour
- Production Monitoring: Last 6 hours
- Performance Analysis: Last 24 hours
- Incident Investigation: Custom range

### Refresh Rate

Default refresh: **30 seconds**

Adjust based on your needs:
- High-traffic production: 10-30 seconds
- Development: 1-5 minutes
- Historical analysis: Disable auto-refresh

## Customization

### Adding Panels

To add custom panels:

1. Edit the dashboard in Grafana UI
2. Add new panel with desired metric
3. Export the updated dashboard JSON
4. Replace the file in this directory

### Metric Examples

Query Prometheus metrics directly:

```promql
# Request rate by status
rate(reasoning_bounded_context_requests_total{status="success"}[5m])

# P95 latency
histogram_quantile(0.95, sum(rate(reasoning_bounded_context_duration_seconds_bucket[5m])) by (le))

# Average compute savings
rate(reasoning_bounded_context_compute_savings_pct_sum[5m]) / rate(reasoning_bounded_context_compute_savings_pct_count[5m])

# Error rate percentage
rate(reasoning_bounded_context_requests_total{status="error"}[5m]) / rate(reasoning_bounded_context_requests_total[5m])

# LLM failure rate
rate(reasoning_bounded_context_llm_failures_total[5m])
```

## Alerting Integration

The dashboard metrics can be used for alerting. See `prometheus/alerts/reasoning-alerts.yml` for pre-configured alert rules.

## Troubleshooting

### Dashboard Shows "No Data"

1. Verify Prometheus is scraping the `/metrics` endpoint:
   ```bash
   curl http://localhost:8001/metrics | grep reasoning_bounded_context
   ```

2. Check Prometheus targets are healthy:
   - Navigate to Prometheus UI → Status → Targets
   - Verify AgentCore target is `UP`

3. Verify datasource configuration in Grafana:
   - Configuration → Data Sources → Prometheus
   - Test connection and ensure it succeeds

### Metrics Missing or Incorrect

1. Ensure `ENABLE_METRICS=true` in AgentCore configuration
2. Verify reasoning endpoint has been called (metrics only appear after first request)
3. Check Prometheus scrape interval matches dashboard refresh rate
4. Review Prometheus logs for scrape errors

### Permission Issues

If dashboard is read-only:
- Check Grafana user permissions (need Editor or Admin role)
- Verify dashboard is not provisioned with `disableDeletion: true`

## Support

For issues or questions:
- Check AgentCore documentation: `docs/`
- Review Prometheus metrics: `src/agentcore/reasoning/services/metrics.py`
- Open an issue in the AgentCore repository
