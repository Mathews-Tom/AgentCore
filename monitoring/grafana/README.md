# Grafana Dashboards for AgentCore

This directory contains Grafana dashboard configurations and provisioning setup for monitoring AgentCore's tool integration framework.

## Dashboards

### Tool Integration Metrics Dashboard

**File:** `tool-metrics-dashboard.json`

Comprehensive dashboard for monitoring tool execution performance, success rates, and error patterns.

#### Panels Overview

**KPI Stat Panels (Top Row):**
- **Total Tool Executions (24h)**: Aggregate execution count with color thresholds (green <100, yellow 100-1000, red >1000)
- **Success Rate (24h)**: Percentage of successful executions (green >95%, yellow 80-95%, red <80%)
- **Active Tools**: Count of unique tools with recorded executions
- **Error Rate (24h)**: Total error count with thresholds (green <10, yellow 10-50, red >50)
- **Avg Latency (p50)**: Median execution time in milliseconds (green <100ms, yellow 100-500ms, red >500ms)
- **p95 Latency**: 95th percentile latency (green <200ms, yellow 200-1000ms, red >1000ms)

**Time Series Graphs:**
- **Tool Execution Rate (by status)**: Executions per second broken down by success/failed/timeout status
- **Tool Execution Latency (Percentiles)**: p50/p95/p99 latency trends over time
- **Top 10 Tools by Execution Count**: Most frequently used tools ranked by execution rate
- **Tool Success vs Failure Rate**: Comparison of success, failure, and timeout rates
- **Error Distribution by Type**: Breakdown of errors by category (validation, timeout, rate_limit, etc.)
- **Tool Execution Duration by Tool**: Heatmap showing execution time distribution per tool

**Tables:**
- **Tool Executions by Tool ID (24h)**: Detailed breakdown of executions per tool and status
- **Tool Error Details (24h)**: Error counts grouped by tool and error type

#### Accessing the Dashboard

**Development Environment:**
1. Start the monitoring stack:
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```

2. Access Grafana:
   - URL: http://localhost:3000
   - Default credentials: admin/admin (change on first login)

3. Navigate to:
   - Dashboards → AgentCore → Tool Integration Metrics

**Production Environment:**
- Dashboard auto-loads via provisioning (see `dashboards.yml`)
- Access via Grafana instance at your configured domain
- Look for "AgentCore" folder in dashboard list

#### Interpreting Metrics

**Success Rate Analysis:**
- **>95% (Green)**: Healthy operation, tools executing reliably
- **80-95% (Yellow)**: Investigate recent failures, check error distribution panel
- **<80% (Red)**: Critical issue, immediate investigation required

**Latency Analysis:**
- **p50 <100ms (Green)**: Good performance for most requests
- **p95 >1000ms (Red)**: Performance degradation, check slow tools in heatmap
- Compare p50/p95/p99 to identify outliers

**Error Pattern Analysis:**
- Check "Error Distribution by Type" to identify root causes
- Common patterns:
  - High `validation_error`: Review tool parameter schemas
  - High `timeout_error`: Increase timeout or optimize tool implementation
  - High `rate_limit_error`: Adjust rate limits or increase capacity
  - High `quota_exceeded_error`: Review daily/monthly quota settings

**Tool Usage Analysis:**
- "Top 10 Tools" shows most critical tools for capacity planning
- Compare execution counts to success rates to identify problematic tools
- Use "Tool Execution Duration" heatmap to identify slow tools

#### Prometheus Metrics Reference

The dashboard queries these Prometheus metrics:

**Counters:**
- `agentcore_tool_executions_total{tool_id, status}`: Total executions by tool and status
- `agentcore_tool_errors_total{tool_id, error_type}`: Total errors by tool and error category

**Histograms:**
- `agentcore_tool_execution_seconds_bucket{tool_id, le}`: Execution duration histogram
- `agentcore_tool_execution_seconds_count{tool_id}`: Total execution count
- `agentcore_tool_execution_seconds_sum{tool_id}`: Total execution time

#### Dashboard Configuration

**Provisioning:**
Dashboards are automatically loaded via `dashboards.yml`:
- Update interval: 30 seconds
- UI updates: Enabled
- Deletion: Enabled (can delete from UI)
- Organization: AgentCore folder

**Customization:**
1. Edit dashboard in Grafana UI
2. Save changes
3. Export JSON model
4. Update `tool-metrics-dashboard.json`
5. Restart Grafana or wait for auto-reload (30s)

**Time Range:**
- Default: Last 6 hours
- Refresh: 30 seconds
- Adjust via time picker in dashboard header

## Adding New Dashboards

1. Create dashboard JSON file in this directory
2. Add provider entry to `dashboards.yml`:
   ```yaml
   - name: 'Dashboard Name'
     orgId: 1
     folder: 'AgentCore'
     type: file
     disableDeletion: false
     updateIntervalSeconds: 30
     allowUiUpdates: true
     options:
       path: /etc/grafana/provisioning/dashboards/your-dashboard.json
   ```

3. Restart Grafana or wait for auto-reload

## Troubleshooting

**Dashboard not loading:**
- Check Grafana logs: `docker logs agentcore-grafana`
- Verify JSON syntax: `jq . tool-metrics-dashboard.json`
- Ensure provisioning path is correct in `dashboards.yml`

**No data displayed:**
- Verify Prometheus data source is configured
- Check Prometheus is scraping metrics: http://localhost:9090/targets
- Verify tool executor has `metrics_collector` configured
- Execute some tools to generate metrics

**Metrics not updating:**
- Check dashboard refresh rate (top-right corner)
- Verify Prometheus scrape interval (default: 15s)
- Check time range selection (ensure it covers recent data)

## Related Documentation

- Prometheus metrics implementation: `src/agentcore/agent_runtime/services/metrics_collector.py:1`
- Executor metrics integration: `src/agentcore/agent_runtime/tools/executor.py:159`
- Metrics tests: `tests/agent_runtime/tools/test_executor_metrics.py:1`
- Tool Integration Framework: `.docs/PR_DESCRIPTION.md:1`
