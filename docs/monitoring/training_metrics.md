# Training Metrics Documentation

Prometheus metrics for GRPO training system monitoring and observability.

## Overview

The training system exports metrics via Prometheus for monitoring training jobs, performance, and budget usage. Metrics are available at the `/metrics` endpoint and can be scraped by Prometheus for visualization in Grafana.

## Metrics Categories

### 1. Training Job Lifecycle Metrics

**Counter Metrics:**

- `training_jobs_created_total{agent_id}` - Total training jobs created
- `training_jobs_completed_total{agent_id}` - Total jobs completed successfully
- `training_jobs_failed_total{agent_id}` - Total jobs failed
- `training_jobs_cancelled_total{agent_id}` - Total jobs cancelled

**Gauge Metrics:**

- `training_jobs_active` - Number of currently active training jobs

**Usage:**
```python
from agentcore.training.metrics import TrainingMetrics

# Record job lifecycle
TrainingMetrics.job_created(agent_id)
TrainingMetrics.job_completed(agent_id)
TrainingMetrics.job_failed(agent_id)
TrainingMetrics.job_cancelled(agent_id)
```

### 2. Performance Metrics

**Histogram Metrics (duration in seconds):**

- `training_trajectory_generation_duration_seconds{agent_id}` - Time to generate trajectory batch
  - Buckets: [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]

- `training_policy_update_duration_seconds{agent_id}` - Time to update policy
  - Buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

- `training_iteration_duration_seconds{agent_id}` - Time for complete training iteration
  - Buckets: [5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]

- `training_checkpoint_save_duration_seconds{agent_id}` - Time to save checkpoint
  - Buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

**Usage:**
```python
from agentcore.training.metrics import TrainingMetrics

# Measure operations
with TrainingMetrics.measure_trajectory_generation(agent_id):
    trajectories = generate_trajectories(...)

with TrainingMetrics.measure_policy_update(agent_id):
    update_policy(...)

with TrainingMetrics.measure_training_iteration(agent_id):
    run_iteration(...)

with TrainingMetrics.measure_checkpoint_save(agent_id):
    save_checkpoint(...)
```

### 3. Budget Metrics

**Gauge Metrics:**

- `training_budget_usage_usd{agent_id, job_id}` - Current budget usage in USD
- `training_budget_limit_usd{agent_id, job_id}` - Budget limit in USD
- `training_budget_utilization_percent{agent_id, job_id}` - Budget utilization percentage (0-100)

**Usage:**
```python
from agentcore.training.metrics import TrainingMetrics

# Update budget metrics
TrainingMetrics.update_budget(
    agent_id="agent-123",
    job_id="job-456",
    usage_usd=75.50,
    limit_usd=100.00,
)
```

### 4. Training Progress Metrics

**Gauge Metrics:**

- `training_iteration_current{agent_id, job_id}` - Current training iteration
- `training_iteration_total{agent_id, job_id}` - Total iterations planned
- `training_loss{agent_id, job_id}` - Current training loss
- `training_reward_mean{agent_id, job_id}` - Mean reward across trajectories

**Usage:**
```python
from agentcore.training.metrics import TrainingMetrics

# Update progress
TrainingMetrics.update_progress(
    agent_id="agent-123",
    job_id="job-456",
    current_iteration=50,
    total_iterations=100,
)

# Update training metrics
TrainingMetrics.update_training_metrics(
    agent_id="agent-123",
    job_id="job-456",
    loss=0.25,
    mean_reward=0.75,
)
```

## Grafana Dashboard Configuration

### Recommended Dashboards

#### 1. Training Overview Dashboard

**Panels:**

1. **Active Jobs** (Gauge)
   - Query: `training_jobs_active`

2. **Job Success Rate** (Graph)
   - Query: `rate(training_jobs_completed_total[5m]) / (rate(training_jobs_completed_total[5m]) + rate(training_jobs_failed_total[5m]))`

3. **Jobs Created/Completed/Failed** (Graph)
   - Queries:
     - Created: `rate(training_jobs_created_total[5m])`
     - Completed: `rate(training_jobs_completed_total[5m])`
     - Failed: `rate(training_jobs_failed_total[5m])`

#### 2. Performance Dashboard

**Panels:**

1. **Trajectory Generation Time (p95)** (Graph)
   - Query: `histogram_quantile(0.95, rate(training_trajectory_generation_duration_seconds_bucket[5m]))`

2. **Policy Update Time (p95)** (Graph)
   - Query: `histogram_quantile(0.95, rate(training_policy_update_duration_seconds_bucket[5m]))`

3. **Training Iteration Time (p95)** (Graph)
   - Query: `histogram_quantile(0.95, rate(training_iteration_duration_seconds_bucket[5m]))`

4. **Checkpoint Save Time (p95)** (Graph)
   - Query: `histogram_quantile(0.95, rate(training_checkpoint_save_duration_seconds_bucket[5m]))`

#### 3. Budget Dashboard

**Panels:**

1. **Budget Utilization by Job** (Graph)
   - Query: `training_budget_utilization_percent`

2. **Total Budget Usage** (Gauge)
   - Query: `sum(training_budget_usage_usd)`

3. **Budget Usage vs Limit** (Graph)
   - Queries:
     - Usage: `training_budget_usage_usd`
     - Limit: `training_budget_limit_usd`

4. **Jobs Near Budget Limit** (Table)
   - Query: `training_budget_utilization_percent > 90`

#### 4. Training Progress Dashboard

**Panels:**

1. **Training Progress** (Gauge)
   - Query: `training_iteration_current / training_iteration_total * 100`

2. **Training Loss** (Graph)
   - Query: `training_loss`

3. **Mean Reward** (Graph)
   - Query: `training_reward_mean`

### Alert Rules

#### Critical Alerts

**High Failure Rate:**
```yaml
- alert: HighTrainingFailureRate
  expr: rate(training_jobs_failed_total[5m]) / rate(training_jobs_created_total[5m]) > 0.2
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "High training job failure rate"
    description: "More than 20% of training jobs are failing"
```

**Budget Exceeded:**
```yaml
- alert: TrainingBudgetExceeded
  expr: training_budget_utilization_percent > 100
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Training budget exceeded"
    description: "Job {{ $labels.job_id }} has exceeded budget limit"
```

**Slow Trajectory Generation:**
```yaml
- alert: SlowTrajectoryGeneration
  expr: histogram_quantile(0.95, rate(training_trajectory_generation_duration_seconds_bucket[5m])) > 60
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Slow trajectory generation"
    description: "p95 trajectory generation time exceeds 60 seconds"
```

#### Warning Alerts

**Budget Warning (90%):**
```yaml
- alert: TrainingBudgetWarning
  expr: training_budget_utilization_percent > 90 and training_budget_utilization_percent <= 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Training budget warning"
    description: "Job {{ $labels.job_id }} has used 90% of budget"
```

**Long Running Job:**
```yaml
- alert: LongRunningTrainingJob
  expr: training_iteration_current > 0 and (time() - training_jobs_created_total) > 3600
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "Long running training job"
    description: "Job {{ $labels.job_id }} has been running for over 1 hour"
```

## Prometheus Configuration

### Scrape Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'agentcore-training'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8001']  # Adjust port as needed
    metrics_path: '/metrics'
```

### Recording Rules

Create recording rules for common queries:

```yaml
groups:
  - name: training_metrics
    interval: 30s
    rules:
      - record: training:job_success_rate:5m
        expr: rate(training_jobs_completed_total[5m]) / (rate(training_jobs_completed_total[5m]) + rate(training_jobs_failed_total[5m]))

      - record: training:trajectory_generation_p95:5m
        expr: histogram_quantile(0.95, rate(training_trajectory_generation_duration_seconds_bucket[5m]))

      - record: training:policy_update_p95:5m
        expr: histogram_quantile(0.95, rate(training_policy_update_duration_seconds_bucket[5m]))
```

## Best Practices

1. **Label Cardinality**: Be mindful of label cardinality. Use `agent_id` and `job_id` sparingly.

2. **Histogram Buckets**: Adjust histogram buckets based on your actual latency distribution.

3. **Retention**: Configure Prometheus retention based on your monitoring needs (default: 15 days).

4. **Aggregation**: Use recording rules for frequently-queried aggregations to improve dashboard performance.

5. **Alerting**: Start with conservative alert thresholds and tune based on observed behavior.

## Troubleshooting

### Missing Metrics

If metrics are not appearing:

1. Check Prometheus scrape status: `http://localhost:9090/targets`
2. Verify `/metrics` endpoint is accessible
3. Check application logs for errors

### High Cardinality

If metrics have too many label combinations:

1. Review label usage (avoid high-cardinality labels like timestamps)
2. Use label relabeling in Prometheus to drop unnecessary labels
3. Consider aggregating at the application level

### Slow Queries

If Grafana dashboards are slow:

1. Use recording rules for complex queries
2. Reduce query time ranges
3. Use rate() and increase() functions appropriately

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
