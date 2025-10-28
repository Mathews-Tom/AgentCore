# MLflow Experiment Tracking Setup

This document describes how to set up and use MLflow experiment tracking for DSPy optimization.

## Overview

MLflow provides experiment tracking, model versioning, and artifact management for the DSPy optimization engine. It enables:

- **Experiment Logging**: Track optimization runs with parameters, metrics, and results
- **Model Versioning**: Save and version optimized models with metadata
- **Artifact Management**: Store training data, optimization artifacts, and model files
- **Performance Analytics**: Compare optimization runs and analyze improvements

## Quick Start

### 1. Start MLflow Server

Using Docker Compose (recommended):

```bash
# Start MLflow with PostgreSQL backend
docker compose -f docker-compose.mlflow.yml up -d

# Check MLflow is running
curl http://localhost:5000/health
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.mlflow.example .env.mlflow
```

Edit `.env.mlflow` with your settings:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=dspy-optimization
```

### 3. Use in Code

```python
from agentcore.dspy_optimization.pipeline import OptimizationPipeline
from agentcore.dspy_optimization.tracking import MLflowConfig

# Configure MLflow
mlflow_config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment_name="dspy-optimization",
)

# Create pipeline with tracking enabled
pipeline = OptimizationPipeline(
    mlflow_config=mlflow_config,
    enable_tracking=True,
)

# Run optimization (automatically tracked)
result = await pipeline.run_optimization(
    request=optimization_request,
    baseline_metrics=baseline_metrics,
    training_data=training_data,
)
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│                  MLflow Tracking                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  Experiment  │  │   Model      │  │ Artifact │ │
│  │   Logging    │  │  Versioning  │  │ Storage  │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         PostgreSQL Backend Storage           │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **Optimization Start**: Create MLflow run with request parameters
2. **Baseline Logging**: Log baseline performance metrics
3. **Training Data**: Save training data as artifact
4. **Algorithm Execution**: Track optimization iterations
5. **Result Logging**: Log optimized metrics and improvements
6. **Model Storage**: Save optimized model artifacts
7. **Run Completion**: Close run with status

## Configuration

### MLflowConfig Options

```python
class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "dspy-optimization"
    artifact_location: str | None = None  # S3, local path, etc.
    registry_uri: str | None = None  # Model registry URI
```

### Environment Variables

```bash
# Required
MLFLOW_TRACKING_URI=http://localhost:5000

# Optional
MLFLOW_EXPERIMENT_NAME=dspy-optimization
MLFLOW_ARTIFACT_LOCATION=s3://my-bucket/mlflow
MLFLOW_REGISTRY_URI=http://localhost:5000
```

## Tracked Metrics

### Baseline Metrics

- `baseline_success_rate`: Task success rate before optimization
- `baseline_avg_cost`: Average cost per task
- `baseline_avg_latency_ms`: Average latency in milliseconds
- `baseline_quality_score`: Quality score

### Optimized Metrics

- `optimized_success_rate`: Task success rate after optimization
- `optimized_avg_cost`: Average cost per task
- `optimized_avg_latency_ms`: Average latency in milliseconds
- `optimized_quality_score`: Quality score

### Improvement Metrics

- `success_rate_improvement`: Absolute improvement in success rate
- `cost_reduction`: Cost savings per task
- `latency_reduction_ms`: Latency reduction in milliseconds
- `quality_improvement`: Quality score improvement
- `improvement_percentage`: Overall improvement percentage
- `statistical_significance`: P-value for statistical significance

### Algorithm Metrics

- `iterations`: Number of optimization iterations
- `duration_seconds`: Total optimization duration

## Artifacts

### Logged Artifacts

1. **Training Data**: `training_data.json`
2. **Key Improvements**: `key_improvements.txt`
3. **Algorithm Parameters**: `algorithm_parameters.json`
4. **Optimized Model**: `models/optimized_model.pkl`
5. **Model Metadata**: `models/optimized_model_metadata.json`

### Loading Artifacts

```python
from agentcore.dspy_optimization.tracking import MLflowTracker

tracker = MLflowTracker()

# Load optimized model
model = await tracker.load_model_artifact(
    run_id="abc123",
    artifact_name="optimized_model",
)
```

## Searching Runs

### Find Best Run

```python
# Get best run by improvement percentage
best_run = await tracker.get_best_run(
    metric="improvement_percentage",
    order_by="DESC",
)

print(f"Best run: {best_run.info.run_id}")
print(f"Improvement: {best_run.data.metrics['improvement_percentage']:.2%}")
```

### Search with Filters

```python
# Find runs with >20% improvement
runs = await tracker.search_runs(
    filter_string="metrics.improvement_percentage > 0.2",
    max_results=10,
)

for run in runs:
    print(f"Run {run.info.run_id}: {run.data.metrics['improvement_percentage']:.2%}")
```

## Production Deployment

### S3 Artifact Storage

```bash
# Configure S3 backend
MLFLOW_ARTIFACT_LOCATION=s3://my-bucket/mlflow-artifacts

# Set AWS credentials
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

### Remote Tracking Server

```bash
# Point to remote MLflow server
MLFLOW_TRACKING_URI=https://mlflow.example.com

# Use authentication if required
MLFLOW_TRACKING_USERNAME=user
MLFLOW_TRACKING_PASSWORD=pass
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.19.0
        args:
        - mlflow
        - server
        - --backend-store-uri
        - postgresql://user:pass@postgres:5432/mlflow
        - --default-artifact-root
        - s3://my-bucket/mlflow
        - --host
        - 0.0.0.0
        ports:
        - containerPort: 5000
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-access-key
```

## Web UI

Access the MLflow web UI at: `http://localhost:5000`

Features:
- View all experiments and runs
- Compare metrics across runs
- Visualize performance trends
- Download artifacts
- Search and filter runs

## Best Practices

### 1. Naming Conventions

```python
# Use descriptive run names
run_name = f"opt_{target_type}_{target_id}_{timestamp}"

# Tag runs consistently
tags = {
    "target_type": "agent",
    "environment": "production",
    "version": "v1.0",
}
```

### 2. Parameter Tracking

```python
# Log all important parameters
mlflow.log_param("algorithm", "miprov2")
mlflow.log_param("target_id", agent_id)
mlflow.log_param("optimization_time", max_time)
```

### 3. Artifact Management

```python
# Save models with metadata
await tracker.log_model_artifact(
    model=optimized_model,
    artifact_name="optimized_model",
    metadata={
        "algorithm": "miprov2",
        "improvement": 0.25,
        "created_at": datetime.utcnow().isoformat(),
    },
)
```

### 4. Cleanup

```python
# Clean up old experiments (careful!)
# tracker.cleanup_experiment()  # Deletes all runs!

# Better: Archive old runs via UI or API
```

## Troubleshooting

### MLflow Server Not Starting

```bash
# Check PostgreSQL connection
docker compose -f docker-compose.mlflow.yml logs postgres

# Check MLflow logs
docker compose -f docker-compose.mlflow.yml logs mlflow

# Restart services
docker compose -f docker-compose.mlflow.yml restart
```

### Connection Errors

```python
# Test connection
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")
```

### Artifact Upload Failures

```bash
# Check disk space
df -h

# Check permissions on artifact directory
ls -la /path/to/artifacts

# Verify S3 credentials if using S3
aws s3 ls s3://my-bucket/mlflow-artifacts/
```

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [DSPy Optimization Spec](../docs/specs/dspy-optimization/spec.md)
- [Implementation Plan](../docs/specs/dspy-optimization/plan.md)
