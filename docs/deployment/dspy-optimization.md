# DSPy Optimization Service - Deployment Guide

This guide covers deployment procedures for the DSPy optimization service in production environments.

## Overview

The DSPy optimization service provides REST API endpoints for systematic AI optimization using MIPROv2 and GEPA algorithms. It runs as a containerized service in Kubernetes with auto-scaling, monitoring, and MLflow integration.

**Service Details:**
- **Port:** 8002
- **Base Path:** `/api/v1`
- **Health Checks:** `/api/v1/health/live`, `/api/v1/health/ready`
- **Metrics:** `/metrics` (Prometheus format)

## Prerequisites

- Kubernetes cluster (v1.28+)
- kubectl configured with cluster access
- Docker or container registry access
- MLflow tracking server (optional but recommended)
- Redis for job queue management
- Prometheus/Grafana for monitoring (optional)

## Quick Start

### Local Development

```bash
# Run service locally with hot reload
uv run uvicorn agentcore.dspy_optimization.main:app \
  --host 0.0.0.0 \
  --port 8002 \
  --reload

# Service available at http://localhost:8002
# API docs at http://localhost:8002/docs
```

### Docker Build

```bash
# Build Docker image
docker build -f Dockerfile.dspy -t agentcore/dspy-optimization:latest .

# Run container locally
docker run -p 8002:8002 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e REDIS_URL=redis://redis:6379/1 \
  agentcore/dspy-optimization:latest
```

### Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/dspy-serviceaccount.yaml
kubectl apply -f k8s/dspy-configmap.yaml
kubectl apply -f k8s/dspy-secret.yaml
kubectl apply -f k8s/dspy-deployment.yaml
kubectl apply -f k8s/dspy-service.yaml
kubectl apply -f k8s/dspy-hpa.yaml
kubectl apply -f k8s/dspy-servicemonitor.yaml

# Verify deployment
kubectl get pods -n agentcore -l app=dspy-optimization
kubectl rollout status deployment/dspy-optimization -n agentcore
```

## Configuration

### Environment Variables

Required configuration in `k8s/dspy-configmap.yaml`:

```yaml
LOG_LEVEL: "INFO"
MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
MLFLOW_EXPERIMENT_NAME: "dspy-optimization"
MAX_OPTIMIZATION_TIME: "3600"
MAX_ITERATIONS: "100"
MAX_CONCURRENT_OPTIMIZATIONS: "10"
REDIS_URL: "redis://redis-service:6379/1"
```

### Secrets

Required secrets in `k8s/dspy-secret.yaml`:

```yaml
OPENAI_API_KEY: ""           # OpenAI API key
ANTHROPIC_API_KEY: ""        # Anthropic API key
GOOGLE_API_KEY: ""           # Google API key
MODEL_ENCRYPTION_KEY: ""     # Encryption key for model storage
```

**Important:** Encode secrets as base64 before applying:

```bash
echo -n "your-api-key" | base64
```

### Resource Limits

Default resource allocation per pod:

```yaml
requests:
  memory: "1Gi"
  cpu: "1000m"
limits:
  memory: "4Gi"
  cpu: "4000m"
```

Adjust based on optimization workload and cluster capacity.

## Scaling

### Horizontal Pod Autoscaling

Auto-scaling is configured via `k8s/dspy-hpa.yaml`:

- **Min replicas:** 2
- **Max replicas:** 8
- **CPU threshold:** 75%
- **Memory threshold:** 85%

Manual scaling:

```bash
# Scale to specific replica count
kubectl scale deployment/dspy-optimization -n agentcore --replicas=5

# View current scale
kubectl get hpa -n agentcore dspy-optimization-hpa
```

### Vertical Scaling

To increase resources per pod, edit `k8s/dspy-deployment.yaml` and apply:

```bash
kubectl apply -f k8s/dspy-deployment.yaml
kubectl rollout restart deployment/dspy-optimization -n agentcore
```

## Monitoring

### Prometheus Metrics

Metrics exposed at `/metrics`:

- `dspy_optimization_active_jobs` - Active optimization jobs
- `dspy_optimization_queue_size` - Queued jobs waiting
- `dspy_optimization_completed_total` - Total completed optimizations
- `dspy_optimization_duration_seconds` - Optimization duration
- `dspy_optimization_score` - Algorithm performance scores
- `http_requests_total` - HTTP request count by status
- `http_request_duration_seconds` - Request latency histogram

### Grafana Dashboard

Import dashboard from `monitoring/grafana/dspy-dashboard.json`:

1. Open Grafana
2. Navigate to **Dashboards** > **Import**
3. Upload `monitoring/grafana/dspy-dashboard.json`
4. Select Prometheus data source

### Alerts

Prometheus alerts configured in `monitoring/prometheus-rules.yaml`:

- **DSPyHighCPUUsage** - CPU usage > 80% for 5m
- **DSPyHighMemoryUsage** - Memory usage > 90% for 5m
- **DSPyPodRestarts** - Pod restart detected
- **DSPyServiceDown** - Service unavailable for 2m
- **DSPyHighErrorRate** - Error rate > 5% for 5m
- **DSPySlowResponseTime** - p95 latency > 5s for 5m
- **DSPyLongRunningOptimization** - Optimization > 1h
- **DSPyQueueBacklog** - Queue size > 50 for 10m

Apply alert rules:

```bash
kubectl apply -f monitoring/prometheus-rules.yaml
```

## Health Checks

### Liveness Probe

Checks if service is alive:

```bash
curl http://dspy-optimization.agentcore.svc.cluster.local/api/v1/health/live
```

Response:
```json
{"status": "healthy", "version": "0.1.0"}
```

### Readiness Probe

Checks if service is ready to accept traffic:

```bash
curl http://dspy-optimization.agentcore.svc.cluster.local/api/v1/health/ready
```

Response:
```json
{"status": "ready", "version": "0.1.0"}
```

## CI/CD Pipeline

Automated deployment via GitHub Actions (`.github/workflows/dspy-deployment.yml`):

### Triggers

- **Push to main** - Automatic deployment to staging
- **Manual dispatch** - Choose staging or production
- **Pull request** - Build and test only (no deployment)

### Workflow Steps

1. **Test** - Run pytest with 90% coverage requirement
2. **Build** - Build multi-arch Docker image (amd64, arm64)
3. **Deploy** - Apply Kubernetes manifests to target environment
4. **Verify** - Run health checks and smoke tests

### Manual Deployment

```bash
# Trigger workflow via GitHub CLI
gh workflow run dspy-deployment.yml -f environment=production
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n agentcore -l app=dspy-optimization

# View pod logs
kubectl logs -n agentcore -l app=dspy-optimization --tail=100

# Describe pod for events
kubectl describe pod -n agentcore <pod-name>
```

Common issues:
- Missing secrets (API keys)
- MLflow connection failure
- Redis connection failure
- Insufficient resources

### High Memory Usage

```bash
# Check current memory usage
kubectl top pods -n agentcore -l app=dspy-optimization

# Increase memory limits in deployment
kubectl edit deployment/dspy-optimization -n agentcore
```

### Slow Response Times

1. Check active optimizations:
```bash
curl http://dspy-optimization/metrics | grep dspy_optimization_active_jobs
```

2. Scale up replicas if queue is backed up:
```bash
kubectl scale deployment/dspy-optimization -n agentcore --replicas=6
```

3. Review optimization timeouts in ConfigMap

### MLflow Connection Issues

```bash
# Test MLflow connectivity from pod
kubectl exec -it -n agentcore <pod-name> -- curl http://mlflow-service:5000/health

# Check MLflow credentials in secret
kubectl get secret dspy-optimization-secrets -n agentcore -o yaml
```

## Security

### Container Security

- **Non-root user:** Runs as UID 1001
- **Read-only filesystem:** Enabled
- **No privilege escalation:** Enforced
- **Capabilities dropped:** All Linux capabilities dropped
- **Seccomp profile:** RuntimeDefault

### Network Security

- **Internal traffic only:** ClusterIP service type
- **TLS recommended:** Use Ingress with TLS termination
- **CORS configured:** Adjust origins in `main.py` for production

### Secret Management

Best practices:
1. Use external secret managers (Vault, AWS Secrets Manager)
2. Rotate API keys regularly
3. Enable secret encryption at rest in Kubernetes
4. Audit secret access via RBAC

## Rollback

### Rollback to Previous Version

```bash
# View rollout history
kubectl rollout history deployment/dspy-optimization -n agentcore

# Rollback to previous revision
kubectl rollout undo deployment/dspy-optimization -n agentcore

# Rollback to specific revision
kubectl rollout undo deployment/dspy-optimization -n agentcore --to-revision=2
```

### Emergency Rollback

```bash
# Scale down to zero (stop all traffic)
kubectl scale deployment/dspy-optimization -n agentcore --replicas=0

# Investigate issues, then scale back up
kubectl scale deployment/dspy-optimization -n agentcore --replicas=2
```

## Performance Tuning

### Optimization Workers

Adjust worker count in `Dockerfile.dspy`:

```dockerfile
CMD ["uvicorn", "agentcore.dspy_optimization.main:app", \
     "--workers", "4", \  # Increase for more concurrency
     ...]
```

### Redis Configuration

For high-throughput scenarios:
- Enable Redis clustering
- Adjust connection pool size
- Configure persistence based on requirements

### MLflow Performance

- Use remote artifact storage (S3, GCS, Azure Blob)
- Enable result caching
- Configure database backend (PostgreSQL recommended)

## Support

For deployment issues or questions:
- **GitHub Issues:** https://github.com/agentcore/agentcore/issues
- **Documentation:** https://docs.agentcore.ai
- **Slack:** #dspy-optimization channel

## Version History

- **v0.1.0** (2025-01-29) - Initial production deployment
  - MIPROv2 and GEPA algorithms
  - MLflow integration
  - Kubernetes deployment
  - Prometheus monitoring
  - CI/CD pipeline
