# Kubernetes Deployment Guide

Production-ready Kubernetes manifests for AgentCore A2A Protocol Layer.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- PostgreSQL database (deployed separately or using operator)
- Redis cache (optional, for future features)
- Prometheus Operator (for monitoring)

## Quick Start

### 1. Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### 2. Configure Secrets

**IMPORTANT**: Update secrets before deploying:

```bash
# Generate secure JWT secret
kubectl create secret generic a2a-protocol-secrets \
  --namespace=agentcore \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -base64 32)" \
  --from-literal=POSTGRES_USER="agentcore" \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 32)"
```

Or edit `secret.yaml` with your values and apply:

```bash
kubectl apply -f secret.yaml
```

### 3. Update ConfigMap

Edit `configmap.yaml` to match your environment:
- Update `POSTGRES_HOST` if using external database
- Adjust `DATABASE_POOL_SIZE` based on expected load
- Configure `REDIS_URL` when Redis is available

```bash
kubectl apply -f configmap.yaml
```

### 4. Create RBAC Resources

```bash
kubectl apply -f serviceaccount.yaml
```

### 5. Deploy Application

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 6. Setup Autoscaling

```bash
# Ensure metrics-server is installed
kubectl apply -f hpa.yaml
```

### 7. Setup Monitoring (Optional)

If using Prometheus Operator:

```bash
kubectl apply -f servicemonitor.yaml
```

## Verify Deployment

```bash
# Check pod status
kubectl get pods -n agentcore -l app=a2a-protocol

# Check logs
kubectl logs -n agentcore -l app=a2a-protocol --tail=100

# Check service
kubectl get svc -n agentcore a2a-protocol

# Port forward for local testing
kubectl port-forward -n agentcore svc/a2a-protocol 8001:80
```

Test endpoints:
```bash
curl http://localhost:8001/api/v1/health
curl http://localhost:8001/api/v1/health/ready
curl http://localhost:8001/metrics
```

## Architecture

### Deployment Strategy

- **Replicas**: 3 (minimum for high availability)
- **Update Strategy**: RollingUpdate with maxSurge=1, maxUnavailable=0
- **Autoscaling**: HPA scales 3-10 replicas based on CPU/memory

### Security Features

- Non-root container (UID 1000)
- Read-only root filesystem
- Security context with seccomp profile
- Dropped all capabilities
- Pod anti-affinity for distribution across nodes
- RBAC with minimal permissions

### Resource Limits

**Per Pod:**
- Requests: 512Mi memory, 500m CPU
- Limits: 2Gi memory, 2000m CPU

**Init Container (migrations):**
- Requests: 128Mi memory, 100m CPU
- Limits: 256Mi memory, 200m CPU

### Health Checks

- **Liveness**: `/api/v1/health/live` - checks if app is alive
- **Readiness**: `/api/v1/health/ready` - checks database/dependencies
- Initial delay: 30s (liveness), 10s (readiness)

## Database Migrations

Migrations run automatically via init container before app starts.

To run manually:

```bash
kubectl exec -it -n agentcore deployment/a2a-protocol -- \
  alembic upgrade head
```

## Monitoring

### Prometheus Metrics

Metrics available at `/metrics`:
- HTTP request duration
- Request count by endpoint
- Active connections
- Database connection pool stats
- Custom business metrics

### Grafana Dashboards

Import dashboard from `monitoring/grafana-dashboard.json` (create this separately)

### Logs

Structured JSON logs with levels:
- ERROR: Application errors
- WARNING: Degraded service
- INFO: Normal operations
- DEBUG: Detailed debugging (disable in production)

View logs:
```bash
kubectl logs -n agentcore -l app=a2a-protocol -f
```

## Scaling

### Horizontal Scaling

HPA automatically scales based on:
- CPU > 70%
- Memory > 80%

Manual scaling:
```bash
kubectl scale deployment a2a-protocol -n agentcore --replicas=5
```

### Vertical Scaling

Update resource limits in `deployment.yaml` and reapply.

## Troubleshooting

### Pods Not Starting

```bash
# Check events
kubectl describe pod -n agentcore <pod-name>

# Check logs
kubectl logs -n agentcore <pod-name>

# Check init container
kubectl logs -n agentcore <pod-name> -c migrate-db
```

### Database Connection Issues

1. Verify database is accessible
2. Check credentials in secrets
3. Verify POSTGRES_HOST in configmap
4. Check network policies

### High Memory Usage

1. Check for memory leaks
2. Reduce DATABASE_POOL_SIZE
3. Increase pod memory limits
4. Scale horizontally

## Production Checklist

- [ ] Update all secrets with strong passwords
- [ ] Configure external PostgreSQL with backups
- [ ] Setup Redis for caching (future)
- [ ] Configure ingress/load balancer
- [ ] Enable TLS termination
- [ ] Setup log aggregation (ELK, Loki)
- [ ] Configure alert rules in Prometheus
- [ ] Test disaster recovery procedures
- [ ] Document runbook for common issues
- [ ] Configure network policies
- [ ] Enable pod security policies
- [ ] Setup automated backups

## Advanced Configuration

### Multi-Region Deployment

For global deployment, use:
- External database with replication
- Redis cluster for session storage
- Cross-region load balancing
- Geo-routing with DNS

### High Availability

- Deploy across multiple availability zones
- Use pod anti-affinity (already configured)
- Configure PodDisruptionBudget
- Use StatefulSet for stateful components

## Cleanup

```bash
kubectl delete namespace agentcore
```

Or delete resources individually:
```bash
kubectl delete -f .
```