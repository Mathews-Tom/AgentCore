# Staging Deployment Guide - Bounded Context Reasoning

## Pre-Deployment Checklist

### Environment Preparation
- [ ] Staging Kubernetes cluster accessible
- [ ] kubectl configured for staging context
- [ ] Staging database (PostgreSQL) provisioned
- [ ] Staging Redis instance provisioned
- [ ] Staging LLM API keys configured
- [ ] Monitoring stack (Prometheus + Grafana) deployed

### Configuration Review
- [ ] Environment variables validated (see `docs/configuration/reasoning-config.md`)
- [ ] Secrets created in Kubernetes
- [ ] ConfigMaps updated with staging values
- [ ] Rate limits appropriate for staging load
- [ ] JWT secret key generated (not production key)

### Code Review
- [ ] All tests passing (`uv run pytest`)
- [ ] Code coverage >= 90%
- [ ] Security scan completed (bandit, safety)
- [ ] Performance benchmarks meet targets
- [ ] Documentation up to date

## Deployment Steps

### 1. Build Docker Image

```bash
# Build image
docker build -t agentcore/reasoning:staging-$(git rev-parse --short HEAD) .

# Tag latest
docker tag agentcore/reasoning:staging-$(git rev-parse --short HEAD) agentcore/reasoning:staging-latest

# Push to registry
docker push agentcore/reasoning:staging-$(git rev-parse --short HEAD)
docker push agentcore/reasoning:staging-latest
```

### 2. Apply Kubernetes Manifests

```bash
# Set kubectl context
kubectl config use-context staging

# Create namespace if needed
kubectl create namespace agentcore-staging || true

# Apply configs
kubectl apply -f k8s/staging/configmap.yaml
kubectl apply -f k8s/staging/secrets.yaml

# Deploy application
kubectl apply -f k8s/staging/deployment.yaml
kubectl apply -f k8s/staging/service.yaml

# Deploy monitoring
kubectl apply -f k8s/staging/servicemonitor.yaml
```

### 3. Database Migration

```bash
# Connect to staging database pod
kubectl exec -it deployment/agentcore-staging -- /bin/bash

# Run migrations
uv run alembic upgrade head

# Verify migration
uv run alembic current
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n agentcore-staging -l app=agentcore

# Check logs
kubectl logs -f deployment/agentcore-staging --tail=100

# Check readiness
kubectl get deployment agentcore-staging -o jsonpath='{.status.readyReplicas}'
```

## Health Checks

### Application Health

```bash
# Health check endpoint
curl http://staging.agentcore.internal/api/v1/health

# Expected response
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-10-16T00:00:00Z"
}
```

### Database Connectivity

```bash
# Database health
curl http://staging.agentcore.internal/api/v1/health/db

# Expected: {"database": "connected"}
```

### Redis Connectivity

```bash
# Redis health
curl http://staging.agentcore.internal/api/v1/health/redis

# Expected: {"redis": "connected"}
```

### Metrics Endpoint

```bash
# Check metrics
curl http://staging.agentcore.internal/metrics | grep reasoning_bounded_context

# Should see metrics like:
# reasoning_bounded_context_requests_total{status="success"} 0
```

## Smoke Tests

### 1. Authentication Test

```bash
# Get JWT token
TOKEN=$(curl -X POST http://staging.agentcore.internal/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "test-agent", "credentials": {"api_key": "test-key"}}' \
  | jq -r .access_token)

# Verify token
echo $TOKEN
```

### 2. Basic Reasoning Request

```bash
# Send test reasoning request
curl -X POST http://staging.agentcore.internal/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "auth_token": "'$TOKEN'",
      "query": "What is 2+2?",
      "max_iterations": 1
    },
    "id": 1
  }'

# Expected: Success response with answer
```

### 3. Rate Limit Test

```bash
# Send burst of requests
for i in {1..10}; do
  curl -X POST http://staging.agentcore.internal/api/v1/jsonrpc \
    -H "Content-Type: application/json" \
    -d '{
      "jsonrpc": "2.0",
      "method": "reasoning.bounded_context",
      "params": {
        "auth_token": "'$TOKEN'",
        "query": "Test query '$i'"
      },
      "id": '$i'
    }' &
done
wait

# Check for rate limit responses (HTTP 429)
```

### 4. Error Handling Test

```bash
# Test invalid request
curl -X POST http://staging.agentcore.internal/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "auth_token": "invalid-token",
      "query": ""
    },
    "id": 1
  }'

# Expected: Error response with proper error code
```

## Monitoring Verification

### Grafana Dashboard

1. Open Grafana: `https://grafana.staging.internal`
2. Navigate to "AgentCore - Bounded Context Reasoning" dashboard
3. Verify panels are populating with data
4. Check for any error spikes or anomalies

### Prometheus Alerts

```bash
# Check Prometheus rules
curl http://prometheus.staging.internal/api/v1/rules | jq '.data.groups[] | select(.name == "reasoning")'

# Verify alerts are not firing
curl http://prometheus.staging.internal/api/v1/alerts | jq '.data.alerts[] | select(.labels.component == "reasoning")'
```

## Rollback Procedure

If deployment fails or issues are detected:

### 1. Quick Rollback

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/agentcore-staging -n agentcore-staging

# Verify rollback
kubectl rollout status deployment/agentcore-staging -n agentcore-staging
```

### 2. Database Rollback

```bash
# Rollback one migration
kubectl exec -it deployment/agentcore-staging -- uv run alembic downgrade -1

# Verify
kubectl exec -it deployment/agentcore-staging -- uv run alembic current
```

### 3. Verify Service Recovery

```bash
# Check health after rollback
curl http://staging.agentcore.internal/api/v1/health

# Check logs for errors
kubectl logs deployment/agentcore-staging --tail=100
```

## Post-Deployment Validation

### Performance Testing

```bash
# Run load test
uv run locust -f tests/load/reasoning_load_test.py \
  --host=http://staging.agentcore.internal \
  --users=10 \
  --spawn-rate=2 \
  --run-time=5m \
  --headless

# Check results for:
# - Average response time < 2s
# - Error rate < 1%
# - Token savings > 40%
```

### Integration Testing

```bash
# Run integration tests against staging
STAGING_URL=http://staging.agentcore.internal \
  uv run pytest tests/integration/ -v
```

### Security Testing

```bash
# Run security scan
bandit -r src/agentcore/reasoning/

# Check for vulnerabilities
safety check --json
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n agentcore-staging

# Check logs
kubectl logs <pod-name> -n agentcore-staging --previous

# Common issues:
# - ImagePullBackOff: Check image tag and registry credentials
# - CrashLoopBackOff: Check application logs for startup errors
# - Pending: Check resource quotas and node capacity
```

### Database Connection Failures

```bash
# Test database connectivity from pod
kubectl exec -it deployment/agentcore-staging -- psql $DATABASE_URL -c "SELECT 1"

# Check database secrets
kubectl get secret agentcore-db-credentials -n agentcore-staging -o yaml

# Verify network policies allow database access
```

### High Error Rates

```bash
# Check error logs
kubectl logs deployment/agentcore-staging -n agentcore-staging | grep ERROR

# Check Prometheus for error metrics
curl http://prometheus.staging.internal/api/v1/query?query=reasoning_bounded_context_errors_total

# Review Grafana error dashboard
```

## Success Criteria

Deployment is considered successful when:

- [ ] All pods are running and healthy (Ready 1/1)
- [ ] Health checks passing for 5 minutes
- [ ] Smoke tests all passing
- [ ] Metrics being collected and visible in Grafana
- [ ] No alerts firing in Prometheus
- [ ] Performance tests meet targets (p95 < 2s)
- [ ] Error rate < 1% over 10 minute period
- [ ] Integration tests passing

## Next Steps

After successful staging deployment:

1. Monitor for 24-48 hours
2. Run extended load tests
3. Gather performance metrics
4. Document any issues encountered
5. Update rollout plan based on staging learnings
6. Schedule production deployment review

## References

- Configuration Guide: `docs/configuration/reasoning-config.md`
- Production Rollout Plan: `docs/deployment/production-rollout.md`
- Kubernetes Manifests: `k8s/staging/`
- Monitoring: `grafana/README.md`
