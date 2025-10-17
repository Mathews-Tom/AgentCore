# Training System Operational Runbook

**Version:** 1.0
**Last Updated:** 2025-10-17
**Team:** ML Infrastructure / Training Operations

---

## Overview

This runbook provides operational procedures, incident response playbooks, and maintenance guidelines for the AgentCore training infrastructure. It covers deployment, monitoring, troubleshooting, and emergency procedures.

**System Components:**
- Training API (JSON-RPC 2.0)
- GRPO Training Engine
- Redis Job Queue
- PostgreSQL Database
- Prometheus Metrics
- Grafana Dashboards
- Worker Pools (Kubernetes)

---

## Architecture Overview

### System Diagram

```
┌─────────────┐
│   Clients   │
│   (API)     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Training API (FastAPI)            │
│   - JSON-RPC Handler                │
│   - Authentication                  │
│   - Rate Limiting                   │
└──────┬──────────────────────────────┘
       │
       ├────────────────┬──────────────┐
       ▼                ▼              ▼
┌────────────┐   ┌──────────┐   ┌──────────┐
│  Redis     │   │PostgreSQL│   │Prometheus│
│  Queue     │   │ Database │   │ Metrics  │
└──────┬─────┘   └──────────┘   └──────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Worker Pool (K8s Deployment)      │
│   - GRPO Trainer                    │
│   - Trajectory Collection           │
│   - Reward Computation              │
└─────────────────────────────────────┘
```

### Key Services

| Service | Technology | Port | Purpose |
|---------|-----------|------|---------|
| Training API | FastAPI | 8001 | JSON-RPC endpoints |
| Redis | Redis Cluster | 6379 | Job queue and caching |
| PostgreSQL | PostgreSQL 15 | 5432 | Training data storage |
| Prometheus | Prometheus | 9090 | Metrics collection |
| Grafana | Grafana | 3000 | Metrics visualization |
| Workers | Python | - | Background job processing |

---

## Deployment Procedures

### Initial Deployment

#### Prerequisites

```bash
# Verify Kubernetes cluster access
kubectl cluster-info

# Verify Helm is installed
helm version

# Create namespace
kubectl create namespace training
```

#### Deploy Database (PostgreSQL)

```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy PostgreSQL with persistence
helm install training-db bitnami/postgresql \
  --namespace training \
  --set auth.postgresPassword=<secure-password> \
  --set auth.database=training \
  --set primary.persistence.size=100Gi \
  --set primary.resources.requests.memory=4Gi \
  --set primary.resources.requests.cpu=2
```

#### Deploy Redis

```bash
# Deploy Redis cluster
helm install training-redis bitnami/redis-cluster \
  --namespace training \
  --set cluster.nodes=6 \
  --set cluster.replicas=1 \
  --set persistence.size=20Gi \
  --set password=<secure-password>
```

#### Run Database Migrations

```bash
# Get database pod name
DB_POD=$(kubectl get pods -n training -l app.kubernetes.io/name=postgresql -o jsonpath='{.items[0].metadata.name}')

# Run migrations
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c "SELECT version();"

# Apply Alembic migrations
kubectl run -i --rm --restart=Never migrate \
  --image=agentcore/training:latest \
  --namespace=training \
  --env="DATABASE_URL=postgresql://postgres:<password>@training-db:5432/training" \
  -- uv run alembic upgrade head
```

#### Deploy Training API

```bash
# Create secret for credentials
kubectl create secret generic training-secrets \
  --namespace=training \
  --from-literal=postgres-password=<postgres-password> \
  --from-literal=redis-password=<redis-password> \
  --from-literal=jwt-secret=<jwt-secret>

# Deploy API
kubectl apply -f k8s/training-api-deployment.yaml

# Verify deployment
kubectl rollout status deployment/training-api -n training
kubectl get pods -n training -l app=training-api
```

#### Deploy Worker Pool

```bash
# Deploy workers
kubectl apply -f k8s/training-worker-deployment.yaml

# Verify workers
kubectl get pods -n training -l app=training-worker
```

#### Deploy Monitoring Stack

```bash
# Add Prometheus community repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Deploy Prometheus stack (includes Grafana)
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace training \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
```

### Rolling Updates

```bash
# Update API image
kubectl set image deployment/training-api \
  training-api=agentcore/training:v1.2.0 \
  --namespace=training

# Watch rollout
kubectl rollout status deployment/training-api -n training

# Rollback if needed
kubectl rollout undo deployment/training-api -n training
```

### Scaling Procedures

#### Scale Workers

```bash
# Manual scaling
kubectl scale deployment/training-worker --replicas=10 -n training

# Enable HPA (auto-scaling)
kubectl apply -f k8s/training-worker-hpa.yaml

# Verify HPA
kubectl get hpa -n training
```

#### Scale Database Connections

```bash
# Update PostgreSQL max connections
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "ALTER SYSTEM SET max_connections = 200;"

# Restart PostgreSQL
kubectl rollout restart statefulset/training-db-postgresql -n training
```

---

## Monitoring and Alerting

### Key Metrics

#### Training Job Metrics

```promql
# Active training jobs
sum(training_jobs_active)

# Job completion rate (last 1h)
rate(training_jobs_completed_total[1h])

# Job failure rate
rate(training_jobs_failed_total[1h]) / rate(training_jobs_total[1h])

# Average job duration
rate(training_job_duration_seconds_sum[1h]) / rate(training_job_duration_seconds_count[1h])
```

#### System Metrics

```promql
# API request rate
rate(http_requests_total{service="training-api"}[5m])

# API latency (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Worker CPU usage
sum(rate(container_cpu_usage_seconds_total{pod=~"training-worker.*"}[5m])) by (pod)

# Redis queue depth
redis_queue_length{queue="training_jobs"}
```

#### Resource Metrics

```promql
# Database connections
pg_stat_activity_count

# Database query latency
pg_stat_statements_mean_exec_time_seconds

# Redis memory usage
redis_memory_used_bytes / redis_memory_max_bytes

# Disk usage (PostgreSQL)
node_filesystem_avail_bytes{mountpoint="/var/lib/postgresql/data"}
```

### Grafana Dashboards

**Training Overview Dashboard:**
- Active jobs count
- Job success/failure rates
- Average iteration time
- Budget consumption
- Worker utilization

**System Health Dashboard:**
- API request rate and latency
- Database query performance
- Redis queue metrics
- Resource usage (CPU, memory, disk)

**Training Job Details Dashboard:**
- Per-job metrics (loss, reward, accuracy)
- Trajectory generation rate
- Checkpoint creation status
- Cost tracking

### Alert Rules

#### Critical Alerts

**Training Job Failures:**
```yaml
- alert: HighTrainingJobFailureRate
  expr: rate(training_jobs_failed_total[10m]) / rate(training_jobs_total[10m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High training job failure rate detected"
    description: "{{ $value | humanizePercentage }} of jobs are failing"
```

**Database Connection Exhaustion:**
```yaml
- alert: DatabaseConnectionsHigh
  expr: pg_stat_activity_count / pg_settings_max_connections > 0.9
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Database connection pool nearly exhausted"
```

**Redis Queue Backup:**
```yaml
- alert: RedisQueueBackup
  expr: redis_queue_length{queue="training_jobs"} > 1000
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Training job queue is backing up"
```

#### Warning Alerts

**Budget Overrun Risk:**
```yaml
- alert: TrainingBudgetWarning
  expr: training_job_cost_usd / training_job_budget_usd > 0.9
  labels:
    severity: warning
  annotations:
    summary: "Training job {{ $labels.job_id }} near budget limit"
```

---

## Incident Response

### Incident Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| **P0 - Critical** | System down, data loss risk | < 15 min | API outage, database failure |
| **P1 - High** | Major functionality impaired | < 1 hour | High error rate, job failures |
| **P2 - Medium** | Degraded performance | < 4 hours | Slow queries, queue backup |
| **P3 - Low** | Minor issues, no user impact | < 24 hours | Logging errors, metric gaps |

### Incident Response Playbooks

#### P0: Training API Outage

**Symptoms:**
- 5xx errors on `/api/v1/jsonrpc`
- Health check failures
- Alert: `TrainingAPIDown`

**Investigation:**

```bash
# Check pod status
kubectl get pods -n training -l app=training-api

# Check pod logs
kubectl logs -n training deployment/training-api --tail=100

# Check events
kubectl get events -n training --sort-by='.lastTimestamp' | head -20
```

**Resolution:**

```bash
# If pod is CrashLooping
kubectl describe pod -n training <pod-name>

# Check database connectivity
kubectl exec -n training <api-pod> -- nc -zv training-db 5432

# Check Redis connectivity
kubectl exec -n training <api-pod> -- nc -zv training-redis 6379

# Restart deployment if needed
kubectl rollout restart deployment/training-api -n training

# Rollback if recent deployment
kubectl rollout undo deployment/training-api -n training
```

**Escalation:**
- Notify on-call engineer immediately
- Page database team if DB is unreachable
- Create incident postmortem

#### P1: High Job Failure Rate

**Symptoms:**
- > 10% of training jobs failing
- Alert: `HighTrainingJobFailureRate`
- User reports of failed jobs

**Investigation:**

```bash
# Check recent failed jobs
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT job_id, agent_id, error_message, created_at
   FROM training_jobs
   WHERE status = 'failed'
   ORDER BY created_at DESC
   LIMIT 10;"

# Check worker logs
kubectl logs -n training deployment/training-worker --tail=200 | grep ERROR

# Check worker resource usage
kubectl top pods -n training -l app=training-worker
```

**Common Causes:**
- Out of memory errors → Scale workers or reduce batch size
- Model loading failures → Check model registry connectivity
- Budget exhaustion → Verify budget enforcement logic
- Database deadlocks → Check slow query log

**Resolution:**

```bash
# If OOM errors, increase worker memory
kubectl patch deployment training-worker -n training -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"worker","resources":{"limits":{"memory":"8Gi"}}}]}}}}'

# Scale workers if queue is backed up
kubectl scale deployment/training-worker --replicas=20 -n training

# Restart workers if stale
kubectl rollout restart deployment/training-worker -n training
```

#### P1: Database Performance Degradation

**Symptoms:**
- Slow API responses (p95 > 5s)
- Alert: `DatabaseSlowQueries`
- Timeouts in logs

**Investigation:**

```bash
# Check active queries
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT pid, now() - query_start as duration, state, query
   FROM pg_stat_activity
   WHERE state != 'idle'
   ORDER BY duration DESC;"

# Check table bloat
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT schemaname, tablename,
          pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables
   WHERE schemaname = 'public'
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Check locks
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Resolution:**

```bash
# Kill long-running query
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT pg_terminate_backend(<pid>);"

# Run VACUUM ANALYZE
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "VACUUM ANALYZE;"

# Add missing index (if identified)
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "CREATE INDEX CONCURRENTLY idx_training_jobs_status ON training_jobs(status);"
```

#### P2: Redis Queue Backup

**Symptoms:**
- Queue length > 1000
- Alert: `RedisQueueBackup`
- Jobs slow to start

**Investigation:**

```bash
# Check queue length
kubectl exec -n training training-redis-master-0 -- redis-cli LLEN training_jobs

# Check worker count
kubectl get pods -n training -l app=training-worker --no-headers | wc -l

# Check worker CPU/memory
kubectl top pods -n training -l app=training-worker
```

**Resolution:**

```bash
# Scale workers
kubectl scale deployment/training-worker --replicas=30 -n training

# Check for stuck jobs
kubectl exec -n training training-redis-master-0 -- redis-cli LRANGE training_jobs 0 10

# If jobs are stuck, drain and restart
kubectl exec -n training training-redis-master-0 -- redis-cli DEL training_jobs_processing
```

---

## Maintenance Procedures

### Database Maintenance

#### Daily VACUUM

```bash
# Run VACUUM ANALYZE (non-blocking)
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "VACUUM ANALYZE;"
```

#### Weekly VACUUM FULL (Scheduled Maintenance)

```bash
# Schedule downtime window

# Stop API and workers
kubectl scale deployment/training-api --replicas=0 -n training
kubectl scale deployment/training-worker --replicas=0 -n training

# Run VACUUM FULL (requires lock)
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "VACUUM FULL ANALYZE;"

# Restart services
kubectl scale deployment/training-api --replicas=3 -n training
kubectl scale deployment/training-worker --replicas=10 -n training
```

#### Backup Database

```bash
# Create backup
kubectl exec -n training $DB_POD -- pg_dump -U postgres training | gzip > training_backup_$(date +%Y%m%d).sql.gz

# Upload to S3
aws s3 cp training_backup_$(date +%Y%m%d).sql.gz s3://agentcore-backups/training/
```

#### Restore Database

```bash
# Download backup
aws s3 cp s3://agentcore-backups/training/training_backup_20251017.sql.gz .

# Restore
gunzip -c training_backup_20251017.sql.gz | \
  kubectl exec -i -n training $DB_POD -- psql -U postgres training
```

### Redis Maintenance

#### Clear Old Completed Jobs

```bash
# Remove completed job data older than 7 days
kubectl exec -n training training-redis-master-0 -- redis-cli \
  EVAL "local keys = redis.call('keys', 'job:*:completed');
        for i=1,#keys do
          local age = redis.call('ttl', keys[i]);
          if age < 0 or age > 604800 then
            redis.call('del', keys[i])
          end
        end" 0
```

### Log Rotation

Logs are automatically rotated by Kubernetes, but for manual cleanup:

```bash
# Clear old logs (keeping last 1000 lines)
kubectl logs -n training deployment/training-api --tail=1000 > /tmp/api_logs.txt
```

### Certificate Renewal

```bash
# Check certificate expiration
kubectl get secret training-tls -n training -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -noout -dates

# Renew with cert-manager
kubectl delete certificate training-tls -n training
kubectl apply -f k8s/training-tls-certificate.yaml
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: Jobs Stuck in "queued" Status

**Symptoms:**
- Jobs remain in "queued" for > 5 minutes
- No workers processing jobs

**Investigation:**
```bash
# Check worker count
kubectl get pods -n training -l app=training-worker

# Check worker logs
kubectl logs -n training deployment/training-worker --tail=50
```

**Resolution:**
```bash
# Restart workers
kubectl rollout restart deployment/training-worker -n training

# Scale workers if needed
kubectl scale deployment/training-worker --replicas=20 -n training
```

#### Issue: Out of Memory Errors

**Symptoms:**
- Pods in CrashLoopBackoff
- Logs show "OOMKilled"

**Investigation:**
```bash
# Check pod status
kubectl describe pod -n training <pod-name> | grep -A 10 "Last State"

# Check resource usage
kubectl top pods -n training
```

**Resolution:**
```bash
# Increase memory limits
kubectl patch deployment training-worker -n training -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"worker","resources":{"limits":{"memory":"16Gi"}}}]}}}}'

# Or reduce batch size in training config
```

#### Issue: Database Connection Errors

**Symptoms:**
- API returns "connection pool exhausted"
- Logs show "FATAL: too many connections"

**Investigation:**
```bash
# Check active connections
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT count(*) FROM pg_stat_activity;"

# Check max connections
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SHOW max_connections;"
```

**Resolution:**
```bash
# Increase max connections
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "ALTER SYSTEM SET max_connections = 300;"

# Restart PostgreSQL
kubectl rollout restart statefulset/training-db-postgresql -n training

# Or increase connection pool size in API config
```

#### Issue: Budget Overruns

**Symptoms:**
- Jobs consuming more than max_budget_usd
- Alert: `TrainingBudgetExceeded`

**Investigation:**
```bash
# Check job costs
kubectl exec -n training $DB_POD -- psql -U postgres -d training -c \
  "SELECT job_id, cost_usd, budget_usd, (cost_usd / budget_usd * 100) as percent_used
   FROM training_jobs
   WHERE cost_usd > budget_usd * 0.9
   ORDER BY created_at DESC
   LIMIT 20;"
```

**Resolution:**
```bash
# Cancel expensive jobs
curl -X POST https://api.agentcore.ai/api/v1/jsonrpc \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{"jsonrpc":"2.0","method":"training.cancel","params":{"job_id":"<job-id>"},"id":"1"}'

# Update budget enforcement logic if needed
# Review cost tracking in src/agentcore/training/cost_tracker.py
```

---

## Emergency Procedures

### Emergency Shutdown

```bash
# Stop all training jobs
kubectl scale deployment/training-worker --replicas=0 -n training

# Stop API (prevents new jobs)
kubectl scale deployment/training-api --replicas=0 -n training

# Verify no jobs running
kubectl get pods -n training
```

### Emergency Restart

```bash
# Start database first
kubectl rollout restart statefulset/training-db-postgresql -n training
kubectl rollout status statefulset/training-db-postgresql -n training

# Start Redis
kubectl rollout restart statefulset/training-redis-master -n training

# Start API
kubectl scale deployment/training-api --replicas=3 -n training

# Start workers
kubectl scale deployment/training-worker --replicas=10 -n training

# Verify health
kubectl get pods -n training
```

### Data Recovery

If database corruption occurs:

```bash
# Stop all services
kubectl scale deployment/training-api --replicas=0 -n training
kubectl scale deployment/training-worker --replicas=0 -n training

# Restore from latest backup
aws s3 cp s3://agentcore-backups/training/training_backup_latest.sql.gz .
gunzip -c training_backup_latest.sql.gz | \
  kubectl exec -i -n training $DB_POD -- psql -U postgres training

# Run migrations to ensure schema is current
kubectl run -i --rm --restart=Never migrate \
  --image=agentcore/training:latest \
  --namespace=training \
  --env="DATABASE_URL=postgresql://postgres:<password>@training-db:5432/training" \
  -- uv run alembic upgrade head

# Restart services
kubectl scale deployment/training-api --replicas=3 -n training
kubectl scale deployment/training-worker --replicas=10 -n training
```

---

## Contact Information

**Primary On-Call:** training-oncall@agentcore.ai

**Escalation Contacts:**
- ML Infrastructure Lead: ml-infra-lead@agentcore.ai
- Database Team: db-team@agentcore.ai
- Platform Team: platform-team@agentcore.ai

**Incident Management:** PagerDuty / Slack #training-incidents

---

## Further Reading

- [Training API Reference](../api/training-api.md)
- [Developer Guide](../guides/training-agents.md)
- [Custom Rewards Guide](../guides/custom_rewards.md)
- [Architecture Overview](../architecture/training-system.md)
- [Kubernetes Deployment Guide](../ops/k8s-deployment.md)

---

**Document Version:** 1.0
**Last Reviewed:** 2025-10-17
**Next Review:** 2025-11-17
