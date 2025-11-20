# AgentCore Tool Integration Production Runbook

**Version:** 1.0
**Last Updated:** 2025-01-13
**Status:** Production Ready
**Component:** agent_runtime/tools

## Overview

This runbook provides comprehensive operational guidance for deploying, monitoring, and troubleshooting the AgentCore Tool Integration Framework in production environments.

**Quick Links:**
- [Deployment Guide](#1-deployment-guide)
- [Monitoring Setup](#2-monitoring-and-alerting)
- [Incident Response](#3-incident-response)
- [Troubleshooting](#4-troubleshooting-guide)
- [Maintenance Procedures](#5-maintenance-procedures)

---

## 1. Deployment Guide

### 1.1 Prerequisites

**Infrastructure Requirements:**
- Kubernetes 1.24+ or Docker Compose 2.0+
- PostgreSQL 14+ (primary database)
- Redis 7+ (rate limiting, quota management)
- Prometheus + Grafana (monitoring)
- Docker Registry (for custom tool images)

**Resource Requirements per Instance:**
- CPU: 4 cores minimum, 8 cores recommended
- Memory: 5GB minimum, 8GB recommended
- Disk: 20GB minimum for logs and temporary files
- Network: 200 Mbps bandwidth

**Access Requirements:**
- Database credentials (PostgreSQL)
- Redis connection URL
- External API keys (Google Search, etc.) - stored in Vault or env vars
- Docker Registry credentials (for pulling tool images)

### 1.2 Environment Configuration

**Production Environment Variables:**

```bash
# Core Configuration
export A2A_PROTOCOL_VERSION="0.2"
export LOG_LEVEL="INFO"
export DEBUG=false

# Database Configuration
export DATABASE_URL="postgresql+asyncpg://user:pass@postgres:5432/agentcore"
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=20
export DATABASE_POOL_TIMEOUT=30

# Redis Configuration
export REDIS_URL="redis://redis:6379/0"
export REDIS_POOL_SIZE=50
export REDIS_TIMEOUT=5

# Tool Framework Configuration
export TOOL_EXECUTION_TIMEOUT=30
export TOOL_MAX_RETRIES=3
export TOOL_RETRY_BACKOFF=1.5
export TOOL_RATE_LIMIT_ENABLED=true
export TOOL_SANDBOX_ENABLED=true

# External API Keys (Vault preferred)
export GOOGLE_API_KEY="$(vault kv get -field=api_key secret/tools/google)"
export GOOGLE_SEARCH_ENGINE_ID="$(vault kv get -field=engine_id secret/tools/google)"

# Security Configuration
export JWT_SECRET_KEY="$(vault kv get -field=jwt_secret secret/auth)"
export JWT_ALGORITHM="RS256"
export JWT_EXPIRATION_HOURS=1

# Monitoring Configuration
export ENABLE_METRICS=true
export ENABLE_TRACING=true
export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
```

### 1.3 Docker Compose Deployment

**Production Docker Compose:**

```yaml
# docker-compose.prod.yml
version: "3.9"

services:
  agentcore:
    image: agentcore:latest
    restart: always
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    volumes:
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock  # For code execution sandbox
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    restart: always
    environment:
      POSTGRES_DB: agentcore
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 4G

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 2G

  prometheus:
    image: prom/prometheus:latest
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

**Deploy:**
```bash
# Load environment variables
source .env.prod

# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Start services
docker compose -f docker-compose.prod.yml up -d

# Verify deployment
docker compose -f docker-compose.prod.yml ps
curl http://localhost:8001/health
```

### 1.4 Kubernetes Deployment

**Production Kubernetes Manifests:**

```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentcore
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentcore
  template:
    metadata:
      labels:
        app: agentcore
    spec:
      containers:
      - name: agentcore
        image: agentcore:1.0.0
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agentcore-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: agentcore-config
              key: redis-url
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: tool-api-keys
              key: google-api-key
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 10
        volumeMounts:
        - name: docker-socket
          mountPath: /var/run/docker.sock
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
---
apiVersion: v1
kind: Service
metadata:
  name: agentcore
  namespace: production
spec:
  selector:
    app: agentcore
  ports:
  - port: 8001
    targetPort: 8001
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentcore-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentcore
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
```

**Deploy to Kubernetes:**
```bash
# Create namespace
kubectl create namespace production

# Apply secrets
kubectl apply -f k8s/production/secrets.yaml

# Apply config maps
kubectl apply -f k8s/production/configmap.yaml

# Deploy application
kubectl apply -f k8s/production/deployment.yaml

# Verify deployment
kubectl get pods -n production
kubectl logs -n production deployment/agentcore --tail=100

# Check health
kubectl port-forward -n production svc/agentcore 8001:8001
curl http://localhost:8001/health
```

---

## 2. Monitoring and Alerting

### 2.1 Prometheus Configuration

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agentcore'
    static_configs:
      - targets: ['agentcore:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### 2.2 Grafana Dashboards

**Import Dashboards:**
1. Tool Integration Dashboard (ID: tool-integration.json)
2. Tool Performance Dashboard (ID: tool-performance.json)
3. Tool Costs & Quotas Dashboard (ID: tool-costs.json)

**Dashboard Locations:**
- `k8s/monitoring/tool-dashboards.yaml`

**Import via UI:**
```bash
# Access Grafana
http://localhost:3000

# Login with admin credentials
# Navigate to: Dashboards > Import
# Upload: k8s/monitoring/tool-dashboards.yaml
```

### 2.3 Alert Rules

**Critical Alerts (PagerDuty):**

```yaml
# prometheus-alerts.yml
groups:
  - name: tool_integration_critical
    interval: 1m
    rules:
      - alert: HighFrameworkOverhead
        expr: histogram_quantile(0.95, sum(rate(framework_overhead_seconds_bucket[5m])) by (le)) > 0.1
        for: 5m
        labels:
          severity: critical
          component: tool_integration
        annotations:
          summary: "Framework overhead p95 > 100ms"
          description: "Tool framework overhead has been above 100ms for 5 minutes. Current: {{ $value }}ms"

      - alert: LowToolSuccessRate
        expr: sum(rate(agentcore_tool_executions_total{status="success"}[5m])) / sum(rate(agentcore_tool_executions_total[5m])) < 0.95
        for: 10m
        labels:
          severity: critical
          component: tool_integration
        annotations:
          summary: "Tool success rate < 95%"
          description: "Tool success rate has been below 95% for 10 minutes. Current: {{ $value | humanizePercentage }}"

      - alert: HighErrorRate
        expr: sum(rate(agentcore_tool_errors_total[5m])) / sum(rate(agentcore_tool_executions_total[5m])) > 0.10
        for: 5m
        labels:
          severity: critical
          component: tool_integration
        annotations:
          summary: "Error rate > 10%"
          description: "Tool error rate has been above 10% for 5 minutes. Current: {{ $value | humanizePercentage }}"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
          component: tool_integration
        annotations:
          summary: "Redis is down"
          description: "Redis has been down for 1 minute. Rate limiting and quotas disabled."
```

**Warning Alerts (Slack):**

```yaml
  - name: tool_integration_warning
    interval: 1m
    rules:
      - alert: ModerateFrameworkOverhead
        expr: histogram_quantile(0.95, sum(rate(framework_overhead_seconds_bucket[5m])) by (le)) > 0.08
        for: 10m
        labels:
          severity: warning
          component: tool_integration
        annotations:
          summary: "Framework overhead p95 > 80ms"

      - alert: HighConcurrency
        expr: tool_execution_concurrency > 1200
        for: 10m
        labels:
          severity: warning
          component: tool_integration
        annotations:
          summary: "High concurrent executions > 1200"
          description: "Consider scaling horizontally"
```

### 2.4 Key Metrics to Monitor

**Latency Metrics:**
- `framework_overhead_seconds{quantile="0.95"}` < 100ms
- `tool_execution_duration_seconds{quantile="0.95"}` by tool_id

**Throughput Metrics:**
- `rate(agentcore_tool_executions_total[5m])` by tool_id, status
- `tool_execution_concurrency` (current concurrent executions)

**Error Metrics:**
- `rate(agentcore_tool_errors_total[5m])` by error_type
- `tool_execution_success_rate` > 95%

**Resource Metrics:**
- `system_cpu_percent` < 80%
- `system_memory_percent` < 80%
- `database_connection_pool_active` / `database_connection_pool_size`

---

## 3. Incident Response

### 3.1 Incident Severity Levels

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **P0 (Critical)** | Service down, data loss | 15 minutes | All tool executions failing |
| **P1 (High)** | Major functionality impaired | 1 hour | Success rate < 80% |
| **P2 (Medium)** | Minor functionality impaired | 4 hours | Single tool failing |
| **P3 (Low)** | Cosmetic issue, minor bug | 1 business day | Slow performance |

### 3.2 Incident Response Procedures

#### P0: Service Down

**Symptoms:**
- All tool executions failing
- Health check endpoint returning 500
- No metrics being collected

**Response Steps:**

1. **Acknowledge Incident** (0-5 minutes)
   ```bash
   # Check service status
   kubectl get pods -n production
   docker compose ps

   # Check logs
   kubectl logs -n production deployment/agentcore --tail=200
   ```

2. **Initial Diagnosis** (5-15 minutes)
   ```bash
   # Check dependencies
   kubectl get pods -n production | grep -E "(postgres|redis)"

   # Check database connectivity
   kubectl exec -n production deployment/agentcore -- \
     psql $DATABASE_URL -c "SELECT 1"

   # Check Redis connectivity
   kubectl exec -n production deployment/agentcore -- \
     redis-cli -u $REDIS_URL PING
   ```

3. **Mitigation** (15-30 minutes)
   ```bash
   # Restart service
   kubectl rollout restart deployment/agentcore -n production

   # Scale up if resource exhaustion
   kubectl scale deployment/agentcore --replicas=6 -n production

   # Rollback if recent deployment
   kubectl rollout undo deployment/agentcore -n production
   ```

4. **Recovery Verification** (30-45 minutes)
   ```bash
   # Verify health
   curl http://agentcore.prod/health

   # Test tool execution
   curl -X POST http://agentcore.prod/api/v1/jsonrpc \
     -H "Content-Type: application/json" \
     -d '{
       "jsonrpc": "2.0",
       "method": "tools.execute",
       "params": {
         "tool_id": "echo",
         "parameters": {"message": "test"}
       },
       "id": "test-1"
     }'

   # Check metrics
   curl http://agentcore.prod/metrics | grep tool_executions_total
   ```

5. **Post-Incident** (1-2 hours)
   - Write incident report
   - Update runbook with learnings
   - Schedule postmortem meeting

#### P1: High Error Rate

**Symptoms:**
- Tool success rate < 80%
- Multiple tools timing out
- Rate limit errors increasing

**Response Steps:**

1. **Identify Failing Tools**
   ```bash
   # Query Prometheus for failing tools
   curl -G http://prometheus:9090/api/v1/query \
     --data-urlencode 'query=topk(10, rate(agentcore_tool_errors_total[5m]))'
   ```

2. **Check External Dependencies**
   ```bash
   # Test Google Search API
   curl "https://www.googleapis.com/customsearch/v1?key=$GOOGLE_API_KEY&cx=$SEARCH_ENGINE_ID&q=test"

   # Check network connectivity
   kubectl exec -n production deployment/agentcore -- ping -c 3 8.8.8.8
   ```

3. **Adjust Rate Limits** (Temporary)
   ```bash
   # Reduce rate limits to prevent cascading failures
   kubectl set env deployment/agentcore -n production \
     TOOL_RATE_LIMIT_PER_MINUTE=30
   ```

4. **Scale Resources**
   ```bash
   # Increase replicas
   kubectl scale deployment/agentcore --replicas=10 -n production

   # Increase resource limits
   kubectl set resources deployment/agentcore -n production \
     --limits=cpu=8,memory=16Gi
   ```

### 3.3 Incident Communication Template

**Slack Notification:**
```
🚨 [P0] Tool Integration Service Down
Status: INVESTIGATING
Impact: All tool executions failing
Start Time: 2025-01-13 14:23 UTC
Affected Components: Tool Execution, Search Tools, Code Execution

Actions Taken:
- Service restarted (14:25 UTC)
- Database connectivity verified
- Redis cluster healthy

Next Steps:
- Monitoring recovery metrics
- Investigating root cause
- ETA for full recovery: 15 minutes

Incident Commander: @john-doe
```

---

## 4. Troubleshooting Guide

### 4.1 Common Issues

#### Issue: Tool Execution Timeout

**Symptoms:**
- `error_type="timeout"` in logs
- `TOOL_E2001` error code
- Execution time > 30 seconds

**Diagnosis:**
```bash
# Check slow tools
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.99, sum(rate(tool_execution_duration_seconds_bucket[5m])) by (tool_id, le))'

# Check logs for specific tool
kubectl logs -n production deployment/agentcore | grep "tool_execution_timeout" | grep "tool_id=google_search"
```

**Resolution:**
```python
# Increase timeout for specific tool
# src/agentcore/agent_runtime/tools/builtin/search_tools.py
GoogleSearchTool(
    timeout_seconds=60,  # Increase from 30 to 60
)
```

**Temporary Workaround:**
```bash
# Increase global timeout via environment variable
kubectl set env deployment/agentcore -n production \
  TOOL_EXECUTION_TIMEOUT=60
```

---

#### Issue: Rate Limit Exceeded

**Symptoms:**
- `error_type="rate_limit"` in logs
- `TOOL_E1401` error code
- 429 responses increasing

**Diagnosis:**
```bash
# Check rate limit metrics
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(rate_limit_exceeded_total[5m])'

# Check Redis connection
kubectl exec -n production deployment/agentcore -- \
  redis-cli -u $REDIS_URL --scan --pattern "agentcore:ratelimit:*" | head -20
```

**Resolution:**
```bash
# Option 1: Increase rate limit for tool
kubectl patch configmap agentcore-config -n production \
  --type merge -p '{"data":{"GOOGLE_SEARCH_RATE_LIMIT":"120"}}'

# Option 2: Reset rate limit for specific user
kubectl exec -n production deployment/agentcore -- \
  python -c "
from agentcore.agent_runtime.services.rate_limiter import get_rate_limiter
limiter = get_rate_limiter()
await limiter.reset('google_search', identifier='user123')
"
```

**Root Cause Prevention:**
- Review user behavior (potential abuse)
- Implement progressive rate limiting
- Add user education for rate limits

---

#### Issue: Redis Connection Failure

**Symptoms:**
- `dependency_unavailable` errors
- Rate limiting disabled
- Quota management disabled

**Diagnosis:**
```bash
# Check Redis health
kubectl get pods -n production | grep redis

# Test connection
kubectl exec -n production deployment/redis -- redis-cli PING

# Check connection pool
kubectl logs -n production deployment/agentcore | grep "redis_connection_failed"
```

**Resolution:**
```bash
# Restart Redis
kubectl rollout restart statefulset/redis -n production

# Scale Redis (if using cluster)
kubectl scale statefulset/redis --replicas=3 -n production

# Emergency: Disable rate limiting temporarily
kubectl set env deployment/agentcore -n production \
  TOOL_RATE_LIMIT_ENABLED=false \
  TOOL_QUOTA_ENABLED=false
```

**Monitoring:**
```bash
# Watch Redis recovery
watch -n 5 'kubectl exec -n production deployment/redis -- redis-cli INFO stats'
```

---

#### Issue: Database Connection Pool Exhausted

**Symptoms:**
- `database_error` in logs
- Slow query execution
- Connection timeouts

**Diagnosis:**
```bash
# Check connection pool usage
kubectl exec -n production deployment/agentcore -- \
  python -c "
from agentcore.a2a_protocol.database.connection import get_engine
engine = await get_engine()
print(f'Pool size: {engine.pool.size()}')
print(f'Checked out: {engine.pool.checkedout()}')
"

# Check PostgreSQL connections
kubectl exec -n production deployment/postgres -- \
  psql -U agentcore -c "SELECT count(*), state FROM pg_stat_activity WHERE datname='agentcore' GROUP BY state;"
```

**Resolution:**
```bash
# Increase pool size
kubectl set env deployment/agentcore -n production \
  DATABASE_POOL_SIZE=50 \
  DATABASE_MAX_OVERFLOW=50

# Kill idle connections (if stale)
kubectl exec -n production deployment/postgres -- \
  psql -U agentcore -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '5 minutes';"
```

---

### 4.2 Debugging Tools

**Log Analysis:**
```bash
# Search for errors
kubectl logs -n production deployment/agentcore | grep -E "(ERROR|CRITICAL)" | tail -50

# Filter by tool_id
kubectl logs -n production deployment/agentcore | grep "tool_id=google_search"

# Follow logs in real-time
kubectl logs -n production deployment/agentcore -f --tail=100
```

**Metrics Query:**
```bash
# Prometheus query for error rate
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum(rate(agentcore_tool_errors_total[5m])) by (error_type)'

# Grafana explore
# Navigate to: http://grafana:3000/explore
# Query: rate(agentcore_tool_executions_total[5m])
```

**Database Query:**
```sql
-- Check recent failed executions
SELECT tool_id, error, error_type, created_at
FROM tool_executions
WHERE success = false
  AND created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC
LIMIT 50;

-- Check slow executions
SELECT tool_id, AVG(execution_time_ms), COUNT(*)
FROM tool_executions
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY tool_id
HAVING AVG(execution_time_ms) > 1000
ORDER BY AVG(execution_time_ms) DESC;
```

---

## 5. Maintenance Procedures

### 5.1 Database Maintenance

**Weekly Maintenance:**
```bash
# Vacuum analyze (reclaim space, update stats)
kubectl exec -n production deployment/postgres -- \
  psql -U agentcore -c "VACUUM ANALYZE tool_executions;"

# Reindex (improve query performance)
kubectl exec -n production deployment/postgres -- \
  psql -U agentcore -c "REINDEX TABLE tool_executions;"

# Check table bloat
kubectl exec -n production deployment/postgres -- \
  psql -U agentcore -c "
SELECT
  schemaname || '.' || tablename as table,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
  n_dead_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;
"
```

**Monthly Archival:**
```bash
# Archive old tool executions (> 90 days)
kubectl exec -n production deployment/postgres -- \
  psql -U agentcore -c "
INSERT INTO tool_executions_archive
SELECT * FROM tool_executions
WHERE created_at < NOW() - INTERVAL '90 days';

DELETE FROM tool_executions
WHERE created_at < NOW() - INTERVAL '90 days';
"
```

### 5.2 Log Rotation

**Configure Log Rotation:**
```bash
# /etc/logrotate.d/agentcore
/var/log/agentcore/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 agentcore agentcore
    sharedscripts
    postrotate
        kubectl rollout restart deployment/agentcore -n production
    endscript
}
```

### 5.3 Dependency Updates

**Monthly Updates:**
```bash
# Update Docker base images
docker pull python:3.12-slim
docker build -t agentcore:latest .

# Update Python dependencies
uv update

# Update Kubernetes manifests
kubectl apply -f k8s/production/deployment.yaml
```

**Security Patches:**
```bash
# Scan for vulnerabilities
trivy image agentcore:latest

# Apply security patches immediately (weekly)
apt-get update && apt-get upgrade -y
```

---

## 6. Backup and Recovery

### 6.1 Database Backup

**Daily Backup:**
```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="agentcore_backup_${BACKUP_DATE}.sql.gz"

kubectl exec -n production deployment/postgres -- \
  pg_dump -U agentcore agentcore | gzip > /backups/${BACKUP_FILE}

# Upload to S3
aws s3 cp /backups/${BACKUP_FILE} s3://agentcore-backups/database/

# Verify backup
gunzip -c /backups/${BACKUP_FILE} | head -100

echo "Backup completed: ${BACKUP_FILE}"
```

**Restore Procedure:**
```bash
#!/bin/bash
# scripts/restore_database.sh

BACKUP_FILE=$1

# Download from S3
aws s3 cp s3://agentcore-backups/database/${BACKUP_FILE} /tmp/

# Restore
gunzip -c /tmp/${BACKUP_FILE} | \
  kubectl exec -i -n production deployment/postgres -- \
  psql -U agentcore agentcore

echo "Database restored from ${BACKUP_FILE}"
```

### 6.2 Redis Backup

**Daily Snapshot:**
```bash
# Trigger Redis save
kubectl exec -n production deployment/redis -- redis-cli BGSAVE

# Copy RDB file
kubectl cp production/redis-0:/data/dump.rdb /backups/redis_$(date +%Y%m%d).rdb

# Upload to S3
aws s3 cp /backups/redis_$(date +%Y%m%d).rdb s3://agentcore-backups/redis/
```

---

## 7. Performance Tuning

### 7.1 Production Tuning Checklist

**Before Launch:**
- [ ] Database connection pool: 20-50 connections
- [ ] Redis connection pool: 50-100 connections
- [ ] HTTP connection pool: 100 connections
- [ ] Tool execution timeout: 30 seconds
- [ ] Rate limits configured per tool
- [ ] Quotas configured per user
- [ ] Horizontal pod autoscaling enabled
- [ ] Resource limits set appropriately
- [ ] Monitoring dashboards configured
- [ ] Alerts configured and tested

**Optimization Settings:**
```python
# config.py (Production)
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 20
REDIS_POOL_SIZE = 50
HTTP_CONNECTION_POOL_SIZE = 100
TOOL_EXECUTION_TIMEOUT = 30
ENABLE_DATABASE_LOGGING = True  # Async, minimal overhead
ENABLE_TRACING = True  # OpenTelemetry with sampling
TRACING_SAMPLE_RATE = 0.1  # 10% sampling
```

---

## 8. Security Hardening

### 8.1 Production Security Checklist

**Before Launch:**
- [ ] JWT authentication enabled with RS256
- [ ] API keys stored in Vault (not environment variables)
- [ ] Docker sandbox enabled with seccomp/AppArmor
- [ ] Network policies configured (deny by default)
- [ ] TLS/SSL enabled for all external connections
- [ ] Rate limiting enabled
- [ ] Quota management enabled
- [ ] Audit logging enabled
- [ ] Security headers configured (HSTS, CSP, etc.)
- [ ] Vulnerability scanning automated in CI/CD

**Security Configuration:**
```yaml
# k8s/production/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agentcore-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: agentcore
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8001
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

---

## 9. Disaster Recovery

### 9.1 Recovery Time Objectives

| Component | RTO (Recovery Time Objective) | RPO (Recovery Point Objective) |
|-----------|-------------------------------|--------------------------------|
| Application | 15 minutes | 5 minutes |
| Database | 30 minutes | 24 hours (daily backups) |
| Redis | 15 minutes | 1 hour (hourly snapshots) |

### 9.2 Disaster Recovery Procedure

**Full System Recovery:**

1. **Restore Database** (0-30 minutes)
   ```bash
   # Restore latest backup
   ./scripts/restore_database.sh agentcore_backup_20250113.sql.gz
   ```

2. **Restore Redis** (30-45 minutes)
   ```bash
   # Copy RDB file
   kubectl cp /backups/redis_20250113.rdb production/redis-0:/data/dump.rdb
   kubectl exec -n production deployment/redis -- redis-cli SHUTDOWN
   kubectl rollout restart statefulset/redis -n production
   ```

3. **Redeploy Application** (45-60 minutes)
   ```bash
   # Deploy from last known good version
   kubectl apply -f k8s/production/deployment.yaml
   kubectl rollout status deployment/agentcore -n production
   ```

4. **Verify Recovery** (60-75 minutes)
   ```bash
   # Health check
   curl http://agentcore.prod/health

   # Test tool execution
   curl -X POST http://agentcore.prod/api/v1/jsonrpc -d '{"method":"tools.execute",...}'

   # Verify metrics
   curl http://agentcore.prod/metrics
   ```

---

## 10. Contact Information

**On-Call Rotation:**
- Primary: ops-primary@agentcore.io
- Secondary: ops-secondary@agentcore.io
- Escalation: engineering-lead@agentcore.io

**Communication Channels:**
- Slack: #agentcore-incidents
- PagerDuty: agentcore-production
- Zoom War Room: https://zoom.us/j/agentcore-incidents

**Key Personnel:**
- Engineering Lead: @john-doe
- DevOps Lead: @jane-smith
- Database Admin: @bob-wilson

---

## Appendix A: Quick Reference

**Health Check:**
```bash
curl http://localhost:8001/health
```

**Metrics Endpoint:**
```bash
curl http://localhost:8001/metrics
```

**Restart Service:**
```bash
kubectl rollout restart deployment/agentcore -n production
```

**View Logs:**
```bash
kubectl logs -n production deployment/agentcore --tail=100 -f
```

**Scale Service:**
```bash
kubectl scale deployment/agentcore --replicas=10 -n production
```

**Check Resource Usage:**
```bash
kubectl top pods -n production
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-01-13 | Initial production runbook | AgentCore Team |

---

**Classification:** INTERNAL USE ONLY
**Distribution:** Engineering Team, DevOps Team, On-Call Engineers
