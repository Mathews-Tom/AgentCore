# Tool Integration Framework - Production Runbook

**Version:** 1.0
**Last Updated:** 2025-11-13
**Owner:** AgentCore Operations Team
**On-Call:** oncall@agentcore.dev

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [System Architecture](#system-architecture)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Troubleshooting](#troubleshooting)
6. [Incident Response](#incident-response)
7. [Maintenance](#maintenance)
8. [Scaling](#scaling)
9. [Disaster Recovery](#disaster-recovery)
10. [Contacts](#contacts)

## Quick Reference

### Critical Metrics Dashboard

**Grafana URL:** https://monitoring.agentcore.dev/d/tool-metrics
**Prometheus URL:** https://prometheus.agentcore.dev

**Health Check:**
```bash
curl https://api.agentcore.dev/api/v1/health
```

**Key SLIs (Service Level Indicators):**
- Success Rate: >99% (P0)
- P95 Latency: <100ms lightweight, <1s medium (P0)
- Availability: >99.9% (P0)
- Error Rate: <0.1% (P1)

### Emergency Contacts

| Role | Contact | Phone | PagerDuty |
|------|---------|-------|-----------|
| On-Call Engineer | oncall@agentcore.dev | +1-XXX-XXX-XXXX | Primary |
| Engineering Lead | lead@agentcore.dev | +1-XXX-XXX-XXXX | Secondary |
| Security Team | security@agentcore.dev | +1-XXX-XXX-XXXX | Escalation |
| SRE Team | sre@agentcore.dev | +1-XXX-XXX-XXXX | Escalation |

### Quick Commands

```bash
# Check service health
kubectl get pods -n agentcore -l app=tool-framework

# View recent logs
kubectl logs -n agentcore -l app=tool-framework --tail=100 -f

# Check resource usage
kubectl top pods -n agentcore -l app=tool-framework

# Scale deployment
kubectl scale deployment tool-framework -n agentcore --replicas=5

# View metrics
curl http://localhost:8001/metrics

# Emergency rollback
kubectl rollout undo deployment/tool-framework -n agentcore
```

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer (ALB)                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼──────┐                       ┌───────▼──────┐
│   AgentCore  │                       │   AgentCore  │
│   Pod 1      │  ... (auto-scaled)    │   Pod N      │
└───────┬──────┘                       └───────┬──────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼──────┐                       ┌───────▼──────┐
│  PostgreSQL  │                       │    Redis     │
│   (Primary)  │                       │   Cluster    │
└──────────────┘                       └──────────────┘
```

### Technology Stack

- **Application:** Python 3.12+, FastAPI, Asyncio
- **Database:** PostgreSQL 15+ (async with asyncpg)
- **Cache:** Redis 7+ (cluster mode for production)
- **Metrics:** Prometheus, Grafana
- **Tracing:** OpenTelemetry, Jaeger
- **Orchestration:** Kubernetes 1.28+
- **Load Balancer:** AWS ALB / NGINX

### Resource Requirements

**Per Pod:**
- CPU: 2 cores (limit: 4 cores)
- Memory: 4GB (limit: 8GB)
- Storage: 10GB ephemeral

**Database (PostgreSQL):**
- CPU: 8 cores
- Memory: 16GB
- Storage: 500GB SSD

**Cache (Redis Cluster):**
- Nodes: 6 (3 primary + 3 replica)
- CPU: 2 cores per node
- Memory: 4GB per node
- Storage: 10GB per node

## Deployment

### Pre-Deployment Checklist

- [ ] All tests passing (unit, integration, load)
- [ ] Security audit completed
- [ ] Performance benchmarks validated
- [ ] Secrets rotated from defaults
- [ ] Environment variables configured
- [ ] Database migrations prepared
- [ ] Monitoring dashboards updated
- [ ] On-call team notified
- [ ] Rollback plan documented

### Deployment Steps

#### 1. Pre-Deployment

```bash
# Create deployment directory
mkdir -p ~/deployments/tool-framework-$(date +%Y%m%d-%H%M%S)
cd ~/deployments/tool-framework-*

# Clone repository
git clone https://github.com/agentcore/agentcore.git
cd agentcore
git checkout v1.0.0  # Replace with target version

# Verify tests
uv run pytest
```

#### 2. Database Migration

```bash
# Backup database first
kubectl exec -it postgres-0 -n agentcore -- \
    pg_dump -U agentcore agentcore > backup-$(date +%Y%m%d-%H%M%S).sql

# Run migrations
kubectl exec -it agentcore-pod -n agentcore -- \
    uv run alembic upgrade head

# Verify migration
kubectl exec -it agentcore-pod -n agentcore -- \
    uv run alembic current
```

#### 3. Deploy Application

```bash
# Update container image
kubectl set image deployment/tool-framework \
    tool-framework=agentcore/tool-framework:v1.0.0 \
    -n agentcore

# Watch rollout
kubectl rollout status deployment/tool-framework -n agentcore

# Verify pods
kubectl get pods -n agentcore -l app=tool-framework
```

#### 4. Post-Deployment Validation

```bash
# Health check
curl https://api.agentcore.dev/api/v1/health

# Test tool execution
curl -X POST https://api.agentcore.dev/api/v1/jsonrpc \
    -H "Content-Type: application/json" \
    -d '{
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "tools.execute",
        "params": {
            "tool_id": "echo",
            "parameters": {"message": "deployment test"}
        }
    }'

# Check metrics
curl https://api.agentcore.dev/metrics | grep agentcore_tool

# View logs
kubectl logs -n agentcore -l app=tool-framework --tail=100
```

#### 5. Smoke Tests

```bash
# Run smoke test suite
uv run pytest tests/smoke/

# Run load test (light)
uv run locust -f tests/load/tool_integration_load_test.py \
    --host https://api.agentcore.dev \
    --users 100 \
    --spawn-rate 10 \
    --run-time 2m \
    --headless
```

### Rollback Procedure

```bash
# Emergency rollback to previous version
kubectl rollout undo deployment/tool-framework -n agentcore

# Verify rollback
kubectl rollout status deployment/tool-framework -n agentcore

# Check health
curl https://api.agentcore.dev/api/v1/health

# If database migration was applied, rollback
kubectl exec -it agentcore-pod -n agentcore -- \
    uv run alembic downgrade -1
```

## Monitoring

### Key Metrics

#### Tool Execution Metrics

**Success Rate:**
```promql
sum(rate(agentcore_tool_executions_total{status="success"}[5m])) /
sum(rate(agentcore_tool_executions_total[5m])) * 100
```

**P95 Latency:**
```promql
histogram_quantile(0.95,
    sum(rate(agentcore_tool_execution_seconds_bucket[5m])) by (le, tool_id)
)
```

**Error Rate:**
```promql
sum(rate(agentcore_tool_errors_total[5m])) /
sum(rate(agentcore_tool_executions_total[5m])) * 100
```

**Throughput:**
```promql
sum(rate(agentcore_tool_executions_total[5m]))
```

#### Rate Limiting Metrics

**Rate Limit Hit Rate:**
```promql
sum(rate(agentcore_rate_limit_hits_total[5m]))
```

**Rate Limit Check Latency:**
```promql
histogram_quantile(0.95,
    sum(rate(agentcore_rate_limit_check_seconds_bucket[5m])) by (le)
)
```

#### Quota Management Metrics

**Quota Exceeded Rate:**
```promql
sum(rate(agentcore_quota_exceeded_total[5m]))
```

**Quota Check Latency:**
```promql
histogram_quantile(0.95,
    sum(rate(agentcore_quota_check_seconds_bucket[5m])) by (le)
)
```

#### System Metrics

**CPU Usage:**
```promql
rate(container_cpu_usage_seconds_total{pod=~"tool-framework-.*"}[5m]) * 100
```

**Memory Usage:**
```promql
container_memory_usage_bytes{pod=~"tool-framework-.*"} / 1024 / 1024 / 1024
```

**Pod Count:**
```promql
count(kube_pod_info{pod=~"tool-framework-.*"})
```

### Alerting Rules

**Critical Alerts (P0):**

```yaml
- alert: ToolFrameworkDown
  expr: up{job="tool-framework"} == 0
  for: 1m
  severity: critical
  annotations:
    summary: "Tool Framework is down"
    description: "No healthy instances of Tool Framework detected"

- alert: HighErrorRate
  expr: |
    sum(rate(agentcore_tool_errors_total[5m])) /
    sum(rate(agentcore_tool_executions_total[5m])) * 100 > 1
  for: 5m
  severity: critical
  annotations:
    summary: "High tool execution error rate"
    description: "Error rate is {{ $value }}% (threshold: 1%)"

- alert: HighLatency
  expr: |
    histogram_quantile(0.95,
        sum(rate(agentcore_tool_execution_seconds_bucket[5m])) by (le)
    ) > 0.5
  for: 5m
  severity: critical
  annotations:
    summary: "High tool execution latency"
    description: "P95 latency is {{ $value }}s (threshold: 0.5s)"
```

**Warning Alerts (P1):**

```yaml
- alert: HighRateLimitHitRate
  expr: sum(rate(agentcore_rate_limit_hits_total[5m])) > 100
  for: 10m
  severity: warning
  annotations:
    summary: "High rate limit hit rate"
    description: "Rate limiting is being triggered frequently"

- alert: HighQuotaExceededRate
  expr: sum(rate(agentcore_quota_exceeded_total[5m])) > 50
  for: 10m
  severity: warning
  annotations:
    summary: "High quota exceeded rate"
    description: "Many tools are hitting quota limits"

- alert: HighMemoryUsage
  expr: |
    container_memory_usage_bytes{pod=~"tool-framework-.*"} /
    container_spec_memory_limit_bytes{pod=~"tool-framework-.*"} * 100 > 80
  for: 5m
  severity: warning
  annotations:
    summary: "High memory usage"
    description: "Memory usage is {{ $value }}% (threshold: 80%)"
```

### Logging

**Log Levels:**
- **DEBUG:** Detailed diagnostic information
- **INFO:** General informational messages
- **WARNING:** Warning messages (degraded performance)
- **ERROR:** Error messages (operation failed)
- **CRITICAL:** Critical messages (system failure)

**Log Aggregation:**
- Logs collected via Fluentd/Fluent Bit
- Centralized in Elasticsearch/CloudWatch
- Searchable via Kibana/CloudWatch Insights

**Useful Log Queries:**

```bash
# Errors in last hour
kubectl logs -n agentcore -l app=tool-framework --since=1h | grep ERROR

# Tool execution failures
kubectl logs -n agentcore -l app=tool-framework | grep "Tool execution failed"

# Rate limit violations
kubectl logs -n agentcore -l app=tool-framework | grep "Rate limit exceeded"

# Quota exceeded
kubectl logs -n agentcore -l app=tool-framework | grep "Quota exceeded"
```

## Troubleshooting

### Common Issues

#### High Error Rate

**Symptoms:**
- Error rate >1%
- Failed tool executions
- User complaints

**Diagnosis:**
```bash
# Check error distribution
kubectl logs -n agentcore -l app=tool-framework | grep ERROR | \
    awk '{print $NF}' | sort | uniq -c | sort -rn

# Check Grafana error distribution panel
# Dashboard: Tool Integration Metrics > Error Distribution by Type
```

**Common Causes:**
1. **Validation Errors:** Invalid parameters
   - Fix: Review parameter schemas, update documentation
2. **Timeout Errors:** Tool taking too long
   - Fix: Increase timeout, optimize tool implementation
3. **Rate Limit Errors:** Too many requests
   - Fix: Increase rate limits, implement backoff
4. **Quota Exceeded:** Daily/monthly quota hit
   - Fix: Increase quotas, notify users

**Resolution:**
```bash
# Restart pods if necessary
kubectl rollout restart deployment/tool-framework -n agentcore

# Increase rate limits (if configured via ConfigMap)
kubectl edit configmap tool-framework-config -n agentcore
# Update RATE_LIMIT_DEFAULT and restart
```

#### High Latency

**Symptoms:**
- P95 latency >100ms for lightweight tools
- Slow response times
- User complaints

**Diagnosis:**
```bash
# Check latency distribution
curl http://localhost:8001/metrics | grep tool_execution_seconds

# Check database connection pool
kubectl exec -it postgres-0 -n agentcore -- \
    psql -U agentcore -c "SELECT * FROM pg_stat_activity;"

# Check Redis latency
kubectl exec -it redis-0 -n agentcore -- redis-cli --latency

# Review Grafana latency percentiles panel
# Dashboard: Tool Integration Metrics > Tool Execution Latency (Percentiles)
```

**Common Causes:**
1. **Database Slow Queries:** Inefficient queries
   - Fix: Add indexes, optimize queries
2. **Redis Latency:** Network issues or overload
   - Fix: Check network, scale Redis cluster
3. **Resource Constraints:** CPU/memory limits
   - Fix: Increase resource limits, scale pods
4. **External Tool Latency:** External API slow
   - Fix: Increase timeout, implement caching

**Resolution:**
```bash
# Scale up pods
kubectl scale deployment tool-framework -n agentcore --replicas=10

# Increase resource limits
kubectl edit deployment tool-framework -n agentcore
# Update resources.limits.cpu and resources.limits.memory
```

#### Service Unavailable

**Symptoms:**
- Health check failing
- 503 errors
- No pods running

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n agentcore -l app=tool-framework

# Check pod events
kubectl describe pod -n agentcore -l app=tool-framework

# Check logs
kubectl logs -n agentcore -l app=tool-framework --tail=100
```

**Common Causes:**
1. **CrashLoopBackOff:** Application crashing on startup
   - Fix: Check logs, fix startup issues
2. **ImagePullBackOff:** Cannot pull container image
   - Fix: Verify image exists, check registry credentials
3. **OOMKilled:** Out of memory
   - Fix: Increase memory limits
4. **Pending:** Insufficient cluster resources
   - Fix: Add nodes, reduce resource requests

**Resolution:**
```bash
# Emergency rollback
kubectl rollout undo deployment/tool-framework -n agentcore

# Force restart
kubectl rollout restart deployment/tool-framework -n agentcore

# Check cluster resources
kubectl describe nodes | grep -A 5 "Allocated resources"
```

#### Database Connection Issues

**Symptoms:**
- Connection pool exhausted errors
- Database timeouts
- Failed queries

**Diagnosis:**
```bash
# Check database connections
kubectl exec -it postgres-0 -n agentcore -- \
    psql -U agentcore -c "
        SELECT count(*) as connections,
               state
        FROM pg_stat_activity
        GROUP BY state;
    "

# Check connection pool metrics
curl http://localhost:8001/metrics | grep db_pool
```

**Resolution:**
```bash
# Increase connection pool size
kubectl edit configmap tool-framework-config -n agentcore
# Update POSTGRES_POOL_SIZE and POSTGRES_MAX_OVERFLOW

# Restart application
kubectl rollout restart deployment/tool-framework -n agentcore

# If database is overloaded, scale read replicas
kubectl scale statefulset postgres-replica -n agentcore --replicas=2
```

#### Redis Connection Issues

**Symptoms:**
- Rate limiting failures
- Quota management errors
- Redis connection errors

**Diagnosis:**
```bash
# Check Redis cluster health
kubectl exec -it redis-0 -n agentcore -- redis-cli cluster info

# Check Redis connections
kubectl exec -it redis-0 -n agentcore -- redis-cli info clients
```

**Resolution:**
```bash
# Increase Redis max connections
kubectl exec -it redis-0 -n agentcore -- \
    redis-cli CONFIG SET maxclients 20000

# Restart Redis cluster
kubectl rollout restart statefulset/redis -n agentcore

# Scale Redis cluster (add more nodes)
kubectl scale statefulset redis -n agentcore --replicas=9
```

## Incident Response

### Incident Severity Levels

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| P0 - Critical | Service down or severe degradation | 15 minutes | All pods down, 100% error rate |
| P1 - High | Partial service degradation | 1 hour | High error rate (>1%), high latency |
| P2 - Medium | Minor service degradation | 4 hours | Elevated error rate (<1%) |
| P3 - Low | No service impact | Next business day | Warning alerts, low priority bugs |

### Incident Response Steps

#### 1. Detection & Alert

- Alert triggers via PagerDuty
- On-call engineer acknowledges within SLA
- Create incident ticket (Jira/ServiceNow)

#### 2. Assessment

```bash
# Quick assessment checklist
- [ ] Service health: curl https://api.agentcore.dev/api/v1/health
- [ ] Pod status: kubectl get pods -n agentcore
- [ ] Recent logs: kubectl logs -n agentcore -l app=tool-framework --tail=100
- [ ] Metrics: Check Grafana dashboard
- [ ] Recent changes: Check deployment history
```

#### 3. Communication

- Update incident ticket with assessment
- Notify stakeholders (Slack #incidents channel)
- Update status page (status.agentcore.dev)

#### 4. Mitigation

**P0 Incident:**
```bash
# Immediate actions
1. kubectl rollout undo deployment/tool-framework -n agentcore
2. kubectl scale deployment tool-framework -n agentcore --replicas=10
3. Check database and Redis health
4. Notify engineering lead
```

**P1 Incident:**
```bash
# Investigation and fix
1. Identify root cause from logs and metrics
2. Apply targeted fix (increase limits, fix bug, etc.)
3. Deploy fix with testing
4. Monitor for 30 minutes
```

#### 5. Resolution

- Verify service health restored
- Update incident ticket with resolution
- Post-incident review scheduled within 48 hours

#### 6. Post-Incident Review

**Template:**
```markdown
# Post-Incident Review - [Incident ID]

## Incident Summary
- **Date:** YYYY-MM-DD
- **Duration:** X hours
- **Severity:** PX
- **Impact:** Description of user impact

## Timeline
- HH:MM - Alert triggered
- HH:MM - On-call acknowledged
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Incident resolved

## Root Cause
- What caused the incident
- Why it wasn't caught earlier

## Resolution
- Actions taken to resolve
- Why it worked

## Action Items
- [ ] Improve monitoring (assigned to: X)
- [ ] Update runbook (assigned to: Y)
- [ ] Fix underlying issue (assigned to: Z)

## Lessons Learned
- What went well
- What could be improved
```

## Maintenance

### Routine Maintenance Tasks

#### Daily

- [ ] Review Grafana dashboards for anomalies
- [ ] Check error logs for new issues
- [ ] Verify backup completion

#### Weekly

- [ ] Review performance metrics trends
- [ ] Check for dependency updates
- [ ] Review rate limit and quota configurations
- [ ] Check disk usage (database, logs)

#### Monthly

- [ ] Security audit review
- [ ] Load testing validation
- [ ] Performance optimization review
- [ ] Update documentation
- [ ] Database index optimization
- [ ] Log retention cleanup

#### Quarterly

- [ ] Disaster recovery drill
- [ ] Capacity planning review
- [ ] Security penetration testing
- [ ] Dependency major version updates

### Database Maintenance

**Vacuum and Analyze:**
```sql
-- Run weekly during low-traffic hours
VACUUM ANALYZE;

-- Check bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Index Maintenance:**
```sql
-- Rebuild indexes quarterly
REINDEX DATABASE agentcore;

-- Find unused indexes
SELECT schemaname, tablename, indexname
FROM pg_stat_user_indexes
WHERE idx_scan = 0 AND schemaname = 'public';
```

**Backup Verification:**
```bash
# Test restore monthly
pg_restore --verbose backup-latest.sql | psql agentcore_test
```

### Redis Maintenance

**Memory Optimization:**
```bash
# Check memory usage
redis-cli info memory

# Analyze key space
redis-cli --bigkeys

# Cleanup expired keys
redis-cli --scan --pattern "*" | xargs redis-cli DEL
```

**Cluster Rebalancing:**
```bash
# Rebalance cluster quarterly
redis-cli --cluster rebalance redis-0:6379 \
    --cluster-use-empty-masters
```

## Scaling

### Horizontal Scaling (Add Pods)

```bash
# Manual scaling
kubectl scale deployment tool-framework -n agentcore --replicas=10

# Auto-scaling (HPA)
kubectl autoscale deployment tool-framework -n agentcore \
    --cpu-percent=70 \
    --min=5 \
    --max=50
```

### Vertical Scaling (Increase Resources)

```bash
# Update deployment with more resources
kubectl edit deployment tool-framework -n agentcore

# Update:
resources:
  requests:
    cpu: 4
    memory: 8Gi
  limits:
    cpu: 8
    memory: 16Gi
```

### Database Scaling

**Read Replicas:**
```bash
# Add read replica
kubectl scale statefulset postgres-replica -n agentcore --replicas=3
```

**Vertical Scaling:**
```bash
# Increase database resources
kubectl edit statefulset postgres -n agentcore

# Update:
resources:
  requests:
    cpu: 16
    memory: 32Gi
  limits:
    cpu: 32
    memory: 64Gi
```

### Redis Scaling

**Add Nodes:**
```bash
# Add 3 more nodes (1 primary + 2 replicas)
kubectl scale statefulset redis -n agentcore --replicas=9

# Rebalance cluster
kubectl exec -it redis-0 -n agentcore -- \
    redis-cli --cluster rebalance redis-0:6379
```

## Disaster Recovery

### Backup Strategy

**Database Backups:**
- Automated daily backups via pg_dump
- Retention: 30 days
- Location: S3 bucket (encrypted at rest)
- Verification: Monthly restore test

**Redis Backups:**
- RDB snapshots every 6 hours
- AOF for durability (if enabled)
- Location: S3 bucket

**Configuration Backups:**
- Kubernetes manifests in Git
- ConfigMaps and Secrets backed up daily
- Environment variables documented

### Recovery Procedures

**Database Recovery:**
```bash
# Restore from backup
kubectl cp backup-YYYYMMDD.sql postgres-0:/tmp/
kubectl exec -it postgres-0 -n agentcore -- \
    psql -U agentcore agentcore < /tmp/backup-YYYYMMDD.sql

# Verify restoration
kubectl exec -it postgres-0 -n agentcore -- \
    psql -U agentcore -c "SELECT COUNT(*) FROM tool_executions;"
```

**Full System Recovery:**
```bash
# 1. Deploy infrastructure (Kubernetes, database, Redis)
# 2. Restore database from backup
# 3. Deploy application
kubectl apply -f k8s/
# 4. Verify health
kubectl get pods -n agentcore
curl https://api.agentcore.dev/api/v1/health
# 5. Run smoke tests
uv run pytest tests/smoke/
```

## Contacts

### Support Channels

- **Slack:** #agentcore-ops, #incidents
- **Email:** support@agentcore.dev
- **PagerDuty:** https://agentcore.pagerduty.com
- **Status Page:** https://status.agentcore.dev

### Escalation Matrix

| Level | Contact | Response Time |
|-------|---------|---------------|
| L1 - On-Call | oncall@agentcore.dev | 15 minutes |
| L2 - Engineering Lead | lead@agentcore.dev | 30 minutes |
| L3 - SRE Team | sre@agentcore.dev | 1 hour |
| L4 - CTO | cto@agentcore.dev | 2 hours |

---

**Document Version:** 1.0
**Last Review:** 2025-11-13
**Next Review:** 2025-12-13
**Owner:** AgentCore Operations Team
