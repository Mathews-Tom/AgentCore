# ACE Operations Guide

**Version:** 1.0
**Component:** ACE (Agent Context Engineering)
**Status:** Production Ready
**Last Updated:** 2025-11-09

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Configuration](#configuration)
5. [Operations](#operations)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)
9. [Scaling](#scaling)
10. [Disaster Recovery](#disaster-recovery)
11. [Performance Tuning](#performance-tuning)
12. [Security](#security)

---

## Overview

### What is ACE?

ACE (Agent Context Engineering) is a Meta-Thinker component that provides real-time performance monitoring, error tracking, and strategic intervention for agentic AI systems. Based on the COMPASS framework, ACE enables long-horizon task planning with continuous oversight and dynamic adaptation.

### Production Readiness Status

✅ **Production Ready** - All validation and load tests passed:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| COMPASS Long-Horizon Accuracy | +20% | +22% | ✅ |
| Critical Error Recall | 90%+ | 95% | ✅ |
| Intervention Precision | 85%+ | 88% | ✅ |
| Monthly Cost (100 agents) | <$150 | $132 | ✅ |
| System Overhead | <5% | 3.2% | ✅ |
| Concurrent Agents | 100 | 100 (0 errors) | ✅ |
| Task Throughput | High | 10.5 tasks/sec | ✅ |
| Intervention Latency (p95) | <200ms | 50ms | ✅ |

### Key Features

- **Real-time Performance Monitoring**: Stage-specific metrics tracking
- **Error Accumulation Detection**: Pattern recognition with 95% recall
- **Strategic Interventions**: Automated with 88% precision
- **Baseline Tracking**: Performance deviation monitoring
- **Context Management**: Staleness detection and refresh
- **COMPASS Compliance**: All 5 benchmark targets met
- **Production Scale**: Handles 100+ concurrent agents

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        AgentCore API                         │
│                    (FastAPI + A2A Protocol)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─ /metrics (Prometheus)
                              │
┌─────────────────────────────┴─────────────────────────────────┐
│                        ACE Components                          │
├───────────────────┬───────────────────┬──────────────────────┤
│ PerformanceMonitor│ ErrorAccumulator  │ InterventionEngine   │
│ - Metrics batching│ - Pattern detect  │ - Strategy selection │
│ - Baseline track  │ - Severity track  │ - Execution          │
└───────────────────┴───────────────────┴──────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────────┐
│                        Database Layer                          │
├───────────────────┬───────────────────┬──────────────────────┤
│ PostgreSQL        │ TimescaleDB       │ Redis Cache          │
│ - Metrics storage │ - Time-series     │ - Fast lookups       │
│ - Agent state     │ - Efficient query │ - 85%+ hit rate      │
└───────────────────┴───────────────────┴──────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────────┐
│                    Monitoring & Alerting                       │
├───────────────────┬───────────────────┬──────────────────────┤
│ Prometheus        │ AlertManager      │ Grafana              │
│ - Metrics collect │ - Alert routing   │ - Visualization      │
│ - Alert rules     │ - Notifications   │ - Dashboards         │
└───────────────────┴───────────────────┴──────────────────────┘
```

### Data Flow

1. **Metrics Recording**: Agents report metrics via PerformanceMonitor
2. **Batching**: Metrics buffered (batch_size=100, timeout=1.0s)
3. **Storage**: Batched writes to PostgreSQL/TimescaleDB
4. **Caching**: Baseline and context data cached in Redis
5. **Monitoring**: Prometheus scrapes /metrics endpoint (15s interval)
6. **Alerting**: AlertManager routes alerts based on severity
7. **Visualization**: Grafana displays real-time dashboards

### Database Schema

**Key Tables:**
- `ace_performance_metrics`: Stage-specific performance data
- `ace_baselines`: Historical baseline tracking
- `ace_errors`: Error tracking with severity and patterns
- `ace_interventions`: Intervention history and effectiveness
- `ace_agent_context`: Agent context and staleness scores

---

## Deployment

### Prerequisites

**System Requirements:**
- CPU: 4+ cores (recommended: 8 cores)
- RAM: 8GB minimum (recommended: 16GB)
- Disk: 50GB+ SSD storage
- Network: Low latency (<10ms internal)

**Software Requirements:**
- Docker 24.0+ and Docker Compose 2.20+
- PostgreSQL 14+ with TimescaleDB extension
- Redis 7.0+
- Python 3.12+

### Quick Start (Docker Compose)

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-org/agentcore.git
   cd agentcore
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Services**:
   ```bash
   # Start AgentCore with dependencies
   docker compose -f docker-compose.dev.yml up -d

   # Start monitoring stack
   docker compose -f docker-compose.monitoring.yml up -d
   ```

4. **Verify Deployment**:
   ```bash
   # Check service health
   curl http://localhost:8001/health

   # Check metrics endpoint
   curl http://localhost:8001/metrics | grep ace_

   # Access Grafana dashboard
   open http://localhost:3000  # admin/admin
   ```

### Production Deployment (Kubernetes)

1. **Apply Kubernetes Manifests**:
   ```bash
   kubectl apply -f k8s/namespace.yml
   kubectl apply -f k8s/configmap.yml
   kubectl apply -f k8s/secret.yml
   kubectl apply -f k8s/deployment.yml
   kubectl apply -f k8s/service.yml
   kubectl apply -f k8s/hpa.yml
   ```

2. **Deploy Monitoring**:
   ```bash
   kubectl apply -f k8s/monitoring/prometheus.yml
   kubectl apply -f k8s/monitoring/grafana.yml
   kubectl apply -f k8s/monitoring/servicemonitor.yml
   ```

3. **Verify Deployment**:
   ```bash
   kubectl get pods -n agentcore
   kubectl logs -f deployment/agentcore -n agentcore
   kubectl port-forward svc/grafana 3000:3000 -n monitoring
   ```

### Database Setup

1. **Enable TimescaleDB Extension**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   ```

2. **Create Hypertables**:
   ```sql
   SELECT create_hypertable('ace_performance_metrics', 'created_at',
     chunk_time_interval => INTERVAL '1 day');
   ```

3. **Run Migrations**:
   ```bash
   uv run alembic upgrade head
   ```

4. **Verify Schema**:
   ```bash
   uv run alembic current
   ```

---

## Configuration

### Environment Variables

**Database Configuration:**
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/agentcore
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure-password
POSTGRES_DB=agentcore
```

**Redis Configuration:**
```bash
REDIS_URL=redis://localhost:6379/0
REDIS_CLUSTER_URLS=redis://node1:6379,redis://node2:6379
REDIS_PASSWORD=secure-password
```

**ACE Configuration:**
```bash
# Performance Monitor
ACE_BATCH_SIZE=100
ACE_BATCH_TIMEOUT=1.0
ACE_ENABLE_CACHING=true
ACE_CACHE_TTL=3600

# Error Tracking
ACE_ERROR_THRESHOLD=3
ACE_ERROR_WINDOW=300

# Interventions
ACE_COOLDOWN_SECONDS=60
ACE_MAX_INTERVENTIONS_PER_TASK=5
ACE_INTERVENTION_ENABLED=true

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=8001
```

**Monitoring Configuration:**
```bash
# Prometheus
PROMETHEUS_SCRAPE_INTERVAL=15s
PROMETHEUS_RETENTION_TIME=30d

# AlertManager
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=alerts@example.com
SMTP_PASSWORD=app-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
PAGERDUTY_SERVICE_KEY=your-key

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure-password
```

### Connection Pool Tuning

**PostgreSQL Connection Pool:**
```bash
# In .env
POSTGRES_MIN_CONNECTIONS=10
POSTGRES_MAX_CONNECTIONS=50
POSTGRES_MAX_IDLE_TIME=300
POSTGRES_MAX_OVERFLOW=10
```

**Redis Connection Pool:**
```bash
# In .env
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5
```

---

## Operations

### Starting ACE

**Development:**
```bash
# Start with hot reload
uv run uvicorn agentcore.a2a_protocol.main:app --reload --port 8001
```

**Production:**
```bash
# Start with Docker Compose
docker compose -f docker-compose.dev.yml up -d agentcore-api

# Or with Gunicorn (production WSGI server)
uv run gunicorn agentcore.a2a_protocol.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --access-logfile - \
  --error-logfile -
```

### Stopping ACE

**Graceful Shutdown:**
```bash
# Docker Compose
docker compose -f docker-compose.dev.yml stop agentcore-api

# Kubernetes
kubectl scale deployment agentcore --replicas=0 -n agentcore
```

**Force Stop (emergency):**
```bash
# Docker Compose
docker compose -f docker-compose.dev.yml kill agentcore-api

# Kubernetes
kubectl delete pod -l app=agentcore -n agentcore --grace-period=0
```

### Health Checks

**Application Health:**
```bash
curl http://localhost:8001/health

# Expected response:
# {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
```

**Database Health:**
```bash
curl http://localhost:8001/health/db

# Expected response:
# {"database": "connected", "latency_ms": 5.2}
```

**Redis Health:**
```bash
curl http://localhost:8001/health/redis

# Expected response:
# {"redis": "connected", "latency_ms": 1.3}
```

### Log Management

**View Logs:**
```bash
# Docker Compose
docker compose -f docker-compose.dev.yml logs -f agentcore-api

# Kubernetes
kubectl logs -f deployment/agentcore -n agentcore

# View last 100 lines
kubectl logs --tail=100 deployment/agentcore -n agentcore
```

**Log Levels:**
- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors requiring immediate attention

**Configure Log Level:**
```bash
# In .env
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'text'
```

---

## Monitoring and Alerting

### Grafana Dashboards

**Access Dashboard:**
1. Navigate to http://localhost:3000
2. Login (default: admin/admin)
3. Go to Dashboards → ACE (Agent Context Engineering) Monitoring

**Key Panels:**
- **Overview**: Concurrent agents, throughput, intervention rate
- **Performance**: Stage duration percentiles (p50, p95, p99)
- **Errors**: Error rate by severity and stage
- **Interventions**: Intervention latency and effectiveness
- **COMPASS Benchmarks**: All 5 targets with thresholds

### Prometheus Queries

**Concurrent Agents:**
```promql
count(count by (agent_id) (rate(ace_performance_updates_total[5m]) > 0))
```

**Task Throughput:**
```promql
sum(rate(ace_performance_updates_total[5m]))
```

**Intervention Latency (p95):**
```promql
histogram_quantile(0.95, rate(ace_intervention_latency_seconds_bucket[5m])) * 1000
```

**Error Rate:**
```promql
(sum(rate(ace_errors_total[5m])) / sum(rate(ace_performance_updates_total[5m]))) * 100
```

### Alert Response

**When Alert Fires:**

1. **Acknowledge Alert**:
   - Check Slack/email notification
   - Review alert description and severity
   - Access AlertManager: http://localhost:9093

2. **Assess Impact**:
   - Check Grafana dashboard for current state
   - Review Prometheus metrics for trends
   - Check application logs for errors

3. **Follow Runbook**:
   - Each alert has a runbook annotation
   - Execute remediation steps
   - Document actions taken

4. **Resolve and Review**:
   - Verify metrics return to normal
   - Alert auto-resolves when condition clears
   - Post-incident review for patterns

**Common Alerts:**

- `ACEInterventionLatencyHigh`: Check database and Redis latency
- `ACEErrorRateHigh`: Review agent logs and task complexity
- `ACECriticalErrorsDetected`: Immediate investigation required
- `ACEApproachingCapacity`: Prepare for scaling

---

## Troubleshooting

### High Intervention Latency

**Symptoms:**
- Intervention latency >200ms p95
- Alert: `ACEInterventionLatencyHigh`
- Slow agent response times

**Diagnosis:**
```bash
# Check database latency
curl http://localhost:8001/health/db

# Check Redis latency
redis-cli --latency

# Query slow queries
docker compose exec postgres psql -U postgres -c "
  SELECT query, mean_time, calls
  FROM pg_stat_statements
  WHERE mean_time > 100
  ORDER BY mean_time DESC
  LIMIT 10;"
```

**Resolution:**
1. **Database Optimization**:
   ```sql
   -- Create missing indexes
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ace_metrics_agent_created
     ON ace_performance_metrics (agent_id, created_at DESC);

   -- Analyze tables
   ANALYZE ace_performance_metrics;
   ```

2. **Connection Pool Tuning**:
   ```bash
   # Increase pool size
   POSTGRES_MAX_CONNECTIONS=75
   ```

3. **Redis Performance**:
   ```bash
   # Check Redis memory usage
   redis-cli INFO memory

   # Enable persistence if needed
   redis-cli CONFIG SET save "900 1 300 10 60 10000"
   ```

### High Error Rate

**Symptoms:**
- Error rate >10% over 5 minutes
- Alert: `ACEErrorRateHigh`
- Degraded agent performance

**Diagnosis:**
```bash
# Check error distribution
curl 'http://localhost:9090/api/v1/query?query=sum by (severity) (rate(ace_errors_total[5m]))'

# View agent logs
docker compose logs agentcore-api | grep ERROR | tail -50
```

**Resolution:**
1. **Identify Error Patterns**:
   - Review error logs for common error types
   - Check if errors are agent-specific or systemic

2. **Task Complexity**:
   - Reduce task complexity if possible
   - Increase timeout values if tasks are timing out

3. **Agent Configuration**:
   - Review agent capabilities and constraints
   - Ensure agent resources are adequate

### Database Connection Exhaustion

**Symptoms:**
- `OperationalError: connection pool exhausted`
- Slow query responses
- Timeout errors

**Diagnosis:**
```bash
# Check active connections
docker compose exec postgres psql -U postgres -c "
  SELECT count(*) as connections, state
  FROM pg_stat_activity
  GROUP BY state;"

# Check connection pool metrics
curl 'http://localhost:9090/api/v1/query?query=pg_stat_activity_count'
```

**Resolution:**
1. **Increase Pool Size**:
   ```bash
   # In .env
   POSTGRES_MAX_CONNECTIONS=100
   ```

2. **Optimize Queries**:
   - Review slow queries
   - Add indexes for frequent queries
   - Use connection pooling (PgBouncer)

3. **Connection Leaks**:
   - Review code for unclosed connections
   - Ensure proper use of async context managers

### Context Staleness High

**Symptoms:**
- Context staleness score >0.8
- Alert: `ACEContextStalenessHigh`
- Suboptimal interventions

**Diagnosis:**
```promql
# Check staleness by agent
ace_context_staleness{agent_id="$agent_id"}
```

**Resolution:**
1. **Trigger Context Refresh**:
   ```python
   # Via API
   import httpx
   async with httpx.AsyncClient() as client:
       await client.post(
           "http://localhost:8001/api/v1/ace/refresh-context",
           json={"agent_id": "agent-123"}
       )
   ```

2. **Reduce Staleness Threshold**:
   ```bash
   # In .env
   ACE_CONTEXT_STALENESS_THRESHOLD=0.6
   ```

3. **Increase Refresh Frequency**:
   - Configure more frequent context updates
   - Adjust intervention cooldown if needed

---

## Maintenance

### Database Maintenance

**Vacuum and Analyze:**
```bash
# Full vacuum (requires downtime)
docker compose exec postgres psql -U postgres -c "VACUUM FULL ANALYZE;"

# Online vacuum (no downtime)
docker compose exec postgres psql -U postgres -c "VACUUM ANALYZE;"

# Schedule regular vacuum
docker compose exec postgres psql -U postgres -c "
  ALTER TABLE ace_performance_metrics SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
  );"
```

**Index Maintenance:**
```bash
# Rebuild indexes
docker compose exec postgres psql -U postgres -c "
  REINDEX TABLE CONCURRENTLY ace_performance_metrics;"
```

**Data Retention:**
```bash
# Delete metrics older than 90 days
docker compose exec postgres psql -U postgres -c "
  DELETE FROM ace_performance_metrics
  WHERE created_at < NOW() - INTERVAL '90 days';"

# Compress old data (TimescaleDB)
docker compose exec postgres psql -U postgres -c "
  SELECT compress_chunk(chunk)
  FROM timescaledb_information.chunks
  WHERE hypertable_name = 'ace_performance_metrics'
    AND range_end < NOW() - INTERVAL '30 days';"
```

### Backup and Restore

**Backup Database:**
```bash
# Full backup
docker compose exec postgres pg_dump -U postgres agentcore > backup_$(date +%Y%m%d).sql

# Compressed backup
docker compose exec postgres pg_dump -U postgres agentcore | gzip > backup_$(date +%Y%m%d).sql.gz

# Backup to S3
docker compose exec postgres pg_dump -U postgres agentcore | \
  aws s3 cp - s3://your-bucket/backups/agentcore_$(date +%Y%m%d).sql
```

**Restore Database:**
```bash
# Restore from backup
docker compose exec -T postgres psql -U postgres agentcore < backup_20240101.sql

# Restore from compressed backup
gunzip -c backup_20240101.sql.gz | \
  docker compose exec -T postgres psql -U postgres agentcore
```

**Backup Grafana:**
```bash
# Backup Grafana data
docker run --rm -v agentcore_grafana-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/grafana-backup_$(date +%Y%m%d).tar.gz /data
```

### Software Updates

**Update Docker Images:**
```bash
# Pull latest images
docker compose -f docker-compose.dev.yml pull

# Restart with new images (rolling update)
docker compose -f docker-compose.dev.yml up -d --no-deps agentcore-api

# Verify update
docker compose exec agentcore-api python -c "import agentcore; print(agentcore.__version__)"
```

**Update Dependencies:**
```bash
# Update Python packages
uv lock --upgrade
uv sync

# Run tests
uv run pytest

# Deploy updates
docker compose -f docker-compose.dev.yml build agentcore-api
docker compose -f docker-compose.dev.yml up -d agentcore-api
```

---

## Scaling

### Horizontal Scaling

**Add AgentCore Replicas (Kubernetes):**
```bash
# Scale to 3 replicas
kubectl scale deployment agentcore --replicas=3 -n agentcore

# Or use HPA (Horizontal Pod Autoscaler)
kubectl apply -f k8s/hpa.yml

# HPA configuration:
# - Target: 70% CPU utilization
# - Min replicas: 2
# - Max replicas: 10
```

**Load Balancing:**
- Use Kubernetes Service for internal load balancing
- Use Ingress with NGINX or Traefik for external access
- Configure health check probes for proper traffic routing

### Database Scaling

**Read Replicas (PostgreSQL):**
```bash
# Configure read replica
docker run -d --name postgres-replica \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=agentcore \
  -e POSTGRES_REPLICATION_MODE=slave \
  -e POSTGRES_MASTER_HOST=postgres \
  postgres:14

# Update connection string for read operations
DATABASE_READ_URL=postgresql+asyncpg://user:pass@postgres-replica:5432/agentcore
```

**TimescaleDB Scaling:**
- Enable compression for old data
- Use continuous aggregates for dashboards
- Partition by time for efficient queries

**Redis Cluster:**
```bash
# Deploy Redis Cluster
docker run -d --name redis-cluster \
  -e REDIS_CLUSTER_ENABLED=yes \
  -e REDIS_CLUSTER_NODE_TIMEOUT=5000 \
  bitnami/redis-cluster:7.0

# Update connection string
REDIS_CLUSTER_URLS=redis://node1:6379,redis://node2:6379,redis://node3:6379
```

### Capacity Planning

**Current Capacity (ACE-031 Load Tests):**
- **Concurrent Agents**: 100 (target), 150+ (max tested)
- **Task Throughput**: 10.5 tasks/sec sustained
- **Intervention Latency**: 50ms p95 (target: <200ms)
- **System Overhead**: 3.2% (target: <5%)

**Scaling Triggers:**
- **Scale Up**: When concurrent agents >80 or CPU >70%
- **Scale Down**: When concurrent agents <30 and CPU <30%

**Resource Requirements per 100 Agents:**
- **CPU**: 4 cores
- **Memory**: 8GB
- **Database**: 50GB storage, 100 IOPS
- **Redis**: 2GB memory

---

## Disaster Recovery

### Backup Strategy

**Frequency:**
- **Database**: Daily full backup + continuous WAL archiving
- **Grafana**: Weekly backup
- **Configuration**: On every change (version controlled)

**Retention:**
- **Hot backups**: 7 days (local)
- **Warm backups**: 30 days (S3)
- **Cold backups**: 1 year (S3 Glacier)

### Recovery Procedures

**Database Recovery:**
```bash
# Stop application
docker compose -f docker-compose.dev.yml stop agentcore-api

# Restore database
gunzip -c backup_20240101.sql.gz | \
  docker compose exec -T postgres psql -U postgres agentcore

# Verify data
docker compose exec postgres psql -U postgres agentcore -c "
  SELECT COUNT(*) FROM ace_performance_metrics;"

# Restart application
docker compose -f docker-compose.dev.yml start agentcore-api
```

**Point-in-Time Recovery (PITR):**
```bash
# Restore to specific timestamp
docker compose exec postgres pg_restore \
  --target-time="2024-01-01 12:00:00" \
  --dbname=agentcore backup.dump
```

**Application Recovery:**
```bash
# Rollback to previous version
kubectl rollout undo deployment/agentcore -n agentcore

# Or deploy specific version
kubectl set image deployment/agentcore \
  agentcore=agentcore:v1.2.3 -n agentcore
```

### Failover Procedures

**Primary Database Failure:**
1. Promote read replica to primary
2. Update connection strings
3. Restart application pods

**Application Pod Failure:**
- Kubernetes automatically restarts failed pods
- Load balancer routes traffic to healthy pods
- No manual intervention required

---

## Performance Tuning

### Database Tuning

**PostgreSQL Configuration:**
```ini
# postgresql.conf
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1  # For SSD
effective_io_concurrency = 200  # For SSD
work_mem = 20MB
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
```

**Key Indexes:**
```sql
-- Performance metrics
CREATE INDEX CONCURRENTLY idx_ace_metrics_agent_created
  ON ace_performance_metrics (agent_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_ace_metrics_stage_created
  ON ace_performance_metrics (stage, created_at DESC);

-- Errors
CREATE INDEX CONCURRENTLY idx_ace_errors_severity_created
  ON ace_errors (severity, created_at DESC);

-- Interventions
CREATE INDEX CONCURRENTLY idx_ace_interventions_agent_created
  ON ace_interventions (agent_id, created_at DESC);
```

### Redis Tuning

**Configuration:**
```ini
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
```

### Application Tuning

**Batch Size Optimization:**
```python
# Increase batch size for high throughput
ACE_BATCH_SIZE=200  # Default: 100
ACE_BATCH_TIMEOUT=2.0  # Default: 1.0
```

**Connection Pool Tuning:**
```python
# Increase pool size for more concurrent agents
POSTGRES_MAX_CONNECTIONS=100  # Default: 50
REDIS_MAX_CONNECTIONS=100  # Default: 50
```

---

## Security

### Authentication

**JWT Configuration:**
```bash
JWT_SECRET_KEY=secure-random-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

**API Key Authentication:**
```bash
# Generate API key
openssl rand -hex 32

# Set in .env
API_KEY=your-generated-api-key
```

### Authorization

**Role-Based Access Control (RBAC):**
- `admin`: Full access to all operations
- `operator`: Read/write access to ACE operations
- `viewer`: Read-only access to metrics and dashboards

### Network Security

**Firewall Rules:**
```bash
# Allow AgentCore API (internal only)
ufw allow from 10.0.0.0/8 to any port 8001

# Allow Prometheus (internal only)
ufw allow from 10.0.0.0/8 to any port 9090

# Allow Grafana (external, behind reverse proxy)
ufw allow 3000/tcp
```

**TLS/SSL:**
```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem

# Use Let's Encrypt (production)
certbot certonly --standalone -d agentcore.example.com
```

### Secrets Management

**Using Kubernetes Secrets:**
```bash
# Create secret
kubectl create secret generic agentcore-secrets \
  --from-literal=database-password=secure-password \
  --from-literal=redis-password=secure-password \
  -n agentcore

# Reference in deployment
env:
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef:
        name: agentcore-secrets
        key: database-password
```

---

## Support and Contact

### Documentation

- **Architecture**: `docs/architecture/ace-architecture.md`
- **API Reference**: `docs/api/ace-api.md`
- **COMPASS Validation**: `docs/ace-compass-validation-report.md`
- **Load Test Report**: `docs/ace-load-test-report.md`
- **Monitoring Setup**: `monitoring/README.md`

### Troubleshooting Resources

- **Alert Runbooks**: See `monitoring/alerts/ace-alerts.yml` annotations
- **Common Issues**: See Troubleshooting section above
- **GitHub Issues**: https://github.com/your-org/agentcore/issues

### Contact

- **Operations Team**: ops@agentcore.example.com
- **ACE Team**: ace-team@agentcore.example.com
- **On-Call**: Via PagerDuty (critical alerts)

---

## Appendix

### Metrics Reference

See Prometheus `/metrics` endpoint for full list. Key metrics:

**Counters:**
- `ace_performance_updates_total{agent_id, stage}`
- `ace_errors_total{agent_id, stage, severity}`
- `ace_interventions_total{agent_id, type}`

**Gauges:**
- `ace_baseline_deviation{agent_id, stage, metric}`
- `ace_error_rate{agent_id, stage}`
- `ace_intervention_effectiveness{agent_id}`
- `ace_context_staleness{agent_id}`

**Histograms:**
- `ace_metric_computation_duration_seconds{operation}`
- `ace_intervention_latency_seconds{type}`
- `ace_mem_query_duration_seconds{query_type}`
- `ace_stage_duration_seconds{agent_id, stage}`

### Alert Rules Reference

See `monitoring/alerts/ace-alerts.yml` for full list. Key alerts:

**Performance (4 rules):**
- `ACEInterventionLatencyHigh`
- `ACEInterventionLatencyCritical`
- `ACEStageDurationAnomaly`
- `ACEMetricComputationSlow`

**Errors (3 rules):**
- `ACEErrorRateHigh`
- `ACECriticalErrorsDetected`
- `ACEErrorAccumulation`

**Interventions (3 rules):**
- `ACEInterventionRateTooHigh`
- `ACEInterventionMissed`
- `ACEInterventionEffectivenessLow`

**System Health (4 rules):**
- `ACEContextStalenessHigh`
- `ACEBaselineDeviationSignificant`
- `ACEMEMQueryLatencyHigh`
- `ACEMetricsNotRecorded`

**Scalability (2 rules):**
- `ACEConcurrentAgentsHigh`
- `ACEApproachingCapacity`

---

**Document Version:** 1.0
**Last Reviewed:** 2025-11-09
**Next Review:** 2025-12-09
**Maintained By:** ACE Team
