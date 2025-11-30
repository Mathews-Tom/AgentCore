# Modular Agent Core Operations Guide

**Version:** 1.0
**Component:** Modular Agent Core
**Status:** Production Ready
**Last Updated:** 2025-11-30

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Configuration](#configuration)
5. [Operations](#operations)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Troubleshooting](#troubleshooting)
8. [Incident Response](#incident-response)
9. [Security](#security)
10. [Scaling](#scaling)
11. [Disaster Recovery](#disaster-recovery)

---

## Overview

### What is Modular Agent Core?

Modular Agent Core decomposes complex agentic workflows into four specialized modules that coordinate through well-defined interfaces:

- **Planner**: Analyzes requests and creates structured execution plans
- **Executor**: Executes plan steps by invoking tools and resources
- **Verifier**: Validates results against success criteria
- **Generator**: Synthesizes final responses from verified results

### Production Readiness Status

✅ **Production Ready** - All validation and load tests passed:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Task Success Rate | +15% over baseline | +18% | ✅ |
| Tool Call Accuracy | +10% improvement | +12% | ✅ |
| End-to-End Latency | <2x baseline | 1.7x | ✅ |
| Cost Efficiency | 30% reduction | 35% | ✅ |
| Error Recovery Rate | >80% | 83% | ✅ |
| Concurrent Executions | 100 | 100 (0 errors) | ✅ |
| Module Transition Latency | <500ms | 120ms avg | ✅ |

### Key Features

- **Four-Module Pipeline**: Planner → Executor → Verifier → Generator
- **Plan Refinement Loop**: Iterative improvement based on verification feedback
- **Module-Specific Models**: Independent LLM configuration per module
- **Distributed Tracing**: End-to-end execution visibility
- **State Persistence**: Crash recovery with checkpoint support
- **Security**: JWT auth, RBAC, audit logging with sensitive data redaction

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgentCore API Layer                         │
│                    (FastAPI + A2A Protocol)                      │
│                                                                  │
│  POST /api/v1/jsonrpc  ─────▶  modular.solve                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ModularCoordinator                             │
│                                                                  │
│  • Orchestrates module pipeline                                  │
│  • Manages execution state and transitions                       │
│  • Enforces iteration limits (max: 5)                           │
│  • Coordinates plan refinement loops                             │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    Planner    │   │   Executor    │   │   Verifier    │
│               │   │               │   │               │
│ • Plan create │   │ • Tool calls  │   │ • Validation  │
│ • Plan refine │   │ • Retries     │   │ • Feedback    │
│ • Dependencies│   │ • State track │   │ • Confidence  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────┐
                    │   Generator   │
                    │               │
                    │ • Response    │
                    │ • Formatting  │
                    │ • Evidence    │
                    └───────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────────┐
│                    Supporting Services                            │
├───────────────┬───────────────┬───────────────┬──────────────────┤
│ StateManager  │ ModularTracer │ SecuritySvc   │ PerformanceOpt   │
│ • Checkpoints │ • Trace IDs   │ • JWT Auth    │ • Response cache │
│ • Recovery    │ • Spans       │ • RBAC        │ • Prompt opt     │
│ • Persistence │ • Events      │ • Audit logs  │ • Parallel exec  │
└───────────────┴───────────────┴───────────────┴──────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────────┐
│                    Infrastructure Layer                           │
├───────────────┬───────────────┬───────────────┬──────────────────┤
│ PostgreSQL    │ Redis         │ Prometheus    │ Grafana          │
│ • Exec plans  │ • Cache       │ • Metrics     │ • Dashboards     │
│ • Audit logs  │ • State       │ • Alerts      │ • Visualization  │
└───────────────┴───────────────┴───────────────┴──────────────────┘
```

### Execution Flow

```
1. Request         2. Planning        3. Execution       4. Verification
   ┌─────┐            ┌─────┐            ┌─────┐            ┌─────┐
   │Query│────────────│Plan │────────────│Exec │────────────│Check│
   └─────┘            └─────┘            └─────┘            └─────┘
                         │                                      │
                         │◄──────── Refine Loop ───────────────┘
                         │         (if failed)
                         ▼
                    5. Generation
                      ┌─────┐
                      │ Gen │─────────▶ Response
                      └─────┘
```

### Module Interfaces

Each module implements a standard interface:

```python
class BaseModule(ABC):
    async def initialize(self) -> None
    async def process(self, input: ModuleInput) -> ModuleOutput
    async def shutdown(self) -> None
    async def health_check(self) -> HealthStatus
```

---

## Deployment

### Prerequisites

- Python 3.12+
- PostgreSQL 14+ with async driver (asyncpg)
- Redis 6+ (for caching and state)
- Docker (optional, for containerized deployment)

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/agentcore

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM Configuration (per-module)
PLANNER_MODEL=gpt-4.1
EXECUTOR_MODEL=gpt-4.1-mini
VERIFIER_MODEL=gpt-4.1-mini
GENERATOR_MODEL=gpt-4.1

# Security
JWT_SECRET_KEY=<secure-random-key>
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Modular Settings
MODULAR_MAX_ITERATIONS=5
MODULAR_STEP_TIMEOUT_SECONDS=300
MODULAR_ENABLE_TRACING=true
MODULAR_ENABLE_CACHING=true
```

### Docker Deployment

```bash
# Build image
docker build -t agentcore:latest .

# Run with dependencies
docker compose -f docker-compose.yml up -d
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/modular-agent-core/

# Verify deployment
kubectl get pods -l app=agentcore-modular
kubectl get svc agentcore-modular
```

### Health Verification

```bash
# Check health endpoint
curl http://localhost:8001/health

# Expected response:
# {"status": "healthy", "modules": {"planner": "ready", "executor": "ready", "verifier": "ready", "generator": "ready"}}
```

---

## Configuration

### Module-Specific LLM Models

Configure different models per module for cost optimization:

```toml
# config.toml
[modular.planner]
model = "gpt-4.1"          # Large model for complex reasoning
temperature = 0.7
max_tokens = 4096

[modular.executor]
model = "gpt-4.1-mini"     # Smaller model for tool calls
temperature = 0.3
max_tokens = 2048

[modular.verifier]
model = "gpt-4.1-mini"     # Smaller model for validation
temperature = 0.1
max_tokens = 1024

[modular.generator]
model = "gpt-4.1"          # Medium model for synthesis
temperature = 0.5
max_tokens = 4096
```

### Execution Limits

```toml
[modular.limits]
max_iterations = 5         # Max refinement loops
max_plan_steps = 50        # Max steps per plan
step_timeout_seconds = 300 # Per-step timeout
total_timeout_seconds = 1800 # Total execution timeout
max_retries_per_step = 3   # Tool retry limit
```

### Caching Configuration

```toml
[modular.cache]
enabled = true
ttl_seconds = 3600         # Cache entry TTL
max_entries = 10000        # Max cache size
compression = "gzip"       # Compression algorithm
```

### Tracing Configuration

```toml
[modular.tracing]
enabled = true
sample_rate = 1.0          # 100% sampling in production
export_endpoint = "http://jaeger:14268/api/traces"
service_name = "modular-agent-core"
```

---

## Operations

### JSON-RPC API

The primary interface is the `modular.solve` JSON-RPC method:

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "method": "modular.solve",
    "params": {
      "query": "Create a Python function to calculate fibonacci",
      "config": {
        "max_iterations": 5,
        "enable_caching": true
      }
    },
    "id": "req-001"
  }'
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "result": {
    "success": true,
    "response": "...",
    "execution_trace": {
      "trace_id": "trace-abc123",
      "modules_invoked": ["planner", "executor", "verifier", "generator"],
      "iterations": 1,
      "total_duration_ms": 2340,
      "plan_steps_executed": 3
    },
    "metadata": {
      "tokens_used": 1250,
      "cost_usd": 0.025,
      "cache_hits": 2
    }
  },
  "id": "req-001"
}
```

### Common Operations

**List Active Executions:**
```bash
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.listExecutions","params":{"status":"running"},"id":"1"}'
```

**Cancel Execution:**
```bash
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.cancelExecution","params":{"execution_id":"exec-123"},"id":"1"}'
```

**Get Execution Details:**
```bash
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.getExecution","params":{"execution_id":"exec-123"},"id":"1"}'
```

---

## Monitoring and Alerting

### Prometheus Metrics

Key metrics exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `modular_executions_total` | Counter | Total executions by status |
| `modular_execution_duration_seconds` | Histogram | Execution duration |
| `modular_module_latency_seconds` | Histogram | Per-module latency |
| `modular_iterations_total` | Counter | Total refinement iterations |
| `modular_cache_hits_total` | Counter | Response cache hits |
| `modular_errors_total` | Counter | Errors by module and type |
| `modular_active_executions` | Gauge | Currently running executions |

### Alert Rules

```yaml
# prometheus/alerts/modular.yml
groups:
  - name: modular-agent-core
    rules:
      - alert: HighErrorRate
        expr: rate(modular_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in Modular Agent Core"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: HighLatency
        expr: histogram_quantile(0.95, modular_execution_duration_seconds) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p95 latency in modular executions"

      - alert: TooManyIterations
        expr: rate(modular_iterations_total[5m]) / rate(modular_executions_total[5m]) > 3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High average iteration count indicates plan quality issues"

      - alert: ModuleUnhealthy
        expr: modular_module_healthy == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Module {{ $labels.module }} is unhealthy"
```

### Grafana Dashboards

Import dashboards from `k8s/grafana/dashboards/`:

1. **Modular Overview**: Execution success rate, latency, throughput
2. **Module Health**: Per-module latency, error rates, queue depth
3. **Cost Analysis**: Token usage, model costs, cache savings
4. **Tracing**: Execution flow visualization, bottleneck detection

---

## Troubleshooting

### Common Issues

#### 1. Execution Timeout

**Symptoms:** Executions exceed `total_timeout_seconds`

**Diagnosis:**
```bash
# Check slow executions
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.getSlowExecutions","params":{"threshold_ms":30000},"id":"1"}'
```

**Resolution:**
- Increase timeout limits if queries are legitimately complex
- Check LLM provider latency
- Review plan complexity (reduce `max_plan_steps` if needed)
- Enable response caching for repeated patterns

#### 2. High Iteration Count

**Symptoms:** Executions frequently hit `max_iterations` limit

**Diagnosis:**
```bash
# Check iteration distribution
curl http://localhost:8001/metrics | grep modular_iterations
```

**Resolution:**
- Review Verifier confidence thresholds (may be too strict)
- Improve Planner prompts for better initial plans
- Check for ambiguous success criteria in plans
- Review recent plan failures for patterns

#### 3. Module Initialization Failure

**Symptoms:** Module reports "not ready" in health check

**Diagnosis:**
```bash
# Check module logs
kubectl logs -l app=agentcore-modular --tail=100 | grep -i "error\|failed"

# Check module status
curl http://localhost:8001/health | jq '.modules'
```

**Resolution:**
- Verify LLM API keys are valid
- Check database connectivity
- Ensure Redis is accessible
- Review module-specific configuration

#### 4. Cache Miss Rate High

**Symptoms:** `modular_cache_hits_total` not increasing

**Diagnosis:**
```bash
# Check cache stats
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.getCacheStats","id":"1"}'
```

**Resolution:**
- Verify caching is enabled in config
- Check Redis connectivity
- Review cache TTL settings
- Analyze query patterns for cacheability

### Log Analysis

Key log patterns to monitor:

```bash
# Execution failures
grep "execution_failed" /var/log/agentcore/modular.log

# Module transitions
grep "module_transition" /var/log/agentcore/modular.log | jq '.from_module, .to_module'

# Plan refinements
grep "plan_refined" /var/log/agentcore/modular.log | jq '.iteration, .reason'

# Security events
grep "security_event" /var/log/agentcore/modular.log | jq '.action, .result'
```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P0 | Service down | 15 minutes | All modules unhealthy |
| P1 | Major degradation | 1 hour | >50% error rate |
| P2 | Minor degradation | 4 hours | Single module slow |
| P3 | Low impact | 24 hours | Cache miss rate high |

### Incident Playbooks

#### P0: All Modules Unhealthy

1. **Verify infrastructure**
   ```bash
   kubectl get pods -l app=agentcore-modular
   kubectl describe pod <pod-name>
   ```

2. **Check dependencies**
   ```bash
   # Database
   psql -h <host> -U <user> -c "SELECT 1"

   # Redis
   redis-cli -h <host> ping

   # LLM API
   curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

3. **Review recent changes**
   ```bash
   git log --oneline -10
   kubectl rollout history deployment/agentcore-modular
   ```

4. **Rollback if needed**
   ```bash
   kubectl rollout undo deployment/agentcore-modular
   ```

5. **Force restart**
   ```bash
   kubectl rollout restart deployment/agentcore-modular
   ```

#### P1: High Error Rate

1. **Identify error source**
   ```bash
   curl http://localhost:8001/metrics | grep modular_errors_total
   ```

2. **Check module-specific errors**
   ```bash
   grep "error" /var/log/agentcore/modular.log | jq '.module, .error_type' | sort | uniq -c
   ```

3. **Review trace data**
   - Open Jaeger UI
   - Filter by `service=modular-agent-core` and `error=true`
   - Identify common failure patterns

4. **Apply mitigation**
   - Enable circuit breaker for failing module
   - Increase retry limits if transient
   - Scale up if resource constrained

### Post-Incident Review

After every P0/P1 incident:

1. Document timeline in incident log
2. Identify root cause
3. Create action items for prevention
4. Update runbook with new learnings
5. Review alert thresholds

---

## Security

### Authentication

All modular API calls require JWT authentication:

```bash
# Get token
curl -X POST http://localhost:8001/api/v1/auth/token \
  -d '{"username":"operator","password":"***"}'

# Use token
curl http://localhost:8001/api/v1/jsonrpc \
  -H "Authorization: Bearer <token>" \
  -d '{"jsonrpc":"2.0","method":"modular.solve",...}'
```

### RBAC Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `modular:execute` | Execute modular queries | user, admin |
| `modular:read` | View execution history | user, admin |
| `module:planner` | Access planner module | admin |
| `module:executor` | Access executor module | admin |
| `module:verifier` | Access verifier module | admin |
| `module:generator` | Access generator module | admin |
| `module:internal` | Internal module comms | system |
| `modular:admin` | Full administrative access | admin |

### Audit Logging

All operations are logged with:

- Trace ID for correlation
- User/agent identity
- Action performed
- Result (success/failure)
- Timestamp

Sensitive data is automatically redacted:
- `api_key` → `***REDACTED***`
- `password` → `***REDACTED***`
- `token` → `***REDACTED***`
- `secret` → `***REDACTED***`

### Security Checklist

- [ ] JWT secret rotated every 90 days
- [ ] TLS enabled for all connections
- [ ] Database credentials in secrets manager
- [ ] Audit logs shipped to SIEM
- [ ] Rate limiting enabled
- [ ] Input validation on all parameters

---

## Scaling

### Horizontal Scaling

```yaml
# k8s/modular-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentcore-modular
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentcore-modular
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: modular_active_executions
        target:
          type: AverageValue
          averageValue: 20
```

### Capacity Planning

| Concurrent Executions | Pods | CPU (cores) | Memory (GB) |
|----------------------|------|-------------|-------------|
| 10 | 2 | 2 | 4 |
| 50 | 4 | 4 | 8 |
| 100 | 8 | 8 | 16 |
| 500 | 20 | 20 | 40 |

### Performance Optimization

1. **Enable Response Caching**: 30-50% latency reduction for repeated patterns
2. **Use Module-Specific Models**: Cost savings without quality loss
3. **Tune Connection Pools**: Match LLM API rate limits
4. **Enable Parallel Execution**: For independent plan steps

---

## Disaster Recovery

### Backup Strategy

| Component | Frequency | Retention | Storage |
|-----------|-----------|-----------|---------|
| PostgreSQL | Hourly | 30 days | S3 |
| Redis (RDB) | 6 hours | 7 days | S3 |
| Configuration | On change | 90 days | Git |
| Audit Logs | Daily | 1 year | S3 |

### Recovery Procedures

#### Database Recovery

```bash
# List available backups
aws s3 ls s3://agentcore-backups/postgres/

# Restore from backup
pg_restore -h <host> -U <user> -d agentcore backup.dump
```

#### State Recovery

Executions in progress at crash time can be recovered:

```bash
# List recoverable executions
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.listRecoverable","id":"1"}'

# Resume execution
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.resumeExecution","params":{"execution_id":"exec-123"},"id":"1"}'
```

### RTO/RPO Targets

| Metric | Target |
|--------|--------|
| Recovery Time Objective (RTO) | 1 hour |
| Recovery Point Objective (RPO) | 1 hour |
| Data Loss Tolerance | Last checkpoint |

---

## Appendix

### JSON-RPC Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32600 | Invalid Request | Malformed JSON-RPC request |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Parameter validation failed |
| -32603 | Internal error | Server error |
| -32001 | Execution timeout | Total timeout exceeded |
| -32002 | Module error | Module-specific failure |
| -32003 | Max iterations | Refinement limit reached |
| -32004 | Authorization failed | RBAC check failed |

### Useful Commands

```bash
# Check overall health
curl http://localhost:8001/health | jq

# Get module metrics
curl http://localhost:8001/metrics | grep modular_

# List recent executions
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.listExecutions","params":{"limit":10},"id":"1"}'

# Get execution trace
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.getTrace","params":{"trace_id":"abc123"},"id":"1"}'

# View cache stats
curl http://localhost:8001/api/v1/jsonrpc \
  -d '{"jsonrpc":"2.0","method":"modular.getCacheStats","id":"1"}'
```

### Contact

- **On-call**: PagerDuty escalation policy `agentcore-modular`
- **Slack**: `#agentcore-incidents`
- **Documentation**: `docs/modular-agent-core-operations-guide.md`
