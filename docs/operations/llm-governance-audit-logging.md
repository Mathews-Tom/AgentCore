# LLM Governance Audit Logging

## Overview

This document describes the audit logging system for LLM model governance violations in AgentCore. The system provides comprehensive logging, monitoring, and alerting for security and compliance requirements.

## Audit Log Structure

All governance violations are logged with structured JSON format to a dedicated audit log stream. Each log entry includes:

### Required Fields

- **timestamp**: Unix timestamp (seconds since epoch) when violation occurred
- **trace_id**: A2A protocol trace ID for distributed tracing
- **source_agent**: Agent identifier that attempted the operation
- **session_id**: Session identifier for multi-turn interactions
- **attempted_model**: Model identifier that was requested
- **allowed_models**: List of models permitted by current configuration
- **reason**: Human-readable explanation of the violation
- **audit_event**: Always "governance_violation" for filtering
- **violation_type**: Type of violation (see below)
- **severity**: Violation severity level

### Violation Types

1. **disallowed_model**: Attempted to use a model not in `ALLOWED_MODELS` configuration
2. **missing_api_key**: Provider API key not configured in environment
3. **request_type**: Additional context for streaming vs non-streaming requests

## Log Examples

### Disallowed Model Violation

```json
{
  "timestamp": 1704067200.0,
  "level": "WARNING",
  "message": "AUDIT: Model governance violation - disallowed model",
  "audit_event": "governance_violation",
  "violation_type": "disallowed_model",
  "trace_id": "trace-abc-123",
  "source_agent": "agent-researcher-001",
  "session_id": "session-xyz-789",
  "attempted_model": "gpt-4-turbo",
  "allowed_models": ["gpt-4.1-mini", "gpt-5-mini", "claude-3-5-haiku-20241022", "gemini-1.5-flash"],
  "reason": "Model 'gpt-4-turbo' is not in ALLOWED_MODELS configuration",
  "severity": "high"
}
```

### Missing API Key Violation

```json
{
  "timestamp": 1704067200.0,
  "level": "ERROR",
  "message": "AUDIT: Model governance violation - missing API key",
  "audit_event": "governance_violation",
  "violation_type": "missing_api_key",
  "provider": "openai",
  "reason": "OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
  "severity": "critical"
}
```

## Prometheus Metrics

Governance violations are tracked via Prometheus metrics:

```
llm_governance_violations_total{model="gpt-4-turbo", source_agent="agent-001"}
```

### Alert Rules

The system includes five Prometheus alert rules:

1. **HighModelGovernanceViolationRate**
   - Threshold: >10 violations/hour
   - Severity: critical
   - Action: Immediate investigation required

2. **HighModelGovernanceViolationsPerAgent**
   - Threshold: >5 violations/hour per agent
   - Severity: warning
   - Action: Investigate agent configuration

3. **HighModelGovernanceViolationsPerModel**
   - Threshold: >5 violations/hour per model
   - Severity: warning
   - Action: Review ALLOWED_MODELS configuration

4. **ModelGovernanceViolationSpike**
   - Threshold: 3x hourly average within 5 minutes
   - Severity: critical
   - Action: Potential security incident

5. **MissingAPIKeyError**
   - Threshold: Any occurrence
   - Severity: critical
   - Action: Immediate configuration fix

## Monitoring System Integration

### Alertmanager Configuration

Configure Alertmanager to route governance alerts to appropriate channels:

```yaml
route:
  group_by: ['alertname', 'component']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    # LLM Governance violations route to security team
    - match:
        component: llm-service
        security: true
      receiver: 'security-team'
      continue: true

    # Critical violations also go to on-call
    - match:
        component: llm-service
        severity: critical
      receiver: 'pagerduty-oncall'

receivers:
  - name: 'default'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'

  - name: 'security-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#security-alerts'
        title: 'LLM Governance Violation'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}\n{{ end }}'

  - name: 'pagerduty-oncall'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

### Slack Notification Format

Governance violation alerts in Slack include:

- **Alert Name**: High Model Governance Violation Rate
- **Severity**: critical/warning
- **Component**: llm-service
- **Description**: Violation rate, threshold, affected agent/model
- **Runbook Link**: Investigation procedures
- **Dashboard Link**: Grafana dashboard for detailed analysis

### PagerDuty Integration

Critical governance violations (severity: critical) trigger PagerDuty incidents:

- **Service**: LLM Service Governance
- **Incident Title**: Alert name + summary
- **Incident Description**: Full alert details with context
- **Urgency**: High for critical, Low for warning
- **Auto-resolve**: After 1 hour of no violations

## Audit Log Retention Policy

### Retention Periods

- **Hot Storage (Loki/Elasticsearch)**: 90 days minimum
- **Cold Storage (S3/GCS)**: 7 years for compliance
- **Metrics (Prometheus)**: 15 days (high resolution), 1 year (downsampled)

### Storage Location

- **Application Logs**: `/var/log/agentcore/llm-service.log`
- **Structured Logs**: Sent to Loki/Elasticsearch via log shipper
- **Metrics**: Prometheus TSDB
- **Long-term Archive**: S3 bucket `agentcore-audit-logs` with lifecycle policies

### Compliance Requirements

- Logs MUST be immutable once written (append-only)
- Access to audit logs MUST be tracked (access audit)
- Logs MUST be encrypted at rest (AES-256)
- Logs MUST be encrypted in transit (TLS 1.3+)
- Log retention MUST meet regulatory requirements (SOC 2, GDPR, HIPAA)

## Query Examples

### Common Audit Scenarios

#### 1. Find all violations by a specific agent

**Loki Query:**
```logql
{component="llm-service"}
  | json
  | audit_event="governance_violation"
  | source_agent="agent-researcher-001"
```

**Elasticsearch Query:**
```json
{
  "query": {
    "bool": {
      "must": [
        { "term": { "audit_event": "governance_violation" }},
        { "term": { "source_agent": "agent-researcher-001" }}
      ]
    }
  }
}
```

#### 2. Find all attempts to use a specific model

**Loki Query:**
```logql
{component="llm-service"}
  | json
  | audit_event="governance_violation"
  | attempted_model="gpt-4-turbo"
```

#### 3. Find violations in a specific time range

**Loki Query:**
```logql
{component="llm-service"}
  | json
  | audit_event="governance_violation"
  | __timestamp__ >= 1704067200
  | __timestamp__ <= 1704153600
```

#### 4. Count violations per agent (last 24 hours)

**PromQL Query:**
```promql
sum by (source_agent) (
  increase(llm_governance_violations_total[24h])
)
```

#### 5. Top 10 most violated models

**PromQL Query:**
```promql
topk(10,
  sum by (model) (
    increase(llm_governance_violations_total[7d])
  )
)
```

#### 6. Find all missing API key errors

**Loki Query:**
```logql
{component="llm-service"}
  | json
  | audit_event="governance_violation"
  | violation_type="missing_api_key"
```

#### 7. Violations by session (multi-turn analysis)

**Loki Query:**
```logql
{component="llm-service"}
  | json
  | audit_event="governance_violation"
  | session_id="session-xyz-789"
```

#### 8. High severity violations only

**Loki Query:**
```logql
{component="llm-service"}
  | json
  | audit_event="governance_violation"
  | severity="critical"
```

## Security Response Procedures

### When Alert Fires

1. **Acknowledge Alert**: Acknowledge in PagerDuty/Slack
2. **Initial Assessment**: Check Grafana dashboard for context
3. **Query Audit Logs**: Use Loki/Elasticsearch to find violation details
4. **Identify Pattern**: Single agent? Specific model? Time-based?
5. **Investigate Root Cause**:
   - Agent misconfiguration?
   - Malicious activity?
   - Outdated documentation?
   - Integration test gone wrong?
6. **Take Action**:
   - Update ALLOWED_MODELS if legitimate need
   - Revoke agent credentials if malicious
   - Fix agent configuration if error
   - Update documentation if confusion
7. **Document Incident**: Create incident report
8. **Post-Mortem**: Review and prevent recurrence

### Escalation Path

1. **Level 1 (0-15 min)**: On-call engineer investigates
2. **Level 2 (15-30 min)**: Security team notified if suspicious
3. **Level 3 (30+ min)**: Engineering manager + CISO for security incidents

## Configuration

### Environment Variables

```bash
# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json  # Required for structured logging

# Audit log retention
AUDIT_LOG_RETENTION_DAYS=90

# Alertmanager endpoint
ALERTMANAGER_URL=http://alertmanager:9093
```

### Python Logging Configuration

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure JSON formatter for audit logs
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(timestamp)s %(level)s %(name)s %(message)s',
    rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
)
logHandler.setFormatter(formatter)

logger = logging.getLogger('agentcore.a2a_protocol.services.llm_service')
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

## Testing

### Manual Testing

```python
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest

# Trigger governance violation
request = LLMRequest(
    model="gpt-4-turbo",  # Not in ALLOWED_MODELS
    messages=[{"role": "user", "content": "test"}],
    trace_id="test-trace-123",
    source_agent="test-agent",
    session_id="test-session",
)

try:
    await llm_service.complete(request)
except ModelNotAllowedError:
    # Check logs for audit entry
    pass
```

### Automated Testing

See `tests/integration/test_llm_governance.py` for comprehensive test suite.

## References

- **Prometheus Alert Rules**: `deploy/prometheus/alert-rules.yml`
- **LLM Service**: `src/agentcore/a2a_protocol/services/llm_service.py`
- **Metrics**: `src/agentcore/a2a_protocol/metrics/llm_metrics.py`
- **Grafana Dashboard**: https://grafana/d/llm-governance
- **Runbook**: `docs/ops/llm-governance-runbook.md`

## Changelog

- **2025-10-26**: Initial audit logging implementation
- **2025-10-26**: Added Prometheus alert rules
- **2025-10-26**: Documented retention policy and query examples
