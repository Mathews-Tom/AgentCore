# ACE Monitoring and Alerting Setup

Production monitoring and alerting infrastructure for AgentCore ACE (Agent Context Engineering) system.

## Overview

This monitoring stack provides comprehensive observability for ACE performance, error tracking, and intervention effectiveness, aligned with COMPASS benchmarks and production readiness requirements.

### Components

- **Prometheus**: Time-series metrics collection and alerting engine
- **AlertManager**: Alert routing, grouping, and notification dispatch
- **Grafana**: Visualization dashboards and metrics exploration
- **Exporters**: System (node), database (PostgreSQL), and cache (Redis) metrics

### Metrics Coverage

**ACE Performance:**
- Stage duration (planning, execution, reflection, verification)
- Task throughput and concurrent agent count
- System overhead and resource utilization

**Error Tracking:**
- Error accumulation by severity (low, medium, high, critical)
- Error patterns and compounding detection
- Critical error recall (COMPASS target: 90%+)

**Interventions:**
- Intervention rate and latency (COMPASS target: <200ms p95)
- Intervention effectiveness and precision (COMPASS target: 85%+)
- Context staleness and baseline deviation

**COMPASS Benchmarks:**
- Long-horizon accuracy improvement (target: +20%)
- Critical error recall (target: 90%+)
- Intervention precision (target: 85%+)
- Monthly cost tracking (target: <$150 for 100 agents)
- System overhead (target: <5%)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- AgentCore API running and exposing `/metrics` endpoint
- Environment variables configured (see Configuration section)

### Start Monitoring Stack

```bash
# Start all monitoring services
docker compose -f docker-compose.monitoring.yml up -d

# Verify services are running
docker compose -f docker-compose.monitoring.yml ps

# View logs
docker compose -f docker-compose.monitoring.yml logs -f
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (default: admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### Import ACE Dashboard

1. Open Grafana at http://localhost:3000
2. Navigate to Dashboards → Import
3. Upload `monitoring/dashboards/ace-dashboard.json`
4. Select "Prometheus" as the datasource
5. Click "Import"

The ACE dashboard provides real-time visibility into all key metrics.

## Configuration

### Environment Variables

Create a `.env` file in the project root with monitoring configuration:

```bash
# SMTP Settings (for email alerts)
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-app-password
ALERT_FROM_EMAIL=alerts@agentcore.example.com
DEFAULT_EMAIL=ops@agentcore.example.com

# Slack Integration (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CRITICAL_CHANNEL=#agentcore-critical
SLACK_WARNING_CHANNEL=#agentcore-warnings
SLACK_ACE_CHANNEL=#agentcore-ace

# PagerDuty Integration (optional)
PAGERDUTY_SERVICE_KEY=your-pagerduty-service-key

# Grafana Settings
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure-password
GRAFANA_ROOT_URL=http://localhost:3000

# Database Connection (for PostgreSQL Exporter)
POSTGRES_EXPORTER_DSN=postgresql://postgres:password@postgres:5432/agentcore?sslmode=disable

# Redis Connection (for Redis Exporter)
REDIS_ADDR=redis:6379
REDIS_PASSWORD=
```

### Alert Routing

Alerts are routed based on severity:

- **Critical**: PagerDuty + Slack (#agentcore-critical)
- **Warning**: Slack (#agentcore-warnings) + Email
- **Info**: Slack (#agentcore-info)
- **ACE-specific**: Slack (#agentcore-ace)

Edit `monitoring/alertmanager.yml` to customize alert routing.

## Support

For issues or questions:

- **Documentation**: See `docs/ace-load-test-report.md` and `docs/ace-compass-validation-report.md`
- **Alert Runbooks**: See annotations in `monitoring/alerts/ace-alerts.yml`
- **GitHub Issues**: https://github.com/your-org/agentcore/issues

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [COMPASS Paper](https://example.com/compass-paper)
- [ACE Integration Documentation](../docs/ace-integration.md)
