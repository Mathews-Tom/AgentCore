# LLM-CLIENT-011: Prometheus Metrics Instrumentation

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** llm-client-service
**Effort:** 5 SP
**Sprint:** Sprint 2
**Phase:** Features
**Parent:** LLM-001

## Description

Add comprehensive Prometheus metrics tracking all LLM operations for observability.

Production observability requirement for monitoring costs, performance, and errors.

## Acceptance Criteria

- [x] llm_requests_total counter with labels (provider, model, status)
- [x] llm_requests_duration_seconds histogram with labels (provider, model)
- [x] llm_tokens_total counter with labels (provider, model, token_type: prompt/completion)
- [x] llm_errors_total counter with labels (provider, model, error_type)
- [x] llm_active_requests gauge with label (provider)
- [x] llm_governance_violations_total counter (model attempted, source_agent)
- [x] Metrics exposed at /metrics endpoint
- [x] Metrics updated in real-time
- [x] Grafana dashboard template provided
- [x] Unit tests for metrics (verify counters increment)

## Dependencies

**Blocks:** LLM-CLIENT-013 (JSON-RPC metrics handler), LLM-CLIENT-018 (audit logging)

**Requires:** LLM-CLIENT-009 (LLMService facade)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/metrics/llm_metrics.py`

**SDK:** prometheus-client ^0.21.0 (already in AgentCore)

```python
from prometheus_client import Counter, Histogram, Gauge

llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'model', 'status']
)

llm_requests_duration_seconds = Histogram(
    'llm_requests_duration_seconds',
    'LLM request duration',
    ['provider', 'model']
)
```

## Estimated Time

- **Story Points:** 5 SP
- **Time:** 2-3 days (Backend Engineer 2)
- **Sprint:** Sprint 2, Days 11-13

## Owner

Backend Engineer 2

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
