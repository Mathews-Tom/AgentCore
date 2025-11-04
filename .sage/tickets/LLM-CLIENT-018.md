# LLM-CLIENT-018: Governance Audit Logging

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story
**Component:** llm-client-service
**Effort:** 2 SP
**Sprint:** Sprint 1
**Phase:** Risk Mitigation
**Parent:** LLM-001

## Description

Implement comprehensive audit logging for model governance violations with alerts.

Compliance and security requirement for model usage tracking.

## Acceptance Criteria

- [ ] All governance violations logged to dedicated log stream
- [ ] Log entries include: timestamp, trace_id, source_agent, session_id, attempted_model, reason
- [ ] Structured logging format (JSON)
- [ ] Prometheus alert rule for violations (>10/hour threshold)
- [ ] Integration with monitoring system (alert to Slack/PagerDuty)
- [ ] Audit log retention policy documented (90 days minimum)
- [ ] Query examples for common audit scenarios

## Dependencies

**Requires:**
- LLM-CLIENT-009 (LLMService facade)
- LLM-CLIENT-011 (metrics)

## Technical Notes

**File Location:** `src/agentcore/a2a_protocol/services/llm_service.py` (audit logging logic)

**Structured Logging:**
```python
if request.model not in self.config.ALLOWED_MODELS:
    logger.warning(
        "Model governance violation",
        extra={
            "trace_id": request.trace_id,
            "source_agent": request.source_agent,
            "session_id": request.session_id,
            "attempted_model": request.model,
            "allowed_models": self.config.ALLOWED_MODELS,
            "violation_type": "disallowed_model"
        }
    )
    llm_governance_violations_total.labels(
        model=request.model,
        source_agent=request.source_agent
    ).inc()
    raise ModelNotAllowedError(request.model)
```

## Estimated Time

- **Story Points:** 2 SP
- **Time:** 1 day (Backend Engineer 2)
- **Sprint:** Sprint 1, Day 10

## Owner

Backend Engineer 2

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-25
**Updated:** 2025-10-25
