# TOOL-026: Prometheus Metrics

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story

## Description
Add Prometheus metrics for tool execution monitoring: latency histograms, success/failure counters, rate limit counters

## Acceptance Criteria
- [ ] `tool_execution_duration_seconds` histogram (p50, p95, p99) by tool_id, success
- [ ] `tool_execution_total` counter by tool_id, success
- [ ] `rate_limit_exceeded_total` counter by tool_id, user_id
- [ ] `framework_overhead_seconds` histogram
- [ ] `tool_registry_size` gauge
- [ ] Metrics integrated with existing Prometheus endpoint
- [ ] Unit tests for metric recording

## Dependencies
#TOOL-017, #TOOL-023

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 5
**Estimated Duration:** 5 days
**Sprint:** 4

## Implementation Details
**Owner:** DevOps Engineer
**Files:** src/agentcore/tools/metrics.py
