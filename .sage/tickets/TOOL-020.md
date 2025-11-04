# TOOL-020: Distributed Tracing Support

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story

## Description
Integrate OpenTelemetry for distributed tracing with span creation, trace ID propagation, and tool execution linking

## Acceptance Criteria
- [ ] OpenTelemetry SDK integrated
- [ ] Spans created for each tool execution
- [ ] Trace ID propagated via A2A context
- [ ] Tool executions linked to parent trace (agent request)
- [ ] Span attributes include tool_id, user_id, success, execution_time_ms
- [ ] Trace export to OpenTelemetry collector
- [ ] Integration tests validate trace propagation

## Dependencies
#TOOL-017

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 5
**Estimated Duration:** 5 days
**Sprint:** 3

## Implementation Details
**Owner:** DevOps Engineer
**Files:** src/agentcore/tools/tracing.py
