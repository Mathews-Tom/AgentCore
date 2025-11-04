# COORD-002: Data Models and Enums

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** coordination-service
**Effort:** 3 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** COORD-001

## Description

Create Pydantic models for SensitivitySignal, AgentCoordinationState, CoordinationMetrics with validation

## Acceptance Criteria

- [ ] SignalType enum (LOAD, CAPACITY, QUALITY, COST, LATENCY, AVAILABILITY)
- [ ] SensitivitySignal model with all fields (agent_id, signal_type, value 0.0-1.0, timestamp, ttl_seconds, confidence)
- [ ] AgentCoordinationState model with score fields (load, capacity, quality, cost, availability, routing_score)
- [ ] CoordinationMetrics model for Prometheus
- [ ] All models have 100% type coverage (mypy strict)
- [ ] Pydantic validators for value ranges (0.0-1.0, ttl >0)

## Dependencies

**Blocks:** Subsequent development

**Requires:** COORD-001 (epic)

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/models/coordination.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 3 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
