# COORD-006: Signal History & TTL Management

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** coordination-service
**Effort:** 5 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** COORD-001

## Description

Implement signal history tracking and time-based expiry with temporal decay

## Acceptance Criteria

- [ ] Signal history stored per agent (max 100 signals)
- [ ] Oldest signals evicted when history full
- [ ] is_expired(signal) checks TTL against current time
- [ ] get_active_signals(agent_id) filters expired signals
- [ ] Temporal decay applied to signal values based on age
- [ ] Decay formula: value * e^(-age / ttl)
- [ ] Scores recomputed excluding expired signals
- [ ] Unit tests for TTL and decay (95%+ coverage)

## Dependencies

**Blocks:** Dependent tasks

**Requires:** COORD-005

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_signal_ttl.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 5 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
