# COORD-004: CoordinationService Core

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** coordination-service
**Effort:** 5 SP
**Sprint:** Sprint 1
**Phase:** Foundation
**Parent:** COORD-001

## Description

Implement CoordinationService class with signal registration, validation, and normalization

## Acceptance Criteria

- [ ] CoordinationService class in coordination_service.py
- [ ] register_signal(signal) validates and normalizes signal values
- [ ] Signal validation: value in 0.0-1.0, ttl >0, valid agent_id
- [ ] Signal normalization ensures consistent 0.0-1.0 range
- [ ] Signals stored in coordination_states dict
- [ ] UUID generation for signal_id
- [ ] Timestamp recorded on registration
- [ ] Unit tests for validation and normalization (90%+ coverage)

## Dependencies

**Blocks:** Dependent tasks

**Requires:** COORD-002, COORD-003

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/unit/test_signal_registration.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 5 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
