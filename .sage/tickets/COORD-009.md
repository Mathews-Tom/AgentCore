# COORD-009: Signal Cleanup Service

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story
**Component:** coordination-service
**Effort:** 3 SP
**Sprint:** Sprint 1
**Phase:** Optimization
**Parent:** COORD-001

## Description

Implement background task for periodic cleanup of expired signals and stale states

## Acceptance Criteria

- [ ] cleanup_expired_signals() removes all expired signals
- [ ] Iterates all agent coordination states
- [ ] Removes signals where is_expired() == True
- [ ] Removes agent states with no active signals
- [ ] Scores recomputed after cleanup
- [ ] Cleanup statistics logged (signals removed, agents removed)
- [ ] Background task runs every COORDINATION_CLEANUP_INTERVAL
- [ ] Integration tests for cleanup behavior

## Dependencies

**Blocks:** Dependent tasks

**Requires:** COORD-006

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/services/coordination_service.py`
  - `tests/coordination/integration/test_cleanup.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 3 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
