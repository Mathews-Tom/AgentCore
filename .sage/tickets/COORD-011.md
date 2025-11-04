# COORD-011: MessageRouter Integration

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story
**Component:** coordination-service
**Effort:** 8 SP
**Sprint:** Sprint 1
**Phase:** Integration
**Parent:** COORD-001

## Description

Add RIPPLE_COORDINATION routing strategy to MessageRouter and integrate with CoordinationService

## Acceptance Criteria

- [ ] RoutingStrategy enum extended with RIPPLE_COORDINATION
- [ ] _ripple_coordination_select(candidates) method in MessageRouter
- [ ] Integration with coordination_service.select_optimal_agent()
- [ ] Fallback to RANDOM routing on coordination errors
- [ ] Selection logged with trace_id
- [ ] Metrics updated (coordination_routing_selections_total)
- [ ] Backward compatibility with existing strategies validated
- [ ] Integration tests for RIPPLE_COORDINATION strategy

## Dependencies

**Blocks:** Dependent tasks

**Requires:** COORD-007

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/services/message_router.py`
  - `tests/coordination/integration/test_message_router.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 8 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
