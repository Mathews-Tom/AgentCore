# COORD-012: JSON-RPC Methods

**State:** COMPLETED
**Priority:** P0
**Type:** Story
**Component:** coordination-service
**Effort:** 3 SP
**Sprint:** Sprint 1
**Phase:** Integration
**Parent:** COORD-001

## Description

Expose coordination service via JSON-RPC 2.0 protocol for A2A integration

## Acceptance Criteria

- [x] coordination.signal handler (register signal from agent)
- [x] coordination.state handler (get agent coordination state)
- [x] coordination.metrics handler (get current metrics snapshot)
- [x] coordination.predict_overload handler (get overload prediction)
- [x] All methods registered with jsonrpc_processor
- [x] A2A context extraction from JsonRpcRequest
- [x] Error mapping to JsonRpcErrorCode
- [x] Request/response validation with Pydantic
- [x] Unit tests for all handlers
- [ ] Integration test via JSON-RPC endpoint (COORD-013)

## Dependencies

**Blocks:** Dependent tasks

**Requires:** COORD-007

## Technical Notes

**Files:**
  - `src/agentcore/a2a_protocol/services/coordination_jsonrpc.py`
  - `tests/coordination/integration/test_jsonrpc.py`

**Owner:** Backend Engineer

## Estimated Time

- **Story Points:** 3 SP
- **Sprint:** Sprint 1 (2 weeks)

## Progress

**Status:** COMPLETED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-11-05T00:25:12Z
**Completed:** 2025-11-05T00:25:12Z

## Implementation

- **Commit:** `fa61e0e` - feat(coord): #COORD-012 add JSON-RPC methods for coordination service
- **Tests:** 16/16 passed (100%)
- **Files Modified:**
  - `src/agentcore/a2a_protocol/services/coordination_jsonrpc.py`
  - `tests/coordination/unit/test_coordination_jsonrpc.py`
- **Methods Implemented:**
  - coordination.signal
  - coordination.state
  - coordination.metrics
  - coordination.predict_overload
