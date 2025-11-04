# COORD-012: JSON-RPC Methods

**State:** UNPROCESSED
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

- [ ] coordination.signal handler (register signal from agent)
- [ ] coordination.state handler (get agent coordination state)
- [ ] coordination.metrics handler (get current metrics snapshot)
- [ ] coordination.predict_overload handler (get overload prediction)
- [ ] All methods registered with jsonrpc_processor
- [ ] A2A context extraction from JsonRpcRequest
- [ ] Error mapping to JsonRpcErrorCode
- [ ] Request/response validation with Pydantic
- [ ] Unit tests for all handlers
- [ ] Integration test via JSON-RPC endpoint

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

**Status:** UNPROCESSED
**Created:** 2025-10-24T23:00:58.098754+00:00
**Updated:** 2025-10-24T23:00:58.098754+00:00
