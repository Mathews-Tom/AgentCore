# TOOL-019: A2A Authentication Integration

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Integrate with A2A authentication to extract user_id/agent_id from JWT and pass to ExecutionContext for RBAC enforcement

## Acceptance Criteria
- [ ] Extract user_id and agent_id from JWT claims
- [ ] Pass auth information to ExecutionContext
- [ ] RBAC policy enforcement for tool access (basic implementation)
- [ ] Authentication errors return 401 Unauthorized
- [ ] Unit tests for auth extraction and validation
- [ ] Integration tests with valid/invalid JWTs

## Dependencies
#TOOL-017

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 3
**Estimated Duration:** 3 days
**Sprint:** 3

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/auth.py
