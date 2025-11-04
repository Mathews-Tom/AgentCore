# TOOL-021: Error Categorization

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Define error types (auth, validation, timeout, execution) and map tool errors to JSON-RPC error codes with structured error responses

## Acceptance Criteria
- [ ] Error type enum: AuthError, ValidationError, TimeoutError, ExecutionError, RateLimitError
- [ ] Mapping from tool errors to JSON-RPC error codes (401, 400, 408, 500, 429)
- [ ] Structured error responses with error type, message, details
- [ ] Error handling in ToolExecutor
- [ ] Unit tests for error categorization
- [ ] Integration tests for each error type

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
**Files:** src/agentcore/tools/errors.py
