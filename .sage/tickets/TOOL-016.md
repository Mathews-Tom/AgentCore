# TOOL-016: tools.list JSON-RPC Method

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Register `tools.list` JSON-RPC method with `@register_jsonrpc_method` decorator for tool discovery with category filtering

## Acceptance Criteria
- [ ] `tools.list` method registered with JSON-RPC handler
- [ ] Optional category parameter for filtering
- [ ] Returns tool metadata (tool_id, name, description, parameters, authentication, rate_limit)
- [ ] Integration with ToolRegistry
- [ ] Error handling (invalid category)
- [ ] Unit tests for method logic
- [ ] Integration test via HTTP POST to /api/v1/jsonrpc

## Dependencies
#TOOL-004, #TOOL-014

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
**Files:** src/agentcore/tools/jsonrpc.py
