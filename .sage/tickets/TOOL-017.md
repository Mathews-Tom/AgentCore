# TOOL-017: tools.execute JSON-RPC Method

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Register `tools.execute` JSON-RPC method for tool invocation with parameter validation, authentication, and result formatting

## Acceptance Criteria
- [ ] `tools.execute` method registered with JSON-RPC handler
- [ ] Parameters: tool_id, parameters, context (optional)
- [ ] Integration with ToolExecutor
- [ ] Parameter validation before execution
- [ ] A2A context extraction (trace_id, source_agent)
- [ ] Error handling with proper JSON-RPC error codes (400, 404, 429, 408, 500)
- [ ] Returns ToolResult with success, result, error, execution_time_ms
- [ ] Unit tests with mocked ToolExecutor
- [ ] Integration tests with real tools

## Dependencies
#TOOL-006, #TOOL-014

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 5
**Estimated Duration:** 5 days
**Sprint:** 3

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/jsonrpc.py
