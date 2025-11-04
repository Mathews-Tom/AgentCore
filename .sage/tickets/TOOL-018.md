# TOOL-018: tools.search JSON-RPC Method

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story

## Description
Register `tools.search` JSON-RPC method as convenience wrapper around tools.list with query-based search

## Acceptance Criteria
- [ ] `tools.search` method registered with JSON-RPC handler
- [ ] Parameters: query, category (optional), limit (optional)
- [ ] Delegates to ToolRegistry.search()
- [ ] Returns matching tools sorted by relevance
- [ ] Unit tests for search logic
- [ ] Integration test via JSON-RPC

## Dependencies
#TOOL-016

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 2
**Estimated Duration:** 2 days
**Sprint:** 3

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/jsonrpc.py
