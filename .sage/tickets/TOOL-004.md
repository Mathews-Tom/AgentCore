# TOOL-004: Tool Registry

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Implement ToolRegistry class with in-memory storage, category indexing, search capabilities, and tool discovery methods

## Acceptance Criteria
- [ ] ToolRegistry class with `_tools` dict and `_categories` index
- [ ] `register(tool: Tool) -> None` method validates and stores tools
- [ ] `get(tool_id: str) -> Tool | None` method with <10ms lookup
- [ ] `search(query: str, category: str) -> list[Tool]` with fuzzy matching
- [ ] `list_by_category(category: str) -> list[Tool]` method
- [ ] Unit tests for all registry operations
- [ ] Performance test validates <10ms lookup for 1000+ tools

## Dependencies
#TOOL-002

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 5
**Estimated Duration:** 5 days
**Sprint:** 1

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/registry.py
