# TOOL-014: Tool Registration on Startup

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Create `register_builtin_tools()` function to auto-register all built-in tools on application startup with configuration via environment variables

## Acceptance Criteria
- [ ] `register_builtin_tools(registry)` function in builtin.py
- [ ] Auto-registration of all 5 built-in tools
- [ ] Environment variable configuration (GEMINI_API_KEY, ENABLE_CODE_EXECUTION, etc.)
- [ ] Conditional registration based on config (e.g., skip Google if no API key)
- [ ] Integration with FastAPI startup event
- [ ] Logging of registered tools on startup
- [ ] Unit tests for registration logic

## Dependencies
#TOOL-009, #TOOL-010, #TOOL-011, #TOOL-012, #TOOL-013

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 2
**Estimated Duration:** 2 days
**Sprint:** 2

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/builtin.py
